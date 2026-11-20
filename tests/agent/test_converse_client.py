"""Property tests for the Bedrock Converse client (offline, fake boto client)."""

from __future__ import annotations

from typing import Any

import pytest
from botocore.exceptions import ClientError
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.agent.converse import (
    BedrockConverseClient,
    ConverseError,
    ConverseMessage,
    ConverseRequestError,
    ConverseTransientError,
    TextBlock,
    ToolDefinition,
    ToolResultBlock,
    ToolUseBlock,
)
from tests.agent.strategies import SETTINGS, _ident, _text, json_trees, tool_arguments

MODEL_ID = "anthropic.claude-test-v1"


class FakeBoto:
    """Records each ``converse`` call and returns a canned response."""

    def __init__(self, response: dict[str, Any]) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        return self.response


# ----- generators ---------------------------------------------------------------------------

_request_ids = st.from_regex(r"tu-[a-z0-9]{1,8}", fullmatch=True)
_json_objects = st.dictionaries(_ident, json_trees, max_size=3)

blocks = st.one_of(
    st.builds(TextBlock, _text),
    st.builds(ToolUseBlock, _request_ids, _ident, tool_arguments),
    st.builds(ToolResultBlock, _request_ids, _json_objects),
)
messages = st.lists(
    st.builds(
        ConverseMessage,
        st.sampled_from(["user", "assistant"]),
        st.lists(blocks, min_size=1, max_size=4).map(tuple),
    ),
    min_size=1,
    max_size=5,
)
tool_definitions = st.lists(
    st.builds(ToolDefinition, _ident, _text, _json_objects), max_size=4, unique_by=lambda t: t.name
)

_wire_text = st.builds(lambda t: {"text": t}, _text)
_wire_tool_use = st.builds(
    lambda rid, name, args: {"toolUse": {"toolUseId": rid, "name": name, "input": args}},
    _request_ids,
    _ident,
    tool_arguments,
)
responses = st.builds(
    lambda content, stop, i, o: {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop,
        "usage": {"inputTokens": i, "outputTokens": o, "totalTokens": i + o},
    },
    st.lists(st.one_of(_wire_text, _wire_tool_use), max_size=6),
    st.sampled_from(["end_turn", "tool_use", "max_tokens", "stop_sequence", "guardrail_intervened"]),
    st.integers(0, 200_000),
    st.integers(0, 8192),
)

def _expected_block(block: Any) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"text": block.text}
    if isinstance(block, ToolUseBlock):
        return {"toolUse": {"toolUseId": block.request_id, "name": block.name, "input": block.arguments}}
    return {"toolResult": {"toolUseId": block.request_id, "content": [{"json": block.content}]}}


# Feature: ml-assistant, Property 45: Converse requests and responses map faithfully
@settings(**SETTINGS)
@given(
    system=_text,
    msgs=messages,
    tools=tool_definitions,
    max_tokens=st.integers(1, 8192),
    response=responses,
)
def test_converse_requests_and_responses_map_faithfully(
    system: str,
    msgs: list[ConverseMessage],
    tools: list[ToolDefinition],
    max_tokens: int,
    response: dict[str, Any],
) -> None:
    """**Validates: Requirements 2.1, 2.7, 2.10**"""
    boto = FakeBoto(response)
    client = BedrockConverseClient(boto, model_id=MODEL_ID, retries=0, backoff_seconds=0.0)

    turn = client.converse(system=system, messages=msgs, tools=tools, max_tokens=max_tokens)

    # Request side: one call, deterministic inference config, every block and tool carried.
    assert len(boto.calls) == 1
    request = boto.calls[0]
    assert request["modelId"] == MODEL_ID
    assert request["system"] == [{"text": system}]
    assert request["inferenceConfig"]["temperature"] == 0
    assert request["inferenceConfig"]["maxTokens"] == max_tokens
    assert request["messages"] == [
        {"role": m.role, "content": [_expected_block(b) for b in m.blocks]} for m in msgs
    ]
    if tools:
        assert request["toolConfig"]["tools"] == [
            {"toolSpec": {"name": t.name, "description": t.description, "inputSchema": {"json": t.input_schema}}}
            for t in tools
        ]
    else:
        assert "toolConfig" not in request

    # Response side: tool uses preserved in order; text, model id and usage reported.
    content = response["output"]["message"]["content"]
    expected_uses = [b["toolUse"] for b in content if "toolUse" in b]
    assert [(u.request_id, u.name, u.arguments) for u in turn.tool_uses] == [
        (u["toolUseId"], u["name"], u["input"]) for u in expected_uses
    ]
    assert turn.text == "".join(b["text"] for b in content if "text" in b)
    assert turn.model_id == MODEL_ID
    assert turn.usage.input_tokens == response["usage"]["inputTokens"]
    assert turn.usage.output_tokens == response["usage"]["outputTokens"]
    raw_stop = response["stopReason"]
    assert turn.stop_reason == (raw_stop if raw_stop in ("end_turn", "tool_use", "max_tokens") else "other")


# ----- retry backoff ------------------------------------------------------------------------

_TRANSIENT = ["ThrottlingException", "ServiceUnavailableException", "InternalServerException",
              "ModelTimeoutException", "ModelNotReadyException"]
_NON_TRANSIENT = ["AccessDeniedException", "ValidationException", "ResourceNotFoundException"]
_OK_RESPONSE = {
    "output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
    "stopReason": "end_turn",
    "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2},
}


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "secret detail"}}, "Converse")


class ScriptedBoto:
    """Raises the scripted errors in order, then returns a canned success response."""

    def __init__(self, errors: list[ClientError]) -> None:
        self.errors = list(errors)
        self.attempts = 0

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.attempts += 1
        if self.errors:
            raise self.errors.pop(0)
        return _OK_RESPONSE


# Feature: ml-assistant, Property 46: Retry follows the configured backoff
@settings(**SETTINGS)
@given(
    retries=st.integers(0, 5),
    backoff=st.floats(0.0, 30.0, allow_nan=False),
    transient=st.lists(st.sampled_from(_TRANSIENT), max_size=8),
    terminal=st.one_of(st.none(), st.sampled_from(_NON_TRANSIENT)),
)
def test_retry_follows_the_configured_backoff(
    retries: int, backoff: float, transient: list[str], terminal: str | None
) -> None:
    """**Validates: Requirements 2.8**"""
    errors = [_client_error(c) for c in transient]
    if terminal is not None:
        errors.append(_client_error(terminal))
    boto = ScriptedBoto(errors)
    delays: list[float] = []
    client = BedrockConverseClient(
        boto, model_id=MODEL_ID, retries=retries, backoff_seconds=backoff, sleep=delays.append
    )

    def call() -> Any:
        return client.converse(system="s", messages=[], tools=[], max_tokens=16)

    if len(transient) > retries:
        # Retries exhausted on transient errors: r + 1 attempts, the last transient error raised.
        with pytest.raises(ConverseTransientError) as info:
            call()
        assert boto.attempts == retries + 1
        assert info.value.code == transient[retries]
    elif terminal is not None:
        # Transient failures retried, then the non-transient error raised with no further delay.
        with pytest.raises(ConverseRequestError) as info:
            call()
        assert boto.attempts == len(transient) + 1
        assert info.value.code == terminal
    else:
        turn = call()
        assert turn.text == "ok"
        assert boto.attempts == len(transient) + 1

    # One delay per retry, doubling from the base: d * 2^(i-1) before attempt i + 1.
    assert delays == [backoff * 2 ** (i - 1) for i in range(1, boto.attempts)]


@pytest.mark.parametrize("code", _NON_TRANSIENT)
def test_non_transient_error_is_raised_after_one_attempt_without_delay(code: str) -> None:
    boto = ScriptedBoto([_client_error(code)])
    delays: list[float] = []
    client = BedrockConverseClient(boto, model_id=MODEL_ID, retries=5, backoff_seconds=1.0, sleep=delays.append)

    with pytest.raises(ConverseError) as info:
        client.converse(system="s", messages=[], tools=[], max_tokens=16)

    assert not isinstance(info.value, ConverseTransientError)
    assert "secret detail" not in str(info.value)
    assert boto.attempts == 1
    assert delays == []
