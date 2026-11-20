"""Contract tests of the Converse client against recorded Bedrock response shapes (offline).

Fixtures live in ``tests/fixtures/converse/``. Success shapes are returned by a fake boto client;
error shapes are raised as ``botocore.exceptions.ClientError`` exactly as botocore builds them.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

import backend.aws
from backend.agent import converse as converse_module
from backend.agent.converse import (
    BedrockConverseClient,
    ConverseCredentialsError,
    ConverseModelUnsupportedError,
    ConverseTransientError,
    ConverseTurn,
    TokenUsage,
    ToolUseRequest,
)
from backend.config import Settings
from tests.support.datasets import ARTIFACT_ROOT

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures" / "converse"
MODEL_ID = "anthropic.claude-test-v1"
ROLE = "arn:aws:iam::123456789012:role/mlf-sagemaker-job"


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _recorded_error(name: str) -> ClientError:
    shape = _fixture(name)
    return ClientError(
        {"Error": shape["Error"], "ResponseMetadata": shape["ResponseMetadata"]},
        shape["operation_name"],
    )


class RecordedBoto:
    """Plays back recorded outcomes in order: a dict is returned, an exception is raised."""

    def __init__(self, *outcomes: Any) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def _client(boto: RecordedBoto, delays: list[float], retries: int = 3) -> BedrockConverseClient:
    return BedrockConverseClient(
        boto, model_id=MODEL_ID, retries=retries, backoff_seconds=1.0, sleep=delays.append
    )


def _call(client: BedrockConverseClient) -> ConverseTurn:
    return client.converse(system="s", messages=[], tools=[], max_tokens=64)


# ----- success shapes ------------------------------------------------------------------------


def test_recorded_text_turn_maps_to_a_text_only_turn() -> None:
    delays: list[float] = []
    turn = _call(_client(RecordedBoto(_fixture("text_turn")), delays))

    assert turn == ConverseTurn(
        text="The dataset has 400 rows.",
        tool_uses=(),
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=120, output_tokens=9),
        model_id=MODEL_ID,
    )
    assert delays == []


def test_recorded_tool_use_turn_maps_to_tool_use_requests() -> None:
    turn = _call(_client(RecordedBoto(_fixture("tool_use_turn")), []))

    assert turn.text == "I will look at your datasets."
    assert turn.tool_uses == (ToolUseRequest("tooluse_abc123", "list_datasets", {}),)
    assert turn.stop_reason == "tool_use"
    assert turn.usage == TokenUsage(input_tokens=200, output_tokens=30)
    assert turn.model_id == MODEL_ID


# ----- error shapes --------------------------------------------------------------------------


def test_recorded_throttling_is_transient_and_retried() -> None:
    boto = RecordedBoto(_recorded_error("throttling"), _fixture("text_turn"))
    delays: list[float] = []

    turn = _call(_client(boto, delays))

    assert turn.text == "The dataset has 400 rows."
    assert len(boto.calls) == 2
    assert delays == [1.0]


def test_recorded_throttling_exhausts_retries_as_transient_error() -> None:
    boto = RecordedBoto(*[_recorded_error("throttling") for _ in range(3)])
    delays: list[float] = []

    with pytest.raises(ConverseTransientError) as info:
        _call(_client(boto, delays, retries=2))

    assert info.value.retriable is True
    assert info.value.code == "ThrottlingException"
    assert len(boto.calls) == 3
    assert delays == [1.0, 2.0]


def test_recorded_unsupported_model_names_the_model_and_is_not_retried() -> None:
    """Req 13.8: the first call reports the model id and reason; no retry, no fallback."""
    boto = RecordedBoto(_recorded_error("unsupported_model"))
    delays: list[float] = []

    with pytest.raises(ConverseModelUnsupportedError) as info:
        _call(_client(boto, delays, retries=5))

    error = info.value
    assert error.model_id == MODEL_ID
    assert MODEL_ID in str(error)
    assert error.reason
    assert error.reason in str(error)
    assert error.retriable is False
    assert error.code == "ValidationException"
    # Exactly one attempt, with the configured model only, and no backoff.
    assert len(boto.calls) == 1
    assert boto.calls[0]["modelId"] == MODEL_ID
    assert delays == []


def test_recorded_expired_credentials_is_a_retriable_credentials_error() -> None:
    """Req 14.9: expired credentials surface as a resumable credentials error, never retried here."""
    boto = RecordedBoto(_recorded_error("expired_credentials"))
    delays: list[float] = []

    with pytest.raises(ConverseCredentialsError) as info:
        _call(_client(boto, delays, retries=5))

    error = info.value
    assert error.retriable is True
    assert error.code == "ExpiredTokenException"
    assert "security token" not in str(error)  # the Bedrock message text is not carried
    assert len(boto.calls) == 1
    assert delays == []


# ----- lazy construction through backend.aws.build_client ------------------------------------


def _settings(**overrides: Any) -> Settings:
    return Settings(artifact_root=ARTIFACT_ROOT, sagemaker_execution_role_arn=ROLE, **overrides)


def test_boto_client_is_built_lazily_through_build_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Req 13.7: no client at construction; one ``build_client("bedrock-runtime", ...)`` on first use."""
    built: list[tuple[str, Any, dict[str, Any]]] = []
    boto = RecordedBoto(_fixture("text_turn"), _fixture("tool_use_turn"))

    def fake_build_client(service: str, settings: Any, **overrides: Any) -> RecordedBoto:
        built.append((service, settings, overrides))
        return boto

    monkeypatch.setattr(backend.aws, "build_client", fake_build_client)
    settings = _settings(agent_model_id=MODEL_ID, agent_converse_read_timeout_seconds=45)

    client = BedrockConverseClient.from_settings(settings, sleep=lambda _: None)
    assert built == []
    assert client.model_id == MODEL_ID

    _call(client)
    _call(client)

    assert built == [("bedrock-runtime", settings, {"read_timeout": 45})]
    assert len(boto.calls) == 2


def test_from_settings_falls_back_to_the_bedrock_model_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(backend.aws, "build_client", lambda *a, **k: pytest.fail("built eagerly"))
    settings = _settings(bedrock_model_id=MODEL_ID)

    assert BedrockConverseClient.from_settings(settings).model_id == MODEL_ID


# ----- BedrockReasoningClient is unchanged ---------------------------------------------------


def test_bedrock_reasoning_client_is_unchanged() -> None:
    """Req 2.2: the Converse client is new; ``ReasoningClient``/``BedrockReasoningClient`` are untouched."""
    from ml_engine.reasoning import client as reasoning

    manifest = json.loads((ROOT / "tests" / "inherited_manifest.json").read_text(encoding="utf-8"))
    source = (ROOT / "ml_engine" / "reasoning" / "client.py").read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(source).hexdigest() == manifest["files"]["ml_engine/reasoning/client.py"]

    assert not issubclass(BedrockConverseClient, reasoning.BedrockReasoningClient)
    assert not hasattr(BedrockConverseClient, "invoke")
    assert str(inspect.signature(reasoning.BedrockReasoningClient.invoke)) == (
        "(self, system_prompt: str, user_prompt: str) -> str"
    )
    assert "ml_engine.reasoning" not in inspect.getsource(converse_module)
