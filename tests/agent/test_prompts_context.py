from datetime import UTC, datetime

from pydantic import BaseModel

from backend.agent.context import build_messages
from backend.agent.converse import TextBlock, ToolResultBlock, ToolUseBlock
from backend.agent.prompts import SYSTEM_PROMPT_VERSION, system_prompt
from backend.agent.tools import ToolOutput, ToolRegistry
from ml_engine.contracts.agent import (
    CallState,
    Session,
    SessionLimits,
    SessionStatus,
    ToolCallRecord,
    ToolClass,
    TranscriptEntry,
    args_hash,
)

NOW = datetime(2025, 1, 1, tzinfo=UTC)


class _In(BaseModel):
    pass


def _session(goal: str = "predict churn for ACME") -> Session:
    return Session(
        session_id="s1",
        owner="u1",
        created_at=NOW,
        updated_at=NOW,
        last_activity_at=NOW,
        status=SessionStatus.RUNNING,
        limits=SessionLimits(
            max_converse_turns=5,
            max_tool_calls=5,
            max_costly_tool_calls=1,
            max_estimated_spend_usd=1,
            max_active_seconds=60,
            max_repeated_identical_calls=3,
        ),
        goal=goal,
    )


def _entry(seq: int, kind: str, **kw) -> TranscriptEntry:
    return TranscriptEntry(seq=seq, at=NOW, kind=kind, **kw)


def test_system_prompt_has_tools_and_no_user_data() -> None:
    registry = ToolRegistry()
    registry.register(
        "list_datasets",
        classes=[ToolClass.READ],
        description="List datasets.",
        input_model=_In,
        handler=lambda c, a: ToolOutput(data={}),
    )
    session = _session("secret-goal-text")
    text = system_prompt(session, registry)
    assert SYSTEM_PROMPT_VERSION == "1"
    assert "list_datasets [read]" in text
    assert "secret-goal-text" not in text
    assert "untrusted_data" in text


def test_build_messages_maps_transcript() -> None:
    session = _session()
    session.tool_calls.append(
        ToolCallRecord(
            tool_call_id="c1",
            turn=1,
            tool="list_datasets",
            arguments={},
            args_hash=args_hash("list_datasets", {}),
            tool_class=ToolClass.READ,
            state=CallState.SUCCEEDED,
            result_summary="2 datasets",
        )
    )
    session.transcript.extend(
        [
            _entry(1, "user_message", text=session.goal),
            _entry(2, "assistant_text", text="Looking."),
            _entry(3, "tool_call", tool_call_id="c1"),
            _entry(4, "tool_result", tool_call_id="c1"),
            _entry(5, "job_update", data={"state": "succeeded"}),
            _entry(6, "status", text="ignored"),
        ]
    )
    messages, info = build_messages(session, 1000)
    assert [m.role for m in messages] == ["user", "assistant", "user"]
    assert messages[0].blocks == (TextBlock(session.goal),)
    assert isinstance(messages[1].blocks[1], ToolUseBlock)
    result, update = messages[2].blocks
    assert isinstance(result, ToolResultBlock)
    assert result.content["untrusted_data"]["data"]["summary"] == "2 datasets"
    assert isinstance(update, TextBlock) and "untrusted_data" in update.text
    assert info.compacted_through_seq == 0
