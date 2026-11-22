from datetime import UTC, datetime

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.agent.context import build_messages
from backend.agent.converse import TextBlock, ToolResultBlock, ToolUseBlock
from backend.agent.results import envelope
from ml_engine.contracts.agent import (
    CallState,
    Plan,
    PlanStep,
    Session,
    SessionLimits,
    SessionStatus,
    ToolCallRecord,
    ToolClass,
    TranscriptEntry,
    args_hash,
)
from tests.agent.strategies import SETTINGS

NOW = datetime(2025, 1, 1, tzinfo=UTC)


def _session(turns=5) -> Session:
    session = Session(
        session_id="s1",
        owner="u1",
        created_at=NOW,
        updated_at=NOW,
        last_activity_at=NOW,
        status=SessionStatus.RUNNING,
        limits=SessionLimits(
            max_converse_turns=50,
            max_tool_calls=50,
            max_costly_tool_calls=1,
            max_estimated_spend_usd=1,
            max_active_seconds=60,
            max_repeated_identical_calls=3,
        ),
        goal="predict churn",
        plan=Plan(steps=[PlanStep(text="profile the data")], accepted_at=NOW),
    )
    seq = 1
    session.transcript.append(
        TranscriptEntry(seq=seq, at=NOW, kind="user_message", text="predict churn")
    )
    for i in range(1, turns + 1):
        call_id = f"c{i}"
        session.tool_calls.append(
            ToolCallRecord(
                tool_call_id=call_id,
                turn=i,
                tool="get_metrics",
                arguments={"i": i},
                args_hash=args_hash("get_metrics", {"i": i}),
                tool_class=ToolClass.READ,
                state=CallState.SUCCEEDED,
            )
        )
        view = envelope(
            call_id,
            "get_metrics",
            {"metrics": {"auc": 0.5 + i / 100, "n": 100 * i}},
            truncated=False,
        )
        session.transcript.append(
            TranscriptEntry(seq=seq + 1, at=NOW, kind="tool_call", tool_call_id=call_id)
        )
        session.transcript.append(
            TranscriptEntry(
                seq=seq + 2,
                at=NOW,
                kind="tool_result",
                tool_call_id=call_id,
                data={"model_view": view},
            )
        )
        seq += 2
    return session


def test_no_compaction_within_budget() -> None:
    session = _session()
    messages, info = build_messages(session, 100_000)
    assert len(messages) == 11
    assert info.compacted_through_seq == 0


def test_compaction_summarizes_older_turns_and_keeps_numbers() -> None:
    session = _session()
    before = session.model_dump()
    messages, info = build_messages(session, 10)

    assert session.model_dump() == before  # persisted session untouched
    # goal + summary, then the last 6 messages (3 tool use / result pairs)
    assert [m.role for m in messages] == ["user"] + ["assistant", "user"] * 3
    goal, summary = messages[0].blocks
    assert goal == TextBlock("predict churn")
    assert isinstance(summary, TextBlock)
    assert "1. profile the data" in summary.text
    assert "[c1] get_metrics succeeded metrics.auc=0.51 metrics.n=100" in summary.text
    assert "[c2] get_metrics succeeded metrics.auc=0.52 metrics.n=200" in summary.text
    assert "[c3]" not in summary.text
    assert isinstance(messages[1].blocks[0], ToolUseBlock) and messages[1].blocks[0].request_id == "c3"
    assert isinstance(messages[2].blocks[0], ToolResultBlock)
    assert info.compacted_through_seq == 5
    assert info.summary_tokens_estimate > 0
    # deterministic
    again, _ = build_messages(session, 10)
    assert again == messages


def test_last_reported_input_tokens_triggers_compaction() -> None:
    session = _session()
    session.transcript.append(
        TranscriptEntry(seq=99, at=NOW, kind="status", data={"input_tokens": 50_000})
    )
    _, info = build_messages(session, 40_000)
    assert info.compacted_through_seq == 5


# Feature: ml-assistant, Property 29: Compaction preserves ids and engine numbers
@settings(**SETTINGS)
@given(turns=st.integers(4, 15), budget=st.integers(1, 2000))
def test_compaction_preserves_each_summarized_call(turns, budget):
    session = _session(turns)
    before = session.model_dump()
    messages, info = build_messages(session, budget)
    assert session.model_dump() == before
    for i in range(1, turns + 1):
        if 2 * i + 1 <= info.compacted_through_seq:
            summary = messages[0].blocks[-1].text
            assert f"[c{i}] get_metrics succeeded metrics.auc={0.5 + i / 100} metrics.n={100*i}" in summary
