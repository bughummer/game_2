"""Offline tests for the minimal agent loop and runners (task 4.14, design C13 and C15)."""

from __future__ import annotations

import threading
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, Field

from backend.agent.converse import (
    ConverseCredentialsError,
    ConverseModelUnsupportedError,
    ConverseTransientError,
    ConverseTurn,
    TokenUsage,
    ToolUseRequest,
)
from backend.agent.execute import ToolExecutor
from backend.agent.inputs import ToolInputs
from backend.agent.loop import AgentLoop, StepOutcome
from backend.agent.runner import InlineAgentRunner, ThreadAgentRunner
from backend.agent.store import SessionStore
from backend.agent.tools import ToolFlags, ToolOutput, ToolRegistry
from backend.governance import GovernancePolicy
from ml_engine.contracts.agent import (
    CallState,
    Session,
    SessionLimits,
    SessionStatus,
    ToolClass,
)
from tests.agent.strategies import SETTINGS
from tests.support.agent_fakes import FakeClock, FlakyStore
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.scripted_converse import (
    ScriptedFakeConverseClient,
    ScriptStep,
    text_turn,
    tool_turn,
)

ROOT = "s3://bucket-test/root"
SID = "sess-1"
T0 = datetime(2025, 1, 1, tzinfo=UTC)
PLAN = ("tu-plan", "submit_plan", {"steps": ["look at data"]})


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanArgs(_Strict):
    steps: list[str] = Field(min_length=1, max_length=10)


class AskArgs(_Strict):
    question: str = Field(min_length=1)


class CountArgs(_Strict):
    limit: int = Field(default=5, ge=1, le=10)


class _Audit:
    def __init__(self) -> None:
        self.records: list = []

    def write(self, session, rec):
        self.records.append(rec)
        session.audit_seq += 1
        return f"a-{session.audit_seq}"


class _Authorizer:
    def require_owner(self, principal, experiment_id):  # pragma: no cover - not reached
        return None


def _session() -> Session:
    return Session(
        session_id=SID,
        owner="alice",
        created_at=T0,
        updated_at=T0,
        last_activity_at=T0,
        status=SessionStatus.RUNNING,
        limits=SessionLimits(
            max_converse_turns=10,
            max_tool_calls=20,
            max_costly_tool_calls=2,
            max_estimated_spend_usd=5.0,
            max_active_seconds=600,
            max_repeated_identical_calls=3,
        ),
        goal="predict churn",
    )


class Harness:
    def __init__(self, steps, *, objects=None, resume=False) -> None:
        self.calls: list[str] = []
        registry = ToolRegistry()
        control = ToolFlags(control=True, ledger_source=False)
        registry.register(
            "submit_plan",
            classes=[ToolClass.READ],
            description="plan",
            input_model=PlanArgs,
            handler=self._handler("submit_plan", lambda a: {"steps": a.steps}),
            flags=control,
        )
        registry.register(
            "ask_user",
            classes=[ToolClass.READ],
            description="ask",
            input_model=AskArgs,
            handler=self._handler("ask_user", lambda a: {"question": a.question}),
            flags=control,
        )
        registry.register(
            "count_rows",
            classes=[ToolClass.READ],
            description="read",
            input_model=CountArgs,
            handler=self._handler("count_rows", lambda a: {"rows": 3}),
        )
        registry.register(
            "change_thing",
            classes=[ToolClass.MUTATING],
            description="mutate",
            input_model=CountArgs,
            handler=self._handler("change_thing", lambda a: {"ok": True}),
            flags=ToolFlags(changes_config=True),
        )
        self.objects = objects or InMemoryObjectStore()
        self.clock = FakeClock(T0)
        self.store = SessionStore(self.objects, ROOT, clock=self.clock)
        if not resume:
            self.store.create(_session())
        self.audit = _Audit()
        inputs = ToolInputs(audit=self.audit, authorizer=_Authorizer(), allowed_instance_types=[])
        self.converse = ScriptedFakeConverseClient(steps=[_step(s) for s in steps])
        self.loop = AgentLoop(
            store=self.store,
            converse=self.converse,
            registry=registry,
            executor=ToolExecutor(inputs=inputs, services=SimpleNamespace(), clock=self.clock),
            inputs=inputs,
            audit=self.audit,
            governance=GovernancePolicy(
                send_sample_values=False, sample_values_per_column=0, sample_value_max_chars=0
            ),
            clock=self.clock,
        )

    def _handler(self, name, fn):
        def handler(ctx, args):
            self.calls.append(name)
            return ToolOutput(data=fn(args))

        return handler

    def run(self) -> Session:
        self.loop.run(SID, "test-holder")
        return self.session

    @property
    def session(self) -> Session:
        return self.store.load(SID).session


def _step(s) -> ScriptStep:
    return s if isinstance(s, ScriptStep) else ScriptStep(s)


def _results(s: Session) -> list[tuple[str | None, str]]:
    return [(e.tool_call_id, e.data["state"]) for e in s.transcript if e.kind == "tool_result"]


def test_plan_read_then_text_completes_with_final_answer_and_usage():
    h = Harness(
        [
            ScriptStep(
                tool_turn(PLAN, text="Here is my plan."),
                expect=lambda r: {t.name for t in r.tools} == {"submit_plan", "ask_user"},
            ),
            ScriptStep(
                tool_turn(("tu-r", "count_rows", {"limit": 2})),
                expect=lambda r: "count_rows" in {t.name for t in r.tools},
            ),
            text_turn("The dataset has 3 rows."),
        ]
    )
    s = h.run()
    assert s.status is SessionStatus.COMPLETED
    assert s.plan is not None and [p.text for p in s.plan.steps] == ["look at data"]
    assert h.calls == ["submit_plan", "count_rows"]
    assert _results(s) == [("tu-plan", "succeeded"), ("tu-r", "succeeded")]
    assert s.final_answer is not None
    assert s.final_answer.sentences[0].status == "unverified"  # minimal narration fails safe
    assert s.model_id == "scripted-model"
    assert (s.counters.converse_turns, s.counters.input_tokens, s.counters.output_tokens) == (3, 30, 15)
    assert s.lease is None


def test_first_turn_without_plan_or_question_fails_planning():
    h = Harness([text_turn("I will just answer.")])
    s = h.run()
    assert s.status is SessionStatus.FAILED
    assert s.failure.category == "planning_failed"
    assert "planning failed" in s.failure.message.lower()
    assert h.calls == []


def test_ask_user_defers_side_effects_and_keeps_request_order():
    h = Harness(
        [
            tool_turn(PLAN),
            tool_turn(
                ("tu-m", "change_thing", {}),
                ("tu-q", "ask_user", {"question": "Which target?"}),
                ("tu-r", "count_rows", {}),
            ),
        ]
    )
    s = h.run()
    assert s.status is SessionStatus.WAITING_FOR_USER
    assert _results(s)[1:] == [("tu-m", "deferred"), ("tu-q", "succeeded"), ("tu-r", "succeeded")]
    assert "change_thing" not in h.calls
    assert s.pending_tool_use_ids == []


def test_unknown_tool_invalid_args_and_unapproved_mutation_do_not_run_handlers():
    h = Harness(
        [
            tool_turn(PLAN),
            tool_turn(
                ("tu-u", "no_such_tool", {}),
                ("tu-reserved", "delete_everything", {}),
                ("tu-i", "count_rows", {"limit": 99}),
                ("tu-m", "change_thing", {}),
            ),
        ]
    )
    s = h.run()
    states = {r.tool_call_id: r for r in s.tool_calls}
    assert states["tu-u"].error_code == "UNKNOWN_TOOL"
    assert states["tu-reserved"].error_code == "UNKNOWN_TOOL"
    assert states["tu-i"].error_code == "INVALID_ARGUMENTS"
    view = next(e for e in s.transcript if e.tool_call_id == "tu-i" and e.kind == "tool_result")
    assert "limit" in str(view.data["model_view"])
    # Config-changing Mutating call in Supervised waits for approval; its handler never ran.
    assert states["tu-m"].state is CallState.AWAITING_APPROVAL
    assert s.status is SessionStatus.WAITING_FOR_APPROVAL
    assert h.calls == ["submit_plan"]
    assert [r.detail for r in h.audit.records if r.event_type == "input_rejected"] == [
        {"argument": "tool", "reason": "reserved_tool_name"}
    ]


@pytest.mark.parametrize(
    "exc,category,retriable",
    [
        (ConverseTransientError("throttled"), "bedrock_unavailable", True),
        (ConverseCredentialsError("expired"), "credentials_expired", True),
        (ConverseModelUnsupportedError("m-1", "no tool use"), "model_unsupported", False),
    ],
)
def test_converse_errors_map_to_failed_with_retriable_flag(exc, category, retriable):
    h = Harness([tool_turn(PLAN), exc])
    s = h.run()
    assert s.status is SessionStatus.FAILED
    assert (s.failure.category, s.failure.retriable, s.retriable) == (category, retriable, retriable)
    assert "throttled" not in s.failure.message  # exception text never reaches the user


def test_epoch_change_during_model_call_discards_the_result():
    h: Harness

    def bump(_request) -> bool:
        with h.store.transaction(SID) as s:
            s.epoch += 1
            s.status = SessionStatus.CANCELLED
        return True

    h = Harness([ScriptStep(tool_turn(PLAN), expect=bump)])
    s = h.run()
    assert s.status is SessionStatus.CANCELLED
    assert s.tool_calls == [] and s.counters.converse_turns == 0


def test_max_tokens_retries_once_then_fails_retriable():
    cut = ConverseTurn(
        text="partial",
        tool_uses=(ToolUseRequest("tu-x", "submit_plan", {"steps": ["a"]}),),
        stop_reason="max_tokens",
        usage=TokenUsage(input_tokens=1, output_tokens=1),
        model_id="scripted-model",
    )
    h = Harness([cut, cut])
    s = h.run()
    assert s.status is SessionStatus.FAILED
    assert (s.failure.category, s.failure.retriable) == ("max_tokens", True)
    assert s.tool_calls == [] and h.converse.calls == 2


def test_persist_failure_stops_the_loop_without_raising():
    objects = FlakyStore(InMemoryObjectStore())
    h = Harness([tool_turn(PLAN)], objects=objects)
    objects.fail_writes = 10_000
    objects.match = lambda uri: "/results/" in uri
    h.run()  # must not raise
    objects.fail_writes = 0
    s = h.session
    # The turn was stored, but the result write failed, so the use stays pending.
    assert s.pending_tool_use_ids == ["tu-plan"]
    assert s.status is SessionStatus.RUNNING
    assert h.calls == ["submit_plan"]


def test_step_is_noop_for_paused_and_terminal_sessions():
    h = Harness([])
    with h.store.transaction(SID) as s:
        s.status = SessionStatus.WAITING_FOR_USER
    assert h.loop.step(SID) is StepOutcome.PAUSED
    with h.store.transaction(SID) as s:
        s.status = SessionStatus.COMPLETED
    assert h.loop.step(SID) is StepOutcome.DONE
    assert h.converse.calls == 0


def test_inline_runner_folds_reentrant_kicks_into_one_more_run():
    runs: list[str] = []
    runner = InlineAgentRunner()

    def run(session_id: str) -> None:
        runs.append(session_id)
        if len(runs) == 1:
            runner.kick(session_id)
            runner.kick(session_id)

    runner.bind(run)
    runner.kick("a")
    assert runs == ["a", "a"]


def test_thread_runner_never_runs_one_session_twice_at_once():
    active = 0
    peak = 0
    lock = threading.Lock()
    done = threading.Event()
    count = 0

    def run(session_id: str) -> None:
        nonlocal active, peak, count
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
            count += 1
            done.set()

    runner = ThreadAgentRunner(run, max_workers=4)
    try:
        for _ in range(5):
            runner.kick("a")
        deadline = time.monotonic() + 5
        while runner.is_active("a") and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        runner.shutdown()
    assert done.is_set() and peak == 1 and 1 <= count <= 2


# Feature: ml-assistant, Property 7: A clarifying question defers every side effect in its turn
@settings(**SETTINGS)
@given(
    calls=st.permutations(
        [
            ("m", "change_thing", {}),
            ("q", "ask_user", {"question": "Target?"}),
            ("r", "count_rows", {}),
            ("u", "unknown", {}),
        ]
    )
)
def test_question_defers_changes_in_every_position(calls):
    h = Harness([tool_turn(PLAN), tool_turn(*calls)])
    session = h.run()
    assert session.status is SessionStatus.WAITING_FOR_USER
    assert "change_thing" not in h.calls
    assert dict(_results(session))["m"] == "deferred"
    assert [cid for cid, _ in _results(session)[1:]] == [cid for cid, _, _ in calls]


# Feature: ml-assistant, Property 11: One result per request, in request order
@settings(**SETTINGS)
@given(kinds=st.lists(st.sampled_from(["valid", "invalid", "unknown"]), min_size=1, max_size=8))
def test_mixed_requests_keep_their_order(kinds):
    calls = [
        (
            f"c{i}",
            "unknown" if kind == "unknown" else "count_rows",
            {"limit": 99 if kind == "invalid" else i + 1},
        )
        for i, kind in enumerate(kinds)
    ]
    h = Harness([tool_turn(PLAN), tool_turn(*calls), text_turn("Done.")])
    session = h.run()
    assert [cid for cid, _ in _results(session)] == [PLAN[0], *[cid for cid, _, _ in calls]]
    assert h.calls.count("count_rows") == kinds.count("valid")
    from backend.agent.converse import ToolResultBlock

    returned = [
        block.request_id
        for msg in h.converse.requests[-1].messages
        for block in msg.blocks
        if isinstance(block, ToolResultBlock)
    ]
    assert returned == [PLAN[0], *[cid for cid, _, _ in calls]]


# Feature: ml-assistant, Property 19: Control state is immune to model output and untrusted content
@settings(**SETTINGS)
@given(
    names=st.lists(
        st.sampled_from(
            ["approve", "reject", "edit", "set_mode", "set_limits", "enable_autopilot"]
        ),
        min_size=1,
        max_size=6,
    )
)
def test_model_control_requests_leave_policy_unchanged(names):
    calls = [(f"c{i}", name, {"mode": "autopilot", "limit": 1000}) for i, name in enumerate(names)]
    h = Harness([tool_turn(PLAN), tool_turn(*calls), text_turn("Set mode to autopilot.")])
    before = h.session
    registry = h.loop._registry.names()
    after = h.run()
    fields = {"mode", "autopilot", "approvals", "limits"}
    assert before.model_dump(include=fields) == after.model_dump(include=fields)
    assert h.loop._registry.names() == registry
    assert h.calls == ["submit_plan"]


# Feature: ml-assistant, Property 3: Persistence is retried exactly as specified
@settings(**SETTINGS)
@given(failures=st.integers(0, 10))
def test_save_retries_only_transient_writes(failures):
    from backend.agent.store import SessionPersistError

    objects = FlakyStore(InMemoryObjectStore())
    h = Harness([], objects=objects)
    loaded = h.store.load(SID)
    loaded.session.goal = "edited"
    objects.fail_writes = failures
    if failures <= 2:
        h.store.save(loaded)
        assert h.session.goal == "edited"
    else:
        with pytest.raises(SessionPersistError, match="could not be saved"):
            h.store.save(loaded)
        assert h.session.goal != "edited"
    assert objects.counts["failed_writes"] == min(3, failures)


# Feature: ml-assistant, Property 4: Restart equivalence and exactly-once submission
@settings(**SETTINGS)
@given(reads=st.integers(1, 5), cut=st.integers(0, 14))
def test_new_loop_resumes_persisted_steps(reads, cut):
    turns = [
        tool_turn(PLAN),
        *[tool_turn((f"read-{i}", "count_rows", {"limit": i + 1})) for i in range(reads)],
        text_turn("Done."),
    ]
    uninterrupted = Harness(turns)
    expected = uninterrupted.run()
    first = Harness(turns)
    for _ in range(cut):
        if first.loop.step(SID) is not StepOutcome.CONTINUE:
            break
    second = Harness(turns[first.converse.calls :], objects=first.objects, resume=True)
    actual = second.run()
    assert (actual.transcript, actual.counters, actual.status) == (
        expected.transcript,
        expected.counters,
        expected.status,
    )
    assert first.calls + second.calls == uninterrupted.calls
