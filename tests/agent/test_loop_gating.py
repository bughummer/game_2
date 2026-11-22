"""Offline tests for loop gating, approvals and the Costly path (task 8.6, design C6, C7, C13)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, Field

from backend.agent.estimate import Estimate
from backend.agent.execute import ToolExecutor
from backend.agent.inputs import ToolInputs
from backend.agent.loop import AgentLoop
from backend.agent.service import AgentService
from backend.agent.store import SessionStore
from backend.agent.tools import ToolFlags, ToolOutput, ToolRegistry
from backend.auth import Principal
from backend.governance import GovernancePolicy
from backend.orchestration.guards import GuardViolation
from ml_engine.contracts.agent import (
    ApprovalStatus,
    AutopilotGrant,
    CallState,
    Mode,
    Session,
    SessionLimits,
    SessionStatus,
    ToolClass,
    args_hash,
)
from ml_engine.io.layout import AgentLayout
from tests.agent.strategies import SETTINGS
from tests.support.agent_fakes import FakeClock
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.scripted_converse import (
    ScriptedFakeConverseClient,
    ScriptStep,
    text_turn,
    tool_turn,
)

ROOT = "s3://bucket-test/root"
SID = "sess-gate"
T0 = datetime(2025, 1, 1, tzinfo=UTC)
PLAN = ("tu-plan", "submit_plan", {"steps": ["train a model"]})
ALICE = Principal(user="alice")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanArgs(_Strict):
    steps: list[str] = Field(min_length=1, max_length=10)


class TrainArgs(_Strict):
    experiment_id: str
    n: int = Field(ge=1, le=10)


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
    def require_owner(self, principal, experiment_id):
        return None


class _Estimator:
    """Prices every Costly call at a fixed amount, labelled like the real estimator."""

    def __init__(self, usd: float | None) -> None:
        self.usd = usd

    def estimate(self, tool, args, ctx, *, estimator=None):
        return Estimate(self.usd, "upper bound for compute only", "test-prices", "v1")


class _Runner:
    def kick(self, session_id: str) -> None:  # the test runs the loop itself
        pass


def _session(*, mode: Mode = Mode.SUPERVISED, spend: float = 5.0) -> Session:
    grant = (
        AutopilotGrant(enabled_by="alice", enabled_at=T0, max_estimated_spend_usd=4.0)
        if mode is Mode.AUTOPILOT
        else None
    )
    return Session(
        session_id=SID,
        owner="alice",
        created_at=T0,
        updated_at=T0,
        last_activity_at=T0,
        status=SessionStatus.RUNNING,
        mode=mode,
        autopilot=grant,
        limits=SessionLimits(
            max_converse_turns=10,
            max_tool_calls=20,
            max_costly_tool_calls=3,
            max_estimated_spend_usd=spend,
            max_active_seconds=600,
            max_repeated_identical_calls=3,
        ),
        goal="predict churn",
        experiment_ids=["exp-1"],
    )


class Harness:
    def __init__(self, turns, *, mode=Mode.SUPERVISED, estimate=1.0, spend=5.0) -> None:
        self.train_args: list[dict] = []
        self.reads = 0
        self.guard_code: str | None = None
        registry = ToolRegistry()
        registry.register(
            "submit_plan",
            classes=[ToolClass.READ],
            description="plan",
            input_model=PlanArgs,
            handler=lambda ctx, a: ToolOutput(data={"steps": a.steps}),
            flags=ToolFlags(control=True, ledger_source=False),
        )
        registry.register(
            "train",
            classes=[ToolClass.COSTLY],
            description="costly",
            input_model=TrainArgs,
            handler=self._train,
            flags=ToolFlags(experiment_arg="experiment_id", estimator="training"),
        )
        registry.register(
            "count_rows",
            classes=[ToolClass.READ],
            description="read",
            input_model=CountArgs,
            handler=self._read,
        )
        self.registry = registry
        self.objects = InMemoryObjectStore()
        self.clock = FakeClock(T0)
        self.store = SessionStore(self.objects, ROOT, clock=self.clock)
        self.store.create(_session(mode=mode, spend=spend))
        self.audit = _Audit()
        self.inputs = ToolInputs(
            audit=self.audit, authorizer=_Authorizer(), allowed_instance_types=[]
        )
        self.converse = ScriptedFakeConverseClient(steps=[ScriptStep(t) for t in turns])
        self.loop = AgentLoop(
            store=self.store,
            converse=self.converse,
            registry=registry,
            executor=ToolExecutor(inputs=self.inputs, services=SimpleNamespace(), clock=self.clock),
            inputs=self.inputs,
            audit=self.audit,
            governance=GovernancePolicy(
                send_sample_values=False, sample_values_per_column=0, sample_value_max_chars=0
            ),
            clock=self.clock,
            estimator=_Estimator(estimate),
        )
        self.service = AgentService(
            settings=SimpleNamespace(agent_enabled=True),
            store=self.store,
            runner=_Runner(),
            audit=self.audit,
            clock=self.clock,
            registry=registry,
            inputs=self.inputs,
        )

    def _train(self, ctx, args):
        self.train_args.append(args.model_dump())
        if self.guard_code is not None:
            raise GuardViolation(self.guard_code, "The budget guard refused the job.")
        return ToolOutput(data={"submitted": True})

    def _read(self, ctx, args):
        self.reads += 1
        return ToolOutput(data={"rows": 3})

    def run(self) -> Session:
        self.loop.run(SID, "test-holder")
        return self.session

    @property
    def session(self) -> Session:
        return self.store.load(SID).session


def _call(s: Session, call_id: str):
    return next(r for r in s.tool_calls if r.tool_call_id == call_id)


def _approval_for(s: Session, call_id: str):
    return next(a for a in s.approvals if a.tool_call_id == call_id)


# Feature: ml-assistant, Property 14: The approved call is exactly the shown call, run at most once
@settings(**SETTINGS)
@given(original=st.integers(1, 5), replacement=st.integers(6, 10), replays=st.integers(1, 5))
def test_approved_call_runs_once_with_the_shown_arguments(original, replacement, replays):
    h = Harness(
        [
            tool_turn(PLAN),
            tool_turn(("tu-t", "train", {"experiment_id": "exp-1", "n": original})),
            text_turn("Training was submitted."),
        ]
    )
    s = h.run()
    assert s.status is SessionStatus.WAITING_FOR_APPROVAL
    approval = _approval_for(s, "tu-t")
    assert (approval.status, approval.estimate_usd) == (ApprovalStatus.PENDING, 1.0)
    assert "test-prices" in approval.estimate_basis
    assert h.train_args == []

    # The user edits the arguments, then approves exactly what is now shown.
    assert approval.arguments == {"experiment_id": "exp-1", "n": original}
    edited = {"experiment_id": "exp-1", "n": replacement}
    h.service.decide_approval(
        ALICE, SID, approval.approval_id, "edit", args_hash=approval.args_hash, edited_args=edited
    )
    shown = args_hash("train", edited)
    h.service.decide_approval(ALICE, SID, approval.approval_id, "approve", args_hash=shown)
    s = h.run()

    assert h.train_args == [edited]
    record = _call(s, "tu-t")
    assert record.state is CallState.SUCCEEDED
    assert _approval_for(s, "tu-t").status is ApprovalStatus.EXECUTED
    assert (s.counters.costly_tool_calls, s.counters.estimated_spend_usd) == (1, 1.0)
    marker = json.loads(
        h.objects.read_bytes(AgentLayout.for_session(ROOT, SID).inflight(record.idempotency_key))
    )
    assert (marker["tool"], marker["args_hash"], marker["experiment_id"]) == (
        "train",
        shown,
        "exp-1",
    )
    assert s.status is SessionStatus.COMPLETED

    for _ in range(replays):
        h.run()
    assert len(h.train_args) == 1


def test_pending_approval_pauses_and_batch_requests_are_precreated():
    h = Harness(
        [
            tool_turn(PLAN),
            tool_turn(
                ("tu-a", "train", {"experiment_id": "exp-1", "n": 1}),
                ("tu-b", "train", {"experiment_id": "exp-1", "n": 2}),
            ),
        ]
    )
    s = h.run()
    assert s.status is SessionStatus.WAITING_FOR_APPROVAL
    assert [_approval_for(s, c).status for c in ("tu-a", "tu-b")] == [ApprovalStatus.PENDING] * 2
    assert h.train_args == []

    first = _approval_for(s, "tu-a")
    h.service.decide_approval(ALICE, SID, first.approval_id, "approve", args_hash=first.args_hash)
    s = h.run()
    # The approved call ran; the loop paused again on the next call that is still pending.
    assert h.train_args == [{"experiment_id": "exp-1", "n": 1}]
    assert s.status is SessionStatus.WAITING_FOR_APPROVAL
    assert _call(s, "tu-b").state is CallState.AWAITING_APPROVAL
    assert s.pending_tool_use_ids == ["tu-b"]


def test_costly_call_over_the_spend_cap_gets_limit_reached_and_no_approval():
    h = Harness(
        [
            tool_turn(PLAN),
            tool_turn(("tu-t", "train", {"experiment_id": "exp-1", "n": 2})),
            text_turn("The limit was reached."),
        ],
        estimate=10.0,
        spend=5.0,
    )
    s = h.run()
    record = _call(s, "tu-t")
    assert (record.state, record.error_code) == (CallState.REFUSED, "LIMIT_REACHED")
    assert h.train_args == [] and s.approvals == []
    error = h.store.get_result(SID, record.result_ref)["error"]
    assert error["details"]["limit"] == "max_estimated_spend_usd"


def test_guard_refusal_is_recorded_and_not_retried_with_changed_arguments():
    h = Harness(
        [
            tool_turn(PLAN),
            tool_turn(("tu-1", "train", {"experiment_id": "exp-1", "n": 2})),
            tool_turn(("tu-2", "train", {"experiment_id": "exp-1", "n": 1})),
        ],
        mode=Mode.AUTOPILOT,
    )
    h.guard_code = "BUDGET_EXCEEDED"
    s = h.run()
    assert _call(s, "tu-1").error_code == "BUDGET_EXCEEDED"
    assert [(g.tool, g.experiment_id, g.code) for g in s.guard_refusals] == [
        ("train", "exp-1", "BUDGET_EXCEEDED")
    ]
    # The changed retry is not run unattended: it waits for the user.
    assert len(h.train_args) == 1
    assert _call(s, "tu-2").state is CallState.AWAITING_APPROVAL
    assert s.status is SessionStatus.WAITING_FOR_APPROVAL


def test_repeated_identical_call_is_refused_and_counted():
    same = {"limit": 2}
    h = Harness(
        [
            tool_turn(PLAN),
            tool_turn(*[(f"tu-{i}", "count_rows", same) for i in range(4)]),
            text_turn("Done."),
        ]
    )
    s = h.run()
    assert h.reads == 3
    record = _call(s, "tu-3")
    assert (record.state, record.error_code) == (CallState.REFUSED, "REPEATED_CALL")
    assert s.counters.tool_calls == 5  # the plan plus four uses, including the refused one


def test_expired_approval_is_swept_at_the_next_step():
    h = Harness(
        [tool_turn(PLAN), tool_turn(("tu-t", "train", {"experiment_id": "exp-1", "n": 2}))]
    )
    h.run()
    h.clock.advance(901)
    h.loop.step(SID)
    s = h.session
    assert _approval_for(s, "tu-t").status is ApprovalStatus.EXPIRED
    assert _call(s, "tu-t").error_code == "APPROVAL_EXPIRED"
    assert s.status is SessionStatus.WAITING_FOR_USER
    assert h.train_args == []


# Feature: ml-assistant, Property 23: Counters are bounded and exact
@settings(**SETTINGS)
@given(limit=st.integers(1, 5), requested=st.integers(1, 8))
def test_costly_counter_and_spend_match_executions(limit, requested):
    calls = [(f"tu-{i}", "train", {"experiment_id": "exp-1", "n": i + 1}) for i in range(requested)]
    h = Harness(
        [tool_turn(PLAN), tool_turn(*calls), text_turn("Done.")], mode=Mode.AUTOPILOT, estimate=0.25
    )
    with h.store.transaction(SID) as session:
        session.limits.max_costly_tool_calls = limit
    s = h.run()
    expected = min(limit, requested)
    assert len(h.train_args) == s.counters.costly_tool_calls == expected
    assert s.counters.estimated_spend_usd == expected * 0.25
    assert s.counters.tool_calls <= s.limits.max_tool_calls
    assert s.counters.converse_turns <= s.limits.max_converse_turns


def test_autopilot_preserves_selection_and_budget_after_guard_refusal():

    from backend.agent.tools_features import DecideFeaturesArgs, decide_features_protected

    selection = ["age", "tenure"]
    attempts = [
        ("first", "train", {"experiment_id": "exp-1", "n": 1}),
        ("retry", "train", {"experiment_id": "exp-1", "n": 2}),
        ("drop", "decide_features", {"experiment_id": "exp-1", "selected_features": ["age"]}),
    ]
    h = Harness([tool_turn(PLAN), tool_turn(*attempts), text_turn("Done.")], mode=Mode.AUTOPILOT)
    h.guard_code = "BUDGET_EXCEEDED"
    review = SimpleNamespace(features=[SimpleNamespace(feature=name) for name in selection])
    h.loop._services = SimpleNamespace(
        experiments=SimpleNamespace(feature_review=lambda eid: review)
    )

    def save(ctx, args):
        selection[:] = args.selected_features
        return ToolOutput(data={})

    h.registry.register(
        "decide_features",
        classes=[ToolClass.MUTATING],
        description="choose features",
        input_model=DecideFeaturesArgs,
        handler=save,
        protected=decide_features_protected,
    )
    session = h.run()
    assert _call(session, "first").error_code == "BUDGET_EXCEEDED"
    for call_id in ("retry", "drop"):
        approval = _approval_for(session, call_id)
        assert approval.status is ApprovalStatus.PENDING
        h.service.decide_approval(
            ALICE, SID, approval.approval_id, "reject", args_hash=approval.args_hash
        )
        session = h.run()
    assert h.train_args == [attempts[0][2]]
    assert session.counters.estimated_spend_usd < session.autopilot.max_estimated_spend_usd
    assert selection == ["age", "tenure"]


def test_changed_persisted_arguments_require_a_new_approval():
    args = {"experiment_id": "exp-1", "n": 1}
    h = Harness([tool_turn(PLAN), tool_turn(("t", "train", args)), text_turn("Done.")])
    approval = _approval_for(h.run(), "t")
    h.service.decide_approval(
        ALICE, SID, approval.approval_id, "approve", args_hash=approval.args_hash
    )
    # An inconsistent persisted record still has the old digest.
    with h.store.transaction(SID) as session:
        _call(session, "t").arguments["n"] = 2
    session = h.run()
    assert _call(session, "t").error_code == "APPROVAL_MISMATCH"
    assert h.train_args == []
    assert session.counters.costly_tool_calls == 0


def test_failed_costly_call_cannot_reuse_earlier_approval():
    args = {"experiment_id": "exp-1", "n": 1}
    h = Harness(
        [
            tool_turn(PLAN),
            tool_turn(("first", "train", args)),
            tool_turn(("retry", "train", args)),
            text_turn("Done."),
        ]
    )
    h.guard_code = "BUDGET_EXCEEDED"
    first = _approval_for(h.run(), "first")
    h.service.decide_approval(ALICE, SID, first.approval_id, "approve", args_hash=first.args_hash)
    session = h.run()
    second = _approval_for(session, "retry")
    assert (second.status, session.status) == (
        ApprovalStatus.PENDING,
        SessionStatus.WAITING_FOR_APPROVAL,
    )
    assert first.approval_id != second.approval_id and first.args_hash == second.args_hash
    assert _call(session, "first").error_code == "BUDGET_EXCEEDED"
    assert h.train_args == [args]


# Feature: ml-assistant, Property 20: Autopilot enablement and switching
@settings(**SETTINGS)
@given(original=st.integers(1, 10), following=st.integers(1, 10))
def test_switch_during_costly_handler_applies_to_the_next_call(original, following):
    first = {"experiment_id": "exp-1", "n": original}
    second = {"experiment_id": "exp-1", "n": following}
    from unittest.mock import patch

    train = Harness._train

    def switch_and_train(harness, ctx, args):
        harness.service.set_mode(ALICE, SID, Mode.SUPERVISED)
        return train(harness, ctx, args)

    with patch.object(Harness, "_train", switch_and_train):
        h = Harness(
            [tool_turn(PLAN), tool_turn(("one", "train", first), ("two", "train", second))],
            mode=Mode.AUTOPILOT,
        )
        session = h.run()
    assert h.train_args == [first]
    assert _call(session, "one").state is CallState.SUCCEEDED
    assert _approval_for(session, "two").status is ApprovalStatus.PENDING


# Feature: ml-assistant, Property 21: Autopilot executes only within every limit
@settings(**SETTINGS)
@given(cents=st.integers(1, 100), count=st.integers(2, 6))
def test_decimal_spend_allows_the_last_affordable_call(cents, count):
    cap = cents * count / 100
    calls = [(f"c{i}", "train", {"experiment_id": "exp-1", "n": i + 1}) for i in range(count + 1)]
    h = Harness(
        [tool_turn(PLAN), tool_turn(*calls), text_turn("Done.")],
        mode=Mode.AUTOPILOT,
        estimate=cents / 100,
        spend=cap,
    )
    with h.store.transaction(SID) as s:
        s.autopilot.max_estimated_spend_usd = cap
        s.limits.max_costly_tool_calls = count + 1
    session = h.run()
    assert h.train_args == [args for _, _, args in calls[:-1]]
    assert session.counters.estimated_spend_usd == cap
    assert _call(session, calls[-1][0]).error_code == "LIMIT_REACHED"
