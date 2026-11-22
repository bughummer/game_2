"""Offline tests for job waiting, job refresh, the watcher and in-flight recovery (task 8.7).

Design C13, C15, D11, D13; Req 6.3 to 6.6, 14.1, 14.5, 14.6, 14.11.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, Field

from backend.agent.converse import TextBlock
from backend.agent.estimate import Estimate
from backend.agent.execute import ToolExecutor
from backend.agent.inputs import ToolInputs
from backend.agent.loop import AgentLoop, LoopSettings
from backend.agent.service import AgentService
from backend.agent.store import SessionStore
from backend.agent.tools import AwaitedJobSpec, ToolFlags, ToolOutput, ToolRegistry
from backend.agent.watcher import AgentJobWatcher
from backend.auth import Principal
from backend.errors import NotFoundError
from backend.governance import GovernancePolicy
from backend.schemas import ExecutionStatusView
from ml_engine.contracts.agent import (
    ApprovalStatus,
    AutopilotGrant,
    CallState,
    Mode,
    Session,
    SessionLimits,
    SessionStatus,
    ToolClass,
)
from ml_engine.contracts.execution import ExecutionRecord, WorkflowKind
from ml_engine.io import ExperimentLayout, write_model
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
SID = "sess-jobs"
T0 = datetime(2025, 1, 1, tzinfo=UTC)
EXP = "exp-1"
PREFIX = f"{ROOT}/ml-factory/experiments/{EXP}"
PLAN = ("tu-plan", "submit_plan", {"steps": ["train a model"]})
TRAIN = ("tu-t", "start_training", {"experiment_id": EXP})
ALICE = Principal(user="alice")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanArgs(_Strict):
    steps: list[str] = Field(min_length=1, max_length=10)


class TrainArgs(_Strict):
    experiment_id: str


class _Crash(BaseException):
    """Stands in for the process stopping while a handler runs."""


class _Audit:
    def write(self, session, rec):
        session.audit_seq += 1
        return f"a-{session.audit_seq}"


class _Authorizer:
    def require_owner(self, principal, experiment_id):
        return None


class _Estimator:
    def estimate(self, tool, args, ctx, *, estimator=None):
        return Estimate(1.0, "upper bound", "test-prices", "v1")


class _Runner:
    def kick(self, session_id: str) -> None:  # the tests run the loop themselves
        pass


class _Analysis:
    """``execution_status`` from a table: a status string, or ``None`` for not found."""

    def __init__(self) -> None:
        self.states: dict[str, tuple[str | None, str | None]] = {}

    def execution_status(self, experiment_id, execution_id=None):
        status, message = self.states.get(execution_id, ("RUNNING", None))
        if status is None:
            raise NotFoundError(f"No execution '{execution_id}'.")
        return ExecutionStatusView(
            execution_id=execution_id,
            kind="training",
            status=status,
            stage="train",
            started_at=T0,
            updated_at=T0,
            failure_message=message,
        )


class _Repository:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay

    def get(self, experiment_id):
        if self.delay:
            time.sleep(self.delay)
        return SimpleNamespace(artifact_prefix=PREFIX)


def _session(mode: Mode) -> Session:
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
            max_estimated_spend_usd=5.0,
            max_active_seconds=600,
            max_repeated_identical_calls=3,
        ),
        goal="predict churn",
        experiment_ids=[EXP],
    )


def _extract(data: dict) -> list[AwaitedJobSpec]:
    return [AwaitedJobSpec(execution_id=i, experiment_id=EXP, kind="training") for i in data["ids"]]


class Harness:
    def __init__(
        self, turns, *, mode=Mode.AUTOPILOT, job_ids=("exec-1",), timeout=30.0, existing_objects=None
    ) -> None:
        self.calls = 0
        self.job_ids = list(job_ids)
        self.crash = False
        self.submit_on_crash = False
        self.objects = existing_objects if existing_objects is not None else InMemoryObjectStore()
        self.clock = FakeClock(T0)
        self.analysis = _Analysis()
        self.repository = _Repository()
        services = SimpleNamespace(
            analysis=self.analysis, repository=self.repository, store=self.objects
        )
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
            "start_training",
            classes=[ToolClass.COSTLY],
            description="costly",
            input_model=TrainArgs,
            handler=self._train,
            flags=ToolFlags(experiment_arg="experiment_id", estimator="training"),
            job_extractor=_extract,
        )
        self.store = SessionStore(self.objects, ROOT, clock=self.clock)
        if existing_objects is None:
            self.store.create(_session(mode))
        inputs = ToolInputs(audit=_Audit(), authorizer=_Authorizer(), allowed_instance_types=[])
        self.converse = ScriptedFakeConverseClient(steps=[ScriptStep(t) for t in turns])
        self.watcher = AgentJobWatcher(self.store, self.analysis, clock=self.clock)
        self.loop = AgentLoop(
            store=self.store,
            converse=self.converse,
            registry=registry,
            executor=ToolExecutor(inputs=inputs, services=services, clock=self.clock),
            inputs=inputs,
            audit=_Audit(),
            governance=GovernancePolicy(
                send_sample_values=False, sample_values_per_column=0, sample_value_max_chars=0
            ),
            settings=LoopSettings(reconcile_timeout_seconds=timeout),
            services=services,
            clock=self.clock,
            estimator=_Estimator(),
            on_waiting_for_job=self.watcher.register,
        )
        self.service = AgentService(
            settings=SimpleNamespace(agent_enabled=True),
            store=self.store,
            runner=_Runner(),
            audit=_Audit(),
            clock=self.clock,
        )

    def _train(self, ctx, args):
        self.calls += 1
        if self.crash:
            if self.submit_on_crash:
                self.write_execution("exec-new", self.clock() + timedelta(seconds=1))
            raise _Crash()
        first, *rest = self.job_ids
        awaited = (AwaitedJobSpec(execution_id=first, experiment_id=EXP, kind="training"),)
        # The handler reports the first job directly; the extractor reports all of them.
        return ToolOutput(data={"ids": [first, *rest]}, awaited=awaited)

    def write_execution(self, execution_id, started_at, kind=WorkflowKind.TRAINING) -> None:
        layout = ExperimentLayout(base=PREFIX)
        record = ExecutionRecord(
            execution_id=execution_id,
            experiment_id=EXP,
            kind=kind,
            artifact_prefix=PREFIX,
            started_at=started_at,
            updated_at=started_at,
            stage="train",
        )
        write_model(self.objects, layout.execution(execution_id), record)

    def run(self) -> Session:
        self.loop.run(SID, "test-holder")
        return self.session

    @property
    def session(self) -> Session:
        return self.store.load(SID).session


def _call(s: Session, call_id: str):
    return next(r for r in s.tool_calls if r.tool_call_id == call_id)


def _job_updates(request) -> list[dict]:
    found = []
    for message in request.messages:
        for block in message.blocks:
            if isinstance(block, TextBlock) and '"job_update"' in block.text:
                found.append(json.loads(block.text)["untrusted_data"])
    return found


# ---------------------------------------------------------------------- job waiting


def test_started_job_pauses_in_waiting_for_job_without_a_model_call():
    h = Harness([tool_turn(PLAN), tool_turn(TRAIN)])
    s = h.run()
    assert s.status is SessionStatus.WAITING_FOR_JOB
    assert h.converse.calls == 2  # plan + the training turn; nothing while waiting
    # Reported both by ToolOutput.awaited and the job_extractor, recorded once.
    assert [j.execution_id for j in s.awaited_jobs] == ["exec-1"]
    assert s.jobs_ready is False
    assert h.watcher.active == {SID}

    h.run()  # running again while the job runs never calls the model
    h.watcher.sweep_jobs()
    assert h.converse.calls == 2
    assert h.session.status is SessionStatus.WAITING_FOR_JOB


def test_terminal_states_deliver_one_enveloped_update_each_on_resume():
    ids = ("exec-ok", "exec-bad", "exec-stop", "exec-gone")
    h = Harness([tool_turn(PLAN), tool_turn(TRAIN), text_turn("Done.")], job_ids=ids)
    h.run()
    assert h.session.status is SessionStatus.WAITING_FOR_JOB

    h.analysis.states.update(
        {
            "exec-ok": ("SUCCEEDED", None),
            "exec-bad": ("FAILED", "Out of memory"),
            "exec-stop": ("STOPPED", None),
            "exec-gone": (None, None),
        }
    )
    assert h.watcher.sweep_jobs() == 1
    s = h.session
    assert s.jobs_ready is True
    assert s.status is SessionStatus.WAITING_FOR_JOB  # the watcher never resumes
    assert h.converse.calls == 2  # and never calls the model
    assert h.watcher.active == set()

    h.service.resume(ALICE, SID)
    s = h.run()
    assert s.status is SessionStatus.COMPLETED
    updates = _job_updates(h.converse.requests[-1])
    by_id = {u["data"]["execution_id"]: u["data"] for u in updates}
    assert len(updates) == 4 and set(by_id) == set(ids)
    assert by_id["exec-ok"]["state"] == "succeeded"
    assert (by_id["exec-bad"]["state"], by_id["exec-bad"]["failure_reason"]) == (
        "failed",
        "Out of memory",
    )
    assert by_id["exec-stop"]["state"] == "stopped"
    assert (by_id["exec-gone"]["state"], by_id["exec-gone"]["error_code"]) == (
        "not_found",
        "JOB_NOT_FOUND",
    )
    assert all(j.delivered for j in s.awaited_jobs) and s.jobs_ready is False


# ---------------------------------------------------------------------- recovery


def _crash_after_approval(h: Harness) -> None:
    s = h.run()
    approval = next(a for a in s.approvals if a.tool_call_id == "tu-t")
    h.service.decide_approval(ALICE, SID, approval.approval_id, "approve", args_hash=approval.args_hash)
    h.crash = True
    with pytest.raises(_Crash):
        h.run()
    assert _call(h.session, "tu-t").state is CallState.EXECUTING


# Feature: ml-assistant, Property 4: Restart equivalence and exactly-once submission
@settings(**SETTINGS)
@given(age=st.integers(1, 86400), replays=st.integers(1, 4))
def test_crash_during_costly_call_adopts_exactly_one_execution_and_never_resubmits(age, replays):
    h = Harness([tool_turn(PLAN), tool_turn(TRAIN)], mode=Mode.SUPERVISED)
    h.write_execution("exec-old", T0 - timedelta(seconds=age))  # started before the marker
    h.write_execution("exec-eda", T0 + timedelta(seconds=2), WorkflowKind.EDA)  # other kind
    h.submit_on_crash = True
    _crash_after_approval(h)

    for _ in range(replays):
        restarted = Harness([], existing_objects=h.objects)
        s = restarted.run()
        assert restarted.calls == restarted.converse.calls == 0
    assert h.calls == 1  # never executed again
    record = _call(s, "tu-t")
    assert record.state is CallState.SUCCEEDED
    assert [j.execution_id for j in s.awaited_jobs] == ["exec-new"]
    assert (
        next(a for a in s.approvals if a.tool_call_id == "tu-t").status is ApprovalStatus.EXECUTED
    )
    assert s.counters.costly_tool_calls == 1  # not counted again
    assert s.status is SessionStatus.WAITING_FOR_JOB
    assert h.converse.calls == 2


def test_crash_without_a_submission_reports_not_submitted():
    h = Harness([tool_turn(PLAN), tool_turn(TRAIN), text_turn("It was not submitted.")],
                mode=Mode.SUPERVISED)
    _crash_after_approval(h)

    s = h.run()
    assert h.calls == 1
    record = _call(s, "tu-t")
    assert (record.state, record.error_code) == (CallState.FAILED, "NOT_SUBMITTED")
    approval = next(a for a in s.approvals if a.tool_call_id == "tu-t")
    assert approval.status is ApprovalStatus.CANCELLED  # a new decision is needed
    assert s.awaited_jobs == []
    assert s.counters.costly_tool_calls == 1
    assert s.status is SessionStatus.COMPLETED


def test_slow_platform_query_gives_unknown_outcome():
    h = Harness([tool_turn(PLAN), tool_turn(TRAIN), text_turn("Outcome unknown.")],
                mode=Mode.SUPERVISED, timeout=0.05)
    _crash_after_approval(h)
    h.repository.delay = 0.5

    s = h.run()
    assert h.calls == 1
    record = _call(s, "tu-t")
    assert (record.state, record.error_code) == (CallState.UNKNOWN_OUTCOME, "UNKNOWN_OUTCOME")


# Feature: ml-assistant, Property 27: Waiting holds nothing open, and job results are delivered
@settings(**SETTINGS)
@given(
    states=st.lists(
        st.sampled_from(["SUCCEEDED", "FAILED", "STOPPED", None]), min_size=1, max_size=5
    )
)
def test_partial_job_completion_waits_for_the_last_job(states):
    ids = [f"exec-{i}" for i in range(len(states))]
    h = Harness([tool_turn(PLAN), tool_turn(TRAIN), text_turn("Done.")], job_ids=ids)
    h.run()
    for i, (job, state) in enumerate(zip(ids, states, strict=True)):
        h.analysis.states[job] = (state, "bad job" if state == "FAILED" else None)
        h.watcher.sweep_jobs()
        assert h.session.jobs_ready == (i + 1 == len(states))
        assert h.converse.calls == 2
    h.service.resume(ALICE, SID)
    h.run()
    updates = [u["data"] for u in _job_updates(h.converse.requests[-1])]
    assert [(u["execution_id"], u["state"]) for u in updates] == [
        (job, state.lower() if state else "not_found")
        for job, state in zip(ids, states, strict=True)
    ]


# Feature: ml-assistant, Property 11: One result per request, in request order
@settings(**SETTINGS)
@given(requests=st.integers(2, 3))
def test_unfinished_job_refuses_later_submissions_in_order(requests):
    from backend.agent.loop import CODE_JOB_PENDING

    calls = [(f"train-{i}", "start_training", {"experiment_id": EXP}) for i in range(requests)]
    h = Harness([tool_turn(PLAN), tool_turn(*calls)])
    s = h.run()
    assert s.status is SessionStatus.WAITING_FOR_JOB and h.calls == 1
    assert [e.tool_call_id for e in s.transcript if e.kind == "tool_result"] == [
        PLAN[0],
        *[cid for cid, _, _ in calls],
    ]
    assert [_call(s, cid).error_code for cid, _, _ in calls[1:]] == [CODE_JOB_PENDING] * (requests-1)
