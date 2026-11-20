"""Audit trail and audit-failure handling (task 11.3, design C11, Req 11)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import BaseModel, ConfigDict, Field

from backend.agent.audit import AgentAudit
from backend.agent.estimate import Estimate
from backend.agent.execute import ToolExecutor
from backend.agent.inputs import ToolInputs
from backend.agent.loop import CODE_AUDIT_FAILED, AgentLoop
from backend.agent.service import AgentService
from backend.agent.store import SessionStore
from backend.agent.tools import ToolFlags, ToolOutput, ToolRegistry
from backend.audit import AuditLog, AuditWriteError
from backend.auth import Principal
from backend.errors import AuthorizationError
from backend.governance import GovernancePolicy
from ml_engine.contracts.agent import (
    AutopilotGrant,
    CallState,
    Mode,
    Session,
    SessionLimits,
    SessionStatus,
    ToolClass,
)
from tests.support.agent_fakes import FakeClock
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.scripted_converse import (
    ScriptedFakeConverseClient,
    ScriptStep,
    text_turn,
    tool_turn,
)

ROOT = "s3://bucket-test/root"
EXP_PREFIX = "s3://bucket-test/root/experiments/exp-1"
SID = "sess-audit"
T0 = datetime(2025, 1, 1, tzinfo=UTC)
PLAN = ("tu-plan", "submit_plan", {"steps": ["train a model"]})
ALICE = Principal(user="alice")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanArgs(_Strict):
    steps: list[str] = Field(min_length=1, max_length=10)


class ExpArgs(_Strict):
    experiment_id: str
    n: int = Field(ge=1, le=10)


class CountArgs(_Strict):
    limit: int = Field(default=5, ge=1, le=10)


class _FlakyLog(AuditLog):
    """The real ``AuditLog``; writes of the listed actions fail."""

    def __init__(self, store, *, enabled: bool = True, fail: frozenset[str] = frozenset()):
        super().__init__(store, enabled=enabled)
        self.fail = fail

    def write(self, **kwargs):
        if kwargs["action"] in self.fail:
            raise AuditWriteError(f"audit write failed for action {kwargs['action']}")
        return super().write(**kwargs)


class _Authorizer:
    def require_owner(self, principal, experiment_id):
        return None


class _Estimator:
    def estimate(self, tool, args, ctx, *, estimator=None):
        return Estimate(1.0, "upper bound for compute only", "test-prices", "v1")


class _Runner:
    def kick(self, session_id: str) -> None:
        pass


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
        experiment_ids=["exp-1"],
    )


class Harness:
    def __init__(self, turns, *, mode=Mode.SUPERVISED, enabled=True, fail=frozenset()) -> None:
        self.mutations = 0
        self.reads = 0
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
            input_model=ExpArgs,
            handler=self._train,
            flags=ToolFlags(
                experiment_arg="experiment_id", estimator="training", audit_action="train"
            ),
        )
        registry.register(
            "mutate",
            classes=[ToolClass.MUTATING],
            description="mutating",
            input_model=ExpArgs,
            handler=self._mutate,
            flags=ToolFlags(experiment_arg="experiment_id", audit_action="mutate"),
        )
        registry.register(
            "count_rows",
            classes=[ToolClass.READ],
            description="read",
            input_model=CountArgs,
            handler=self._read,
        )
        self.objects = InMemoryObjectStore()
        self.clock = FakeClock(T0)  # every record gets the same timestamp
        self.log = _FlakyLog(self.objects, enabled=enabled, fail=frozenset(fail))
        self.audit = AgentAudit(self.log, ROOT)
        self.store = SessionStore(self.objects, ROOT, clock=self.clock)
        self.store.create(_session(mode))
        inputs = ToolInputs(audit=self.audit, authorizer=_Authorizer(), allowed_instance_types=[])
        services = SimpleNamespace(
            experiments=SimpleNamespace(get=lambda eid: SimpleNamespace(artifact_prefix=EXP_PREFIX)),
            audit=self.log,
        )
        self.loop = AgentLoop(
            store=self.store,
            converse=ScriptedFakeConverseClient(steps=[ScriptStep(t) for t in turns]),
            registry=registry,
            executor=ToolExecutor(inputs=inputs, services=services, clock=self.clock),
            inputs=inputs,
            audit=self.audit,
            governance=GovernancePolicy(
                send_sample_values=False, sample_values_per_column=0, sample_value_max_chars=0
            ),
            clock=self.clock,
            estimator=_Estimator(),
        )
        self.service = AgentService(
            settings=SimpleNamespace(agent_enabled=True),
            store=self.store,
            runner=_Runner(),
            audit=self.audit,
            clock=self.clock,
            registry=registry,
            inputs=inputs,
        )

    def _train(self, ctx, args):
        return ToolOutput(data={"submitted": True}, audit_detail={"n": str(args.n)})

    def _mutate(self, ctx, args):
        self.mutations += 1
        return ToolOutput(data={"changed": True}, audit_detail={"n": str(args.n)})

    def _read(self, ctx, args):
        self.reads += 1
        return ToolOutput(data={"rows": 3})

    def run(self) -> Session:
        self.loop.run(SID, "test-holder")
        return self.store.load(SID).session


def _executed(items):
    return [i for i in items if i.record.event_type == "tool_executed"]


def _call(s: Session, call_id: str):
    return next(r for r in s.tool_calls if r.tool_call_id == call_id)


def test_supervised_approval_names_the_approver_and_resolves_the_service_entry():
    h = Harness(
        [tool_turn(PLAN), tool_turn(("tu-t", "train", {"experiment_id": "exp-1", "n": 2})),
         text_turn("Training submitted.")]
    )
    s = h.run()
    approval = next(a for a in s.approvals if a.tool_call_id == "tu-t")
    h.service.decide_approval(ALICE, SID, approval.approval_id, "approve", args_hash=approval.args_hash)
    h.run()

    (item,) = _executed(h.service.audit_trail(ALICE, SID))
    rec = item.record
    assert (rec.tool, rec.args_hash, rec.result_status) == ("train", approval.args_hash, "success")
    assert (rec.decision, rec.decided_by) == ("approve", "alice")
    assert rec.prompt_injection_suspected is False
    # The service's own entry is returned exactly as stored.
    assert rec.service_audit_ref is not None
    assert item.service_entry == h.log.read(rec.service_audit_ref)
    entry = item.service_entry
    assert (entry.experiment_id, entry.action, entry.actor, entry.detail) == (
        "exp-1", "train", "alice", {"n": "2"}
    )


def test_autopilot_execution_names_the_enabling_user():
    h = Harness(
        [tool_turn(PLAN), tool_turn(("tu-t", "train", {"experiment_id": "exp-1", "n": 1})),
         text_turn("Done.")],
        mode=Mode.AUTOPILOT,
    )
    assert h.run().status is SessionStatus.COMPLETED
    (item,) = _executed(h.service.audit_trail(ALICE, SID))
    assert (item.record.decision, item.record.decided_by) == ("approve", "autopilot:alice")


def test_trail_is_ordered_by_seq_when_timestamps_are_equal():
    h = Harness(
        [tool_turn(PLAN),
         tool_turn(("tu-1", "mutate", {"experiment_id": "exp-1", "n": 1})),
         tool_turn(("tu-2", "mutate", {"experiment_id": "exp-1", "n": 2})),
         tool_turn(("tu-3", "mutate", {"experiment_id": "exp-1", "n": 3})),
         text_turn("Done.")]
    )
    h.run()
    records = [i.record for i in h.service.audit_trail(ALICE, SID)]
    assert len(records) == 3
    assert len({r.at for r in records}) == 1
    assert [r.seq for r in records] == [1, 2, 3]
    assert [r.detail["tool_call_id"] for r in records] == ["tu-1", "tu-2", "tu-3"]
    # Mutating without approval in Supervised mode: no decision to record.
    assert all(r.decision is None and r.decided_by is None for r in records)


def test_trail_is_empty_without_records_and_refused_for_another_user():
    h = Harness([])
    assert h.service.audit_trail(ALICE, SID) == []
    with pytest.raises(AuthorizationError):
        h.service.audit_trail(Principal(user="bob"), SID)


def test_failed_tool_executed_write_blocks_side_effects_but_not_reads():
    h = Harness(
        [tool_turn(PLAN),
         tool_turn(("tu-m1", "mutate", {"experiment_id": "exp-1", "n": 1})),
         tool_turn(
             ("tu-m2", "mutate", {"experiment_id": "exp-1", "n": 2}),
             ("tu-r", "count_rows", {"limit": 2}),
         ),
         text_turn("Done.")],
        fail={"agent.tool_executed"},
    )
    s = h.run()
    assert s.audit_failed
    assert _call(s, "tu-m1").state is CallState.SUCCEEDED
    second = _call(s, "tu-m2")
    assert (second.state, second.error_code) == (CallState.REFUSED, CODE_AUDIT_FAILED)
    assert _call(s, "tu-r").state is CallState.SUCCEEDED
    assert (h.mutations, h.reads) == (1, 1)
    # The owner is told, once, and the session and its records are kept.
    view = h.service.get_session(ALICE, SID)
    errors = [e for e in view.transcript if e.kind == "error"]
    assert [e.data["category"] for e in errors] == ["audit_failed"]
    assert view.status is SessionStatus.COMPLETED


def test_disabled_audit_writes_nothing_and_the_session_continues():
    h = Harness(
        [tool_turn(PLAN), tool_turn(("tu-m", "mutate", {"experiment_id": "exp-1", "n": 1})),
         text_turn("Done.")],
        enabled=False,
    )
    s = h.run()
    assert s.status is SessionStatus.COMPLETED and not s.audit_failed
    assert h.mutations == 1
    assert not [u for u in h.objects.list_uris(ROOT + "/") if "/audit/" in u]
    assert h.service.audit_trail(ALICE, SID) == []
