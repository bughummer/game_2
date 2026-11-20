"""Offline tests for the minimal agent service and its wiring (task 4.15, design C15)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from backend.agent.runner import InlineAgentRunner
from backend.agent.service import AgentService
from backend.agent.store import SessionInUse
from backend.agent.wiring import UnavailableConverseClient, build_agent_service
from backend.config import Settings
from backend.errors import AuthorizationError, ConflictError, FeatureDisabledError, ValidationError
from ml_engine.contracts.agent import (
    ApprovalRequest,
    ApprovalStatus,
    CallState,
    Lease,
    Mode,
    SessionStatus,
    ToolCallRecord,
    ToolClass,
)
from ml_engine.io.layout import AgentLayout
from tests.support.agent_fakes import CountingStore, FakeClock
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.scripted_converse import (
    ScriptedFakeConverseClient,
    ScriptStep,
    text_turn,
    tool_turn,
)

ROLE = "arn:aws:iam::123456789012:role/mlf-sagemaker-job"
T0 = datetime(2025, 1, 1, tzinfo=UTC)
ALICE = type("P", (), {"user": "alice"})()
BOB = type("P", (), {"user": "bob"})()
PLAN = ("tu-plan", "submit_plan", {"steps": ["look at the data"]})
ASK = ("tu-ask", "ask_user", {"question": "Which column is the target?"})


def _settings(**overrides) -> Settings:
    base = {
        "artifact_root": ARTIFACT_ROOT,
        "sagemaker_execution_role_arn": ROLE,
        "agent_enabled": True,
    }
    return Settings(**{**base, **overrides})


def _service(steps=(), *, runner=None, **overrides):
    objects = CountingStore(InMemoryObjectStore())
    converse = ScriptedFakeConverseClient(steps=[ScriptStep(t) for t in steps])
    runner = runner if runner is not None else InlineAgentRunner()
    service = build_agent_service(
        _settings(**overrides),
        store=objects,
        converse_client=converse,
        runner=runner,
        clock=FakeClock(T0),
    )
    return service, objects, converse, runner


def test_build_opens_nothing_and_writes_nothing():
    service, objects, _, _ = _service()
    assert isinstance(service, AgentService)
    assert objects.writes == 0
    disabled = build_agent_service(_settings(agent_enabled=False), store=InMemoryObjectStore())
    assert isinstance(disabled.loop._converse, UnavailableConverseClient)
    disabled.runner.shutdown()


def test_disabled_refuses_create_and_writes_nothing():
    service, objects, _, runner = _service(agent_enabled=False)
    with pytest.raises(FeatureDisabledError, match="ML_FACTORY_AGENT_ENABLED"):
        service.create_session(ALICE, "predict churn")
    assert objects.writes == 0
    assert runner.kicked == []


@pytest.mark.parametrize("goal", ["", "   \n\t", "x" * 4001, None])
def test_invalid_goal_creates_nothing(goal):
    service, objects, _, runner = _service()
    with pytest.raises(ValidationError, match="4,000"):
        service.create_session(ALICE, goal)
    assert objects.writes == 0 and runner.kicked == []


def test_invalid_limit_names_the_setting():
    service, objects, _, _ = _service(agent_limit_max_tool_calls=0)
    with pytest.raises(ValidationError, match="ML_FACTORY_AGENT_LIMIT_MAX_TOOL_CALLS"):
        service.create_session(ALICE, "predict churn")
    assert objects.writes == 0


@pytest.mark.parametrize("spend", [None, 0, -1, 10.5, float("nan")])
def test_autopilot_needs_spend_within_limit(spend):
    service, objects, _, _ = _service()
    with pytest.raises(ValidationError, match="maximum estimated spend"):
        service.create_session(
            ALICE, "predict churn", mode=Mode.AUTOPILOT, autopilot_max_spend_usd=spend
        )
    assert objects.writes == 0


def test_autopilot_refused_when_not_allowed():
    service, _, _, _ = _service(agent_autopilot_allowed=False)
    with pytest.raises(FeatureDisabledError):
        service.create_session(ALICE, "g", mode="autopilot", autopilot_max_spend_usd=1.0)


def test_create_runs_to_question_then_message_completes():
    service, objects, converse, runner = _service(
        [tool_turn(PLAN, ASK, text="Plan."), text_turn("Done: the target is churn.")]
    )
    view = service.create_session(ALICE, "predict churn")
    sid = view.session_id
    assert runner.kicked == [sid]
    assert view.owner == "alice" and view.mode is Mode.SUPERVISED
    assert view.status is SessionStatus.WAITING_FOR_USER
    assert view.transcript[0].kind == "user_message" and view.transcript[0].text == "predict churn"
    base = AgentLayout.for_session(_settings().artifact_root, sid).base
    assert any("agent.session_created" in uri or "audit" in uri for uri in objects.list_uris(base))

    with pytest.raises(AuthorizationError):
        service.send_message(BOB, sid, "churn")
    view = service.send_message(ALICE, sid, "churn")
    assert view.status is SessionStatus.COMPLETED
    assert any(e.kind == "user_message" and e.text == "churn" for e in view.transcript)
    assert len(converse.requests) == 2

    with pytest.raises(ConflictError):
        service.send_message(ALICE, sid, "again")


def test_autopilot_grant_recorded():
    service, _, _, _ = _service()  # no loop bound to a model step: script is empty
    service.runner.bind(None)
    view = service.create_session(
        ALICE, "predict churn", mode="autopilot", autopilot_max_spend_usd=2.5
    )
    s = service.store.load(view.session_id).session
    assert s.mode is Mode.AUTOPILOT
    assert s.autopilot is not None and s.autopilot.enabled_by == "alice"
    assert s.autopilot.max_estimated_spend_usd == 2.5


def test_ownership_on_reads_and_list():
    service, _, _, _ = _service()
    service.runner.bind(None)
    sid = service.create_session(ALICE, "predict churn").session_id
    with pytest.raises(AuthorizationError):
        service.get_session(BOB, sid)
    with pytest.raises(AuthorizationError):
        service.resume(BOB, sid)
    with pytest.raises(AuthorizationError):
        service.cancel(BOB, sid)
    assert [i.session_id for i in service.list_sessions(ALICE).items] == [sid]
    assert service.list_sessions(BOB).items == []


def test_resume_rejects_in_use_and_restarts_retriable_failure():
    service, _, _, runner = _service()
    runner.bind(None)
    sid = service.create_session(ALICE, "predict churn").session_id
    with service.store.transaction(sid) as s:
        s.lease = Lease(holder="other", expires_at=T0 + timedelta(seconds=60))
    with pytest.raises(SessionInUse):
        service.resume(ALICE, sid)
    with service.store.transaction(sid) as s:
        s.lease = None
        s.status = SessionStatus.FAILED
        s.retriable = True
    view = service.resume(ALICE, sid)
    assert view.status is SessionStatus.RUNNING and view.failure is None
    assert runner.kicked == [sid, sid]


def test_cancel_rejects_approvals_and_resolves_pending_uses():
    service, _, _, runner = _service()
    runner.bind(None)
    sid = service.create_session(ALICE, "predict churn").session_id
    with service.store.transaction(sid) as s:
        epoch = s.epoch
        s.status = SessionStatus.WAITING_FOR_APPROVAL
        for call_id, state in (("tu-1", CallState.AWAITING_APPROVAL), ("tu-2", CallState.REQUESTED)):
            s.tool_calls.append(
                ToolCallRecord(
                    tool_call_id=call_id, turn=1, tool="train", arguments={}, args_hash="h",
                    tool_class=ToolClass.COSTLY, state=state,
                )
            )
        s.pending_tool_use_ids = ["tu-1", "tu-2"]
        s.approvals.append(
            ApprovalRequest(
                approval_id="ap-1", kind="tool", tool_call_id="tu-1", tool="train",
                arguments={}, args_hash="h", rationale="", estimate_usd=None,
                estimate_basis=None, status=ApprovalStatus.PENDING, created_at=T0,
                expires_at=T0 + timedelta(seconds=900),
            )
        )
    view = service.cancel(ALICE, sid)
    assert view.status is SessionStatus.CANCELLED and view.pending_approvals == []
    s = service.store.load(sid).session
    assert s.epoch == epoch + 1 and s.pending_tool_use_ids == []
    assert [r.state for r in s.tool_calls] == [CallState.CANCELLED, CallState.CANCELLED]
    assert s.approvals[0].status is ApprovalStatus.REJECTED
    results = [e.tool_call_id for e in s.transcript if e.kind == "tool_result"]
    assert results == ["tu-1", "tu-2"]
    assert service.cancel(ALICE, sid).status is SessionStatus.CANCELLED  # idempotent
