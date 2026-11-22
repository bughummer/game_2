"""Offline tests for cancel with job stopping, job refresh on read and decision order (task 8.8).

Design C15; Req 2.6, 6.5, 6.6, 6.12, 6.13, 14.1.
"""

from __future__ import annotations

from types import SimpleNamespace

from hypothesis import given, settings
from hypothesis import strategies as st

from backend.agent.service import CODE_APPROVAL_REJECTED, CODE_CANCELLED, AgentService
from ml_engine.contracts.agent import ApprovalStatus, CallState, Mode, SessionStatus
from tests.agent.strategies import SETTINGS
from tests.agent.test_jobs_recovery import ALICE, EXP, PLAN, SID, TRAIN, Harness, _call
from tests.support.scripted_converse import text_turn, tool_turn


class _Orchestrator:
    def __init__(self) -> None:
        self.stopped: list[str] = []

    def stop(self, execution_id: str) -> None:
        self.stopped.append(execution_id)


class _RecordingAudit:
    def __init__(self) -> None:
        self.records = []

    def write(self, session, record):
        session.audit_seq += 1
        self.records.append(record)
        return f"a-{session.audit_seq}"


def _service(h: Harness) -> tuple[AgentService, _Orchestrator, _RecordingAudit]:
    orchestrator, audit = _Orchestrator(), _RecordingAudit()
    service = AgentService(
        settings=SimpleNamespace(agent_enabled=True),
        store=h.store,
        runner=SimpleNamespace(kick=lambda sid: None),
        audit=audit,
        clock=h.clock,
        analysis=h.analysis,
        orchestrator=orchestrator,
    )
    return service, orchestrator, audit


def _waiting_on_jobs() -> Harness:
    h = Harness([tool_turn(PLAN), tool_turn(TRAIN)], job_ids=("exec-1", "exec-2"))
    h.run()
    assert h.session.status is SessionStatus.WAITING_FOR_JOB
    return h


def test_cancel_by_default_leaves_jobs_running():
    h = _waiting_on_jobs()
    service, orchestrator, audit = _service(h)
    view = service.cancel(ALICE, SID)
    assert view.status is SessionStatus.CANCELLED
    assert orchestrator.stopped == []
    assert [j.execution_id for j in view.running_jobs] == ["exec-1", "exec-2"]
    (record,) = audit.records
    assert record.event_type == "cancelled" and record.detail["stop_jobs"] == "false"
    assert record.detail["running_jobs"] == "2"


def test_cancel_with_stop_jobs_stops_each_unfinished_job_and_audits_it():
    h = _waiting_on_jobs()
    h.analysis.states["exec-1"] = ("SUCCEEDED", None)
    service, orchestrator, audit = _service(h)
    service.get_session(ALICE, SID)  # records exec-1 as succeeded
    view = service.cancel(ALICE, SID, stop_jobs=True)
    assert view.status is SessionStatus.CANCELLED
    assert orchestrator.stopped == ["exec-2"]  # the ended job is not stopped again
    stops = [r for r in audit.records if r.event_type == "job_stopped"]
    assert [(r.detail["execution_id"], r.result_status) for r in stops] == [("exec-2", "success")]
    assert stops[0].actor == "alice"
    cancelled = next(r for r in audit.records if r.event_type == "cancelled")
    assert cancelled.detail["stop_jobs"] == "true"


def test_get_session_refreshes_jobs_and_shows_jobs_ready():
    h = _waiting_on_jobs()
    service, _, _ = _service(h)
    revision = h.session.revision
    view = service.get_session(ALICE, SID)
    assert view.jobs_ready is False and h.session.revision == revision  # nothing ended: no save

    h.analysis.states.update({"exec-1": ("SUCCEEDED", None), "exec-2": ("FAILED", "OOM")})
    view = service.get_session(ALICE, SID)
    assert view.jobs_ready is True and view.running_jobs == []
    s = h.session
    assert [(j.state, j.failure_reason) for j in s.awaited_jobs] == [
        ("succeeded", None),
        ("failed", "OOM"),
    ]
    revision = s.revision
    service.get_session(ALICE, SID)
    assert h.session.revision == revision  # already recorded: no further save


def test_rejecting_a_later_precreated_request_keeps_request_order():
    first = ("tu-1", "start_training", {"experiment_id": EXP})
    second = ("tu-2", "start_training", {"experiment_id": EXP})
    h = Harness([tool_turn(PLAN), tool_turn(first, second), text_turn("Done.")],
                mode=Mode.SUPERVISED)
    h.run()
    s = h.session
    assert s.status is SessionStatus.WAITING_FOR_APPROVAL
    ap1, ap2 = (_call(s, c).approval_id for c in ("tu-1", "tu-2"))
    assert ap1 and ap2  # tu-2's request was pre-created

    service, _, _ = _service(h)
    hashes = {a.approval_id: a.args_hash for a in s.approvals}
    service.decide_approval(ALICE, SID, ap2, "reject", args_hash=hashes[ap2], reason="too big")
    s = h.session
    assert next(a for a in s.approvals if a.approval_id == ap2).status is ApprovalStatus.REJECTED
    assert s.pending_tool_use_ids == ["tu-1", "tu-2"]  # tu-2 waits for its turn
    assert [e.tool_call_id for e in s.transcript if e.kind == "tool_result"] == ["tu-plan"]

    service.decide_approval(ALICE, SID, ap1, "reject", args_hash=hashes[ap1])
    h.run()
    s = h.session
    assert h.calls == 0  # neither call ran
    results = [e.tool_call_id for e in s.transcript if e.kind == "tool_result"]
    assert results == ["tu-plan", "tu-1", "tu-2"]
    assert _call(s, "tu-1").state is CallState.REJECTED
    # The loop gives tu-2 its rejection result when it reaches it.
    tu2 = _call(s, "tu-2")
    assert tu2.state is CallState.REJECTED and tu2.error_code == CODE_APPROVAL_REJECTED
    assert tu2.result_summary == "The user rejected this call. It was not run. Reason: too big"
    assert s.pending_tool_use_ids == []


def test_rejected_precreated_request_never_runs_after_switching_to_autopilot():
    first = ("tu-1", "start_training", {"experiment_id": EXP})
    second = ("tu-2", "start_training", {"experiment_id": EXP})
    h = Harness([tool_turn(PLAN), tool_turn(first, second), text_turn("Done.")],
                mode=Mode.SUPERVISED)
    h.run()
    s = h.session
    ap1, ap2 = (_call(s, c).approval_id for c in ("tu-1", "tu-2"))
    hashes = {a.approval_id: a.args_hash for a in s.approvals}

    service, _, _ = _service(h)
    service._settings = SimpleNamespace(agent_enabled=True, agent_autopilot_allowed=True)
    service.decide_approval(ALICE, SID, ap2, "reject", args_hash=hashes[ap2])
    service.decide_approval(ALICE, SID, ap1, "reject", args_hash=hashes[ap1])
    service.set_mode(ALICE, SID, Mode.AUTOPILOT, max_estimated_spend_usd=4.0)
    assert h.session.mode is Mode.AUTOPILOT and h.session.autopilot is not None

    h.run()
    s = h.session
    assert h.calls == 0  # the rejected tu-2 did not run under Autopilot
    tu2 = _call(s, "tu-2")
    assert tu2.state is CallState.REJECTED and tu2.error_code == CODE_APPROVAL_REJECTED
    assert s.pending_tool_use_ids == []


def test_cancel_resolves_pending_uses_in_request_order():
    first = ("tu-1", "start_training", {"experiment_id": EXP})
    second = ("tu-2", "start_training", {"experiment_id": EXP})
    h = Harness([tool_turn(PLAN), tool_turn(first, second)], mode=Mode.SUPERVISED)
    h.run()
    service, orchestrator, _ = _service(h)
    service.cancel(ALICE, SID, stop_jobs=True)
    s = h.session
    results = [e for e in s.transcript if e.kind == "tool_result"]
    assert [e.tool_call_id for e in results] == ["tu-plan", "tu-1", "tu-2"]
    assert all(e.data["error_code"] == CODE_CANCELLED for e in results[1:])
    assert all(a.status is ApprovalStatus.REJECTED for a in s.approvals)
    assert orchestrator.stopped == []  # no job was awaited


# Feature: ml-assistant, Property 30: Cancellation is final
@settings(**SETTINGS)
@given(finished=st.lists(st.booleans(), min_size=1, max_size=6), stop=st.booleans())
def test_cancel_with_mixed_job_states(finished, stop):
    ids = [f"exec-{i}" for i in range(len(finished))]
    h = Harness([tool_turn(PLAN), tool_turn(TRAIN)], job_ids=ids)
    h.run()
    service, orchestrator, _ = _service(h)
    for job, done in zip(ids, finished, strict=True):
        if done:
            h.analysis.states[job] = ("SUCCEEDED", None)
    service.get_session(ALICE, SID)
    service.cancel(ALICE, SID, stop_jobs=stop)
    assert h.run().status is SessionStatus.CANCELLED
    assert h.calls == 1
    assert orchestrator.stopped == [job for job, done in zip(ids, finished, strict=True) if stop and not done]
