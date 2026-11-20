"""Offline tests for approval handling in ``AgentService`` (task 8.4, design C15; Req 4.6 to 4.13).

The service only records decisions. Executing an approved call is the loop's job (task 8.6), so
these tests never bind a loop to the runner.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from backend.errors import AuthorizationError, ConflictError, ValidationError
from ml_engine.contracts.agent import (
    ApprovalRequest,
    ApprovalStatus,
    CallState,
    SessionStatus,
    ToolCallRecord,
    ToolClass,
    args_hash,
)
from tests.agent.test_service import ALICE, BOB, T0, _service

LIST = ("tu-1", "list_datasets", {"limit": 10})
H_LIST = args_hash("list_datasets", {"limit": 10})


def _awaiting(service, sid, *calls):
    """Put ``(call_id, tool, arguments)`` uses into ``awaiting_approval`` with pending requests."""
    with service.store.transaction(sid) as s:
        s.status = SessionStatus.WAITING_FOR_APPROVAL
        for call_id, tool, arguments in calls:
            h = args_hash(tool, arguments)
            s.tool_calls.append(
                ToolCallRecord(
                    tool_call_id=call_id, turn=1, tool=tool, arguments=arguments, args_hash=h,
                    tool_class=ToolClass.COSTLY, state=CallState.AWAITING_APPROVAL,
                    approval_id=f"ap-{call_id}",
                )
            )
            s.pending_tool_use_ids.append(call_id)
            s.approvals.append(
                ApprovalRequest(
                    approval_id=f"ap-{call_id}", kind="tool", tool_call_id=call_id, tool=tool,
                    arguments=arguments, args_hash=h, rationale="", estimate_usd=None,
                    estimate_basis=None, status=ApprovalStatus.PENDING, created_at=T0,
                    expires_at=T0 + timedelta(seconds=900),
                )
            )


def _approval_session(*calls):
    service, _, _, runner = _service()
    runner.bind(None)
    sid = service.create_session(ALICE, "predict churn").session_id
    _awaiting(service, sid, *calls)
    return service, runner, sid


def test_approve_records_decision_and_never_runs_the_handler():
    service, runner, sid = _approval_session(LIST)
    before = service.store.load(sid).session

    view = service.decide_approval(ALICE, sid, "ap-tu-1", "approve", args_hash=H_LIST)
    s = service.store.load(sid).session
    approval = s.approvals[0]
    assert approval.status is ApprovalStatus.APPROVED and approval.approved_hash == H_LIST
    assert approval.decided_by == "alice" and approval.decided_at == T0
    assert s.tool_calls[0].state is CallState.AWAITING_APPROVAL  # the loop executes, not us
    assert s.tool_calls[0].args_hash == approval.approved_hash
    assert view.status is SessionStatus.RUNNING and runner.kicked[-1] == sid
    assert s.audit_seq == before.audit_seq + 1
    with pytest.raises(ConflictError, match="cannot be decided again"):
        service.decide_approval(ALICE, sid, "ap-tu-1", "approve", args_hash=H_LIST)


@pytest.mark.parametrize("decision", ["approve", "reject", "edit"])
def test_non_owner_is_refused_and_nothing_changes(decision):
    service, runner, sid = _approval_session(LIST)
    before = service.store.load(sid).session
    kicks = len(runner.kicked)
    with pytest.raises(AuthorizationError):
        service.decide_approval(
            BOB, sid, "ap-tu-1", decision, args_hash=H_LIST, edited_args={"limit": 20}
        )
    assert service.store.load(sid).session == before
    assert len(runner.kicked) == kicks


def test_stale_hash_is_refused_with_approval_mismatch():
    service, _, sid = _approval_session(LIST)
    before = service.store.load(sid).session
    with pytest.raises(ConflictError) as stale:
        service.decide_approval(ALICE, sid, "ap-tu-1", "approve", args_hash="stale")
    assert stale.value.details["reason"] == "approval_mismatch"
    assert service.store.load(sid).session == before


def test_reject_caps_reason_and_resolves_the_call():
    service, runner, sid = _approval_session(LIST)
    view = service.decide_approval(
        ALICE, sid, "ap-tu-1", "reject", args_hash=H_LIST, reason="x" * 600
    )
    s = service.store.load(sid).session
    assert s.approvals[0].status is ApprovalStatus.REJECTED
    assert s.approvals[0].reason == "x" * 500
    assert s.approvals[0].decided_by == "alice" and s.approvals[0].decided_at == T0
    assert s.tool_calls[0].state is CallState.REJECTED and s.pending_tool_use_ids == []
    results = [e for e in s.transcript if e.kind == "tool_result"]
    assert [e.tool_call_id for e in results] == ["tu-1"]
    assert "x" * 500 in results[0].text and "x" * 501 not in results[0].text
    assert view.status is SessionStatus.RUNNING and runner.kicked[-1] == sid


def test_invalid_edit_names_arguments_and_leaves_request_unchanged():
    exp = ("tu-2", "get_experiment", {"experiment_id": "exp-1"})
    service, runner, sid = _approval_session(LIST, exp)
    before = service.store.load(sid).session.approvals
    kicks = len(runner.kicked)
    with pytest.raises(ValidationError) as bad_model:
        service.decide_approval(
            ALICE, sid, "ap-tu-1", "edit", args_hash=H_LIST, edited_args={"limit": 0}
        )
    assert bad_model.value.details["failed_arguments"] == ["limit"]
    assert bad_model.value.details["reason"] == "invalid_edit"
    with pytest.raises(ValidationError) as bad_input:
        service.decide_approval(
            ALICE, sid, "ap-tu-2", "edit", args_hash=args_hash(*exp[1:]),
            edited_args={"experiment_id": "s3://bucket/key"},
        )
    assert bad_input.value.details["failed_arguments"] == ["experiment_id"]
    s = service.store.load(sid).session
    assert s.approvals == before and s.status is SessionStatus.WAITING_FOR_APPROVAL
    assert len(runner.kicked) == kicks


def test_valid_edit_needs_approval_of_the_edited_hash():
    service, runner, sid = _approval_session(LIST)
    edited = {"limit": 20}
    h_edit = args_hash("list_datasets", edited)
    kicks = len(runner.kicked)
    view = service.decide_approval(
        ALICE, sid, "ap-tu-1", "edit", args_hash=H_LIST, edited_args=edited
    )
    assert view.status is SessionStatus.WAITING_FOR_APPROVAL and len(runner.kicked) == kicks
    s = service.store.load(sid).session
    approval = s.approvals[0]
    assert approval.status is ApprovalStatus.EDIT_PROPOSED and approval.edited_hash == h_edit
    assert approval.arguments == {"limit": 10} and approval.decided_by is None
    assert s.tool_calls[0].arguments == {"limit": 10}  # nothing runs before the confirmation
    with pytest.raises(ConflictError) as stale:
        service.decide_approval(ALICE, sid, "ap-tu-1", "approve", args_hash=H_LIST)
    assert stale.value.details["reason"] == "approval_mismatch"

    service.decide_approval(ALICE, sid, "ap-tu-1", "approve", args_hash=h_edit)
    s = service.store.load(sid).session
    assert s.approvals[0].status is ApprovalStatus.APPROVED
    assert s.approvals[0].approved_hash == h_edit
    assert s.tool_calls[0].arguments == edited and s.tool_calls[0].args_hash == h_edit


def test_expired_approval_is_marked_and_refused():
    service, runner, sid = _approval_session(LIST)
    service._clock.advance(901)
    kicks = len(runner.kicked)
    with pytest.raises(ConflictError) as expired:
        service.decide_approval(ALICE, sid, "ap-tu-1", "approve", args_hash=H_LIST)
    assert expired.value.details["reason"] == "approval_expired"
    s = service.store.load(sid).session
    assert s.approvals[0].status is ApprovalStatus.EXPIRED
    assert s.status is SessionStatus.WAITING_FOR_USER
    assert s.tool_calls[0].state is CallState.REFUSED and s.pending_tool_use_ids == []
    assert len(runner.kicked) == kicks
    with pytest.raises(ConflictError) as again:  # a later decision is still refused
        service.decide_approval(ALICE, sid, "ap-tu-1", "reject", args_hash=H_LIST)
    assert again.value.details["reason"] == "approval_expired"


def test_get_session_sweeps_expired_approvals():
    service, _, sid = _approval_session(LIST)
    service._clock.advance(900)  # exactly at the TTL is still valid
    assert service.get_session(ALICE, sid).status is SessionStatus.WAITING_FOR_APPROVAL
    service._clock.advance(1)
    with pytest.raises(AuthorizationError):
        service.get_session(BOB, sid)
    assert service.store.load(sid).session.approvals[0].status is ApprovalStatus.PENDING
    view = service.get_session(ALICE, sid)
    assert view.status is SessionStatus.WAITING_FOR_USER and view.pending_approvals == []


def test_batch_approves_only_matching_items_and_takes_at_most_10():
    other = ("tu-2", "list_datasets", {"limit": 5})
    service, runner, sid = _approval_session(LIST, other)
    item = {"approval_id": "ap-tu-1", "args_hash": H_LIST}
    with pytest.raises(ValidationError, match="1 to 10"):
        service.decide_batch(ALICE, sid, [item] * 11)
    with pytest.raises(ValidationError, match="1 to 10"):
        service.decide_batch(ALICE, sid, [])
    with pytest.raises(AuthorizationError):
        service.decide_batch(BOB, sid, [item])
    view = service.decide_batch(
        ALICE, sid, [item, {"approval_id": "ap-tu-2", "args_hash": "not-what-was-shown"}]
    )
    s = service.store.load(sid).session
    assert [a.status for a in s.approvals] == [ApprovalStatus.APPROVED, ApprovalStatus.PENDING]
    assert s.approvals[0].batch_id is not None
    assert [a.approval_id for a in view.pending_approvals] == ["ap-tu-2"]
    assert runner.kicked[-1] == sid


def test_batch_of_ten_is_accepted():
    calls = [(f"tu-{i}", "list_datasets", {"limit": i + 1}) for i in range(10)]
    service, _, sid = _approval_session(*calls)
    items = [
        {"approval_id": f"ap-{cid}", "args_hash": args_hash(tool, a)} for cid, tool, a in calls
    ]
    service.decide_batch(ALICE, sid, items)
    s = service.store.load(sid).session
    assert all(a.status is ApprovalStatus.APPROVED for a in s.approvals)
    assert len({a.batch_id for a in s.approvals}) == 1


def test_edit_recomputes_the_review_or_is_refused_when_the_view_fails():
    from types import SimpleNamespace

    from pydantic import BaseModel, ConfigDict

    from backend.agent.tools import ToolOutput

    class RunArgs(BaseModel):
        model_config = ConfigDict(extra="forbid")
        revision: int

    def view(ctx, args):
        code = ctx.services.scripts[args.revision]  # KeyError for an unknown revision
        return {"revision": args.revision, "content": code}

    service, _, sid = _approval_session(("tu-1", "run_script", {"revision": 1}))
    service._services = SimpleNamespace(scripts={1: "print(1)", 2: "print(2)"})
    service._registry.register(
        "run_script",
        classes=[ToolClass.COSTLY],
        description="run a stored script revision",
        input_model=RunArgs,
        handler=lambda ctx, a: ToolOutput(data={}),
        approval_view=view,
    )
    h1 = args_hash("run_script", {"revision": 1})
    with service.store.transaction(sid) as s:
        s.approvals[0].review = {"revision": 1, "content": "print(1)"}

    service.decide_approval(
        ALICE, sid, "ap-tu-1", "edit", args_hash=h1, edited_args={"revision": 2}
    )
    approval = service.store.load(sid).session.approvals[0]
    assert approval.status is ApprovalStatus.EDIT_PROPOSED
    assert approval.review == {"revision": 2, "content": "print(2)"}

    before = service.store.load(sid).session.approvals
    with pytest.raises(ValidationError) as failed:
        service.decide_approval(
            ALICE, sid, "ap-tu-1", "edit", args_hash=approval.edited_hash,
            edited_args={"revision": 3},
        )
    assert failed.value.details["reason"] == "invalid_edit"
    assert service.store.load(sid).session.approvals == before
