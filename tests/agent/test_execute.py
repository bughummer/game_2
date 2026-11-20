"""Unit tests for the execution shell and input checks (task 4.8)."""

from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from backend.agent.execute import ToolExecutor
from backend.agent.inputs import ToolInputs
from backend.agent.tools import ToolFlags, ToolOutput, ToolRegistry
from backend.audit import AuditWriteError
from backend.errors import AppError, AuthorizationError, NotFoundError
from backend.orchestration.guards import GuardViolation
from ml_engine.contracts.agent import DatasetRef, Session, ToolClass


class Args(BaseModel):
    experiment_id: str | None = None
    dataset_id: str | None = None
    instance_type: str | None = None
    note: str | None = None


class FakeAgentAudit:
    def __init__(self) -> None:
        self.records = []

    def write(self, session, rec):
        self.records.append(rec)
        session.audit_seq += 1
        return f"agent-audit-{session.audit_seq}"


class FakeAuthorizer:
    def __init__(self, owned: set[str]) -> None:
        self.owned = owned
        self.principals = []

    def require_owner(self, principal, experiment_id):
        self.principals.append(principal.user)
        if experiment_id not in self.owned:
            raise AuthorizationError("not yours")


class FakeAuditLog:
    def __init__(self, fail: bool = False) -> None:
        self.events = []
        self.fail = fail

    def write(self, **kwargs):
        if self.fail:
            raise AuditWriteError("boom")
        self.events.append(kwargs)
        return "s3://bucket/audit/1.json"


def make_session() -> Session:
    return Session.model_construct(
        session_id="s1",
        owner="alice",
        audit_seq=0,
        experiment_ids=["e1", "e2"],
        datasets=[
            DatasetRef.model_construct(dataset_id="d1", uri="s3://b/d1.csv"),
        ],
    )


def build(handler, flags=None, *, owned=None, audit_fail=False):
    registry = ToolRegistry()
    registry.register(
        "t",
        classes=[ToolClass.MUTATING],
        description="t",
        input_model=Args,
        handler=handler,
        flags=flags
        or ToolFlags(
            experiment_arg="experiment_id",
            dataset_args=("dataset_id",),
            instance_args=("instance_type",),
            audit_action="thing_done",
        ),
    )
    audit = FakeAgentAudit()
    log = FakeAuditLog(fail=audit_fail)
    services = SimpleNamespace(
        audit=log,
        experiments=SimpleNamespace(get=lambda eid: SimpleNamespace(artifact_prefix=f"p/{eid}")),
    )
    inputs = ToolInputs(
        audit=audit,
        authorizer=FakeAuthorizer({"e1"} if owned is None else owned),
        allowed_instance_types=["ml.m5.large"],
    )
    return ToolExecutor(inputs=inputs, services=services), registry.get("t"), audit, log


def ok_handler(ctx, args):
    return ToolOutput(data={"user": ctx.principal.user, "uri": args.dataset_id}, audit_detail={"k": 1})


@pytest.mark.parametrize(
    "arguments,reason",
    [
        ({"note": "C:\\data\\x.csv"}, "local_path"),
        ({"note": "/etc/passwd"}, "local_path"),
        ({"note": "s3://b/k"}, "uri_not_allowed"),
        ({"note": "https://h/x?X-Amz-Signature=abc"}, "signed_url"),
        ({"instance_type": "ml.p4d.24xlarge"}, "instance_not_allowed"),
        ({"dataset_id": "nope"}, "unknown_dataset"),
        ({"experiment_id": "zzz"}, "experiment_not_linked"),
        ({"experiment_id": "e2"}, "experiment_not_owned"),
    ],
)
def test_rejections_are_audited_without_the_value(arguments, reason):
    ex, spec, audit, log = build(ok_handler)
    result = ex.execute(make_session(), spec, arguments)
    assert result.error and result.error.code == "INPUT_REJECTED"
    assert [r.reason for r in result.rejections] == [reason]
    assert len(audit.records) == 1
    rec = audit.records[0]
    assert rec.event_type == "input_rejected"
    assert set(rec.detail) == {"argument", "reason"}
    assert not log.events


def test_success_uses_owner_identity_resolves_dataset_and_records_ref():
    ex, spec, audit, log = build(ok_handler)
    result = ex.execute(
        make_session(), spec, {"experiment_id": "e1", "dataset_id": "d1", "instance_type": "ml.m5.large"}
    )
    assert result.ok
    assert result.output.data == {"user": "alice", "uri": "s3://b/d1.csv"}
    assert result.service_audit_ref == "s3://bucket/audit/1.json"
    event = log.events[0]
    assert event["action"] == "thing_done"
    assert event["actor"] == "alice"
    assert event["artifact_prefix"] == "p/e1"
    assert event["detail"] == {"k": "1"}


def test_audit_write_failure_keeps_the_result_and_flags_it():
    ex, spec, *_ = build(ok_handler, audit_fail=True)
    result = ex.execute(make_session(), spec, {"experiment_id": "e1"})
    assert result.ok and result.audit_failed and result.service_audit_ref is None


@pytest.mark.parametrize(
    "exc,code,kind",
    [
        (NotFoundError("missing"), "NOT_FOUND", None),
        (GuardViolation("OVER_BUDGET", "too much"), "OVER_BUDGET", "guard"),
    ],
)
def test_known_errors_are_structured(exc, code, kind):
    def handler(ctx, args):
        raise exc

    ex, spec, *_ = build(handler)
    result = ex.execute(make_session(), spec, {})
    assert result.error.code == code
    assert result.error_kind == kind
    assert isinstance(exc, (AppError, GuardViolation))


def test_unexpected_exception_is_generic_and_names_only_the_tool():
    def handler(ctx, args):
        raise RuntimeError("secret detail s3://b/k")

    ex, spec, *_ = build(handler)
    result = ex.execute(make_session(), spec, {})
    assert result.error.code == "TOOL_FAILED"
    assert "secret" not in result.error.message and "s3" not in str(result.error.as_dict())
    assert result.error.details == {"tool": "t"}
