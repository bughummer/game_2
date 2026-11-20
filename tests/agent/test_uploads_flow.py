"""Offline tests for the local upload flow (task 12.2, design C16; Req 7.4 to 7.12).

The service, loop and tool run for real against the in-memory object store and the real
``DatasetService``. The only stand-in is the S3 client: it signs nothing and reads from the
in-memory store, and the test writes the bytes the user's client would PUT.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import BytesIO
from types import SimpleNamespace

import pytest

from backend.agent.runner import InlineAgentRunner
from backend.agent.service import LocalFileInput
from backend.agent.tools_core import dataset_id_for
from backend.agent.wiring import build_agent_service
from backend.config import Settings
from backend.errors import AuthorizationError, ValidationError
from backend.services.datasets import DatasetService
from ml_engine.contracts.agent import CallState, DatasetRef, SessionStatus
from ml_engine.io import ExperimentLayout
from tests.support.agent_fakes import FakeClock
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
ALICE = SimpleNamespace(user="alice")
BOB = SimpleNamespace(user="bob")
LIMIT = 10_000
SHA = "ab" * 32
SECRET = "zz-secret-cell"
GOOD_CSV = f"id,label\n1,{SECRET}\n2,{SECRET}\n3,{SECRET}\n".encode()
UPLOADS = ExperimentLayout.uploads_prefix(ARTIFACT_ROOT)

PLAN_ASK = tool_turn(
    ("tu-plan", "submit_plan", {"steps": ["upload the data"]}),
    ("tu-ask", "ask_user", {"question": "Which file?"}),
)


class FakeS3:
    """Signs nothing; ``get_object`` reads the in-memory store (with ``Range``)."""

    def __init__(self, objects: InMemoryObjectStore) -> None:
        self._objects = objects

    def generate_presigned_url(self, operation, Params, ExpiresIn):  # noqa: N803
        return f"https://s3.example/{Params['Key']}?X-Amz-Signature=sig"

    def get_object(self, Bucket, Key, Range=None):  # noqa: N803
        data = self._objects.read_bytes(f"s3://{Bucket}/{Key}")
        if Range:
            start, end = Range.removeprefix("bytes=").split("-")
            data = data[int(start) : int(end) + 1]
        return {"Body": BytesIO(data)}


def _file(name="data.csv", size=len(GOOD_CSV), sha=SHA) -> dict:
    return {"name": name, "size_bytes": size, "modified_at": T0.isoformat(), "sha256": sha}


def _file_id(**kwargs) -> str:
    return LocalFileInput.model_validate(_file(**kwargs)).to_ref().file_id


def _upload_turn(file_id: str | None):
    args = {} if file_id is None else {"file_id": file_id}
    return tool_turn(("tu-up", "request_local_upload", args))


def _env(*turns, mode="supervised", files=(_file(),), datasets=()):
    """A session that asked the user, got its local files, and made its upload call."""
    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn=ROLE,
        agent_enabled=True,
        max_dataset_upload_bytes=LIMIT,
        bedrock_send_sample_values=False,
    )
    objects = InMemoryObjectStore()
    services = SimpleNamespace(
        settings=settings,
        store=objects,
        datasets=DatasetService(settings=settings, store=objects, s3_client=FakeS3(objects)),
    )
    converse = ScriptedFakeConverseClient(steps=[ScriptStep(t) for t in (PLAN_ASK, *turns)])
    service = build_agent_service(
        settings,
        store=objects,
        services=services,
        converse_client=converse,
        runner=InlineAgentRunner(),
        clock=FakeClock(T0),
    )
    spend = 1.0 if mode == "autopilot" else None
    sid = service.create_session(
        ALICE, "predict churn", mode=mode, autopilot_max_spend_usd=spend
    ).session_id
    if datasets:
        with service.store.transaction(sid) as s:
            s.datasets = list(datasets)
    service.register_local_files(ALICE, sid, list(files))
    service.send_message(ALICE, sid, "Use the file I registered.")
    return SimpleNamespace(service=service, sid=sid, objects=objects, converse=converse)


def _session(env):
    return env.service.store.load(env.sid).session


def _upload_objects(env) -> list[str]:
    return env.objects.list_uris(UPLOADS)


def _upload_record(env):
    return next(r for r in _session(env).tool_calls if r.tool == "request_local_upload")


def _to_client_action(env):
    """Approve the first upload, so the loop hands it to the client."""
    approval = _session(env).approvals[0]
    env.service.decide_approval(ALICE, env.sid, approval.approval_id, "approve",
                                args_hash=approval.args_hash)
    return _session(env).client_action


@pytest.mark.parametrize(
    "file",
    [_file(size=LIMIT + 1), _file(name="data.tsv"), _file(name="data.xlsx")],
    ids=["oversized", "tsv", "xlsx"],
)
def test_oversized_or_wrong_extension_is_refused_before_any_transfer(file):
    file_id = LocalFileInput.model_validate(file).to_ref().file_id
    env = _env(_upload_turn(file_id), text_turn("Cannot upload that file."), files=[file])
    s = _session(env)
    record = _upload_record(env)
    assert record.state is CallState.REFUSED and record.error_code == "VALIDATION_ERROR"
    assert s.approvals == [] and s.client_action is None and s.datasets == []
    assert _upload_objects(env) == []
    assert s.status is SessionStatus.COMPLETED


def test_duplicate_sha_reuses_the_dataset_without_approval():
    uri = f"{UPLOADS}old-1234/data.csv"
    existing = DatasetRef(
        dataset_id=dataset_id_for(uri), uri=uri, filename="data.csv", sha256=SHA,
        size_bytes=10, file_format="csv", columns=[], row_count=3,
    )
    env = _env(_upload_turn(_file_id()), text_turn("Reused."), datasets=[existing])
    s = _session(env)
    assert _upload_record(env).state is CallState.SUCCEEDED
    assert s.approvals == [] and s.client_action is None
    assert [d.dataset_id for d in s.datasets] == [existing.dataset_id]
    result = next(e for e in s.transcript if e.kind == "tool_result" and e.tool_call_id == "tu-up")
    assert existing.dataset_id in json.dumps(result.data["model_view"])
    assert _upload_objects(env) == []


@pytest.mark.parametrize("mode", ["supervised", "autopilot"])
def test_first_upload_needs_approval_in_both_modes(mode):
    env = _env(_upload_turn(_file_id()), mode=mode)
    s = _session(env)
    assert s.status is SessionStatus.WAITING_FOR_APPROVAL
    (approval,) = s.approvals
    assert approval.kind == "upload" and s.client_action is None
    assert approval.review == {
        "file_id": _file_id(),
        "file_name": "data.csv",
        "size_bytes": len(GOOD_CSV),
        "destination_prefix": UPLOADS,
    }
    action = _to_client_action(env)
    s = _session(env)
    assert action is not None and action.status == "ready"
    assert s.status is SessionStatus.WAITING_FOR_USER
    assert _upload_record(env).state is CallState.EXECUTING  # pending until the client uploads
    assert _upload_objects(env) == []


def test_several_candidates_ask_the_user_and_create_no_approval():
    files = [_file(name="a.csv"), _file(name="b.csv", sha="cd" * 32)]
    env = _env(_upload_turn(_file_id(name="a.csv")), files=files)
    s = _session(env)
    assert _upload_record(env).error_code == "USER_CHOICE_REQUIRED"
    assert s.approvals == [] and s.status is SessionStatus.WAITING_FOR_USER


def test_failed_completion_leaves_no_object_and_the_same_file_can_be_retried():
    env = _env(_upload_turn(_file_id()), text_turn("Uploaded."))
    action = _to_client_action(env)

    ticket = env.service.client_action(ALICE, env.sid)
    assert "X-Amz-Signature" not in _session(env).model_dump_json()  # nothing signed is saved
    env.objects.write_bytes(_session(env).client_action.dataset_uri, b"x" * (LIMIT + 1))
    with pytest.raises(ValidationError, match="bytes"):
        env.service.complete_upload(ALICE, env.sid, ticket.action_id)
    s = _session(env)
    assert _upload_objects(env) == [] and s.datasets == []
    assert s.client_action.status == "ready" and s.client_action.action_id == action.action_id

    env.service.client_action(ALICE, env.sid)
    env.objects.write_bytes(_session(env).client_action.dataset_uri, GOOD_CSV)
    view = env.service.complete_upload(ALICE, env.sid, action.action_id)
    assert view.status is SessionStatus.COMPLETED
    assert len(_session(env).datasets) == 1


def test_success_registers_the_dataset_and_the_model_sees_no_raw_rows():
    env = _env(_upload_turn(_file_id()), text_turn("Uploaded."))
    action = _to_client_action(env)
    env.service.client_action(ALICE, env.sid)
    uri = _session(env).client_action.dataset_uri
    env.objects.write_bytes(uri, GOOD_CSV)
    env.service.complete_upload(ALICE, env.sid, action.action_id)

    s = _session(env)
    (ref,) = s.datasets
    assert ref.dataset_id == dataset_id_for(uri) and ref.row_count == 3 and ref.sha256 == SHA
    assert [c.name for c in ref.columns] == ["id", "label"]
    assert s.upload_approved_once and s.client_action.status == "done"
    assert _upload_record(env).state is CallState.SUCCEEDED
    result = next(e for e in s.transcript if e.kind == "tool_result" and e.tool_call_id == "tu-up")
    view = json.dumps(result.data["model_view"])
    assert ref.dataset_id in view and '"row_count": 3' in view and "label" in view
    sent = repr(env.converse.requests[-1].messages)
    assert SECRET not in view and SECRET not in sent and uri not in sent
    assert s.status is SessionStatus.COMPLETED


def test_non_owner_is_refused_on_every_upload_action():
    env = _env(_upload_turn(_file_id()))
    action = _to_client_action(env)
    before = _session(env)
    with pytest.raises(AuthorizationError):
        env.service.register_local_files(BOB, env.sid, [_file()])
    with pytest.raises(AuthorizationError):
        env.service.client_action(BOB, env.sid)
    with pytest.raises(AuthorizationError):
        env.service.complete_upload(BOB, env.sid, action.action_id)
    assert _session(env) == before
