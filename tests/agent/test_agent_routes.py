"""ML Assistant REST routes (ml-assistant task 14.1, design C17; Req 12.9, 13.5, 15.8).

Routes are resolved on the real HTTP app (``mcp_server.app.build_app``) and their endpoints
called with a crafted request, the pattern ``tests/mcp/test_api_routes.py`` uses. The service,
loop and tools run for real over the in-memory store; the Converse client is scripted and the
S3 client is the signing-free fake from the upload-flow tests.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from starlette.requests import Request

from backend.agent.runner import InlineAgentRunner
from backend.agent.wiring import build_agent_service
from backend.auth import Principal
from backend.config import Settings
from backend.container import build_container
from mcp_server.app import build_app
from ml_engine.contracts.agent import SessionStatus, TranscriptEntry
from tests.agent.test_approvals import H_LIST, LIST, _awaiting
from tests.agent.test_uploads_flow import (
    GOOD_CSV,
    LIMIT,
    PLAN_ASK,
    T0,
    FakeS3,
    _file,
    _file_id,
    _upload_turn,
)
from tests.backend.conftest import IMAGE, ROLE
from tests.support.agent_fakes import CountingStore, FakeClock
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.fake_sagemaker import FakeLogs, FakeSageMaker
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.scripted_converse import ScriptedFakeConverseClient, ScriptStep, text_turn

OWNER = "alice"
BOB = Principal(user="bob")
SID = "/api/agent/sessions/{session_id}"

#: (method, path, body) for every session-scoped route, with bodies valid enough to reach the
#: session load (so ownership and existence are what decide the answer).
SESSION_ROUTES = [
    ("GET", SID, None),
    ("POST", SID + "/messages", {"text": "hello"}),
    ("POST", SID + "/resume", {}),
    ("POST", SID + "/approvals/{approval_id}", {"decision": "approve", "args_hash": H_LIST}),
    ("POST", SID + "/approvals/batch", {"items": [{"approval_id": "ap-1", "args_hash": H_LIST}]}),
    ("POST", SID + "/mode", {"mode": "supervised"}),
    ("POST", SID + "/limits/continue", {"limit": "max_tool_calls", "amount": 5}),
    ("POST", SID + "/cancel", {}),
    ("GET", SID + "/audit", None),
    ("POST", SID + "/local-files", {"files": []}),
    ("GET", SID + "/client-action", None),
    ("POST", SID + "/client-action/{action_id}/complete", {}),
]
ALL_ROUTES = [
    ("GET", "/api/agent/config", None),
    ("POST", "/api/agent/sessions", {"goal": "predict churn"}),
    ("GET", "/api/agent/sessions", None),
    *SESSION_ROUTES,
]


def _env(price_table_path, *turns, enabled=True):
    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn=ROLE,
        stage_image_uri=IMAGE,
        price_table_path=price_table_path,
        dev_identity=OWNER,
        agent_enabled=enabled,
        max_dataset_upload_bytes=LIMIT,
        bedrock_send_sample_values=False,
    )
    objects = InMemoryObjectStore()
    store = CountingStore(objects)
    converse = ScriptedFakeConverseClient(steps=[ScriptStep(t) for t in turns])
    container = build_container(
        settings,
        store=store,
        sagemaker_client=FakeSageMaker(),
        s3_client=FakeS3(objects),
        logs_client=FakeLogs(),
        converse_client=converse,
    )
    container.agent._runner.shutdown()  # replaced by an inline runner: no threads in tests
    runner = InlineAgentRunner()
    container.agent = build_agent_service(
        settings,
        store=store,
        services=container,
        converse_client=converse,
        audit_log=container.audit,
        authorizer=container.authorizer,
        runner=runner,
        clock=FakeClock(T0),
    )
    app = build_app(container)
    store.counts.clear()
    return SimpleNamespace(
        app=app, container=container, store=store, objects=objects, converse=converse,
        runner=runner, service=container.agent,
    )


def _request(method: str, body: dict | None, query: bytes) -> Request:
    raw = json.dumps(body).encode() if body is not None else b""
    scope = {"type": "http", "method": method, "headers": [], "query_string": query,
             "path_params": {}}
    request = Request(scope)

    async def receive() -> dict:
        return {"type": "http.request", "body": raw, "more_body": False}

    request._receive = receive
    return request


def _run(coroutine):
    """Drive a handler to completion without an event loop.

    The agent suite blocks sockets, and the Windows event loop opens a socket pair; these
    handlers never truly suspend (the body arrives in one message), so one ``send`` finishes.
    """
    try:
        coroutine.send(None)
    except StopIteration as done:
        return done.value
    coroutine.close()
    raise AssertionError("the handler suspended; it must complete without an event loop")


def _call(env, method, path, *, body=None, query=b"", **path_params):
    route = next(
        r
        for r in env.app.http_app().routes
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set())
    )
    request = _request(method, body, query)
    request.scope["path_params"] = {
        "session_id": "ses_missing", "approval_id": "ap-1", "action_id": "ca-1", **path_params
    }
    response = _run(route.endpoint(request))
    return response.status_code, json.loads(bytes(response.body))


def _session(env, sid):
    return env.service.store.load(sid).session


# --- the owner drives a whole session through the routes ------------------------------------


def test_owner_runs_an_upload_session_end_to_end(price_table_path):
    env = _env(price_table_path, PLAN_ASK, _upload_turn(_file_id()), text_turn("Uploaded."))

    status, view = _call(env, "POST", "/api/agent/sessions", body={"goal": "predict churn"})
    assert status == 200, view
    sid = view["session_id"]
    assert view["owner"] == OWNER and view["status"] == "waiting_for_user"

    status, view = _call(env, "POST", SID + "/local-files", body={"files": [_file()]},
                               session_id=sid)
    assert status == 200, view
    status, view = _call(env, "POST", SID + "/messages", body={"text": "Use my file."},
                               session_id=sid)
    assert status == 200 and view["status"] == "waiting_for_approval", view
    approval = view["pending_approvals"][0]

    status, view = _call(
        env, "POST", SID + "/approvals/{approval_id}",
        body={"decision": "approve", "args_hash": approval["args_hash"]},
        session_id=sid, approval_id=approval["approval_id"],
    )
    assert status == 200 and view["client_action"]["status"] == "ready", view
    assert _call(env, "GET", SID, session_id=sid) == (200, view)

    # The pre-signed PUT goes to the owner in this response only; the session view and the
    # stored record never carry it.
    status, action = _call(env, "GET", SID + "/client-action", session_id=sid)
    assert status == 200 and "X-Amz-Signature" in action["upload_url"]
    status, view = _call(env, "GET", SID, session_id=sid)
    assert status == 200
    assert "X-Amz-Signature" not in json.dumps(view) and "s3://" not in json.dumps(view)
    assert "X-Amz-Signature" not in _session(env, sid).model_dump_json()

    env.objects.write_bytes(_session(env, sid).client_action.dataset_uri, GOOD_CSV)
    status, view = _call(
        env, "POST", SID + "/client-action/{action_id}/complete",
        session_id=sid, action_id=action["action_id"],
    )
    assert status == 200 and view["status"] == "completed", view
    assert view["datasets"][0]["filename"] == "data.csv"
    assert "s3://" not in json.dumps(view)

    status, view = _call(env, "GET", SID, session_id=sid, query=b"after_seq=1000")
    assert status == 200 and view["transcript"] == []
    status, trail = _call(env, "GET", SID + "/audit", session_id=sid)
    assert status == 200 and trail["session_id"] == sid and trail["records"]
    status, page = _call(env, "GET", "/api/agent/sessions", query=b"limit=10")
    assert status == 200 and [i["session_id"] for i in page["items"]] == [sid]


def test_owner_mode_limits_batch_resume_and_cancel(price_table_path):
    env = _env(price_table_path, PLAN_ASK)
    status, view = _call(env, "POST", "/api/agent/sessions", body={"goal": "predict churn"})
    sid = view["session_id"]
    env.runner.bind(None)  # decisions below are recorded only; no further loop turn runs

    status, view = _call(
        env, "POST", SID + "/mode", body={"mode": "autopilot", "max_estimated_spend_usd": 1.0},
        session_id=sid,
    )
    assert status == 200 and view["mode"] == "autopilot", view
    assert view["autopilot_max_estimated_spend_usd"] == 1.0

    with env.service.store.transaction(sid) as s:
        s.status = SessionStatus.WAITING_FOR_USER
        s.transcript.append(TranscriptEntry(
            seq=s.next_seq, at=T0, kind="limit_stop", data={"limit": "max_tool_calls"}
        ))
        s.next_seq += 1
    before = _session(env, sid).limits.max_tool_calls
    status, view = _call(
        env, "POST", SID + "/limits/continue", body={"limit": "max_tool_calls", "amount": 5},
        session_id=sid,
    )
    assert status == 200 and view["limits"]["max_tool_calls"] == before + 5, view

    status, view = _call(env, "POST", SID + "/resume", session_id=sid)
    assert status == 200 and view["status"] == "running", view

    _awaiting(env.service, sid, LIST)
    status, view = _call(
        env, "POST", SID + "/approvals/batch",
        body={"items": [{"approval_id": "ap-tu-1", "args_hash": H_LIST}]}, session_id=sid,
    )
    assert status == 200 and view["pending_approvals"] == [], view

    status, body = _call(env, "POST", SID + "/cancel", body={"stop_jobs": "yes"},
                               session_id=sid)
    assert status == 400 and body["code"] == "VALIDATION_ERROR"
    status, view = _call(env, "POST", SID + "/cancel", body={"stop_jobs": False},
                               session_id=sid)
    assert status == 200 and view["status"] == "cancelled", view


def test_config_reports_poll_intervals_and_flags(price_table_path):
    env = _env(price_table_path)
    status, body = _call(env, "GET", "/api/agent/config")
    settings = env.container.settings
    assert status == 200
    assert body == {
        "enabled": True,
        "autopilot_allowed": settings.agent_autopilot_allowed,
        "cli_poll_seconds": settings.agent_cli_poll_seconds,
        "web_poll_seconds": settings.agent_web_poll_seconds,
    }


def test_bad_input_maps_to_400_not_500(price_table_path):
    env = _env(price_table_path)
    status, body = _call(env, "POST", "/api/agent/sessions", body={"goal": "   "})
    assert status == 400 and body["code"] == "VALIDATION_ERROR"
    assert env.store.writes == 0 and env.converse.calls == 0


# --- refusals ------------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "path", "body"), SESSION_ROUTES)
def test_non_owner_is_refused(price_table_path, method, path, body):
    env = _env(price_table_path, PLAN_ASK)
    sid = env.service.create_session(BOB, "bob's goal").session_id
    before = _session(env, sid)
    status, payload = _call(env, method, path, body=body, session_id=sid)
    assert status == 403 and payload["code"] == "NOT_AUTHORIZED", payload
    assert _session(env, sid) == before
    status, page = _call(env, "GET", "/api/agent/sessions")
    assert status == 200 and page["items"] == []  # bob's session is not listed for alice


@pytest.mark.parametrize(("method", "path", "body"), SESSION_ROUTES)
def test_unknown_session_is_404(price_table_path, method, path, body):
    env = _env(price_table_path)
    status, payload = _call(env, method, path, body=body, session_id="ses_missing")
    assert status == 404 and payload["code"] == "NOT_FOUND", payload


@pytest.mark.parametrize(("method", "path", "body"), ALL_ROUTES)
def test_disabled_agent_refuses_every_route_before_any_store_access(
    price_table_path, method, path, body
):
    env = _env(price_table_path, enabled=False)
    status, payload = _call(env, method, path, body=body)
    assert status == 403 and payload["code"] == "FEATURE_DISABLED", payload
    assert payload["details"]["setting"] == "ML_FACTORY_AGENT_ENABLED"
    assert sum(env.store.counts.values()) == 0  # no read, no write, no listing
    assert env.converse.calls == 0


def test_agent_routes_are_added_after_the_existing_ones():
    from backend.api.app import _collect_routes
    from backend.api.routes import agent

    routes = _collect_routes()
    added = [(m, p) for m, p, _ in agent.routes()]
    assert [(m, p) for m, p, _ in routes[-len(added):]] == added
    existing = [(m, p) for m, p, _ in routes[: -len(added)]]
    assert not any(p.startswith("/agent") for _, p in existing)
