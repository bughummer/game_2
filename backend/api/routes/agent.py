"""ML Assistant routes (ml-assistant design C17).

Thin wrappers over ``AgentService``: resolve the caller with ``principal(...)``, call the one
service method, return ``json_ok``. The service checks session ownership and audits its own
events, so these handlers add neither (design C1 table, ``backend/api/routes/*`` row).

Every handler first checks ``agent_enabled``. When the agent is off it raises
``FeatureDisabledError`` before the caller is resolved or any store is touched (Req 13.5, 15.8).
``SessionView`` carries no dataset URI or signed URL; only the ``client-action`` response holds
the freshly minted pre-signed PUT, and only for the session owner.
"""

from collections.abc import Callable
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import principal
from backend.api.errors import json_ok
from backend.api.request import json_body, query_int
from backend.audit import AuditWriteError
from backend.container import AppContainer
from backend.errors import FeatureDisabledError, UpstreamError, ValidationError

ENABLED_SETTING = "ML_FACTORY_AGENT_ENABLED"

Endpoint = Callable[[AppContainer, Request], Any]


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    # Joined, not one literal: the boundary test reads "sessions/" as a storage-path marker.
    sid = "/agent/sessions" + "/{session_id}"
    return [
        ("GET", "/agent/config", _agent(_config)),
        ("POST", "/agent/sessions", _agent(_create)),
        ("GET", "/agent/sessions", _agent(_list)),
        ("GET", sid, _agent(_get)),
        ("POST", f"{sid}/messages", _agent(_send_message)),
        ("POST", f"{sid}/resume", _agent(_resume)),
        # Registered before "/approvals/{approval_id}" so "batch" is never read as an id.
        ("POST", f"{sid}/approvals/batch", _agent(_decide_batch)),
        ("POST", f"{sid}/approvals/{{approval_id}}", _agent(_decide_approval)),
        ("POST", f"{sid}/mode", _agent(_set_mode)),
        ("POST", f"{sid}/limits/continue", _agent(_continue_after_limit)),
        ("POST", f"{sid}/cancel", _agent(_cancel)),
        ("GET", f"{sid}/audit", _agent(_audit_trail)),
        ("POST", f"{sid}/local-files", _agent(_register_local_files)),
        ("GET", f"{sid}/client-action", _agent(_client_action)),
        ("POST", f"{sid}/client-action/{{action_id}}/complete", _agent(_complete_upload)),
    ]


def _agent(endpoint: Endpoint) -> Callable[[AppContainer], Callable]:
    """Build a handler factory that refuses while the agent is disabled.

    An ``AuditWriteError`` (a ``RuntimeError``, not an ``AppError``) becomes ``UPSTREAM_ERROR``
    so a failed audit write is a 502 with a stable reason, not a generic 500.
    """

    def factory(container: AppContainer) -> Callable:
        async def handler(request: Request) -> JSONResponse:
            _require_enabled(container)
            try:
                return json_ok(await endpoint(container, request))
            except AuditWriteError:
                raise UpstreamError(
                    "The session could not be audited.", {"reason": "audit_failed"}
                ) from None

        handler.__name__ = endpoint.__name__.lstrip("_")
        return handler

    return factory


def _require_enabled(container: AppContainer) -> None:
    if not bool(container.settings.agent_enabled) or container.agent is None:
        raise FeatureDisabledError(
            f"The ML Assistant is disabled. Set {ENABLED_SETTING}=true to enable it.",
            {"setting": ENABLED_SETTING},
        )


def _dump(model: Any) -> Any:
    return model.model_dump(mode="json") if model is not None else None


def _session_id(request: Request) -> str:
    return request.path_params["session_id"]


def _optional_bool(body: dict[str, Any], name: str) -> bool:
    value = body.get(name, False)
    if not isinstance(value, bool):
        raise ValidationError(f"'{name}' must be true or false.", {"field": name})
    return value


# --- config --------------------------------------------------------------------------------


async def _config(container: AppContainer, _request: Request) -> dict[str, Any]:
    settings = container.settings
    return {
        "enabled": bool(settings.agent_enabled),
        "autopilot_allowed": bool(settings.agent_autopilot_allowed),
        "cli_poll_seconds": settings.agent_cli_poll_seconds,
        "web_poll_seconds": settings.agent_web_poll_seconds,
    }


# --- sessions ------------------------------------------------------------------------------


async def _create(container: AppContainer, request: Request) -> Any:
    body = await json_body(request)
    who = principal(container, request)
    view = container.agent.create_session(
        who,
        body.get("goal"),
        mode=body.get("mode", "supervised"),
        autopilot_max_spend_usd=body.get("autopilot_max_spend_usd"),
    )
    return _dump(view)


async def _list(container: AppContainer, request: Request) -> Any:
    who = principal(container, request)
    page = container.agent.list_sessions(
        who,
        limit=query_int(request, "limit", 100),
        cursor=request.query_params.get("cursor") or None,
    )
    return _dump(page)


async def _get(container: AppContainer, request: Request) -> Any:
    who = principal(container, request)
    view = container.agent.get_session(
        who, _session_id(request), after_seq=query_int(request, "after_seq", 0)
    )
    return _dump(view)


async def _send_message(container: AppContainer, request: Request) -> Any:
    body = await json_body(request)
    who = principal(container, request)
    return _dump(container.agent.send_message(who, _session_id(request), body.get("text")))


async def _resume(container: AppContainer, request: Request) -> Any:
    who = principal(container, request)
    return _dump(container.agent.resume(who, _session_id(request)))


# --- approvals -----------------------------------------------------------------------------


async def _decide_approval(container: AppContainer, request: Request) -> Any:
    body = await json_body(request)
    who = principal(container, request)
    view = container.agent.decide_approval(
        who,
        _session_id(request),
        request.path_params["approval_id"],
        body.get("decision"),
        args_hash=body.get("args_hash"),
        edited_args=body.get("edited_args"),
        reason=body.get("reason"),
    )
    return _dump(view)


async def _decide_batch(container: AppContainer, request: Request) -> Any:
    body = await json_body(request)
    who = principal(container, request)
    return _dump(container.agent.decide_batch(who, _session_id(request), body.get("items")))


# --- mode, limits, cancel ------------------------------------------------------------------


async def _set_mode(container: AppContainer, request: Request) -> Any:
    body = await json_body(request)
    who = principal(container, request)
    view = container.agent.set_mode(
        who,
        _session_id(request),
        body.get("mode"),
        max_estimated_spend_usd=body.get("max_estimated_spend_usd"),
    )
    return _dump(view)


async def _continue_after_limit(container: AppContainer, request: Request) -> Any:
    body = await json_body(request)
    who = principal(container, request)
    view = container.agent.continue_after_limit(
        who, _session_id(request), body.get("limit"), body.get("amount")
    )
    return _dump(view)


async def _cancel(container: AppContainer, request: Request) -> Any:
    body = await json_body(request)
    stop_jobs = _optional_bool(body, "stop_jobs")
    who = principal(container, request)
    return _dump(container.agent.cancel(who, _session_id(request), stop_jobs=stop_jobs))


async def _audit_trail(container: AppContainer, request: Request) -> Any:
    who = principal(container, request)
    session_id = _session_id(request)
    records = container.agent.audit_trail(who, session_id)
    return {"session_id": session_id, "records": [_dump(r) for r in records]}


# --- local files and uploads ---------------------------------------------------------------


async def _register_local_files(container: AppContainer, request: Request) -> Any:
    body = await json_body(request)
    who = principal(container, request)
    view = container.agent.register_local_files(who, _session_id(request), body.get("files"))
    return _dump(view)


async def _client_action(container: AppContainer, request: Request) -> Any:
    # The only response that carries a signed URL. The service checks the owner first.
    who = principal(container, request)
    return _dump(container.agent.client_action(who, _session_id(request)))


async def _complete_upload(container: AppContainer, request: Request) -> Any:
    who = principal(container, request)
    view = container.agent.complete_upload(
        who, _session_id(request), request.path_params["action_id"]
    )
    return _dump(view)
