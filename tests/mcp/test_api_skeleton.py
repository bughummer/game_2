"""The REST API skeleton: mounting, error mapping, principal resolution (control-ui-api task 1).

The API is a peer of the MCP on the same HTTP app over the same container. No socket is opened —
routes are resolved on the app and their endpoints called directly with a minimal ASGI scope, the
same way the MCP's /health is tested.
"""

import json

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import USER_HEADER, principal
from backend.api.errors import error_response, generic_error_response, safe, status_for
from backend.errors import (
    ArtifactNotReadyError,
    AuthorizationError,
    ConflictError,
    NotFoundError,
    UpstreamError,
    ValidationError,
)
from mcp_server.app import build_app


def _route_paths(app) -> set[str]:
    return {getattr(route, "path", None) for route in app.routes}


def _request(headers: dict[str, str] | None = None) -> Request:
    raw = [(k.lower().encode(), v.encode()) for k, v in (headers or {}).items()]
    return Request({"type": "http", "method": "GET", "headers": raw, "query_string": b""})


async def _call(app, path: str, request: Request):
    http_app = app.http_app()
    route = next(r for r in http_app.routes if getattr(r, "path", None) == path)
    response = await route.endpoint(request)
    return response.status_code, json.loads(bytes(response.body))


# --- mounting -------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_the_api_health_route_is_mounted_beside_the_mcp(container):
    app = build_app(container)
    paths = _route_paths(app.http_app())
    assert "/api/health" in paths  # REST API
    assert "/health" in paths  # shared load-balancer probe


@pytest.mark.anyio
async def test_api_health_reports_ok_for_a_complete_configuration(container):
    app = build_app(container)
    status, body = await _call(app, "/api/health", _request())
    assert status == 200
    assert body["status"] == "ok"
    assert body["configuration_problems"] == []


# --- error mapping --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (ValidationError("bad"), 400),
        (AuthorizationError("no"), 403),
        (NotFoundError("missing"), 404),
        (ConflictError("state"), 409),
        (ArtifactNotReadyError("wait"), 409),
        (UpstreamError("aws"), 502),
    ],
)
def test_app_errors_map_to_their_status(error, expected):
    assert status_for(error) == expected
    response = error_response(error)
    assert response.status_code == expected
    body = json.loads(bytes(response.body))
    assert body["code"] == error.code
    assert body["message"] == error.message


@pytest.mark.anyio
async def test_safe_maps_an_app_error_and_hides_everything_else():
    async def raises_app_error(_request):
        raise NotFoundError("no such experiment", {"experiment_id": "x"})

    async def raises_unexpected(_request):
        raise RuntimeError("secret internal detail")

    status, body = await _run(safe(raises_app_error))
    assert status == 404
    assert body["code"] == "NOT_FOUND"
    assert body["details"] == {"experiment_id": "x"}

    status, body = await _run(safe(raises_unexpected))
    assert status == 500
    assert body["code"] == "INTERNAL_ERROR"
    # The internal message never leaks to the client.
    assert "secret internal detail" not in json.dumps(body)


async def _run(handler):
    response: JSONResponse = await handler(_request())
    return response.status_code, json.loads(bytes(response.body))


def test_generic_error_response_leaks_nothing():
    body = json.loads(bytes(generic_error_response().body))
    assert body == {
        "code": "INTERNAL_ERROR",
        "message": "An internal error occurred.",
        "details": {},
    }


# --- principal resolution -------------------------------------------------------------------


def test_principal_uses_the_dev_identity_not_the_claim_header(container):
    # A claim header is never trusted on its own; the dev identity (configured) wins.
    who = principal(container, _request({USER_HEADER: "attacker@evil"}))
    assert who.user == container.settings.dev_identity
    assert who.trusted is False
