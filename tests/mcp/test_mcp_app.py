"""The HTTP entrypoint (task 6.5): a /health route and the MCP tools, served over HTTP.

No live server is started — that needs a socket and is a deployment concern. These tests check
that the app is assembled correctly: the health route exists and reports configuration honestly,
and the full tool set is still mounted on the HTTP app.
"""

import json

import pytest
from starlette.requests import Request

from mcp_server.app import MCP_PATH, build_app


def _route_paths(app) -> set[str]:
    return {getattr(route, "path", None) for route in app.routes}


async def _call_health(mcp_app, container):
    """Invoke the registered /health handler directly with a minimal ASGI scope."""
    http_app = mcp_app.http_app()
    health = next(r for r in http_app.routes if getattr(r, "path", None) == "/health")
    request = Request({"type": "http", "method": "GET", "headers": [], "query_string": b""})
    response = await health.endpoint(request)
    return response.status_code, json.loads(bytes(response.body))


@pytest.mark.anyio
async def test_health_reports_ok_for_a_complete_configuration(container):
    mcp_app = build_app(container)
    status, body = await _call_health(mcp_app, container)
    assert status == 200
    assert body["status"] == "ok"
    assert body["orchestrator"] in {"sagemaker", "inline"}
    assert body["auth"] == "dev-identity"
    assert body["configuration_problems"] == []
    # Health never leaks a credential.
    assert "secret" not in json.dumps(body).lower()


@pytest.mark.anyio
async def test_health_reports_degraded_when_configuration_is_incomplete(container):
    # stage_image_uri is now optional (stock-image migration); the role ARN is still required, so
    # clearing it is a genuine configuration problem.
    container.settings.sagemaker_execution_role_arn = ""  # break the config
    mcp_app = build_app(container)
    status, body = await _call_health(mcp_app, container)
    assert status == 503
    assert body["status"] == "degraded"
    assert any("EXECUTION_ROLE_ARN" in p for p in body["configuration_problems"])


@pytest.mark.anyio
async def test_the_http_app_mounts_health_and_the_sse_endpoint(container):
    # SSE is the default transport (matching the existing LibreChat MCP servers); it serves
    # /sse (plus /messages) alongside the health route.
    http_app = build_app(container).http_app(transport="sse", path=MCP_PATH)
    paths = _route_paths(http_app)
    assert "/health" in paths
    assert MCP_PATH in paths


@pytest.mark.anyio
async def test_every_tool_is_available_over_http(container):
    """build_app must not drop tools: the HTTP server exposes the same set as the in-memory one."""
    from fastmcp import Client

    from mcp_server import build_mcp_server

    async with Client(build_mcp_server(container)) as c:
        expected = {t.name for t in await c.list_tools()}
    async with Client(build_app(container)) as c:
        served = {t.name for t in await c.list_tools()}
    assert served == expected
