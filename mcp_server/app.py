"""HTTP entrypoint for the MCP server — what runs on Fargate behind an internal load balancer.

Builds the control-plane container from ``Settings`` (environment), mounts a ``/health`` route
for the load balancer, and serves the MCP tools over SSE at ``/sse`` for LibreChat — matching
the transport the existing MCP servers in this deployment use. Transport and path are
overridable via ``ML_FACTORY_MCP_TRANSPORT`` / ``ML_FACTORY_MCP_PATH``.

Run locally:
    python -m mcp_server.app            # reads ML_FACTORY_* from the environment

The identity story is unchanged from Phase 6/9: a transport header is a *claim*, never trusted on
its own. The ``Authorizer`` resolves the principal (today: the configured dev identity, flagged);
real authentication is wired here when the LibreChat auth mechanism is decided (task 6.5 / Q8).
"""

import logging
import os

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api import register_api_routes
from backend.config import Settings
from backend.container import AppContainer, build_container
from mcp_server.server import build_mcp_server

LOGGER = logging.getLogger("ml_factory.mcp.app")

# Bound inside a private subnet behind an internal load balancer, not exposed to the internet.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
# SSE transport at /sse, matching the existing LibreChat MCP servers in this deployment
# (e.g. successfactors). Override with ML_FACTORY_MCP_TRANSPORT / ML_FACTORY_MCP_PATH.
DEFAULT_TRANSPORT = "sse"
MCP_PATH = "/sse"


def build_app(container: AppContainer):
    """A FastMCP server with a health route, ready to serve over HTTP."""
    mcp = build_mcp_server(container)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> JSONResponse:
        """Liveness + readiness for the load balancer. Never leaks a credential."""
        problems = container.settings.problems()
        body = {
            "status": "ok" if not problems else "degraded",
            "orchestrator": container.orchestrator.name,
            "artifact_root": container.settings.artifact_root,
            "bedrock_enabled": container.settings.bedrock_enabled,
            "budget_enforced": container.settings.enforce_budget,
            "auth": "dev-identity" if container.settings.dev_identity else "none",
            "configuration_problems": problems,
        }
        return JSONResponse(body, status_code=200 if not problems else 503)

    # The REST API is a peer of the MCP over the same container and the same HTTP app, so
    # /api/... , /sse and /health coexist in one process (control-ui-api design C1).
    register_api_routes(mcp, container)

    return mcp


def serve(settings: Settings | None = None) -> None:
    logging.basicConfig(level=os.environ.get("ML_FACTORY_LOG_LEVEL", "INFO").upper())
    settings = settings or Settings()
    problems = settings.problems()
    if problems:
        # Log, but still start: /health reports "degraded" so the operator can see why, rather
        # than a container that crash-loops with no endpoint to inspect.
        LOGGER.warning("Starting with configuration problems: %s", "; ".join(problems))
    container = build_container(settings)
    if settings.status_poll_interval_seconds > 0:
        # Drives every workflow forward (EDA -> feature review, preparation -> training ->
        # evaluation). Without it, experiments never leave their first stage.
        container.poller.start()
    else:
        LOGGER.warning(
            "Status poller disabled (ML_FACTORY_STATUS_POLL_INTERVAL_SECONDS=0); workflows "
            "advance only if another process drives StatusMonitor (e.g. EventBridge)."
        )
    watcher = getattr(container.agent, "watcher", None)
    if settings.agent_enabled and watcher is not None:
        # Marks agent sessions ready when their jobs end. It never calls the model (D13).
        watcher.start()
    app = build_app(container)
    host = os.environ.get("ML_FACTORY_MCP_HOST", DEFAULT_HOST)
    port = int(os.environ.get("ML_FACTORY_MCP_PORT", DEFAULT_PORT))
    transport = os.environ.get("ML_FACTORY_MCP_TRANSPORT", DEFAULT_TRANSPORT)
    path = os.environ.get("ML_FACTORY_MCP_PATH", MCP_PATH)
    LOGGER.info("Serving ML Factory MCP (%s) on http://%s:%d%s", transport, host, port, path)
    app.run(transport=transport, host=host, port=port, path=path, show_banner=False)


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    serve()
