"""Register the REST API onto the control-plane HTTP app (control-ui-api design C1).

The API is a peer of the MCP in one process over one ``AppContainer``. Rather than a second ASGI
server, its routes are attached to the same FastMCP HTTP app the MCP already serves (the mechanism
``/health`` uses), so ``/api/...``, ``/sse`` and ``/health`` coexist and ``app.run()`` is unchanged.

Route modules (experiments, datasets, training, model versions, validation, …) register through
``register_api_routes``; task 1 provides the skeleton and the error-mapped wrapper they all use.
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.errors import json_ok, safe
from backend.container import AppContainer

API_PREFIX = "/api"

#: Route modules append (method, path_suffix, handler_factory) here. A handler_factory takes the
#: container and returns an async Starlette handler; app wiring wraps it with ``safe`` and mounts
#: it under API_PREFIX. Keeping this list lets each resource live in its own module (task 2+).
RouteFactory = Callable[[AppContainer], Callable]


def register_api_routes(mcp, container: AppContainer) -> None:
    """Attach every REST route to the FastMCP HTTP app ``mcp``.

    Called from ``mcp_server/app.py::build_app`` after the MCP server is built, so the REST API and
    the MCP tools share the one app and the one container.
    """
    for method, suffix, factory in _collect_routes():
        handler = safe(factory(container))
        mcp.custom_route(f"{API_PREFIX}{suffix}", methods=[method])(handler)
    # The thin static UI (design C6): the shell at "/" and its assets at "/ui/<name>", outside the
    # API prefix. Served from backend/api/ui/; it calls only the /api routes above.
    from backend.api.static import ui_routes

    for method, suffix, factory in ui_routes():
        mcp.custom_route(suffix, methods=[method])(safe(factory(container)))


def _collect_routes() -> list[tuple[str, str, RouteFactory]]:
    """Every (method, path_suffix, handler_factory) the API exposes.

    The readiness probe plus each resource module's collector. Later tasks extend this with the
    remaining resource modules (training, model versions, validation).
    """
    from backend.api.routes import (
        agent,
        analysis,
        datasets,
        experiments,
        features,
        free_form,
        model_versions,
        reports,
        training,
        validation,
    )

    routes: list[tuple[str, str, RouteFactory]] = [
        ("GET", "/health", _api_health),
    ]
    routes.extend(experiments.routes())
    routes.extend(datasets.routes())
    routes.extend(features.routes())
    routes.extend(training.routes())
    routes.extend(analysis.routes())
    routes.extend(reports.routes())
    routes.extend(model_versions.routes())
    routes.extend(validation.routes())
    routes.extend(free_form.routes())
    routes.extend(agent.routes())
    return routes


def _api_health(container: AppContainer) -> Callable:
    async def handler(_request: Request) -> JSONResponse:
        problems = container.settings.problems()
        return json_ok(
            {
                "status": "ok" if not problems else "degraded",
                "configuration_problems": problems,
            },
            status_code=200 if not problems else 503,
        )

    return handler
