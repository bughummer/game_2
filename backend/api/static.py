"""Serve the thin static UI (control-ui-api design C6).

The UI is a single HTML page plus one stylesheet and one script, served from ``backend/api/ui/``.
It holds no logic and computes no metric — it only calls the REST API. Served here with plain
file responses (no new dependency, no build step), so ``/`` returns the app shell and ``/ui/<file>``
returns its two assets. Unknown asset names are refused rather than reaching the filesystem.
"""

from collections.abc import Callable
from pathlib import Path

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from backend.container import AppContainer

UI_DIR = Path(__file__).resolve().parent / "ui"

#: The only files the UI serves. A fixed allow-list keeps the route from reading arbitrary paths.
_ASSETS: dict[str, str] = {
    "app.js": "text/javascript; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
}


def ui_routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    """The UI shell at ``/`` and its assets at ``/ui/<name>`` (no API prefix)."""
    return [
        ("GET", "/", _index),
        ("GET", "/ui/{name}", _asset),
    ]


def _index(_container: AppContainer) -> Callable:
    async def handler(_request: Request) -> Response:
        html = (UI_DIR / "index.html").read_text(encoding="utf-8")
        return Response(html, media_type="text/html; charset=utf-8")

    return handler


def _asset(_container: AppContainer) -> Callable:
    async def handler(request: Request) -> Response:
        name = request.path_params["name"]
        media_type = _ASSETS.get(name)
        if media_type is None:
            return PlainTextResponse("Not found", status_code=404)
        return Response((UI_DIR / name).read_text(encoding="utf-8"), media_type=media_type)

    return handler
