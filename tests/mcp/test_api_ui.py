"""The thin static UI is served by the API (control-ui-api task 8).

No socket: the shell and asset routes are resolved on the HTTP app and their endpoints are called
directly, the same pattern as the other API tests. The UI holds no logic — these tests only check
that it is mounted, served with the right content type, and that unknown assets are refused.
"""

import pytest
from starlette.requests import Request

from mcp_server.app import build_app


def _request() -> Request:
    return Request(
        {"type": "http", "method": "GET", "headers": [], "query_string": b"", "path_params": {}}
    )


async def _call(app, path, *, path_params=None):
    http_app = app.http_app()
    route = next(r for r in http_app.routes if getattr(r, "path", None) == path)
    request = _request()
    request.scope["path_params"] = path_params or {}
    response = await route.endpoint(request)
    body = response.body if isinstance(response.body, bytes) else bytes(response.body)
    return response.status_code, response.media_type, body.decode("utf-8")


@pytest.mark.anyio
async def test_the_ui_shell_is_served_at_root(container):
    app = build_app(container)
    status, media_type, html = await _call(app, "/")
    assert status == 200
    assert media_type.startswith("text/html")
    assert "ML" in html and "control console" in html
    assert "/ui/app.js" in html and "/ui/styles.css" in html


@pytest.mark.anyio
async def test_the_ui_assets_are_served(container):
    app = build_app(container)
    status, media_type, js = await _call(app, "/ui/{name}", path_params={"name": "app.js"})
    assert status == 200
    assert "javascript" in media_type
    assert "/api" in js  # the client calls the REST API

    status, media_type, _css = await _call(app, "/ui/{name}", path_params={"name": "styles.css"})
    assert status == 200
    assert "css" in media_type


@pytest.mark.anyio
async def test_an_unknown_asset_is_refused(container):
    app = build_app(container)
    status, _media_type, _body = await _call(
        app, "/ui/{name}", path_params={"name": "../secrets.py"}
    )
    assert status == 404
