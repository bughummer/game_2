"""Offline checks of the web chat panel assets (ml-assistant task 17.1, design C19; Req 12.10).

Text scans only: the files exist, never assign HTML markup, and call only routes that
``backend/api/routes/agent.py`` defines.
"""

from __future__ import annotations

import re
from pathlib import Path

from backend.api.routes import agent as agent_routes

UI = Path(__file__).resolve().parents[2] / "backend" / "api" / "ui"
JS = UI / "assistant.js"
CSS = UI / "assistant.css"

_HTML_SINKS = re.compile(r"\.(innerHTML|outerHTML)\s*\+?=|insertAdjacentHTML|document\.write|\bhtml\s*:")
_AGENT_PATH = re.compile(r"[`\"'](/agent/[^`\"'\s]*)")
_JS_PLACEHOLDER = re.compile(r"\$\{[^}]*\}")
_ROUTE_PLACEHOLDER = re.compile(r"\{[^}]*\}")


def _normalize(path: str, placeholder: re.Pattern[str]) -> str:
    return placeholder.sub("{}", path.split("?", 1)[0])


def test_assets_exist_and_are_not_empty():
    assert JS.read_text(encoding="utf-8").strip()
    assert CSS.read_text(encoding="utf-8").strip()


def test_no_html_assignment():
    assert not _HTML_SINKS.findall(JS.read_text(encoding="utf-8"))


def test_only_defined_agent_routes_are_called():
    defined = {_normalize(path, _ROUTE_PLACEHOLDER) for _, path, _ in agent_routes.routes()}
    called = {
        _normalize(p, _JS_PLACEHOLDER) for p in _AGENT_PATH.findall(JS.read_text(encoding="utf-8"))
    }
    assert called, "assistant.js calls no /agent/ route"
    assert called <= defined, sorted(called - defined)
