"""CLI safety and ordering checks that also run without AWS fixtures."""

from __future__ import annotations

import contextlib
import io
import sys
from types import SimpleNamespace
from unittest.mock import patch

from ml_factory_cli.__main__ import SESSION_ID, _stream
from ml_factory_cli.transport import HttpTransport


class FakeTransport:
    def __init__(self, view):
        self.view = view
        self.calls = []

    def request(self, method, path, body=None):
        self.calls.append((method, path, body))
        return self.view


def test_noninteractive_approval_is_never_sent():
    view = {
        "session_id": "ses_0123456789abcdef",
        "status": "waiting_for_approval",
        "mode": "supervised",
        "transcript": [
            {"seq": 2, "kind": "assistant_text", "text": "second", "data": {}},
            {"seq": 1, "kind": "assistant_text", "text": "first", "data": {}},
        ],
        "pending_approvals": [{
            "approval_id": "apr_1", "status": "pending", "tool": "create_experiment",
            "arguments": {"name": "sample"}, "args_hash": "abc", "rationale": "Review this",
        }],
    }
    transport = FakeTransport(view)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = _stream(transport, view["session_id"], poll=1, watch=False, interval=30)
    assert code == 75
    assert all(method == "GET" for method, _, _ in transport.calls)
    text = output.getvalue()
    assert text.index("first") < text.index("second")
    assert "APPROVAL REQUEST apr_1" in text


def test_cli_recognizes_existing_service_session_ids():
    assert SESSION_ID.fullmatch("ses_" + "a" * 16)
    assert SESSION_ID.fullmatch("ses_" + "a" * 32)


def test_http_transport_uses_api_route_and_claim_header():
    calls = []

    class FakeClient:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))

        def request(self, method, path, **kwargs):
            calls.append((method, path, kwargs))
            return SimpleNamespace(is_error=False, json=lambda: {"items": []})

        def close(self):
            pass

    with patch.dict(sys.modules, {"httpx": SimpleNamespace(Client=FakeClient)}):
        transport = HttpTransport("http://localhost:8000", "alice")
        assert transport.request("GET", "/agent/sessions") == {"items": []}
    assert calls[1] == (
        "GET", "/api/agent/sessions",
        {"json": None, "headers": {"X-ML-Factory-User": "alice"}},
    )
