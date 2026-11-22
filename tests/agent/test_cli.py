"""Property checks for the streamed CLI view (Req 12.5 and 12.6)."""

from __future__ import annotations

import contextlib
import io

from hypothesis import given, settings
from hypothesis import strategies as st

from ml_factory_cli.__main__ import _stream
from ml_factory_cli.render import ARG_SUMMARY_MAX, RESULT_SUMMARY_MAX
from tests.agent.test_cli_stdlib import FakeTransport

SID = "ses_0123456789abcdef"
_text = st.text(alphabet="abcXYZ0123456789", max_size=1200)


# Feature: ml-assistant, Property 49: CLI stream output is bounded and ordered
@settings(max_examples=100, deadline=None)
@given(values=st.lists(_text, min_size=1, max_size=10))
def test_stream_orders_entries_and_bounds_tool_summaries(values):
    entries = []
    for seq, value in enumerate(values, 1):
        if seq % 2:
            entries.append(
                {
                    "seq": seq,
                    "kind": "tool_call",
                    "data": {"tool": "t", "arguments": {"value": value}},
                }
            )
        else:
            entries.append(
                {
                    "seq": seq,
                    "kind": "tool_result",
                    "data": {"tool": "t", "model_view": {"value": value}},
                }
            )
    view = {
        "session_id": SID,
        "status": "completed",
        "mode": "supervised",
        "transcript": list(reversed(entries)),
        "pending_approvals": [],
    }
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert _stream(FakeTransport(view), SID, poll=1, watch=False, interval=30) == 0
    lines = output.getvalue().splitlines()
    rendered = [line for line in lines if line.startswith("[")]
    assert [int(line.split("]", 1)[0][1:]) for line in rendered] == list(range(1, len(entries) + 1))
    for line in rendered:
        if "-> call " in line:
            assert len(line.split("-> call ", 1)[1]) <= ARG_SUMMARY_MAX
        if "<- result " in line:
            assert len(line.split("<- result ", 1)[1]) <= RESULT_SUMMARY_MAX


# Feature: ml-assistant, Property 48: A non-interactive CLI never approves
@settings(max_examples=100, deadline=None)
@given(
    tool=st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=30), argument=_text
)
def test_noninteractive_stream_never_decides(tool, argument):
    view = {
        "session_id": SID,
        "status": "waiting_for_approval",
        "mode": "supervised",
        "transcript": [],
        "pending_approvals": [
            {
                "approval_id": "apr_1",
                "status": "pending",
                "tool": tool,
                "arguments": {"value": argument},
                "args_hash": "hash",
                "rationale": "Review",
            }
        ],
    }
    transport = FakeTransport(view)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        assert _stream(transport, SID, poll=1, watch=False, interval=30) == 75
    assert "APPROVAL REQUEST apr_1" in output.getvalue()
    assert all(method == "GET" for method, _, _ in transport.calls)
