"""CLI rendering of a session view (Req 12.5, 5.10)."""

from __future__ import annotations

from ml_factory_cli.render import (
    ARG_SUMMARY_MAX,
    RESULT_SUMMARY_MAX,
    render_approval,
    render_entry,
    render_final_answer,
    render_view,
)


def _entry(seq: int, kind: str, **kw) -> dict:
    return {"seq": seq, "at": "2025-01-01T00:00:00+00:00", "kind": kind, "data": {}, **kw}


def _approval(**kw) -> dict:
    base = {
        "approval_id": "apr_1",
        "kind": "costly",
        "tool_call_id": "tc_1",
        "tool": "start_free_form_run",
        "arguments": {"experiment_id": "exp_1", "draft_id": "d1", "revision": 2},
        "args_hash": "h",
        "rationale": "Try a custom model.",
        "estimate_usd": None,
        "estimate_basis": None,
        "status": "pending",
        "created_at": "2025-01-01T00:00:00+00:00",
        "expires_at": "2025-01-01T00:15:00+00:00",
        "review": None,
    }
    return {**base, **kw}


def _view(**kw) -> dict:
    base = {
        "session_id": "ses_0123456789abcdef",
        "status": "running",
        "mode": "supervised",
        "autopilot_max_estimated_spend_usd": None,
        "transcript": [],
        "pending_approvals": [],
    }
    return {**base, **kw}


def test_transcript_is_rendered_in_seq_order() -> None:
    view = _view(
        transcript=[
            _entry(3, "assistant_text", text="third"),
            _entry(1, "user_message", text="first"),
            _entry(2, "assistant_text", text="second"),
        ]
    )
    out = render_view(view)
    assert out.index("first") < out.index("second") < out.index("third")


def test_argument_and_result_summaries_are_capped() -> None:
    call = _entry(1, "tool_call", data={"tool": "t", "arguments": {"x": "a" * 1000}})
    result = _entry(
        2, "tool_result", data={"tool": "t", "state": "succeeded", "model_view": {"y": "b" * 2000}}
    )
    call_line = render_entry(call).split("-> call ", 1)[1]
    result_line = render_entry(result).split("<- result ", 1)[1]
    assert len(call_line) <= ARG_SUMMARY_MAX
    assert len(result_line) <= RESULT_SUMMARY_MAX
    assert call_line.endswith("...") and result_line.endswith("...")


def test_short_summaries_are_not_truncated() -> None:
    result = _entry(2, "tool_result", text="profile_dataset succeeded")
    assert render_entry(result).endswith("<- result profile_dataset succeeded")


def test_mode_is_shown() -> None:
    assert "Mode: supervised" in render_view(_view())
    auto = render_view(_view(mode="autopilot", autopilot_max_estimated_spend_usd=25.0))
    assert "Mode: autopilot (max estimated spend $25)" in auto


def test_pending_approval_printed_with_review_and_no_estimate() -> None:
    code = "import pandas as pd\nprint('hi')"
    review = {"draft_id": "d1", "revision": 2, "content": code, "content_hash": "sha256:abc"}
    out = render_view(
        _view(status="waiting_for_approval", pending_approvals=[_approval(review=review)])
    )
    assert "tool: start_free_form_run" in out
    assert '"experiment_id": "exp_1"' in out
    assert "rationale: Try a custom model." in out
    assert "no estimate available" in out
    assert "content_hash: sha256:abc" in out
    assert "import pandas as pd" in out and "print('hi')" in out


def test_approval_with_estimate() -> None:
    out = render_approval(_approval(estimate_usd=1.5, estimate_basis="3 trials"))
    assert "estimate: $1.50 (3 trials)" in out


def test_final_answer_markers_attributions_notes_labels() -> None:
    answer = {
        "sentences": [
            {
                "text": "AUC is 0.91.",
                "status": "verified",
                "attributions": [
                    {
                        "tool_call_id": "tc_9",
                        "experiment_id": "exp_1",
                        "artifact_ref": None,
                        "path": "metrics.auc",
                        "value_text": "0.91",
                        "metric_label": "engine",
                    }
                ],
            },
            {"text": "It is the best possible.", "status": "overclaim", "attributions": []},
            {"text": "Recall is 0.7.", "status": "unverified", "attributions": []},
        ],
        "verbatim": [],
        "notes": ["Leakage findings are advisory."],
        "labels": ["Advisory"],
    }
    out = render_final_answer(answer)
    assert "0.91 <- exp_1:metrics.auc, engine" in out
    assert "[OVERCLAIM] It is the best possible." in out
    assert "[UNVERIFIED] Recall is 0.7." in out
    assert "[VERIFIED]" not in out
    assert "note: Leakage findings are advisory." in out
    assert "labels: Advisory" in out
