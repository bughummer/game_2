"""Plain-text formatting of a session view for the ``mlf`` CLI.

Every function takes the JSON ``SessionView`` shape (plain dicts, see ``backend/agent/view.py``)
and returns a string. Nothing here prints, reads files or touches the network.
"""

from __future__ import annotations

import json
from typing import Any

ARG_SUMMARY_MAX = 200
RESULT_SUMMARY_MAX = 500
_ELLIPSIS = "..."


def _compact(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def summarize(value: Any, limit: int) -> str:
    """One-line summary of ``value`` of at most ``limit`` characters."""
    text = " ".join(_compact(value).split())
    if len(text) <= limit:
        return text
    return text[: max(limit - len(_ELLIPSIS), 0)] + _ELLIPSIS


def render_mode(view: dict[str, Any]) -> str:
    mode = view.get("mode") or "unknown"
    cap = view.get("autopilot_max_estimated_spend_usd")
    if mode == "autopilot" and cap is not None:
        return f"Mode: autopilot (max estimated spend ${cap:g})"
    return f"Mode: {mode}"


def _tool_call_line(entry: dict[str, Any]) -> str:
    data = entry.get("data") or {}
    tool = data.get("tool") or "tool"
    summary = tool
    if "arguments" in data:
        summary = f"{tool}({_compact(data['arguments'])})"
    return f"-> call {summarize(summary, ARG_SUMMARY_MAX)}"


def _tool_result_line(entry: dict[str, Any]) -> str:
    data = entry.get("data") or {}
    if entry.get("text"):
        body: Any = entry["text"]
    else:
        parts = [str(data.get("tool") or "tool"), str(data.get("state") or "")]
        if data.get("error_code"):
            parts.append(f"error={data['error_code']}")
        if data.get("model_view") is not None:
            parts.append(_compact(data["model_view"]))
        body = " ".join(p for p in parts if p)
    return f"<- result {summarize(body, RESULT_SUMMARY_MAX)}"


def render_entry(entry: dict[str, Any]) -> str:
    """One transcript entry as text."""
    kind = entry.get("kind")
    text = entry.get("text")
    data = entry.get("data") or {}
    seq = entry.get("seq")
    prefix = f"[{seq}] "
    if kind == "tool_call":
        return prefix + _tool_call_line(entry)
    if kind == "tool_result":
        return prefix + _tool_result_line(entry)
    if kind == "user_message":
        return prefix + f"you: {text or ''}"
    if kind == "assistant_text":
        return prefix + f"assistant: {text or ''}"
    if kind == "plan":
        steps = data.get("steps") or []
        lines = [prefix + "plan:"] + [f"  {i}. {s}" for i, s in enumerate(steps, 1)]
        return "\n".join(lines)
    if kind == "approval_request":
        reason = f" ({data['reason']})" if data.get("reason") else ""
        return prefix + (
            f"approval requested: {data.get('tool', 'tool')} [{entry.get('approval_id')}]{reason}"
        )
    if kind == "mode":
        line = f"mode changed to {data.get('mode')}"
        if data.get("max_estimated_spend_usd") is not None:
            line += f" (max estimated spend ${data['max_estimated_spend_usd']:g})"
        return prefix + line
    if kind == "final_answer":
        return prefix + "final answer:\n" + render_final_answer(data)
    label = str(kind or "event").replace("_", " ")
    body = text if text else summarize(data, RESULT_SUMMARY_MAX)
    return prefix + f"{label}: {body}"


def render_transcript(entries: list[dict[str, Any]]) -> str:
    """Entries in ``seq`` order, one block per entry."""
    ordered = sorted(entries, key=lambda e: e.get("seq", 0))
    return "\n".join(render_entry(e) for e in ordered)


def render_approval(approval: dict[str, Any]) -> str:
    """Full approval request: tool, exact arguments, rationale, estimate and review block."""
    lines = [
        f"APPROVAL REQUEST {approval.get('approval_id')} ({approval.get('status')})",
        f"  tool: {approval.get('tool')}",
        "  arguments:",
        _indent(json.dumps(approval.get("arguments") or {}, indent=2, sort_keys=True), 4),
    ]
    if approval.get("edited_arguments") is not None:
        lines.append("  edited arguments (awaiting confirmation):")
        lines.append(_indent(json.dumps(approval["edited_arguments"], indent=2, sort_keys=True), 4))
    lines.append(f"  rationale: {approval.get('rationale') or ''}")
    usd = approval.get("estimate_usd")
    if usd is None:
        lines.append("  estimate: no estimate available")
    else:
        basis = approval.get("estimate_basis")
        lines.append(f"  estimate: ${usd:.2f}" + (f" ({basis})" if basis else ""))
    review = approval.get("review")
    if review:
        lines.append("  review:")
        for key in sorted(k for k in review if k != "content"):
            lines.append(f"    {key}: {_compact(review[key])}")
        if "content" in review:
            lines.append("    content:")
            lines.append(_indent(str(review["content"]), 6))
    if approval.get("expires_at"):
        lines.append(f"  expires at: {approval['expires_at']}")
    return "\n".join(lines)


def render_final_answer(answer: dict[str, Any]) -> str:
    """Sentences with their markers and attributions, then verbatim blocks, notes and labels."""
    lines: list[str] = []
    for sentence in answer.get("sentences") or []:
        status = sentence.get("status")
        marker = f"[{status.upper()}] " if status in ("unverified", "overclaim") else ""
        lines.append(f"  {marker}{sentence.get('text', '')}")
        for a in sentence.get("attributions") or []:
            source = a.get("experiment_id") or a.get("tool_call_id")
            label = f", {a['metric_label']}" if a.get("metric_label") else ""
            lines.append(f"      {a.get('value_text')} <- {source}:{a.get('path')}{label}")
    for block in answer.get("verbatim") or []:
        lines.append(f"  [{block.get('title')}] (from {block.get('tool_call_id')})")
        lines.append(_indent(json.dumps(block.get("content") or {}, indent=2, sort_keys=True), 4))
    for note in answer.get("notes") or []:
        lines.append(f"  note: {note}")
    labels = answer.get("labels") or []
    if labels:
        lines.append("  labels: " + ", ".join(labels))
    return "\n".join(lines)


def render_view(view: dict[str, Any]) -> str:
    """Header with status and mode, the transcript, then every pending approval request."""
    parts = [
        f"Session {view.get('session_id')}: {view.get('status')}",
        render_mode(view),
    ]
    transcript = view.get("transcript") or []
    if transcript:
        parts.append(render_transcript(transcript))
    for approval in view.get("pending_approvals") or []:
        parts.append(render_approval(approval))
    return "\n".join(parts)


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines()) if text else pad
