"""Rebuild the model conversation from the persisted transcript (design C14).

Messages are rebuilt on every call. Tool results appear only as ``untrusted_data`` envelopes: the
model view stored with the ``tool_result`` entry (``data["model_view"]``) when present, otherwise
an envelope made from the tool call record's summary. A job update is an enveloped text block in
the same user message as the results it follows.

Compaction: when the estimated size (characters / 4, or the last reported ``input_tokens`` if
larger) exceeds ``budget_tokens``, messages older than the most recent 6 are replaced by one
deterministic summary. Each summarized tool call keeps one line ``[<call_id>] <tool> <status>``
followed by every ``path=value`` pair of the numbers in its result. The goal and the accepted
plan are kept. The session is only read, never changed (Req 6.11).

Requirements: 8.4, 8.5, 2.1, 6.11.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from backend.agent.converse import (
    Block,
    ConverseMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from backend.agent.results import _ledger, envelope
from ml_engine.contracts.agent import (
    CompactionInfo,
    Session,
    ToolCallRecord,
    TranscriptEntry,
)

__all__ = ["KEEP_RECENT_MESSAGES", "build_messages"]

#: Messages after the goal that are never summarized.
KEEP_RECENT_MESSAGES = 6
_USER_TEXT_CHARS = 200


@dataclass(slots=True)
class _Part:
    role: str
    blocks: list[Block]
    entries: list[TranscriptEntry] = field(default_factory=list)


def _append(
    parts: list[_Part], role: str, block: Block, entry: TranscriptEntry | None = None
) -> None:
    if not parts or parts[-1].role != role:
        parts.append(_Part(role, []))
    parts[-1].blocks.append(block)
    if entry is not None:
        parts[-1].entries.append(entry)


def _result_content(entry: TranscriptEntry, record: ToolCallRecord | None) -> dict[str, Any]:
    view = entry.data.get("model_view")
    if isinstance(view, dict) and "untrusted_data" in view:
        return view
    call_id = entry.tool_call_id or ""
    tool = record.tool if record is not None else str(entry.data.get("tool", ""))
    body: dict[str, Any] = {}
    summary = entry.text or (record.result_summary if record is not None else None)
    if summary:
        body["summary"] = summary
    error = record.error_code if record is not None else entry.data.get("error_code")
    if error:
        body["error_code"] = error
    if record is not None:
        body["state"] = record.state.value
    return envelope(call_id, tool, body, truncated=bool(record and record.truncated))


# ----------------------------------------------------------------------------- size


def _json_len(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _block_chars(block: Block) -> int:
    if isinstance(block, TextBlock):
        return len(block.text)
    if isinstance(block, ToolUseBlock):
        return len(block.request_id) + len(block.name) + _json_len(block.arguments)
    if isinstance(block, ToolResultBlock):
        return len(block.request_id) + _json_len(block.content)
    return 0


def _last_input_tokens(session: Session) -> int:
    for entry in sorted(session.transcript, key=lambda e: e.seq, reverse=True):
        value = entry.data.get("input_tokens")
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return 0


def _estimate_tokens(parts: list[_Part], session: Session) -> float:
    chars = sum(_block_chars(b) for p in parts for b in p.blocks)
    return max(chars / 4, _last_input_tokens(session))


# ----------------------------------------------------------------------------- summary


def _format_number(value: float) -> str:
    if value.is_integer() and abs(value) < 2**53:
        return str(int(value))
    return repr(value)


def _number_pairs(content: dict[str, Any], call_id: str) -> list[str]:
    data = (content.get("untrusted_data") or {}).get("data") if isinstance(content, dict) else None
    if not isinstance(data, (dict, list)):
        return []
    return [f"{e.path}={_format_number(e.value)}" for e in _ledger(data, call_id, None)]


def _summary_text(
    session: Session, entries: list[TranscriptEntry], records: dict[str, ToolCallRecord]
) -> str:
    calls: dict[str, TranscriptEntry | None] = {}  # call id -> tool_result entry, first-seen order
    lines: list[str] = []
    for entry in entries:
        call_id = entry.tool_call_id or ""
        if entry.kind == "tool_call":
            calls.setdefault(call_id, None)
        elif entry.kind == "tool_result":
            calls[call_id] = entry
        elif entry.kind == "user_message" and entry.text:
            lines.append(f"user: {entry.text[:_USER_TEXT_CHARS]}")
        elif entry.kind == "job_update":
            state = entry.data.get("state", "")
            pairs = [
                f"{e.path}={_format_number(e.value)}"
                for e in _ledger(dict(entry.data), call_id, None)
            ]
            head = f"[{call_id or 'job_update'}] job_update {state}".rstrip()
            lines.append(" ".join([head, *pairs]))

    call_lines: list[str] = []
    for call_id, result in calls.items():
        record = records.get(call_id)
        tool = record.tool if record is not None else ""
        status = record.state.value if record is not None else ""
        pairs: list[str] = []
        if result is not None:
            content = _result_content(result, record)
            if not tool:
                tool = str((content.get("untrusted_data") or {}).get("tool", ""))
            if not status:
                status = str(result.data.get("state", "unknown"))
            pairs = _number_pairs(content, call_id)
        head = f"[{call_id}] {tool or 'unknown'} {status or 'unknown'}"
        call_lines.append(" ".join([head, *pairs]))

    first = min(e.seq for e in entries)
    last = max(e.seq for e in entries)
    out = [f"[context summary] Earlier turns (transcript seq {first} to {last}) are summarized."]
    if session.plan is not None:
        out.append("Accepted plan:")
        out.extend(f"{i}. {step.text}" for i, step in enumerate(session.plan.steps, start=1))
    if call_lines:
        out.append("Tool calls:")
        out.extend(call_lines)
    if lines:
        out.append("Other events:")
        out.extend(lines)
    return "\n".join(out)


# ----------------------------------------------------------------------------- entry point


def build_messages(
    session: Session, budget_tokens: int
) -> tuple[list[ConverseMessage], CompactionInfo]:
    """Map the transcript to Converse messages. The first message is always the goal."""
    records = {r.tool_call_id: r for r in session.tool_calls}
    parts: list[_Part] = []
    _append(parts, "user", TextBlock(session.goal))
    goal_skipped = False

    for entry in sorted(session.transcript, key=lambda e: e.seq):
        kind = entry.kind
        if kind == "user_message":
            text = entry.text or ""
            if not goal_skipped and text == session.goal:
                goal_skipped = True  # the goal was already added as the first message
                continue
            goal_skipped = True
            if text:
                _append(parts, "user", TextBlock(text), entry)
        elif kind == "assistant_text":
            if entry.text:
                _append(parts, "assistant", TextBlock(entry.text), entry)
        elif kind == "tool_call":
            record = records.get(entry.tool_call_id or "")
            if record is not None:
                _append(
                    parts,
                    "assistant",
                    ToolUseBlock(record.tool_call_id, record.tool, dict(record.arguments)),
                    entry,
                )
        elif kind == "tool_result":
            call_id = entry.tool_call_id or ""
            _append(
                parts,
                "user",
                ToolResultBlock(call_id, _result_content(entry, records.get(call_id))),
                entry,
            )
        elif kind == "job_update":
            body = dict(entry.data)
            if entry.text:
                body.setdefault("message", entry.text)
            call_id = entry.tool_call_id or "job_update"
            wrapped = envelope(call_id, "job_update", body, truncated=False)
            _append(
                parts,
                "user",
                TextBlock(json.dumps(wrapped, ensure_ascii=False, sort_keys=True)),
                entry,
            )
        # plan, approvals, status, mode, limit_stop, error, final_answer: not sent to the model.

    info = session.compaction or CompactionInfo()
    if _estimate_tokens(parts, session) > budget_tokens:
        # Keep at least the last N messages; the kept tail starts with an assistant message so
        # every kept tool result still follows its tool use and roles keep alternating.
        start = len(parts) - KEEP_RECENT_MESSAGES
        while start > 1 and parts[start].role == "user":
            start -= 1
        summarized = [e for p in parts[1:start] for e in p.entries]
        if start > 1 and summarized:
            summary = _summary_text(session, summarized, records)
            head = _Part("user", [*parts[0].blocks, TextBlock(summary)])
            parts = [head, *parts[start:]]
            info = CompactionInfo(
                compacted_through_seq=max(e.seq for e in summarized),
                summary_tokens_estimate=math.ceil(len(summary) / 4),
                compacted_at=datetime.now(UTC),
            )

    messages = [ConverseMessage(role=p.role, blocks=tuple(p.blocks)) for p in parts]  # type: ignore[arg-type]
    return messages, info
