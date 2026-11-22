"""Result pipeline: what a tool result looks like before it reaches the model (design C8).

Pure functions, no I/O. ``shape_result`` runs, in order: JSON-safe conversion, scrub, govern,
injection detection, string truncation, size cap, ledger. The model only ever sees
``ShapedResult.model_view``, which is inside the ``untrusted_data`` envelope and never larger
than ``max_bytes``.

Requirements: 3.6, 3.8, 8.1 to 8.4, 8.7 to 8.9.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
import math
import numbers
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import PurePath
from typing import TYPE_CHECKING, Any

from backend.governance import GovernancePolicy, apply_governance
from ml_engine.contracts.agent import LedgerEntry

if TYPE_CHECKING:  # ToolSpec lives in backend/agent/tools.py; only duck-typed here.
    from backend.agent.tools import ToolSpec

__all__ = [
    "ScrubContext",
    "ShapedResult",
    "detect_injection",
    "envelope",
    "shape_result",
]

TRUNCATION_SUFFIX = "…[truncated]"
REDACTED = "[redacted]"
REDACTED_URI = "[redacted-uri]"
_MAX_DEPTH = 40

#: Keys whose value is dropped entirely (Req 8.9).
_DROP_KEYS = frozenset({"upload_url", "download_url", "presigned_url", "headers"})
#: Keys whose value is replaced because the key says it is a secret.
_SECRET_KEYS = frozenset(
    {
        "secret_access_key",
        "aws_secret_access_key",
        "session_token",
        "aws_session_token",
        "security_token",
        "secret_key",
        "password",
        "api_key",
        "authorization",
    }
)
#: Row-shaped keys removed before governance (Req 8.1).
_ROW_KEYS = frozenset({"rows", "preview_rows", "sample_rows", "records", "head"})
_MAX_ROW_LIST = 5


@dataclass(frozen=True, slots=True)
class ScrubContext:
    """What the scrubber needs to rewrite storage URIs into references.

    ``experiment_prefixes`` maps a linked experiment id to its S3 prefix.
    ``dataset_uris`` maps a registered dataset URI to its dataset id.
    """

    experiment_prefixes: Mapping[str, str] = field(default_factory=dict)
    dataset_uris: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShapedResult:
    full: dict[str, Any]  # governed, scrubbed, not truncated
    model_view: dict[str, Any]  # enveloped, within max_bytes
    truncated: bool
    injection_suspected: bool
    ledger: tuple[LedgerEntry, ...]


# ----------------------------------------------------------------------------- envelope


def envelope(call_id: str, tool: str, body: dict, *, truncated: bool) -> dict:
    return {
        "untrusted_data": {
            "tool_call_id": call_id,
            "tool": tool,
            "truncated": truncated,
            "data": body,
        }
    }


# ----------------------------------------------------------------------------- injection

_INJECTION_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"ignore\s+(?:all\s+|any\s+)?(?:the\s+)?(?:previous|earlier|prior|above)?\s*instructions",
        r"system\s+prompt",
        r"you\s+are\s+now\b",
        r"\bdisregard\b",
        r"call\s+the\s+tool",
        r"\bapprove\b",
        r"\bautopilot\b",
        r"set\s+mode",
        r"</?\s*untrusted_data\s*>",
    )
)


def detect_injection(strings: Iterable[str]) -> bool:
    """True when any string looks like an instruction to the agent. Only sets a flag."""
    for text in strings:
        if isinstance(text, str) and any(p.search(text) for p in _INJECTION_PATTERNS):
            return True
    return False


# ----------------------------------------------------------------------------- step 1


def _json_safe(value: Any, depth: int = 0) -> Any:
    if depth > _MAX_DEPTH:
        return "[depth-limit]"
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, enum.Enum):
        return _json_safe(value.value, depth + 1)
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            return _json_safe(value.model_dump(mode="json"), depth + 1)
        except Exception:  # fall through to a generic rendering
            return f"[unserializable {type(value).__name__}]"
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _json_safe(dataclasses.asdict(value), depth + 1)
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[_unique_key(out, str(key))] = _json_safe(item, depth + 1)
        return out
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item, depth + 1) for item in sorted(value, key=repr)]
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value]
    if isinstance(value, (_dt.datetime, _dt.date, _dt.time)):
        return value.isoformat()
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, (bytes, bytearray)):
        return "[bytes]"
    return str(value)


def _unique_key(existing: Mapping[str, Any], key: str) -> str:
    if key not in existing:
        return key
    n = 2
    while f"{key}#{n}" in existing:
        n += 1
    return f"{key}#{n}"


# ----------------------------------------------------------------------------- step 2

_S3_URI = re.compile(r"s3a?://[^\s\"'<>,;)\]}]+", re.IGNORECASE)
_SIGNED_URL = re.compile(r"https?://\S*X-Amz-\S*", re.IGNORECASE)
_AMZ_PARAM = re.compile(r"X-Amz-(?:Signature|Security-Token|Credential)=[^\s&\"']*", re.IGNORECASE)
_ACCESS_KEY_ID = re.compile(r"\b(?:AKIA|ASIA|AIDA|AROA|AGPA)[A-Z0-9]{16}\b")
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)\b(?:aws_secret_access_key|secret_access_key|secret[_-]?key|aws_session_token"
    r"|session_token|password|passwd|api[_-]?key)\b\s*[=:]\s*[\"']?[^\s\"',;]+"
)


def _scrub_string(text: str, ctx: ScrubContext) -> str:
    text = _SIGNED_URL.sub(REDACTED, text)
    text = _AMZ_PARAM.sub(REDACTED, text)
    text = _ACCESS_KEY_ID.sub(REDACTED, text)
    text = _SECRET_ASSIGNMENT.sub(REDACTED, text)
    if "s3" in text.lower():
        text = _S3_URI.sub(lambda m: _rewrite_uri(m.group(0), ctx), text)
    return text


def _rewrite_uri(uri: str, ctx: ScrubContext) -> str:
    dataset_id = ctx.dataset_uris.get(uri)
    if dataset_id is not None:
        return f"dataset:{dataset_id}"
    best: tuple[int, str, str] | None = None
    for experiment_id, prefix in ctx.experiment_prefixes.items():
        base = prefix if prefix.endswith("/") else prefix + "/"
        if uri.startswith(base) and (best is None or len(base) > best[0]):
            best = (len(base), experiment_id, uri[len(base):])
    if best is not None:
        return f"artifact://{best[1]}/{best[2]}"
    return REDACTED_URI


def scrub_sensitive(value: Any, ctx: ScrubContext) -> Any:
    """Remove credentials and rewrite storage URIs before persisting untrusted JSON."""
    if isinstance(value, str):
        return _scrub_string(value, ctx)
    if isinstance(value, list):
        return [scrub_sensitive(item, ctx) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower()
            if lowered in _DROP_KEYS:
                continue
            new_key = _unique_key(out, _scrub_string(key, ctx))
            out[new_key] = REDACTED if lowered in _SECRET_KEYS else scrub_sensitive(item, ctx)
        return out
    return value


# ----------------------------------------------------------------------------- step 3


def _drop_rows(value: Any) -> Any:
    if isinstance(value, list):
        return [_drop_rows(item) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower()
            if lowered in _ROW_KEYS:
                continue
            if (
                "row" in lowered
                and isinstance(item, list)
                and sum(isinstance(x, dict) for x in item) > _MAX_ROW_LIST
            ):
                continue
            out[key] = _drop_rows(item)
        return out
    return value


# ----------------------------------------------------------------------------- strings


def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _walk_strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _walk_strings(item)


class _Counter:
    __slots__ = ("n",)

    def __init__(self) -> None:
        self.n = 0


def _truncate_strings(value: Any, max_chars: int, counter: _Counter) -> Any:
    def cut(text: str) -> str:
        if len(text) > max_chars:
            counter.n += 1
            return text[:max_chars] + TRUNCATION_SUFFIX
        return text

    if isinstance(value, str):
        return cut(value)
    if isinstance(value, list):
        return [_truncate_strings(item, max_chars, counter) for item in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            out[_unique_key(out, cut(key))] = _truncate_strings(item, max_chars, counter)
        return out
    return value


# ----------------------------------------------------------------------------- size cap


def _size(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def _lists(node: Any, parent: Any = None, key: Any = None) -> list[tuple[Any, Any, list]]:
    found: list[tuple[Any, Any, list]] = []
    if isinstance(node, list):
        if node:
            found.append((parent, key, node))
        for index, item in enumerate(node):
            found.extend(_lists(item, node, index))
    elif isinstance(node, dict):
        for k, item in node.items():
            found.extend(_lists(item, node, k))
    return found


def _shrink_largest_list(body: dict[str, Any]) -> bool:
    candidates = _lists(body)
    if not candidates:
        return False
    parent, key, target = max(candidates, key=lambda c: _size(c[2]))
    keep = len(target) // 2
    omitted = len(target) - keep
    del target[keep:]
    if isinstance(parent, dict):
        marker = f"{key}_omitted"
        previous = parent.get(marker)
        parent[marker] = omitted + (previous if isinstance(previous, int) else 0)
    return True


def _drop_largest_key(body: dict[str, Any]) -> bool:
    candidates = [k for k in body if not str(k).startswith("_")]
    if not candidates:
        return False
    victim = max(candidates, key=lambda k: _size(body[k]))
    del body[victim]
    dropped = body.setdefault("_dropped_keys", [])
    dropped.append(str(victim)[:64])
    return True


def _cap(
    call_id: str, tool: str, body: dict[str, Any], truncated: bool, max_bytes: int
) -> tuple[dict, dict, bool]:
    """Return (final body, envelope, truncated) with the envelope within ``max_bytes``."""
    env = envelope(call_id, tool, body, truncated=truncated)
    if _size(env) <= max_bytes:
        return body, env, truncated
    truncated = True
    while True:
        env = envelope(call_id, tool, body, truncated=True)
        if _size(env) <= max_bytes:
            return body, env, True
        if _shrink_largest_list(body):
            continue
        if _drop_largest_key(body):
            continue
        break
    names = [str(k)[:64] for k in body if k != "_dropped_keys"] + list(
        body.get("_dropped_keys", [])
    )
    while True:
        body = {"_omitted_keys": names} if names else {}
        env = envelope(call_id, tool, body, truncated=True)
        if _size(env) <= max_bytes or not names:
            return body, env, True
        names = names[: len(names) // 2]


# ----------------------------------------------------------------------------- ledger


def _ledger(full: Any, call_id: str, experiment_id: str | None) -> tuple[LedgerEntry, ...]:
    entries: list[LedgerEntry] = []

    def visit(node: Any, path: str, label: str | None) -> None:
        if isinstance(node, bool):
            return
        if isinstance(node, (int, float)):
            entries.append(
                LedgerEntry(
                    value=float(node),
                    tool_call_id=call_id,
                    experiment_id=experiment_id,
                    path=path,
                    metric_label=label,
                )
            )
        elif isinstance(node, list):
            for index, item in enumerate(node):
                visit(item, f"{path}[{index}]", label)
        elif isinstance(node, dict):
            sibling = node.get("metric_label", node.get("label"))
            own = sibling if isinstance(sibling, str) else label
            for key, item in node.items():
                visit(item, f"{path}.{key}" if path else key, own)

    visit(full, "", None)
    return tuple(entries)


# ----------------------------------------------------------------------------- entry point


def shape_result(
    raw: Any,
    *,
    spec: ToolSpec,
    call_id: str,
    policy: GovernancePolicy,
    ctx: ScrubContext,
    max_bytes: int,
    max_string_chars: int,
    experiment_id: str | None = None,
) -> ShapedResult:
    """Turn a handler's raw output into the governed, bounded view the model may see."""
    safe = _json_safe(raw)
    if not isinstance(safe, dict):
        safe = {"result": safe}

    scrubbed = scrub_sensitive(safe, ctx)
    full = apply_governance(_drop_rows(scrubbed), policy)

    injection = detect_injection(_walk_strings(full))

    counter = _Counter()
    view = _truncate_strings(full, max_string_chars, counter)
    tool = str(getattr(spec, "name", ""))
    _, env, truncated = _cap(call_id, tool, view, counter.n > 0, max_bytes)

    flags = getattr(spec, "flags", None)
    ledger = _ledger(full, call_id, experiment_id) if getattr(flags, "ledger_source", True) else ()

    return ShapedResult(
        full=full,
        model_view=env,
        truncated=truncated,
        injection_suspected=injection,
        ledger=ledger,
    )
