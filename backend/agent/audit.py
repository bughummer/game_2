"""Agent audit records stored as ``AuditEvent`` objects (ml-assistant C11).

An ``AgentAuditRecord`` maps to an ``AuditEvent`` with action ``agent.<event_type>``, stored
under the session's own prefix (``AgentLayout.audit_prefix``). ``AuditEvent.detail`` is a flat
``dict[str, str]``, so record fields without a slot are encoded as reserved ``@``-prefixed keys
and caller detail keys are prefixed ``d.``.
"""

import logging
import re
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from backend.audit import AuditEvent, AuditLog
from ml_engine.contracts.agent import AgentAuditRecord, Session
from ml_engine.io.layout import AgentLayout

LOGGER = logging.getLogger("ml_factory.agent.audit")
ACTION_PREFIX = "agent."


class AuditTrailItem(BaseModel):
    """One agent record and, when it has a ``service_audit_ref``, that service's own entry."""

    model_config = ConfigDict(extra="forbid")

    record: AgentAuditRecord
    service_entry: AuditEvent | None = None
REDACTED = "[REDACTED]"

_CREDENTIAL_PATTERNS = (
    re.compile(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA)[A-Z0-9]{12,}\b"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]{8,}"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]*"),
    re.compile(
        r"(?i)(X-Amz-(?:Signature|Credential|Security-Token)|aws_secret_access_key|"
        r"aws_session_token|secret|token|password|api[_-]?key)\s*[=:]\s*[^\s&\"',;]+"
    ),
    re.compile(r"(?<![A-Za-z0-9/+=])[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])(?=.*[/+])"),
)
_KV = re.compile(r"(?i)^(X-Amz-|aws_|secret|token|password|api[_-]?key)")


def scrub_text(value: str) -> str:
    """Replace credential-shaped substrings with a fixed marker."""
    for pattern in _CREDENTIAL_PATTERNS[:3]:
        value = pattern.sub(REDACTED, value)
    value = _CREDENTIAL_PATTERNS[3].sub(lambda m: f"{m.group(1)}={REDACTED}", value)
    return _CREDENTIAL_PATTERNS[4].sub(REDACTED, value)


def scrub_detail(detail: dict[str, str]) -> dict[str, str]:
    return {
        str(k): (REDACTED if _KV.match(str(k)) else scrub_text(str(v))) for k, v in detail.items()
    }


def to_event(rec: AgentAuditRecord) -> AuditEvent:
    """Convert a record to an ``AuditEvent`` (event id is a stable function of session and seq)."""
    detail: dict[str, str] = {f"d.{k}": v for k, v in rec.detail.items()}
    detail["@owner"] = rec.owner
    detail["@seq"] = str(rec.seq)
    detail["@result_status"] = rec.result_status
    detail["@prompt_injection_suspected"] = "true" if rec.prompt_injection_suspected else "false"
    for key in ("tool", "args_hash", "decision", "decided_by", "service_audit_ref"):
        value = getattr(rec, key)
        if value is not None:
            detail[f"@{key}"] = value
    return AuditEvent(
        event_id=f"{rec.seq:06d}",
        experiment_id=rec.session_id,
        action=f"{ACTION_PREFIX}{rec.event_type}",
        actor=rec.actor,
        at=rec.at,
        detail=detail,
    )


def from_event(event: AuditEvent) -> AgentAuditRecord:
    """Inverse of ``to_event``."""
    if not event.action.startswith(ACTION_PREFIX):
        raise ValueError("not an agent audit event")
    d = event.detail
    return AgentAuditRecord(
        session_id=event.experiment_id,
        owner=d["@owner"],
        actor=event.actor or "",
        seq=int(d["@seq"]),
        event_type=event.action[len(ACTION_PREFIX) :],
        at=event.at,
        result_status=d["@result_status"],  # type: ignore[arg-type]
        tool=d.get("@tool"),
        args_hash=d.get("@args_hash"),
        decision=d.get("@decision"),
        decided_by=d.get("@decided_by"),
        service_audit_ref=d.get("@service_audit_ref"),
        prompt_injection_suspected=d.get("@prompt_injection_suspected") == "true",
        detail={k[2:]: v for k, v in d.items() if k.startswith("d.")},
    )


class AgentAudit:
    """Writes agent audit records; failures raise ``AuditWriteError`` (never swallowed)."""

    def __init__(self, log: AuditLog, root: str) -> None:
        self._log = log
        self._root = root

    def write(self, session: Session, rec: AgentAuditRecord) -> str | None:
        """Assign the next per-session ``audit_seq``, scrub, write, and return the object URI.

        ``session.audit_seq`` is incremented only after a successful write, so a failed write
        leaves the sequence unchanged. The caller persists the session.
        """
        seq = session.audit_seq + 1
        rec = rec.model_copy(
            update={
                "seq": seq,
                "session_id": session.session_id,
                "owner": session.owner,
                "detail": scrub_detail(rec.detail),
                "at": rec.at or datetime.now(UTC),
            }
        )
        event = to_event(rec)
        uri = self._log.write(
            artifact_prefix=AgentLayout.for_session(self._root, session.session_id).base,
            experiment_id=event.experiment_id,
            action=event.action,
            actor=event.actor,
            detail=event.detail,
            at=event.at,
        )
        session.audit_seq = seq
        return uri

    def trail(self, session: Session) -> list[AuditTrailItem]:
        """The session's records ordered by ``(at, seq)``, each with its service entry (Req 11.3,
        11.6). A reference that cannot be read yields ``service_entry=None``; the entry itself is
        returned as stored."""
        base = AgentLayout.for_session(self._root, session.session_id).base
        records = [
            from_event(e)
            for e in self._log.list_events(base)
            if e.action.startswith(ACTION_PREFIX) and e.experiment_id == session.session_id
        ]
        records.sort(key=lambda r: (r.at, r.seq))
        items: list[AuditTrailItem] = []
        for rec in records:
            entry: AuditEvent | None = None
            if rec.service_audit_ref:
                try:
                    entry = self._log.read(rec.service_audit_ref)
                except Exception:
                    LOGGER.warning("Service audit entry of session %s unreadable", rec.session_id)
            items.append(AuditTrailItem(record=rec, service_entry=entry))
        return items
