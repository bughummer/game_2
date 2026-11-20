"""Append-only audit of mutations (requirement 15.4).

Every state-changing action records who did what, when, to which experiment — written to the
experiment's own prefix as one object per event, so the trail lives with the artifacts and has a
single writer per object (no shared log to contend on). Reads are never audited; only mutations.

This is deliberately simple: a durable, queryable-by-prefix trail, not a SIEM. It records the
action name and a small, caller-supplied detail dict that must not contain data values.
"""

import logging
import uuid
from datetime import UTC, datetime

from pydantic import Field

from ml_engine.contracts.common import StrictModel
from ml_engine.io import ExperimentLayout, ObjectStore, join_uri, read_model, write_model

LOGGER = logging.getLogger("ml_factory.audit")
AUDIT_SCHEMA_VERSION = "1.0"
AUDIT_DIR = "audit"


class AuditEvent(StrictModel):
    schema_version: str = AUDIT_SCHEMA_VERSION
    event_id: str
    experiment_id: str
    action: str
    actor: str | None = None
    at: datetime
    detail: dict[str, str] = Field(default_factory=dict)


class AuditWriteError(RuntimeError):
    """An audit object could not be written (raised by ``AuditLog.write`` only)."""


class AuditLog:
    def __init__(self, store: ObjectStore, *, enabled: bool = True) -> None:
        self._store = store
        self._enabled = enabled

    def record(
        self,
        *,
        artifact_prefix: str,
        experiment_id: str,
        action: str,
        actor: str | None,
        detail: dict[str, str] | None = None,
    ) -> None:
        if not self._enabled:
            return
        now = datetime.now(UTC)
        event = AuditEvent(
            event_id=uuid.uuid4().hex[:12],
            experiment_id=experiment_id,
            action=action,
            actor=actor,
            at=now,
            detail=detail or {},
        )
        layout = ExperimentLayout(base=artifact_prefix)
        uri = join_uri(
            layout.path(AUDIT_DIR), f"{now.strftime('%Y%m%dT%H%M%S')}-{event.event_id}.json"
        )
        try:
            write_model(self._store, uri, event)
        except Exception:  # auditing must never break the operation it records
            LOGGER.exception("Could not write audit event for %s/%s", experiment_id, action)

    def write(
        self,
        *,
        artifact_prefix: str,
        experiment_id: str,
        action: str,
        actor: str | None,
        detail: dict[str, str] | None = None,
        at: datetime | None = None,
    ) -> str | None:
        """Write one event and return its object URI.

        Unlike ``record`` this raises ``AuditWriteError`` on failure, so callers that must not
        proceed without an audit trail can stop. Returns ``None`` when auditing is disabled.
        """
        if not self._enabled:
            return None
        now = at or datetime.now(UTC)
        try:
            event = AuditEvent(
                event_id=uuid.uuid4().hex[:12],
                experiment_id=experiment_id,
                action=action,
                actor=actor,
                at=now,
                detail=detail or {},
            )
            layout = ExperimentLayout(base=artifact_prefix)
            uri = join_uri(
                layout.path(AUDIT_DIR), f"{now.strftime('%Y%m%dT%H%M%S')}-{event.event_id}.json"
            )
            write_model(self._store, uri, event)
        except Exception as exc:
            LOGGER.error("Could not write audit event for %s/%s", experiment_id, action)
            raise AuditWriteError(f"audit write failed for action {action}") from exc
        return uri

    def read(self, uri: str) -> AuditEvent:
        """Read one event by the URI ``write`` returned."""
        return read_model(self._store, uri, AuditEvent)

    def list_events(self, artifact_prefix: str) -> list[AuditEvent]:
        """Every event under an artifact prefix, in object-name order."""
        prefix = ExperimentLayout(base=artifact_prefix).path(AUDIT_DIR) + "/"
        uris = sorted(self._store.list_uris(prefix))
        return [read_model(self._store, uri, AuditEvent) for uri in uris]
