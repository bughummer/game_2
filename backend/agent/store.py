"""Session store: versioned agent sessions in the object store (ml-assistant design C10).

Concurrency limits, stated plainly. ``ObjectStore`` has no conditional put (no If-Match), so
``save`` compares the stored ETag (from ``stat``) with the one captured at load and then writes.
The window between that check and the write is not closed. Protection is therefore best effort:
an in-process re-entrant lock per session id (``transaction``), the ETag check, and the lease.
A store that reports no ETag (``ObjectMetadata.etag is None``) gets no conflict detection.
This matches the design's "cross-replica safety is best effort" note.
"""

import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import ValidationError as PydanticValidationError

from backend.errors import AppError, ConflictError
from ml_engine.contracts.agent import (
    SUPPORTED_VERSIONS,
    Lease,
    ResultRef,
    Session,
    SessionSummary,
)
from ml_engine.io.layout import AgentLayout
from ml_engine.io.object_store import ObjectNotFoundError, ObjectStore

_SAFE_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
_LoadCategory = Literal["unknown_contract_version", "corrupt_content", "not_found"]


class SessionLoadError(AppError):
    """A session could not be loaded. Carries the session id and a category, never content."""

    code = "VALIDATION_ERROR"

    def __init__(self, session_id: str, category: _LoadCategory, message: str) -> None:
        super().__init__(
            f"Session {session_id}: {message}",
            {"session_id": session_id, "category": category},
        )
        self.session_id = session_id
        self.category: _LoadCategory = category
        if category == "not_found":
            self.code = "NOT_FOUND"


class SessionConflict(ConflictError):
    """The stored session changed since it was loaded."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Session {session_id} was changed by another writer",
            {"session_id": session_id, "reason": "session_conflict"},
        )
        self.session_id = session_id


class SessionPersistError(AppError):
    """The session could not be saved after every attempt. The loop must stop."""

    code = "UPSTREAM_ERROR"

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Session {session_id} could not be saved",
            {"session_id": session_id, "reason": "persist_failed"},
        )
        self.session_id = session_id


class SessionInUse(ConflictError):
    """An unexpired lease is held by another holder."""

    def __init__(self, session_id: str) -> None:
        super().__init__(
            f"Session {session_id} is in use",
            {"session_id": session_id, "reason": "session_in_use"},
        )
        self.session_id = session_id


@dataclass(slots=True)
class LoadedSession:
    session: Session
    etag: str | None


@dataclass(frozen=True, slots=True)
class Page[ItemT]:
    items: list[ItemT] = field(default_factory=list)
    next_cursor: str | None = None


def _utcnow() -> datetime:
    return datetime.now(UTC)


# Versioned contracts stored inside a Session record: (contract name, field, is a list).
_EMBEDDED_CONTRACTS: tuple[tuple[str, str, bool], ...] = (
    ("SessionLimits", "limits", False),
    ("ToolCallRecord", "tool_calls", True),
    ("ApprovalRequest", "approvals", True),
)


def _embedded_versions(payload: dict[str, Any]) -> Iterator[tuple[str, object]]:
    """Yield ``(contract, schema_version)`` for the Session and each embedded contract.

    The Session comes first. An embedded record that is missing or not an object is left to
    model validation (``corrupt_content``); a missing embedded ``schema_version`` means the
    default and is not a version error. The top-level Session version is required.
    """
    if "schema_version" in payload:
        yield "Session", payload["schema_version"]
    for contract, field_name, many in _EMBEDDED_CONTRACTS:
        value = payload.get(field_name)
        records = value if many and isinstance(value, list) else [value]
        for record in records:
            if isinstance(record, dict) and "schema_version" in record:
                yield contract, record["schema_version"]


class SessionStore:
    def __init__(
        self,
        store: ObjectStore,
        root: str,
        *,
        retries: int = 3,
        lease_seconds: int = 300,
        clock: Callable[[], datetime] = _utcnow,
    ) -> None:
        """``retries`` is the total number of write attempts (3 means up to 3 tries)."""
        self._store = store
        self._root = root
        self._attempts = max(1, retries)
        self._lease_seconds = lease_seconds
        self._clock = clock
        self._locks: dict[str, threading.RLock] = {}
        self._locks_guard = threading.Lock()

    # --- helpers ----------------------------------------------------------------------
    def _layout(self, session_id: str) -> AgentLayout:
        if not _SAFE_ID.match(session_id):
            raise SessionLoadError(session_id[:64], "not_found", "no such session")
        return AgentLayout.for_session(self._root, session_id)

    def _lock(self, session_id: str) -> threading.RLock:
        with self._locks_guard:
            return self._locks.setdefault(session_id, threading.RLock())

    def _etag(self, uri: str) -> str | None:
        meta = self._store.stat(uri)
        return None if meta is None else meta.etag

    # --- create / load / save -----------------------------------------------------------
    def create(self, session: Session) -> Session:
        layout = self._layout(session.session_id)
        if self._store.stat(layout.session) is not None:
            raise SessionConflict(session.session_id)
        session.revision = 0
        self._write_with_retries(session, layout, loaded_etag=None, check_conflict=False)
        return session

    def load(self, session_id: str) -> LoadedSession:
        """Read, check ``schema_version``, then validate. Never writes or deletes."""
        layout = self._layout(session_id)
        try:
            raw = self._store.read_bytes(layout.session)
        except (ObjectNotFoundError, FileNotFoundError):
            raise SessionLoadError(session_id, "not_found", "no such session") from None
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            raise SessionLoadError(
                session_id, "corrupt_content", "record is not valid JSON"
            ) from None
        if not isinstance(payload, dict):
            raise SessionLoadError(session_id, "corrupt_content", "record is not an object")
        if "schema_version" not in payload:
            raise SessionLoadError(session_id, "corrupt_content", "record has no schema_version")
        for contract, version in _embedded_versions(payload):
            if not isinstance(version, str) or version not in SUPPORTED_VERSIONS[contract]:
                raise SessionLoadError(
                    session_id,
                    "unknown_contract_version",
                    f"unsupported {contract} schema_version {str(version)[:32]!r}",
                )
        try:
            session = Session.model_validate(payload)
        except PydanticValidationError:
            raise SessionLoadError(
                session_id, "corrupt_content", "record failed validation"
            ) from None
        if session.session_id != session_id:
            raise SessionLoadError(session_id, "corrupt_content", "record names another session")
        return LoadedSession(session=session, etag=self._etag(layout.session))

    def save(self, loaded: LoadedSession) -> LoadedSession:
        """Write revision+1, then the owner index. Up to ``retries`` attempts in total.

        Raises ``SessionConflict`` (no retry) when the stored ETag differs from the loaded one,
        and ``SessionPersistError`` once every attempt failed.
        """
        session = loaded.session
        layout = self._layout(session.session_id)
        etag = self._write_with_retries(
            session, layout, loaded_etag=loaded.etag, check_conflict=True
        )
        return LoadedSession(session=session, etag=etag)

    def _write_with_retries(
        self,
        session: Session,
        layout: AgentLayout,
        *,
        loaded_etag: str | None,
        check_conflict: bool,
    ) -> str | None:
        now = self._clock()
        revision = session.revision + (1 if check_conflict else 0)
        update: dict[str, Any] = {"revision": revision, "updated_at": now}
        if session.lease is not None and session.lease.expires_at > now:
            update["lease"] = session.lease.model_copy(
                update={"expires_at": now + timedelta(seconds=self._lease_seconds)}
            )
        payload = session.model_copy(update=update)
        data = payload.model_dump_json(indent=2, by_alias=True).encode("utf-8")
        summary = SessionSummary(
            session_id=session.session_id,
            owner=session.owner,
            status=session.status,
            mode=session.mode,
            experiment_ids=list(session.experiment_ids),
            last_activity_at=session.last_activity_at,
            created_at=session.created_at,
        )
        index_uri = AgentLayout.owner_index_object(self._root, session.owner, session.session_id)
        index_data = summary.model_dump_json(indent=2).encode("utf-8")

        written = False
        new_etag: str | None = None
        for _ in range(self._attempts):
            try:
                if not written:
                    if check_conflict and self._etag(layout.session) != loaded_etag:
                        raise SessionConflict(session.session_id)
                    self._store.write_bytes(layout.session, data)
                    written = True
                    new_etag = self._etag(layout.session)
                self._store.write_bytes(index_uri, index_data)
            except SessionConflict:
                raise
            except Exception:  # any store failure is retried, none is logged
                continue
            # Success: reflect what was stored on the caller's object.
            session.revision = revision
            session.updated_at = now
            if "lease" in update:
                session.lease = update["lease"]
            return new_etag
        raise SessionPersistError(session.session_id)

    @contextmanager
    def transaction(self, session_id: str) -> Iterator[Session]:
        """Lock the session id in-process, load, yield, save. Not saved if the body raises."""
        with self._lock(session_id):
            loaded = self.load(session_id)
            yield loaded.session
            self.save(loaded)

    # --- tool results -------------------------------------------------------------------
    def put_result(self, session_id: str, call_id: str, full: dict) -> ResultRef:
        layout = self._layout(session_id)
        if not _SAFE_ID.match(call_id):
            raise ValueError("invalid tool call id")
        data = json.dumps(full, sort_keys=True, default=str).encode("utf-8")
        uri = layout.tool_result(call_id)
        for _ in range(self._attempts):
            try:
                self._store.write_bytes(uri, data)
            except Exception:
                continue
            return ResultRef(uri=uri, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        raise SessionPersistError(session_id)

    def put_inflight(self, session_id: str, idempotency_key: str, marker: dict) -> str:
        """Write the in-flight marker of a Costly call before its handler runs (design D11)."""
        layout = self._layout(session_id)
        if not _SAFE_ID.match(idempotency_key):
            raise ValueError("invalid idempotency key")
        data = json.dumps(marker, sort_keys=True, default=str).encode("utf-8")
        uri = layout.inflight(idempotency_key)
        for _ in range(self._attempts):
            try:
                self._store.write_bytes(uri, data)
            except Exception:
                continue
            return uri
        raise SessionPersistError(session_id)

    def get_inflight(self, session_id: str, idempotency_key: str) -> dict | None:
        """The in-flight marker written by ``put_inflight``; ``None`` when there is none.

        Raises ``SessionLoadError`` when the marker exists but is not a JSON object. Store errors
        other than "not found" propagate, so the reconciler can report an unknown outcome.
        """
        layout = self._layout(session_id)
        if not _SAFE_ID.match(idempotency_key):
            raise ValueError("invalid idempotency key")
        try:
            data = self._store.read_bytes(layout.inflight(idempotency_key))
        except (ObjectNotFoundError, FileNotFoundError):
            return None
        try:
            marker = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            raise SessionLoadError(session_id, "corrupt_content", "marker is not JSON") from None
        if not isinstance(marker, dict):
            raise SessionLoadError(session_id, "corrupt_content", "marker is not an object")
        return marker

    def get_result(self, session_id: str, ref: ResultRef) -> dict:
        layout = self._layout(session_id)
        results_prefix = layout.tool_result("x").rsplit("/", 1)[0] + "/"
        if not ref.uri.startswith(results_prefix):
            raise SessionLoadError(session_id, "corrupt_content", "result is outside the session")
        try:
            data = self._store.read_bytes(ref.uri)
        except (ObjectNotFoundError, FileNotFoundError):
            raise SessionLoadError(session_id, "not_found", "no such tool result") from None
        if hashlib.sha256(data).hexdigest() != ref.sha256:
            raise SessionLoadError(session_id, "corrupt_content", "tool result hash mismatch")
        try:
            result = json.loads(data)
        except (ValueError, UnicodeDecodeError):
            raise SessionLoadError(
                session_id, "corrupt_content", "tool result is not JSON"
            ) from None
        if not isinstance(result, dict):
            raise SessionLoadError(session_id, "corrupt_content", "tool result is not an object")
        return result

    # --- listing ------------------------------------------------------------------------
    def list_for_owner(
        self, owner: str, *, limit: int, cursor: str | None = None
    ) -> Page[SessionSummary]:
        """Newest activity first. ``cursor`` is an opaque offset from a previous page."""
        prefix = AgentLayout.owner_index_prefix(self._root, owner)
        summaries: list[SessionSummary] = []
        for uri in self._store.list_uris(prefix):
            try:
                summary = SessionSummary.model_validate_json(self._store.read_bytes(uri))
            except (ObjectNotFoundError, FileNotFoundError, PydanticValidationError, ValueError):
                continue  # an unreadable index object hides one session, never the list
            if summary.owner == owner:
                summaries.append(summary)
        summaries.sort(key=lambda s: (s.last_activity_at, s.session_id), reverse=True)
        try:
            start = max(0, int(cursor)) if cursor else 0
        except ValueError:
            start = 0
        size = max(1, limit)
        items = summaries[start : start + size]
        more = start + size < len(summaries)
        return Page(items=items, next_cursor=str(start + size) if more else None)

    def list_all(self) -> list[SessionSummary]:
        """Every owner's index objects (job watcher discovery). Unreadable ones are skipped."""
        summaries: list[SessionSummary] = []
        for uri in self._store.list_uris(AgentLayout.owners_prefix(self._root)):
            try:
                summaries.append(SessionSummary.model_validate_json(self._store.read_bytes(uri)))
            except (ObjectNotFoundError, FileNotFoundError, PydanticValidationError, ValueError):
                continue
        return summaries

    # --- lease --------------------------------------------------------------------------
    def acquire_lease(self, session_id: str, holder: str, ttl: int) -> None:
        """Raise ``SessionInUse`` while an unexpired lease is held by another holder."""
        with self._lock(session_id):
            loaded = self.load(session_id)
            session = loaded.session
            now = self._clock()
            lease = session.lease
            if lease is not None and lease.holder != holder and lease.expires_at > now:
                raise SessionInUse(session_id)
            session.lease = Lease(holder=holder, expires_at=now + timedelta(seconds=ttl))
            self.save(loaded)

    def release_lease(self, session_id: str, holder: str) -> None:
        """Clear the lease if ``holder`` owns it. Anything else is a no-op."""
        with self._lock(session_id):
            loaded = self.load(session_id)
            lease = loaded.session.lease
            if lease is None or lease.holder != holder:
                return
            loaded.session.lease = None
            self.save(loaded)
