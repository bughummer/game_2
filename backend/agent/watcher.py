"""The agent job watcher (design C15, D13; Req 6.5).

A daemon thread in the server process, built like ``backend.tracking.ExecutionPoller``. It
records the state of the jobs that sessions wait on and sets ``jobs_ready`` once they have all
ended. It never calls the model and never resumes a session: the user's surface does that
(D13). Nothing raises into its thread loop; a failed sweep is logged with the session id only
and retried on the next tick.

How it finds sessions:
* ``register`` - the loop tells it the moment a session starts waiting for a job.
* ``discover`` - a scan of every owner index for ``waiting_for_job`` sessions, at startup (so a
  restart keeps watching) and then periodically as a safety net.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from backend.agent.jobs import apply_observations, observe_jobs
from backend.agent.store import SessionStore
from ml_engine.contracts.agent import SessionStatus

LOGGER = logging.getLogger("ml_factory.agent.watcher")

_FINISHED = frozenset({SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED})


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentJobWatcher:
    def __init__(
        self,
        store: SessionStore,
        analysis: Any,
        *,
        interval_seconds: float = 15,
        rediscover_seconds: float = 60,
        clock: Callable[[], datetime] = _utcnow,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store = store
        self._analysis = analysis
        self._interval = interval_seconds
        self._rediscover = rediscover_seconds
        self._clock = clock
        self._monotonic = monotonic
        self._active: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- bookkeeping --------------------------------------------------------------------------
    def register(self, session_id: str) -> None:
        """Start watching a session. Safe to call from any thread."""
        with self._lock:
            self._active.add(session_id)

    @property
    def active(self) -> set[str]:
        with self._lock:
            return set(self._active)

    def discover(self) -> int:
        """Watch every session that waits for a job, found via the owner indexes."""
        found = {
            s.session_id
            for s in self._store.list_all()
            if s.status is SessionStatus.WAITING_FOR_JOB
        }
        with self._lock:
            self._active |= found
        LOGGER.info("Discovered %d session(s) waiting for a job", len(found))
        return len(found)

    # --- one sweep ----------------------------------------------------------------------------
    def sweep_jobs(self) -> int:
        """Refresh the awaited jobs of every watched session once. Returns how many became ready.

        Job states are read outside the session transaction; only a change is saved. A session
        that finished, or no longer has a running job, is dropped.
        """
        ready = 0
        for session_id in sorted(self.active):
            try:
                if self._sweep_one(session_id):
                    ready += 1
            except Exception:
                LOGGER.warning("Job sweep for session %s failed; will retry", session_id)
        return ready

    def _sweep_one(self, session_id: str) -> bool:
        session = self._store.load(session_id).session
        if session.status in _FINISHED or not _running_jobs(session):
            self._drop(session_id)
            return False
        if session.status is not SessionStatus.WAITING_FOR_JOB:
            return False  # the loop is driving it; keep watching, never write under the loop
        observed = observe_jobs(session, self._analysis)
        if not observed:
            return False
        with self._store.transaction(session_id) as s:
            if s.status is not SessionStatus.WAITING_FOR_JOB:
                return False
            was_ready = s.jobs_ready
            apply_observations(s, observed, self._clock())
            became_ready = s.jobs_ready and not was_ready
            still_running = _running_jobs(s)
        if not still_running:
            self._drop(session_id)
        return became_ready

    def _drop(self, session_id: str) -> None:
        with self._lock:
            self._active.discard(session_id)

    # --- background thread --------------------------------------------------------------------
    def start(self) -> threading.Thread:
        """Run discovery once, then sweep on the interval in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mlf-agent-watcher", daemon=True)
        self._thread.start()
        LOGGER.info("Agent job watcher started (every %ss)", self._interval)
        return self._thread

    def stop(self, timeout: float | None = 5) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        last_discovery = self._safe_discover()
        while not self._stop.wait(self._interval):
            try:
                if self._monotonic() - last_discovery >= self._rediscover:
                    last_discovery = self._safe_discover()
                self.sweep_jobs()
            except Exception:  # never raises into the thread loop
                LOGGER.warning("Agent job watcher tick failed; will retry")

    def _safe_discover(self) -> float:
        try:
            self.discover()
        except Exception:
            LOGGER.warning("Agent session discovery failed; will retry")
        return self._monotonic()


def _running_jobs(session: Any) -> bool:
    return any(j.state in ("running", "unknown") for j in session.awaited_jobs)
