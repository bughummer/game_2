"""The built-in status poller: what actually drives a workflow forward.

``SageMakerOrchestrator.advance()`` submits each next stage, and ``StatusMonitor.poll()`` calls
it — but something has to call the monitor. Without EventBridge, that is this poller: a daemon
thread inside the MCP server process that polls every RUNNING execution on a fixed interval.

How it finds executions:
* ``register`` — the orchestrator notifies it the moment a workflow's first job is submitted.
* ``discover`` — a scan of every experiment's ``executions/`` prefix for RUNNING records, at
  startup (so a restart resumes in-flight work) and then periodically as a safety net.

Run exactly ONE poller per artifact root. ``advance()`` is not safe against two concurrent
callers on the same execution (its read-modify-write is unlocked), so a second MCP replica must
set ``ML_FACTORY_STATUS_POLL_INTERVAL_SECONDS=0``.
"""

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

from ml_engine.contracts.execution import ExecutionRecord, is_settled
from ml_engine.io import ExperimentLayout, ObjectStore, read_model_if_exists

LOGGER = logging.getLogger("ml_factory.tracking.poller")


class ExecutionPoller:
    def __init__(
        self,
        monitor: Any,
        store: ObjectStore,
        artifact_root: str,
        *,
        interval_seconds: float = 30,
        rediscover_seconds: float = 900,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._monitor = monitor
        self._store = store
        self._root = artifact_root
        self._interval = interval_seconds
        self._rediscover = rediscover_seconds
        self._clock = clock
        self._active: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # --- bookkeeping --------------------------------------------------------------------------
    def register(self, execution_id: str) -> None:
        """Start watching an execution. Safe to call from any thread."""
        with self._lock:
            self._active.add(execution_id)

    @property
    def active(self) -> set[str]:
        with self._lock:
            return set(self._active)

    def discover(self) -> int:
        """Scan the artifact root for RUNNING executions and watch them. Returns how many."""
        found: set[str] = set()
        experiments = ExperimentLayout.experiments_prefix(self._root)
        for experiment_prefix in self._store.list_prefixes(experiments):
            layout = ExperimentLayout(base=experiment_prefix.rstrip("/"))
            prefix = layout.executions_prefix
            try:
                uris = self._store.list_uris(prefix)
            except Exception:
                LOGGER.exception("Could not list executions under %s", prefix)
                continue
            for uri in uris:
                name = uri[len(prefix) :]
                # Execution records sit directly under executions/; observations and diagnoses
                # live in per-execution subfolders and are skipped.
                if "/" in name or not name.endswith(".json"):
                    continue
                try:
                    record = read_model_if_exists(self._store, uri, ExecutionRecord)
                except Exception:
                    LOGGER.exception("Unreadable execution record %s", uri)
                    continue
                if record is not None and not is_settled(record):
                    found.add(record.execution_id)
        with self._lock:
            self._active |= found
        LOGGER.info("Discovered %d running execution(s)", len(found))
        return len(found)

    # --- one sweep ----------------------------------------------------------------------------
    def tick(self) -> int:
        """Poll every watched execution once. Finished ones are dropped. Returns how many polled.

        One failing execution never stops the others: errors are logged and the execution is
        kept, so a transient AWS error is retried on the next tick.
        """
        polled = 0
        for execution_id in sorted(self.active):
            try:
                result = self._monitor.poll(execution_id)
            except Exception:
                LOGGER.exception("Polling %s failed; will retry", execution_id)
                continue
            polled += 1
            # A stopped execution stays watched until AWS confirms its jobs have ended.
            if result is None or is_settled(result):
                with self._lock:
                    self._active.discard(execution_id)
        return polled

    # --- background thread --------------------------------------------------------------------
    def start(self) -> threading.Thread:
        """Run discovery once, then poll on the interval in a daemon thread."""
        if self._thread is not None and self._thread.is_alive():
            return self._thread
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="mlf-status-poller", daemon=True)
        self._thread.start()
        LOGGER.info("Status poller started (every %ss)", self._interval)
        return self._thread

    def stop(self, timeout: float | None = 5) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)

    def _run(self) -> None:
        last_discovery = self._safe_discover()
        while not self._stop.wait(self._interval):
            if self._clock() - last_discovery >= self._rediscover:
                last_discovery = self._safe_discover()
            self.tick()

    def _safe_discover(self) -> float:
        try:
            self.discover()
        except Exception:
            LOGGER.exception("Execution discovery failed; will retry")
        return self._clock()
