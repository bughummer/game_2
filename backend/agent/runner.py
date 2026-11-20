"""Runners that drive ``AgentLoop.run`` (design C15).

``InlineAgentRunner`` runs the loop to its next pause on the caller's thread (tests, CLI
``--in-process``). ``ThreadAgentRunner`` runs it on a bounded thread pool and de-duplicates per
session id: a kick for a session that is already running is folded into one more run after the
current one finishes, so at most one worker drives a session at a time.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

LOGGER = logging.getLogger("ml_factory.agent.runner")

RunFn = Callable[[str], Any]


class AgentRunner(Protocol):
    def kick(self, session_id: str) -> None: ...


def _holder(kind: str) -> str:
    return f"{socket.gethostname()}:{os.getpid()}:{kind}:{uuid.uuid4().hex[:8]}"


def _as_run(loop: Any, holder: str) -> RunFn | None:
    if loop is None:
        return None
    if hasattr(loop, "run"):
        return lambda session_id: loop.run(session_id, holder)
    return loop  # a plain callable taking the session id


class InlineAgentRunner:
    """Runs the loop synchronously on ``kick``. A kick during a run of the same session (for
    example from inside a handler) is folded into one more run once the current one returns."""

    def __init__(self, loop: Any = None, *, holder: str | None = None) -> None:
        self._holder = holder or _holder("inline")
        self._run = _as_run(loop, self._holder)
        self._active: set[str] = set()
        self._again: set[str] = set()
        self.kicked: list[str] = []

    def bind(self, loop: Any) -> None:
        """Attach the loop after construction (the service and loop are built together)."""
        self._run = _as_run(loop, self._holder)

    def kick(self, session_id: str) -> None:
        self.kicked.append(session_id)
        if self._run is None:
            return
        if session_id in self._active:
            self._again.add(session_id)
            return
        self._active.add(session_id)
        try:
            while True:
                self._run(session_id)
                if session_id not in self._again:
                    break
                self._again.discard(session_id)
        finally:
            self._active.discard(session_id)
            self._again.discard(session_id)


class ThreadAgentRunner:
    """Runs the loop on a thread pool, at most one worker per session id."""

    def __init__(
        self, loop: Any = None, *, max_workers: int = 4, holder: str | None = None
    ) -> None:
        self._holder = holder or _holder("thread")
        self._run = _as_run(loop, self._holder)
        self._pool = ThreadPoolExecutor(
            max_workers=max(1, max_workers), thread_name_prefix="agent-loop"
        )
        self._lock = threading.Lock()
        self._active: set[str] = set()
        self._again: set[str] = set()

    def bind(self, loop: Any) -> None:
        self._run = _as_run(loop, self._holder)

    def kick(self, session_id: str) -> None:
        if self._run is None:
            raise RuntimeError("ThreadAgentRunner has no loop bound")
        with self._lock:
            if session_id in self._active:
                self._again.add(session_id)
                return
            self._active.add(session_id)
        self._pool.submit(self._work, session_id)

    def is_active(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._active

    def _work(self, session_id: str) -> None:
        while True:
            try:
                assert self._run is not None
                self._run(session_id)
            except Exception:
                LOGGER.exception("Agent loop for session %s failed", session_id)
            with self._lock:
                if session_id in self._again:
                    self._again.discard(session_id)
                    continue
                self._active.discard(session_id)
                return

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)
