"""Small test doubles for the agent suite: store wrappers, a fake clock and a runner stub.

``InlineAgentRunner`` is a stub used only until ``backend.agent.runner`` exists; once it does,
the real class is re-exported so tests always get the production behaviour.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


class _Delegating:
    """Forwards every attribute not overridden to the wrapped store."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class CountingStore(_Delegating):
    """Counts calls per method, so tests can assert "no write happened"."""

    def __init__(self, inner: Any) -> None:
        super().__init__(inner)
        self.counts: Counter[str] = Counter()
        self._lock = threading.Lock()

    def _count(self, name: str) -> None:
        with self._lock:
            self.counts[name] += 1

    def read_bytes(self, uri: str) -> bytes:
        self._count("read_bytes")
        return self._inner.read_bytes(uri)

    def write_bytes(self, uri: str, data: bytes) -> None:
        self._count("write_bytes")
        self._inner.write_bytes(uri, data)

    def delete(self, uri: str) -> None:
        self._count("delete")
        self._inner.delete(uri)

    def list_uris(self, prefix: str) -> list[str]:
        self._count("list_uris")
        return self._inner.list_uris(prefix)

    def stat(self, uri: str) -> Any:
        self._count("stat")
        return self._inner.stat(uri)

    @property
    def writes(self) -> int:
        return self.counts["write_bytes"]


class FlakyStore(CountingStore):
    """Fails the next ``fail_writes`` writes (optionally only where ``match(uri)`` is true)."""

    def __init__(
        self,
        inner: Any,
        *,
        fail_writes: int = 0,
        match: Callable[[str], bool] | None = None,
        error: Callable[[], BaseException] = lambda: OSError("injected write failure"),
    ) -> None:
        super().__init__(inner)
        self.fail_writes = fail_writes
        self.match = match
        self.error = error

    def write_bytes(self, uri: str, data: bytes) -> None:
        self._count("write_bytes")
        if self.fail_writes > 0 and (self.match is None or self.match(uri)):
            self.fail_writes -= 1
            self._count("failed_writes")
            raise self.error()
        self._inner.write_bytes(uri, data)

    def upload(self, source: Path, uri: str) -> None:
        self.write_bytes(uri, Path(source).read_bytes())


class FakeClock:
    """A controllable clock. ``sleep`` advances time and records the requested delays."""

    def __init__(self, start: datetime | None = None) -> None:
        self._now = start or datetime(2025, 1, 1, tzinfo=UTC)
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return self._now

    def __call__(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._now.timestamp()

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)


class _StubInlineAgentRunner:
    """Runs the loop synchronously on ``kick``. Replaced by the real runner once it exists."""

    def __init__(self, run: Callable[[str], Any] | None = None) -> None:
        self._run = run
        self.kicked: list[str] = []

    def bind(self, run: Callable[[str], Any]) -> None:
        self._run = run

    def kick(self, session_id: str) -> None:
        self.kicked.append(session_id)
        if self._run is not None:
            self._run(session_id)


try:  # pragma: no cover - depends on build order
    from backend.agent.runner import InlineAgentRunner  # type: ignore[attr-defined]
except ImportError:
    InlineAgentRunner = _StubInlineAgentRunner  # type: ignore[misc,assignment]
