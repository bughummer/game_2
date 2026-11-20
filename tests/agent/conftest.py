"""Agent suite fixtures: offline guarantee and a service factory."""

from __future__ import annotations

import socket
from collections.abc import Callable
from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _block_sockets(monkeypatch):
    """Any attempt to open a network connection fails the test (Req 16.1)."""

    def refuse(*args: Any, **kwargs: Any):
        raise RuntimeError("network access is blocked in the agent test suite")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture
def agent_service_factory() -> Callable[..., Any]:
    """Build an ``AgentService`` through ``backend.agent.wiring.build_agent_service``.

    Tolerant of build order: while the wiring module does not exist, tests that use this
    fixture are skipped instead of erroring at collection.
    """
    try:
        from backend.agent.wiring import build_agent_service
    except ImportError:
        pytest.skip("backend.agent.wiring is not available yet")

    def factory(**kwargs: Any) -> Any:
        return build_agent_service(**kwargs)

    return factory
