"""Property tests for session intake (design C15, Requirements 1.x)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.agent.runner import InlineAgentRunner
from backend.agent.wiring import build_agent_service
from backend.config import Settings
from backend.errors import ValidationError
from ml_engine.contracts.agent import SessionStatus
from tests.support.agent_fakes import CountingStore, FakeClock
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.in_memory_store import InMemoryObjectStore

ROLE = "arn:aws:iam::123456789012:role/mlf-sagemaker-job"
T0 = datetime(2025, 1, 1, tzinfo=UTC)
ALICE = type("P", (), {"user": "alice"})()
LIMIT = 4_000

_WHITESPACE = st.sampled_from([" ", "\t", "\n", "\r", "\x0b", "\x0c", "\u00a0", "\u2003", "\u3000"])


def _service():
    objects = CountingStore(InMemoryObjectStore())
    runner = InlineAgentRunner()
    service = build_agent_service(
        Settings(
            artifact_root=ARTIFACT_ROOT,
            sagemaker_execution_role_arn=ROLE,
            agent_enabled=True,
        ),
        store=objects,
        runner=runner,
        clock=FakeClock(T0),
    )
    runner.bind(None)  # no loop: nothing calls a model
    return service, objects, runner


def _padded(core: str, pad: int) -> str:
    return core + "x" * pad


# Goals near and far from both boundaries: blank, whitespace-only, short text, and lengths
# around the 4,000-character limit.
goals = st.one_of(
    st.text(max_size=50),
    st.text(alphabet=_WHITESPACE, max_size=LIMIT + 5),
    st.builds(_padded, st.text(max_size=5), st.integers(LIMIT - 10, LIMIT + 10)),
    st.builds(
        lambda ws, n: ws * n,
        _WHITESPACE,
        st.integers(LIMIT - 2, LIMIT + 2),
    ),
)


# Feature: ml-assistant, Property 5: Goal validation
@settings(max_examples=100, deadline=None)
@given(goal=goals)
def test_goal_validation(goal: str) -> None:
    """**Validates: Requirements 1.1, 1.2**"""
    service, objects, runner = _service()
    valid = any(not c.isspace() for c in goal) and len(goal) <= LIMIT
    if valid:
        view = service.create_session(ALICE, goal)
        assert view.owner == "alice"
        assert view.status is SessionStatus.RUNNING
        assert runner.kicked == [view.session_id]
        assert service.store.load(view.session_id).session.goal == goal
    else:
        with pytest.raises(ValidationError, match="4,000"):
            service.create_session(ALICE, goal)
        assert objects.writes == 0
        assert runner.kicked == []
