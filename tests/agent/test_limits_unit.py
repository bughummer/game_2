"""Unit tests for the pure session-limit functions (design C7)."""

from __future__ import annotations

import pytest

from backend.agent.limits import (
    LimitName,
    accrue_active,
    check_before_model_call,
    check_before_tool_call,
    check_loop_top,
    raise_limit,
    register_call,
    validate_limits,
    would_exceed_costly,
)
from backend.errors import ValidationError
from ml_engine.contracts.agent import SessionCounters, SessionLimits, SessionStatus

RAW = {
    "max_converse_turns": 40,
    "max_tool_calls": 60,
    "max_costly_tool_calls": 10,
    "max_estimated_spend_usd": 25.5,
    "max_active_seconds": 7200,
    "max_repeated_identical_calls": 3,
}


def _limits(**overrides: object) -> SessionLimits:
    return validate_limits({**RAW, **overrides})


def test_validate_limits_accepts_valid_mapping() -> None:
    assert _limits().max_estimated_spend_usd == 25.5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_converse_turns", 0),
        ("max_tool_calls", -1),
        ("max_costly_tool_calls", 2.5),
        ("max_active_seconds", True),
        ("max_estimated_spend_usd", 0),
        ("max_estimated_spend_usd", float("nan")),
        ("max_repeated_identical_calls", None),
    ],
)
def test_validate_limits_names_invalid_limit(field: str, value: object) -> None:
    with pytest.raises(ValidationError) as caught:
        validate_limits({**RAW, field: value})
    assert caught.value.details["limit"] == field
    assert field in caught.value.message


def test_validate_limits_rejects_missing_limit() -> None:
    raw = dict(RAW)
    del raw["max_tool_calls"]
    with pytest.raises(ValidationError) as caught:
        validate_limits(raw)
    assert caught.value.details["limit"] == "max_tool_calls"


def test_loop_top_stops_on_costly_count() -> None:
    stop = check_loop_top(SessionCounters(costly_tool_calls=10), _limits())
    assert stop is not None
    assert (stop.name, stop.configured, stop.current) == (LimitName.COSTLY, 10, 10)
    assert check_loop_top(SessionCounters(costly_tool_calls=9), _limits()) is None


def test_model_and_tool_checks() -> None:
    limits = _limits()
    assert (
        check_before_model_call(SessionCounters(converse_turns=40), limits).name is LimitName.TURNS
    )
    assert check_before_model_call(SessionCounters(converse_turns=39), limits) is None
    assert (
        check_before_tool_call(SessionCounters(tool_calls=60), limits).name is LimitName.TOOL_CALLS
    )
    assert check_before_tool_call(SessionCounters(tool_calls=59), limits) is None


def test_would_exceed_costly() -> None:
    limits = _limits()
    assert (
        would_exceed_costly(SessionCounters(costly_tool_calls=10), limits, 1.0) is LimitName.COSTLY
    )
    assert (
        would_exceed_costly(SessionCounters(estimated_spend_usd=25.0), limits, 1.0)
        is LimitName.SPEND
    )
    assert would_exceed_costly(SessionCounters(estimated_spend_usd=25.0), limits, 0.5) is None
    assert would_exceed_costly(SessionCounters(), limits, None) is None


def test_register_call_counts_and_detects_repeats() -> None:
    limits = _limits()
    counters = SessionCounters()
    violations = []
    for _ in range(4):
        counters, violated = register_call(counters, "same", limits)
        violations.append(violated)
    assert violations == [False, False, False, True]
    assert counters.tool_calls == 4
    counters, violated = register_call(counters, "other", limits)
    assert (counters.consecutive_identical, violated, counters.tool_calls) == (1, False, 5)


def test_raise_limit_changes_only_that_limit() -> None:
    limits = _limits()
    raised = raise_limit(limits, LimitName.TURNS, 5)
    assert raised.max_converse_turns == 45
    assert raised.model_dump(exclude={"max_converse_turns"}) == limits.model_dump(
        exclude={"max_converse_turns"}
    )
    assert raise_limit(limits, LimitName.SPEND, 1.25).max_estimated_spend_usd == 26.75
    for bad in (0, -2, 1.5):
        with pytest.raises(ValidationError):
            raise_limit(limits, LimitName.TOOL_CALLS, bad)


def test_accrue_active_only_while_running() -> None:
    counters = SessionCounters(active_seconds=10.0)
    assert accrue_active(counters, SessionStatus.RUNNING, 2.5).active_seconds == 12.5
    for status in SessionStatus:
        if status is not SessionStatus.RUNNING:
            assert accrue_active(counters, status, 5.0).active_seconds == 10.0
    assert accrue_active(counters, SessionStatus.RUNNING, -1.0).active_seconds == 10.0
