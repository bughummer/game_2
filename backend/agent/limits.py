"""Session limits (design C7). Pure: no IO, no clock, no AWS.

Every function takes counters and limits and returns a decision or new objects; nothing is
mutated. Limits built from ``Settings`` come from ``backend.agent.service.limits_from_settings``;
``validate_limits`` validates an arbitrary mapping (for example a stored or user-supplied one).

Spend is a positive decimal in USD; every other limit is a positive integer (owner decision).
Active time accrues only while the session status is ``running`` (owner decision).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from numbers import Real

from backend.errors import ValidationError
from ml_engine.contracts.agent import SessionCounters, SessionLimits, SessionStatus

#: Error code returned to the model for a Costly call over a cap, in every mode (Req 9.8, 14.3).
CODE_LIMIT_REACHED = "LIMIT_REACHED"


class LimitName(StrEnum):
    TURNS = "max_converse_turns"
    TOOL_CALLS = "max_tool_calls"
    COSTLY = "max_costly_tool_calls"
    SPEND = "max_estimated_spend_usd"
    ACTIVE = "max_active_seconds"
    REPEATS = "max_repeated_identical_calls"


#: The counter each limit is compared with.
_COUNTER: dict[LimitName, str] = {
    LimitName.TURNS: "converse_turns",
    LimitName.TOOL_CALLS: "tool_calls",
    LimitName.COSTLY: "costly_tool_calls",
    LimitName.SPEND: "estimated_spend_usd",
    LimitName.ACTIVE: "active_seconds",
    LimitName.REPEATS: "consecutive_identical",
}


@dataclass(frozen=True, slots=True)
class LimitStop:
    """A reached limit: its name, configured value and current value (Req 9.2)."""

    name: LimitName
    configured: float
    current: float

    @property
    def message(self) -> str:
        return (
            f"Session limit {self.name.value} reached: configured {self.configured:g}, "
            f"current {self.current:g}."
        )


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_positive_decimal(value: object) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    )


def _invalid(name: LimitName, value: object, expected: str) -> ValidationError:
    return ValidationError(
        f"Session limit {name.value} must be {expected} (got {value!r}).",
        {"limit": name.value},
    )


def validate_limits(raw: Mapping[str, object]) -> SessionLimits:
    """Validate a mapping of limits; raise ``ValidationError`` naming the first invalid limit."""
    values: dict[str, object] = {}
    for name in LimitName:
        value = raw.get(name.value)
        if name is LimitName.SPEND:
            if not _is_positive_decimal(value):
                raise _invalid(name, value, "a positive decimal amount in USD")
            values[name.value] = float(value)  # type: ignore[arg-type]
        else:
            if not _is_positive_int(value):
                raise _invalid(name, value, "a positive integer")
            values[name.value] = value
    unknown = sorted(set(raw) - {n.value for n in LimitName} - {"schema_version"})
    if unknown:
        raise ValidationError(f"Unknown session limit {unknown[0]}.", {"limit": unknown[0]})
    return SessionLimits(**values)  # type: ignore[arg-type]


def _stop_if_reached(
    c: SessionCounters, limits: SessionLimits, name: LimitName
) -> LimitStop | None:
    configured = getattr(limits, name.value)
    current = getattr(c, _COUNTER[name])
    if current >= configured:
        return LimitStop(name=name, configured=configured, current=current)
    return None


def _first_stop(
    c: SessionCounters, limits: SessionLimits, names: tuple[LimitName, ...]
) -> LimitStop | None:
    for name in names:
        stop = _stop_if_reached(c, limits, name)
        if stop is not None:
            return stop
    return None


def check_loop_top(c: SessionCounters, limits: SessionLimits) -> LimitStop | None:
    """Stop at the top of each loop step: active time, Costly count (Req 9.9), spend.

    The Costly-count stop applies whether or not budget enforcement is on.
    """
    return _first_stop(c, limits, (LimitName.ACTIVE, LimitName.COSTLY, LimitName.SPEND))


def check_before_model_call(c: SessionCounters, limits: SessionLimits) -> LimitStop | None:
    """Stop before the next Converse request: turns, then active time (Req 9.2, 9.5)."""
    return _first_stop(c, limits, (LimitName.TURNS, LimitName.ACTIVE))


def check_before_tool_call(c: SessionCounters, limits: SessionLimits) -> LimitStop | None:
    """Stop before processing the next tool use, so ``tool_calls`` never passes the limit."""
    return _first_stop(c, limits, (LimitName.TOOL_CALLS,))


def projected_spend(current: float, increase: float) -> Decimal:
    """Add the decimal USD amounts represented by the session's JSON numbers."""
    return Decimal(str(current)) + Decimal(str(increase))


def would_exceed_costly(
    c: SessionCounters, limits: SessionLimits, estimate_usd: float | None
) -> LimitName | None:
    """The cap a Costly call would break, if any. Such a call is never executed (Req 9.8, 14.3).

    An unknown estimate (``None``) cannot be checked against spend, so it breaks the spend cap
    only when spend has already reached it; the policy handles the missing estimate itself.
    """
    if c.costly_tool_calls + 1 > limits.max_costly_tool_calls:
        return LimitName.COSTLY
    if estimate_usd is None:
        return LimitName.SPEND if c.estimated_spend_usd >= limits.max_estimated_spend_usd else None
    if projected_spend(c.estimated_spend_usd, max(estimate_usd, 0.0)) > Decimal(
        str(limits.max_estimated_spend_usd)
    ):
        return LimitName.SPEND
    return None


def register_call(
    c: SessionCounters, fingerprint: str, limits: SessionLimits
) -> tuple[SessionCounters, bool]:
    """Count one tool use and apply the repeated-call rule (Req 9.4).

    Every use that reaches processing adds 1 to ``tool_calls``, including a repeat violation.
    A different fingerprint resets the consecutive run to this call alone.
    """
    consecutive = c.consecutive_identical + 1 if fingerprint == c.last_fingerprint else 1
    updated = c.model_copy(
        update={
            "tool_calls": c.tool_calls + 1,
            "consecutive_identical": consecutive,
            "last_fingerprint": fingerprint,
        }
    )
    return updated, consecutive > limits.max_repeated_identical_calls


def raise_limit(limits: SessionLimits, name: LimitName, amount: float) -> SessionLimits:
    """Raise exactly one limit by a confirmed positive amount (Req 9.3).

    Spend takes a positive decimal; every other limit takes a positive integer.
    """
    name = LimitName(name)
    if name is LimitName.SPEND:
        if not _is_positive_decimal(amount):
            raise _invalid(name, amount, "raised by a positive decimal amount in USD")
        new_value: float = float(projected_spend(limits.max_estimated_spend_usd, amount))
    else:
        if not _is_positive_int(amount):
            raise _invalid(name, amount, "raised by a positive integer")
        new_value = getattr(limits, name.value) + amount
    return SessionLimits.model_validate({**limits.model_dump(), name.value: new_value})


def accrue_active(c: SessionCounters, status: SessionStatus, elapsed: float) -> SessionCounters:
    """Add ``elapsed`` seconds of active time, only while ``running`` (Req 9.5)."""
    if status is not SessionStatus.RUNNING or not math.isfinite(elapsed) or elapsed <= 0:
        return c
    return c.model_copy(update={"active_seconds": c.active_seconds + elapsed})
