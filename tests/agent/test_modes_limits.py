"""Offline tests for ``set_mode`` and ``continue_after_limit`` (task 8.5; Req 5.2-5.4, 5.9, 9.3).

No loop is bound to the runner, so a kick only records the session id.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest

from backend.agent.limits import LimitName
from backend.errors import AuthorizationError, ConflictError, FeatureDisabledError, ValidationError
from ml_engine.contracts.agent import Mode, SessionStatus, TranscriptEntry
from tests.agent.test_service import ALICE, BOB, T0, _service
from tests.support.agent_fakes import FakeClock


def _session(**overrides):
    service, objects, _, runner = _service(**overrides)
    runner.bind(None)
    sid = service.create_session(ALICE, "predict churn").session_id
    return service, objects, runner, sid


def _stop_at(service, sid, limit: str = LimitName.TURNS.value):
    """Put the session where the loop leaves it after a limit stop."""
    with service.store.transaction(sid) as s:
        s.transcript.append(
            TranscriptEntry(
                seq=s.next_seq, at=T0, kind="limit_stop", data={"limit": limit}
            )
        )
        s.next_seq += 1
        s.status = SessionStatus.WAITING_FOR_USER


# ------------------------------------------------------------------ set_mode


def test_autopilot_accepted_records_grant_and_audit():
    service, _, _, sid = _session()
    service._clock = FakeClock(  # a non-UTC clock with sub-second part
        datetime(2025, 1, 1, 3, 0, 0, 750_000, tzinfo=timezone(timedelta(hours=2)))
    )
    before = service.store.load(sid).session
    view = service.set_mode(ALICE, sid, Mode.AUTOPILOT, max_estimated_spend_usd=2.5)
    s = service.store.load(sid).session
    assert view.mode is Mode.AUTOPILOT and s.mode is Mode.AUTOPILOT
    grant = s.autopilot
    assert grant is not None and grant.enabled_by == "alice"
    assert grant.max_estimated_spend_usd == 2.5
    assert grant.enabled_at == datetime(2025, 1, 1, 1, 0, 0, tzinfo=UTC)
    assert grant.enabled_at.utcoffset() == timedelta(0) and grant.enabled_at.microsecond == 0
    assert s.audit_seq == before.audit_seq + 1
    assert s.transcript[-1].kind == "mode" and s.transcript[-1].data["mode"] == "autopilot"


@pytest.mark.parametrize("spend", [None, 0, -1, 10.5, float("nan")])
def test_autopilot_bad_spend_refused_and_nothing_changes(spend):
    service, _, _, sid = _session()
    before = service.store.load(sid).session
    with pytest.raises(ValidationError, match="maximum estimated spend"):
        service.set_mode(ALICE, sid, "autopilot", max_estimated_spend_usd=spend)
    assert service.store.load(sid).session == before


def test_autopilot_refused_when_not_allowed():
    service, _, _, sid = _session(agent_autopilot_allowed=False)
    before = service.store.load(sid).session
    with pytest.raises(FeatureDisabledError):
        service.set_mode(ALICE, sid, Mode.AUTOPILOT, max_estimated_spend_usd=1.0)
    assert service.store.load(sid).session == before


def test_switch_back_to_supervised_clears_grant():
    service, _, _, sid = _session()
    service.set_mode(ALICE, sid, Mode.AUTOPILOT, max_estimated_spend_usd=1.0)
    view = service.set_mode(ALICE, sid, Mode.SUPERVISED)
    s = service.store.load(sid).session
    assert view.mode is Mode.SUPERVISED and s.mode is Mode.SUPERVISED and s.autopilot is None
    assert [e.data["mode"] for e in s.transcript if e.kind == "mode"] == [
        "autopilot",
        "supervised",
    ]


def test_set_mode_non_owner_refused():
    service, _, _, sid = _session()
    before = service.store.load(sid).session
    with pytest.raises(AuthorizationError):
        service.set_mode(BOB, sid, Mode.AUTOPILOT, max_estimated_spend_usd=1.0)
    assert service.store.load(sid).session == before


# ------------------------------------------------------------------ continue_after_limit


def test_raise_changes_only_the_named_limit_and_resumes():
    service, _, runner, sid = _session()
    _stop_at(service, sid, LimitName.TURNS.value)
    before = service.store.load(sid).session
    kicks = len(runner.kicked)
    view = service.continue_after_limit(ALICE, sid, LimitName.TURNS, 5)
    s = service.store.load(sid).session
    expected = before.limits.model_dump()
    expected["max_converse_turns"] += 5
    assert s.limits.model_dump() == expected
    assert view.status is SessionStatus.RUNNING
    assert len(runner.kicked) == kicks + 1 and runner.kicked[-1] == sid
    assert s.audit_seq == before.audit_seq + 1


@pytest.mark.parametrize("amount", [0, -3, None, 1.5, True])
def test_bad_amount_leaves_everything_unchanged(amount):
    service, _, runner, sid = _session()
    _stop_at(service, sid)
    before = service.store.load(sid).session
    kicks = len(runner.kicked)
    with pytest.raises(ValidationError):
        service.continue_after_limit(ALICE, sid, LimitName.TURNS, amount)
    after = service.store.load(sid).session
    assert after == before and after.status is SessionStatus.WAITING_FOR_USER
    assert len(runner.kicked) == kicks


def test_only_the_reached_limit_can_be_raised():
    service, _, _, sid = _session()
    _stop_at(service, sid, LimitName.TURNS.value)
    before = service.store.load(sid).session
    with pytest.raises(ValidationError, match="max_converse_turns"):
        service.continue_after_limit(ALICE, sid, LimitName.TOOL_CALLS, 5)
    assert service.store.load(sid).session == before


def test_continue_needs_a_limit_stop():
    service, _, _, sid = _session()
    with service.store.transaction(sid) as s:
        s.status = SessionStatus.WAITING_FOR_USER  # waiting on a question, not a limit
    before = service.store.load(sid).session
    with pytest.raises(ConflictError) as err:
        service.continue_after_limit(ALICE, sid, LimitName.TURNS, 5)
    assert err.value.details["reason"] == "not_at_limit"
    assert service.store.load(sid).session == before


def test_continue_non_owner_refused():
    service, _, _, sid = _session()
    _stop_at(service, sid)
    before = service.store.load(sid).session
    with pytest.raises(AuthorizationError):
        service.continue_after_limit(BOB, sid, LimitName.TURNS, 5)
    assert service.store.load(sid).session == before
