"""Registry and input-validation properties for the ML Assistant (design C5)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel

from backend.agent import catalogue
from backend.agent.tools import (
    RESERVED_PRIVILEGED_PREFIXES,
    ToolOutput,
    ToolRegistrationError,
    ToolRegistry,
)
from ml_engine.contracts.agent import ToolClass
from tests.agent.strategies import SETTINGS


class _NoArgs(BaseModel):
    pass


def _handler(ctx, args) -> ToolOutput:
    return ToolOutput(data={})


def _snapshot(registry: ToolRegistry) -> dict:
    return {name: registry.get(name) for name in registry.names()}


# --- Property 8 --------------------------------------------------------------------------------

_ident = st.from_regex(r"[a-z][a-z0-9_]{0,15}", fullmatch=True)
# A class is either the enum member or its string value; both denote the same class.
_tool_class = st.sampled_from(list(ToolClass)).flatmap(lambda c: st.sampled_from([c, c.value]))
_class_collections = st.lists(_tool_class, max_size=4)
_reserved_names = st.builds(
    lambda prefix, suffix: prefix + suffix, st.sampled_from(RESERVED_PRIVILEGED_PREFIXES), _ident | st.just("")
)


@st.composite
def _registration_cases(draw: st.DrawFn):
    existing = draw(
        st.lists(_ident.filter(lambda n: not n.startswith(RESERVED_PRIVILEGED_PREFIXES)), max_size=4, unique=True)
    )
    choices = [_ident, _reserved_names]
    if existing:
        choices.append(st.sampled_from(existing))
    name = draw(st.one_of(*choices))
    classes = draw(_class_collections)
    return existing, name, classes


# Feature: ml-assistant, Property 8: Registration classifies every tool exactly once
@settings(**SETTINGS)
@given(case=_registration_cases())
def test_registration_classifies_every_tool_exactly_once(case):
    """**Validates: Requirements 3.2, 3.3, 3.7**"""
    existing, name, classes = case
    registry = ToolRegistry()
    for i, prior in enumerate(existing):
        registry.register(
            prior,
            classes=[list(ToolClass)[i % len(ToolClass)]],
            description="prior",
            input_model=_NoArgs,
            handler=_handler,
        )
    before = _snapshot(registry)

    distinct = {ToolClass(c) for c in classes}
    should_succeed = (
        len(distinct) == 1
        and name not in before
        and not name.startswith(RESERVED_PRIVILEGED_PREFIXES)
    )

    if should_succeed:
        registry.register(
            name, classes=classes, description="d", input_model=_NoArgs, handler=_handler
        )
        spec = registry.get(name)
        assert spec is not None
        assert spec.tool_class is next(iter(distinct))
        assert isinstance(spec.tool_class, ToolClass)
        assert registry.names() == (*before, name)
        assert all(registry.get(n) is s for n, s in before.items())
    else:
        with pytest.raises(ToolRegistrationError) as excinfo:
            registry.register(
                name, classes=classes, description="d", input_model=_NoArgs, handler=_handler
            )
        assert excinfo.value.tool == name
        assert repr(name) in str(excinfo.value)
        assert _snapshot(registry) == before
        assert registry.names() == tuple(before)


def test_shipped_catalogue_classifies_each_tool_once_with_no_reserved_names():
    """**Validates: Requirements 3.2, 3.7**"""
    registry = catalogue.build_registry(SimpleNamespace())
    names = registry.names()
    assert names, "the shipped catalogue registers at least one tool"
    assert len(set(names)) == len(names)
    for name in names:
        spec = registry.get(name)
        assert isinstance(spec.tool_class, ToolClass), name
        assert not name.startswith(RESERVED_PRIVILEGED_PREFIXES), name


# --- Property 9 --------------------------------------------------------------------------------
# Driven through the real AgentLoop, store and executor; only the model and the tools are stubs.

from datetime import UTC, datetime  # noqa: E402

from pydantic import ConfigDict, Field  # noqa: E402

from backend.agent.execute import ToolExecutor  # noqa: E402
from backend.agent.inputs import ToolInputs  # noqa: E402
from backend.agent.loop import AgentLoop  # noqa: E402
from backend.agent.store import SessionStore  # noqa: E402
from backend.agent.tools import ToolFlags  # noqa: E402
from backend.governance import GovernancePolicy  # noqa: E402
from ml_engine.contracts.agent import (  # noqa: E402
    CallState,
    Session,
    SessionLimits,
    SessionStatus,
)
from tests.support.agent_fakes import FakeClock  # noqa: E402
from tests.support.in_memory_store import InMemoryObjectStore  # noqa: E402
from tests.support.scripted_converse import (  # noqa: E402
    ScriptedFakeConverseClient,
    ScriptStep,
    tool_turn,
)

_P9_SID = "sess-p9"
_P9_T0 = datetime(2025, 1, 1, tzinfo=UTC)
_P9_CALL = "tu-probe"


class _PlanArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    steps: list[str] = Field(min_length=1, max_length=10)


class _ProbeArgs(BaseModel):
    """Strict, flat input model: every failure is a missing, unexpected or wrongly typed field."""

    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    count: int
    flag: bool = False


_REQUIRED = ("name", "count")
_FIELDS = ("name", "count", "flag")
_VALID = {"name": st.text(max_size=8), "count": st.integers(-5, 5), "flag": st.booleans()}
_WRONG = {
    "name": st.one_of(st.none(), st.integers(-5, 5), st.booleans(), st.lists(st.integers(), max_size=2)),
    "count": st.one_of(st.none(), st.text(max_size=4), st.booleans(), st.lists(st.integers(), max_size=2)),
    "flag": st.one_of(st.none(), st.integers(-5, 5), st.text(max_size=4), st.lists(st.integers(), max_size=2)),
}
_json_value = st.one_of(st.none(), st.booleans(), st.integers(-5, 5), st.text(max_size=4))
_extra_key = st.text(alphabet="abcxyz019_.- ", max_size=8).filter(lambda k: k not in _FIELDS)


class _P9Audit:
    def __init__(self) -> None:
        self.records: list = []

    def write(self, session, rec):
        self.records.append(rec)
        session.audit_seq += 1
        return f"a-{session.audit_seq}"


class _P9Authorizer:
    def require_owner(self, principal, experiment_id):  # pragma: no cover - not reached
        return None


def _p9_session() -> Session:
    return Session(
        session_id=_P9_SID,
        owner="alice",
        created_at=_P9_T0,
        updated_at=_P9_T0,
        last_activity_at=_P9_T0,
        status=SessionStatus.RUNNING,
        limits=SessionLimits(
            max_converse_turns=10,
            max_tool_calls=20,
            max_costly_tool_calls=2,
            max_estimated_spend_usd=5.0,
            max_active_seconds=600,
            max_repeated_identical_calls=3,
        ),
        goal="predict churn",
    )


class _P9Harness:
    """A loop with a plan tool, a Read probe and a Mutating probe, each counting its calls."""

    def __init__(self, name: str, arguments: dict) -> None:
        self.calls: list[str] = []
        self.registry = ToolRegistry()
        self.registry.register(
            "submit_plan",
            classes=[ToolClass.READ],
            description="plan",
            input_model=_PlanArgs,
            handler=self._handler("submit_plan", lambda a: {"steps": a.steps}),
            flags=ToolFlags(control=True, ledger_source=False),
        )
        self.registry.register(
            "probe_read",
            classes=[ToolClass.READ],
            description="read probe",
            input_model=_ProbeArgs,
            handler=self._handler("probe_read", lambda a: {"ok": True}),
        )
        self.registry.register(
            "probe_mutate",
            classes=[ToolClass.MUTATING],
            description="mutating probe",
            input_model=_ProbeArgs,
            handler=self._handler("probe_mutate", lambda a: {"ok": True}),
            # A config change, so a valid call waits for approval in Supervised (gate rule 6).
            flags=ToolFlags(changes_config=True),
        )
        self.clock = FakeClock(_P9_T0)
        self.store = SessionStore(InMemoryObjectStore(), "s3://bucket-test/root", clock=self.clock)
        self.store.create(_p9_session())
        self.audit = _P9Audit()
        inputs = ToolInputs(audit=self.audit, authorizer=_P9Authorizer(), allowed_instance_types=[])
        self.converse = ScriptedFakeConverseClient(
            steps=[
                ScriptStep(tool_turn(("tu-plan", "submit_plan", {"steps": ["look at data"]}))),
                ScriptStep(tool_turn((_P9_CALL, name, arguments))),
            ]
        )
        self.loop = AgentLoop(
            store=self.store,
            converse=self.converse,
            registry=self.registry,
            executor=ToolExecutor(inputs=inputs, services=SimpleNamespace(), clock=self.clock),
            inputs=inputs,
            audit=self.audit,
            governance=GovernancePolicy(
                send_sample_values=False, sample_values_per_column=0, sample_value_max_chars=0
            ),
            clock=self.clock,
        )

    def _handler(self, name, fn):
        def handler(ctx, args):
            self.calls.append(name)
            return ToolOutput(data=fn(args))

        return handler

    @property
    def session(self) -> Session:
        return self.store.load(_P9_SID).session


@st.composite
def _p9_cases(draw: st.DrawFn):
    """A tool name and arguments, plus the error the loop must return (``None`` when valid)."""
    registered = ("submit_plan", "probe_read", "probe_mutate")
    name = draw(
        st.one_of(
            st.sampled_from(["probe_read", "probe_mutate"]),
            _ident.filter(
                lambda n: n not in registered and not n.startswith(RESERVED_PRIVILEGED_PREFIXES)
            ),
        )
    )
    arguments: dict = {}
    bad: set[str] = set()
    for field_name in _FIELDS:
        choice = draw(st.sampled_from(["absent", "valid", "wrong"]))
        if choice == "valid":
            arguments[field_name] = draw(_VALID[field_name])
        elif choice == "wrong":
            arguments[field_name] = draw(_WRONG[field_name])
            bad.add(field_name)
        elif field_name in _REQUIRED:
            bad.add(field_name)
    extras = draw(st.dictionaries(_extra_key, _json_value, max_size=3))
    arguments.update(extras)
    bad |= set(extras)
    if name not in registered:
        return name, arguments, ("UNKNOWN_TOOL", None)
    if bad:
        return name, arguments, ("INVALID_ARGUMENTS", sorted(bad))
    return name, arguments, None


# Fields the loop may change when it records one error result for the pending call.
_RESULT_RECORD_FIELDS = {
    "state", "error_code", "result_ref", "truncated", "injection_suspected", "result_summary",
    "finished_at",
}
_RESULT_SESSION_FIELDS = {"revision", "updated_at", "last_activity_at", "next_seq"}


def _without_result(before: dict, after: dict) -> tuple[dict, dict]:
    """Strip exactly the changes that recording one error result is allowed to make."""
    before, after = dict(before), dict(after)
    for key in _RESULT_SESSION_FIELDS:
        before.pop(key, None)
        after.pop(key, None)
    assert after["counters"]["tool_calls"] == before["counters"]["tool_calls"] + 1
    # Counting a use also feeds the repeated-call rule (Req 9.4).
    counted = {"tool_calls": None, "consecutive_identical": None, "last_fingerprint": None}
    before["counters"] = {**before["counters"], **counted}
    after["counters"] = {**after["counters"], **counted}
    assert before["pending_tool_use_ids"] == [_P9_CALL] and after["pending_tool_use_ids"] == []
    before.pop("pending_tool_use_ids")
    after.pop("pending_tool_use_ids")
    assert after["transcript"][: len(before["transcript"])] == before["transcript"]
    added = after["transcript"][len(before["transcript"]) :]
    assert [(e["kind"], e["tool_call_id"]) for e in added] == [("tool_result", _P9_CALL)]
    before.pop("transcript")
    after.pop("transcript")
    strip = lambda r: {k: v for k, v in r.items() if k not in _RESULT_RECORD_FIELDS}  # noqa: E731
    before["tool_calls"] = [strip(r) for r in before["tool_calls"]]
    after["tool_calls"] = [strip(r) for r in after["tool_calls"]]
    return before, after


# Feature: ml-assistant, Property 9: Calls that cannot validate never execute
@settings(max_examples=100, deadline=None)
@given(case=_p9_cases())
def test_calls_that_cannot_validate_never_execute(case):
    """**Validates: Requirements 2.3, 2.4, 2.5, 3.10**"""
    name, arguments, expected = case
    h = _P9Harness(name, arguments)

    # Plan turn, plan result, then the turn that requests the probe call.
    for _ in range(3):
        h.loop.step(_P9_SID)
    assert h.session.plan is not None
    assert h.calls == ["submit_plan"]
    before = h.session.model_dump(mode="json")
    assert before["pending_tool_use_ids"] == [_P9_CALL]

    h.loop.step(_P9_SID)
    after_session = h.session
    record = next(r for r in after_session.tool_calls if r.tool_call_id == _P9_CALL)

    if expected is None:
        # Registered and valid: Read runs exactly once; Mutating waits for approval, unrun.
        if name == "probe_read":
            assert h.calls == ["submit_plan", "probe_read"]
            assert record.state is CallState.SUCCEEDED
        else:
            assert h.calls == ["submit_plan"]
            assert record.state is CallState.AWAITING_APPROVAL
        return

    code, fields = expected
    assert h.calls == ["submit_plan"], "a handler ran for a call that cannot validate"
    assert record.state is CallState.FAILED
    assert record.error_code == code
    error = h.store.get_result(_P9_SID, record.result_ref)["error"]
    assert error["code"] == code
    entry = next(
        e for e in after_session.transcript
        if e.kind == "tool_result" and e.tool_call_id == _P9_CALL
    )
    assert entry.data["error_code"] == code
    if code == "UNKNOWN_TOOL":
        assert error["details"]["tool"] == name
        assert repr(name) in error["message"]
        assert sorted(error["details"]["registered_tools"]) == sorted(h.registry.names())
    else:
        assert error["details"]["fields"] == fields

    stripped_before, stripped_after = _without_result(before, after_session.model_dump(mode="json"))
    assert stripped_after == stripped_before
    assert after_session.status is SessionStatus.RUNNING
    assert [r for r in h.audit.records if r.event_type == "input_rejected"] == []
