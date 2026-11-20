"""Contract and session-store tests for the ML Assistant (ml-assistant design C10)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from backend.agent.store import SessionLoadError, SessionStore
from backend.agent.view import build_view
from ml_engine.contracts.agent import (
    SUPPORTED_VERSIONS,
    AnswerSentence,
    ApprovalRequest,
    ApprovalStatus,
    Attribution,
    AutopilotGrant,
    AwaitedJob,
    CallState,
    ClientAction,
    ColumnRef,
    CompactionInfo,
    DatasetRef,
    FinalAnswer,
    GuardRefusal,
    Lease,
    LocalFileRef,
    Mode,
    Plan,
    PlanStep,
    ResultRef,
    Session,
    SessionCounters,
    SessionFailure,
    SessionLimits,
    SessionStatus,
    ToolCallRecord,
    ToolClass,
    TranscriptEntry,
    VerbatimBlock,
)
from ml_engine.io.layout import AgentLayout
from tests.support.in_memory_store import InMemoryObjectStore

ROOT = "s3://bucket-test/root"
SESSION_ID = "sess-1"
_T0 = datetime(2024, 1, 1, tzinfo=UTC)


def _session(session_id: str = SESSION_ID) -> Session:
    return Session(
        session_id=session_id,
        owner="alice",
        created_at=_T0,
        updated_at=_T0,
        last_activity_at=_T0,
        status=SessionStatus.WAITING_FOR_USER,
        limits=SessionLimits(
            max_converse_turns=10,
            max_tool_calls=10,
            max_costly_tool_calls=2,
            max_estimated_spend_usd=5.0,
            max_active_seconds=600,
            max_repeated_identical_calls=3,
        ),
        goal="predict churn",
        tool_calls=[
            ToolCallRecord(
                tool_call_id="tc-1",
                turn=1,
                tool="list_datasets",
                arguments={},
                args_hash="h",
                tool_class=ToolClass.READ,
                state=CallState.SUCCEEDED,
            )
        ],
        approvals=[
            ApprovalRequest(
                approval_id="ap-1",
                kind="tool",
                tool_call_id="tc-1",
                tool="train",
                arguments={},
                args_hash="h",
                rationale="r",
                estimate_usd=None,
                estimate_basis=None,
                status=ApprovalStatus.PENDING,
                created_at=_T0,
                expires_at=_T0 + timedelta(seconds=900),
            )
        ],
    )


def _new_store() -> tuple[InMemoryObjectStore, SessionStore, str]:
    objects = InMemoryObjectStore()
    store = SessionStore(objects, ROOT)
    store.create(_session())
    return objects, store, AgentLayout.for_session(ROOT, SESSION_ID).session


def _snapshot(objects: InMemoryObjectStore) -> dict[str, bytes]:
    return {uri: objects.read_bytes(uri) for uri in objects.list_uris(ROOT)}


def _valid_payload() -> dict[str, Any]:
    return _session().model_dump(mode="json", by_alias=True)


# --- generators ---------------------------------------------------------------------------

# Where each embedded contract sits inside a stored Session record.
_CONTRACT_PATHS: dict[str, tuple[Any, ...]] = {
    "Session": (),
    "SessionLimits": ("limits",),
    "ToolCallRecord": ("tool_calls", 0),
    "ApprovalRequest": ("approvals", 0),
}

contract_names = st.sampled_from(sorted(_CONTRACT_PATHS))
_version_text = st.text(
    alphabet=st.characters(min_codepoint=0x21, max_codepoint=0x7E, blacklist_characters="'\"\\"),
    min_size=1,
    max_size=20,
)


@st.composite
def unsupported_versions(draw: st.DrawFn, contract: str) -> str:
    supported = SUPPORTED_VERSIONS[contract]
    return draw(_version_text.filter(lambda v: v not in supported))


def _drop_required(payload: dict[str, Any], key: str) -> dict[str, Any]:
    payload.pop(key, None)
    return payload


_required_keys = st.sampled_from(
    ["session_id", "owner", "created_at", "status", "limits", "goal", "updated_at"]
)
# Invalid for every field below (an integer is not: it parses as a Unix timestamp).
_bad_values = st.one_of(st.none(), st.lists(st.integers(), max_size=2), st.just("x"))

corrupt_records = st.one_of(
    # Arbitrary bytes (almost never JSON).
    st.binary(max_size=200),
    # Truncated valid record.
    st.integers(1, 200).map(lambda n: json.dumps(_valid_payload()).encode("utf-8")[:n]),
    # Valid JSON that is not an object.
    st.one_of(st.integers(), st.lists(st.integers(), max_size=3), st.text(max_size=10)).map(
        lambda v: json.dumps(v).encode("utf-8")
    ),
    # Supported version, but a required field is missing.
    _required_keys.map(lambda k: json.dumps(_drop_required(_valid_payload(), k)).encode("utf-8")),
    # Supported version, but a field has the wrong type.
    st.tuples(st.sampled_from(["status", "limits", "created_at", "revision"]), _bad_values).map(
        lambda kv: json.dumps({**_valid_payload(), kv[0]: kv[1]}).encode("utf-8")
    ),
    # Supported version, but an unknown field (contracts forbid extras).
    st.just(json.dumps({**_valid_payload(), "unexpected": 1}).encode("utf-8")),
)


# --- Property 1 ---------------------------------------------------------------------------

_ids = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-_", min_size=1, max_size=12)
_safe_session_ids = st.from_regex(r"\A[A-Za-z0-9_.-]{1,40}\Z")
_texts = st.text(max_size=30)
# Contracts store finite numbers; NaN/inf are not representable in JSON.
_money = st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False)
_opt_money = st.none() | _money
_pos_int = st.integers(min_value=1, max_value=10_000)
_nat = st.integers(min_value=0, max_value=10_000)
_times = st.datetimes(
    min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1), timezones=st.just(UTC)
)
_opt_times = st.none() | _times
_seconds = _times.map(lambda t: t.replace(microsecond=0))
_json_scalars = (
    st.none()
    | st.booleans()
    | st.integers(-(10**12), 10**12)
    | _texts
    | st.floats(allow_nan=False, allow_infinity=False)
)
_json_values = st.recursive(
    _json_scalars,
    lambda inner: st.lists(inner, max_size=3) | st.dictionaries(_texts, inner, max_size=3),
    max_leaves=8,
)
_json_dicts = st.dictionaries(_texts, _json_values, max_size=4)

_limits = st.builds(
    SessionLimits,
    max_converse_turns=_pos_int,
    max_tool_calls=_pos_int,
    max_costly_tool_calls=_pos_int,
    max_estimated_spend_usd=_money,
    max_active_seconds=_pos_int,
    max_repeated_identical_calls=_pos_int,
)
_counters = st.builds(
    SessionCounters,
    converse_turns=_nat,
    tool_calls=_nat,
    costly_tool_calls=_nat,
    estimated_spend_usd=st.just(0.0) | _money,
    active_seconds=st.just(0.0) | _money,
    consecutive_identical=_nat,
    last_fingerprint=st.none() | _ids,
    input_tokens=_nat,
    output_tokens=_nat,
)
_tool_calls = st.builds(
    ToolCallRecord,
    tool_call_id=_ids,
    turn=_nat,
    tool=_ids,
    arguments=_json_dicts,
    args_hash=_ids,
    tool_class=st.none() | st.sampled_from(ToolClass),
    state=st.sampled_from(CallState),
    approval_id=st.none() | _ids,
    idempotency_key=st.none() | _ids,
    estimate_usd=_opt_money,
    result_ref=st.none() | st.builds(ResultRef, uri=_texts, sha256=_ids, size_bytes=_nat),
    result_summary=st.none() | _texts,
    error_code=st.none() | _ids,
    truncated=st.booleans(),
    injection_suspected=st.booleans(),
    started_at=_opt_times,
    finished_at=_opt_times,
)
_approvals = st.builds(
    ApprovalRequest,
    approval_id=_ids,
    kind=st.sampled_from(["tool", "upload"]),
    tool_call_id=_ids,
    tool=_ids,
    arguments=_json_dicts,
    args_hash=_ids,
    approved_hash=st.none() | _ids,
    rationale=_texts,
    estimate_usd=_opt_money,
    estimate_basis=st.none() | _texts,
    status=st.sampled_from(ApprovalStatus),
    created_at=_times,
    expires_at=_times,
    edited_arguments=st.none() | _json_dicts,
    edited_hash=st.none() | _ids,
    decided_by=st.none() | _ids,
    decided_at=_opt_times,
    reason=st.none() | st.text(max_size=50),
    batch_id=st.none() | _ids,
)
_jobs = st.builds(
    AwaitedJob,
    execution_id=_ids,
    experiment_id=_ids,
    tool_call_id=_ids,
    kind=_ids,
    state=st.sampled_from(["running", "succeeded", "failed", "stopped", "not_found", "unknown"]),
    failure_reason=st.none() | _texts,
    observed_at=_opt_times,
    delivered=st.booleans(),
)
_datasets = st.builds(
    DatasetRef,
    dataset_id=_ids,
    uri=st.builds(lambda k: f"s3://bucket/{k}.csv", _ids),
    filename=_texts,
    sha256=st.none() | _ids,
    size_bytes=_nat,
    file_format=st.sampled_from(["csv", "parquet"]),
    columns=st.lists(
        st.builds(ColumnRef, name=_texts, dtype=st.none() | _ids, semantic_type=st.none() | _ids),
        max_size=3,
    ),
    row_count=st.none() | _nat,
)
_transcript_kinds = st.sampled_from(
    [
        "user_message",
        "assistant_text",
        "plan",
        "tool_call",
        "tool_result",
        "approval_request",
        "approval_decision",
        "status",
        "mode",
        "limit_stop",
        "job_update",
        "error",
        "final_answer",
    ]
)
_entries = st.builds(
    TranscriptEntry,
    seq=_nat,
    at=_times,
    kind=_transcript_kinds,
    text=st.none() | _texts,
    tool_call_id=st.none() | _ids,
    approval_id=st.none() | _ids,
    data=_json_dicts,
)
_final_answers = st.builds(
    FinalAnswer,
    sentences=st.lists(
        st.builds(
            AnswerSentence,
            text=_texts,
            status=st.sampled_from(["verified", "unverified", "overclaim"]),
            attributions=st.lists(
                st.builds(
                    Attribution,
                    tool_call_id=_ids,
                    experiment_id=st.none() | _ids,
                    artifact_ref=st.none() | _ids,
                    path=_texts,
                    value_text=_texts,
                    metric_label=st.none() | st.sampled_from(["self_reported", "engine"]),
                ),
                max_size=2,
            ),
        ),
        max_size=3,
    ),
    verbatim=st.lists(
        st.builds(VerbatimBlock, tool_call_id=_ids, title=_texts, content=_json_dicts),
        max_size=2,
    ),
    notes=st.lists(_texts, max_size=2),
    labels=st.lists(_ids, max_size=2),
)


@st.composite
def sessions(draw: st.DrawFn) -> Session:
    return Session(
        session_id=draw(_safe_session_ids),
        owner=draw(_ids),
        created_at=draw(_times),
        updated_at=draw(_times),
        last_activity_at=draw(_times),
        revision=draw(_nat),
        epoch=draw(_nat),
        status=draw(st.sampled_from(SessionStatus)),
        retriable=draw(st.booleans()),
        failure=draw(
            st.none()
            | st.builds(
                SessionFailure,
                category=_ids,
                message=_texts,
                retriable=st.booleans(),
                at=_opt_times,
            )
        ),
        mode=draw(st.sampled_from(Mode)),
        autopilot=draw(
            st.none()
            | st.builds(
                AutopilotGrant,
                enabled_by=_ids,
                enabled_at=_seconds,
                max_estimated_spend_usd=_money,
                spend_at_enable=st.just(0.0) | _money,
            )
        ),
        limits=draw(_limits),
        counters=draw(_counters),
        goal=draw(_texts),
        plan=draw(
            st.none()
            | st.builds(
                Plan,
                steps=st.lists(
                    st.builds(PlanStep, text=st.text(min_size=1, max_size=40)),
                    min_size=1,
                    max_size=4,
                ),
                accepted_at=_times,
            )
        ),
        transcript=draw(st.lists(_entries, max_size=4)),
        next_seq=draw(_pos_int),
        audit_seq=draw(_nat),
        tool_calls=draw(st.lists(_tool_calls, max_size=3)),
        pending_tool_use_ids=draw(st.lists(_ids, max_size=3)),
        approvals=draw(st.lists(_approvals, max_size=3)),
        awaited_jobs=draw(st.lists(_jobs, max_size=3)),
        jobs_ready=draw(st.booleans()),
        experiment_ids=draw(st.lists(_ids, max_size=3)),
        datasets=draw(st.lists(_datasets, max_size=2)),
        local_files=draw(
            st.lists(
                st.builds(
                    LocalFileRef,
                    file_id=_ids,
                    name=_texts,
                    size_bytes=_nat,
                    modified_at=_times,
                    sha256=_ids,
                ),
                max_size=2,
            )
        ),
        client_action=draw(
            st.none()
            | st.builds(
                ClientAction,
                action_id=_ids,
                kind=st.just("upload"),
                file_id=_ids,
                approval_id=_ids,
                status=st.sampled_from(["ready", "done"]),
            )
        ),
        upload_approved_once=draw(st.booleans()),
        guard_refusals=draw(
            st.lists(
                st.builds(GuardRefusal, tool=_ids, experiment_id=st.none() | _ids, code=_ids),
                max_size=2,
            )
        ),
        audit_failed=draw(st.booleans()),
        model_id=draw(st.none() | _ids),
        final_answer=draw(st.none() | _final_answers),
        compaction=draw(
            st.none()
            | st.builds(
                CompactionInfo,
                compacted_through_seq=_nat,
                summary_tokens_estimate=_nat,
                compacted_at=_opt_times,
            )
        ),
        lease=draw(st.none() | st.builds(Lease, holder=_ids, expires_at=_times)),
    )


# Feature: ml-assistant, Property 1: Session round trip and view determinism
@settings(max_examples=100, deadline=None)
@given(session=sessions(), after_seq=st.integers(min_value=0, max_value=10_000))
def test_session_round_trip_and_view_determinism(session, after_seq):
    """**Validates: Requirements 6.1, 6.9, 15.1**"""
    objects = InMemoryObjectStore()
    store = SessionStore(objects, ROOT)
    uri = AgentLayout.for_session(ROOT, session.session_id).session
    # The exact serialization SessionStore writes.
    stored = session.model_dump_json(indent=2, by_alias=True).encode("utf-8")
    objects.write_bytes(uri, stored)

    first = store.load(session.session_id).session
    second = store.load(session.session_id).session

    assert first == session
    assert second == session
    assert first.model_dump_json(indent=2, by_alias=True).encode("utf-8") == stored
    assert objects.read_bytes(uri) == stored  # load never writes

    view_a = build_view(first, after_seq)
    view_b = build_view(second, after_seq)
    assert view_a == view_b
    assert view_a.model_dump_json() == view_b.model_dump_json()


# --- Property 2 ---------------------------------------------------------------------------


# Feature: ml-assistant, Property 2: Unsupported or corrupt records are rejected untouched
@settings(max_examples=100, deadline=None)
@given(data=st.data(), contract=contract_names)
def test_unsupported_contract_version_is_rejected_untouched(data, contract):
    """**Validates: Requirements 14.8, 15.2**"""
    version = data.draw(unsupported_versions(contract))
    objects, store, uri = _new_store()
    payload = _valid_payload()
    target: Any = payload
    for step in _CONTRACT_PATHS[contract]:
        target = target[step]
    target["schema_version"] = version
    objects.write_bytes(uri, json.dumps(payload).encode("utf-8"))
    before = _snapshot(objects)

    with pytest.raises(SessionLoadError) as caught:
        store.load(SESSION_ID)

    err = caught.value
    assert err.session_id == SESSION_ID
    assert err.category == "unknown_contract_version"
    assert SESSION_ID in err.message
    assert contract in err.message
    assert version in err.message
    assert err.details["session_id"] == SESSION_ID
    assert err.details["category"] == "unknown_contract_version"
    assert _snapshot(objects) == before


# Feature: ml-assistant, Property 2: Unsupported or corrupt records are rejected untouched
@settings(max_examples=100, deadline=None)
@given(raw=corrupt_records)
def test_corrupt_record_is_rejected_untouched(raw):
    """**Validates: Requirements 14.8, 15.2**"""
    objects, store, uri = _new_store()
    objects.write_bytes(uri, raw)
    before = _snapshot(objects)

    with pytest.raises(SessionLoadError) as caught:
        store.load(SESSION_ID)

    err = caught.value
    assert err.session_id == SESSION_ID
    assert err.category == "corrupt_content"
    assert SESSION_ID in err.message
    assert err.details == {"session_id": SESSION_ID, "category": "corrupt_content"}
    assert _snapshot(objects) == before


# --- Property 28 --------------------------------------------------------------------------

_owners = st.sampled_from(["alice", "bob", "carol"])
# Few distinct minutes so ties on last_activity_at are common.
_activity = st.integers(min_value=0, max_value=20).map(lambda m: _T0 + timedelta(minutes=m))
_page_sizes = st.integers(min_value=1, max_value=5) | st.integers(min_value=1, max_value=100)


def _collect(fetch: Any) -> list[Any]:
    """Follow next_cursor until exhausted; every page must hold at most 100 items."""
    items: list[Any] = []
    cursor: str | None = None
    for _ in range(1000):
        page = fetch(cursor)
        assert len(page.items) <= 100
        items.extend(page.items)
        cursor = page.next_cursor
        if cursor is None:
            return items
    raise AssertionError("cursor never ended")


# Feature: ml-assistant, Property 28: Session listing
@settings(max_examples=100, deadline=None)
@given(
    rows=st.lists(st.tuples(_owners, _activity), max_size=25),
    who=_owners,
    page_size=_page_sizes,
)
def test_session_listing(rows, who, page_size):
    """**Validates: Requirements 6.8**"""
    from backend.agent.service import AgentService
    from backend.auth import Principal

    store = SessionStore(InMemoryObjectStore(), ROOT)
    for i, (owner, at) in enumerate(rows):
        store.create(_session(f"s-{i}").model_copy(update={"owner": owner, "last_activity_at": at}))
    expected = {f"s-{i}" for i, (owner, _) in enumerate(rows) if owner == who}

    service = AgentService(settings=None, store=store, runner=None, audit=None)
    listings = [
        _collect(lambda c: store.list_for_owner(who, limit=page_size, cursor=c)),
        _collect(lambda c: service.list_sessions(Principal(user=who), limit=page_size, cursor=c)),
    ]
    for items in listings:
        ids = [s.session_id for s in items]
        assert len(ids) == len(set(ids))  # each session exactly once
        assert set(ids) == expected  # only, and all of, the owner's sessions
        assert all(s.owner == who for s in items)
        times = [s.last_activity_at for s in items]
        assert times == sorted(times, reverse=True)  # most recent first
