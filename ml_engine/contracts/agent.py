"""Versioned contracts for the ML Assistant agent (sessions, approvals, audit).

Imports only ``pydantic`` and the standard library. Storage paths for these
objects live in ``ml_engine/io/layout.py``, never here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Any, Literal

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0"

# Owner decision: approval time-to-live defaults to 900 s, allowed 60..86400 s.
DEFAULT_APPROVAL_TTL_SECONDS = 900
MIN_APPROVAL_TTL_SECONDS = 60
MAX_APPROVAL_TTL_SECONDS = 86_400


class _Strict(BaseModel):
    """Local strict base (same config as ``common.StrictModel``).

    Kept local so this module imports only pydantic and the standard library.
    """

    model_config = ConfigDict(extra="forbid", use_enum_values=False, validate_assignment=True)

    schema_version: str = "1.0"


def _second_precision(value: datetime) -> datetime:
    return value.replace(microsecond=0)


# Timezone-aware timestamps (UTC by convention).
UtcDatetime = AwareDatetime
UtcSeconds = Annotated[AwareDatetime, AfterValidator(_second_precision)]


# --------------------------------------------------------------------- enums


class Mode(StrEnum):
    SUPERVISED = "supervised"
    AUTOPILOT = "autopilot"


class SessionStatus(StrEnum):
    RUNNING = "running"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    WAITING_FOR_JOB = "waiting_for_job"
    WAITING_FOR_USER = "waiting_for_user"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolClass(StrEnum):
    READ = "read"
    MUTATING = "mutating"
    COSTLY = "costly"


class CallState(StrEnum):
    REQUESTED = "requested"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    REJECTED = "rejected"
    REFUSED = "refused"
    DEFERRED = "deferred"
    UNKNOWN_OUTCOME = "unknown_outcome"
    CANCELLED = "cancelled"


class ApprovalStatus(StrEnum):
    PENDING = "pending"
    EDIT_PROPOSED = "edit_proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    EXECUTED = "executed"
    CANCELLED = "cancelled"


# ------------------------------------------------------------------- helpers


def args_hash(tool: str, arguments: dict[str, Any]) -> str:
    """Canonical sha256 of ``{"tool", "arguments"}``; key order never matters."""
    payload = json.dumps(
        {"tool": tool, "arguments": arguments},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def idempotency_key(session_id: str, approval_id: str, args_hash: str) -> str:
    """sha256 of ``session_id | approval_id | args_hash``."""
    return hashlib.sha256(f"{session_id}|{approval_id}|{args_hash}".encode()).hexdigest()


# -------------------------------------------------------------------- limits


class SessionLimits(_Strict):
    """Positive integers, except spend which is a positive decimal (USD)."""

    schema_version: str = SCHEMA_VERSION
    max_converse_turns: int = Field(gt=0)
    max_tool_calls: int = Field(gt=0)
    max_costly_tool_calls: int = Field(gt=0)
    max_estimated_spend_usd: float = Field(gt=0)
    max_active_seconds: int = Field(gt=0)
    max_repeated_identical_calls: int = Field(gt=0)


class SessionCounters(_Strict):
    converse_turns: int = 0
    tool_calls: int = 0
    costly_tool_calls: int = 0
    estimated_spend_usd: float = 0.0
    active_seconds: float = 0.0
    consecutive_identical: int = 0
    last_fingerprint: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0


class AutopilotGrant(_Strict):
    enabled_by: str
    enabled_at: UtcSeconds
    max_estimated_spend_usd: float = Field(gt=0)
    spend_at_enable: float = 0.0


class PlanStep(_Strict):
    text: str = Field(min_length=1, max_length=200)


class Plan(_Strict):
    steps: list[PlanStep] = Field(min_length=1, max_length=10)
    accepted_at: UtcDatetime


# --------------------------------------------------------------- tool calls


class ResultRef(_Strict):
    uri: str
    sha256: str
    size_bytes: int


class ToolCallRecord(_Strict):
    schema_version: str = SCHEMA_VERSION
    tool_call_id: str
    turn: int
    tool: str
    arguments: dict[str, Any]
    args_hash: str
    tool_class: ToolClass | None  # None for an unknown tool
    state: CallState
    approval_id: str | None = None
    idempotency_key: str | None = None  # Costly only
    estimate_usd: float | None = None
    result_ref: ResultRef | None = None
    result_summary: str | None = None
    error_code: str | None = None
    truncated: bool = False
    injection_suspected: bool = False
    service_audit_ref: str | None = None
    started_at: UtcDatetime | None = None
    finished_at: UtcDatetime | None = None


class ApprovalRequest(_Strict):
    schema_version: str = SCHEMA_VERSION
    approval_id: str
    kind: Literal["tool", "upload"]
    tool_call_id: str
    tool: str
    arguments: dict[str, Any]  # exactly what the user sees
    args_hash: str
    approved_hash: str | None = None
    rationale: str  # model text, truncated, shown as untrusted
    estimate_usd: float | None  # None = "no estimate available"
    estimate_basis: str | None
    status: ApprovalStatus
    created_at: UtcDatetime
    expires_at: UtcDatetime
    edited_arguments: dict[str, Any] | None = None
    edited_hash: str | None = None
    decided_by: str | None = None
    decided_at: UtcDatetime | None = None
    reason: str | None = Field(default=None, max_length=500)
    batch_id: str | None = None
    #: What the user must see besides the arguments, e.g. a free-form script and its hash.
    review: dict[str, Any] | None = None


class ApprovalDecision(_Strict):
    schema_version: str = SCHEMA_VERSION
    approval_id: str
    decision: Literal["approve", "reject", "edit"]
    decided_by: str
    decided_at: UtcDatetime
    args_hash: str
    reason: str | None = Field(default=None, max_length=500)
    edited_arguments: dict[str, Any] | None = None


class AwaitedJob(_Strict):
    execution_id: str
    experiment_id: str
    tool_call_id: str
    kind: str
    state: Literal["running", "succeeded", "failed", "stopped", "not_found", "unknown"] = "running"
    failure_reason: str | None = None
    observed_at: UtcDatetime | None = None
    delivered: bool = False  # already shown to the model


# ----------------------------------------------------------------- datasets


class ColumnRef(_Strict):
    """A column of a registered dataset as the agent may describe it (no rows)."""

    name: str
    dtype: str | None = None
    semantic_type: str | None = None


class DatasetRef(_Strict):
    dataset_id: str
    uri: str
    filename: str
    sha256: str | None
    size_bytes: int
    file_format: Literal["csv", "parquet"]
    columns: list[ColumnRef]
    row_count: int | None


class LocalFileRef(_Strict):
    file_id: str
    name: str
    size_bytes: int
    modified_at: UtcDatetime
    sha256: str


class ClientAction(_Strict):
    action_id: str
    kind: Literal["upload"]
    file_id: str
    approval_id: str
    status: Literal["ready", "done"]
    #: The pending ``request_local_upload`` use this action resolves.
    tool_call_id: str | None = None
    #: Where the last minted PUT points. A plain location, never a signed URL.
    dataset_uri: str | None = None


# --------------------------------------------------------------- transcript


class TranscriptEntry(_Strict):
    seq: int
    at: UtcDatetime
    kind: Literal[
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
    text: str | None = None
    tool_call_id: str | None = None
    approval_id: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)  # JSON-safe, small


class Attribution(_Strict):
    tool_call_id: str
    experiment_id: str | None
    artifact_ref: str | None
    path: str
    value_text: str
    metric_label: Literal["self_reported", "engine"] | None = None


class AnswerSentence(_Strict):
    text: str
    status: Literal["verified", "unverified", "overclaim"]
    attributions: list[Attribution] = Field(default_factory=list)


class VerbatimBlock(_Strict):
    tool_call_id: str
    title: str
    content: dict[str, Any]


class FinalAnswer(_Strict):
    sentences: list[AnswerSentence] = Field(default_factory=list)
    verbatim: list[VerbatimBlock] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    labels: list[str] = Field(default_factory=list)


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One engine number a narration may cite.

    Lives here so ``results.py`` and ``narration.py`` need not import each other.
    """

    value: float
    tool_call_id: str
    experiment_id: str | None
    path: str
    metric_label: str | None


# ------------------------------------------------------------------ session


class Lease(_Strict):
    holder: str
    expires_at: UtcDatetime


class SessionFailure(_Strict):
    """Why a session is ``failed``. Never carries exception text or credentials."""

    category: str  # e.g. planning_failed, bedrock_unavailable, persist_failed
    message: str
    retriable: bool = False
    at: UtcDatetime | None = None


class GuardRefusal(_Strict):
    tool: str
    experiment_id: str | None = None
    code: str


class CompactionInfo(_Strict):
    """Records that earlier turns were summarised for the model context."""

    compacted_through_seq: int = 0
    summary_tokens_estimate: int = 0
    compacted_at: UtcDatetime | None = None


class Session(_Strict):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    owner: str
    created_at: UtcDatetime
    updated_at: UtcDatetime
    last_activity_at: UtcDatetime
    revision: int = 0
    epoch: int = 0  # bumped by user actions that must win over an in-flight model call
    status: SessionStatus
    retriable: bool = False
    failure: SessionFailure | None = None
    mode: Mode = Mode.SUPERVISED
    autopilot: AutopilotGrant | None = None
    limits: SessionLimits
    counters: SessionCounters = Field(default_factory=SessionCounters)
    goal: str
    plan: Plan | None = None
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    next_seq: int = 1
    audit_seq: int = 0
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    pending_tool_use_ids: list[str] = Field(default_factory=list)
    approvals: list[ApprovalRequest] = Field(default_factory=list)
    awaited_jobs: list[AwaitedJob] = Field(default_factory=list)
    jobs_ready: bool = False
    experiment_ids: list[str] = Field(default_factory=list)
    datasets: list[DatasetRef] = Field(default_factory=list)
    local_files: list[LocalFileRef] = Field(default_factory=list)
    client_action: ClientAction | None = None
    upload_approved_once: bool = False
    guard_refusals: list[GuardRefusal] = Field(default_factory=list)  # cleared on user action
    audit_failed: bool = False
    model_id: str | None = None
    final_answer: FinalAnswer | None = None
    compaction: CompactionInfo | None = None
    lease: Lease | None = None


class SessionSummary(_Strict):
    """The owner-index object."""

    schema_version: str = SCHEMA_VERSION
    session_id: str
    owner: str
    status: SessionStatus
    mode: Mode
    experiment_ids: list[str]
    last_activity_at: UtcDatetime
    created_at: UtcDatetime


class AgentAuditRecord(_Strict):
    schema_version: str = SCHEMA_VERSION
    session_id: str
    owner: str
    actor: str
    seq: int
    event_type: str
    at: UtcDatetime
    result_status: Literal["success", "failure", "rejected", "cancelled"]
    tool: str | None = None
    args_hash: str | None = None
    decision: str | None = None
    decided_by: str | None = None  # user id, or "autopilot:<user>"
    service_audit_ref: str | None = None
    prompt_injection_suspected: bool = False
    detail: dict[str, str] = Field(default_factory=dict)


# Drives the version check in ``SessionStore.load``.
SUPPORTED_VERSIONS: dict[str, frozenset[str]] = {
    "Session": frozenset({SCHEMA_VERSION}),
    "SessionLimits": frozenset({SCHEMA_VERSION}),
    "ToolCallRecord": frozenset({SCHEMA_VERSION}),
    "ApprovalRequest": frozenset({SCHEMA_VERSION}),
    "ApprovalDecision": frozenset({SCHEMA_VERSION}),
    "SessionSummary": frozenset({SCHEMA_VERSION}),
    "AgentAuditRecord": frozenset({SCHEMA_VERSION}),
}
