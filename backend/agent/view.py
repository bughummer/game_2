"""Client-facing view of a session.

``build_view`` is a pure function of the session and ``after_seq``: the same
inputs always give an equal view (no clock, no I/O). The view never carries
dataset URIs, result-object URIs or signed URLs. Datasets appear by id only.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ml_engine.contracts.agent import (
    ApprovalRequest,
    ApprovalStatus,
    AwaitedJob,
    FinalAnswer,
    Mode,
    Session,
    SessionCounters,
    SessionFailure,
    SessionLimits,
    SessionStatus,
    TranscriptEntry,
)

VIEW_SCHEMA_VERSION = "1.0"

_URI_RE = re.compile(r"\b(?:s3|s3a|s3n|https?|file)://[^\s\"'<>)]*", re.IGNORECASE)
_SIGNED_PARAM_RE = re.compile(
    r"(X-Amz-[A-Za-z-]+|AWSAccessKeyId|Signature|Expires|x-amz-[a-z-]+)=[^&\s\"']*"
)
_URI_KEYS = frozenset(
    {"uri", "url", "s3_uri", "presigned_url", "signed_url", "upload_url", "download_url"}
)
_PENDING = (ApprovalStatus.PENDING, ApprovalStatus.EDIT_PROPOSED)


def _scrub_text(text: str) -> str:
    text = _URI_RE.sub("[uri removed]", text)
    return _SIGNED_PARAM_RE.sub("[redacted]", text)


def _scrub(value: Any) -> Any:
    """Drop URI-like keys and rewrite URI-like strings, recursively."""
    if isinstance(value, str):
        return _scrub_text(value)
    if isinstance(value, dict):
        return {
            k: _scrub(v)
            for k, v in value.items()
            if not (isinstance(k, str) and k.lower() in _URI_KEYS)
        }
    if isinstance(value, (list, tuple)):
        return [_scrub(v) for v in value]
    return value


class DatasetView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    filename: str
    file_format: str
    size_bytes: int
    row_count: int | None
    column_count: int


class ApprovalView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    approval_id: str
    kind: str
    tool_call_id: str
    tool: str
    arguments: dict[str, Any]
    args_hash: str
    rationale: str
    estimate_usd: float | None
    estimate_basis: str | None
    status: str
    created_at: str
    expires_at: str
    edited_arguments: dict[str, Any] | None = None
    edited_hash: str | None = None
    review: dict[str, Any] | None = None


class SessionView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = VIEW_SCHEMA_VERSION
    session_id: str
    owner: str
    status: SessionStatus
    retriable: bool
    failure: SessionFailure | None
    mode: Mode
    autopilot_max_estimated_spend_usd: float | None
    goal: str
    plan: list[str] = Field(default_factory=list)
    limits: SessionLimits
    counters: SessionCounters
    pending_approvals: list[ApprovalView] = Field(default_factory=list)
    running_jobs: list[AwaitedJob] = Field(default_factory=list)
    jobs_ready: bool
    experiment_ids: list[str] = Field(default_factory=list)
    datasets: list[DatasetView] = Field(default_factory=list)
    client_action: dict[str, Any] | None = None
    transcript: list[TranscriptEntry] = Field(default_factory=list)
    last_seq: int
    final_answer: FinalAnswer | None = None
    created_at: str
    last_activity_at: str


def _approval_view(a: ApprovalRequest) -> ApprovalView:
    return ApprovalView(
        approval_id=a.approval_id,
        kind=a.kind,
        tool_call_id=a.tool_call_id,
        tool=a.tool,
        arguments=_scrub(a.arguments),
        args_hash=a.args_hash,
        rationale=_scrub_text(a.rationale),
        estimate_usd=a.estimate_usd,
        estimate_basis=a.estimate_basis,
        status=a.status.value,
        created_at=a.created_at.isoformat(),
        expires_at=a.expires_at.isoformat(),
        edited_arguments=(
            _scrub(a.edited_arguments) if a.edited_arguments is not None else None
        ),
        edited_hash=a.edited_hash,
        review=_scrub(a.review) if a.review is not None else None,
    )


def _entry_view(e: TranscriptEntry) -> TranscriptEntry:
    return e.model_copy(
        update={
            "text": _scrub_text(e.text) if e.text is not None else None,
            "data": _scrub(e.data),
        }
    )


def build_view(session: Session, after_seq: int = 0) -> SessionView:
    """Build the client view; transcript holds entries with ``seq > after_seq``."""
    pending = [_approval_view(a) for a in session.approvals if a.status in _PENDING]
    running = [j for j in session.awaited_jobs if j.state == "running"]
    datasets = [
        DatasetView(
            dataset_id=d.dataset_id,
            filename=_scrub_text(d.filename),
            file_format=d.file_format,
            size_bytes=d.size_bytes,
            row_count=d.row_count,
            column_count=len(d.columns),
        )
        for d in session.datasets
    ]
    action = None
    if session.client_action is not None:
        ca = session.client_action
        action = {
            "action_id": ca.action_id,
            "kind": ca.kind,
            "file_id": ca.file_id,
            "approval_id": ca.approval_id,
            "status": ca.status,
        }
    return SessionView(
        session_id=session.session_id,
        owner=session.owner,
        status=session.status,
        retriable=session.retriable,
        failure=session.failure,
        mode=session.mode,
        autopilot_max_estimated_spend_usd=(
            session.autopilot.max_estimated_spend_usd if session.autopilot else None
        ),
        goal=_scrub_text(session.goal),
        plan=[s.text for s in session.plan.steps] if session.plan else [],
        limits=session.limits,
        counters=session.counters,
        pending_approvals=pending,
        running_jobs=running,
        jobs_ready=session.jobs_ready,
        experiment_ids=list(session.experiment_ids),
        datasets=datasets,
        client_action=action,
        transcript=[_entry_view(e) for e in session.transcript if e.seq > after_seq],
        last_seq=session.next_seq - 1,
        final_answer=session.final_answer,
        created_at=session.created_at.isoformat(),
        last_activity_at=session.last_activity_at.isoformat(),
    )
