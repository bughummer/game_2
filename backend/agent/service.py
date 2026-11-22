"""Service facade for the ML Assistant (design C15).

Minimal version (task 4.15): ``create_session``, ``send_message``, ``resume``, ``get_session``,
``list_sessions`` and a minimal ``cancel``. Task 8.4 adds ``decide_approval``, ``decide_batch``
and the approval expiry sweep. Task 8.5 adds ``set_mode`` and ``continue_after_limit``.
Audit trail, local uploads and job refresh are added by later tasks without changing these
signatures.

Every method first loads the session and compares ``session.owner`` with ``who.user``. A
mismatch raises ``AuthorizationError`` before any content is read or anything is written. Write
methods check the owner inside the session transaction, so a raise leaves the record unsaved.

The service never runs the loop itself. After a change that sets ``running`` it kicks the
runner, which calls ``AgentLoop.run(session_id, holder)``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from backend.agent.execute import (
    CODE_INPUT_REJECTED,
    CODE_INVALID_ARGUMENTS,
    ToolError,
    input_rejected_error,
    invalid_arguments_error,
)
from backend.agent.jobs import (
    TERMINAL_JOB_STATES,
    JobObservation,
    apply_observations,
    jobs_ready,
    observe_jobs,
)
from backend.agent.limits import LimitName, raise_limit
from backend.agent.results import ScrubContext, scrub_sensitive
from backend.agent.store import SessionInUse, SessionStore
from backend.agent.tools import ToolContext, ToolOutput
from backend.agent.uploads import RowCounter, dataset_id_for, format_for
from backend.agent.view import SessionView, build_view
from backend.audit import AuditWriteError
from backend.auth import Principal
from backend.errors import (
    AppError,
    AuthorizationError,
    ConflictError,
    FeatureDisabledError,
    ValidationError,
)
from backend.services.datasets import DatasetParseError
from ml_engine.contracts.agent import (
    AgentAuditRecord,
    ApprovalRequest,
    ApprovalStatus,
    AutopilotGrant,
    CallState,
    ClientAction,
    ColumnRef,
    DatasetRef,
    LocalFileRef,
    Mode,
    Session,
    SessionFailure,
    SessionLimits,
    SessionStatus,
    SessionSummary,
    TranscriptEntry,
)
from ml_engine.contracts.agent import args_hash as hash_arguments

LOGGER = logging.getLogger("ml_factory.agent.service")

ENABLED_SETTING = "ML_FACTORY_AGENT_ENABLED"
AUTOPILOT_SETTING = "ML_FACTORY_AGENT_AUTOPILOT_ALLOWED"
MAX_TEXT_CHARS = 4_000
MAX_PAGE_SIZE = 100

CODE_CANCELLED = "CANCELLED"
CODE_APPROVAL_REJECTED = "APPROVAL_REJECTED"
CODE_APPROVAL_EXPIRED = "APPROVAL_EXPIRED"
CODE_DEFERRED = "DEFERRED_WAITING_FOR_USER"
FAIL_AUDIT = "audit_failed"
MAX_REASON_CHARS = 500
MAX_BATCH_ITEMS = 10
DECISIONS = ("approve", "reject", "edit")
MAX_LOCAL_FILES = 100

#: (SessionLimits field, Settings field) for the integer limits (Req 9.1).
_INT_LIMITS: tuple[tuple[str, str], ...] = (
    ("max_converse_turns", "agent_limit_max_converse_turns"),
    ("max_tool_calls", "agent_limit_max_tool_calls"),
    ("max_costly_tool_calls", "agent_limit_max_costly_tool_calls"),
    ("max_active_seconds", "agent_limit_max_active_seconds"),
    ("max_repeated_identical_calls", "agent_limit_max_repeated_identical_calls"),
)
_SPEND_LIMIT = ("max_estimated_spend_usd", "agent_limit_max_estimated_spend_usd")

_PENDING_APPROVALS = (ApprovalStatus.PENDING, ApprovalStatus.EDIT_PROPOSED)
#: Tool-call states the cancel resolves. An ``executing`` call is left to the loop, which
#: records its real outcome when the handler returns.
_CANCELLABLE_CALLS = (CallState.REQUESTED, CallState.AWAITING_APPROVAL)


class SessionPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[SessionSummary] = Field(default_factory=list)
    next_cursor: str | None = None


class LocalFileInput(BaseModel):
    """One candidate file as the client lists it. A plain name, never a path."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=255, pattern=r"^[^/\\]+$")
    size_bytes: int = Field(ge=0)
    modified_at: datetime
    sha256: str = Field(pattern=r"^[0-9a-fA-F]{64}$")

    def to_ref(self) -> LocalFileRef:
        digest = hashlib.sha256(f"{self.name}\0{self.sha256.lower()}".encode()).hexdigest()
        return LocalFileRef(
            file_id=f"lf_{digest[:16]}",
            name=self.name,
            size_bytes=self.size_bytes,
            modified_at=self.modified_at,
            sha256=self.sha256.lower(),
        )


class ClientActionView(BaseModel):
    """What the client needs to perform one upload. Returned, never persisted."""

    model_config = ConfigDict(extra="forbid")

    action_id: str
    kind: str
    file_id: str
    file_name: str
    size_bytes: int
    sha256: str
    upload_url: str
    method: str
    headers: dict[str, str]
    expires_at: datetime
    max_bytes: int


class BatchItem(BaseModel):
    """One displayed approval the user approves in a batch, bound to the hash they saw."""

    model_config = ConfigDict(extra="forbid")

    approval_id: str
    args_hash: str


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _setting_name(field: str) -> str:
    return f"ML_FACTORY_{field.upper()}"


def validate_text(value: Any, *, what: str) -> str:
    """1 to 4,000 characters that are not all whitespace (Req 1.2). Returns ``value`` unchanged."""
    if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT_CHARS:
        raise ValidationError(
            f"The {what} must be 1 to {MAX_TEXT_CHARS:,} characters and not blank.",
            {"field": what, "min_length": "1", "max_length": str(MAX_TEXT_CHARS)},
        )
    return value


def limits_from_settings(settings: Any) -> SessionLimits:
    """Session limits from the ``agent_limit_*`` settings. Raises naming the invalid limit."""
    values: dict[str, Any] = {}
    for field, name in _INT_LIMITS:
        value = getattr(settings, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValidationError(
                f"Session limit {_setting_name(name)} must be a positive integer (got {value!r}).",
                {"limit": field, "setting": _setting_name(name)},
            )
        values[field] = value
    field, name = _SPEND_LIMIT
    raw = getattr(settings, name, None)
    try:
        spend = float(raw) if raw is not None and not isinstance(raw, bool) else math.nan
    except (TypeError, ValueError):
        spend = math.nan
    if not math.isfinite(spend) or spend <= 0:
        raise ValidationError(
            f"Session limit {_setting_name(name)} must be greater than 0 (got {raw!r}).",
            {"limit": field, "setting": _setting_name(name)},
        )
    values[field] = spend
    return SessionLimits(**values)


class AgentService:
    def __init__(
        self,
        *,
        settings: Any,
        store: SessionStore,
        runner: Any,
        audit: Any,
        loop: Any = None,
        clock: Callable[[], datetime] = _utcnow,
        registry: Any = None,
        inputs: Any = None,
        analysis: Any = None,
        orchestrator: Any = None,
        services: Any = None,
    ) -> None:
        self._settings = settings
        self._store = store
        self._runner = runner
        self._audit = audit
        self._loop = loop
        self._clock = clock
        #: Used only to validate edited approval arguments (the tool's input model and checks).
        self._registry = registry
        self._inputs = inputs
        #: ``AnalysisService``: reads awaited job states (``refresh_jobs``). None: no refresh.
        self._analysis = analysis
        #: The platform orchestrator: ``stop(execution_id)`` for ``cancel(stop_jobs=True)``.
        self._orchestrator = orchestrator
        #: What tools reach through ``ctx.services``; used to rebuild an edited request's review.
        self._services = services

    # ------------------------------------------------------------------ accessors
    @property
    def store(self) -> SessionStore:
        return self._store

    @property
    def runner(self) -> Any:
        return self._runner

    @property
    def loop(self) -> Any:
        return self._loop

    # ------------------------------------------------------------------ create
    def create_session(
        self,
        who: Principal,
        goal: str,
        *,
        mode: Mode | str = Mode.SUPERVISED,
        autopilot_max_spend_usd: float | None = None,
    ) -> SessionView:
        # Order (design C15): enabled, goal, limits, Autopilot arguments. Nothing is written
        # until every check passed.
        self._require_enabled()
        user = self._user(who)
        goal = scrub_sensitive(validate_text(goal, what="goal"), ScrubContext())
        limits = limits_from_settings(self._settings)
        now = self._clock()
        chosen, grant = self._autopilot(user, mode, autopilot_max_spend_usd, limits, now)

        session = Session(
            session_id=f"ses_{uuid.uuid4().hex}",
            owner=user,
            created_at=now,
            updated_at=now,
            last_activity_at=now,
            status=SessionStatus.RUNNING,
            mode=chosen,
            autopilot=grant,
            limits=limits,
            goal=goal,
        )
        self._add(session, "user_message", text=goal)
        self._add(session, "status", data={"status": SessionStatus.RUNNING.value})
        self._store.create(session)
        sid = session.session_id

        record = AgentAuditRecord(
            session_id=sid,
            owner=user,
            actor=user,
            seq=0,
            event_type="session_created",
            at=now,
            result_status="success",
            detail={"mode": chosen.value, "goal_chars": str(len(goal))},
        )
        try:
            with self._store.transaction(sid) as s:
                self._audit.write(s, record)
        except AuditWriteError:
            with self._store.transaction(sid) as s:
                s.audit_failed = True
                self._fail(
                    s, FAIL_AUDIT, "The session could not be audited, so it was not started."
                )
            raise
        self._kick(sid)
        return self._view(sid)

    # ------------------------------------------------------------------ messages / resume
    def send_message(self, who: Principal, session_id: str, text: str) -> SessionView:
        """Answer the agent while the session waits for the user (Req 1.6)."""
        self._require_enabled()
        user = self._user(who)
        with self._owned(session_id, user) as s:
            text = scrub_sensitive(validate_text(text, what="message"), ScrubContext())
            if s.status is not SessionStatus.WAITING_FOR_USER:
                raise ConflictError(
                    f"Session {session_id} is {s.status.value}; a message can be sent only "
                    "while it waits for the user.",
                    {"session_id": session_id, "reason": "not_waiting_for_user",
                     "status": s.status.value},
                )
            s.epoch += 1
            s.guard_refusals = []
            self._add(s, "user_message", text=text)
            self._set_status(s, SessionStatus.RUNNING)
        self._kick(session_id)
        return self._view(session_id)

    def resume(self, who: Principal, session_id: str) -> SessionView:
        """Restart the loop of a retriable failure, a stale run or a job wait (Req 2.9, 6.10)."""
        self._require_enabled()
        user = self._user(who)
        current = self._store.load(session_id).session
        self._check_owner(current, user)
        observed = self._observe_jobs(current)
        with self._owned(session_id, user) as s:
            if self._in_use(s):
                raise SessionInUse(session_id)
            self._check_resumable(s)
            apply_observations(s, observed, self._clock())
            s.epoch += 1
            s.failure = None
            s.retriable = False
            s.guard_refusals = []
            s.last_activity_at = self._clock()
            self._set_status(s, SessionStatus.RUNNING)
        self._kick(session_id)
        return self._view(session_id)

    # ------------------------------------------------------------------ reads
    def get_session(self, who: Principal, session_id: str, *, after_seq: int = 0) -> SessionView:
        user = self._user(who)
        session = self._store.load(session_id).session
        self._check_owner(session, user)
        if self._overdue(session, self._clock()):
            self.expire_approvals(session_id)
            session = self._store.load(session_id).session
        observed = self._observe_jobs(session)
        if observed or session.jobs_ready != jobs_ready(session):
            # Saved only when a job ended (or ``jobs_ready`` is stale) (Req 6.5, 6.6).
            with self._store.transaction(session_id) as s:
                apply_observations(s, observed, self._clock())
            session = self._store.load(session_id).session
        return build_view(session, max(0, int(after_seq)))

    def _observe_jobs(self, session: Session) -> list[JobObservation]:
        """Terminal states of unfinished awaited jobs, read outside any transaction."""
        if self._analysis is None:
            return []
        return observe_jobs(session, self._analysis)

    def list_sessions(
        self, who: Principal, *, limit: int = MAX_PAGE_SIZE, cursor: str | None = None
    ) -> SessionPage:
        """The caller's sessions, newest activity first, at most 100 per page (Req 6.8)."""
        user = self._user(who)
        size = min(MAX_PAGE_SIZE, max(1, int(limit)))
        page = self._store.list_for_owner(user, limit=size, cursor=cursor)
        items = [item for item in page.items if item.owner == user]
        return SessionPage(items=items, next_cursor=page.next_cursor)

    def audit_trail(self, who: Principal, session_id: str) -> list[Any]:
        """The session's audit records in chronological order; owner only (Req 11.6, 11.7)."""
        user = self._user(who)
        session = self._store.load(session_id).session
        self._check_owner(session, user)
        return self._audit.trail(session)

    # ------------------------------------------------------------------ approvals
    def decide_approval(
        self,
        who: Principal,
        session_id: str,
        approval_id: str,
        decision: str,
        *,
        args_hash: str,
        edited_args: dict[str, Any] | None = None,
        reason: str | None = None,
    ) -> SessionView:
        """Record one user decision (Req 4.6 to 4.12). Never runs a handler.

        ``args_hash`` must equal the hash of the arguments the user was shown: ``edited_hash``
        while an edit is proposed, otherwise ``args_hash``. Approve and reject set the session
        ``running`` and kick the runner; the loop executes an approved call only when the
        approval's ``approved_hash`` equals the call's ``args_hash``.
        """
        self._require_enabled()
        user = self._user(who)
        if decision not in DECISIONS:
            raise ValidationError(
                "The decision must be 'approve', 'reject' or 'edit'.", {"field": "decision"}
            )
        if reason is not None and not isinstance(reason, str):
            raise ValidationError("The reason must be text.", {"field": "reason"})
        if decision == "edit" and not isinstance(edited_args, dict):
            raise ValidationError(
                "An edit needs the edited arguments as an object.", {"field": "edited_args"}
            )

        refusal: Exception | None = None
        with self._owned(session_id, user) as s:
            now = self._clock()
            swept = self._expire_in(s, now)
            approval = next((a for a in s.approvals if a.approval_id == approval_id), None)
            refusal = self._refusal(session_id, approval_id, approval, args_hash)
            if refusal is not None:
                if not swept:
                    raise refusal  # nothing changed, so nothing is saved
            elif decision == "approve":
                self._approve(s, approval, user, now)  # type: ignore[arg-type]
            elif decision == "reject":
                self._reject(s, approval, user, now, reason)  # type: ignore[arg-type]
            else:
                refusal = self._edit(s, approval, user, now, edited_args)  # type: ignore[arg-type]
            running = s.status is SessionStatus.RUNNING
        if refusal is not None:
            raise refusal
        if running and decision != "edit":
            self._kick(session_id)
        return self._view(session_id)

    def decide_batch(
        self, who: Principal, session_id: str, items: list[BatchItem | dict[str, Any]]
    ) -> SessionView:
        """Approve up to 10 displayed requests at once (Req 4.13).

        Each item is approved by the same rules as ``decide_approval``. An item whose hash does
        not match what is displayed, or whose request is no longer pending, stays as it is.
        """
        self._require_enabled()
        user = self._user(who)
        if not isinstance(items, list) or not 1 <= len(items) <= MAX_BATCH_ITEMS:
            raise ValidationError(
                f"A batch approval takes 1 to {MAX_BATCH_ITEMS} items.",
                {"field": "items", "max_items": str(MAX_BATCH_ITEMS)},
            )
        try:
            parsed = [BatchItem.model_validate(item) for item in items]
        except PydanticValidationError:
            raise ValidationError(
                "Each batch item needs an approval_id and an args_hash.", {"field": "items"}
            ) from None
        batch_id = f"bat_{uuid.uuid4().hex[:16]}"
        with self._owned(session_id, user) as s:
            now = self._clock()
            self._expire_in(s, now)
            approved = 0
            for item in parsed:
                approval = next(
                    (a for a in s.approvals if a.approval_id == item.approval_id), None
                )
                if self._refusal(session_id, item.approval_id, approval, item.args_hash):
                    continue
                self._approve(s, approval, user, now, batch_id=batch_id)  # type: ignore[arg-type]
                approved += 1
            running = s.status is SessionStatus.RUNNING
        if running and approved:
            self._kick(session_id)
        return self._view(session_id)

    def expire_approvals(self, session_id: str, now: datetime | None = None) -> int:
        """Mark every undecided request past ``expires_at`` as ``expired`` (Req 4.12).

        The expired call gets an ``APPROVAL_EXPIRED`` result, the other pending uses get
        deferred results (their requests are cancelled) and the session waits for the user.
        Callers check ownership first. Returns the number of requests expired.
        """
        now = now or self._clock()
        if not self._overdue(self._store.load(session_id).session, now):
            return 0
        with self._store.transaction(session_id) as s:
            return self._expire_in(s, now)

    @staticmethod
    def _displayed_hash(approval: ApprovalRequest) -> str:
        if approval.status is ApprovalStatus.EDIT_PROPOSED and approval.edited_hash:
            return approval.edited_hash
        return approval.args_hash

    def _refusal(
        self,
        session_id: str,
        approval_id: str,
        approval: ApprovalRequest | None,
        supplied_hash: str,
    ) -> Exception | None:
        details = {"session_id": session_id, "approval_id": approval_id}
        if approval is None:
            return ConflictError(
                f"Approval request {approval_id} does not exist in this session.",
                {**details, "reason": "approval_not_found"},
            )
        if approval.status is ApprovalStatus.EXPIRED:
            return ConflictError(
                f"Approval request {approval_id} has expired. Ask the agent to propose the "
                "step again.",
                {**details, "reason": "approval_expired"},
            )
        if approval.status not in _PENDING_APPROVALS:
            return ConflictError(
                f"Approval request {approval_id} is {approval.status.value} and cannot be "
                "decided again.",
                {**details, "reason": "approval_not_pending", "status": approval.status.value},
            )
        if supplied_hash != self._displayed_hash(approval):
            return ConflictError(
                f"Approval request {approval_id} no longer matches the arguments you saw. "
                "Review the request again.",
                {**details, "reason": "approval_mismatch"},
            )
        return None

    def _approve(
        self,
        s: Session,
        approval: ApprovalRequest,
        user: str,
        now: datetime,
        *,
        batch_id: str | None = None,
    ) -> None:
        approved_hash = self._displayed_hash(approval)
        if approval.status is ApprovalStatus.EDIT_PROPOSED:
            # The call runs with exactly the confirmed edited arguments.
            record = self._call(s, approval.tool_call_id)
            if record is not None:
                record.arguments = dict(approval.edited_arguments or {})
                record.args_hash = approved_hash
        approval.status = ApprovalStatus.APPROVED
        approval.approved_hash = approved_hash
        approval.decided_by = user
        approval.decided_at = now
        approval.batch_id = batch_id
        self._record_decision(s, approval, user, "approve", "success", approved_hash)
        self._resume_after_decision(s)

    def _reject(
        self, s: Session, approval: ApprovalRequest, user: str, now: datetime, reason: str | None
    ) -> None:
        reason = scrub_sensitive(reason, ScrubContext())[:MAX_REASON_CHARS] if reason else None
        approval.status = ApprovalStatus.REJECTED
        approval.decided_by = user
        approval.decided_at = now
        approval.reason = reason
        summary = "The user rejected this call. It was not run."
        if reason:
            summary += f" Reason: {reason}"
        # Only the head use gets its result now. A later (pre-created) one stays pending, so
        # the loop gives it the rejection result in request order (Req 2.6).
        pending = s.pending_tool_use_ids
        if pending and pending[0] == approval.tool_call_id:
            self._resolve_call(
                s, approval.tool_call_id, CallState.REJECTED, CODE_APPROVAL_REJECTED, summary
            )
        self._record_decision(s, approval, user, "reject", "rejected", approval.args_hash)
        self._resume_after_decision(s)

    def _edit(
        self,
        s: Session,
        approval: ApprovalRequest,
        user: str,
        now: datetime,
        edited: dict[str, Any],
    ) -> ValidationError | None:
        """Validate an edit. On failure the request is unchanged and the error names arguments."""
        del now  # an edit is not a decision yet; ``decided_*`` is set by the later approve
        new_hash = hash_arguments(approval.tool, edited)
        error = self._edit_error(s, approval.tool, edited, new_hash)
        review = approval.review
        if error is None:
            review, error = self._edited_review(s, approval.tool, edited)
        if error is not None:
            names = error.details.get("fields") or [
                a["argument"] for a in error.details.get("arguments", [])
            ]
            self._record_decision(s, approval, user, "edit", "failure", new_hash)
            return ValidationError(
                "The edited arguments are invalid; the request is unchanged.",
                {
                    "approval_id": approval.approval_id,
                    "reason": "invalid_edit",
                    "failed_arguments": list(names),
                    "error": error.as_dict(),
                },
            )
        approval.status = ApprovalStatus.EDIT_PROPOSED
        approval.edited_arguments = dict(edited)
        approval.edited_hash = new_hash
        approval.review = review
        self._record_decision(s, approval, user, "edit", "success", new_hash)
        return None

    def _edited_review(
        self, s: Session, tool: str, edited: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, ToolError | None]:
        """The tool's ``approval_view`` for the edited arguments, so the review is not stale."""
        spec = self._registry.get(tool)
        if spec.approval_view is None:
            return None, None
        ctx = ToolContext(
            session=s,
            principal=Principal(user=s.owner),
            services=self._services,
            clock=self._clock,
            inputs=self._inputs,
        )
        try:
            model = spec.input_model.model_validate(edited)
            return dict(spec.approval_view(ctx, model)), None
        except Exception:
            LOGGER.exception("Approval view of tool %s failed for an edit", tool)
            return None, ToolError(
                CODE_INVALID_ARGUMENTS,
                "The content this approval must show could not be loaded for the edit.",
                {"fields": [], "tool": tool, "reason": "approval_view_failed"},
            )

    def _edit_error(
        self, s: Session, tool: str, edited: dict[str, Any], new_hash: str
    ) -> ToolError | None:
        spec = self._registry.get(tool) if self._registry is not None else None
        if spec is None or self._inputs is None:
            return ToolError(
                CODE_INVALID_ARGUMENTS,
                f"Edits cannot be validated for the tool {tool}.",
                {"fields": [], "tool": tool},
            )
        try:
            spec.input_model.model_validate(edited)
        except PydanticValidationError as exc:
            return invalid_arguments_error(exc)
        try:
            checked = self._inputs.check(s, spec, dict(edited), args_hash=new_hash)
        except AuditWriteError:
            LOGGER.error("An input rejection in session %s could not be audited", s.session_id)
            s.audit_failed = True
            return ToolError(CODE_INPUT_REJECTED, "The edited arguments could not be checked.")
        return None if checked.ok else input_rejected_error(checked.rejections)

    @staticmethod
    def _overdue(s: Session, now: datetime) -> bool:
        return any(a.status in _PENDING_APPROVALS and now > a.expires_at for a in s.approvals)

    def _expire_in(self, s: Session, now: datetime) -> int:
        """The expiry sweep, inside a transaction. Returns the number of requests expired."""
        overdue = [
            a for a in s.approvals if a.status in _PENDING_APPROVALS and now > a.expires_at
        ]
        if not overdue:
            return 0
        for approval in overdue:
            approval.status = ApprovalStatus.EXPIRED
            self._add(
                s,
                "approval_decision",
                tool_call_id=approval.tool_call_id,
                approval_id=approval.approval_id,
                data={"tool": approval.tool, "status": ApprovalStatus.EXPIRED.value},
            )
            self._write_audit(
                s,
                AgentAuditRecord(
                    session_id=s.session_id,
                    owner=s.owner,
                    actor=s.owner,
                    seq=0,
                    event_type="approval_expired",
                    at=now,
                    result_status="cancelled",
                    tool=approval.tool,
                    args_hash=approval.args_hash,
                    detail={"approval_id": approval.approval_id},
                ),
            )
        # The session waits for the user, so the other uses of the turn cannot run now.
        for approval in s.approvals:
            if approval.status in _PENDING_APPROVALS:
                approval.status = ApprovalStatus.CANCELLED
                approval.reason = "Another approval request in this turn expired."
        # One pass in request order, so the results keep the order of the uses (Req 2.6).
        expired = {a.tool_call_id for a in overdue}
        for call_id in list(s.pending_tool_use_ids):
            if call_id in expired:
                self._resolve_call(
                    s,
                    call_id,
                    CallState.REFUSED,
                    CODE_APPROVAL_EXPIRED,
                    "Not run: the approval request expired. Propose the step again if it is "
                    "still needed.",
                )
            else:
                self._resolve_call(
                    s,
                    call_id,
                    CallState.DEFERRED,
                    CODE_DEFERRED,
                    "Not run: an approval request in this turn expired. Ask again after the "
                    "user answers.",
                )
        if s.status in (SessionStatus.WAITING_FOR_APPROVAL, SessionStatus.RUNNING):
            self._set_status(s, SessionStatus.WAITING_FOR_USER)
        return len(overdue)

    def _resume_after_decision(self, s: Session) -> None:
        if s.status is SessionStatus.WAITING_FOR_APPROVAL:
            self._set_status(s, SessionStatus.RUNNING)

    def _record_decision(
        self,
        s: Session,
        approval: ApprovalRequest,
        user: str,
        decision: str,
        result_status: str,
        hash_: str,
    ) -> None:
        """Transcript entry and ``approval_decision`` audit record for one decision (Req 11.1)."""
        now = self._clock()
        self._add(
            s,
            "approval_decision",
            tool_call_id=approval.tool_call_id,
            approval_id=approval.approval_id,
            data={
                "tool": approval.tool,
                "decision": decision,
                "status": approval.status.value,
                "args_hash": hash_,
                "result": result_status,
            },
        )
        detail = {"approval_id": approval.approval_id, "status": approval.status.value}
        if approval.batch_id:
            detail["batch_id"] = approval.batch_id
        self._write_audit(
            s,
            AgentAuditRecord(
                session_id=s.session_id,
                owner=s.owner,
                actor=user,
                seq=0,
                event_type="approval_decision",
                at=now,
                result_status=result_status,  # type: ignore[arg-type]
                tool=approval.tool,
                args_hash=hash_,
                decision=decision,
                decided_by=user,
                detail=detail,
            ),
        )

    def _write_audit(self, s: Session, record: AgentAuditRecord) -> None:
        try:
            self._audit.write(s, record)
        except AuditWriteError:
            # Recorded on the session; the loop then runs no further Mutating or Costly tool.
            LOGGER.error("Event %s of session %s could not be audited", record.event_type,
                         s.session_id)
            s.audit_failed = True

    @staticmethod
    def _call(s: Session, call_id: str) -> Any:
        return next((r for r in reversed(s.tool_calls) if r.tool_call_id == call_id), None)

    def _resolve_call(
        self, s: Session, call_id: str, state: CallState, code: str, summary: str
    ) -> bool:
        """Give a not-yet-executing use its single result. An ``executing`` call is left alone."""
        record = self._call(s, call_id)
        if record is None and call_id not in s.pending_tool_use_ids:
            return False
        if record is not None and record.state not in _CANCELLABLE_CALLS:
            return False
        if record is not None:
            record.state = state
            record.error_code = code
            record.result_summary = summary
            record.finished_at = self._clock()
        self._add(
            s,
            "tool_result",
            text=summary,
            tool_call_id=call_id,
            data={
                "tool": record.tool if record is not None else "",
                "state": state.value,
                "error_code": code,
            },
        )
        s.pending_tool_use_ids = [i for i in s.pending_tool_use_ids if i != call_id]
        return True

    # ------------------------------------------------------------------ local uploads
    def register_local_files(
        self, who: Principal, session_id: str, files: list[LocalFileInput | dict[str, Any]]
    ) -> SessionView:
        """Replace the session's manifest of local candidate files (Req 7.1, 7.2).

        The client lists and hashes the files on the user's machine; the server only keeps
        names, sizes, times and hashes. Refused while an upload is awaiting a decision or the
        client, so an approval never shows a file the manifest no longer holds.
        """
        self._require_enabled()
        user = self._user(who)
        if not isinstance(files, list) or len(files) > MAX_LOCAL_FILES:
            raise ValidationError(
                f"Register at most {MAX_LOCAL_FILES} local files.",
                {"field": "files", "max_items": str(MAX_LOCAL_FILES)},
            )
        try:
            parsed = [LocalFileInput.model_validate(f) for f in files]
        except PydanticValidationError as exc:
            fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
            raise ValidationError(
                "Each local file needs a plain file name, a size in bytes, a modification "
                "time and a SHA-256 hash.",
                {"field": "files", "fields": fields},
            ) from None
        with self._owned(session_id, user) as s:
            if s.status in (SessionStatus.COMPLETED, SessionStatus.CANCELLED):
                raise ConflictError(
                    f"Session {session_id} is {s.status.value}.",
                    {"session_id": session_id, "reason": "session_finished"},
                )
            busy = (s.client_action is not None and s.client_action.status == "ready") or any(
                a.kind == "upload" and a.status in _PENDING_APPROVALS for a in s.approvals
            )
            if busy:
                raise ConflictError(
                    "An upload is in progress; finish or cancel it before listing files again.",
                    {"session_id": session_id, "reason": "upload_in_progress"},
                )
            s.local_files = [f.to_ref() for f in parsed]
            self._add(
                s, "status", data={"status": s.status.value, "local_files": len(parsed)}
            )
        return self._view(session_id)

    def client_action(self, who: Principal, session_id: str) -> ClientActionView | None:
        """The upload the client must perform, with a freshly minted pre-signed PUT.

        Only the plain destination location is saved; the signed URL and headers are returned
        and never persisted. A previous, never-completed destination is deleted.
        """
        self._require_enabled()
        user = self._user(who)
        current = self._store.load(session_id).session
        self._check_owner(current, user)
        if current.client_action is None or current.client_action.status != "ready":
            return None
        with self._owned(session_id, user) as s:
            action = s.client_action
            if action is None or action.status != "ready":
                return None
            file = self._local_file(s, action.file_id)
            ticket = self._services.datasets.create_upload_ticket(file.name)
            if action.dataset_uri and action.dataset_uri != ticket.dataset_uri:
                self._discard(action.dataset_uri)
            action.dataset_uri = ticket.dataset_uri
            return ClientActionView(
                action_id=action.action_id,
                kind=action.kind,
                file_id=file.file_id,
                file_name=file.name,
                size_bytes=file.size_bytes,
                sha256=file.sha256,
                upload_url=ticket.upload_url,
                method=ticket.method,
                headers=dict(ticket.headers),
                expires_at=ticket.expires_at,
                max_bytes=ticket.max_bytes,
            )

    def complete_upload(self, who: Principal, session_id: str, action_id: str) -> SessionView:
        """Register the uploaded object and give the waiting tool use its result (Req 7.9, 7.10).

        On any failure the object is deleted, nothing is registered, the action stays
        ``ready`` for a retry of the same file, and the error names the reason.
        """
        self._require_enabled()
        user = self._user(who)
        current = self._store.load(session_id).session
        self._check_owner(current, user)
        action = self._ready_action(current, action_id)
        file = self._local_file(current, action.file_id)
        uri = action.dataset_uri
        if not uri:
            raise ConflictError(
                "No upload was started for this action. Fetch the client action first.",
                {"session_id": session_id, "action_id": action_id, "reason": "not_started"},
            )
        try:
            ref, profile = self._register_upload(uri, file)
        except Exception as exc:
            self._discard(uri)
            error = exc if isinstance(exc, AppError) else ValidationError(
                "The uploaded file could not be registered.", {"reason": type(exc).__name__}
            )
            with self._owned(session_id, user) as s:
                if s.client_action is not None and s.client_action.action_id == action_id:
                    s.client_action.dataset_uri = None  # stays ready: the user can retry
                self._add(
                    s,
                    "error",
                    text=f"The upload of {file.name} failed: {error.message} "
                    "Nothing was registered; you can retry.",
                    data={"category": "upload_failed", "code": error.code,
                          "action_id": action_id},
                )
            raise error from None
        with self._owned(session_id, user) as s:
            try:
                live = self._ready_action(s, action_id)
            except ConflictError:
                self._discard(uri)
                raise
            s.upload_approved_once = True
            if s.status is SessionStatus.WAITING_FOR_USER:
                self._set_status(s, SessionStatus.RUNNING)
            # Resolve while the action is still ``ready``: that is how the loop finds the call.
            self._loop.complete_client_upload(s, ToolOutput(data=profile, datasets=(ref,)))
            live.status = "done"
            live.dataset_uri = None
            running = s.status is SessionStatus.RUNNING
        if running:
            self._kick(session_id)
        return self._view(session_id)

    def _register_upload(self, uri: str, file: LocalFileRef) -> tuple[DatasetRef, dict[str, Any]]:
        """Re-check size and format, register through ``DatasetService`` and count rows."""
        store = self._object_store()
        limit = int(self._settings.max_dataset_upload_bytes)
        metadata = store.stat(uri)
        if metadata is None:
            raise ValidationError(
                "No uploaded object was found. The transfer did not complete.",
                {"reason": "object_missing"},
            )
        if metadata.size_bytes > limit:
            raise ValidationError(
                f"The uploaded file is {metadata.size_bytes} bytes, over the upload limit of "
                f"{limit} bytes.",
                {"reason": "too_large", "size_bytes": metadata.size_bytes, "max_bytes": limit},
            )
        file_format = format_for(file.name)
        if file_format is None:
            raise ValidationError(
                "Supported formats: CSV (.csv) and Parquet (.parquet).",
                {"reason": "unsupported_format", "supported_formats": ["CSV", "Parquet"]},
            )
        datasets = self._services.datasets
        try:
            registered = datasets.register(uri)
        except DatasetParseError as exc:
            raise ValidationError(
                f"The file could not be read as {file_format}. Supported formats: CSV and Parquet.",
                {"reason": "invalid_content", "supported_formats": ["CSV", "Parquet"]},
            ) from exc
        # The S3 client lets the counter read a Parquet footer or stream a CSV, not load it.
        rows = RowCounter(store, getattr(datasets, "_s3", None)).count(uri, file_format)
        ref = DatasetRef(
            dataset_id=dataset_id_for(uri),
            uri=uri,
            filename=file.name,
            sha256=file.sha256.lower(),
            size_bytes=metadata.size_bytes,
            file_format=file_format,
            columns=[ColumnRef(name=c.name, dtype=c.dtype) for c in registered.columns],
            row_count=rows,
        )
        profile = {
            "uploaded": True,
            "dataset_id": ref.dataset_id,
            "filename": ref.filename,
            "file_format": file_format,
            "size_bytes": ref.size_bytes,
            "row_count": rows,
            "column_count": len(registered.columns),
            "columns": [
                {
                    "name": c.name,
                    "dtype": c.dtype,
                    "non_null_in_sample": c.non_null_in_sample,
                    "sample_values": list(c.sample_values),  # capped by governance
                }
                for c in registered.columns
            ],
        }
        return ref, profile

    @staticmethod
    def _ready_action(s: Session, action_id: str) -> ClientAction:
        action = s.client_action
        if action is None or action.action_id != action_id or action.status != "ready":
            raise ConflictError(
                f"Client action {action_id} is not waiting for an upload.",
                {"session_id": s.session_id, "action_id": action_id, "reason": "not_ready"},
            )
        return action

    @staticmethod
    def _local_file(s: Session, file_id: str) -> LocalFileRef:
        file = next((f for f in s.local_files if f.file_id == file_id), None)
        if file is None:
            raise ConflictError(
                "The file of this upload is no longer registered.",
                {"session_id": s.session_id, "file_id": file_id, "reason": "unknown_file"},
            )
        return file

    def _object_store(self) -> Any:
        store = getattr(self._services, "store", None)
        if store is None:
            raise ValidationError("Uploads need an object store; none is configured.")
        return store

    def _discard(self, uri: str) -> None:
        """Delete an upload object that must not stay. A missing object is not an error."""
        try:
            self._object_store().delete(uri)
        except Exception:
            LOGGER.exception("Upload object could not be deleted")

    # ------------------------------------------------------------------ mode and limits
    def set_mode(
        self,
        who: Principal,
        session_id: str,
        mode: Mode | str,
        *,
        max_estimated_spend_usd: float | None = None,
    ) -> SessionView:
        """Switch between Supervised and Autopilot (Req 5.2, 5.3, 5.4, 5.9, 5.10).

        Autopilot needs ``agent_autopilot_allowed`` and a spend cap in ``(0, limit]``; the grant
        is saved before the loop's next gating transaction, which re-reads the mode. Supervised
        clears the grant. A refused request saves nothing.
        """
        self._require_enabled()
        user = self._user(who)
        with self._owned(session_id, user) as s:
            if s.status in (SessionStatus.COMPLETED, SessionStatus.CANCELLED):
                raise ConflictError(
                    f"Session {session_id} is {s.status.value}; its mode cannot be changed.",
                    {"session_id": session_id, "reason": "session_finished",
                     "status": s.status.value},
                )
            now = self._clock()
            chosen, grant = self._autopilot(user, mode, max_estimated_spend_usd, s.limits, now)
            if grant is not None:
                grant = grant.model_copy(
                    update={"spend_at_enable": s.counters.estimated_spend_usd}
                )
            previous = s.mode
            s.mode = chosen
            s.autopilot = grant
            data: dict[str, Any] = {"mode": chosen.value, "previous": previous.value}
            detail = {"mode": chosen.value, "previous": previous.value}
            if grant is not None:
                data["max_estimated_spend_usd"] = grant.max_estimated_spend_usd
                detail["max_estimated_spend_usd"] = f"{grant.max_estimated_spend_usd:g}"
                detail["enabled_at"] = grant.enabled_at.isoformat()
            self._add(s, "mode", data=data)
            self._write_audit(
                s,
                AgentAuditRecord(
                    session_id=s.session_id,
                    owner=s.owner,
                    actor=user,
                    seq=0,
                    event_type="mode_changed",
                    at=now,
                    result_status="success",
                    decided_by=user,
                    detail=detail,
                ),
            )
        return self._view(session_id)

    def continue_after_limit(
        self, who: Principal, session_id: str, limit: LimitName | str, amount: Any
    ) -> SessionView:
        """Raise the reached limit by the confirmed positive amount and resume (Req 9.3, 9.7).

        Applies only while the session waits for the user after a limit stop, and only to the
        limit that stopped it. A missing, zero or negative amount raises ``ValidationError`` and
        leaves the limits and the ``waiting_for_user`` status unchanged.
        """
        self._require_enabled()
        user = self._user(who)
        try:
            name = LimitName(limit)
        except ValueError:
            raise ValidationError(
                f"Unknown session limit {limit!r}.", {"field": "limit"}
            ) from None
        with self._owned(session_id, user) as s:
            reached = self._reached_limit(s)
            if reached is None:
                raise ConflictError(
                    f"Session {session_id} is not stopped at a session limit.",
                    {"session_id": session_id, "reason": "not_at_limit",
                     "status": s.status.value},
                )
            if name.value != reached:
                raise ValidationError(
                    f"Only the reached limit {reached} can be raised (got {name.value}).",
                    {"field": "limit", "limit": reached},
                )
            before = getattr(s.limits, name.value)
            s.limits = raise_limit(s.limits, name, amount)
            after = getattr(s.limits, name.value)
            now = self._clock()
            s.epoch += 1
            s.guard_refusals = []
            self._write_audit(
                s,
                AgentAuditRecord(
                    session_id=s.session_id,
                    owner=s.owner,
                    actor=user,
                    seq=0,
                    event_type="limit_raised",
                    at=now,
                    result_status="success",
                    decided_by=user,
                    detail={
                        "limit": name.value,
                        "amount": f"{amount:g}",
                        "previous": f"{before:g}",
                        "configured": f"{after:g}",
                    },
                ),
            )
            self._set_status(s, SessionStatus.RUNNING)
        self._kick(session_id)
        return self._view(session_id)

    @staticmethod
    def _reached_limit(s: Session) -> str | None:
        """The limit named by the stop the session is waiting on, if it waits on one."""
        if s.status is not SessionStatus.WAITING_FOR_USER:
            return None
        for entry in reversed(s.transcript):
            if entry.kind == "limit_stop":
                reached = entry.data.get("limit")
                return reached if isinstance(reached, str) else None
            if entry.kind == "user_message" or (
                entry.kind == "status" and entry.data.get("status") == SessionStatus.RUNNING.value
            ):
                return None  # the session ran again after the last stop
        return None

    # ------------------------------------------------------------------ cancel
    def cancel(self, who: Principal, session_id: str, *, stop_jobs: bool = False) -> SessionView:
        """Set ``cancelled``, reject pending approvals and give pending uses cancelled results.

        Running awaited jobs keep running unless ``stop_jobs`` is true (the user answered yes,
        Req 6.13); then each is stopped through the orchestrator's stop path and audited.
        """
        self._require_enabled()
        user = self._user(who)
        current = self._store.load(session_id).session
        self._check_owner(current, user)
        if current.status is SessionStatus.CANCELLED:
            return build_view(current)
        with self._owned(session_id, user) as s:
            if s.status is SessionStatus.CANCELLED:
                pass  # cancelled between the load and the lock; saving again is harmless
            elif s.status is SessionStatus.COMPLETED:
                raise ConflictError(
                    f"Session {session_id} is completed and cannot be cancelled.",
                    {"session_id": session_id, "reason": "session_finished"},
                )
            else:
                self._apply_cancel(s, user, stop_jobs=stop_jobs)
        return self._view(session_id)

    def _stop_jobs(self, s: Session, user: str) -> None:
        """Stop every awaited job that has not ended, one ``job_stopped`` audit record each."""
        stop = getattr(self._orchestrator, "stop", None)
        for job in s.awaited_jobs:
            if job.state in TERMINAL_JOB_STATES:
                continue
            result = "success"
            detail = {"execution_id": job.execution_id, "experiment_id": job.experiment_id}
            if not callable(stop):
                result = "failure"
                detail["reason"] = "no_stop_path"
            else:
                try:
                    stop(job.execution_id)
                except Exception as exc:  # one failed stop must not leave the others running
                    LOGGER.warning("Could not stop job %s: %s", job.execution_id,
                                   type(exc).__name__)
                    result = "failure"
                    detail["reason"] = type(exc).__name__
            self._write_audit(
                s,
                AgentAuditRecord(
                    session_id=s.session_id,
                    owner=s.owner,
                    actor=user,
                    seq=0,
                    event_type="job_stopped",
                    at=self._clock(),
                    result_status=result,  # type: ignore[arg-type]
                    decided_by=user,
                    detail=detail,
                ),
            )

    def _apply_cancel(self, s: Session, user: str, *, stop_jobs: bool = False) -> None:
        now = self._clock()
        s.epoch += 1  # an in-flight model result is discarded by the loop
        rejected = 0
        for approval in s.approvals:
            if approval.status in _PENDING_APPROVALS:
                approval.status = ApprovalStatus.REJECTED
                approval.decided_by = user
                approval.decided_at = now
                approval.reason = "The session was cancelled."
                rejected += 1
        action = s.client_action
        if action is not None and action.status == "ready":
            # The upload call waits for the client, not for a handler: nothing ran on the
            # server, so it is cancelled like a requested call and its destination removed.
            record = self._call(s, action.tool_call_id or "")
            if record is not None and record.state is CallState.EXECUTING:
                record.state = CallState.REQUESTED
            if action.dataset_uri:
                self._discard(action.dataset_uri)
            s.client_action = None
        # An ``executing`` call stays pending: the loop records its real outcome.
        for call_id in list(s.pending_tool_use_ids):
            self._resolve_call(
                s,
                call_id,
                CallState.CANCELLED,
                CODE_CANCELLED,
                "The session was cancelled before this call ran.",
            )
        running = sum(1 for j in s.awaited_jobs if j.state not in TERMINAL_JOB_STATES)
        if stop_jobs:
            self._stop_jobs(s, user)
        self._set_status(s, SessionStatus.CANCELLED)
        record = AgentAuditRecord(
            session_id=s.session_id,
            owner=s.owner,
            actor=user,
            seq=0,
            event_type="cancelled",
            at=now,
            result_status="cancelled",
            detail={
                "stop_jobs": "true" if stop_jobs else "false",
                "running_jobs": str(running),
                "approvals_rejected": str(rejected),
            },
        )
        try:
            self._audit.write(s, record)
        except AuditWriteError:
            # A cancel must not be blocked by auditing; the gap is recorded on the session.
            LOGGER.error("Cancel of session %s could not be audited", s.session_id)
            s.audit_failed = True

    # ------------------------------------------------------------------ checks
    def _require_enabled(self) -> None:
        if not bool(getattr(self._settings, "agent_enabled", False)):
            raise FeatureDisabledError(
                f"The ML Assistant is disabled. Set {ENABLED_SETTING}=true to enable it.",
                {"setting": ENABLED_SETTING},
            )

    @staticmethod
    def _user(who: Principal | None) -> str:
        user = getattr(who, "user", None)
        if not isinstance(user, str) or not user:
            raise AuthorizationError("An authenticated user is required.")
        return user

    @staticmethod
    def _check_owner(session: Session, user: str) -> None:
        if session.owner != user:
            raise AuthorizationError(
                f"Session {session.session_id} belongs to another user.",
                {"session_id": session.session_id},
            )

    @contextmanager
    def _owned(self, session_id: str, user: str) -> Iterator[Session]:
        """A store transaction whose body runs only for the owner. A raise saves nothing."""
        with self._store.transaction(session_id) as s:
            self._check_owner(s, user)
            yield s

    def _in_use(self, s: Session) -> bool:
        lease = s.lease
        if lease is not None and lease.expires_at > self._clock():
            return True
        is_active = getattr(self._runner, "is_active", None)
        return bool(callable(is_active) and is_active(s.session_id))

    @staticmethod
    def _check_resumable(s: Session) -> None:
        sid = s.session_id
        status = s.status
        if status in (SessionStatus.COMPLETED, SessionStatus.CANCELLED):
            raise ConflictError(
                f"Session {sid} is {status.value} and cannot be resumed.",
                {"session_id": sid, "reason": "session_finished", "status": status.value},
            )
        if status is SessionStatus.FAILED and not s.retriable:
            raise ConflictError(
                f"Session {sid} failed and cannot be resumed.",
                {"session_id": sid, "reason": "not_retriable", "status": status.value},
            )
        if status is SessionStatus.WAITING_FOR_APPROVAL:
            raise ConflictError(
                f"Session {sid} waits for an approval decision.",
                {"session_id": sid, "reason": "approval_pending", "status": status.value},
            )
        if status is SessionStatus.WAITING_FOR_USER:
            raise ConflictError(
                f"Session {sid} waits for the user; send a message to continue.",
                {"session_id": sid, "reason": "waiting_for_user", "status": status.value},
            )

    def _autopilot(
        self,
        user: str,
        mode: Mode | str,
        max_spend: float | None,
        limits: SessionLimits,
        now: datetime,
    ) -> tuple[Mode, AutopilotGrant | None]:
        """Autopilot only on an explicit request with a spend cap in (0, limit] (Req 5.3, 5.4)."""
        try:
            chosen = Mode(mode)
        except ValueError:
            raise ValidationError(
                "The mode must be 'supervised' or 'autopilot'.", {"field": "mode"}
            ) from None
        if chosen is Mode.SUPERVISED:
            if max_spend is not None:
                raise ValidationError(
                    "A maximum Autopilot spend applies only when the mode is 'autopilot'.",
                    {"field": "autopilot_max_spend_usd"},
                )
            return chosen, None
        if not bool(getattr(self._settings, "agent_autopilot_allowed", False)):
            raise FeatureDisabledError(
                f"Autopilot is not allowed in this deployment ({AUTOPILOT_SETTING}).",
                {"setting": AUTOPILOT_SETTING},
            )
        ceiling = limits.max_estimated_spend_usd
        valid = (
            isinstance(max_spend, (int, float))
            and not isinstance(max_spend, bool)
            and math.isfinite(max_spend)
            and 0 < max_spend <= ceiling
        )
        if not valid:
            raise ValidationError(
                "Autopilot needs a maximum estimated spend greater than 0 and at most "
                f"{ceiling:g} USD (got {max_spend!r}).",
                {"field": "autopilot_max_spend_usd", "max": f"{ceiling:g}"},
            )
        grant = AutopilotGrant(
            enabled_by=user,
            enabled_at=now.astimezone(UTC).replace(microsecond=0),
            max_estimated_spend_usd=float(max_spend),  # type: ignore[arg-type]
            spend_at_enable=0.0,
        )
        return chosen, grant

    # ------------------------------------------------------------------ helpers
    def _kick(self, session_id: str) -> None:
        try:
            self._runner.kick(session_id)
        except Exception:
            # The session stays ``running`` without a lease, so ``resume`` restarts it.
            LOGGER.exception("Runner kick for session %s failed", session_id)

    def _view(self, session_id: str) -> SessionView:
        return build_view(self._store.load(session_id).session)

    def _add(self, s: Session, kind: str, **kwargs: Any) -> TranscriptEntry:
        entry = TranscriptEntry(seq=s.next_seq, at=self._clock(), kind=kind, **kwargs)  # type: ignore[arg-type]
        s.transcript.append(entry)
        s.next_seq += 1
        return entry

    def _set_status(self, s: Session, status: SessionStatus) -> None:
        if s.status is status:
            return
        s.status = status
        s.last_activity_at = self._clock()
        self._add(s, "status", data={"status": status.value})

    def _fail(self, s: Session, category: str, message: str) -> None:
        now = self._clock()
        s.failure = SessionFailure(category=category, message=message, retriable=False, at=now)
        s.retriable = False
        self._add(s, "error", text=message, data={"category": category, "retriable": False})
        self._set_status(s, SessionStatus.FAILED)
        self._write_audit(
            s,
            AgentAuditRecord(
                session_id=s.session_id,
                owner=s.owner,
                actor=s.owner,
                seq=0,
                event_type="session_failed",
                at=now,
                result_status="failure",
                detail={"category": category, "retriable": "false"},
            ),
        )
