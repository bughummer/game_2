"""The agent loop (design C13; Req 1.3 to 1.8, 2.3 to 2.6, 2.9, 2.10, 6.2, 6.14, 13.8, 14.9).

``AgentLoop.step`` does one unit of work chosen by the persisted session and returns a
``StepOutcome``. ``AgentLoop.run`` takes the lease and calls ``step`` until it pauses.

Every step is a short transaction: load, one change, save. Model calls and tool handlers run
outside a transaction, so a session is never locked across network or service time. After a
model call, the result is applied only if the session epoch is unchanged and the status is still
``running``, so a cancel (or any user action that bumps the epoch) wins over an in-flight call.

Task 8.6 adds gating (design C6), limits (C7), cost estimates (C12), approval execution and the
Costly path. The hand-off with ``AgentService.decide_approval``: the service only records the
decision and sets ``running``; the loop executes a call in ``awaiting_approval`` only when its
approval is ``approved`` and ``approved_hash`` equals the call's ``args_hash``, then marks the
approval ``executed``. Approval expiry runs at the top of every step through the service's own
sweep. A Costly call gets an idempotency key and an in-flight marker before its handler runs.

Task 8.7 adds job waiting: jobs a handler started (``ToolOutput.awaited`` and the tool's
``job_extractor``) are recorded as awaited jobs, the session moves to ``waiting_for_job`` and
the loop returns without a model call. When every awaited job has ended, the next step delivers
one ``job_update`` per job (enveloped by ``context.py``) before the model is called. Recovery
reconciles each Costly call left ``executing`` against the platform's execution records and
never runs it again (design D11). Active time accrues only while the status is ``running``.

The final answer's narration is checked against a number ledger rebuilt from the session's
stored full results (design C9). Not here yet: per-call audit records.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from backend.agent.context import build_messages
from backend.agent.converse import (
    ConverseClient,
    ConverseCredentialsError,
    ConverseError,
    ConverseModelUnsupportedError,
    ConverseTransientError,
    ConverseTurn,
)
from backend.agent.estimate import NO_ESTIMATE, Estimate
from backend.agent.execute import (
    ExecutionResult,
    ToolError,
    ToolExecutor,
    input_rejected_error,
    invalid_arguments_error,
)
from backend.agent.inputs import ToolInputs
from backend.agent.jobs import (
    TERMINAL_JOB_STATES,
    Submission,
    apply_observations,
    find_submissions,
    observe_jobs,
    run_bounded,
)
from backend.agent.limits import (
    CODE_LIMIT_REACHED,
    LimitName,
    LimitStop,
    accrue_active,
    check_before_model_call,
    check_before_tool_call,
    check_loop_top,
    register_call,
    would_exceed_costly,
)
from backend.agent.narration import (
    NumberLedger,
    Policy,
    ToolResult,
    VerifiedNarration,
    build_final_answer,
    verify_narration,
)
from backend.agent.policy import GateAction, GateInput, gate
from backend.agent.prompts import system_prompt
from backend.agent.results import ScrubContext, _ledger, shape_result
from backend.agent.service import CODE_APPROVAL_EXPIRED, CODE_APPROVAL_REJECTED, AgentService
from backend.agent.store import (
    SessionConflict,
    SessionInUse,
    SessionLoadError,
    SessionPersistError,
    SessionStore,
)
from backend.agent.tools import (
    RESERVED_PRIVILEGED_PREFIXES,
    ToolContext,
    ToolFlags,
    ToolOutput,
    ToolRegistry,
    ToolSpec,
)
from backend.agent.tools_uploads import CODE_CHOICE_REQUIRED
from backend.audit import AuditWriteError
from backend.auth import Principal
from backend.governance import GovernancePolicy
from ml_engine.contracts.agent import (
    AgentAuditRecord,
    ApprovalRequest,
    ApprovalStatus,
    AwaitedJob,
    CallState,
    ClientAction,
    GuardRefusal,
    Mode,
    Plan,
    PlanStep,
    Session,
    SessionFailure,
    SessionStatus,
    ToolCallRecord,
    ToolClass,
    TranscriptEntry,
    args_hash,
    idempotency_key,
)

LOGGER = logging.getLogger("ml_factory.agent.loop")

SUBMIT_PLAN = "submit_plan"
ASK_USER = "ask_user"

# Structured tool-result codes the model receives.
CODE_UNKNOWN_TOOL = "UNKNOWN_TOOL"
CODE_PLAN_REQUIRED = "PLAN_REQUIRED"
CODE_DEFERRED = "DEFERRED_WAITING_FOR_USER"
CODE_AUDIT_FAILED = "AUDIT_FAILED"
CODE_REFUSED = "REFUSED"
CODE_UNKNOWN_OUTCOME = "UNKNOWN_OUTCOME"
CODE_REPEATED_CALL = "REPEATED_CALL"
CODE_APPROVAL_MISMATCH = "APPROVAL_MISMATCH"
CODE_JOB_PENDING = "PREREQUISITE_PENDING"
CODE_DUPLICATE_CALL = "DUPLICATE_CALL"
CODE_NOT_SUBMITTED = "NOT_SUBMITTED"
CODE_JOB_NOT_FOUND = "JOB_NOT_FOUND"
CODE_JOB_FAILED = "JOB_FAILED"
CODE_APPROVAL_VIEW_FAILED = "APPROVAL_VIEW_FAILED"

# Session failure categories.
FAIL_PLANNING = "planning_failed"
FAIL_BEDROCK = "bedrock_unavailable"
FAIL_MODEL_UNSUPPORTED = "model_unsupported"
FAIL_CREDENTIALS = "credentials_expired"
FAIL_MODEL_REQUEST = "model_request_failed"
FAIL_MAX_TOKENS = "max_tokens"

_TERMINAL = frozenset({SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED})
_MAX_RATIONALE = 500
_MAX_SUMMARY = 200
#: Most approval requests open at once for one turn (the batch size, Req 4.13).
_MAX_OPEN_APPROVALS = 10
_UNDECIDED = (ApprovalStatus.PENDING, ApprovalStatus.EDIT_PROPOSED)


class StepOutcome(StrEnum):
    CONTINUE = "continue"
    PAUSED = "paused"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class LoopSettings:
    max_output_tokens: int = 2048
    context_budget_tokens: int = 100_000
    tool_result_max_bytes: int = 8192
    untrusted_string_max_chars: int = 256
    approval_ttl_seconds: int = 900
    lease_seconds: int = 300
    #: Longest the reconciler may spend querying the platform for one in-flight call (Req 14.6).
    reconcile_timeout_seconds: float = 30.0
    #: Unverified narration sentences are marked (default) or withheld (Req 10.3).
    narration_policy: Policy = "mark"

    @classmethod
    def from_settings(cls, settings: Any) -> LoopSettings:
        return cls(
            max_output_tokens=settings.agent_max_output_tokens,
            context_budget_tokens=settings.agent_context_budget_tokens,
            tool_result_max_bytes=settings.agent_tool_result_max_bytes,
            untrusted_string_max_chars=settings.agent_untrusted_string_max_chars,
            approval_ttl_seconds=settings.agent_approval_ttl_seconds,
            lease_seconds=settings.agent_lease_seconds,
        )


@dataclass(frozen=True, slots=True)
class _NamedTool:
    """Stands in for a ``ToolSpec`` when shaping an error for an unknown tool."""

    name: str
    flags: ToolFlags = field(default_factory=lambda: ToolFlags(ledger_source=False))


class _ApprovalViewFailed(Exception):
    """A tool's ``approval_view`` hook failed, so no approval request was created."""


@dataclass(slots=True)
class _Execute:
    """A gated call that must run its handler outside the transaction."""

    call_id: str
    spec: ToolSpec
    snapshot: Session
    approval_id: str | None = None
    experiment_id: str | None = None


@dataclass(frozen=True, slots=True)
class _Reconciled:
    """What the reconciler found for one in-flight Costly call."""

    outcome: str  # "adopt", "not_submitted" or "unknown"
    submission: Submission | None = None


def _utcnow() -> datetime:
    return datetime.now(UTC)


class AgentLoop:
    def __init__(
        self,
        *,
        store: SessionStore,
        converse: ConverseClient,
        registry: ToolRegistry,
        executor: ToolExecutor,
        inputs: ToolInputs,
        audit: Any,
        governance: GovernancePolicy,
        settings: LoopSettings | None = None,
        services: Any = None,
        clock: Callable[[], datetime] = _utcnow,
        scrub_context: Callable[[Session], ScrubContext] | None = None,
        estimator: Any = None,
        on_waiting_for_job: Callable[[str], None] | None = None,
    ) -> None:
        self._estimator = estimator
        #: Told the session id whenever a session starts waiting for a job (the job watcher).
        self.on_waiting_for_job = on_waiting_for_job
        #: When active time was last accrued, per session this process is running.
        self._active_marks: dict[str, datetime] = {}
        # The service's approval expiry sweep, reused so expiry has one implementation.
        self._expiry = AgentService(
            settings=None, store=store, runner=None, audit=audit, clock=clock
        )
        self._store = store
        self._converse = converse
        self._registry = registry
        self._executor = executor
        self._inputs = inputs
        self._audit = audit
        self._governance = governance
        self._settings = settings or LoopSettings()
        self._services = services
        self._clock = clock
        self._scrub_context = scrub_context

    # ================================================================== run / step
    def run(self, session_id: str, holder: str) -> None:
        """Take the lease, recover stale calls, then step until the session pauses."""
        try:
            self._store.acquire_lease(session_id, holder, self._settings.lease_seconds)
        except SessionInUse:
            LOGGER.info("Session %s is in use; not running", session_id)
            return
        except SessionPersistError:
            LOGGER.error("Session %s lease could not be saved; loop not started", session_id)
            return
        self._active_marks[session_id] = self._clock()
        try:
            self._recover(session_id)
            while self.step(session_id) is StepOutcome.CONTINUE:
                pass
        except SessionPersistError:
            LOGGER.error("Session %s could not be saved; loop stopped", session_id)
        except SessionConflict:
            LOGGER.warning("Session %s changed under the loop; loop stopped", session_id)
        finally:
            self._active_marks.pop(session_id, None)
            try:
                self._store.release_lease(session_id, holder)
            except Exception:
                LOGGER.warning("Lease on session %s could not be released", session_id)

    def step(self, session_id: str) -> StepOutcome:
        session = self._store.load(session_id).session
        now = self._clock()
        if session.status not in _TERMINAL and any(
            a.status in _UNDECIDED and now > a.expires_at for a in session.approvals
        ):
            # Approval expiry at the next step (Req 4.12), by the service's sweep.
            self._expiry.expire_approvals(session_id, now)
            session = self._store.load(session_id).session
        if session.status in _TERMINAL:
            return StepOutcome.DONE
        if session.status is not SessionStatus.RUNNING:
            return StepOutcome.PAUSED
        if session.pending_tool_use_ids:
            return self._process_pending(session_id)
        if _unresolved_jobs(session):
            # Read job states outside the transaction; the watcher may not have run yet.
            analysis = getattr(self._services, "analysis", None)
            observed = observe_jobs(session, analysis) if analysis is not None else []
            with self._store.transaction(session_id) as s:
                if s.status is not SessionStatus.RUNNING:
                    return StepOutcome.PAUSED
                apply_observations(s, observed, self._clock())
                if _unresolved_jobs(s):
                    self._set_status(s, SessionStatus.WAITING_FOR_JOB)
                    return StepOutcome.PAUSED
            return StepOutcome.CONTINUE
        if _undelivered_jobs(session):
            with self._store.transaction(session_id) as s:
                if s.status is SessionStatus.RUNNING and not _unresolved_jobs(s):
                    self._deliver_jobs(s)
            return StepOutcome.CONTINUE
        return self._model_step(session_id, session)

    # ================================================================== recovery
    def _recover(self, session_id: str) -> None:
        """Resolve calls left ``executing`` by a stopped process.

        A Read call is safe to run again, so it goes back to ``requested``. Nothing else is ever
        run again. A Costly call is reconciled against the platform's execution records (design
        D11, Req 14.5, 14.6): exactly one match is adopted as its result, no match gives
        ``NOT_SUBMITTED`` (a new decision is needed), and a failed or slow query or several
        matches give ``UNKNOWN_OUTCOME``. A Mutating call has no marker, so its outcome is
        unknown. The Costly counters were added when the call started and are not added again.
        """
        session = self._store.load(session_id).session
        executing = [r for r in session.tool_calls if r.state is CallState.EXECUTING]
        if not executing:
            return
        # The platform is queried outside the transaction, each query bounded in time.
        found = {
            r.tool_call_id: self._reconcile(session, r)
            for r in executing
            if r.tool_class is ToolClass.COSTLY
        }
        with self._store.transaction(session_id) as s:
            waiting = _client_upload_call(s)
            for record in s.tool_calls:
                if record.state is not CallState.EXECUTING or record.tool_call_id == waiting:
                    continue  # an upload waiting for the client is not a stale call
                if record.tool_class is ToolClass.READ:
                    record.state = CallState.REQUESTED
                    record.started_at = None
                    continue
                spec = self._registry.get(record.tool)
                result = found.get(record.tool_call_id, _Reconciled("unknown"))
                if result.outcome == "adopt" and result.submission is not None:
                    self._adopt(s, record, spec, result.submission)
                elif result.outcome == "not_submitted":
                    self._not_submitted(s, record, spec)
                else:
                    self._unknown_outcome(s, record, spec)
            self._after_resolution(s)

    def _reconcile(self, session: Session, record: ToolCallRecord) -> _Reconciled:
        """Look for the job a crashed Costly call may have started. Reads only, never raises."""
        key = record.idempotency_key
        if not key:
            return _Reconciled("unknown")
        exclude = frozenset(j.execution_id for j in session.awaited_jobs)
        dataset_uris = _dataset_uris(session, record.arguments)
        name = record.arguments.get("name")

        def query() -> list[Submission]:
            marker = self._store.get_inflight(session.session_id, key)
            if marker is None:
                raise LookupError("no in-flight marker")
            return find_submissions(
                self._services,
                owner=session.owner,
                tool=record.tool,
                marker=marker,
                dataset_uris=dataset_uris,
                exclude=exclude,
                name=name if isinstance(name, str) else None,
            )

        try:
            matches = run_bounded(query, self._settings.reconcile_timeout_seconds)
        except Exception:
            LOGGER.warning(
                "In-flight call %s of session %s could not be reconciled",
                record.tool_call_id,
                session.session_id,
            )
            return _Reconciled("unknown")
        if len(matches) == 1:
            return _Reconciled("adopt", matches[0])
        if not matches:
            return _Reconciled("not_submitted")
        return _Reconciled("unknown")

    def _adopt(
        self, s: Session, record: ToolCallRecord, spec: ToolSpec | None, found: Submission
    ) -> None:
        """The one matching record becomes the call's result; its job is awaited."""
        approval = _approval(s, record.approval_id)
        if approval is not None:
            approval.status = ApprovalStatus.EXECUTED
        if found.experiment_id not in s.experiment_ids:
            s.experiment_ids.append(found.experiment_id)
        if found.execution_id and all(j.execution_id != found.execution_id for j in s.awaited_jobs):
            s.awaited_jobs.append(
                AwaitedJob(
                    execution_id=found.execution_id,
                    experiment_id=found.experiment_id,
                    tool_call_id=record.tool_call_id,
                    kind=found.kind,
                )
            )
            s.jobs_ready = False
        data = {
            "recovered": True,
            "experiment_id": found.experiment_id,
            "execution_id": found.execution_id,
            "kind": found.kind,
            "message": "The call was submitted before the process stopped; its job is tracked.",
        }
        self._resolve(
            s,
            record,
            spec,
            data=data,
            state=CallState.SUCCEEDED,
            experiment_id=found.experiment_id,
        )

    def _not_submitted(self, s: Session, record: ToolCallRecord, spec: ToolSpec | None) -> None:
        approval = _approval(s, record.approval_id)
        if approval is not None and approval.status is ApprovalStatus.APPROVED:
            approval.status = ApprovalStatus.CANCELLED
            approval.reason = "Not submitted before the process stopped; a new decision is needed."
        error = ToolError(
            CODE_NOT_SUBMITTED,
            "The call was not submitted before the process stopped and was not run again. "
            "Request it again; it needs a new decision.",
            {"reason": "not_submitted"},
        )
        self._resolve(s, record, spec, error=error, state=CallState.FAILED)

    def _unknown_outcome(self, s: Session, record: ToolCallRecord, spec: ToolSpec | None) -> None:
        message = "The outcome of this call is unknown. It will not be run again."
        error = ToolError(CODE_UNKNOWN_OUTCOME, message, {"reason": "outcome_unknown"})
        self._resolve(s, record, spec, error=error, state=CallState.UNKNOWN_OUTCOME)
        self._add(
            s,
            "error",
            text=f"{record.tool}: {message}",
            tool_call_id=record.tool_call_id,
            data={"category": "outcome_unknown"},
        )

    # ================================================================== awaited jobs
    def _deliver_jobs(self, s: Session) -> None:
        """One ``job_update`` per ended, undelivered job (Req 6.6, 14.1, 14.11)."""
        for job in s.awaited_jobs:
            if job.delivered or job.state not in TERMINAL_JOB_STATES:
                continue
            data: dict[str, Any] = {
                "execution_id": job.execution_id,
                "experiment_id": job.experiment_id,
                "kind": job.kind,
                "state": job.state,
            }
            if job.failure_reason:
                data["failure_reason"] = job.failure_reason[:_MAX_RATIONALE]
            if job.state == "not_found":
                data["error_code"] = CODE_JOB_NOT_FOUND
            elif job.state == "failed":
                data["error_code"] = CODE_JOB_FAILED
            self._add(
                s,
                "job_update",
                text=f"Job {job.execution_id} {job.state.replace('_', ' ')}.",
                tool_call_id=job.tool_call_id,
                data=data,
            )
            job.delivered = True
        s.jobs_ready = False

    # ================================================================== tool uses
    def _process_pending(self, session_id: str) -> StepOutcome:
        with self._store.transaction(session_id) as s:
            if s.status is not SessionStatus.RUNNING or not s.pending_tool_use_ids:
                return StepOutcome.CONTINUE
            action = self._decide_next(s)
        if not isinstance(action, _Execute):
            return action

        result = self._run_handler(action)
        with self._store.transaction(session_id) as s:
            record = _record(s, action.call_id)
            if record is None or record.state is not CallState.EXECUTING:
                LOGGER.info("Result for call %s discarded: call already resolved", action.call_id)
                return StepOutcome.CONTINUE
            audit_failed_before = s.audit_failed
            self._accrue(s)  # handler time is running time
            s.audit_seq = max(s.audit_seq, action.snapshot.audit_seq)
            if action.snapshot.audit_failed:
                s.audit_failed = True
            approval = _approval(s, action.approval_id)
            if approval is not None:
                approval.status = ApprovalStatus.EXECUTED
            if result.error_kind == "guard" and result.error is not None:
                # Recorded so Autopilot cannot retry this tool for the experiment (Req 5.6).
                s.guard_refusals.append(
                    GuardRefusal(
                        tool=action.spec.name,
                        experiment_id=action.experiment_id,
                        code=result.error.code,
                    )
                )
            if result.output is not None and result.output.client_upload is not None:
                # The call stays pending until the client uploads (AgentService.complete_upload).
                self._await_client_upload(s, record, action, result.output.client_upload)
                self._audit_executed(s, record, action, approval, ok=True)
                return StepOutcome.PAUSED
            self._record_execution(s, record, action.spec, result)
            if action.spec.tool_class is not ToolClass.READ:
                self._audit_executed(s, record, action, approval, ok=result.output is not None)
            if s.audit_failed and not audit_failed_before:
                # Tell the owner once; later Mutating and Costly calls are refused (Req 11.9).
                self._add(
                    s,
                    "error",
                    text="Auditing failed, so no further changes or costly steps will run in "
                    "this session.",
                    tool_call_id=record.tool_call_id,
                    data={"category": "audit_failed", "code": CODE_AUDIT_FAILED},
                )
            return self._after_resolution(s)

    def _audit_executed(
        self,
        s: Session,
        record: ToolCallRecord,
        action: _Execute,
        approval: ApprovalRequest | None,
        *,
        ok: bool,
    ) -> None:
        """One ``tool_executed`` record for a Mutating or Costly execution (Req 11.1-11.3)."""
        decision: str | None = None
        decided_by: str | None = None
        grant = action.snapshot.autopilot
        if approval is not None:
            decision, decided_by = "approve", approval.decided_by
        elif action.snapshot.mode is Mode.AUTOPILOT and grant is not None:
            decision, decided_by = "approve", f"autopilot:{grant.enabled_by}"
        try:
            self._audit.write(
                s,
                AgentAuditRecord(
                    session_id=s.session_id,
                    owner=s.owner,
                    actor=s.owner,
                    seq=0,
                    event_type="tool_executed",
                    at=self._clock(),
                    result_status="success" if ok else "failure",
                    tool=record.tool,
                    args_hash=record.args_hash,
                    decision=decision,
                    decided_by=decided_by,
                    service_audit_ref=record.service_audit_ref,
                    prompt_injection_suspected=record.injection_suspected,
                    detail={"tool_call_id": record.tool_call_id, "state": record.state.value},
                ),
            )
        except AuditWriteError:
            LOGGER.error("tool_executed audit write failed for %s", s.session_id)
            s.audit_failed = True

    def _run_handler(self, action: _Execute) -> ExecutionResult:
        record = _record(action.snapshot, action.call_id)
        assert record is not None
        try:
            return self._executor.execute(
                action.snapshot, action.spec, dict(record.arguments), args_hash=record.args_hash
            )
        except AuditWriteError:
            action.snapshot.audit_failed = True
            return ExecutionResult(
                error=ToolError(CODE_AUDIT_FAILED, "An audit record could not be written."),
                audit_failed=True,
            )

    def _decide_next(self, s: Session) -> StepOutcome | _Execute:
        """Resolve the next pending use, or prepare it for execution. Runs inside a transaction.

        The mode, limits and approvals are read here, inside the gating transaction.
        """
        call_id = s.pending_tool_use_ids[0]
        record = _record(s, call_id)
        if record is None:  # defensive: an id without a record cannot get a result
            s.pending_tool_use_ids = s.pending_tool_use_ids[1:]
            return StepOutcome.CONTINUE
        if record.state is CallState.AWAITING_APPROVAL:
            return self._approved_or_wait(s, record)
        if record.state is CallState.EXECUTING:
            return StepOutcome.PAUSED  # in flight; only recovery may resolve it
        if record.state is not CallState.REQUESTED:  # defensive: already has its result
            s.pending_tool_use_ids = [i for i in s.pending_tool_use_ids if i != call_id]
            return StepOutcome.CONTINUE

        # A pre-created request the user rejected (or that expired) is final, whatever the
        # mode is now: the call never runs (Req 4.8).
        approval = _approval(s, record.approval_id)
        if approval is not None and approval.status in (
            ApprovalStatus.REJECTED,
            ApprovalStatus.EXPIRED,
        ):
            return self._approved_or_wait(s, record)

        spec = self._registry.get(record.tool)

        # 1. Tool-call limit: the count never passes it (Req 9.2). Not counted as a call.
        stop = check_before_tool_call(s.counters, s.limits)
        if stop is not None:
            return self._refuse(s, record, spec, _limit_error(stop), CallState.REFUSED)

        # 2. Count the use and apply the repeated-call rule (Req 9.4).
        counters, repeated = register_call(s.counters, record.args_hash, s.limits)
        s.counters = counters
        if repeated:
            error = ToolError(
                CODE_REPEATED_CALL,
                "Not run: the same tool was requested with identical arguments too many times "
                "in a row.",
                {
                    "limit": LimitName.REPEATS.value,
                    "configured": s.limits.max_repeated_identical_calls,
                    "current": counters.consecutive_identical,
                },
            )
            return self._refuse(s, record, spec, error, CallState.REFUSED)

        # 3. Unknown tool.
        if spec is None:
            if record.tool.startswith(RESERVED_PRIVILEGED_PREFIXES):
                self._audit_reserved(s, record)
            names = sorted(self._registry.names())
            error = ToolError(
                CODE_UNKNOWN_TOOL,
                f"The tool {record.tool!r} is not registered.",
                {"tool": record.tool, "registered_tools": names},
            )
            return self._refuse(s, record, None, error, CallState.FAILED)

        # 4. A clarifying question defers every side effect in its turn (Req 1.5).
        if spec.tool_class is not ToolClass.READ and _turn_has(s, record.turn, ASK_USER):
            error = ToolError(
                CODE_DEFERRED,
                "Not run: a question to the user is open in this turn. Ask again after "
                "the user answers.",
                {"reason": "deferred_waiting_for_user"},
            )
            return self._refuse(s, record, spec, error, CallState.DEFERRED)

        # 5. Validate the arguments against the input model (Req 2.3, 2.4).
        try:
            model = spec.input_model.model_validate(record.arguments)
        except PydanticValidationError as exc:
            return self._refuse(s, record, spec, invalid_arguments_error(exc), CallState.FAILED)

        # 6. No further side effects after a failed audit write (Req 11.9).
        refusal = self._side_effect_refusal(s, spec, model)
        if refusal is not None:
            return self._refuse(s, record, spec, refusal, CallState.REFUSED)

        # 7. Input checks for anything with a side effect (Read calls are checked when run).
        if spec.tool_class is not ToolClass.READ:
            try:
                checked = self._inputs.check(
                    s, spec, dict(record.arguments), args_hash=record.args_hash
                )
            except AuditWriteError:
                s.audit_failed = True
                error = ToolError(CODE_AUDIT_FAILED, "An audit record could not be written.")
                return self._refuse(s, record, spec, error, CallState.FAILED)
            if not checked.ok:
                error = input_rejected_error(checked.rejections)
                return self._refuse(s, record, spec, error, CallState.FAILED)

        # 8. Estimate, then the Costly caps, which no approval can lift (Req 9.8, 14.3).
        experiment_id = _experiment_of(spec, model)
        estimate = self._estimate(s, spec, model)
        record.estimate_usd = estimate.usd
        if spec.tool_class is ToolClass.COSTLY:
            error = _costly_cap_error(s, estimate.usd)
            if error is not None:
                return self._refuse(s, record, spec, error, CallState.REFUSED)

        # 9. Tool checks that decide before any approval exists (uploads: refuse, reuse, ask).
        if spec.pre_gate is not None:
            try:
                early = spec.pre_gate(self._context(s), model)
            except Exception:
                LOGGER.exception("Pre-gate check of tool %s failed", spec.name)
                early = ToolError(CODE_REFUSED, "Not run: the call could not be checked.")
            if isinstance(early, ToolOutput):
                self._record_execution(s, record, spec, ExecutionResult(output=early))
                return self._after_resolution(s)
            if isinstance(early, ToolError):
                state = (
                    CallState.DEFERRED if early.code == CODE_CHOICE_REQUIRED else CallState.REFUSED
                )
                return self._refuse(s, record, spec, early, state)

        # 10. Gate.
        decision = gate(
            GateInput(
                spec=spec,
                mode=s.mode,
                plan_accepted=s.plan is not None,
                protected=self._protected(s, spec, model),
                estimate_usd=estimate.usd,
                session_spend_usd=s.counters.estimated_spend_usd,
                autopilot_max_spend_usd=(
                    s.autopilot.max_estimated_spend_usd if s.autopilot is not None else None
                ),
                guard_refused_before=_guard_refused(s, spec.name, experiment_id),
                upload_approved_before=s.upload_approved_once,
            )
        )
        if decision.action is GateAction.REFUSE:
            code = CODE_PLAN_REQUIRED if decision.reason == "plan_required" else CODE_REFUSED
            message = (
                "Not run: submit a plan with submit_plan first."
                if code == CODE_PLAN_REQUIRED
                else "Not run: this call is refused."
            )
            error = ToolError(code, message, {"reason": decision.reason})
            return self._refuse(s, record, spec, error, CallState.REFUSED)

        if decision.action is GateAction.REQUIRE_APPROVAL:
            return self._require_approval(s, record, spec, decision.reason, estimate)

        # EXECUTE: an approval pre-created for this call is no longer needed.
        approval = _approval(s, record.approval_id)
        if approval is not None and approval.status in _UNDECIDED:
            approval.status = ApprovalStatus.CANCELLED
            approval.reason = "Not needed: the call ran without approval."
        return self._start(s, record, spec, experiment_id, approval_id=None)

    def _approved_or_wait(self, s: Session, record: ToolCallRecord) -> StepOutcome | _Execute:
        """Shared hand-off rule: run only an approved call whose hash equals the approved hash."""
        spec = self._registry.get(record.tool)
        approval = _approval(s, record.approval_id)
        if approval is not None and approval.status in _UNDECIDED:
            record.state = CallState.AWAITING_APPROVAL
            self._set_status(s, SessionStatus.WAITING_FOR_APPROVAL)
            return StepOutcome.PAUSED
        if approval is not None and approval.status is ApprovalStatus.REJECTED:
            message = "The user rejected this call. It was not run."
            if approval.reason:
                message += f" Reason: {approval.reason}"
            error = ToolError(
                CODE_APPROVAL_REJECTED, message, {"approval_id": approval.approval_id}
            )
            return self._refuse(s, record, spec, error, CallState.REJECTED)
        if approval is not None and approval.status is ApprovalStatus.EXPIRED:
            error = ToolError(
                CODE_APPROVAL_EXPIRED,
                "Not run: the approval request expired. Propose the step again if it is "
                "still needed.",
                {"approval_id": approval.approval_id},
            )
            return self._refuse(s, record, spec, error, CallState.REFUSED)
        if approval is None or approval.status is not ApprovalStatus.APPROVED:
            status = approval.status.value if approval is not None else "missing"
            error = ToolError(
                CODE_REFUSED, "Not run: the call has no approval.", {"approval_status": status}
            )
            return self._refuse(s, record, spec, error, CallState.REFUSED)
        if approval.approved_hash != record.args_hash:
            error = ToolError(
                CODE_APPROVAL_MISMATCH,
                "Not run: the call's arguments do not match the approved arguments.",
                {"approval_id": approval.approval_id},
            )
            return self._refuse(s, record, spec, error, CallState.REFUSED)
        return self._execute_approved(s, record, spec, approval)

    def _execute_approved(
        self,
        s: Session,
        record: ToolCallRecord,
        spec: ToolSpec | None,
        approval: ApprovalRequest,
    ) -> StepOutcome | _Execute:
        """Last checks before an approved call runs. Caps and prerequisites still apply."""
        if spec is None:
            error = ToolError(
                CODE_UNKNOWN_TOOL,
                f"The tool {record.tool!r} is not registered.",
                {"tool": record.tool, "registered_tools": sorted(self._registry.names())},
            )
            return self._refuse(s, record, None, error, CallState.FAILED)
        try:
            model = spec.input_model.model_validate(record.arguments)
        except PydanticValidationError as exc:
            return self._refuse(s, record, spec, invalid_arguments_error(exc), CallState.FAILED)
        refusal = self._side_effect_refusal(s, spec, model)
        if refusal is not None:
            return self._refuse(s, record, spec, refusal, CallState.REFUSED)
        if spec.tool_class is ToolClass.COSTLY:
            error = _costly_cap_error(s, record.estimate_usd)
            if error is not None:
                return self._refuse(s, record, spec, error, CallState.REFUSED)
        if spec.flags.upload:
            s.upload_approved_once = True
        return self._start(
            s, record, spec, _experiment_of(spec, model), approval_id=approval.approval_id
        )

    def _start(
        self,
        s: Session,
        record: ToolCallRecord,
        spec: ToolSpec,
        experiment_id: str | None,
        *,
        approval_id: str | None,
    ) -> StepOutcome | _Execute:
        """Persist ``executing`` (and, for Costly, the key, marker and counters); run later."""
        now = self._clock()
        if spec.tool_class is ToolClass.COSTLY:
            # The approval id makes a new approval a new call; an unapproved (Autopilot) call
            # uses its own request id (design D11, Req 14.7).
            key = idempotency_key(
                s.session_id, approval_id or record.tool_call_id, record.args_hash
            )
            if any(r is not record and r.idempotency_key == key for r in s.tool_calls):
                error = ToolError(
                    CODE_DUPLICATE_CALL, "Not run: this approved call was already submitted."
                )
                return self._refuse(s, record, spec, error, CallState.REFUSED)
            record.idempotency_key = key
            self._store.put_inflight(
                s.session_id,
                key,
                {
                    "tool": spec.name,
                    "tool_call_id": record.tool_call_id,
                    "args_hash": record.args_hash,
                    "experiment_id": experiment_id,
                    "started_at": now.isoformat(),
                    "known_execution_ids": [
                        j.execution_id
                        for j in s.awaited_jobs
                        if experiment_id is None or j.experiment_id == experiment_id
                    ],
                },
            )
            # Counted when the call is committed, so a crash cannot let a cap be passed (Req 9.6).
            s.counters.costly_tool_calls += 1
            s.counters.estimated_spend_usd += max(record.estimate_usd or 0.0, 0.0)
        record.state = CallState.EXECUTING
        record.started_at = now
        s.last_activity_at = now
        return _Execute(
            call_id=record.tool_call_id,
            spec=spec,
            snapshot=s.model_copy(deep=True),
            approval_id=approval_id,
            experiment_id=experiment_id,
        )

    def _side_effect_refusal(
        self, s: Session, spec: ToolSpec, model: BaseModel
    ) -> ToolError | None:
        """Audit failure (Req 11.9) and the awaited-job prerequisite (Req 1.8)."""
        if spec.tool_class is ToolClass.READ:
            return None
        if s.audit_failed:
            return ToolError(
                CODE_AUDIT_FAILED,
                "Not run: an earlier audit record could not be written.",
                {"reason": "audit_failed"},
            )
        experiment_id = _experiment_of(spec, model)
        if experiment_id is not None and any(
            j.experiment_id == experiment_id and j.state in ("running", "unknown")
            for j in s.awaited_jobs
        ):
            return ToolError(
                CODE_JOB_PENDING,
                "Not run: a job for this experiment has not finished yet.",
                {"experiment_id": experiment_id, "reason": "awaited_job_unresolved"},
            )
        return None

    def _refuse(
        self,
        s: Session,
        record: ToolCallRecord,
        spec: ToolSpec | None,
        error: ToolError,
        state: CallState,
    ) -> StepOutcome:
        self._resolve(s, record, spec, error=error, state=state)
        return self._after_resolution(s)

    def _require_approval(
        self,
        s: Session,
        record: ToolCallRecord,
        spec: ToolSpec,
        reason: str,
        estimate: Estimate,
    ) -> StepOutcome | _Execute:
        """Create (or reuse a pre-created) approval request, pre-create the turn's others."""
        approval = _approval(s, record.approval_id)
        if approval is None:
            try:
                approval = self._new_approval(s, record, spec, estimate, reason)
            except _ApprovalViewFailed:
                error = ToolError(
                    CODE_APPROVAL_VIEW_FAILED,
                    "Not run: the content this approval must show could not be loaded.",
                    {"tool": spec.name},
                )
                return self._refuse(s, record, spec, error, CallState.FAILED)
        record.state = CallState.AWAITING_APPROVAL
        if s.mode is Mode.SUPERVISED:
            self._precreate(s, record)
        return self._approved_or_wait(s, record)

    def _new_approval(
        self,
        s: Session,
        record: ToolCallRecord,
        spec: ToolSpec,
        estimate: Estimate,
        reason: str,
    ) -> ApprovalRequest:
        review: dict[str, Any] | None = None
        if spec.approval_view is not None:
            # Fails closed: no request (so no execution) without the content it must show.
            try:
                model = spec.input_model.model_validate(record.arguments)
                review = dict(spec.approval_view(self._context(s), model))
            except Exception as exc:
                LOGGER.exception("Approval view of tool %s failed", spec.name)
                raise _ApprovalViewFailed(spec.name) from exc
        now = self._clock()
        basis = estimate.basis
        if estimate.price_source:
            basis = f"{basis}; prices: {estimate.price_source} {estimate.price_version or ''}"
        approval = ApprovalRequest(
            approval_id=f"apr_{uuid.uuid4().hex[:16]}",
            kind="upload" if spec.flags.upload else "tool",
            tool_call_id=record.tool_call_id,
            tool=spec.name,
            arguments=dict(record.arguments),
            args_hash=record.args_hash,
            rationale=_rationale(s, record.turn),
            estimate_usd=estimate.usd,
            estimate_basis=basis.strip(),
            status=ApprovalStatus.PENDING,
            created_at=now,
            expires_at=now + timedelta(seconds=self._settings.approval_ttl_seconds),
            review=review,
        )
        s.approvals.append(approval)
        record.approval_id = approval.approval_id
        self._add(
            s,
            "approval_request",
            tool_call_id=record.tool_call_id,
            approval_id=approval.approval_id,
            data={"tool": spec.name, "reason": reason, "args_hash": record.args_hash},
        )
        return approval

    def _precreate(self, s: Session, current: ToolCallRecord) -> None:
        """Pre-create requests for the turn's remaining Costly uses (at most 10 open at once).

        A pre-created call stays ``requested``: it is counted, checked and gated when the loop
        reaches it, and runs then only if its request was approved with the same hash.
        """
        pending = set(s.pending_tool_use_ids)
        open_count = sum(
            1 for a in s.approvals if a.status in _UNDECIDED and a.tool_call_id in pending
        )
        for call_id in s.pending_tool_use_ids:
            if open_count >= _MAX_OPEN_APPROVALS:
                return
            record = _record(s, call_id)
            if (
                record is None
                or record is current
                or record.state is not CallState.REQUESTED
                or record.approval_id is not None
                or _turn_has(s, record.turn, ASK_USER)
            ):
                continue
            spec = self._registry.get(record.tool)
            if spec is None or spec.tool_class is not ToolClass.COSTLY:
                continue
            try:
                model = spec.input_model.model_validate(record.arguments)
            except PydanticValidationError:
                continue  # it gets its validation error when the loop reaches it
            estimate = self._estimate(s, spec, model)
            record.estimate_usd = estimate.usd
            try:
                self._new_approval(s, record, spec, estimate, "costly")
            except _ApprovalViewFailed:
                continue  # it fails closed when the loop reaches it
            open_count += 1

    def _estimate(self, s: Session, spec: ToolSpec, model: BaseModel) -> Estimate:
        """The Guards' estimate for a Costly call; ``usd=None`` when unknown (fails safe)."""
        if spec.tool_class is not ToolClass.COSTLY:
            return Estimate(None, NO_ESTIMATE, None, None)
        if self._estimator is None:
            return Estimate(None, f"{NO_ESTIMATE}: no estimator configured", None, None)
        try:
            return self._estimator.estimate(
                spec.name, model, self._context(s), estimator=spec.flags.estimator
            )
        except Exception:
            LOGGER.exception("Cost estimate for tool %s failed", spec.name)
            return Estimate(None, NO_ESTIMATE, None, None)

    def _context(self, s: Session) -> ToolContext:
        return ToolContext(
            session=s,
            principal=Principal(user=s.owner),
            services=self._services,
            clock=self._clock,
            inputs=self._inputs,
        )

    def _protected(self, s: Session, spec: ToolSpec, model: BaseModel) -> bool:
        if spec.protected is None:
            return False
        try:
            return bool(spec.protected(self._context(s), model))
        except Exception:
            return True

    def _audit_reserved(self, s: Session, record: ToolCallRecord) -> None:
        try:
            self._audit.write(
                s,
                AgentAuditRecord(
                    session_id=s.session_id,
                    owner=s.owner,
                    actor=s.owner,
                    seq=0,
                    event_type="input_rejected",
                    at=self._clock(),
                    result_status="rejected",
                    tool=record.tool[:128],
                    args_hash=record.args_hash,
                    detail={"argument": "tool", "reason": "reserved_tool_name"},
                ),
            )
        except AuditWriteError:
            s.audit_failed = True

    # ================================================================== client uploads
    def _await_client_upload(
        self, s: Session, record: ToolCallRecord, action: _Execute, file_id: str
    ) -> None:
        """An approved upload: create the ``ClientAction`` and wait for the user's client."""
        s.client_action = ClientAction(
            action_id=f"act_{uuid.uuid4().hex[:16]}",
            kind="upload",
            file_id=file_id,
            approval_id=action.approval_id or "",
            status="ready",
            tool_call_id=record.tool_call_id,
        )
        name = next((f.name for f in s.local_files if f.file_id == file_id), file_id)
        self._set_status(
            s,
            SessionStatus.WAITING_FOR_USER,
            text=f"Waiting for your client to upload {name}.",
        )

    def complete_client_upload(self, s: Session, output: ToolOutput) -> StepOutcome:
        """Give the waiting upload call its result. Runs inside the caller's transaction.

        The result goes through the same shaping and governance as every tool result, so the
        model gets the governed profile and no rows (Req 7.9).
        """
        call_id = _client_upload_call(s)
        record = _record(s, call_id) if call_id else None
        spec = self._registry.get(record.tool) if record is not None else None
        if record is None or spec is None or record.state is not CallState.EXECUTING:
            raise ValueError("no upload call is waiting for the client")
        self._record_execution(s, record, spec, ExecutionResult(output=output))
        return self._after_resolution(s)

    # ================================================================== results
    def _record_execution(
        self, s: Session, record: ToolCallRecord, spec: ToolSpec, result: ExecutionResult
    ) -> None:
        if result.audit_failed:
            s.audit_failed = True
        if result.output is None:
            self._resolve(s, record, spec, error=result.error, state=CallState.FAILED)
            return
        output = result.output
        record.service_audit_ref = result.service_audit_ref
        for experiment_id in output.experiment_ids:
            if experiment_id not in s.experiment_ids:
                s.experiment_ids.append(experiment_id)
        known = {d.dataset_id for d in s.datasets}
        for ref in output.datasets:
            if getattr(ref, "dataset_id", None) not in known:
                s.datasets.append(ref)
                known.add(ref.dataset_id)
        jobs = list(output.awaited)
        if spec.job_extractor is not None:
            try:
                jobs.extend(spec.job_extractor(output.data) or [])
            except Exception:
                LOGGER.exception("Job extractor of tool %s failed", spec.name)
        seen = {j.execution_id for j in s.awaited_jobs}
        for job in jobs:
            if not job.execution_id or job.execution_id in seen:
                continue
            seen.add(job.execution_id)
            s.awaited_jobs.append(
                AwaitedJob(
                    execution_id=job.execution_id,
                    experiment_id=job.experiment_id,
                    tool_call_id=record.tool_call_id,
                    kind=job.kind,
                )
            )
            s.jobs_ready = False
        if spec.name == SUBMIT_PLAN:
            steps = [str(x) for x in output.data.get("steps", [])]
            s.plan = Plan(steps=[PlanStep(text=t) for t in steps], accepted_at=self._clock())
            self._add(s, "plan", tool_call_id=record.tool_call_id, data={"steps": steps})
        experiment_id = output.experiment_ids[0] if output.experiment_ids else None
        self._resolve(
            s,
            record,
            spec,
            data=output.data,
            state=CallState.SUCCEEDED,
            experiment_id=experiment_id,
        )

    def _resolve(
        self,
        s: Session,
        record: ToolCallRecord,
        spec: ToolSpec | None,
        *,
        state: CallState,
        data: dict[str, Any] | None = None,
        error: ToolError | None = None,
        experiment_id: str | None = None,
    ) -> None:
        """Give one tool use its single result: shape, store, transcript, un-pend."""
        was_executing = record.state is CallState.EXECUTING
        raw: dict[str, Any] = {"error": error.as_dict()} if error is not None else (data or {})
        shaped = shape_result(
            raw,
            spec=spec if spec is not None else _NamedTool(record.tool),  # type: ignore[arg-type]
            call_id=record.tool_call_id,
            policy=self._governance,
            ctx=self._scrub(s),
            max_bytes=self._settings.tool_result_max_bytes,
            max_string_chars=self._settings.untrusted_string_max_chars,
            experiment_id=experiment_id,
        )
        try:
            record.result_ref = self._store.put_result(
                s.session_id, record.tool_call_id, shaped.full
            )
        except ValueError:  # call id not usable as a key; the model view is still recorded
            record.result_ref = None
        now = self._clock()
        record.state = state
        record.error_code = error.code if error is not None else None
        record.truncated = shaped.truncated
        record.injection_suspected = shaped.injection_suspected
        record.result_summary = (
            error.message[:_MAX_SUMMARY] if error is not None else f"{record.tool} {state.value}"
        )
        record.finished_at = now
        approval = _approval(s, record.approval_id)
        if (
            not was_executing
            and approval is not None
            and approval.status in (*_UNDECIDED, ApprovalStatus.APPROVED)
        ):
            # The call was resolved without running, so its request can no longer be used.
            approval.status = ApprovalStatus.CANCELLED
        entry_data: dict[str, Any] = {
            "tool": record.tool,
            "state": state.value,
            "model_view": shaped.model_view,
        }
        if error is not None:
            entry_data["error_code"] = error.code
        self._add(s, "tool_result", tool_call_id=record.tool_call_id, data=entry_data)
        s.pending_tool_use_ids = [i for i in s.pending_tool_use_ids if i != record.tool_call_id]
        s.last_activity_at = now

    def _after_resolution(self, s: Session) -> StepOutcome:
        """Once the last use of a turn has its result, decide where the session goes."""
        if s.pending_tool_use_ids or s.status is not SessionStatus.RUNNING:
            return StepOutcome.CONTINUE
        turn = s.counters.converse_turns
        question = _answered_question(s, turn)
        if question is not None:
            self._set_status(s, SessionStatus.WAITING_FOR_USER, text=question)
            return StepOutcome.PAUSED
        if _unresolved_jobs(s):
            self._set_status(s, SessionStatus.WAITING_FOR_JOB)
            return StepOutcome.PAUSED
        return StepOutcome.CONTINUE

    def _scrub(self, s: Session) -> ScrubContext:
        if self._scrub_context is not None:
            return self._scrub_context(s)
        return ScrubContext(dataset_uris={d.uri: d.dataset_id for d in s.datasets})

    # ================================================================== model call
    def _model_step(self, session_id: str, session: Session) -> StepOutcome:
        if _limit_stop(session) is not None:
            with self._store.transaction(session_id) as s:
                stop = _limit_stop(s)
                if s.status is SessionStatus.RUNNING and stop is not None:
                    self._stop_for_limit(s, stop)
            return StepOutcome.PAUSED

        epoch = session.epoch
        planning = session.plan is None
        tools = [
            d
            for d in self._registry.definitions()
            if not planning or self._is_control(d.name)
        ]
        messages, _ = build_messages(session, self._settings.context_budget_tokens)
        try:
            turn = self._converse.converse(
                system=system_prompt(session, self._registry),
                messages=messages,
                tools=tools,
                max_tokens=self._settings.max_output_tokens,
            )
        except ConverseError as exc:
            category, message, retriable = _classify(exc, self._converse)
            if planning:
                message = f"Planning failed: {message}"
            with self._store.transaction(session_id) as s:
                if s.epoch == epoch and s.status is SessionStatus.RUNNING:
                    self._fail(s, category, message, retriable=retriable)
            return StepOutcome.DONE

        with self._store.transaction(session_id) as s:
            if s.epoch != epoch or s.status is not SessionStatus.RUNNING:
                LOGGER.info("Model result for session %s discarded (epoch or status)", session_id)
                return StepOutcome.PAUSED
            return self._apply_turn(s, turn)

    def _apply_turn(self, s: Session, turn: ConverseTurn) -> StepOutcome:
        self._accrue(s)
        s.counters.converse_turns += 1
        s.counters.input_tokens += max(0, turn.usage.input_tokens)
        s.counters.output_tokens += max(0, turn.usage.output_tokens)
        s.model_id = turn.model_id
        s.last_activity_at = self._clock()
        number = s.counters.converse_turns
        usage = {
            "turn": number,
            "model_id": turn.model_id,
            "input_tokens": turn.usage.input_tokens,
            "output_tokens": turn.usage.output_tokens,
            "stop_reason": turn.stop_reason,
        }

        if turn.stop_reason == "max_tokens":
            # Partial tool uses are dropped. Ask once more; a second time fails (retriable).
            repeated = _last_is_max_tokens(s)
            self._add(s, "error", text=turn.text or None, data={**usage, "reason": FAIL_MAX_TOKENS})
            if repeated:
                self._fail(
                    s,
                    FAIL_MAX_TOKENS,
                    "The model reply was cut off at the output token limit twice.",
                    retriable=True,
                )
                return StepOutcome.DONE
            return StepOutcome.CONTINUE

        if s.plan is None and not any(self._is_control(u.name) for u in turn.tool_uses):
            self._add(s, "error", text=turn.text or None, data={**usage, "reason": FAIL_PLANNING})
            self._fail(s, FAIL_PLANNING, "Planning failed: the model returned no plan.")
            return StepOutcome.DONE

        if turn.text:
            self._add(s, "assistant_text", text=turn.text, data=usage)

        if not turn.tool_uses:
            results, ledger = self._session_results(s)
            narration = verify_narration(
                turn.text, ledger, policy=self._settings.narration_policy
            )
            answer = build_final_answer(narration, results)
            self._audit_unverified(s, narration)
            s.final_answer = answer
            self._add(s, "final_answer", data=answer.model_dump(mode="json"))
            self._set_status(s, SessionStatus.COMPLETED)
            return StepOutcome.DONE

        pending: list[str] = []
        for use in turn.tool_uses:
            spec = self._registry.get(use.name)
            arguments = dict(use.arguments)
            s.tool_calls.append(
                ToolCallRecord(
                    tool_call_id=use.request_id,
                    turn=number,
                    tool=use.name,
                    arguments=arguments,
                    args_hash=args_hash(use.name, arguments),
                    tool_class=spec.tool_class if spec is not None else None,
                    state=CallState.REQUESTED,
                )
            )
            self._add(s, "tool_call", tool_call_id=use.request_id, data={"tool": use.name})
            pending.append(use.request_id)
        s.pending_tool_use_ids = pending
        return StepOutcome.CONTINUE

    def _session_results(self, s: Session) -> tuple[list[ToolResult], NumberLedger]:
        """The stored full results of the session's successful calls, and their ledger.

        Only tools with ``flags.ledger_source`` feed the ledger. A result that cannot be read
        is skipped, so its numbers stay unverified (fails safe).
        """
        results: list[ToolResult] = []
        ledger = NumberLedger()
        for record in s.tool_calls:
            spec = self._registry.get(record.tool)
            if spec is None or record.result_ref is None or record.state is not CallState.SUCCEEDED:
                continue
            try:
                full = self._store.get_result(s.session_id, record.result_ref)
            except SessionLoadError:
                LOGGER.warning("Stored result of call %s unreadable", record.tool_call_id)
                continue
            results.append(
                ToolResult(
                    tool_call_id=record.tool_call_id,
                    tool=record.tool,
                    turn=record.turn,
                    full=full,
                    verbatim=spec.flags.verbatim,
                    advisory=spec.flags.advisory,
                )
            )
            if spec.flags.ledger_source:
                arg = spec.flags.experiment_arg
                experiment_id = record.arguments.get(arg) if arg else None
                ledger.add(
                    _ledger(
                        full,
                        record.tool_call_id,
                        experiment_id if isinstance(experiment_id, str) else None,
                    )
                )
        return results, ledger

    def _audit_unverified(self, s: Session, narration: VerifiedNarration) -> None:
        """One ``narration_unverified`` record per affected sentence (Req 10.3)."""
        for sentence in narration.unverified:
            try:
                self._audit.write(
                    s,
                    AgentAuditRecord(
                        session_id=s.session_id,
                        owner=s.owner,
                        actor=s.owner,
                        seq=0,
                        event_type="narration_unverified",
                        at=self._clock(),
                        result_status="failure",
                        detail={
                            "unmatched": ",".join(sentence.unmatched)[:500],
                            "policy": narration.policy,
                        },
                    ),
                )
            except AuditWriteError:
                LOGGER.exception("narration_unverified audit write failed for %s", s.session_id)

    def _stop_for_limit(self, s: Session, stop: LimitStop) -> None:
        """Pause in ``waiting_for_user`` naming the limit, its value and the count (Req 9.2)."""
        self._add(
            s,
            "limit_stop",
            text=stop.message,
            data={"limit": stop.name.value, "configured": stop.configured, "current": stop.current},
        )
        try:
            self._audit.write(
                s,
                AgentAuditRecord(
                    session_id=s.session_id,
                    owner=s.owner,
                    actor=s.owner,
                    seq=0,
                    event_type="limit_stop",
                    at=self._clock(),
                    result_status="cancelled",
                    detail={
                        "limit": stop.name.value,
                        "configured": f"{stop.configured:g}",
                        "current": f"{stop.current:g}",
                    },
                ),
            )
        except AuditWriteError:
            s.audit_failed = True
        self._set_status(s, SessionStatus.WAITING_FOR_USER)

    def _is_control(self, name: str) -> bool:
        spec = self._registry.get(name)
        return spec is not None and spec.flags.control

    # ================================================================== helpers
    def _add(self, s: Session, kind: str, **kwargs: Any) -> TranscriptEntry:
        entry = TranscriptEntry(seq=s.next_seq, at=self._clock(), kind=kind, **kwargs)  # type: ignore[arg-type]
        s.transcript.append(entry)
        s.next_seq += 1
        return entry

    def _accrue(self, s: Session) -> None:
        """Add the time since the last mark to active time, only if ``running`` (Req 9.5).

        Marks exist only while ``run`` drives the session, so waiting time is never counted.
        """
        mark = self._active_marks.get(s.session_id)
        if mark is None:
            return
        now = self._clock()
        s.counters = accrue_active(s.counters, s.status, (now - mark).total_seconds())
        self._active_marks[s.session_id] = now

    def _set_status(self, s: Session, status: SessionStatus, *, text: str | None = None) -> None:
        if s.status is status:
            return
        self._accrue(s)  # counts up to now under the old status
        s.status = status
        if status is SessionStatus.WAITING_FOR_JOB and self.on_waiting_for_job is not None:
            try:
                self.on_waiting_for_job(s.session_id)
            except Exception:
                LOGGER.warning("Job watcher was not told about session %s", s.session_id)
        s.last_activity_at = self._clock()
        self._add(s, "status", text=text, data={"status": status.value})

    def _fail(self, s: Session, category: str, message: str, *, retriable: bool = False) -> None:
        now = self._clock()
        s.failure = SessionFailure(category=category, message=message, retriable=retriable, at=now)
        s.retriable = retriable
        self._add(s, "error", text=message, data={"category": category, "retriable": retriable})
        self._set_status(s, SessionStatus.FAILED)
        try:
            self._audit.write(
                s,
                AgentAuditRecord(
                    session_id=s.session_id,
                    owner=s.owner,
                    actor=s.owner,
                    seq=0,
                    event_type="session_failed",
                    at=now,
                    result_status="failure",
                    detail={"category": category, "retriable": str(retriable).lower()},
                ),
            )
        except AuditWriteError:  # not a tool execution, so it does not block side effects
            LOGGER.error("session_failed audit write failed for %s", s.session_id)


# ====================================================================== module helpers


def _record(s: Session, call_id: str) -> ToolCallRecord | None:
    for record in reversed(s.tool_calls):
        if record.tool_call_id == call_id:
            return record
    return None


def _approval(s: Session, approval_id: str | None) -> ApprovalRequest | None:
    if approval_id is None:
        return None
    return next((a for a in s.approvals if a.approval_id == approval_id), None)


def _experiment_of(spec: ToolSpec, model: BaseModel) -> str | None:
    arg = spec.flags.experiment_arg
    value = getattr(model, arg, None) if arg else None
    return value if isinstance(value, str) else None


def _guard_refused(s: Session, tool: str, experiment_id: str | None) -> bool:
    """A Guard refused this tool for this experiment since the last user action (Req 5.6)."""
    return any(g.tool == tool and g.experiment_id == experiment_id for g in s.guard_refusals)


def _limit_stop(s: Session) -> LimitStop | None:
    """The limit that stops the loop before the next model call, if any (Req 9.2, 9.5, 9.9)."""
    return (
        check_loop_top(s.counters, s.limits)
        or check_before_model_call(s.counters, s.limits)
        or check_before_tool_call(s.counters, s.limits)
    )


def _limit_error(stop: LimitStop) -> ToolError:
    return ToolError(
        CODE_LIMIT_REACHED,
        f"Not run: {stop.message}",
        {"limit": stop.name.value, "configured": stop.configured, "current": stop.current},
    )


def _costly_cap_error(s: Session, estimate_usd: float | None) -> ToolError | None:
    """``LIMIT_REACHED`` when a Costly call would pass a cap, in every mode (Req 9.8, 14.3)."""
    name = would_exceed_costly(s.counters, s.limits, estimate_usd)
    if name is None:
        return None
    if name is LimitName.COSTLY:
        configured: float = s.limits.max_costly_tool_calls
        current: float = s.counters.costly_tool_calls
    else:
        configured = s.limits.max_estimated_spend_usd
        current = s.counters.estimated_spend_usd
    error = _limit_error(LimitStop(name=name, configured=configured, current=current))
    if estimate_usd is not None:
        error.details["estimate_usd"] = estimate_usd
    return error


def _turn_has(s: Session, turn: int, tool: str) -> bool:
    return any(r.turn == turn and r.tool == tool for r in s.tool_calls)


def _answered_question(s: Session, turn: int) -> str | None:
    """The question text if ``ask_user`` succeeded in ``turn``."""
    for record in s.tool_calls:
        if record.turn == turn and record.tool == ASK_USER and record.state is CallState.SUCCEEDED:
            question = record.arguments.get("question")
            return str(question).strip()[:_MAX_RATIONALE] if question else ""
        if record.turn == turn and record.error_code == CODE_CHOICE_REQUIRED:
            return record.result_summary or ""  # an upload asked which file to use (Req 7.8)
    return None


def _client_upload_call(s: Session) -> str | None:
    """The call id a ready client upload will resolve, if any."""
    action = s.client_action
    if action is None or action.status != "ready":
        return None
    return action.tool_call_id


def _unresolved_jobs(s: Session) -> bool:
    return any(j.state in ("running", "unknown") for j in s.awaited_jobs)


def _undelivered_jobs(s: Session) -> bool:
    return any(j.state in TERMINAL_JOB_STATES and not j.delivered for j in s.awaited_jobs)


def _dataset_uris(s: Session, arguments: dict[str, Any]) -> frozenset[str]:
    """Dataset URIs a call names, directly or by a session dataset id (reconciler matching)."""
    by_id = {d.dataset_id: d.uri for d in s.datasets}
    found: set[str] = set()
    for value in arguments.values():
        if isinstance(value, str):
            if value in by_id:
                found.add(by_id[value])
            elif value.startswith("s3://"):
                found.add(value)
    return frozenset(found)


def _rationale(s: Session, turn: int) -> str:
    for entry in reversed(s.transcript):
        if entry.kind == "assistant_text" and entry.data.get("turn") == turn:
            return (entry.text or "")[:_MAX_RATIONALE]
    return ""


def _last_is_max_tokens(s: Session) -> bool:
    if not s.transcript:
        return False
    last = s.transcript[-1]
    return last.kind == "error" and last.data.get("reason") == FAIL_MAX_TOKENS


def _classify(exc: ConverseError, client: Any) -> tuple[str, str, bool]:
    """Failure category, user-safe message and retriable flag. Never carries exception text."""
    if isinstance(exc, ConverseModelUnsupportedError):
        return (
            FAIL_MODEL_UNSUPPORTED,
            f"The model {exc.model_id} does not support tool use: {exc.reason}.",
            False,
        )
    if isinstance(exc, ConverseCredentialsError):
        return (
            FAIL_CREDENTIALS,
            "AWS credentials expired or were not recognized. Refresh your credentials, then "
            "resume the session.",
            True,
        )
    if isinstance(exc, ConverseTransientError):
        return (
            FAIL_BEDROCK,
            "Bedrock is unavailable. The session was saved and can be resumed.",
            True,
        )
    model_id = getattr(client, "model_id", "")
    return (
        FAIL_MODEL_REQUEST,
        f"The model call to {model_id} was rejected.",
        bool(getattr(exc, "retriable", False)),
    )
