"""Awaited-job state and in-flight reconciliation (design C13, C15, D11).

``refresh_jobs`` reads ``AnalysisService.execution_status`` for every awaited job that has not
ended and records ``succeeded``, ``failed`` (with the reason), ``stopped`` or ``not_found``
(Req 6.5, 14.1, 14.11). ``jobs_ready`` is true once every awaited job has ended and at least one
update has not yet been shown to the model.

``find_submissions`` is the read side of the reconciler: it lists the execution records a
crashed Costly call may have started (Req 14.5). It only reads; it never submits anything.
``run_bounded`` runs such a query with a time limit (Req 14.6).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from backend.errors import NotFoundError
from ml_engine.contracts.agent import Session
from ml_engine.contracts.execution import ExecutionRecord, JobState
from ml_engine.io import ExperimentLayout, read_model_if_exists

LOGGER = logging.getLogger("ml_factory.agent.jobs")

#: Awaited-job states after which nothing more can happen to the job.
TERMINAL_JOB_STATES = frozenset({"succeeded", "failed", "stopped", "not_found"})
#: Job states that mean SageMaker still has work in flight, mirroring ``is_settled``.
_LIVE_JOB_STATES = frozenset({JobState.IN_PROGRESS.value, JobState.STOPPING.value})
_TERMINAL_EXECUTION = {"SUCCEEDED": "succeeded", "FAILED": "failed", "STOPPED": "stopped"}
_MAX_REASON = 500
_MAX_EXPERIMENTS = 1000

#: The workflow kind (``WorkflowKind`` value) each Costly tool starts. A tool missing here is
#: matched against every kind, so more than one candidate makes the outcome unknown.
TOOL_WORKFLOW_KINDS: dict[str, str] = {
    "create_experiment": "eda",
    "define_derived_features": "derivedprofiling",
    "approve_features": "derivedprofiling",
    "start_training": "training",
    "rerun_training": "training",
    "start_validation": "validation",
    "start_holdout_evaluation": "holdout",
    "start_ablation": "ablation",
    "start_free_form_run": "freeform",
}


# ====================================================================== job refresh


@dataclass(frozen=True, slots=True)
class JobObservation:
    execution_id: str
    state: str
    failure_reason: str | None = None


def observe_jobs(session: Session, analysis: Any) -> list[JobObservation]:
    """Terminal states of the session's unfinished awaited jobs. Reads only, never raises.

    A job whose status cannot be read is left out, so the next refresh retries it.
    """
    found: list[JobObservation] = []
    for job in session.awaited_jobs:
        if job.state in TERMINAL_JOB_STATES:
            continue
        try:
            view = analysis.execution_status(job.experiment_id, job.execution_id)
        except NotFoundError:
            found.append(
                JobObservation(job.execution_id, "not_found", "The job could not be found.")
            )
            continue
        except Exception:
            LOGGER.warning("Status of job %s could not be read; will retry", job.execution_id)
            continue
        state = _terminal_state(view)
        if state is not None:
            found.append(JobObservation(job.execution_id, state, _reason(view, state)))
    return found


def apply_observations(
    session: Session, observations: Iterable[JobObservation], now: datetime
) -> bool:
    """Record observed terminal states and recompute ``jobs_ready``. True if anything changed."""
    changed = False
    by_id = {o.execution_id: o for o in observations}
    for job in session.awaited_jobs:
        seen = by_id.get(job.execution_id)
        if seen is None or job.state in TERMINAL_JOB_STATES:
            continue
        job.state = seen.state  # type: ignore[assignment]
        job.failure_reason = seen.failure_reason
        job.observed_at = now
        changed = True
    ready = jobs_ready(session)
    if session.jobs_ready != ready:
        session.jobs_ready = ready
        changed = True
    return changed


def refresh_jobs(session: Session, analysis: Any, *, now: datetime | None = None) -> bool:
    """Read and record the state of every unfinished awaited job, in place. True if changed.

    The caller persists the session (it holds the transaction).
    """
    observed = observe_jobs(session, analysis)
    return apply_observations(session, observed, now or datetime.now(UTC))


def jobs_ready(session: Session) -> bool:
    """Every awaited job has ended and at least one update has not been shown to the model."""
    jobs = session.awaited_jobs
    return (
        any(not j.delivered for j in jobs)
        and all(j.state in TERMINAL_JOB_STATES for j in jobs)
    )


def _terminal_state(view: Any) -> str | None:
    state = _TERMINAL_EXECUTION.get(str(getattr(view, "status", "")))
    if state is None:
        return None
    if any(getattr(j, "state", None) in _LIVE_JOB_STATES for j in getattr(view, "jobs", [])):
        return None  # e.g. stopped, but SageMaker has not confirmed the stop yet
    return state


def _reason(view: Any, state: str) -> str | None:
    if state == "succeeded":
        return None
    reason = getattr(view, "failure_message", None)
    if not reason:
        reason = next((j.failure_reason for j in view.jobs if j.failure_reason), None)
    if not reason:
        reason = getattr(view, "failure_code", None)
    return str(reason)[:_MAX_REASON] if reason else None


# ====================================================================== reconciler


@dataclass(frozen=True, slots=True)
class Submission:
    """A platform record a crashed Costly call started."""

    experiment_id: str
    execution_id: str | None
    kind: str


def find_submissions(
    services: Any,
    *,
    owner: str,
    tool: str,
    marker: dict[str, Any],
    dataset_uris: frozenset[str] = frozenset(),
    exclude: frozenset[str] = frozenset(),
    name: str | None = None,
) -> list[Submission]:
    """Execution records started at or after the marker, of the tool's kind, not yet known.

    ``create_experiment`` has no experiment yet, so it lists the owner's experiments created
    after the marker with the same name (and dataset, when the call named one). Raises when the
    platform cannot be queried; the caller turns that into an unknown outcome.
    """
    started = _aware(datetime.fromisoformat(str(marker["started_at"])))
    known = set(marker.get("known_execution_ids") or []) | set(exclude)
    kind = TOOL_WORKFLOW_KINDS.get(tool)
    experiment_id = marker.get("experiment_id")
    if experiment_id:
        return [
            Submission(e.experiment_id, e.execution_id, e.kind.value)
            for e in list_executions(services, str(experiment_id))
            if _aware(e.started_at) >= started
            and (kind is None or e.kind.value == kind)
            and e.execution_id not in known
        ]
    if tool == "create_experiment":
        found: list[Submission] = []
        for record in services.repository.list(limit=_MAX_EXPERIMENTS, created_by=owner):
            if _aware(record.created_at) < started:
                continue
            if name is not None and record.name != name:
                continue
            uris = {record.dataset.uri, getattr(record.dataset, "source_uri", None)}
            if dataset_uris and not dataset_uris & uris:
                continue
            execution_id = record.eda_execution_arn
            if execution_id in known:
                continue
            found.append(Submission(record.experiment_id, execution_id, "eda"))
        return found
    raise LookupError(f"no experiment to reconcile tool {tool!r} against")


def list_executions(services: Any, experiment_id: str) -> list[ExecutionRecord]:
    """Every execution record of an experiment, same listing as the orchestrator uses."""
    record = services.repository.get(experiment_id)
    store = services.store
    prefix = ExperimentLayout(base=record.artifact_prefix).executions_prefix
    found: list[ExecutionRecord] = []
    for uri in store.list_uris(prefix):
        rest = uri[len(prefix) :]
        if "/" in rest or not rest.endswith(".json"):
            continue  # observations and diagnoses live in per-execution folders
        execution = read_model_if_exists(store, uri, ExecutionRecord)
        if execution is not None:
            found.append(execution)
    return found


def run_bounded[T](fn: Callable[[], T], timeout_seconds: float) -> T:
    """Run ``fn`` on a daemon thread; raise ``TimeoutError`` if it takes longer than allowed.

    A query that times out keeps running on its daemon thread, but its result is ignored.
    """
    box: dict[str, Any] = {}

    def work() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # re-raised on the caller's thread
            box["error"] = exc

    thread = threading.Thread(target=work, name="agent-reconcile", daemon=True)
    thread.start()
    thread.join(timeout_seconds)
    if thread.is_alive():
        raise TimeoutError(f"query took longer than {timeout_seconds:g} seconds")
    if "error" in box:
        raise box["error"]
    return box["value"]


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
