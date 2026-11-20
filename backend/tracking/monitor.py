"""The status monitor (requirement 6): persist what SageMaker reports, then let the orchestrator
advance the workflow, and diagnose any job that has just failed (requirement 8).

Two entry points, same body:
* ``on_event`` — called by an EventBridge rule on a SageMaker job state change (timely).
* ``poll`` — called on a timer as a fallback for a missed event (eventually consistent).

The monitor does not decide what runs next; it records observations and calls
``SageMakerOrchestrator.advance()``, which owns stage sequencing and is idempotent. Observations
and diagnoses are written independently of any chat history. No cost or Spot accounting lives
here — that is the MLOps team's domain.
"""

import logging
from datetime import UTC, datetime
from typing import Any

from backend.tracking.diagnostics import diagnose_failure, fetch_logs
from backend.tracking.status import normalize_processing_status, normalize_training_status
from ml_engine.contracts.execution import ExecutionRecord, JobKind, JobState, SubmittedJob
from ml_engine.contracts.tracking import JobObservation
from ml_engine.io import ExperimentLayout, ObjectStore, read_model_if_exists, write_model

LOGGER = logging.getLogger("ml_factory.tracking.monitor")


class StatusMonitor:
    def __init__(
        self,
        orchestrator: Any,
        store: ObjectStore,
        sagemaker_client: Any,
        *,
        logs_client: Any = None,
        log_group: str = "/aws/sagemaker/TrainingJobs",
        processing_log_group: str = "/aws/sagemaker/ProcessingJobs",
    ) -> None:
        self._orchestrator = orchestrator
        self._store = store
        self._client = sagemaker_client
        self._logs = logs_client
        self._log_group = log_group
        self._processing_log_group = processing_log_group

    def on_event(self, execution_id: str, job_name: str) -> ExecutionRecord | None:
        """Handle a SageMaker job-state-change event for one job of an execution.

        The event names a single job, but we re-observe the whole execution so a missed earlier
        event cannot leave another job's state stale.
        """
        LOGGER.info("Event for %s in %s", job_name, execution_id)
        return self._tick(execution_id, source="event")

    def poll(self, execution_id: str) -> ExecutionRecord | None:
        """Fallback sweep of an execution's in-flight jobs."""
        return self._tick(execution_id, source="poll")

    # --- internals ----------------------------------------------------------------------------
    def _tick(self, execution_id: str, *, source: str) -> ExecutionRecord | None:
        layout = self._layout_for(execution_id)
        before = read_model_if_exists(self._store, layout.execution(execution_id), ExecutionRecord)
        if before is None:
            LOGGER.warning("No execution record for %s", execution_id)
            return None
        before_states = {job.job_name: job.state for job in before.jobs}

        # advance() re-describes in-flight jobs, writes the new execution record, and submits
        # whatever has become runnable. It is the single owner of job state and sequencing.
        after = self._orchestrator.advance(execution_id)

        for job in after.jobs:
            if job.observed_at is not None:
                self._write_observation(layout, after, job, source)
            was = before_states.get(job.job_name)
            if job.state is JobState.FAILED and was is not JobState.FAILED:
                self._diagnose(layout, after, job)
        return after

    def _write_observation(
        self, layout: ExperimentLayout, execution: ExecutionRecord, job: SubmittedJob, source: str
    ) -> None:
        if job.kind is JobKind.TRAINING:
            normalized = normalize_training_status(job.raw_status, job.secondary_status)
        else:
            normalized = normalize_processing_status(job.raw_status)
        observation = JobObservation(
            execution_id=execution.execution_id,
            job_name=job.job_name,
            stage=job.stage,
            model_name=job.model_name,
            observed_at=job.observed_at or datetime.now(UTC),
            normalized_status=normalized,
            raw_status=job.raw_status,
            secondary_status=job.secondary_status,
            failure_reason=job.failure_reason,
            instance_type=job.compute.instance_type,
            instance_count=job.compute.instance_count,
            source=source,
        )
        write_model(
            self._store, layout.observation(execution.execution_id, job.job_name), observation
        )

    def _diagnose(
        self, layout: ExperimentLayout, execution: ExecutionRecord, job: SubmittedJob
    ) -> None:
        log_group = self._log_group if job.kind is JobKind.TRAINING else self._processing_log_group
        logs = fetch_logs(self._logs, log_group, job.job_name) if self._logs else []
        diagnosis = diagnose_failure(
            self._store,
            layout,
            execution_id=execution.execution_id,
            job_name=job.job_name,
            stage=job.stage,
            model_name=job.model_name,
            failure_reason=job.failure_reason,
            log_lines=logs,
        )
        write_model(self._store, layout.diagnosis(execution.execution_id, job.job_name), diagnosis)
        LOGGER.info("Diagnosed %s as %s", job.job_name, diagnosis.category.value)

    def _layout_for(self, execution_id: str) -> ExperimentLayout:
        from backend.orchestration.sagemaker import parse_execution_id

        experiment_id, _, _ = parse_execution_id(execution_id)
        return self._orchestrator._layout_for(experiment_id)
