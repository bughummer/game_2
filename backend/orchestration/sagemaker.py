"""SageMaker orchestrator — the only production ``ExperimentOrchestrator`` (design C3, DD-1).

Phase A of DD-1: direct boto3. A workflow is a short chain of SageMaker jobs, all running the
one stage image with ``python -m jobs.<stage>.main``:

    EDA:       profiling (Processing)
    Training:  preparation (Processing) → one Training job per model, Managed Spot → evaluation
               (Processing)

``start_*`` checks every guard, reserves the workflow's whole upper-bound cost, submits the
first job and returns at once. Later stages are submitted by ``advance(execution_id)``, which
the status monitor calls when a job changes state (EventBridge, or polling as a fallback).
Step Functions replaces ``advance`` in Phase 5 behind the same Protocol.

Single writers: this class is the only writer of ``executions/<id>.json``, the spend ledgers,
and the stage transitions in ``state/workflow.json`` (the evaluation job writes the terminal
state, as it did in old_repo).
"""

import dataclasses
import hashlib
import logging
import re
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from backend.aws.images import resolve_stage_image
from backend.config import Settings
from backend.orchestration.base import ExecutionHandle
from backend.orchestration.guards import (
    GuardViolation,
    PlannedJob,
    Reservation,
    check_budget,
    check_concurrency_ceiling,
    check_instance,
    check_spot_policy,
    reserve,
)
from backend.pricing import PriceTable
from ml_engine.contracts.common import ExperimentStatus
from ml_engine.contracts.config import ExperimentConfig, SplitConfig
from ml_engine.contracts.cv import EvaluationMode
from ml_engine.contracts.execution import (
    SIDE_WORKFLOWS,
    ComputeSpec,
    ExecutionRecord,
    ExecutionStatus,
    ExperimentCompute,
    JobKind,
    JobState,
    SpendReservation,
    SubmittedJob,
    WorkflowKind,
    is_settled,
)
from ml_engine.contracts.experiment import ExperimentRecord, WorkflowState
from ml_engine.contracts.model import ModelFailure
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    join_uri,
    read_model,
    read_model_if_exists,
    write_model,
)

LOGGER = logging.getLogger("ml_factory.orchestration.sagemaker")

CHECKPOINT_LOCAL_PATH = "/opt/ml/checkpoints"
STAGE_MODULES = {
    "profiling": "jobs.profiling.main",
    "preparation": "jobs.preprocessing.main",
    "training": "jobs.training.main",
    "evaluation": "jobs.evaluation.main",
    "derived_profiling": "jobs.derived_profiling.main",
    "holdout_evaluation": "jobs.holdout_evaluation.main",
    "ablation": "jobs.ablation.main",
    "scoring": "jobs.scoring.main",
    "validation": "jobs.validation.main",
    "free_form": "jobs.free_form.main",
}
#: FailureReason fragments that mean "no Spot capacity", the only failures that may trigger an
#: (opted-in) On-Demand fallback. Anything else is a real failure and is reported as such.
CAPACITY_MARKERS = ("InsufficientInstanceCapacity", "CapacityError", "MaxWaitTimeExceeded")
_JOB_NAME_SAFE = re.compile(r"[^a-zA-Z0-9-]+")
#: SageMaker's hard ceiling on a training job's MaxRuntimeInSeconds (28 days).
SAGEMAKER_MAX_TRAINING_RUNTIME_SECONDS = 2_419_200


class OrchestrationError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class JobPlan:
    """One job of a workflow before submission: what to run, on what, with which arguments."""

    name: str
    kind: JobKind
    stage: str
    compute: ComputeSpec
    arguments: list[str]
    model_name: str | None = None
    checkpoint_uri: str | None = None
    fallback_of: str | None = None
    #: Run on this image instead of the configured one (holdout evaluation: the training image).
    image_uri: str | None = None


# --- naming -----------------------------------------------------------------------------------


def job_name(experiment_id: str, token: str, stage: str, model: str | None = None) -> str:
    """Deterministic, SageMaker-valid (``[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}``) and unique per run.

    Resubmitting the same plan reuses the same name, which SageMaker rejects as a duplicate —
    so a retry of ``advance`` can never start a job twice.
    """
    parts = ["mlf", experiment_id, token, stage, model or ""]
    raw = "-".join(_JOB_NAME_SAFE.sub("-", p).strip("-") for p in parts if p)
    raw = re.sub("-{2,}", "-", raw)
    if len(raw) <= 63:
        return raw
    digest = hashlib.sha256(raw.encode()).hexdigest()[:8]
    return f"{raw[:54].rstrip('-')}-{digest}"


def new_execution_id(experiment_id: str, kind: WorkflowKind) -> str:
    return f"{experiment_id}.{kind.value}.{uuid.uuid4().hex[:8]}"


def parse_execution_id(execution_id: str) -> tuple[str, WorkflowKind, str]:
    try:
        experiment_id, kind, token = execution_id.rsplit(".", 2)
        return experiment_id, WorkflowKind(kind), token
    except ValueError as error:
        raise OrchestrationError(
            "UNKNOWN_EXECUTION", f"Not an execution id: {execution_id!r}"
        ) from error


# --- request builders (pure) ----------------------------------------------------------------


def _image(settings: Settings, plan: "JobPlan") -> str:
    return plan.image_uri or settings.stage_image_uri


def _common_env(settings: Settings, image_uri: str) -> dict[str, str]:
    # The image tells the job which environment it is, so it can record it with the results.
    env = {
        "ML_FACTORY_S3_SSE": settings.s3_sse,
        "PYTHONUNBUFFERED": "1",
        "ML_FACTORY_STAGE_IMAGE_URI": image_uri,
    }
    if settings.s3_kms_key_id:
        env["ML_FACTORY_S3_KMS_KEY_ID"] = settings.s3_kms_key_id
    return env


def _tags(record_id: str, execution_id: str, stage: str, model: str | None) -> list[dict]:
    tags = [
        {"Key": "ml-factory:experiment", "Value": record_id},
        {"Key": "ml-factory:execution", "Value": execution_id},
        {"Key": "ml-factory:stage", "Value": stage},
    ]
    if model:
        tags.append({"Key": "ml-factory:model", "Value": model})
    return tags


def build_processing_request(
    settings: Settings, plan: JobPlan, *, experiment_id: str, execution_id: str
) -> dict[str, Any]:
    resources: dict[str, Any] = {
        "InstanceCount": plan.compute.instance_count,
        "InstanceType": plan.compute.instance_type,
        "VolumeSizeInGB": plan.compute.volume_size_gb,
    }
    if settings.volume_kms_key_id:
        resources["VolumeKmsKeyId"] = settings.volume_kms_key_id
    request: dict[str, Any] = {
        "ProcessingJobName": plan.name,
        "RoleArn": settings.sagemaker_execution_role_arn,
        "AppSpecification": {
            "ImageUri": _image(settings, plan),
            "ContainerEntrypoint": ["python", "-m", STAGE_MODULES[plan.stage]],
            "ContainerArguments": plan.arguments,
        },
        "ProcessingResources": {"ClusterConfig": resources},
        "StoppingCondition": {"MaxRuntimeInSeconds": plan.compute.max_runtime_seconds},
        "Environment": _common_env(settings, _image(settings, plan)),
        "Tags": _tags(experiment_id, execution_id, plan.stage, plan.model_name),
    }
    if settings.vpc_subnet_ids:
        request["NetworkConfig"] = {
            "EnableInterContainerTrafficEncryption": True,
            "VpcConfig": {
                "Subnets": settings.vpc_subnet_ids,
                "SecurityGroupIds": settings.vpc_security_group_ids,
            },
        }
    return request


def build_training_request(
    settings: Settings,
    plan: JobPlan,
    *,
    experiment_id: str,
    execution_id: str,
    output_uri: str,
) -> dict[str, Any]:
    resources: dict[str, Any] = {
        "InstanceType": plan.compute.instance_type,
        "InstanceCount": plan.compute.instance_count,
        "VolumeSizeInGB": plan.compute.volume_size_gb,
    }
    if settings.volume_kms_key_id:
        resources["VolumeKmsKeyId"] = settings.volume_kms_key_id
    stopping: dict[str, int] = {"MaxRuntimeInSeconds": plan.compute.max_runtime_seconds}
    if plan.compute.use_spot:
        stopping["MaxWaitTimeInSeconds"] = plan.compute.max_wait_seconds or 0
    output: dict[str, str] = {"S3OutputPath": output_uri}
    if settings.s3_kms_key_id:
        output["KmsKeyId"] = settings.s3_kms_key_id
    request: dict[str, Any] = {
        "TrainingJobName": plan.name,
        "RoleArn": settings.sagemaker_execution_role_arn,
        "AlgorithmSpecification": {
            "TrainingImage": _image(settings, plan),
            "TrainingInputMode": "File",
            "ContainerEntrypoint": ["python", "-m", STAGE_MODULES["training"]],
            "ContainerArguments": plan.arguments,
        },
        # Stages read and write S3 through ObjectStore; no input channels are needed.
        "OutputDataConfig": output,
        "ResourceConfig": resources,
        "StoppingCondition": stopping,
        "EnableManagedSpotTraining": plan.compute.use_spot,
        "Environment": _common_env(settings, _image(settings, plan)),
        "Tags": _tags(experiment_id, execution_id, plan.stage, plan.model_name),
    }
    if plan.checkpoint_uri:
        request["CheckpointConfig"] = {
            "S3Uri": plan.checkpoint_uri,
            "LocalPath": CHECKPOINT_LOCAL_PATH,
        }
    if settings.training_retry_attempts:
        request["RetryStrategy"] = {"MaximumRetryAttempts": settings.training_retry_attempts}
    if settings.vpc_subnet_ids:
        request["VpcConfig"] = {
            "Subnets": settings.vpc_subnet_ids,
            "SecurityGroupIds": settings.vpc_security_group_ids,
        }
        request["EnableInterContainerTrafficEncryption"] = True
    return request


# --- the orchestrator --------------------------------------------------------------------------


class SageMakerOrchestrator:
    name = "sagemaker"

    def __init__(
        self,
        settings: Settings,
        store: ObjectStore,
        sagemaker_client: Any,
        prices: PriceTable | None = None,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if settings.enforce_budget and prices is None:
            raise OrchestrationError(
                "PRICE_TABLE_REQUIRED",
                "enforce_budget is on but no price table was provided.",
            )
        self._settings = settings
        self._store = store
        self._client = sagemaker_client
        self._prices = prices
        self._now = clock or (lambda: datetime.now(UTC))
        # Budget checks read the ledger, then write to it. In one control-plane process this lock
        # makes that atomic; running several replicas needs a conditional-write ledger (Phase 9).
        self._spend_lock = threading.Lock()
        self._started_listeners: list[Callable[[str], None]] = []

    # --- Protocol -----------------------------------------------------------------------------
    def start_eda(self, record: ExperimentRecord) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        execution_id = new_execution_id(record.experiment_id, WorkflowKind.EDA)
        token = parse_execution_id(execution_id)[2]
        arguments = [
            *self._base_args(record.experiment_id, layout),
            "--dataset-uri",
            record.dataset.uri,
            "--target-column",
            record.target_column,
        ]
        if record.dataset.file_format:
            arguments += ["--file-format", record.dataset.file_format]
        plans = [self._processing_plan(record.experiment_id, token, "profiling", arguments)]
        execution = self._open(record.experiment_id, execution_id, WorkflowKind.EDA, layout, plans)
        self._mark(layout, record.experiment_id, ExperimentStatus.EDA_RUNNING, "profiling")
        self._submit_first(execution, layout, plans[0])
        return ExecutionHandle(execution_id=execution_id, detail="profiling submitted")

    def start_training(self, record: ExperimentRecord) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        config = read_model_if_exists(self._store, layout.experiment_config, ExperimentConfig)
        if config is None:
            # Requirement 3.4: nothing is submitted until the configuration is frozen in S3.
            raise OrchestrationError(
                "EXPERIMENT_CONFIG_MISSING",
                "Training needs the frozen experiment configuration in S3; freeze it first.",
            )
        models = [spec.name for spec in config.enabled_models_specs()]
        if not models:
            raise OrchestrationError("NO_MODELS", "The frozen configuration enables no models.")
        # More models than the parallelism limit are not refused: they are submitted in waves
        # as earlier jobs finish (see _training_slots). The hard ceiling below only stops a
        # runaway configuration.
        check_concurrency_ceiling(self._settings, len(models))

        compute = self._experiment_compute(layout, record.experiment_id)
        execution_id = new_execution_id(record.experiment_id, WorkflowKind.TRAINING)
        token = parse_execution_id(execution_id)[2]
        base = self._base_args(record.experiment_id, layout)
        plans = [
            self._processing_plan(record.experiment_id, token, "preparation", base, compute),
            *(
                self._training_plan(
                    record.experiment_id, token, model, layout, compute, split=config.split
                )
                for model in models
            ),
            self._processing_plan(record.experiment_id, token, "evaluation", base, compute),
        ]
        execution = self._open(
            record.experiment_id, execution_id, WorkflowKind.TRAINING, layout, plans
        )
        self._mark(layout, record.experiment_id, ExperimentStatus.PREPARING, "preparation")
        self._submit_first(execution, layout, plans[0])
        return ExecutionHandle(execution_id=execution_id, detail="preparation submitted")

    def start_derived_profiling(self, record: ExperimentRecord) -> ExecutionHandle:
        """Profile the approved derived features in a Processing job (a side workflow)."""
        return self._start_side_job(record, WorkflowKind.DERIVED_PROFILING, "derived_profiling")

    def start_holdout_evaluation(
        self, record: ExperimentRecord, evaluation_id: str
    ) -> ExecutionHandle:
        """Score the trained models on a separate dataset (a side workflow).

        The request (dataset, who asked) is already in ``evaluations/<id>/request.json``.
        """
        return self._start_side_job(
            record,
            WorkflowKind.HOLDOUT_EVALUATION,
            "holdout_evaluation",
            ["--evaluation-id", evaluation_id],
            # Score in the environment the models were trained in, not whatever image is
            # configured today: a newer scikit-learn/XGBoost may not load, or may score
            # differently. Experiments trained before images were recorded use the current one.
            image_uri=self.training_image(record.experiment_id),
        )

    def start_ablation(
        self, record: ExperimentRecord, *, model_name: str, features: list[str]
    ) -> ExecutionHandle:
        """Retrain one model without each candidate feature (a side workflow)."""
        return self._start_side_job(
            record,
            WorkflowKind.ABLATION,
            "ablation",
            ["--model-name", model_name, "--features", ",".join(features)],
            image_uri=self.training_image(record.experiment_id),
        )

    def start_scoring(self, record: ExperimentRecord, scoring_id: str) -> ExecutionHandle:
        """Apply a trained model to a new file (a side workflow; no endpoint)."""
        return self._start_side_job(
            record,
            WorkflowKind.SCORING,
            "scoring",
            ["--scoring-id", scoring_id],
            image_uri=self.training_image(record.experiment_id),
        )

    def start_validation(self, record: ExperimentRecord, validation_id: str) -> ExecutionHandle:
        """Score chosen model versions on an uploaded labeled dataset (a side workflow).

        Runs in the environment the models were trained in (the recorded training image), like
        holdout evaluation, so a saved version loads with the libraries it was built with.
        """
        return self._start_side_job(
            record,
            WorkflowKind.VALIDATION,
            "validation",
            ["--validation-id", validation_id],
            image_uri=self.training_image(record.experiment_id),
        )

    def start_free_form(self, record: ExperimentRecord, run_id: str) -> ExecutionHandle:
        """Run a user-approved free-form script (a side workflow).

        The run's compute (instance type, Spot) is the user's, read from the already-written
        ``free_form/<id>/request.json``, not the default processing instance — a free-form
        experiment may need a bigger or accelerated box (Req 1.4).
        """
        from ml_engine.contracts.free_form import FreeFormRequest

        layout = ExperimentLayout(base=record.artifact_prefix)
        request = read_model_if_exists(
            self._store, layout.free_form_request(run_id), FreeFormRequest
        )
        compute = request.compute if request is not None else None
        return self._start_side_job(
            record,
            WorkflowKind.FREE_FORM,
            "free_form",
            ["--run-id", run_id],
            compute_override=compute,
        )

    def running_executions(self, experiment_id: str) -> list[ExecutionRecord]:
        """Executions of an experiment that still have, or may have, live SageMaker jobs."""
        layout = self._layout_for(experiment_id)
        prefix = layout.executions_prefix
        found: list[ExecutionRecord] = []
        for uri in self._store.list_uris(prefix):
            name = uri[len(prefix) :]
            if "/" in name or not name.endswith(".json"):
                continue
            execution = read_model_if_exists(self._store, uri, ExecutionRecord)
            if execution is not None and not is_settled(execution):
                found.append(execution)
        return found

    def training_image(self, experiment_id: str) -> str | None:
        """The image of the latest successful training run, if one was recorded."""
        layout = self._layout_for(experiment_id)
        prefix = layout.executions_prefix
        latest: ExecutionRecord | None = None
        for uri in self._store.list_uris(prefix):
            name = uri[len(prefix) :]
            if "/" in name or not name.endswith(".json"):
                continue
            execution = read_model_if_exists(self._store, uri, ExecutionRecord)
            if (
                execution is not None
                and execution.kind is WorkflowKind.TRAINING
                and execution.status is ExecutionStatus.SUCCEEDED
                and execution.image_uri
                and (latest is None or execution.started_at > latest.started_at)
            ):
                latest = execution
        return latest.image_uri if latest else None

    def _start_side_job(
        self,
        record: ExperimentRecord,
        kind: WorkflowKind,
        stage: str,
        extra_arguments: list[str] | None = None,
        *,
        image_uri: str | None = None,
        compute_override: ComputeSpec | None = None,
    ) -> ExecutionHandle:
        """One Processing job that never writes the experiment's workflow state, so neither its
        start nor its failure changes what the experiment is doing (feature review, training…).

        ``compute_override`` lets a free-form run choose its own instance/Spot (Req 1.4); other
        side jobs use the experiment's default processing instance.
        """
        layout = ExperimentLayout(base=record.artifact_prefix)
        execution_id = new_execution_id(record.experiment_id, kind)
        token = parse_execution_id(execution_id)[2]
        arguments = [*self._base_args(record.experiment_id, layout), *(extra_arguments or [])]
        compute = self._experiment_compute(layout, record.experiment_id)
        plan = self._processing_plan(
            record.experiment_id, token, stage, arguments, compute, override=compute_override
        )
        plans = [dataclasses.replace(plan, image_uri=image_uri)]
        execution = self._open(record.experiment_id, execution_id, kind, layout, plans)
        try:
            self._submit(execution, layout, plans[0])
        except Exception:
            execution.status = ExecutionStatus.FAILED
            execution.failure_code = "SUBMISSION_FAILED"
            execution.failure_message = execution.jobs[0].failure_reason
            execution.updated_at = self._now()
            write_model(self._store, layout.execution(execution_id), execution)
            raise
        self._notify_started(execution_id)
        return ExecutionHandle(execution_id=execution_id, detail=f"{stage} submitted")

    def on_started(self, listener: Callable[[str], None]) -> None:
        """Register a callback invoked with the execution id once a workflow's first job is
        submitted. The status poller uses it to start watching new executions immediately."""
        self._started_listeners.append(listener)

    def describe(self, execution_id: str) -> dict[str, str]:
        execution = self._load(execution_id)
        summary = {
            "execution_id": execution_id,
            "orchestrator": self.name,
            "status": execution.status.value,
            "stage": execution.stage,
        }
        for job in execution.jobs:
            summary[f"job:{job.job_name}"] = job.state.value
        return summary

    # --- progression (called by the status monitor) --------------------------------------------
    def advance(self, execution_id: str) -> ExecutionRecord:
        """Observe in-flight jobs and submit whatever has become runnable. Idempotent."""
        experiment_id, kind, token = parse_execution_id(execution_id)
        layout = self._layout_for(experiment_id)
        execution = read_model(self._store, layout.execution(execution_id), ExecutionRecord)
        if execution.status is not ExecutionStatus.RUNNING:
            if not is_settled(execution):
                # Stopped, but SageMaker has not confirmed every job ended: keep observing so
                # the record converges on what AWS reports. Nothing new is ever submitted.
                self._observe(execution)
                execution.updated_at = self._now()
                write_model(self._store, layout.execution(execution_id), execution)
            return execution
        self._observe(execution)
        if kind is WorkflowKind.EDA:
            self._advance_eda(execution, layout)
        elif kind in SIDE_WORKFLOWS:
            self._advance_side_job(execution)
        else:
            self._advance_training(execution, layout, token)
        execution.updated_at = self._now()
        write_model(self._store, layout.execution(execution_id), execution)
        return execution

    def stop(self, execution_id: str) -> ExecutionRecord:
        """Stop every running job of an execution. Nothing further is submitted."""
        experiment_id, kind, _ = parse_execution_id(execution_id)
        layout = self._layout_for(experiment_id)
        execution = read_model(self._store, layout.execution(execution_id), ExecutionRecord)
        if execution.status is not ExecutionStatus.RUNNING:
            return execution  # already finished or stopped; nothing to do
        for job in execution.jobs:
            if job.state in {JobState.IN_PROGRESS, JobState.UNKNOWN}:
                try:
                    if job.kind is JobKind.TRAINING:
                        self._client.stop_training_job(TrainingJobName=job.job_name)
                    else:
                        self._client.stop_processing_job(ProcessingJobName=job.job_name)
                    job.state = JobState.STOPPING
                except Exception as error:
                    # Typically the job already ended. Leave its state for _observe to settle;
                    # one refusal must not leave the other jobs running.
                    LOGGER.warning("Could not stop %s: %s", job.job_name, error)
            elif job.state is JobState.PENDING_SUBMISSION:
                job.state = JobState.STOPPED
        execution.status = ExecutionStatus.STOPPED
        execution.failure_code = "STOPPED_BY_USER"
        execution.failure_message = "The execution was stopped."
        execution.updated_at = self._now()
        write_model(self._store, layout.execution(execution_id), execution)
        if kind not in SIDE_WORKFLOWS:  # a side job never owns the experiment's status
            self._fail_workflow(layout, execution, "STOPPED_BY_USER", "The execution was stopped.")
        for listener in self._started_listeners:  # keep watching until AWS confirms the stop
            try:
                listener(execution_id)
            except Exception:
                LOGGER.exception("Listener failed for %s", execution_id)
        return execution

    # --- planning -----------------------------------------------------------------------------
    def _base_args(self, experiment_id: str, layout: ExperimentLayout) -> list[str]:
        return ["--experiment-id", experiment_id, "--artifact-base", layout.base]

    def _experiment_compute(
        self, layout: ExperimentLayout, experiment_id: str
    ) -> ExperimentCompute:
        return read_model_if_exists(self._store, layout.compute, ExperimentCompute) or (
            ExperimentCompute(experiment_id=experiment_id)
        )

    def _processing_plan(
        self,
        experiment_id: str,
        token: str,
        stage: str,
        arguments: list[str],
        compute: ExperimentCompute | None = None,
        *,
        override: ComputeSpec | None = None,
    ) -> JobPlan:
        # A free-form run supplies its own ComputeSpec (instance, Spot, runtime cap); other side
        # jobs use the experiment's default processing instance.
        if override is not None:
            return JobPlan(
                name=job_name(experiment_id, token, stage),
                kind=JobKind.PROCESSING,
                stage=stage,
                compute=override,
                arguments=list(arguments),
            )
        instance = (
            compute.processing_instance_type if compute else None
        ) or self._settings.processing_instance_type
        return JobPlan(
            name=job_name(experiment_id, token, stage),
            kind=JobKind.PROCESSING,
            stage=stage,
            compute=ComputeSpec(
                instance_type=instance,
                volume_size_gb=self._settings.volume_size_gb,
                max_runtime_seconds=self._settings.processing_max_runtime_seconds,
            ),
            arguments=list(arguments),
        )

    def _training_runtime(self, split: SplitConfig | None) -> int:
        """The runtime ceiling of one training job.

        Single_Split_Mode uses ``training_max_runtime_seconds`` unchanged. CV_Mode runs (K + 1)
        fits sequentially in the one job (design D3), so the ceiling is
        ``cv_training_max_runtime_seconds`` or ``training_max_runtime_seconds x (K + 1)``. A
        runtime above SageMaker's 28-day training limit is refused before submission.
        """
        base = self._settings.training_max_runtime_seconds
        if split is None or split.evaluation_mode is not EvaluationMode.CV or split.cv is None:
            return base
        runtime = self._settings.cv_training_max_runtime_seconds or base * (split.cv.folds + 1)
        if runtime > SAGEMAKER_MAX_TRAINING_RUNTIME_SECONDS:
            raise OrchestrationError(
                "CV_RUNTIME_TOO_LONG",
                f"A CV training job would need {runtime} s of runtime "
                f"(K={split.cv.folds}), over SageMaker's training limit of "
                f"{SAGEMAKER_MAX_TRAINING_RUNTIME_SECONDS} s. Lower K or the per-fit runtime.",
            )
        return runtime

    def _training_plan(
        self,
        experiment_id: str,
        token: str,
        model: str,
        layout: ExperimentLayout,
        compute: ExperimentCompute,
        *,
        on_demand: bool = False,
        split: SplitConfig | None = None,
    ) -> JobPlan:
        instance = compute.training_instance_type or self._settings.training_instance_type
        spot = self._settings.spot_enabled and not on_demand
        # Spot checkpoint/resume is dropped (stock-image migration, Req 5.2): every model is
        # restart-on-interruption, so no job scopes a checkpoint path or gets --checkpoint-dir.
        # RESUMABLE_MODELS is empty, so this is always None; the plumbing is left in place
        # (JobPlan/SubmittedJob keep the field) to make the change minimal and reversible.
        checkpoint_uri = None
        arguments = [*self._base_args(experiment_id, layout), "--model-name", model]
        base_name = job_name(experiment_id, token, "train", model)
        # CV runs (K + 1) fits sequentially in the one job, so the runtime ceiling (and therefore
        # the reservation = rate x runtime x instances) scales by (K + 1); everything else holds.
        runtime = self._training_runtime(split)
        max_wait = max(self._settings.spot_max_wait_seconds, runtime) if spot else None
        return JobPlan(
            name=job_name(experiment_id, token, "train", f"{model}-od") if on_demand else base_name,
            kind=JobKind.TRAINING,
            stage="training",
            model_name=model,
            compute=ComputeSpec(
                instance_type=instance,
                volume_size_gb=self._settings.volume_size_gb,
                max_runtime_seconds=runtime,
                use_spot=spot,
                max_wait_seconds=max_wait,
            ),
            arguments=arguments,
            checkpoint_uri=checkpoint_uri,
            fallback_of=base_name if on_demand else None,
        )

    # --- guarded submission -------------------------------------------------------------------
    def _open(
        self,
        experiment_id: str,
        execution_id: str,
        kind: WorkflowKind,
        layout: ExperimentLayout,
        plans: list[JobPlan],
    ) -> ExecutionRecord:
        """Guard the *whole* workflow before submitting anything (fail closed).

        Safety guards (instance allow-list, CPU-only, Spot-only training) always run. Pricing,
        the budget check and the spend ledger run only when ``enforce_budget`` is on; otherwise
        cost is the MLOps team's concern (AWS Budgets / Cost Explorer), not this gate.
        """
        compute = self._experiment_compute(layout, experiment_id)
        # Resolve the stage image once for the whole workflow (stock-image migration, Req 1.4):
        # the configured override if set, else the stock AWS image for the region. A resolution
        # failure (missing SDK, unknown region) raises before anything is submitted (Req 1.6).
        resolved_image = resolve_stage_image(self._settings).uri
        for plan in plans:
            check_instance(self._settings, plan.compute.instance_type)
            planned = PlannedJob(
                plan.kind, plan.compute, on_demand_fallback=plan.fallback_of is not None
            )
            check_spot_policy(
                self._settings, planned, allow_on_demand=compute.allow_on_demand_fallback
            )
        reservations = self._reservations_for(plans) if self._settings.enforce_budget else None
        now = self._now()
        jobs = [
            SubmittedJob(
                job_name=plan.name,
                kind=plan.kind,
                stage=plan.stage,
                model_name=plan.model_name,
                compute=plan.compute,
                checkpoint_uri=plan.checkpoint_uri,
                resumable=plan.checkpoint_uri is not None,
                reserved_usd=reservations[plan.name].reserved_usd if reservations else None,
            )
            for plan in plans
        ]
        execution = ExecutionRecord(
            execution_id=execution_id,
            experiment_id=experiment_id,
            kind=kind,
            artifact_prefix=layout.base,
            started_at=now,
            updated_at=now,
            stage=plans[0].stage,
            jobs=jobs,
            reserved_total_usd=(
                round(sum(r.reserved_usd for r in reservations.values()), 6)
                if reservations
                else None
            ),
            # Pinned for the whole run: later stages use this even if the resolved image
            # changes (a server restart with a new digest) while the run is in flight.
            image_uri=plans[0].image_uri or resolved_image,
        )
        if reservations is not None:
            with self._spend_lock:
                check_budget(
                    self._settings,
                    new_reservations_usd=execution.reserved_total_usd or 0.0,
                    experiment_reserved_usd=self._reserved(layout.spend_prefix),
                    month_reserved_usd=self._reserved(self._month_prefix(now)),
                )
                for plan in plans:
                    self._record_spend(layout, execution, plan, reservations[plan.name], now)
        write_model(self._store, layout.execution(execution_id), execution)
        return execution

    def _reservations_for(self, plans: list[JobPlan]) -> dict[str, Reservation]:
        assert self._prices is not None
        return {
            plan.name: reserve(
                self._settings,
                self._prices,
                PlannedJob(
                    plan.kind, plan.compute, on_demand_fallback=plan.fallback_of is not None
                ),
            )
            for plan in plans
        }

    def _submit(self, execution: ExecutionRecord, layout: ExperimentLayout, plan: JobPlan) -> None:
        job = execution.job(plan.name)
        # Every job of a workflow runs the image resolved once at _open and pinned on the
        # execution (stock-image migration). A side job may override with the training image.
        plan = dataclasses.replace(plan, image_uri=plan.image_uri or execution.image_uri)
        try:
            if plan.kind is JobKind.TRAINING:
                response = self._client.create_training_job(
                    **build_training_request(
                        self._settings,
                        plan,
                        experiment_id=execution.experiment_id,
                        execution_id=execution.execution_id,
                        output_uri=layout.sagemaker_output,
                    )
                )
                job.arn = response.get("TrainingJobArn")
            else:
                response = self._client.create_processing_job(
                    **build_processing_request(
                        self._settings,
                        plan,
                        experiment_id=execution.experiment_id,
                        execution_id=execution.execution_id,
                    )
                )
                job.arn = response.get("ProcessingJobArn")
        except Exception as error:
            if _error_code(error) == "ResourceInUse":
                LOGGER.info("%s already exists; treating the resubmission as done", plan.name)
            else:
                job.state = JobState.FAILED
                job.failure_reason = f"Submission failed: {error}"
                execution.updated_at = self._now()
                write_model(self._store, layout.execution(execution.execution_id), execution)
                raise
        job.state = JobState.IN_PROGRESS
        job.submitted_at = self._now()
        execution.stage = plan.stage
        execution.updated_at = self._now()
        write_model(self._store, layout.execution(execution.execution_id), execution)

    def _submit_first(
        self, execution: ExecutionRecord, layout: ExperimentLayout, plan: JobPlan
    ) -> None:
        """Submit a workflow's first job.

        If AWS refuses it, the execution is closed as FAILED and the experiment's workflow state
        says so. Otherwise the experiment would stay EDA_RUNNING/PREPARING with nothing running,
        and ``start_training`` would refuse every retry as "already running".
        """
        try:
            self._submit(execution, layout, plan)
        except Exception:
            job = execution.job(plan.name)
            execution.status = ExecutionStatus.FAILED
            execution.failure_code = "SUBMISSION_FAILED"
            execution.failure_message = job.failure_reason or f"{plan.name} was not submitted."
            for other in execution.jobs:
                if other.state is JobState.PENDING_SUBMISSION:
                    other.state = JobState.STOPPED
            execution.updated_at = self._now()
            write_model(self._store, layout.execution(execution.execution_id), execution)
            self._fail_workflow(
                layout, execution, execution.failure_code, execution.failure_message
            )
            raise
        self._notify_started(execution.execution_id)

    def _notify_started(self, execution_id: str) -> None:
        for listener in self._started_listeners:
            try:
                listener(execution_id)
            except Exception:  # a watcher must never break job submission
                LOGGER.exception("Start listener failed for %s", execution_id)

    # --- progression internals ------------------------------------------------------------------
    def _observe(self, execution: ExecutionRecord) -> None:
        for job in execution.jobs:
            if job.state not in {JobState.IN_PROGRESS, JobState.STOPPING}:
                continue
            try:
                if job.kind is JobKind.TRAINING:
                    described = self._client.describe_training_job(TrainingJobName=job.job_name)
                    raw = described.get("TrainingJobStatus")
                    secondary = described.get("SecondaryStatus")
                else:
                    described = self._client.describe_processing_job(ProcessingJobName=job.job_name)
                    raw = described.get("ProcessingJobStatus")
                    secondary = None
            except Exception as error:
                LOGGER.warning("Could not describe %s: %s", job.job_name, error)
                continue  # leave the last known state; never invent one
            job.raw_status = raw
            job.secondary_status = secondary
            try:
                job.state = JobState(raw)
            except ValueError:
                job.state = JobState.UNKNOWN
            reason = described.get("FailureReason")
            if secondary == "MaxWaitTimeExceeded":
                reason = f"MaxWaitTimeExceeded: {reason or 'Spot capacity was not available'}"
            job.failure_reason = reason
            job.observed_at = self._now()

    def _advance_eda(self, execution: ExecutionRecord, layout: ExperimentLayout) -> None:
        job = execution.jobs[0]
        if job.state is JobState.COMPLETED:
            execution.status = ExecutionStatus.SUCCEEDED
            self._mark(
                layout, execution.experiment_id, ExperimentStatus.EDA_COMPLETED, "eda_completed"
            )
            self._mark(
                layout, execution.experiment_id, ExperimentStatus.FEATURE_REVIEW, "feature_review"
            )
        elif job.state in {JobState.FAILED, JobState.STOPPED}:
            self._fail(execution, layout, "profiling", job)

    def _advance_side_job(self, execution: ExecutionRecord) -> None:
        """Finish a one-job side workflow without touching the experiment's workflow state."""
        job = execution.jobs[0]
        if job.state is JobState.COMPLETED:
            execution.status = ExecutionStatus.SUCCEEDED
        elif job.state in {JobState.FAILED, JobState.STOPPED}:
            execution.status = ExecutionStatus.FAILED
            execution.failure_code = f"{job.stage.upper()}_FAILED"
            execution.failure_message = job.failure_reason or f"{job.job_name} {job.state.value}"
            LOGGER.warning("%s failed: %s", job.job_name, execution.failure_message)

    def _advance_training(
        self, execution: ExecutionRecord, layout: ExperimentLayout, token: str
    ) -> None:
        preparation = execution.jobs_for("preparation")[0]
        if preparation.state in {JobState.FAILED, JobState.STOPPED}:
            self._fail(execution, layout, "preparation", preparation)
            return
        if preparation.state is not JobState.COMPLETED:
            return

        training = execution.jobs_for("training")
        compute = self._experiment_compute(layout, execution.experiment_id)
        for job in list(training):
            if job.state not in {JobState.FAILED, JobState.STOPPED}:
                continue
            if any(j.fallback_of == job.job_name for j in execution.jobs):
                # Already replaced by an On-Demand job; that job's own outcome decides the
                # model's result. Recording this Spot failure would mark a model failed even
                # when its replacement succeeds.
                continue
            if self._needs_fallback(execution, job, compute):
                self._fall_back(execution, layout, token, job, compute)
            else:
                self._record_model_failure(layout, execution.experiment_id, job)

        # Submit pending models in waves, never more at once than the limits allow.
        training = execution.jobs_for("training")
        pending = [j for j in training if j.state is JobState.PENDING_SUBMISSION]
        if pending:
            if all(j.state is JobState.PENDING_SUBMISSION for j in training):
                self._mark(layout, execution.experiment_id, ExperimentStatus.TRAINING, "training")
            running = [j for j in training if j.state in {JobState.IN_PROGRESS, JobState.STOPPING}]
            for job in pending[: self._training_slots(len(running))]:
                self._submit(execution, layout, self._plan_from(execution, job))
            return

        live = [j for j in execution.jobs_for("training") if not j.state.is_terminal]
        if live:
            return
        evaluation = execution.jobs_for("evaluation")[0]
        if evaluation.state is JobState.PENDING_SUBMISSION:
            self._mark(layout, execution.experiment_id, ExperimentStatus.EVALUATING, "evaluation")
            self._submit(execution, layout, self._plan_from(execution, evaluation))
        elif evaluation.state is JobState.COMPLETED:
            execution.status = ExecutionStatus.SUCCEEDED
        elif evaluation.state in {JobState.FAILED, JobState.STOPPED}:
            self._fail(execution, layout, "evaluation", evaluation)

    def _training_slots(self, running_in_execution: int) -> int:
        """How many more training jobs may start now: per experiment and platform-wide.

        The platform-wide count comes from SageMaker (``ListTrainingJobs``, ``mlf-`` jobs in
        progress), so it covers every experiment and every control-plane replica. If AWS cannot
        be asked, only the per-experiment limit applies and the failure is logged.
        """
        per_experiment = self._settings.max_parallel_training_jobs - running_in_execution
        platform = self._settings.max_platform_training_jobs - self._platform_running()
        return max(0, min(per_experiment, platform))

    def _platform_running(self) -> int:
        try:
            count, token = 0, None
            while True:
                kwargs: dict[str, Any] = {
                    "StatusEquals": "InProgress",
                    "NameContains": "mlf-",
                    "MaxResults": 100,
                }
                if token:
                    kwargs["NextToken"] = token
                response = self._client.list_training_jobs(**kwargs)
                count += len(response.get("TrainingJobSummaries", []))
                token = response.get("NextToken")
                if not token:
                    return count
        except Exception as error:
            LOGGER.warning(
                "Could not count running training jobs; per-experiment cap only: %s", error
            )
            return 0

    def _needs_fallback(
        self, execution: ExecutionRecord, job: SubmittedJob, compute: ExperimentCompute
    ) -> bool:
        if not (job.compute.use_spot and compute.allow_on_demand_fallback):
            return False
        if any(j.fallback_of == job.job_name for j in execution.jobs):
            return False  # already fell back once
        return any(marker in (job.failure_reason or "") for marker in CAPACITY_MARKERS)

    def _fall_back(
        self,
        execution: ExecutionRecord,
        layout: ExperimentLayout,
        token: str,
        job: SubmittedJob,
        compute: ExperimentCompute,
    ) -> None:
        """Explicitly opted-in On-Demand replacement for a Spot capacity failure. Re-guarded.

        The CV runtime is preserved: the re-plan reads the frozen config's split, so the
        On-Demand job keeps the (K + 1)-scaled runtime ceiling the Spot job had.
        """
        config = read_model_if_exists(self._store, layout.experiment_config, ExperimentConfig)
        plan = self._training_plan(
            execution.experiment_id,
            token,
            job.model_name or "",
            layout,
            compute,
            on_demand=True,
            split=config.split if config else None,
        )
        check_instance(self._settings, plan.compute.instance_type)
        planned = PlannedJob(plan.kind, plan.compute, on_demand_fallback=True)
        check_spot_policy(self._settings, planned, allow_on_demand=compute.allow_on_demand_fallback)
        now = self._now()
        reserved_usd = None
        if self._settings.enforce_budget:
            reservation = reserve(self._settings, self._prices, planned)
            reserved_usd = reservation.reserved_usd
            with self._spend_lock:
                check_budget(
                    self._settings,
                    new_reservations_usd=reserved_usd,
                    experiment_reserved_usd=self._reserved(layout.spend_prefix),
                    month_reserved_usd=self._reserved(self._month_prefix(now)),
                )
                self._record_spend(layout, execution, plan, reservation, now)
            execution.reserved_total_usd = round(
                (execution.reserved_total_usd or 0.0) + reserved_usd, 6
            )
        execution.jobs.append(
            SubmittedJob(
                job_name=plan.name,
                kind=plan.kind,
                stage=plan.stage,
                model_name=plan.model_name,
                compute=plan.compute,
                fallback_of=job.job_name,
                reserved_usd=reserved_usd,
            )
        )
        LOGGER.warning("Spot capacity failure for %s; opted-in On-Demand fallback", job.job_name)
        self._submit(execution, layout, dataclasses.replace(plan, image_uri=execution.image_uri))

    def _plan_from(self, execution: ExecutionRecord, job: SubmittedJob) -> JobPlan:
        """Rebuild a pending job's plan exactly as it was guarded and reserved."""
        layout = ExperimentLayout(base=execution.artifact_prefix)
        arguments = self._base_args(execution.experiment_id, layout)
        if job.model_name:
            arguments += ["--model-name", job.model_name]
            if job.checkpoint_uri:
                arguments += ["--checkpoint-dir", CHECKPOINT_LOCAL_PATH]
        return JobPlan(
            name=job.job_name,
            kind=job.kind,
            stage=job.stage,
            compute=job.compute,
            arguments=arguments,
            model_name=job.model_name,
            checkpoint_uri=job.checkpoint_uri,
            fallback_of=job.fallback_of,
            image_uri=execution.image_uri,
        )

    def _record_model_failure(
        self, layout: ExperimentLayout, experiment_id: str, job: SubmittedJob
    ) -> None:
        """A job that died before the stage could write its own failure still gets one."""
        uri = layout.model_failure(job.model_name or "")
        if self._store.exists(uri):
            return
        write_model(
            self._store,
            uri,
            ModelFailure(
                experiment_id=experiment_id,
                model_name=job.model_name or "",
                error_code="SAGEMAKER_JOB_" + job.state.value.upper(),
                message=job.failure_reason or f"SageMaker job {job.job_name} {job.state.value}.",
                failed_at=self._now(),
                details={"job_name": job.job_name},
            ),
        )

    def _fail(
        self, execution: ExecutionRecord, layout: ExperimentLayout, stage: str, job: SubmittedJob
    ) -> None:
        execution.status = ExecutionStatus.FAILED
        execution.failure_code = f"{stage.upper()}_FAILED"
        execution.failure_message = job.failure_reason or f"{job.job_name} {job.state.value}"
        for other in execution.jobs:
            if other.state is JobState.PENDING_SUBMISSION:
                # Never submitted. Its reservation stays in the ledger (write-once, conservative)
                # until tracking reconciles reservations with measured cost (Phase 4).
                other.state = JobState.STOPPED
        self._fail_workflow(layout, execution, execution.failure_code, execution.failure_message)

    # --- state documents ---------------------------------------------------------------------
    def _mark(
        self, layout: ExperimentLayout, experiment_id: str, status: ExperimentStatus, stage: str
    ) -> None:
        write_model(
            self._store,
            layout.workflow_state,
            WorkflowState(
                experiment_id=experiment_id,
                updated_at=self._now(),
                status=status,
                current_stage=stage,
            ),
        )

    def _fail_workflow(
        self, layout: ExperimentLayout, execution: ExecutionRecord, code: str, message: str
    ) -> None:
        write_model(
            self._store,
            layout.workflow_state,
            WorkflowState(
                experiment_id=execution.experiment_id,
                updated_at=self._now(),
                status=ExperimentStatus.FAILED,
                current_stage=execution.stage,
                failure_code=code,
                failure_message=message,
            ),
        )

    def _load(self, execution_id: str) -> ExecutionRecord:
        experiment_id, _, _ = parse_execution_id(execution_id)
        layout = self._layout_for(experiment_id)
        record = read_model_if_exists(self._store, layout.execution(execution_id), ExecutionRecord)
        if record is None:
            raise OrchestrationError("UNKNOWN_EXECUTION", f"No execution {execution_id!r}")
        return record

    def _layout_for(self, experiment_id: str) -> ExperimentLayout:
        return ExperimentLayout.for_experiment(self._settings.artifact_root, experiment_id)

    # --- spend ledger -------------------------------------------------------------------------
    def _month_prefix(self, now: datetime) -> str:
        return ExperimentLayout.monthly_spend_prefix(
            self._settings.artifact_root, now.strftime("%Y-%m")
        )

    def _reserved(self, prefix: str) -> float:
        total = 0.0
        for uri in self._store.list_uris(prefix):
            total += read_model(self._store, uri, SpendReservation).reserved_usd
        return total

    def _record_spend(
        self,
        layout: ExperimentLayout,
        execution: ExecutionRecord,
        plan: JobPlan,
        reservation: Reservation,
        now: datetime,
    ) -> None:
        entry = SpendReservation(
            job_name=plan.name,
            experiment_id=execution.experiment_id,
            execution_id=execution.execution_id,
            month=now.strftime("%Y-%m"),
            reserved_at=now,
            instance_type=plan.compute.instance_type,
            instance_count=plan.compute.instance_count,
            max_billable_seconds=reservation.max_billable_seconds,
            hourly_rate_usd=reservation.rate.hourly_usd,
            reserved_usd=reservation.reserved_usd,
            region=self._settings.aws_region,
            price_source=reservation.rate.source,
            price_table_version=reservation.rate.version,
            price_retrieved_on=reservation.rate.retrieved_on,
        )
        write_model(self._store, join_uri(layout.spend_prefix, f"{plan.name}.json"), entry)
        write_model(self._store, join_uri(self._month_prefix(now), f"{plan.name}.json"), entry)


def _error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if isinstance(response, dict):
        return response.get("Error", {}).get("Code")
    return None


__all__ = [
    "GuardViolation",
    "JobPlan",
    "OrchestrationError",
    "SageMakerOrchestrator",
    "build_processing_request",
    "build_training_request",
    "job_name",
    "parse_execution_id",
]
