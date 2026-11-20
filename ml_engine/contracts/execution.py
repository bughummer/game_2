"""Execution, compute and spend records for SageMaker workflows (new in ML Factory 2.0).

``config/compute.json``            per-experiment compute choices; written by the control plane
``executions/<id>.json``           one workflow execution and its SageMaker jobs; written only by
                                   the orchestrator
``spend/<job>.json`` (+ monthly)   one reserved upper-bound cost per submitted job; written only
                                   by the orchestrator, once, at submission

Each document has exactly one writer, so parallel jobs never contend for a shared object.
Dollar figures here are *reservations*: the most a job can cost at on-demand rates for its
maximum runtime. They are estimates, never the bill; measured cost arrives with tracking.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import Field, model_validator

from ml_engine.contracts.common import StrictModel

#: No model resumes from a Spot checkpoint: every model is restart-on-interruption (stock-image
#: migration, decision B4). The constant is kept — empty — so the orchestrator and the training
#: stage's checkpointing module still import it; Managed Spot (the discount) is unaffected.
RESUMABLE_MODELS: frozenset[str] = frozenset()

EXECUTION_SCHEMA_VERSION = "1.0"
COMPUTE_SCHEMA_VERSION = "1.0"
SPEND_SCHEMA_VERSION = "1.0"


class WorkflowKind(StrEnum):
    EDA = "eda"
    TRAINING = "training"
    #: Side job after a derived-feature approval. Never changes the experiment's status.
    DERIVED_PROFILING = "derivedprofiling"
    #: Side job scoring trained models on a separate dataset. Never changes the status.
    HOLDOUT_EVALUATION = "holdout"
    #: Side job retraining one model without each candidate feature. Never changes the status.
    ABLATION = "ablation"
    #: Side job applying a trained model to a new file. Never changes the status.
    SCORING = "scoring"
    #: Side job scoring chosen model versions on an uploaded labeled dataset. Never changes status.
    VALIDATION = "validation"
    #: Side job running a user-approved free-form script. Never changes the status.
    FREE_FORM = "freeform"


#: Side workflows: one Processing job, never touching the experiment's workflow state.
SIDE_WORKFLOWS = frozenset(
    {
        WorkflowKind.DERIVED_PROFILING,
        WorkflowKind.HOLDOUT_EVALUATION,
        WorkflowKind.ABLATION,
        WorkflowKind.SCORING,
        WorkflowKind.FREE_FORM,
        WorkflowKind.VALIDATION,
    }
)


class JobKind(StrEnum):
    PROCESSING = "processing"
    TRAINING = "training"


class JobState(StrEnum):
    """Primary SageMaker job status, plus the platform's own pre-submission state."""

    PENDING_SUBMISSION = "PendingSubmission"
    IN_PROGRESS = "InProgress"
    COMPLETED = "Completed"
    FAILED = "Failed"
    STOPPING = "Stopping"
    STOPPED = "Stopped"
    UNKNOWN = "Unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {JobState.COMPLETED, JobState.FAILED, JobState.STOPPED}


class ExecutionStatus(StrEnum):
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"


class ComputeSpec(StrictModel):
    """The compute one job runs on, as submitted."""

    instance_type: str
    instance_count: int = Field(default=1, ge=1)
    volume_size_gb: int = Field(default=30, ge=1)
    max_runtime_seconds: int = Field(gt=0)
    use_spot: bool = False
    max_wait_seconds: int | None = Field(
        default=None, description="Spot only: total time incl. waiting for capacity."
    )

    @model_validator(mode="after")
    def _spot_wait(self) -> "ComputeSpec":
        if self.use_spot:
            if self.max_wait_seconds is None:
                raise ValueError("a Spot job needs max_wait_seconds")
            if self.max_wait_seconds < self.max_runtime_seconds:
                raise ValueError("max_wait_seconds must be >= max_runtime_seconds")
        elif self.max_wait_seconds is not None:
            raise ValueError("max_wait_seconds applies to Spot jobs only")
        return self


class ExperimentCompute(StrictModel):
    """``config/compute.json`` — per-experiment compute choices. Absent means platform defaults."""

    schema_version: str = COMPUTE_SCHEMA_VERSION
    experiment_id: str
    training_instance_type: str | None = None
    processing_instance_type: str | None = None
    allow_on_demand_fallback: bool = Field(
        default=False,
        description=(
            "Explicit opt-in. When false, a Spot capacity failure fails the model and is "
            "reported; the platform never switches to On-Demand on its own."
        ),
    )
    decided_by: str | None = None
    decided_at: datetime | None = None


class SubmittedJob(StrictModel):
    job_name: str
    kind: JobKind
    stage: str
    model_name: str | None = None
    arn: str | None = None
    state: JobState = JobState.PENDING_SUBMISSION
    compute: ComputeSpec
    checkpoint_uri: str | None = None
    resumable: bool = False
    fallback_of: str | None = Field(
        default=None, description="The Spot job this On-Demand job replaced, when it did."
    )
    reserved_usd: float | None = None
    submitted_at: datetime | None = None
    observed_at: datetime | None = None
    failure_reason: str | None = None
    raw_status: str | None = Field(default=None, description="SageMaker primary status, verbatim.")
    secondary_status: str | None = Field(
        default=None, description="SageMaker SecondaryStatus, verbatim (training jobs only)."
    )


class ExecutionRecord(StrictModel):
    """``executions/<id>.json`` — one workflow execution. Written only by the orchestrator."""

    schema_version: str = EXECUTION_SCHEMA_VERSION
    execution_id: str
    experiment_id: str
    kind: WorkflowKind
    artifact_prefix: str
    started_at: datetime
    updated_at: datetime
    stage: str
    status: ExecutionStatus = ExecutionStatus.RUNNING
    jobs: list[SubmittedJob] = Field(default_factory=list)
    reserved_total_usd: float | None = None
    failure_code: str | None = None
    failure_message: str | None = None
    image_uri: str | None = Field(
        default=None,
        description=(
            "The stage image every job of this execution ran on. For training, this is the "
            "environment the models were built in; holdout evaluation reuses it."
        ),
    )

    def job(self, job_name: str) -> SubmittedJob:
        for candidate in self.jobs:
            if candidate.job_name == job_name:
                return candidate
        raise KeyError(job_name)

    def jobs_for(self, stage: str) -> list[SubmittedJob]:
        return [j for j in self.jobs if j.stage == stage]


def is_settled(execution: ExecutionRecord) -> bool:
    """True once nothing more can happen: the execution has ended and AWS reports no live job.

    A stopped execution whose jobs are still ``Stopping`` is not settled — it must keep being
    observed until SageMaker confirms the stop.
    """
    if execution.status is ExecutionStatus.RUNNING:
        return False
    return not any(j.state in {JobState.IN_PROGRESS, JobState.STOPPING} for j in execution.jobs)


class SpendReservation(StrictModel):
    """One job's reserved upper-bound cost. Written once, at submission, never updated."""

    schema_version: str = SPEND_SCHEMA_VERSION
    job_name: str
    experiment_id: str
    execution_id: str
    month: str = Field(pattern=r"^\d{4}-\d{2}$")
    reserved_at: datetime
    instance_type: str
    instance_count: int
    max_billable_seconds: int
    hourly_rate_usd: float
    reserved_usd: float
    region: str
    price_source: str
    price_table_version: str
    price_retrieved_on: str
    label: str = "reserved upper-bound estimate (on-demand rate x max runtime); not the bill"
