"""Status and failure-diagnosis contracts (requirements 6 and 8).

Deliberately money-free. Spot utilization, savings and dollar cost are the MLOps team's domain
(AWS Budgets / Cost Explorer); nothing here prices or sums anything. These contracts answer two
questions only: "where is my job?" and "why did it fail?"

Every field is an AWS-reported fact or a deterministic classification of one. ``UNKNOWN`` is a
first-class value: when AWS evidence is insufficient the platform says so rather than guessing
(requirement 6.4, design Property 10).
"""

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from ml_engine.contracts.common import StrictModel

TRACKING_SCHEMA_VERSION = "1.0"


class NormalizedStatus(StrEnum):
    """User-facing status. A deterministic function of the raw SageMaker fields; never invented."""

    QUEUED = "Queued"
    WAITING_FOR_CAPACITY = "Waiting for capacity"
    PREPARING = "Preparing"
    TRAINING = "Training"
    INTERRUPTED_RECOVERING = "Interrupted/recovering"
    EVALUATING = "Evaluating"
    COMPLETED = "Completed"
    FAILED = "Failed"
    STOPPED = "Stopped"
    UNKNOWN = "Unknown"

    @property
    def is_terminal(self) -> bool:
        return self in {
            NormalizedStatus.COMPLETED,
            NormalizedStatus.FAILED,
            NormalizedStatus.STOPPED,
        }


class JobObservation(StrictModel):
    """One point-in-time reading of a SageMaker job. Written only by the status monitor.

    Raw fields are preserved exactly as AWS returned them, alongside the normalized status, so a
    user never has to read raw AWS output but the raw truth is never lost (requirement 6.2).
    """

    schema_version: str = TRACKING_SCHEMA_VERSION
    execution_id: str
    job_name: str
    stage: str
    model_name: str | None = None
    observed_at: datetime
    normalized_status: NormalizedStatus
    raw_status: str | None = Field(default=None, description="SageMaker primary job status.")
    secondary_status: str | None = Field(
        default=None, description="SageMaker SecondaryStatus (training jobs only)."
    )
    secondary_transitions: list[str] = Field(
        default_factory=list, description="SecondaryStatus history, newest last, as reported."
    )
    failure_reason: str | None = None
    instance_type: str | None = None
    instance_count: int | None = None
    source: str = Field(default="poll", description="How this reading arrived: 'event' or 'poll'.")


class FailureCategory(StrEnum):
    """Deterministically recognizable failure classes (requirement 8.2)."""

    INSUFFICIENT_CAPACITY = "insufficient_capacity"
    SPOT_WAIT_TIMEOUT = "spot_wait_timeout"
    CONTAINER_STARTUP = "container_startup"
    MISSING_DEPENDENCY = "missing_dependency"
    INVALID_DATA = "invalid_data"
    OUT_OF_MEMORY = "out_of_memory"
    IAM_PERMISSION = "iam_permission"
    S3_ACCESS = "s3_access"
    SCRIPT_EXCEPTION = "script_exception"
    STOPPED_BY_USER = "stopped_by_user"
    UNKNOWN = "unknown"


class FailureDiagnosis(StrictModel):
    """What is known about why a job failed. Written only by the diagnoser.

    The LLM may later narrate ``evidence`` and ``recommended_action`` but never invents a cause
    (requirement 8.3). ``log_excerpt`` is already bounded and PII-redacted (requirement 8.4).
    """

    schema_version: str = TRACKING_SCHEMA_VERSION
    execution_id: str
    job_name: str
    stage: str
    model_name: str | None = None
    diagnosed_at: datetime
    category: FailureCategory
    summary: str = Field(description="One plain-language line. Grounded in the evidence below.")
    recommended_action: str | None = None
    failure_reason: str | None = Field(
        default=None, description="SageMaker FailureReason, verbatim (already AWS-sanitised)."
    )
    stage_error_code: str | None = Field(
        default=None, description="Code from the stage's own failure.json, when present."
    )
    exit_code: int | None = None
    evidence: list[str] = Field(
        default_factory=list, description="The bounded, redacted facts the category rests on."
    )
    log_excerpt: list[str] = Field(
        default_factory=list, description="Bounded, PII-redacted CloudWatch lines (may be empty)."
    )
    logs_truncated: bool = False
