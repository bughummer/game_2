"""Free-form experiment contracts (free-form-experiments design C1).

A free-form run executes a user-approved Python script as a SageMaker side job. The platform does
not control what the script does, so it labels results honestly (``free_form``; each metric
``self_reported`` or ``engine``) and restores trust only through the Platform_SDK or independent
validation. Execution binds to the approved script's content hash: only those exact bytes run.

All additive: a no-free-form experiment writes none of these and is byte-identical to today.
"""

import hashlib
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from ml_engine.contracts.common import StrictModel
from ml_engine.contracts.execution import ComputeSpec
from ml_engine.contracts.requirements import RequirementsSpec

FREE_FORM_SCHEMA_VERSION = "1.0"


def content_hash(source: str) -> str:
    """The stable content hash a script's approval binds to (sha256 over the exact bytes)."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


class MetricLabel(StrEnum):
    """How much a free-form number can be trusted."""

    #: Computed by the script itself; stored but never mixed into an engine-metric ranking.
    SELF_REPORTED = "self_reported"
    #: Computed by the deterministic engine, via the Platform_SDK or independent validation.
    ENGINE = "engine"


class DataContextAnswer(StrictModel):
    """One answer to a Req-14 data-context question (answer, don't-know, or skipped)."""

    question_id: str
    question: str
    status: Literal["answered", "dont_know", "skipped"]
    answer: str | None = None


class DataContextAnswers(StrictModel):
    """The data-context the AI must have before drafting (Req 14); reusable as dataset defaults."""

    schema_version: str = FREE_FORM_SCHEMA_VERSION
    answers: list[DataContextAnswer] = Field(default_factory=list)

    @property
    def is_complete(self) -> bool:
        """Every question has been answered, marked don't-know, or skipped."""
        return bool(self.answers)


class ScriptRevision(StrictModel):
    """One Script_Draft revision. Every revision is retained with its author and diff context."""

    schema_version: str = FREE_FORM_SCHEMA_VERSION
    revision: int = Field(ge=1)
    content: str = Field(description="The full script source for this revision.")
    content_hash: str = Field(description="sha256 of the content; approval binds to this.")
    authored_by: Literal["ai", "user"]
    instruction: str | None = Field(
        default=None, description="The revise instruction, when the AI authored this revision."
    )
    assumptions: list[str] = Field(
        default_factory=list, description="The AI's stated assumptions and noticed risks (Req 2.7)."
    )
    created_at: datetime


class ApprovedScript(StrictModel):
    """The exact code a run may execute. Execution binds to ``content_hash`` (Req 2.4)."""

    schema_version: str = FREE_FORM_SCHEMA_VERSION
    content_hash: str
    revision: int
    entry_module: str = Field(default="experiment", description="The script module name.")
    entry_function: str = Field(default="main", description="The callable the job invokes.")
    predict_entry_point: str | None = Field(
        default=None,
        description="A function mapping a feature table to predictions, enabling validation.",
    )
    approved_by: str | None = None
    approved_at: datetime
    acknowledged_high_severity_leakage: bool = Field(
        default=False, description="Set when the user approved with open high-severity findings."
    )


class DataContextQuestion(StrictModel):
    """One Req-14 data-context question the AI must have answered before drafting."""

    question_id: str
    question: str


#: The minimum data-context questions the AI asks before the first draft (Req 14.1).
DATA_CONTEXT_QUESTIONS: list[DataContextQuestion] = [
    DataContextQuestion(
        question_id="label_timing", question="When and how was the label assigned?"
    ),
    DataContextQuestion(
        question_id="entity_repeats",
        question="Can one entity (e.g. a client) appear in more than one row?",
    ),
    DataContextQuestion(
        question_id="post_outcome_columns",
        question="Was any column recorded at or after the time the outcome became known?",
    ),
    DataContextQuestion(
        question_id="time_ordered",
        question="Are rows time-ordered, and which column carries the time?",
    ),
    DataContextQuestion(
        question_id="label_defining_columns",
        question="Was any column used to define the label?",
    ),
]


class DraftSession(StrictModel):
    """A drafting session: the data-context answers plus every retained Script_Draft revision."""

    schema_version: str = FREE_FORM_SCHEMA_VERSION
    draft_id: str
    experiment_id: str
    dataset_uri: str
    data_context: DataContextAnswers = Field(default_factory=DataContextAnswers)
    revisions: list[ScriptRevision] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime

    @property
    def latest_revision(self) -> ScriptRevision | None:
        return self.revisions[-1] if self.revisions else None


class FreeFormRequest(StrictModel):
    """``free_form/<id>/request.json`` — written by the control plane before the job runs."""

    schema_version: str = FREE_FORM_SCHEMA_VERSION
    run_id: str
    experiment_id: str
    approved_script: ApprovedScript
    dataset_uri: str
    file_format: str
    requirements: RequirementsSpec = Field(default_factory=RequirementsSpec)
    compute: ComputeSpec
    target_column: str | None = None
    declared_feature_columns: list[str] = Field(default_factory=list)
    label_columns_for_analysis: list[str] = Field(default_factory=list)
    declared_external_urls: list[str] = Field(default_factory=list)
    data_context: DataContextAnswers = Field(default_factory=DataContextAnswers)
    requested_at: datetime
    requested_by: str | None = None
    execution_id: str | None = None


class FreeFormMetric(StrictModel):
    """One free-form result number, carrying its trust label (Req 3.1)."""

    name: str
    value: float
    label: MetricLabel = MetricLabel.SELF_REPORTED


class FreeFormResult(StrictModel):
    """``free_form/<id>/result.json`` — written only by the job (Req 3, 6)."""

    schema_version: str = FREE_FORM_SCHEMA_VERSION
    run_id: str
    experiment_id: str
    tier: Literal["free_form"] = "free_form"
    status: Literal["completed", "failed"]
    generated_at: datetime
    metrics: list[FreeFormMetric] = Field(default_factory=list)
    produced_model: bool = False
    model_version_name: str | None = Field(
        default=None, description="The models/<name> registered for independent validation."
    )
    content_hash: str = Field(description="The Approved_Script hash this run executed.")
    pip_freeze_uri: str | None = None
    image_digest: str | None = None
    sdk_version: str | None = None
    logs_uri: str | None = None
    instance_type: str | None = None
    used_spot: bool = False
    spot_interrupted: bool = False
    naive_baseline: dict[str, float] = Field(
        default_factory=dict, description="Time-series: the seasonal-naive baseline (Req 5.5)."
    )
    declared_external_urls: list[str] = Field(default_factory=list)
    failure_message: str | None = None

    @property
    def has_engine_metric(self) -> bool:
        return any(m.label is MetricLabel.ENGINE for m in self.metrics)
