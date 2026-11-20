"""Evaluation of already-trained models on a separate dataset (ML Factory 2.0).

After training, a person can score the trained models on another file — a held-out test set,
a later time period, or the full population. The same fitted preprocessing and models are
applied unchanged; nothing is re-fitted, so the file cannot influence the models.

Scoring rows the model was trained on inflates every metric. So the rows of the file are
compared with the training and validation folds, and metrics are reported twice: on all rows,
and on the rows the model has never seen. The unseen-rows numbers are the honest ones.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from ml_engine.contracts.common import ProblemType, StrictModel
from ml_engine.contracts.metrics import MetricSet
from ml_engine.contracts.thresholds import ThresholdPoint
from ml_engine.contracts.warnings import AnalysisWarning

HOLDOUT_SCHEMA_VERSION = "1.0"


class HoldoutRequest(StrictModel):
    """Written by the control plane when an evaluation is started."""

    schema_version: str = HOLDOUT_SCHEMA_VERSION
    evaluation_id: str
    experiment_id: str
    dataset_uri: str
    file_format: str
    dataset_etag: str | None = None
    dataset_version_id: str | None = None
    requested_at: datetime
    requested_by: str | None = None
    execution_id: str | None = None


class RowOverlap(StrictModel):
    """How many rows of the file also appear in the training or validation fold."""

    compared_columns: list[str] = Field(
        description="Model input columns plus the target; rows equal on all of them match."
    )
    rows_in_training_fold: int
    rows_in_validation_fold: int
    unseen_rows: int
    overlap_fraction: float = Field(description="Share of rows found in either fold.")


class HoldoutModelResult(StrictModel):
    model_name: str
    display_name: str
    status: Literal["completed", "failed"]
    primary_metric: str
    validation_score: float | None = Field(
        default=None, description="Primary metric on the training run's validation fold."
    )
    all_rows_score: float | None = None
    unseen_rows_score: float | None = None
    metrics_all_rows: MetricSet | None = None
    metrics_unseen_rows: MetricSet | None = Field(
        default=None, description="Null when too few unseen rows (or one class only) remain."
    )
    at_validation_best_threshold: ThresholdPoint | None = Field(
        default=None,
        description=(
            "Binary only: precision/recall/F1 at the threshold that maximised F1 on the "
            "validation fold, measured here — this is how that threshold holds up on new data."
        ),
    )
    threshold_rows: Literal["unseen", "all"] | None = None
    failure_message: str | None = None


class HoldoutEvaluationReport(StrictModel):
    schema_version: str = HOLDOUT_SCHEMA_VERSION
    evaluation_id: str
    experiment_id: str
    generated_at: datetime
    problem_type: ProblemType
    target_column: str
    dataset_uri: str
    file_format: str
    row_count: int = Field(description="Rows evaluated (after dropping rows without a target).")
    rows_dropped_missing_target: int
    target_rate: float | None = Field(
        default=None, description="Binary only: positive-class share in this file."
    )
    training_target_rate: float | None = None
    overlap: RowOverlap
    models: list[HoldoutModelResult] = Field(default_factory=list)
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    image_uri: str | None = Field(default=None, description="Image this evaluation ran on.")
    training_image_uri: str | None = Field(
        default=None, description="Image the models were trained on, when recorded."
    )
    environment_differences: dict[str, str] = Field(
        default_factory=dict,
        description="Packages whose version differs from training: name -> 'training -> now'.",
    )
