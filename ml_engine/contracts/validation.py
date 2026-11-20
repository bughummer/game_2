"""Validation on an uploaded labeled dataset (control-ui-api design C5).

Generalises holdout evaluation to target a chosen set of Model_Versions (or all of them): the
labels, when present, are excluded from the model input and kept for metrics and a fresh threshold
analysis computed on the validation labels. One SageMaker job loops the versions and writes one
``ValidationReport`` with a result per version. Nothing is re-fit on the validation data.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from ml_engine.contracts.common import ProblemType, StrictModel
from ml_engine.contracts.holdout import RowOverlap
from ml_engine.contracts.metrics import MetricSet
from ml_engine.contracts.thresholds import ThresholdAnalysis, ThresholdPoint
from ml_engine.contracts.warnings import AnalysisWarning

VALIDATION_SCHEMA_VERSION = "1.0"


class ValidationRequest(StrictModel):
    """``validation/<id>/request.json`` — written by the control plane before the job runs."""

    schema_version: str = VALIDATION_SCHEMA_VERSION
    validation_id: str
    experiment_id: str
    dataset_uri: str
    file_format: str
    #: The Model_Versions to score. Empty means every version of the experiment.
    version_ids: list[str] = Field(default_factory=list)
    requested_at: datetime
    requested_by: str | None = None
    execution_id: str | None = None


class ValidationModelResult(StrictModel):
    """One Model_Version scored on the validation dataset."""

    version_id: str
    model_name: str
    run_id: str | None = None
    status: Literal["completed", "failed"]
    primary_metric: str
    #: Metrics on every row and on rows not seen in training; absent when labels are missing.
    metrics_all_rows: MetricSet | None = None
    metrics_unseen_rows: MetricSet | None = None
    all_rows_score: float | None = None
    unseen_rows_score: float | None = None
    #: Binary + labels present: a fresh threshold analysis computed on the validation labels.
    threshold_analysis: ThresholdAnalysis | None = None
    #: Binary + labels present: the point at the training-chosen F1 threshold, on this data.
    at_validation_best_threshold: ThresholdPoint | None = None
    threshold_rows: Literal["unseen", "all"] | None = None
    #: Prediction distribution, always produced (the only output when labels are absent).
    prediction_summary: dict[str, float] = Field(default_factory=dict)
    failure_message: str | None = None


class ValidationReport(StrictModel):
    """``validation/<id>/validation.json`` — written only by the validation job."""

    schema_version: str = VALIDATION_SCHEMA_VERSION
    validation_id: str
    experiment_id: str
    generated_at: datetime
    problem_type: ProblemType
    target_column: str
    dataset_uri: str
    file_format: str
    labels_present: bool
    row_count: int
    rows_dropped_missing_target: int = 0
    overlap: RowOverlap | None = None
    results: list[ValidationModelResult] = Field(default_factory=list)
    image_uri: str | None = None
    training_image_uri: str | None = None
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    note: str = "Scored without refitting; labels (when present) are used only for metrics."
