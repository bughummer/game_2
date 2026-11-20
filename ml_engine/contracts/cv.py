"""Cross-validation and Test_Fold contracts (ML Factory 2.0, cross-validation design).

CV_Mode runs K folds per Run inside the one training job, then a Refit_Model on the whole
Development_Set. A Test_Fold, when enabled, is carved out first and scored once by the
evaluation job, never used for ranking, tuning or thresholds.

This module imports only ``common``, ``metrics``, ``warnings`` and ``explain`` so ``config``,
``preparation``, ``model`` and ``comparison`` can import it without a cycle.
"""

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from ml_engine.contracts.common import StrictModel
from ml_engine.contracts.explain import PermutationEntry
from ml_engine.contracts.metrics import MetricSet
from ml_engine.contracts.warnings import AnalysisWarning

CV_SCHEMA_VERSION = "1.0"

#: Fold-assignment value of a row in the Test_Fold.
TEST = -1
#: Fold-assignment value of a row that is only ever trained on: forward chaining's earliest
#: block and rows without a parseable time, or the training side in Single_Split_Mode.
TRAIN_ONLY = -2


def is_unset(value: object) -> bool:
    """``exclude_if`` predicate for the optional CV fields added to inherited contracts.

    A field holding its ``None`` default is left out of the JSON, so Single_Split_Mode
    artifacts stay byte-identical to those written before this feature.
    """
    return value is None


class EvaluationMode(StrEnum):
    SINGLE_SPLIT = "single_split"
    CV = "cv"


def is_single_split(value: object) -> bool:
    """``exclude_if`` predicate for ``SplitConfig.evaluation_mode`` at its default."""
    return value == EvaluationMode.SINGLE_SPLIT


class CvScheme(StrEnum):
    STRATIFIED_KFOLD = "stratified_kfold"
    KFOLD = "kfold"
    GROUP_KFOLD = "group_kfold"
    FORWARD_CHAINING = "forward_chaining"


class CrossValidationConfig(StrictModel):
    """Frozen into ``SplitConfig.cv``. The stages read thresholds from here, not from Settings,
    so the instability and outlier flags are reproducible from the config alone."""

    folds: int = Field(ge=2)
    scheme: CvScheme
    instability_std: float = Field(default=0.05, gt=0)
    instability_cv: float = Field(default=0.10, gt=0)
    outlier_multiplier: float = Field(default=3.0, gt=0)
    outlier_min_gap: float = Field(default=0.02, ge=0)


class FoldAssignment(StrictModel):
    """``validation/fold_assignment.json``, written only by the preparation job.

    ``row_ids`` are positions in the loaded dataset; ``assignments`` is aligned with them:
    ``TEST`` (-1), ``TRAIN_ONLY`` (-2) or the CV_Fold index 0..K-1. In Single_Split_Mode
    (``folds == 0``) the value ``0`` marks the validation partition.
    """

    schema_version: str = CV_SCHEMA_VERSION
    experiment_id: str
    generated_at: datetime
    evaluation_mode: EvaluationMode
    scheme: CvScheme | None = Field(description="None in Single_Split_Mode.")
    folds: int = Field(ge=0, description="K; 0 in Single_Split_Mode.")
    test_fraction: float | None
    random_seed: int
    dataset_fingerprint: str = Field(description="sha256 over the usable rows and columns.")
    development_digest: str = Field(description="sha256 of the development rows as written.")
    row_ids: list[int] = Field(description="Ascending and unique.")
    assignments: list[int]
    group_column: str | None = None
    time_column: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "FoldAssignment":
        if len(self.row_ids) != len(self.assignments):
            raise ValueError(
                f"row_ids ({len(self.row_ids)}) and assignments ({len(self.assignments)}) "
                "must have the same length"
            )
        if any(b <= a for a, b in zip(self.row_ids, self.row_ids[1:], strict=False)):
            raise ValueError("row_ids must be strictly ascending")
        cv_mode = self.evaluation_mode is EvaluationMode.CV
        if cv_mode and (self.scheme is None or self.folds < 2):
            raise ValueError("CV_Mode needs a scheme and at least 2 folds")
        if not cv_mode and (self.scheme is not None or self.folds != 0):
            raise ValueError("Single_Split_Mode has no scheme and folds == 0")
        fold_values = set(range(self.folds)) if cv_mode else {0}
        allowed = {TEST, TRAIN_ONLY} | fold_values
        unexpected = sorted(set(self.assignments) - allowed)
        if unexpected:
            raise ValueError(f"assignment values outside {sorted(allowed)}: {unexpected}")
        if cv_mode:
            empty = sorted(fold_values - set(self.assignments))
            if empty:
                raise ValueError(f"CV folds without validation rows: {empty}")
            if self.scheme is not CvScheme.FORWARD_CHAINING and TRAIN_ONLY in self.assignments:
                raise ValueError("TRAIN_ONLY rows are only allowed for forward_chaining")
        return self

    @property
    def development_row_count(self) -> int:
        """Rows outside the Test_Fold."""
        return sum(1 for value in self.assignments if value != TEST)

    @property
    def test_row_count(self) -> int:
        return sum(1 for value in self.assignments if value == TEST)


class FoldRowCounts(StrictModel):
    fold_index: int
    train_row_count: int
    validation_row_count: int
    train_class_distribution: dict[str, int] | None = None
    validation_class_distribution: dict[str, int] | None = None


class CvPreparationSummary(StrictModel):
    """``PreparationReport.cross_validation``."""

    scheme: CvScheme
    folds: int
    development_row_count: int
    train_only_row_count: int = 0
    fold_rows: list[FoldRowCounts]
    fold_output_feature_counts: dict[str, list[int]] = Field(
        default_factory=dict, description="Preprocessing strategy -> output features per fold."
    )


class TestFoldSummary(StrictModel):
    """``PreparationReport.test_fold``."""

    __test__ = False  # not a pytest test class

    fraction: float
    row_count: int
    development_row_count: int
    class_distribution: dict[str, int] | None = None


class FoldResult(StrictModel):
    """``models/{m}/cv/folds/fold_{i}.json``: the completion marker of one fold."""

    schema_version: str = CV_SCHEMA_VERSION
    model_name: str
    fold_index: int
    assignment_digest: str
    params_digest: str
    train_row_count: int
    validation_row_count: int
    metrics: MetricSet
    primary_score: float | None
    best_iteration: int | None = None
    permutation: list[PermutationEntry] = Field(
        default_factory=list, description="This fold's permutation drops."
    )
    permutation_baseline: float | None = None
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    job_name: str | None = Field(default=None, description="SageMaker TRAINING_JOB_NAME.")
    resumed: bool = Field(default=False, description="Loaded from a previous attempt.")
    warnings: list[AnalysisWarning] = Field(default_factory=list)


class FoldMetrics(StrictModel):
    """``models/{m}/cv/fold_metrics.json``."""

    schema_version: str = CV_SCHEMA_VERSION
    experiment_id: str
    model_name: str
    scheme: CvScheme
    folds: list[FoldResult] = Field(description="Exactly K, ordered by fold_index.")


class OofPredictions(StrictModel):
    """``models/{m}/cv/oof.json``: manifest of the out-of-fold Parquet payload."""

    schema_version: str = CV_SCHEMA_VERSION
    experiment_id: str
    model_name: str
    row_count: int
    payload_uri: str
    columns: list[str] = Field(description="row_id, fold_index, y_true, y_pred, proba_<label>...")
    class_labels: list[str] | None = None
    positive_class: str | None = None
    payload_sha256: str


class MetricAggregate(StrictModel):
    mean: float
    std: float
    min: float
    max: float
    fold_count: int


class CvRunSummary(StrictModel):
    """``ModelComparisonEntry.cv`` (the CV_Summary of one Run)."""

    folds: int
    succeeded_folds: int
    metrics: dict[str, MetricAggregate]
    primary_fold_values: list[float | None]
    instability_flag: bool = False
    outlier_folds: list[int] = Field(default_factory=list)


class RefitIterations(StrictModel):
    applicable: bool
    per_fold: list[int] = Field(default_factory=list)
    chosen: int | None = None


class RunCvRecord(StrictModel):
    """``ModelMetadata.cross_validation``."""

    scheme: CvScheme
    folds: int
    performance_source: Literal["cv_estimate"] = "cv_estimate"
    note: str = "Metrics are the mean across CV folds; the Refit_Model itself was not measured."
    primary_fold_values: list[float | None]
    fold_metrics_uri: str
    oof_uri: str
    refit_iterations: RefitIterations


class CvComparisonInfo(StrictModel):
    """``ComparisonReport.cross_validation``."""

    scheme: CvScheme
    folds: int
    ranking: Literal["cv_mean_then_std_then_run_id"] = "cv_mean_then_std_then_run_id"
    test_row_count: int | None = None


class TestRunResult(StrictModel):
    __test__ = False  # not a pytest test class

    model_name: str
    status: Literal["completed", "failed"]
    metrics: dict[str, float] = Field(default_factory=dict)
    error_code: str | None = None
    message: str | None = None


class TestEvaluationReport(StrictModel):
    """``comparison/test_evaluation.json``, written only by the evaluation job."""

    __test__ = False  # not a pytest test class

    schema_version: str = CV_SCHEMA_VERSION
    experiment_id: str
    generated_at: datetime
    evaluation_mode: EvaluationMode
    test_row_count: int
    primary_metric: str
    model_source: Literal["refit_model", "trained_model"]
    runs: list[TestRunResult]
    note: str = "Scored once after selection; never used for ranking, tuning or thresholds."
