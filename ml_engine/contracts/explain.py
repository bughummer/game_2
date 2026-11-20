"""Post-training explanation contracts: permutation importance and feature ablation.

Both are deterministic measurements on the validation fold, computed in SageMaker jobs, and both
feed the post-training leakage phase. They answer different questions and are kept apart:

* permutation importance — how much the *trained* model's validation score drops when one input
  column's values are shuffled (no retraining). Cheap, model-agnostic, per input column.
* ablation — how much the score drops when the model is *retrained* without the column on the
  identical folds. Expensive, but it is the direct "with vs. without" comparison.

Neither proves leakage: a large drop says the model depends on the column, not that the column
is unavailable at prediction time.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import MetricDirection, StrictModel

EXPLAIN_SCHEMA_VERSION = "1.0"


def _unset(value: object) -> bool:
    """Optional CV fields are left out of the JSON when unset (single-split bytes unchanged).

    Local rather than imported from ``cv``: ``cv`` imports this module.
    """
    return value is None


class PermutationEntry(StrictModel):
    feature: str
    #: Mean drop in the primary metric (positive = the model is worse without the information).
    mean_drop: float
    std_drop: float
    #: This feature's share of the summed positive drops, 0..1.
    share: float = Field(ge=0.0, le=1.0)


class PermutationImportance(StrictModel):
    schema_version: str = EXPLAIN_SCHEMA_VERSION
    experiment_id: str
    model_name: str
    generated_at: datetime
    primary_metric: str
    direction: MetricDirection
    baseline_score: float
    repeats: int
    random_seed: int
    fold: str = "validation"
    entries: list[PermutationEntry] = Field(default_factory=list)
    truncated_features: int = Field(
        default=0, description="Input columns not measured because of the feature cap."
    )
    # CV_Mode (fold="cv"): entries are the mean ± std across folds of each fold's drop.
    fold_count: int | None = Field(
        default=None, exclude_if=_unset, description="Folds the aggregate is built from."
    )
    fold_baselines: list[float] | None = Field(
        default=None, exclude_if=_unset, description="Baseline score of each fold's model."
    )


class AblationEntry(StrictModel):
    feature: str
    score_without: float | None = None
    #: Baseline minus score-without, oriented so positive always means worse without it.
    drop: float | None = None
    status: str = Field(default="completed", description="'completed' or 'failed'.")
    message: str | None = None
    # CV_Mode (fold="cv"): score_without and drop are fold means.
    fold_drops: list[float] | None = Field(
        default=None, exclude_if=_unset, description="Drop on each CV fold."
    )
    std_drop: float | None = Field(
        default=None, exclude_if=_unset, description="Sample std of the fold drops."
    )


class AblationReport(StrictModel):
    schema_version: str = EXPLAIN_SCHEMA_VERSION
    experiment_id: str
    generated_at: datetime
    model_name: str
    primary_metric: str
    direction: MetricDirection
    #: Retrained in the same job with every feature, so baseline and ablations share one
    #: procedure, one seed and the identical train/validation folds (requirement 13.5).
    baseline_score: float
    entries: list[AblationEntry] = Field(default_factory=list)
    fold: str = "validation"
    note: str = (
        "Each feature was removed and the model retrained on the identical folds. The holdout "
        "test set, if any, was not used."
    )
    fold_count: int | None = Field(
        default=None, exclude_if=_unset, description="CV_Mode: folds each score averages."
    )
    truncated: int = Field(
        default=0,
        description="CV_Mode: candidate features dropped to keep (candidates + 1) x K <= 60.",
    )
