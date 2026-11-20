"""Training-time data-treatment contracts (training-data-treatment design C1).

Outlier handling is a user-chosen, approved, fold-safe step at the *front* of the preprocessing
chain: outlier handling -> imputation -> scaling -> encoding -> model fit. Because the existing
``FeaturePipeline`` is a column-wise ``ColumnTransformer`` that cannot drop rows or touch the
target, outlier handling is a separate fitted object (``OutlierTreatment``, see
``ml_engine/preprocessing/outliers.py``) composed around the pipeline, driven by the versioned
contracts here.

Everything is deliberately additive: ``TreatmentConfig`` defaults to a ``none`` detector, which is
byte-for-byte today's chain, and ``ExperimentConfig.treatment`` is omitted from the JSON at that
default (so pre-feature configs stay identical). Imputation (``PreprocessingConfig``) and
``ClassWeighting`` are *surfaced* here, not duplicated: they stay where they already live on the
training config.
"""

from enum import StrEnum

from pydantic import Field, model_validator

from ml_engine.contracts.common import StrictModel

TREATMENT_SCHEMA_VERSION = "1.0"

#: A ``drop`` must leave at least this many training rows in a fold, else preparation fails.
MIN_TRAINING_ROWS_AFTER_DROP = 10


class OutlierDetector(StrEnum):
    """How outlier rows are identified from feature values. ``none`` changes nothing."""

    NONE = "none"
    IQR = "iqr"
    ZSCORE = "zscore"
    ISOLATION_FOREST = "isolation_forest"

    @property
    def is_per_column(self) -> bool:
        """Per-column detectors (``iqr``/``zscore``) support ``cap``; multivariate ones do not."""
        return self in (OutlierDetector.IQR, OutlierDetector.ZSCORE)


class OutlierHandling(StrEnum):
    """What is done with a detected outlier row."""

    FLAG = "flag"
    CAP = "cap"
    DROP = "drop"


class TreatmentConfig(StrictModel):
    """The user's chosen, approved data treatment, frozen into the experiment configuration.

    The imputation strategy (``PreprocessingConfig``) and ``ClassWeighting`` are *not* duplicated
    here; they stay on the training config and are surfaced alongside this. A ``none`` detector
    reproduces today's behaviour exactly.
    """

    schema_version: str = TREATMENT_SCHEMA_VERSION
    outlier_detector: OutlierDetector = OutlierDetector.NONE
    outlier_handling: OutlierHandling = OutlierHandling.FLAG
    zscore_threshold: float = Field(
        default=3.0, gt=0.0, description="|z| above this is an outlier."
    )
    iqr_multiplier: float = Field(
        default=1.5, gt=0.0, description="Fence = Q1/Q3 -/+ multiplier * IQR."
    )
    isolation_contamination: float = Field(
        default=0.05, gt=0.0, lt=0.5, description="Expected outlier share for isolation_forest."
    )

    @property
    def is_noop(self) -> bool:
        return self.outlier_detector is OutlierDetector.NONE

    @model_validator(mode="after")
    def _handling_matches_detector(self) -> "TreatmentConfig":
        # cap winsorises to a per-column fence; a multivariate detector has no single-column
        # bound to cap to (design open q 5), so cap + isolation_forest is refused here.
        if (
            self.outlier_handling is OutlierHandling.CAP
            and not self.outlier_detector.is_per_column
            and self.outlier_detector is not OutlierDetector.NONE
        ):
            raise ValueError(
                f"outlier_handling 'cap' needs a per-column detector (iqr/zscore); "
                f"'{self.outlier_detector.value}' is multivariate. Use 'flag' or 'drop'."
            )
        return self


class OutlierBounds(StrictModel):
    """The fitted state of an ``OutlierTreatment`` — enough to reproduce the treatment.

    ``column_fences`` holds the per-column lower/upper fences for ``iqr``/``zscore``. For
    ``isolation_forest`` the fences are empty and ``forest_seed``/``forest_contamination`` record
    the seeded fit; the fitted estimator itself is persisted as a joblib artifact alongside.
    """

    schema_version: str = TREATMENT_SCHEMA_VERSION
    detector: OutlierDetector = OutlierDetector.NONE
    column_fences: dict[str, tuple[float, float]] = Field(
        default_factory=dict, description="column -> (lower, upper) fence (iqr/zscore)."
    )
    forest_seed: int | None = None
    forest_contamination: float | None = None


class OutlierProposal(StrictModel):
    """A deterministic, per-detector summary shown to the user before approval (no apply).

    Counts and bounds are computed on the Development_Set so the user can judge each candidate
    detector with evidence. No number here comes from the LLM.
    """

    schema_version: str = TREATMENT_SCHEMA_VERSION
    detector: OutlierDetector
    rows_considered: int
    rows_flagged: int
    affected_columns: list[str] = Field(default_factory=list)
    bounds: OutlierBounds


class AppliedTreatment(StrictModel):
    """What a fit actually did, recorded per fold in the preparation report for audit (Req 6.6)."""

    schema_version: str = TREATMENT_SCHEMA_VERSION
    fitted_on: str = Field(description="e.g. 'train', 'fold_0_train', 'development'.")
    detector: OutlierDetector
    handling: OutlierHandling
    rows_considered: int
    rows_flagged: int = 0
    rows_capped: int = 0
    rows_dropped: int = 0
    affected_columns: list[str] = Field(default_factory=list)
