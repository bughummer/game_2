"""Preparation (validation + split + preprocessing fit) output schema."""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import ProblemType, StrictModel
from ml_engine.contracts.cv import CvPreparationSummary, TestFoldSummary, is_unset
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.model import PreprocessingMetadata
from ml_engine.contracts.proposals import DerivedFeature, RejectedProposal
from ml_engine.contracts.treatment import AppliedTreatment
from ml_engine.contracts.warnings import AnalysisWarning

PREPARATION_SCHEMA_VERSION = "1.0"


class SplitSummary(StrictModel):
    strategy: str
    train_row_count: int
    validation_row_count: int
    validation_fraction_actual: float
    random_seed: int
    stratified: bool
    train_class_distribution: dict[str, int] | None = None
    validation_class_distribution: dict[str, int] | None = None


class PreparationReport(StrictModel):
    schema_version: str = PREPARATION_SCHEMA_VERSION
    experiment_id: str
    generated_at: datetime
    problem_type: ProblemType
    target_column: str
    requested_features: list[str]
    usable_features: list[str]
    derived_features: list[DerivedFeature] = Field(
        default_factory=list, description="Approved proposals that were computed onto the frame."
    )
    rejected_proposals: list[RejectedProposal] = Field(
        default_factory=list,
        description=(
            "Approved proposals that preparation refused to compute, with the reason. A "
            "specification frozen against an earlier version of the dataset can stop being "
            "valid; that must be visible, not silent."
        ),
    )
    derived_leakage: LeakageReport | None = Field(
        default=None,
        description=(
            "Leakage screening of the derived columns against the target. Derived features are "
            "created after the EDA job has run, so they are screened here instead."
        ),
    )
    dropped_features: dict[str, str] = Field(
        default_factory=dict, description="feature -> reason it could not be used"
    )
    rows_before: int
    rows_after: int
    rows_dropped_missing_target: int
    split: SplitSummary
    preprocessing: dict[str, PreprocessingMetadata] = Field(
        default_factory=dict, description="Preprocessing strategy name -> what that pipeline did."
    )
    warnings: list[AnalysisWarning] = Field(default_factory=list)
    # --- ML Factory 2.0: cross-validation and Test_Fold (additive, omitted when unset) ------
    cross_validation: CvPreparationSummary | None = Field(
        default=None, exclude_if=is_unset, description="Per-fold row counts in CV_Mode."
    )
    test_fold: TestFoldSummary | None = Field(
        default=None, exclude_if=is_unset, description="The Test_Fold, when one was carved."
    )
    # --- ML Factory 2.0: training-time data treatment (outlier handling). Omitted from the JSON
    # when empty (no treatment), so a 'none' run stays byte-identical to the pre-feature fixtures.
    applied_treatment: list[AppliedTreatment] = Field(
        default_factory=list,
        exclude_if=lambda value: not value,
        description="Per-fit summary of outlier handling actually applied, for audit (Req 6.6).",
    )
