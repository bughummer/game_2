"""Control-plane request/response shapes that are not themselves artifacts.

Artifact-shaped payloads reuse ``ml_engine.contracts`` verbatim; only genuinely API/MCP-specific
shapes live here. Ported and trimmed from old_repo/backend/schemas.
"""

from datetime import datetime
from typing import Literal

from pydantic import Field

from ml_engine.contracts.common import ClassWeighting, ProblemType, StrictModel
from ml_engine.contracts.config import PreprocessingConfig
from ml_engine.contracts.ensemble import EnsembleType, VotingRule
from ml_engine.contracts.holdout import HoldoutEvaluationReport
from ml_engine.contracts.treatment import OutlierDetector, OutlierHandling


class DatasetColumnPreview(StrictModel):
    name: str
    dtype: str
    non_null_in_sample: int
    sample_values: list[str] = Field(default_factory=list)


class UploadTicket(StrictModel):
    """A pre-signed S3 PUT the caller uploads the dataset bytes to, then registers."""

    dataset_uri: str
    upload_url: str
    method: str = "PUT"
    headers: dict[str, str] = Field(default_factory=dict)
    expires_at: datetime
    max_bytes: int


class RegisteredDataset(StrictModel):
    dataset_uri: str
    filename: str
    file_format: str
    size_bytes: int
    columns: list[DatasetColumnPreview]
    preview_row_count: int
    truncated_preview: bool
    registered_at: datetime


class DatasetSummary(StrictModel):
    """One uploaded dataset in the inbox, enough to reuse it in a new experiment."""

    dataset_uri: str
    filename: str
    file_format: str
    size_bytes: int
    last_modified: datetime | None = None


class DatasetList(StrictModel):
    """Datasets previously uploaded to this platform, newest first."""

    count: int
    datasets: list[DatasetSummary] = Field(default_factory=list)


class FeatureReviewItem(StrictModel):
    feature: str
    selected: bool
    semantic_type: str
    dtype: str
    missing_percentage: float
    unique_count: int
    leakage_risk: str
    recommended_action: str
    reasons: list[str] = Field(default_factory=list)
    origin: str = Field(default="original", description="'original' column or 'derived' feature.")
    status: str | None = Field(
        default=None,
        description=(
            "Derived features only: 'approved' (profiled, will be built at training) or 'built' "
            "(a training run has computed it)."
        ),
    )


class FeatureReview(StrictModel):
    experiment_id: str
    target_column: str
    problem_type: ProblemType | None
    features: list[FeatureReviewItem]
    selected_count: int
    excluded_count: int
    decided: bool = False


class EnsembleBaseLearner(StrictModel):
    """One base learner in an ensemble request."""

    name: str
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    weight: float = Field(default=1.0, gt=0.0)


class EnsembleRequest(StrictModel):
    """A user-approved ensemble to train (ensembles-and-learners design C6)."""

    type: EnsembleType
    base_learners: list[EnsembleBaseLearner] = Field(default_factory=list)
    voting_rule: VotingRule = VotingRule.SOFT
    meta_learner: str | None = Field(
        default=None, description="Stacking only; None resolves to the problem-type default."
    )
    meta_learner_params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class TrainingRequest(StrictModel):
    problem_type: str | None = None
    primary_metric: str | None = None
    models: list[str] = Field(
        default_factory=list, description="Empty means every supported model."
    )
    model_parameters: dict[str, dict[str, float | int | str | bool | None]] = Field(
        default_factory=dict,
        description=(
            "Per-model hyperparameter overrides, e.g. {'xgboost': {'scale_pos_weight': 5}}. "
            "They replace the model's defaults, including the automatic class weights."
        ),
    )
    class_weighting: ClassWeighting = Field(
        default=ClassWeighting.AUTO,
        description="'auto' balances classes in every model that supports it; 'none' disables.",
    )
    validation_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    random_seed: int = 42
    preprocessing: PreprocessingConfig | None = None
    split_strategy: str | None = Field(
        default=None,
        description=(
            "random | stratified_random | group | temporal. None picks: 'group' when the "
            "business context names an entity id column, else the problem-type default."
        ),
    )
    group_column: str | None = Field(
        default=None, description="Keeps every row of one entity in one fold (group split)."
    )
    time_column: str | None = Field(
        default=None, description="Validates on the most recent rows (temporal split)."
    )
    # --- ML Factory 2.0: cross-validation and Test_Fold ------------------------------------
    evaluation_mode: Literal["single_split", "cv"] | None = Field(
        default=None,
        description="None applies the configured default (single_split). 'cv' runs K folds.",
    )
    cv_folds: int | None = Field(
        default=None,
        description="K for CV_Mode; only with evaluation_mode='cv'. None uses the default K.",
    )
    test_fold: bool = Field(
        default=False, description="Carve out an untouched Test_Fold scored once at the end."
    )
    test_fraction: float | None = Field(
        default=None,
        gt=0.0,
        lt=0.5,
        description="Test_Fold share; set implies test_fold. Omitted with test_fold uses the "
        "configured default.",
    )
    # --- ML Factory 2.0: per-experiment package install (stock-image migration, Stage 2) ----
    requirements: list[str] = Field(
        default_factory=list,
        description=(
            "User-approved pip requirements (the experiment's requirements.txt) installed on the "
            "stock image before training, e.g. ['catboost>=1.2', 'lightgbm']. Empty installs "
            "nothing and behaves as today."
        ),
    )
    # --- ML Factory 2.0: training-time data treatment (outlier handling). Absence = 'none' =
    # today. Imputation (preprocessing) and class_weighting above are surfaced, not re-added. --
    outlier_detector: OutlierDetector | None = Field(
        default=None,
        description=(
            "none | iqr | zscore | isolation_forest. None/none leaves the data untreated "
            "(today's behaviour). The treatment is fit on training rows only, per fold under CV."
        ),
    )
    outlier_handling: OutlierHandling | None = Field(
        default=None,
        description=(
            "flag | cap | drop. Only applies when an outlier_detector is chosen; defaults to "
            "'flag' (record only). 'cap' needs a per-column detector (iqr/zscore)."
        ),
    )
    zscore_threshold: float | None = Field(
        default=None, gt=0.0, description="|z| above this is an outlier (zscore detector)."
    )
    iqr_multiplier: float | None = Field(
        default=None, gt=0.0, description="Fence = Q1/Q3 -/+ multiplier * IQR (iqr detector)."
    )
    isolation_contamination: float | None = Field(
        default=None,
        gt=0.0,
        lt=0.5,
        description="Expected outlier share (isolation_forest detector).",
    )
    # --- ML Factory 2.0: grid search over model parameters (grid-search spec) ---------------
    parameter_grid: dict[str, dict[str, list[float | int | str | bool]]] | None = Field(
        default=None,
        description=(
            "Per-model grid over hyperparameters, e.g. "
            "{'xgboost': {'max_depth': [4, 6, 8], 'reg_alpha': [0, 0.1]}}. Each combination is "
            "trained and saved as its own model version. None trains one combination per model "
            "(today's behaviour)."
        ),
    )
    # --- ML Factory 2.0: ensembles (voting/stacking). Absent = today's single-learner training. -
    ensembles: list[EnsembleRequest] = Field(
        default_factory=list,
        description=(
            "User-approved voting/stacking ensembles. Each is trained and saved as its own model "
            "version alongside the single learners; empty trains only the single learners."
        ),
    )


class PreflightCheck(StrictModel):
    """What stands between an experiment and a trustworthy training run. Advisory except
    ``blockers``, which training refuses on."""

    experiment_id: str
    ready: bool
    blockers: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unresolved_leakage_warnings: list[dict[str, str]] = Field(default_factory=list)
    suggested_split: str | None = None


class JobStatusView(StrictModel):
    job_name: str
    stage: str
    model_name: str | None = None
    state: str
    normalized_status: str | None = None
    raw_status: str | None = None
    secondary_status: str | None = None
    failure_reason: str | None = None
    diagnosis_category: str | None = None
    diagnosis_summary: str | None = None
    recommended_action: str | None = None


class ExecutionStatusView(StrictModel):
    execution_id: str
    kind: str
    status: str
    stage: str
    started_at: datetime
    updated_at: datetime
    failure_code: str | None = None
    failure_message: str | None = None
    jobs: list[JobStatusView] = Field(default_factory=list)


class ModelRunState(StrictModel):
    model_name: str
    display_name: str
    status: str
    primary_score: float | None = None
    failure_message: str | None = None


class TrainingStatus(StrictModel):
    experiment_id: str
    status: str
    current_stage: str
    normalized_jobs: list[str] = Field(default_factory=list)
    models: list[ModelRunState] = Field(default_factory=list)
    best_model: str | None = None
    best_score: float | None = None
    primary_metric: str | None = None
    failure_message: str | None = None
    updated_at: datetime


class ProposeFeaturesRequest(StrictModel):
    """Business context the profile cannot supply. Both fields are optional but both help.

    When omitted, the proposal service falls back to the experiment's stored business context.
    """

    target_definition: str | None = Field(
        default=None,
        max_length=2000,
        description="What the target actually means in business terms.",
    )
    prediction_timing: str | None = Field(
        default=None,
        max_length=2000,
        description=(
            "When the prediction is made relative to the outcome. This is what lets the model "
            "judge whether a proposed feature would be known at that moment."
        ),
    )


class ApproveProposalsRequest(StrictModel):
    """The candidates a person chose. An empty list means 'none of them', and is respected."""

    approved_names: list[str] = Field(default_factory=list)


class HoldoutStatus(StrictModel):
    """One evaluation on a separate dataset: where it stands, and its report when finished."""

    evaluation_id: str
    status: str = Field(description="'running', 'completed' or 'failed'.")
    dataset_uri: str
    requested_at: datetime
    execution_id: str | None = None
    report: HoldoutEvaluationReport | None = None
    failure_message: str | None = None
    other_evaluation_ids: list[str] = Field(
        default_factory=list, description="Earlier evaluations of this experiment, newest first."
    )
