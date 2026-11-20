"""Experiment configuration contracts.

``ExperimentConfig`` is frozen at training start and written to
``config/experiment_config.json``. Together with the dataset reference and the captured
environment it is the reproducibility record for the experiment.
"""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from ml_engine.contracts.common import (
    ClassWeighting,
    ProblemType,
    RequestedProblemType,
    StrictModel,
)
from ml_engine.contracts.cv import (
    CrossValidationConfig,
    EvaluationMode,
    is_single_split,
    is_unset,
)
from ml_engine.contracts.ensemble import EnsembleSpec, is_empty_ensembles
from ml_engine.contracts.proposals import FeatureProposal
from ml_engine.contracts.requirements import RequirementsSpec
from ml_engine.contracts.treatment import TreatmentConfig

CONFIG_SCHEMA_VERSION = "1.0"
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_RANDOM_SEED = 42


class DatasetReference(StrictModel):
    """Identity of the dataset an experiment reads.

    ``uri`` is what every stage loads. For a dataset uploaded through the UI that is the
    experiment's own copy, under ``input/<original file name>``, so the experiment carries the
    exact bytes it was trained on; ``source_uri`` then records where those bytes came from. For
    a dataset referenced by URI, ``uri`` is that location and nothing is copied.
    """

    uri: str
    file_format: str
    version_id: str | None = None
    etag: str | None = None
    size_bytes: int | None = None
    last_modified: datetime | None = None
    original_filename: str | None = Field(
        default=None, description="The file name the dataset was uploaded with."
    )
    source_uri: str | None = Field(
        default=None, description="Where the dataset was copied from, when it was copied."
    )


class SplitConfig(StrictModel):
    strategy: str = Field(
        default="stratified_random", description="Registered split strategy name."
    )
    validation_fraction: float = Field(default=DEFAULT_VALIDATION_FRACTION, gt=0.0, lt=1.0)
    random_seed: int = DEFAULT_RANDOM_SEED
    stratify: bool = True
    group_column: str | None = None
    time_column: str | None = Field(
        default=None, description="Reserved for temporal splitting; unused in v1.0."
    )
    # --- ML Factory 2.0: cross-validation and Test_Fold (additive; left out of the JSON at
    # their defaults so single-split configs are byte-identical to pre-feature ones) ---------
    evaluation_mode: EvaluationMode = Field(
        default=EvaluationMode.SINGLE_SPLIT, exclude_if=is_single_split
    )
    cv: CrossValidationConfig | None = Field(
        default=None,
        exclude_if=is_unset,
        description="Folds, scheme and flag thresholds; required iff evaluation_mode is cv.",
    )
    test_fraction: float | None = Field(
        default=None,
        gt=0.0,
        lt=0.5,
        exclude_if=is_unset,
        description="Share of rows carved out as the Test_Fold before CV or the split.",
    )

    @model_validator(mode="after")
    def _cv_matches_mode(self) -> "SplitConfig":
        if self.evaluation_mode is EvaluationMode.CV and self.cv is None:
            raise ValueError("evaluation_mode 'cv' needs a cross-validation configuration")
        if self.evaluation_mode is not EvaluationMode.CV and self.cv is not None:
            raise ValueError("a cross-validation configuration needs evaluation_mode 'cv'")
        return self


class PreprocessingConfig(StrictModel):
    numeric_imputation: str = Field(default="median", description="median | mean | constant")
    numeric_fill_value: float = 0.0
    categorical_imputation: str = Field(default="constant", description="constant | most_frequent")
    categorical_fill_value: str = "__missing__"
    scale_numeric: bool = Field(
        default=True, description="Applied only for plugins that require dense scaled input."
    )
    one_hot_max_categories: int = Field(default=50, ge=2)
    one_hot_min_frequency: float | None = Field(default=None, ge=0.0, lt=1.0)
    datetime_features: list[str] = Field(default_factory=lambda: ["year", "month", "day_of_week"])
    derive_datetime_features: bool = True
    drop_text_columns: bool = True
    drop_constant_columns: bool = True

    @field_validator("numeric_imputation")
    @classmethod
    def _check_numeric_imputation(cls, value: str) -> str:
        allowed = {"median", "mean", "constant"}
        if value not in allowed:
            raise ValueError(f"numeric_imputation must be one of {sorted(allowed)}")
        return value

    @field_validator("categorical_imputation")
    @classmethod
    def _check_categorical_imputation(cls, value: str) -> str:
        allowed = {"constant", "most_frequent"}
        if value not in allowed:
            raise ValueError(f"categorical_imputation must be one of {sorted(allowed)}")
        return value


class ModelSpec(StrictModel):
    """A model to train, plus any user overrides of the plugin defaults."""

    name: str
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    enabled: bool = True


class FeatureSelection(StrictModel):
    """The exact feature decision made by the user. Persisted verbatim."""

    selected_features: list[str]
    excluded_features: list[str] = Field(default_factory=list)
    exclusion_reasons: dict[str, str] = Field(default_factory=dict)
    decided_at: datetime | None = None
    decided_by: str | None = None

    @model_validator(mode="after")
    def _no_overlap(self) -> "FeatureSelection":
        overlap = set(self.selected_features) & set(self.excluded_features)
        if overlap:
            raise ValueError(f"features cannot be both selected and excluded: {sorted(overlap)}")
        if not self.selected_features:
            raise ValueError("at least one feature must be selected")
        return self


class TrainingConfig(StrictModel):
    """User-facing training configuration, validated before the workflow starts."""

    problem_type: RequestedProblemType = RequestedProblemType.AUTO
    primary_metric: str | None = Field(
        default=None, description="None resolves to the default metric for the problem type."
    )
    split: SplitConfig = Field(default_factory=SplitConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    models: list[ModelSpec] = Field(default_factory=list)
    class_weighting: ClassWeighting = ClassWeighting.AUTO

    @property
    def enabled_models(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enabled]


class EnvironmentCapture(StrictModel):
    """Everything needed to explain why a run produced the numbers it produced."""

    python_version: str
    platform: str
    package_versions: dict[str, str] = Field(default_factory=dict)
    ml_factory_version: str
    captured_at: datetime


class ExperimentConfig(StrictModel):
    """Frozen experiment definition. Written once, at training start."""

    schema_version: str = CONFIG_SCHEMA_VERSION
    experiment_id: str
    name: str
    created_at: datetime
    created_by: str | None = None
    dataset: DatasetReference
    target_column: str
    problem_type: ProblemType
    requested_problem_type: RequestedProblemType
    primary_metric: str
    feature_selection: FeatureSelection
    derived_features: list[FeatureProposal] = Field(
        default_factory=list,
        description=(
            "Approved derived-feature specifications, applied during preparation. Stored here "
            "rather than as code so the run can be reproduced exactly."
        ),
    )
    split: SplitConfig
    preprocessing: PreprocessingConfig
    models: list[ModelSpec]
    class_weighting: ClassWeighting
    environment: EnvironmentCapture
    artifact_prefix: str
    # --- ML Factory 2.0: per-experiment package install (stock-image migration, Stage 2).
    # Omitted from the JSON when unset, so configs without requirements stay byte-identical. ---
    requirements: RequirementsSpec | None = Field(default=None, exclude_if=is_unset)
    # --- ML Factory 2.0: training-time data treatment (outlier handling). Optional and omitted
    # from the JSON at the 'none' default, so configs without treatment stay byte-identical. ----
    treatment: TreatmentConfig | None = Field(default=None, exclude_if=is_unset)
    # --- ML Factory 2.0: user-approved ensembles (voting/stacking). Each is expanded into a
    # ModelSpec in `models`; the frozen specs are kept here. Empty = today, omitted from JSON. ---
    ensembles: list[EnsembleSpec] = Field(default_factory=list, exclude_if=is_empty_ensembles)

    def enabled_models_specs(self) -> list[ModelSpec]:
        return [m for m in self.models if m.enabled]
