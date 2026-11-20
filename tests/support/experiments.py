"""Builders for experiment records and frozen configurations used across tests."""

from datetime import UTC, datetime

from ml_engine.contracts.common import ClassWeighting, ProblemType, RequestedProblemType
from ml_engine.contracts.config import (
    DatasetReference,
    EnvironmentCapture,
    ExperimentConfig,
    FeatureSelection,
    ModelSpec,
    PreprocessingConfig,
    SplitConfig,
)
from ml_engine.contracts.cv import CrossValidationConfig, CvScheme, EvaluationMode
from ml_engine.contracts.experiment import ExperimentRecord
from ml_engine.io import ExperimentLayout

FEATURES = ["tenure_months", "monthly_charges", "contract", "is_business", "signup_date"]


def make_record(layout: ExperimentLayout, experiment_id: str, dataset_uri: str) -> ExperimentRecord:
    now = datetime.now(UTC)
    return ExperimentRecord(
        experiment_id=experiment_id,
        name="test",
        created_by="tester",
        created_at=now,
        updated_at=now,
        dataset=DatasetReference(uri=dataset_uri, file_format="csv"),
        target_column="churned",
        artifact_prefix=layout.base,
    )


def make_config(
    layout: ExperimentLayout,
    experiment_id: str,
    dataset_uri: str,
    models: list[str],
    *,
    split: SplitConfig | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        experiment_id=experiment_id,
        name="test",
        created_at=datetime.now(UTC),
        dataset=DatasetReference(uri=dataset_uri, file_format="csv"),
        target_column="churned",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        requested_problem_type=RequestedProblemType.AUTO,
        primary_metric="roc_auc",
        feature_selection=FeatureSelection(selected_features=FEATURES),
        split=split or SplitConfig(),
        preprocessing=PreprocessingConfig(),
        models=[ModelSpec(name=name) for name in models],
        class_weighting=ClassWeighting.AUTO,
        environment=EnvironmentCapture(
            python_version="3.14",
            platform="test",
            ml_factory_version="2.0.0",
            captured_at=datetime.now(UTC),
        ),
        artifact_prefix=layout.base,
    )


def cv_split(folds: int = 5, scheme: CvScheme = CvScheme.STRATIFIED_KFOLD) -> SplitConfig:
    """A CV_Mode SplitConfig for orchestrator/stage tests."""
    return SplitConfig(
        evaluation_mode=EvaluationMode.CV,
        cv=CrossValidationConfig(folds=folds, scheme=scheme),
    )
