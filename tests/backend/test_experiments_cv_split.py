"""ExperimentService CV split resolution and the fold-count guard (CV design C1, Req 1, 3.1).

Every CV check runs in ``_resolve_split`` before ``archive_previous_run`` and before the
orchestrator is called, so a rejected request archives and submits nothing. These tests use a
recording orchestrator (the config is frozen before ``start_training`` is handed to it) and a
failing orchestrator (proves a rejection never reaches submission).
"""

import pytest

from backend.config import Settings
from backend.errors import ValidationError
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import ExperimentService
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.cv import CvScheme, EvaluationMode
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset

FEATURES = ["tenure_months", "monthly_charges", "contract"]


class RecordingOrchestrator:
    """Records EDA/training starts; runs no stages. The frozen config is written before
    ``start_training`` is called, so reading it after a start is enough to test the config."""

    name = "recording"

    def __init__(self) -> None:
        self.eda_started: list[str] = []
        self.training_started: list[str] = []

    def start_eda(self, record):
        from backend.orchestration.base import ExecutionHandle

        self.eda_started.append(record.experiment_id)
        return ExecutionHandle(execution_id=f"rec-eda-{record.experiment_id}")

    def start_training(self, record):
        from backend.orchestration.base import ExecutionHandle

        self.training_started.append(record.experiment_id)
        return ExecutionHandle(execution_id=f"rec-train-{record.experiment_id}")


class FailingOrchestrator(RecordingOrchestrator):
    """Fails loudly if training is ever submitted — a rejected request must never get here."""

    name = "failing"

    def start_training(self, record):
        raise AssertionError("start_training must not be called on a rejected request")


@pytest.fixture
def settings(price_table_path) -> Settings:
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf",
        stage_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/img@sha256:" + "a" * 64,
        price_table_path=price_table_path,
    )


def _service(settings, store, orchestrator):
    repository = ExperimentRepository(settings=settings, store=store)
    return ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )


@pytest.fixture
def ready_experiment(settings, store, classification_frame):
    """An experiment with EDA + feature selection done, and a recording-orchestrator service
    for the start_training call under test. EDA is run by an InlineOrchestrator service so the
    workflow reaches feature review exactly as in production."""
    from tests.support.inline_orchestrator import InlineOrchestrator

    inline = _service(settings, store, InlineOrchestrator(store))
    uri = write_dataset(store, "churn.csv", classification_frame)
    record = inline.create(name="t", dataset_uri=uri, target_column="churned")
    inline.save_features(record.experiment_id, selected_features=FEATURES)

    orchestrator = RecordingOrchestrator()
    service = _service(settings, store, orchestrator)
    return service, orchestrator, record


def _config(store, record) -> ExperimentConfig:
    layout = ExperimentLayout(base=record.artifact_prefix)
    return read_model(store, layout.experiment_config, ExperimentConfig)


# --- defaults ----------------------------------------------------------------------------------


def test_default_request_produces_single_split(ready_experiment, store):
    service, _, record = ready_experiment
    service.start_training(record.experiment_id, TrainingRequest(models=["logistic_regression"]))
    config = _config(store, record)
    assert config.split.evaluation_mode is EvaluationMode.SINGLE_SPLIT
    assert config.split.cv is None
    assert config.split.test_fraction is None


# --- CV config is frozen with the scheme, thresholds and seed ----------------------------------


def test_cv_config_is_frozen_with_scheme_thresholds_and_seed(ready_experiment, store, settings):
    service, _, record = ready_experiment
    service.start_training(
        record.experiment_id,
        TrainingRequest(models=["logistic_regression"], evaluation_mode="cv", random_seed=7),
    )
    config = _config(store, record)
    assert config.split.evaluation_mode is EvaluationMode.CV
    assert config.split.random_seed == 7
    cv = config.split.cv
    assert cv is not None
    assert cv.folds == settings.cv_default_folds == 5
    # stratified_random on a classification problem -> stratified K-fold.
    assert cv.scheme is CvScheme.STRATIFIED_KFOLD
    assert cv.instability_std == settings.cv_instability_std
    assert cv.instability_cv == settings.cv_instability_cv
    assert cv.outlier_multiplier == settings.cv_outlier_multiplier
    assert cv.outlier_min_gap == settings.cv_outlier_min_gap


def test_cv_folds_override_is_recorded(ready_experiment, store):
    service, _, record = ready_experiment
    service.start_training(
        record.experiment_id,
        TrainingRequest(models=["logistic_regression"], evaluation_mode="cv", cv_folds=3),
    )
    assert _config(store, record).split.cv.folds == 3


def test_a_test_fraction_implies_a_test_fold(ready_experiment, store):
    service, _, record = ready_experiment
    service.start_training(
        record.experiment_id,
        TrainingRequest(models=["logistic_regression"], test_fraction=0.3),
    )
    assert _config(store, record).split.test_fraction == 0.3


def test_test_fold_without_a_fraction_uses_the_default(ready_experiment, store, settings):
    service, _, record = ready_experiment
    service.start_training(
        record.experiment_id,
        TrainingRequest(models=["logistic_regression"], test_fold=True),
    )
    assert _config(store, record).split.test_fraction == settings.default_test_fraction == 0.2


# --- rejections: nothing archived or submitted -------------------------------------------------


def _failing_service(settings, store, record):
    orchestrator = FailingOrchestrator()
    repository = ExperimentRepository(settings=settings, store=store)
    return ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )


def _assert_nothing_archived(store, record):
    layout = ExperimentLayout(base=record.artifact_prefix)
    assert store.list_uris(layout.runs_prefix) == []


def test_cv_folds_without_cv_mode_is_rejected(ready_experiment, store, settings):
    _, _, record = ready_experiment
    failing = _failing_service(settings, store, record)
    with pytest.raises(ValidationError) as error:
        failing.start_training(
            record.experiment_id,
            TrainingRequest(models=["logistic_regression"], cv_folds=3),
        )
    assert error.value.details["reason"] == "CV_FOLDS_WITHOUT_CV_MODE"
    _assert_nothing_archived(store, record)


def test_k_below_two_is_rejected_with_the_allowed_range(ready_experiment, store, settings):
    _, _, record = ready_experiment
    failing = _failing_service(settings, store, record)
    with pytest.raises(ValidationError) as error:
        failing.start_training(
            record.experiment_id,
            TrainingRequest(models=["logistic_regression"], evaluation_mode="cv", cv_folds=1),
        )
    details = error.value.details
    assert details["reason"] == "CV_FOLDS_OUT_OF_RANGE"
    assert details["allowed_min"] == 2 and details["allowed_max"] == settings.cv_max_folds
    assert details["requested"] == 1
    _assert_nothing_archived(store, record)


def test_k_above_the_ceiling_is_rejected(ready_experiment, store, settings):
    _, _, record = ready_experiment
    failing = _failing_service(settings, store, record)
    with pytest.raises(ValidationError) as error:
        failing.start_training(
            record.experiment_id,
            TrainingRequest(models=["logistic_regression"], evaluation_mode="cv", cv_folds=11),
        )
    assert error.value.details["reason"] == "CV_FOLDS_OUT_OF_RANGE"
    assert error.value.details["allowed_max"] == 10
    _assert_nothing_archived(store, record)


def test_too_many_fold_fits_is_rejected_with_the_counts(ready_experiment, store, settings):
    _, _, record = ready_experiment
    settings.cv_max_fits = 10  # 3 models x (4 + 1) = 15 > 10
    failing = _failing_service(settings, store, record)
    with pytest.raises(ValidationError) as error:
        failing.start_training(
            record.experiment_id,
            TrainingRequest(
                models=["logistic_regression", "random_forest", "xgboost"],
                evaluation_mode="cv",
                cv_folds=4,
            ),
        )
    details = error.value.details
    assert details["reason"] == "CV_FIT_LIMIT_EXCEEDED"
    assert details["requested_fits"] == 15 and details["allowed_fits"] == 10
    _assert_nothing_archived(store, record)
