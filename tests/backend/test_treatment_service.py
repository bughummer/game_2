"""Service-layer freeze + validation of the outlier treatment (training-data-treatment C6).

The TreatmentConfig is frozen into ExperimentConfig before submission; an invalid combination
(cap + isolation_forest) is refused before any job starts; absence stays byte-identical.
"""

import pytest

from backend.config import Settings
from backend.errors import ValidationError
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import ExperimentService
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.treatment import OutlierDetector, OutlierHandling
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore

FEATURES = ["tenure_months", "monthly_charges", "contract", "is_business", "signup_date"]
TARGET = "churned"


def _service(store, **overrides):
    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf-sagemaker-job",
        **overrides,
    )
    repository = ExperimentRepository(settings=settings, store=store)
    from tests.support.inline_orchestrator import InlineOrchestrator

    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    return experiments, orchestrator


def _ready(experiments, store, name="churn.csv"):
    uri = write_dataset(store, name, golden_frame())
    record = experiments.create(name="t", dataset_uri=uri, target_column=TARGET)
    experiments.save_features(record.experiment_id, selected_features=FEATURES)
    return record


def _frozen(store, record) -> ExperimentConfig:
    layout = ExperimentLayout(base=record.artifact_prefix)
    return read_model(store, layout.experiment_config, ExperimentConfig)


def test_treatment_is_frozen_into_the_config():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(
            models=["logistic_regression"],
            outlier_detector=OutlierDetector.IQR,
            outlier_handling=OutlierHandling.DROP,
        ),
    )
    config = _frozen(store, record)
    assert config.treatment is not None
    assert config.treatment.outlier_detector is OutlierDetector.IQR
    assert config.treatment.outlier_handling is OutlierHandling.DROP


def test_no_treatment_leaves_the_config_untreated():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id, TrainingRequest(models=["logistic_regression"])
    )
    assert _frozen(store, record).treatment is None


def test_cap_with_isolation_forest_is_refused_before_submission():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store)
    record = _ready(experiments, store)
    with pytest.raises(ValidationError) as excinfo:
        experiments.start_training(
            record.experiment_id,
            TrainingRequest(
                models=["logistic_regression"],
                outlier_detector=OutlierDetector.ISOLATION_FOREST,
                outlier_handling=OutlierHandling.CAP,
            ),
        )
    assert "cap" in str(excinfo.value).lower() or "multivariate" in str(excinfo.value).lower()
    assert not orchestrator.failures


def test_default_handling_is_flag_when_a_detector_is_chosen():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(models=["logistic_regression"], outlier_detector=OutlierDetector.ZSCORE),
    )
    config = _frozen(store, record)
    assert config.treatment.outlier_handling is OutlierHandling.FLAG
