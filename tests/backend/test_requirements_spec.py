"""Per-experiment package requirements (stock-image migration Stage 2, task 10).

Offline: the contract, its well-formedness validation, and the service-layer resolution/allowlist.
No install happens here — that is Stage 2 job work confirmed at a first real-AWS run.
"""

import json

import pytest

from backend.config import Settings
from backend.errors import ValidationError
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import ExperimentService
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.requirements import (
    RequirementsError,
    RequirementsSpec,
    requirement_name,
    validate_requirement_line,
)
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

FEATURES = ["tenure_months", "monthly_charges", "contract", "is_business", "signup_date"]
TARGET = "churned"


# --- the contract and its validation --------------------------------------------------------


@pytest.mark.parametrize(
    "line", ["catboost", "catboost>=1.2", "lightgbm==4.3.0", "scikit-learn>=1.3,<2", "pkg[extra]"]
)
def test_well_formed_requirements_are_accepted(line):
    assert validate_requirement_line(line)


@pytest.mark.parametrize("line", ["", "   ", "# comment", "not a package!!", ">=1.2", "a==b==c"])
def test_malformed_requirements_are_rejected(line):
    with pytest.raises(RequirementsError):
        validate_requirement_line(line)


def test_requirement_name_normalises():
    assert requirement_name("scikit_learn>=1.3") == "scikit-learn"
    assert requirement_name("CatBoost") == "catboost"


def test_spec_collapses_duplicates_and_keeps_order():
    spec = RequirementsSpec(packages=["catboost>=1.2", "lightgbm", "catboost>=1.2"])
    assert spec.packages == ["catboost>=1.2", "lightgbm"]
    assert not spec.is_empty


def test_spec_round_trips():
    spec = RequirementsSpec(packages=["catboost>=1.2"])
    assert RequirementsSpec.model_validate_json(spec.model_dump_json()) == spec


def test_empty_requirements_are_omitted_from_experiment_config_json():
    """A config with no requirements stays byte-identical to a pre-feature one: the field is
    excluded at its None default."""
    from tests.support.experiments import make_config

    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, "exp-test")
    config = make_config(layout, "exp-test", "s3://b/x.csv", ["logistic_regression"])
    assert config.requirements is None
    assert "requirements" not in json.loads(config.model_dump_json())


# --- service-layer resolution ---------------------------------------------------------------


def _service(store, **settings_overrides):
    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf-sagemaker-job",
        **settings_overrides,
    )
    repository = ExperimentRepository(settings=settings, store=store)
    orchestrator = InlineOrchestrator(store)
    return ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    ), orchestrator


def _ready(experiments, store, name="churn.csv"):
    uri = write_dataset(store, name, golden_frame())
    record = experiments.create(name="t", dataset_uri=uri, target_column=TARGET)
    experiments.save_features(record.experiment_id, selected_features=FEATURES)
    return record


def test_approved_requirements_are_frozen_into_the_config():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(models=["logistic_regression"], requirements=["catboost>=1.2"]),
    )
    assert not orchestrator.failures, orchestrator.failures
    layout = ExperimentLayout(base=record.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert config.requirements is not None
    assert config.requirements.packages == ["catboost>=1.2"]


def test_no_requirements_leaves_the_config_field_none():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id, TrainingRequest(models=["logistic_regression"])
    )
    assert not orchestrator.failures, orchestrator.failures
    layout = ExperimentLayout(base=record.artifact_prefix)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert config.requirements is None


def test_a_malformed_requirement_is_refused_and_nothing_is_submitted():
    store = InMemoryObjectStore()
    experiments, _orchestrator = _service(store)
    record = _ready(experiments, store)
    with pytest.raises(ValidationError):
        experiments.start_training(
            record.experiment_id,
            TrainingRequest(models=["logistic_regression"], requirements=["not a package!!"]),
        )
    layout = ExperimentLayout(base=record.artifact_prefix)
    assert not store.exists(layout.experiment_config)


def test_the_allowlist_blocks_an_outside_package_when_enforced():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(
        store, enforce_package_allowlist=True, package_install_allowlist=["catboost"]
    )
    record = _ready(experiments, store)
    with pytest.raises(ValidationError) as excinfo:
        experiments.start_training(
            record.experiment_id,
            TrainingRequest(models=["logistic_regression"], requirements=["lightgbm"]),
        )
    assert "allowlist" in str(excinfo.value)
    # An allowed package passes.
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(models=["logistic_regression"], requirements=["catboost>=1.2"]),
    )
    assert not orchestrator.failures, orchestrator.failures
