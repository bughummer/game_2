"""Ensembles end-to-end (ensembles-and-learners design, Testing Strategy).

Offline, with the in-memory store and the inline orchestrator. A voting/stacking ensemble is an
ordinary ModelSpec, so these tests assert the service freeze/validation/guards and that each
ensemble becomes a distinct, loadable, ranked model version through the real stages.
"""

import pytest

from backend.config import Settings
from backend.errors import ValidationError
from backend.repositories import ExperimentRepository
from backend.schemas import EnsembleBaseLearner, EnsembleRequest, TrainingRequest
from backend.services import ExperimentService
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.ensemble import EnsembleType, VotingRule
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

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
    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    return experiments, orchestrator


def _ready(experiments, store, name="churn.csv"):
    uri = write_dataset(store, name, golden_frame())
    record = experiments.create(name="e", dataset_uri=uri, target_column=TARGET)
    experiments.save_features(record.experiment_id, selected_features=FEATURES)
    return record


def _voting(*names, rule=VotingRule.SOFT) -> EnsembleRequest:
    return EnsembleRequest(
        type=EnsembleType.VOTING,
        base_learners=[EnsembleBaseLearner(name=n) for n in names],
        voting_rule=rule,
    )


def _config(store, record) -> ExperimentConfig:
    layout = ExperimentLayout(base=record.artifact_prefix)
    return read_model(store, layout.experiment_config, ExperimentConfig)


# --- validation + freeze ----------------------------------------------------------------------


def test_voting_ensemble_freezes_a_spec_and_adds_one_model():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(
            models=["logistic_regression"],
            ensembles=[_voting("logistic_regression", "random_forest")],
        ),
    )
    assert not orchestrator.failures, orchestrator.failures
    config = _config(store, record)
    assert len(config.ensembles) == 1
    ensemble_id = config.ensembles[0].ensemble_id
    assert any(s.name == ensemble_id for s in config.enabled_models_specs())


def test_fewer_than_two_base_learners_is_refused():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    with pytest.raises(ValidationError):
        experiments.start_training(
            record.experiment_id,
            TrainingRequest(ensembles=[_voting("logistic_regression")]),
        )


def test_an_unknown_base_learner_is_refused():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    with pytest.raises(ValidationError) as excinfo:
        experiments.start_training(
            record.experiment_id,
            TrainingRequest(ensembles=[_voting("not_a_model", "random_forest")]),
        )
    assert "not_a_model" in str(excinfo.value)


def test_no_ensemble_leaves_the_config_empty():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id, TrainingRequest(models=["logistic_regression"])
    )
    assert _config(store, record).ensembles == []


# --- voting end to end ------------------------------------------------------------------------


def test_voting_ensemble_is_a_distinct_loadable_ranked_model_version():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(
            models=["logistic_regression", "random_forest"],
            ensembles=[_voting("logistic_regression", "random_forest")],
        ),
    )
    assert not orchestrator.failures, orchestrator.failures
    layout = ExperimentLayout(base=record.artifact_prefix)
    config = _config(store, record)
    ensemble_id = config.ensembles[0].ensemble_id
    assert store.exists(layout.model_artifact(ensemble_id))
    assert store.exists(layout.model_metadata(ensemble_id))
    comparison = experiments.comparison(record.experiment_id)
    ranked = {e.model_name for e in comparison.models if e.rank is not None}
    assert ensemble_id in ranked


# --- stacking is gated on CV ------------------------------------------------------------------


def _stacking(*names) -> EnsembleRequest:
    return EnsembleRequest(
        type=EnsembleType.STACKING,
        base_learners=[EnsembleBaseLearner(name=n) for n in names],
    )


def test_stacking_is_refused_outside_cv_mode():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    with pytest.raises(ValidationError) as excinfo:
        experiments.start_training(
            record.experiment_id,
            TrainingRequest(ensembles=[_stacking("logistic_regression", "random_forest")]),
        )
    assert "cv" in str(excinfo.value).lower()


def test_stacking_under_cv_is_a_loadable_model_version():
    store = InMemoryObjectStore()
    experiments, orchestrator = _service(store, cv_max_fits=200)
    record = _ready(experiments, store)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(
            models=["logistic_regression"],
            evaluation_mode="cv",
            cv_folds=3,
            ensembles=[_stacking("logistic_regression", "random_forest")],
        ),
    )
    assert not orchestrator.failures, orchestrator.failures
    layout = ExperimentLayout(base=record.artifact_prefix)
    config = _config(store, record)
    assert config.ensembles[0].type is EnsembleType.STACKING
    assert config.ensembles[0].meta_learner is not None  # defaulted
    ensemble_id = config.ensembles[0].ensemble_id
    assert store.exists(layout.model_metadata(ensemble_id))


# --- open catalogue: installed-learner gating -------------------------------------------------


def test_selecting_lightgbm_without_the_package_is_refused():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    with pytest.raises(ValidationError) as excinfo:
        experiments.start_training(record.experiment_id, TrainingRequest(models=["lightgbm"]))
    assert "lightgbm" in str(excinfo.value).lower()


def test_selecting_lightgbm_with_the_package_in_requirements_is_accepted():
    store = InMemoryObjectStore()
    experiments, _ = _service(store)
    record = _ready(experiments, store)
    # The package is approved, so submission is accepted even though lightgbm is not importable in
    # the control plane (it is installed on the stock image at job start). The inline run may then
    # record a per-model failure, but the request itself is not refused with "not installed".
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(models=["lightgbm"], requirements=["lightgbm>=4.0"]),
    )
    config = _config(store, record)
    assert any(s.name == "lightgbm" for s in config.enabled_models_specs())
    assert config.requirements is not None and "lightgbm>=4.0" in config.requirements.packages
