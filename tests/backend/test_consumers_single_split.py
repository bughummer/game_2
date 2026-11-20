"""A pre-feature single-split experiment still runs cleanly through every consumer (CV task 12.2).

The byte-identical comparison against the golden fixtures lives in ``test_golden_single_split.py``.
This test covers the other half of 12.2: a Single_Split_Mode experiment driven through the
downstream consumers (thresholds, permutation importance, ablation, holdout evaluation, scoring,
leakage, archiving) produces unchanged, valid outputs — the cross-validation feature left the
single-split path intact.

Requirements: 7.7, 10.7, 12.3, 13.6.
"""

import pytest

from backend.config import Settings
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import AnalysisService, ExperimentService, LeakageService
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.cv import EvaluationMode
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

MODELS = ["logistic_regression", "random_forest"]
FEATURES = ["tenure_months", "monthly_charges", "contract", "is_business", "signup_date"]
TARGET = "churned"


def _settings() -> Settings:
    # Budget enforcement is off (the production default), so no price table is needed.
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf-sagemaker-job",
        stage_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/img@sha256:" + "a" * 64,
    )


@pytest.fixture(scope="module")
def single_split_run():
    store = InMemoryObjectStore()
    settings = _settings()
    repository = ExperimentRepository(settings=settings, store=store)
    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    leakage = LeakageService(settings=settings, repository=repository, store=store)
    analysis = AnalysisService(
        settings=settings,
        repository=repository,
        store=store,
        orchestrator=orchestrator,
        experiments=experiments,
    )
    uri = write_dataset(store, "churn.csv", golden_frame())
    record = experiments.create(name="single", dataset_uri=uri, target_column=TARGET)
    experiments.save_features(record.experiment_id, selected_features=FEATURES)
    # No evaluation_mode -> the configured default (single_split); no Test_Fold.
    experiments.start_training(record.experiment_id, TrainingRequest(models=MODELS))
    if orchestrator.failures:
        raise RuntimeError(f"single-split experiment failed: {orchestrator.failures}")
    return {
        "store": store,
        "record": record,
        "experiments": experiments,
        "leakage": leakage,
        "analysis": analysis,
        "orchestrator": orchestrator,
        "layout": ExperimentLayout(base=record.artifact_prefix),
    }


def test_frozen_config_is_single_split_with_no_cv(single_split_run):
    config = read_model(
        single_split_run["store"], single_split_run["layout"].experiment_config, ExperimentConfig
    )
    assert config.split.evaluation_mode is EvaluationMode.SINGLE_SPLIT
    assert config.split.cv is None
    assert config.split.test_fraction is None
    assert not single_split_run["store"].exists(single_split_run["layout"].test_evaluation)


def test_comparison_ranks_single_split_without_cv(single_split_run):
    comparison = single_split_run["experiments"].comparison(
        single_split_run["record"].experiment_id
    )
    assert comparison.cross_validation is None
    assert comparison.test_evaluation is None
    ranked = [e for e in comparison.models if e.rank is not None]
    assert {e.model_name for e in ranked} == set(MODELS)
    assert comparison.best_model == ranked[0].model_name


def test_thresholds_available_for_the_binary_models(single_split_run):
    thresholds = single_split_run["experiments"].thresholds(
        single_split_run["record"].experiment_id
    )
    assert thresholds, "a binary single-split experiment produces threshold analyses"


def test_permutation_importance_runs(single_split_run):
    result = single_split_run["analysis"].permutation_importance(
        single_split_run["record"].experiment_id
    )
    assert result is not None


def test_ablation_runs_on_single_split(single_split_run):
    single_split_run["analysis"].start_ablation(
        single_split_run["record"].experiment_id,
        model_name="logistic_regression",
        features=["tenure_months"],
    )
    assert not single_split_run["orchestrator"].failures, single_split_run["orchestrator"].failures


def test_holdout_evaluation_runs_on_single_split(single_split_run):
    holdout_uri = write_dataset(single_split_run["store"], "holdout.csv", golden_frame())
    single_split_run["experiments"].start_holdout_evaluation(
        single_split_run["record"].experiment_id, dataset_uri=holdout_uri
    )
    assert not single_split_run["orchestrator"].failures, single_split_run["orchestrator"].failures


def test_scoring_runs_on_single_split(single_split_run):
    scoring_frame = golden_frame().drop(columns=[TARGET])
    scoring_uri = write_dataset(single_split_run["store"], "score.csv", scoring_frame)
    single_split_run["analysis"].start_scoring(
        single_split_run["record"].experiment_id, dataset_uri=scoring_uri
    )
    assert not single_split_run["orchestrator"].failures, single_split_run["orchestrator"].failures


def test_leakage_report_available(single_split_run):
    report = single_split_run["leakage"].build_post_training_phase(
        single_split_run["record"].experiment_id
    )
    assert report is not None


def test_retrain_archives_the_previous_single_split_run(single_split_run):
    experiments = single_split_run["experiments"]
    exp_id = single_split_run["record"].experiment_id
    experiments.start_training(exp_id, TrainingRequest(models=MODELS))
    assert not single_split_run["orchestrator"].failures, single_split_run["orchestrator"].failures
    runs = experiments.list_runs(exp_id)
    assert len(runs) == 1
    assert runs[0].evaluation_mode is EvaluationMode.SINGLE_SPLIT
