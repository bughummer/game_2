"""End-to-end cross-validation experiment, offline (CV task 12.1).

Runs a whole CV_Mode experiment through the real stages with the ``InlineOrchestrator`` over the
in-memory store: K = 3, two models, a 20% Test_Fold, through preparation, training, evaluation,
holdout evaluation and scoring. Asserts the CV artifacts exist, the comparison ranks by CV mean,
the Test_Fold metrics are kept separate from the CV ranking metrics, and that a second run of the
same frozen configuration reproduces each Run's mean primary metric exactly.

Requirements: 13.1, 13.3, 13.4, 3.7, 8.5, 10.4.
"""

import pytest

from backend.config import Settings
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import AnalysisService, ExperimentService
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.cv import EvaluationMode, TestEvaluationReport
from ml_engine.io import ExperimentLayout, read_model, read_parquet
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

MODELS = ["logistic_regression", "random_forest"]
FEATURES = ["tenure_months", "monthly_charges", "contract", "is_business", "signup_date"]
TARGET = "churned"
FOLDS = 3
TEST_FRACTION = 0.2


def _settings() -> Settings:
    # Budget enforcement is off (the production default), so no price table is needed.
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf-sagemaker-job",
        stage_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/img@sha256:" + "a" * 64,
    )


def _services(settings, store):
    repository = ExperimentRepository(settings=settings, store=store)
    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    analysis = AnalysisService(
        settings=settings,
        repository=repository,
        store=store,
        orchestrator=orchestrator,
        experiments=experiments,
    )
    return experiments, analysis, orchestrator


def _run_cv_experiment(store, settings, *, name="cv.csv"):
    """Create, profile and train one CV_Mode experiment with a Test_Fold through the real stages."""
    experiments, analysis, orchestrator = _services(settings, store)
    uri = write_dataset(store, name, golden_frame())
    record = experiments.create(name="e2e-cv", dataset_uri=uri, target_column=TARGET)
    experiments.save_features(record.experiment_id, selected_features=FEATURES)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(
            models=MODELS,
            evaluation_mode="cv",
            cv_folds=FOLDS,
            test_fraction=TEST_FRACTION,
        ),
    )
    if orchestrator.failures:
        raise RuntimeError(f"CV experiment failed: {orchestrator.failures}")
    return record, experiments, analysis, orchestrator


@pytest.fixture(scope="module")
def cv_run():
    store = InMemoryObjectStore()
    settings = _settings()
    record, experiments, analysis, orchestrator = _run_cv_experiment(store, settings)
    layout = ExperimentLayout(base=record.artifact_prefix)
    return {
        "store": store,
        "settings": settings,
        "record": record,
        "experiments": experiments,
        "analysis": analysis,
        "orchestrator": orchestrator,
        "layout": layout,
    }


def test_cv_experiment_writes_the_cv_artifacts(cv_run):
    store, layout = cv_run["store"], cv_run["layout"]

    assert store.exists(layout.fold_assignment)
    assert store.exists(layout.development_dataset)
    assert store.exists(layout.test_dataset)
    for model_name in MODELS:
        assert store.exists(layout.model_fold_metrics(model_name)), model_name
        assert store.exists(layout.model_oof(model_name)), model_name
        assert store.exists(layout.model_artifact(model_name)), model_name


def test_cv_config_is_frozen_as_cv_mode_with_the_requested_folds(cv_run):
    config = read_model(cv_run["store"], cv_run["layout"].experiment_config, ExperimentConfig)
    assert config.split.evaluation_mode is EvaluationMode.CV
    assert config.split.cv is not None
    assert config.split.cv.folds == FOLDS
    assert config.split.test_fraction == TEST_FRACTION


def test_cv_comparison_ranks_by_cv_mean(cv_run):
    comparison = cv_run["experiments"].comparison(cv_run["record"].experiment_id)

    assert comparison.cross_validation is not None
    assert comparison.cross_validation.folds == FOLDS
    assert comparison.cross_validation.ranking == "cv_mean_then_std_then_run_id"

    ranked = [e for e in comparison.models if e.rank is not None]
    assert {e.model_name for e in ranked} == set(MODELS)
    for entry in ranked:
        assert entry.cv is not None
        assert entry.cv.folds == FOLDS
        for aggregate in entry.cv.metrics.values():
            assert aggregate.min <= aggregate.mean <= aggregate.max

    primary = comparison.primary_metric
    means = {e.model_name: e.cv.metrics[primary].mean for e in ranked}
    best = max(means, key=means.get)  # roc_auc is maximised
    assert comparison.best_model == best
    assert comparison.best_score == pytest.approx(means[best])


def test_test_fold_metrics_are_kept_separate_from_the_cv_ranking(cv_run):
    store, layout = cv_run["store"], cv_run["layout"]
    comparison = cv_run["experiments"].comparison(cv_run["record"].experiment_id)

    assert store.exists(layout.test_evaluation)
    report = read_model(store, layout.test_evaluation, TestEvaluationReport)
    assert report.model_source == "refit_model"
    assert report.evaluation_mode is EvaluationMode.CV
    assert report.test_row_count == len(read_parquet(store, layout.test_dataset))
    assert {run.model_name for run in report.runs} == set(MODELS)
    assert all(run.status == "completed" for run in report.runs)

    for entry in comparison.models:
        if entry.cv is not None:
            assert entry.test_metrics is not None
            assert "roc_auc" in entry.test_metrics


def test_holdout_and_scoring_run_on_a_cv_experiment(cv_run):
    """The downstream consumers that need a dataset work on a CV Run (Test_Fold untouched)."""
    experiments = cv_run["experiments"]
    orchestrator = cv_run["orchestrator"]
    store = cv_run["store"]
    exp_id = cv_run["record"].experiment_id

    # Holdout evaluation on a labelled dataset.
    holdout_uri = write_dataset(store, "holdout.csv", golden_frame())
    experiments.start_holdout_evaluation(exp_id, dataset_uri=holdout_uri)

    # Scoring on an unlabelled dataset.
    scoring_frame = golden_frame().drop(columns=[TARGET])
    scoring_uri = write_dataset(store, "score.csv", scoring_frame)
    cv_run["analysis"].start_scoring(exp_id, dataset_uri=scoring_uri)

    assert not orchestrator.failures, orchestrator.failures


def test_the_same_frozen_config_reproduces_each_run_cv_mean():
    """Two runs of the same experiment agree on each Run's mean primary metric (3.7, 13.5 offline)."""
    settings = _settings()

    first_store = InMemoryObjectStore()
    first_record, first_experiments, *_ = _run_cv_experiment(first_store, settings, name="a.csv")
    first = first_experiments.comparison(first_record.experiment_id)

    second_store = InMemoryObjectStore()
    second_record, second_experiments, *_ = _run_cv_experiment(second_store, settings, name="b.csv")
    second = second_experiments.comparison(second_record.experiment_id)

    primary = first.primary_metric
    first_means = {e.model_name: e.cv.metrics[primary].mean for e in first.models if e.cv}
    second_means = {e.model_name: e.cv.metrics[primary].mean for e in second.models if e.cv}

    assert first_means.keys() == second_means.keys() == set(MODELS)
    for model_name in MODELS:
        assert first_means[model_name] == pytest.approx(second_means[model_name])
