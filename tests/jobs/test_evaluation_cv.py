"""Evaluation stage with cross-validation and a Test_Fold (CV design C6).

Covers the CV_Summary and CV ranking attached to the comparison, the Test_Evaluation written once
per successful Run with the Refit_Model (CV) or trained model (single), the single-split-with-a-
Test_Fold path, the hook being skipped for single split without a Test_Fold, instability/outlier
warnings surfaced in the comparison, and the evaluation job being the only writer of
comparison.json, test_evaluation.json and experiment_summary.json.
"""

import statistics
import sys
from datetime import UTC, datetime

import pytest

from jobs.evaluation import cv as evaluation_cv
from jobs.evaluation.main import run_evaluation
from jobs.preprocessing.main import run_preparation
from jobs.training.main import run_training
from ml_engine.contracts.comparison import ComparisonReport, ModelComparisonEntry
from ml_engine.contracts.config import SplitConfig
from ml_engine.contracts.cv import (
    CrossValidationConfig,
    CvScheme,
    EvaluationMode,
    FoldMetrics,
    FoldResult,
    TestEvaluationReport,
)
from ml_engine.contracts.metrics import MetricSet
from ml_engine.io import read_model, read_parquet, write_model
from tests.support.datasets import write_dataset
from tests.support.experiments import make_config
from tests.support.golden import collect_masked, read_golden, run_single_split_experiment
from tests.support.in_memory_store import InMemoryObjectStore

EXPERIMENT_ID = "exp-test"
MODELS = ["logistic_regression", "random_forest"]


def _cv_split(folds: int = 3, test_fraction: float | None = 0.2, **extra) -> SplitConfig:
    scheme = extra.pop("scheme", CvScheme.STRATIFIED_KFOLD)
    return SplitConfig(
        evaluation_mode=EvaluationMode.CV,
        cv=CrossValidationConfig(folds=folds, scheme=scheme),
        test_fraction=test_fraction,
        **extra,
    )


def _write_config(store, layout, dataset_uri, split, models=MODELS, **update):
    config = make_config(layout, EXPERIMENT_ID, dataset_uri, models)
    config = config.model_copy(update={"split": split, **update})
    write_model(store, layout.experiment_config, config)
    return config


def _run(store, layout, models=MODELS):
    run_preparation(store, layout, experiment_id=EXPERIMENT_ID)
    for model_name in models:
        run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=model_name)
    return run_evaluation(store, layout, experiment_id=EXPERIMENT_ID)


# --- CV: CV_Summary, ranking, Test_Evaluation ----------------------------------------------


def test_cv_comparison_carries_cv_summaries_and_ranks_by_cv_mean(store, layout, dataset_csv):
    _write_config(store, layout, dataset_csv, _cv_split(folds=3, test_fraction=0.2))

    comparison = _run(store, layout)

    assert comparison.cross_validation is not None
    assert comparison.cross_validation.scheme is CvScheme.STRATIFIED_KFOLD
    assert comparison.cross_validation.folds == 3
    assert comparison.cross_validation.ranking == "cv_mean_then_std_then_run_id"

    ranked = [e for e in comparison.models if e.rank is not None]
    assert len(ranked) == len(MODELS)
    for entry in ranked:
        assert entry.cv is not None
        assert entry.cv.folds == 3
        # Each per-metric aggregate is ordered.
        for aggregate in entry.cv.metrics.values():
            assert aggregate.min <= aggregate.mean <= aggregate.max

    # The CV ranking drives best_model/best_score: rank 1 is the highest CV mean roc_auc.
    primary = comparison.primary_metric
    means = {e.model_name: e.cv.metrics[primary].mean for e in ranked}
    best = max(means, key=means.get)  # roc_auc is maximised
    assert comparison.best_model == best
    assert comparison.best_score == pytest.approx(means[best])


def test_cv_test_evaluation_uses_the_refit_model_once_per_successful_run(
    store, layout, dataset_csv
):
    _write_config(store, layout, dataset_csv, _cv_split(folds=3, test_fraction=0.2))

    comparison = _run(store, layout)

    assert store.exists(layout.test_evaluation)
    report = read_model(store, layout.test_evaluation, TestEvaluationReport)
    assert report.model_source == "refit_model"
    assert report.evaluation_mode is EvaluationMode.CV
    assert report.test_row_count == len(read_parquet(store, layout.test_dataset))
    assert {run.model_name for run in report.runs} == set(MODELS)  # once per Run
    assert all(run.status == "completed" for run in report.runs)

    # Test metrics are copied onto the entries, kept separate from the CV ranking metrics.
    for entry in comparison.models:
        if entry.cv is not None:
            assert entry.test_metrics is not None
            assert "roc_auc" in entry.test_metrics
    assert comparison.test_evaluation is not None


def test_cv_without_a_test_fold_has_cv_ranking_but_no_test_evaluation(store, layout, dataset_csv):
    _write_config(store, layout, dataset_csv, _cv_split(folds=4, test_fraction=None))

    comparison = _run(store, layout)

    assert comparison.cross_validation is not None and comparison.cross_validation.folds == 4
    assert comparison.cross_validation.test_row_count is None
    assert comparison.test_evaluation is None
    assert not store.exists(layout.test_evaluation)
    for entry in comparison.models:
        if entry.rank is not None:
            assert entry.test_metrics is None


# --- single split with a Test_Fold ----------------------------------------------------------


def test_single_split_with_a_test_fold_gets_a_test_evaluation_but_no_cv(store, layout, dataset_csv):
    _write_config(store, layout, dataset_csv, SplitConfig(test_fraction=0.25))

    comparison = _run(store, layout)

    assert comparison.cross_validation is None  # no CV in single-split mode
    assert store.exists(layout.test_evaluation)
    report = read_model(store, layout.test_evaluation, TestEvaluationReport)
    assert report.model_source == "trained_model"
    assert report.evaluation_mode is EvaluationMode.SINGLE_SPLIT
    assert {run.model_name for run in report.runs} == set(MODELS)
    for entry in comparison.models:
        if entry.status.value == "COMPLETED":
            assert entry.test_metrics is not None
    # Ranking is still the inherited single-split ranking (by validation primary score).
    ranked = [e for e in comparison.models if e.rank is not None]
    assert comparison.best_model == ranked[0].model_name


# --- single split WITHOUT a Test_Fold: the hook is skipped ----------------------------------


def test_single_split_without_a_test_fold_skips_the_hook(store, layout, dataset_csv, monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("single split without a Test_Fold must not call extend_comparison")

    monkeypatch.setattr(evaluation_cv, "extend_comparison", refuse)
    _write_config(store, layout, dataset_csv, SplitConfig())

    comparison = _run(store, layout)

    assert comparison.cross_validation is None and comparison.test_evaluation is None
    assert not store.exists(layout.test_evaluation)


def test_single_split_artifacts_stay_byte_identical_to_the_golden_fixtures():
    store = InMemoryObjectStore()
    layout = run_single_split_experiment(store)
    assert collect_masked(store, layout) == read_golden()
    assert not store.exists(layout.test_evaluation)


# --- the evaluation job is the only writer --------------------------------------------------


class WriteRecordingStore(InMemoryObjectStore):
    """Records which job module wrote each object (same pattern as test_preparation_cv)."""

    def __init__(self) -> None:
        super().__init__()
        self.writers: dict[str, list[str]] = {}

    def write_bytes(self, uri: str, data: bytes) -> None:
        frame = sys._getframe(1)
        writer = "<unknown>"
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if module.startswith("jobs.") and not module.startswith("jobs._common"):
                writer = module
                break
            frame = frame.f_back
        self.writers.setdefault(uri, []).append(writer)
        super().write_bytes(uri, data)


def test_only_the_evaluation_job_writes_comparison_summary_and_test_evaluation(
    layout, classification_frame
):
    store = WriteRecordingStore()
    uri = write_dataset(store, "churn.csv", classification_frame)
    _write_config(store, layout, uri, _cv_split(folds=3, test_fraction=0.2))

    _run(store, layout)

    # Each is written exactly once, only by the evaluation job (its main or its cv helper).
    for target in (layout.comparison, layout.summary, layout.test_evaluation):
        assert store.exists(target)
        assert store.writers[target] in (
            ["jobs.evaluation.main"],
            ["jobs.evaluation.cv"],
        ), (target, store.writers[target])


# --- instability / outlier warnings surfaced in the comparison ------------------------------


def _fold_metrics(model_name: str, primary: str, values: list[float]) -> FoldMetrics:
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    folds = [
        FoldResult(
            model_name=model_name,
            fold_index=index,
            assignment_digest="a",
            params_digest="p",
            train_row_count=10,
            validation_row_count=5,
            metrics=MetricSet(values={primary: value}),
            primary_score=value,
            started_at=moment,
            completed_at=moment,
            duration_seconds=0.0,
        )
        for index, value in enumerate(values)
    ]
    return FoldMetrics(
        experiment_id=EXPERIMENT_ID, model_name=model_name, scheme=CvScheme.KFOLD, folds=folds
    )


def test_instability_and_outlier_warnings_are_surfaced_and_ranking_prefers_the_steady_run(
    store, layout, dataset_csv
):
    config = _write_config(store, layout, dataset_csv, _cv_split(folds=5, test_fraction=None))

    # Two Runs with hand-built fold metrics: one steady, one with a single wild fold (the design's
    # worked example) that is both unstable and has an outlier fold.
    steady = _fold_metrics("logistic_regression", "roc_auc", [0.90, 0.905, 0.895, 0.90, 0.90])
    wild = _fold_metrics("random_forest", "roc_auc", [0.80, 0.81, 0.79, 0.98, 0.80])
    write_model(store, layout.model_fold_metrics("logistic_regression"), steady)
    write_model(store, layout.model_fold_metrics("random_forest"), wild)

    successes = [
        _metadata("logistic_regression", 0.900),
        _metadata("random_forest", statistics.fmean([0.80, 0.81, 0.79, 0.98, 0.80])),
    ]
    comparison = _build_inherited(config)
    extended = evaluation_cv.extend_comparison(store, layout, config, comparison, successes)

    rules = {w.rule for w in extended.warnings}
    assert "cv_unstable_model" in rules
    assert "cv_outlier_fold" in rules

    wild_entry = next(e for e in extended.models if e.model_name == "random_forest")
    assert wild_entry.cv is not None
    assert wild_entry.cv.instability_flag is True
    assert wild_entry.cv.outlier_folds == [3]  # only fold index 3 (the 0.98 fold)

    # The steady Run ranks first on its higher, stable CV mean.
    assert extended.best_model == "logistic_regression"


def _metadata(model_name: str, primary_score: float):
    from ml_engine.contracts.common import ModelRunStatus, ProblemType
    from ml_engine.contracts.model import ModelMetadata

    return ModelMetadata(
        experiment_id=EXPERIMENT_ID,
        model_name=model_name,
        display_name=model_name,
        status=ModelRunStatus.COMPLETED,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        library="x",
        library_version="1",
        primary_metric="roc_auc",
        primary_score=primary_score,
        metrics=MetricSet(values={"roc_auc": primary_score}),
    )


def _build_inherited(config) -> ComparisonReport:
    from ml_engine.contracts.common import MetricDirection, ModelRunStatus, ProblemType

    return ComparisonReport(
        experiment_id=EXPERIMENT_ID,
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        direction=MetricDirection.MAXIMIZE,
        models=[
            ModelComparisonEntry(
                model_name=name,
                display_name=name,
                status=ModelRunStatus.COMPLETED,
                primary_score=0.9,
                metrics={"roc_auc": 0.9},
            )
            for name in MODELS
        ],
        succeeded_count=2,
    )
