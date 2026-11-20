"""CV_Mode evaluation: CV_Summary per Run, CV ranking and the Test_Evaluation (CV design C6).

``run_evaluation`` (adapted) calls :func:`extend_comparison` once, between ``build_comparison``
and ``write_model(comparison)``, and only when the frozen config is in CV_Mode or carries a
``test_fraction``. Single_Split_Mode without a Test_Fold never reaches this module, so the
inherited comparison stays byte-identical (Req 7.7).

Two things happen here, in order:

1. CV ranking (CV_Mode only): read each successful Run's ``fold_metrics.json``, build its
   CV_Summary, attach it to the comparison entry, surface instability / outlier / primary-missing
   warnings, re-rank by CV mean (then std, then run id) and set ``cross_validation``.
2. Test_Evaluation (both modes, when a Test_Fold exists): score every successful Run once on the
   Test_Fold with the Refit_Model (CV) or trained model (single) and the Development_Set
   preprocessor, write ``comparison/test_evaluation.json`` and copy per-Run ``test_metrics``.

Ranking is always computed before the Test_Evaluation and never reads test metrics (Req 3.8).
The evaluation job remains the only writer of ``comparison.json``, ``test_evaluation.json`` and
``experiment_summary.json`` (Req 7.8).
"""

import logging
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.comparison import ComparisonReport, ModelComparisonEntry
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.cv import (
    CvComparisonInfo,
    CvRunSummary,
    EvaluationMode,
    FoldMetrics,
    TestEvaluationReport,
    TestRunResult,
)
from ml_engine.contracts.model import ModelMetadata
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.evaluation import evaluate, metric_direction, resolve_primary_metric
from ml_engine.features import apply_proposals, column_types_from_frame, validate_proposals
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    load_joblib,
    read_model_if_exists,
    read_parquet,
    write_model,
)
from ml_engine.models import resolve_plugin
from ml_engine.reporting.cv import (
    instability_warnings,
    primary_metric_missing_folds,
    primary_metric_missing_warning,
    rank_cv,
    summarize_folds,
)
from ml_engine.training.runner import _positive_class, build_training_context

LOGGER = logging.getLogger("ml_factory.jobs.evaluation.cv")


def extend_comparison(
    store: ObjectStore,
    layout: ExperimentLayout,
    config: ExperimentConfig,
    comparison: ComparisonReport,
    successes: list[ModelMetadata],
) -> ComparisonReport:
    """Add the CV_Summary, CV ranking and Test_Evaluation to the inherited comparison."""
    cv_mode = config.split.evaluation_mode is EvaluationMode.CV
    primary_metric = resolve_primary_metric(config.problem_type, config.primary_metric)
    direction = metric_direction(config.problem_type, primary_metric)

    if cv_mode:
        comparison = _apply_cv_ranking(
            store, layout, config, comparison, successes, primary_metric, direction
        )

    if store.exists(layout.test_dataset):
        comparison = _apply_test_evaluation(
            store, layout, config, comparison, successes, primary_metric, cv_mode
        )

    return comparison


# --------------------------------------------------------------------------------------------
# CV ranking and CV_Summary
# --------------------------------------------------------------------------------------------


def _apply_cv_ranking(
    store: ObjectStore,
    layout: ExperimentLayout,
    config: ExperimentConfig,
    comparison: ComparisonReport,
    successes: list[ModelMetadata],
    primary_metric: str,
    direction,
) -> ComparisonReport:
    entries = {entry.model_name: entry for entry in comparison.models}
    extra_warnings: list[AnalysisWarning] = []

    for metadata in successes:
        entry = entries.get(metadata.model_name)
        if entry is None:
            continue
        fold_metrics = read_model_if_exists(
            store, layout.model_fold_metrics(metadata.model_name), FoldMetrics
        )
        if fold_metrics is None:
            continue  # a Run trained but produced no fold metrics; left unranked
        summary = summarize_folds(fold_metrics, primary_metric)
        extra_warnings.extend(
            instability_warnings(summary, _cv_config(config), metadata.model_name, primary_metric)
        )
        missing = primary_metric_missing_folds(fold_metrics)
        if missing:
            extra_warnings.append(
                primary_metric_missing_warning(metadata.model_name, primary_metric, missing)
            )
            # A Run missing the primary metric in any fold is unrankable (design C6): drop its
            # primary aggregate so rank_cv skips it.
            summary = _without_primary(summary, primary_metric)
        entry.cv = summary

    best_model, best_score = rank_cv(comparison.models, direction, primary_metric)
    _resort(comparison.models)

    scheme = config.split.cv.scheme if config.split.cv is not None else None
    folds = config.split.cv.folds if config.split.cv is not None else 0
    test_row_count = _test_row_count(store, layout)
    assert scheme is not None  # CV_Mode always has a scheme in the frozen config
    return comparison.model_copy(
        update={
            "best_model": best_model,
            "best_score": best_score,
            "warnings": comparison.warnings + extra_warnings,
            "cross_validation": CvComparisonInfo(
                scheme=scheme, folds=folds, test_row_count=test_row_count
            ),
        }
    )


def _cv_config(config: ExperimentConfig):
    assert config.split.cv is not None  # CV_Mode
    return config.split.cv


def _without_primary(summary: CvRunSummary, primary_metric: str) -> CvRunSummary:
    metrics = {
        name: aggregate for name, aggregate in summary.metrics.items() if name != primary_metric
    }
    return summary.model_copy(update={"metrics": metrics})


def _resort(entries: list[ModelComparisonEntry]) -> None:
    """Re-order the entries list the way the inherited comparison does: rank, then model name."""
    entries.sort(key=lambda entry: (entry.rank is None, entry.rank or 0, entry.model_name))


def _test_row_count(store: ObjectStore, layout: ExperimentLayout) -> int | None:
    if not store.exists(layout.test_dataset):
        return None
    return len(read_parquet(store, layout.test_dataset))


# --------------------------------------------------------------------------------------------
# Test_Evaluation
# --------------------------------------------------------------------------------------------


def _apply_test_evaluation(
    store: ObjectStore,
    layout: ExperimentLayout,
    config: ExperimentConfig,
    comparison: ComparisonReport,
    successes: list[ModelMetadata],
    primary_metric: str,
    cv_mode: bool,
) -> ComparisonReport:
    test_frame = read_parquet(store, layout.test_dataset)
    test_frame = _derive(test_frame, config)
    training_labels = _training_labels(store, layout, config, cv_mode)

    entries = {entry.model_name: entry for entry in comparison.models}
    runs: list[TestRunResult] = []
    for metadata in successes:
        result = _score_run(store, layout, config, metadata, test_frame, training_labels)
        runs.append(result)
        entry = entries.get(metadata.model_name)
        if entry is not None and result.status == "completed":
            entry.test_metrics = dict(result.metrics)

    report = TestEvaluationReport(
        experiment_id=config.experiment_id,
        generated_at=datetime.now(UTC),
        evaluation_mode=config.split.evaluation_mode,
        test_row_count=len(test_frame),
        primary_metric=primary_metric,
        model_source="refit_model" if cv_mode else "trained_model",
        runs=runs,
    )
    write_model(store, layout.test_evaluation, report)
    LOGGER.info("Scored %d Run(s) on %d Test_Fold rows", len(runs), len(test_frame))
    return comparison.model_copy(update={"test_evaluation": report})


def _derive(frame: pd.DataFrame, config: ExperimentConfig) -> pd.DataFrame:
    """Compute the run's derived features exactly as preparation did (row-wise, no fitting).

    Same approach as the holdout job: validate the frozen proposals against the Test_Fold's column
    types, then apply the accepted ones. No fitting happens, so this cannot leak the Test_Fold.
    """
    if not config.derived_features:
        return frame
    accepted, _rejected = validate_proposals(
        config.derived_features,
        column_types_from_frame(frame),
        target_column=config.target_column,
    )
    frame, _derived = apply_proposals(frame, accepted)
    return frame


def _training_labels(
    store: ObjectStore, layout: ExperimentLayout, config: ExperimentConfig, cv_mode: bool
) -> pd.Series:
    """The development/training labels, used only to resolve the positive class (same as training).

    CV: the Development_Set the Refit_Model was fit on. Single: the training fold.
    """
    source = layout.development_dataset if cv_mode else layout.train_dataset
    frame = read_parquet(store, source)
    labels = frame[config.target_column]
    if config.problem_type.is_classification:
        labels = labels.astype("string")
    return labels


def _score_run(
    store: ObjectStore,
    layout: ExperimentLayout,
    config: ExperimentConfig,
    metadata: ModelMetadata,
    test_frame: pd.DataFrame,
    training_labels: pd.Series,
) -> TestRunResult:
    """Score one Run on the Test_Fold. A scoring failure is recorded, never raised (Req 8.5)."""
    try:
        plugin = resolve_plugin(config, metadata.model_name)
        pipeline = load_joblib(store, layout.preprocessor(plugin.preprocessing_strategy))
        estimator = load_joblib(store, layout.model_artifact(metadata.model_name))
        x = pipeline.transform(test_frame)
        predictions = plugin.predict(estimator, x)
        probabilities = plugin.predict_proba(estimator, x)

        problem_type = config.problem_type
        y = test_frame[config.target_column]
        labels = plugin.class_labels(estimator) if problem_type.is_classification else None
        positive = None
        if problem_type.is_classification:
            y = y.astype("string")
        if problem_type is ProblemType.BINARY_CLASSIFICATION:
            context = build_training_context(
                problem_type=problem_type,
                y_train=training_labels,
                feature_count=x.shape[1],
                random_seed=config.split.random_seed,
                class_weighting=config.class_weighting,
                categorical_features=[],
            )
            positive = _positive_class(labels, context)

        metrics = evaluate(
            problem_type,
            y.reset_index(drop=True),
            np.asarray(predictions),
            y_proba=None if probabilities is None else np.asarray(probabilities),
            class_labels=labels,
            positive_class=positive,
        )
    except Exception as error:  # a Run's failure never fails the job (Req 8.5)
        LOGGER.exception("Test_Fold scoring of %s failed", metadata.model_name)
        return TestRunResult(
            model_name=metadata.model_name,
            status="failed",
            error_code="TEST_EVALUATION_FAILED",
            message=f"{type(error).__name__}: {error}"[:500],
        )

    return TestRunResult(
        model_name=metadata.model_name,
        status="completed",
        metrics=dict(metrics.values),
    )
