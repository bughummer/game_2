"""Unit tests for the pure CV reporting helpers (ML Factory 2.0, CV design C6, task 7.1).

These example tests pin the design's worked numbers: ``summarize_folds`` aggregation and the
instability / outlier rules of :func:`ml_engine.reporting.cv.instability_warnings`. The
property-based coverage lives in ``tests/ml_engine/test_cv_properties.py`` (Properties 10-12);
this file documents the concrete cases the design calls out so a regression is easy to read.
"""

import statistics
from datetime import UTC, datetime

import pytest

from ml_engine.contracts.common import MetricDirection, ModelRunStatus
from ml_engine.contracts.comparison import ModelComparisonEntry
from ml_engine.contracts.cv import (
    CrossValidationConfig,
    CvRunSummary,
    CvScheme,
    FoldMetrics,
    FoldResult,
    MetricAggregate,
)
from ml_engine.contracts.metrics import MetricSet
from ml_engine.reporting.cv import (
    instability_warnings,
    primary_metric_missing_folds,
    rank_cv,
    summarize_folds,
)

PRIMARY = "roc_auc"
_MOMENT = datetime(2026, 1, 1, tzinfo=UTC)


def _fold(fold_index: int, values: dict[str, float]) -> FoldResult:
    return FoldResult(
        model_name="m",
        fold_index=fold_index,
        assignment_digest="a",
        params_digest="p",
        train_row_count=10,
        validation_row_count=5,
        metrics=MetricSet(values=dict(values)),
        primary_score=values.get(PRIMARY),
        started_at=_MOMENT,
        completed_at=_MOMENT,
        duration_seconds=0.0,
    )


def _fold_metrics(
    values: list[float], *, extra: dict[int, dict[str, float]] | None = None
) -> FoldMetrics:
    """One primary value per fold; ``extra`` adds other metrics to specific folds."""
    folds = []
    for index, value in enumerate(values):
        metrics = {PRIMARY: value}
        if extra and index in extra:
            metrics.update(extra[index])
        folds.append(_fold(index, metrics))
    return FoldMetrics(experiment_id="e", model_name="m", scheme=CvScheme.KFOLD, folds=folds)


def _summary(values: list[float]) -> CvRunSummary:
    return summarize_folds(_fold_metrics(values), PRIMARY)


def _cfg(**overrides) -> CrossValidationConfig:
    base = {"folds": 5, "scheme": CvScheme.KFOLD}
    base.update(overrides)
    return CrossValidationConfig(**base)


# --- summarize_folds ------------------------------------------------------------------------


def test_summarize_folds_computes_mean_std_min_max_and_fold_count():
    values = [0.80, 0.81, 0.79, 0.98, 0.80]
    summary = summarize_folds(_fold_metrics(values), PRIMARY)

    aggregate = summary.metrics[PRIMARY]
    assert aggregate.fold_count == 5
    assert aggregate.min == pytest.approx(0.79)
    assert aggregate.max == pytest.approx(0.98)
    assert aggregate.mean == pytest.approx(statistics.fmean(values))
    assert aggregate.std == pytest.approx(statistics.stdev(values))  # sample std, ddof=1
    assert aggregate.min <= aggregate.mean <= aggregate.max
    assert summary.folds == 5
    assert summary.succeeded_folds == 5
    assert summary.primary_fold_values == values


def test_summarize_folds_single_fold_std_is_zero():
    """With one value the sample std is defined as 0.0 (ddof=1 would be undefined)."""
    summary = summarize_folds(_fold_metrics([0.9]), PRIMARY)
    aggregate = summary.metrics[PRIMARY]
    assert aggregate.fold_count == 1
    assert aggregate.std == 0.0
    assert aggregate.min == aggregate.mean == aggregate.max == pytest.approx(0.9)


def test_summarize_folds_counts_each_metric_where_present():
    """A metric present in only some folds is aggregated over just those folds."""
    metrics = _fold_metrics(
        [0.80, 0.82, 0.84],
        extra={0: {"pr_auc": 0.70}, 2: {"pr_auc": 0.74}},  # pr_auc missing in fold 1
    )
    summary = summarize_folds(metrics, PRIMARY)

    assert summary.metrics[PRIMARY].fold_count == 3
    pr_auc = summary.metrics["pr_auc"]
    assert pr_auc.fold_count == 2
    assert pr_auc.mean == pytest.approx(0.72)
    assert pr_auc.min == pytest.approx(0.70)
    assert pr_auc.max == pytest.approx(0.74)


def test_summarize_folds_tracks_missing_primary_values():
    metrics = _fold_metrics([0.80, 0.82])
    metrics.folds.append(_fold(2, {"pr_auc": 0.5}))  # no primary score in this fold
    summary = summarize_folds(metrics, PRIMARY)

    assert summary.folds == 3
    assert summary.succeeded_folds == 2
    assert summary.primary_fold_values == [0.80, 0.82, None]
    assert primary_metric_missing_folds(metrics) == [2]


# --- instability rule -----------------------------------------------------------------------


def test_instability_flag_set_when_std_exceeds_absolute_threshold():
    # std of these values is ~0.0837 > the default instability_std of 0.05.
    summary = _summary([0.80, 0.81, 0.79, 0.98, 0.80])
    warnings = instability_warnings(summary, _cfg(), model_name="m", primary_metric=PRIMARY)

    assert summary.instability_flag is True
    assert any(w.rule == "cv_unstable_model" for w in warnings)


def test_instability_flag_clear_when_folds_are_near_equal():
    summary = _summary([0.90, 0.905, 0.895, 0.90, 0.90])
    warnings = instability_warnings(summary, _cfg(), model_name="m", primary_metric=PRIMARY)

    assert summary.instability_flag is False
    assert summary.outlier_folds == []
    assert warnings == []


def test_instability_std_boundary_is_strictly_greater_than():
    """std just below the threshold does not flag; just above it does (the CV rule disabled)."""
    # Two folds: std = |a - b| / sqrt(2). Size the half-gap so std sits either side of 0.05.
    half_gap = 0.05 * (2**0.5) / 2  # a half-gap that yields std ~= 0.05

    below = _summary([0.9 - half_gap * 0.9, 0.9 + half_gap * 0.9])
    assert below.metrics[PRIMARY].std < 0.05
    instability_warnings(below, _cfg(instability_cv=10.0), "m", PRIMARY)
    assert below.instability_flag is False

    above = _summary([0.9 - half_gap * 1.1, 0.9 + half_gap * 1.1])
    assert above.metrics[PRIMARY].std > 0.05
    instability_warnings(above, _cfg(instability_cv=10.0), "m", PRIMARY)
    assert above.instability_flag is True


def test_instability_flag_set_by_coefficient_of_variation():
    """Small absolute std but a small mean: the CV rule fires even when the std rule does not."""
    values = [0.010, 0.014, 0.012, 0.016, 0.013]  # std ~0.0023 < 0.05, but CV ~0.18 > 0.10
    summary = _summary(values)
    instability_warnings(summary, _cfg(), model_name="m", primary_metric=PRIMARY)

    aggregate = summary.metrics[PRIMARY]
    assert aggregate.std < 0.05
    assert aggregate.std / abs(aggregate.mean) > 0.10
    assert summary.instability_flag is True


def test_cv_rule_ignored_when_mean_is_essentially_zero():
    """|mean| <= 1e-12 disables the CV rule; only the absolute std rule can fire."""
    summary = _summary([-1e-13, 1e-13])  # mean ~0, tiny std
    instability_warnings(summary, _cfg(), model_name="m", primary_metric=PRIMARY)
    assert summary.instability_flag is False


# --- outlier rule (leave-one-out, D5) -------------------------------------------------------


def test_worked_example_flags_only_the_wild_fold():
    """Design D5 worked example: 0.80, 0.81, 0.79, 0.98, 0.80 flags fold index 3 only."""
    summary = _summary([0.80, 0.81, 0.79, 0.98, 0.80])
    warnings = instability_warnings(summary, _cfg(), model_name="m", primary_metric=PRIMARY)

    assert summary.outlier_folds == [3]
    outlier_warnings = [w for w in warnings if w.rule == "cv_outlier_fold"]
    assert len(outlier_warnings) == 1
    assert outlier_warnings[0].details["fold_index"] == 3


def test_all_near_equal_folds_flag_no_outlier():
    summary = _summary([0.80, 0.80, 0.80, 0.80, 0.80])
    instability_warnings(summary, _cfg(), model_name="m", primary_metric=PRIMARY)
    assert summary.outlier_folds == []


def test_no_outlier_check_when_k_is_two():
    """With fewer than three folds the leave-one-out rule is skipped (D5)."""
    summary = _summary([0.80, 0.98])
    warnings = instability_warnings(summary, _cfg(), model_name="m", primary_metric=PRIMARY)
    assert summary.outlier_folds == []
    assert not any(w.rule == "cv_outlier_fold" for w in warnings)


def test_outlier_min_gap_suppresses_tiny_deviations():
    """One fold differs by less than outlier_min_gap: not flagged even though others are identical."""
    # Others all 0.800, one at 0.815: gap 0.015 < default outlier_min_gap 0.02 -> not an outlier.
    summary = _summary([0.800, 0.800, 0.800, 0.800, 0.815])
    instability_warnings(summary, _cfg(), model_name="m", primary_metric=PRIMARY)
    assert summary.outlier_folds == []

    # A deviation above the min gap (others std 0, so threshold == min gap) is flagged.
    summary_over = _summary([0.800, 0.800, 0.800, 0.800, 0.825])
    instability_warnings(summary_over, _cfg(), model_name="m", primary_metric=PRIMARY)
    assert summary_over.outlier_folds == [4]


# --- rank_cv --------------------------------------------------------------------------------


def _entry(model_name: str, mean: float, std: float) -> ModelComparisonEntry:
    return ModelComparisonEntry(
        model_name=model_name,
        display_name=model_name,
        status=ModelRunStatus.COMPLETED,
        primary_score=mean,
        cv=CvRunSummary(
            folds=3,
            succeeded_folds=3,
            metrics={
                PRIMARY: MetricAggregate(mean=mean, std=std, min=mean, max=mean, fold_count=3)
            },
            primary_fold_values=[mean, mean, mean],
        ),
    )


def test_rank_cv_orders_by_mean_then_std_then_model_name():
    a = _entry("alpha", mean=0.90, std=0.02)
    b = _entry("bravo", mean=0.90, std=0.01)  # same mean, lower std -> ranks above alpha
    c = _entry("charlie", mean=0.95, std=0.05)  # highest mean -> rank 1
    entries = [a, b, c]

    best_model, best_score = rank_cv(entries, MetricDirection.MAXIMIZE, primary_metric=PRIMARY)

    assert best_model == "charlie"
    assert best_score == pytest.approx(0.95)
    assert {e.model_name: e.rank for e in entries} == {"charlie": 1, "bravo": 2, "alpha": 3}


def test_rank_cv_leaves_runs_without_a_cv_summary_unranked():
    ranked = _entry("alpha", mean=0.90, std=0.01)
    unranked = ModelComparisonEntry(
        model_name="bravo", display_name="bravo", status=ModelRunStatus.FAILED
    )
    entries = [ranked, unranked]

    best_model, _ = rank_cv(entries, MetricDirection.MAXIMIZE, primary_metric=PRIMARY)

    assert best_model == "alpha"
    assert ranked.rank == 1
    assert unranked.rank is None
