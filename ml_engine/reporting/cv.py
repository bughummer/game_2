"""CV aggregation, ranking and instability/outlier warnings, pure (ML Factory 2.0, CV design C6).

No IO. The evaluation job (``jobs/evaluation/cv.py``) reads ``fold_metrics.json`` per Run, calls
``summarize_folds`` to build the CV_Summary, ``instability_warnings`` to flag unstable Runs and
outlier folds (the leave-one-out rule, D5), and ``rank_cv`` to order the Runs by their CV mean.

All thresholds come from the frozen ``CrossValidationConfig`` so the flags are reproducible from
the config alone, never from mutable Settings. The primary metric name is passed explicitly: the
per-fold primary values and the ``metrics[primary]`` aggregate are the same measurement, so the
caller names it rather than the module guessing.
"""

import statistics

from ml_engine.contracts.common import (
    MetricDirection,
    Severity,
    WarningCategory,
)
from ml_engine.contracts.comparison import ModelComparisonEntry
from ml_engine.contracts.cv import (
    CrossValidationConfig,
    CvRunSummary,
    FoldMetrics,
    MetricAggregate,
)
from ml_engine.contracts.warnings import AnalysisWarning


def summarize_folds(fold_metrics: FoldMetrics, primary_metric: str) -> CvRunSummary:  # noqa: ARG001
    """Aggregate one Run's folds into a :class:`CvRunSummary` (Property 10, Req 7.1, 7.2).

    ``primary_metric`` is part of the design's public signature (callers name it alongside the
    aggregates); the aggregation itself covers every metric, so it is not read here.

    For every metric present in at least one fold: ``mean`` (clamped into ``[min, max]`` to absorb
    floating-point rounding), sample ``std`` (``ddof=1``; ``0.0`` with a single value), ``min``,
    ``max`` and ``fold_count``. ``primary_fold_values`` preserves each fold's ``primary_score``
    (None included), ordered by ``fold_index``. ``succeeded_folds`` counts folds with a non-None
    primary score. The instability flag and outlier folds are left at their defaults here and set
    by :func:`instability_warnings`.
    """
    ordered = sorted(fold_metrics.folds, key=lambda result: result.fold_index)

    collected: dict[str, list[float]] = {}
    for result in ordered:
        for name, value in result.metrics.values.items():
            collected.setdefault(name, []).append(value)

    aggregates: dict[str, MetricAggregate] = {}
    for name, values in collected.items():
        low = min(values)
        high = max(values)
        mean = statistics.fmean(values)
        # Clamp for float noise so min <= mean <= max holds exactly (Property 10).
        mean = min(max(mean, low), high)
        std = statistics.stdev(values) if len(values) >= 2 else 0.0
        aggregates[name] = MetricAggregate(
            mean=mean, std=std, min=low, max=high, fold_count=len(values)
        )

    primary_fold_values = [result.primary_score for result in ordered]
    succeeded_folds = sum(1 for value in primary_fold_values if value is not None)

    return CvRunSummary(
        folds=len(ordered),
        succeeded_folds=succeeded_folds,
        metrics=aggregates,
        primary_fold_values=primary_fold_values,
    )


def primary_metric_missing_folds(fold_metrics: FoldMetrics) -> list[int]:
    """Fold indices whose primary metric could not be computed (helper for the HIGH warning)."""
    return [
        result.fold_index
        for result in sorted(fold_metrics.folds, key=lambda r: r.fold_index)
        if result.primary_score is None
    ]


def primary_metric_missing_warning(
    model_name: str, primary_metric: str, missing_folds: list[int]
) -> AnalysisWarning:
    """The HIGH ``cv_primary_metric_missing`` warning that makes the Run unrankable (Req 7.1)."""
    folds = ", ".join(str(index) for index in missing_folds)
    return AnalysisWarning(
        rule="cv_primary_metric_missing",
        category=WarningCategory.METRIC,
        severity=Severity.HIGH,
        message=(
            f"'{model_name}' did not produce the primary metric '{primary_metric}' in "
            f"fold(s) {folds}, so its cross-validation estimate is incomplete and the Run "
            "cannot be ranked."
        ),
        details={"model": model_name, "metric": primary_metric, "folds": folds},
    )


def instability_warnings(
    summary: CvRunSummary,
    cfg: CrossValidationConfig,
    model_name: str,
    primary_metric: str,
) -> list[AnalysisWarning]:
    """Flag unstable Runs and outlier folds, and set the flags on ``summary`` (Property 12, D5).

    Instability (Req 7.5): the primary metric's ``std > cfg.instability_std`` or
    (``|mean| > 1e-12`` and ``std / |mean| > cfg.instability_cv``). Outlier folds (Req 7.6, D5,
    leave-one-out): only when at least 3 folds have a primary value; a fold is flagged when its
    distance from the *other* folds' mean exceeds
    ``max(cfg.outlier_multiplier * stdev(others), cfg.outlier_min_gap)``. Skipped when K = 2.
    """
    warnings: list[AnalysisWarning] = []
    aggregate = summary.metrics.get(primary_metric)
    values = [value for value in summary.primary_fold_values if value is not None]

    unstable = False
    if aggregate is not None and values:
        std = aggregate.std
        mean = aggregate.mean
        by_std = std > cfg.instability_std
        by_cv = abs(mean) > 1e-12 and std / abs(mean) > cfg.instability_cv
        if by_std or by_cv:
            unstable = True
            fold_text = ", ".join(f"{value:.4f}" for value in values)
            warnings.append(
                AnalysisWarning(
                    rule="cv_unstable_model",
                    category=WarningCategory.MODEL,
                    severity=Severity.HIGH,
                    message=(
                        f"'{model_name}' scores vary across folds (std {std:.4f}, "
                        f"values {fold_text}). A high spread means the estimate is unreliable; "
                        "treat the mean with caution."
                    ),
                    details={"model": model_name, "std": round(std, 6)},
                )
            )

    outliers = _outlier_folds(summary, cfg)
    for fold_index, value, others_mean in outliers:
        warnings.append(
            AnalysisWarning(
                rule="cv_outlier_fold",
                category=WarningCategory.MODEL,
                severity=Severity.MEDIUM,
                message=(
                    f"'{model_name}' fold {fold_index} scored {value:.4f}, far from the other "
                    f"folds' mean of {others_mean:.4f}. One fold standing out can signal leakage "
                    "or a distribution shift between folds; review that fold's data."
                ),
                details={
                    "model": model_name,
                    "fold_index": fold_index,
                    "value": round(value, 6),
                    "others_mean": round(others_mean, 6),
                },
            )
        )

    summary.instability_flag = unstable
    summary.outlier_folds = [fold_index for fold_index, _, _ in outliers]
    return warnings


def _outlier_folds(
    summary: CvRunSummary, cfg: CrossValidationConfig
) -> list[tuple[int, float, float]]:
    """Leave-one-out outlier folds (D5). Returns ``(fold_index, value, others_mean)`` tuples."""
    indexed = [
        (index, value)
        for index, value in enumerate(summary.primary_fold_values)
        if value is not None
    ]
    if len(indexed) < 3:  # skipped when fewer than 3 folds have a value (so K = 2 never flags)
        return []

    flagged: list[tuple[int, float, float]] = []
    for position, (fold_index, value) in enumerate(indexed):
        others = [other for other_pos, (_, other) in enumerate(indexed) if other_pos != position]
        others_mean = statistics.fmean(others)
        others_std = statistics.stdev(others) if len(others) >= 2 else 0.0
        threshold = max(cfg.outlier_multiplier * others_std, cfg.outlier_min_gap)
        if abs(value - others_mean) > threshold:
            flagged.append((fold_index, value, others_mean))
    return flagged


def rank_cv(
    entries: list[ModelComparisonEntry], direction: MetricDirection, primary_metric: str
) -> tuple[str | None, float | None]:
    """Rank Runs by their CV mean primary metric, in place (Property 11, Req 7.3, 7.4).

    Only entries with a CV_Summary carrying the primary mean are ranked. Sort key is the
    direction-adjusted mean, then the lower primary std, then the run id (model name). Ranks are
    assigned 1..n; unrankable entries keep ``rank = None``. Does not read test metrics. Returns
    the best Run's ``(model_name, mean)`` or ``(None, None)`` when nothing is rankable.
    """
    rankable = [entry for entry in entries if _cv_primary_mean(entry, primary_metric) is not None]

    maximize = direction is MetricDirection.MAXIMIZE

    def sort_key(entry: ModelComparisonEntry) -> tuple[float, float, str]:
        mean = _cv_primary_mean(entry, primary_metric)
        assert mean is not None  # filtered above
        adjusted = -mean if maximize else mean
        std = _cv_primary_std(entry, primary_metric)
        return (adjusted, std, entry.model_name)

    rankable.sort(key=sort_key)
    ranked_ids = {id(entry) for entry in rankable}
    for rank, entry in enumerate(rankable, start=1):
        entry.rank = rank
    for entry in entries:
        if id(entry) not in ranked_ids:
            entry.rank = None

    if not rankable:
        return None, None
    best = rankable[0]
    return best.model_name, _cv_primary_mean(best, primary_metric)


def _cv_primary_mean(entry: ModelComparisonEntry, primary_metric: str) -> float | None:
    if entry.cv is None:
        return None
    aggregate = entry.cv.metrics.get(primary_metric)
    return aggregate.mean if aggregate is not None else None


def _cv_primary_std(entry: ModelComparisonEntry, primary_metric: str) -> float:
    if entry.cv is None:
        return 0.0
    aggregate = entry.cv.metrics.get(primary_metric)
    return aggregate.std if aggregate is not None else 0.0
