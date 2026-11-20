"""Assemble report documents from existing artifacts (requirement 14). Pure; computes nothing.

Every value in a ``Report`` is copied from the EDA report, the comparison or the summary. These
builders never recompute a statistic, so a report cannot drift from the artifacts it summarises.
The model-comparison report includes everything brief §9 asks for that the artifacts carry:
algorithms, feature sets, metrics, importance, ranking, duration, failures and warnings. Spot/cost
is deliberately absent — that is the MLOps team's domain (owner decision), not ML Factory's report.
"""

from datetime import UTC, datetime

from ml_engine.contracts.comparison import ComparisonReport, ExperimentSummary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.report import Report, ReportKind, ReportSection
from ml_engine.io.layout import COMPARISON_FILE, EDA_FILE, LEAKAGE_FILE, SUMMARY_FILE


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def build_eda_report(
    *,
    eda: EdaReport,
    leakage: LeakageReport | None = None,
    narrative: str | None = None,
    narrative_source: str | None = None,
) -> Report:
    sections: list[ReportSection] = []

    sections.append(
        ReportSection(
            title="Dataset",
            rows=[
                {"metric": "Rows", "value": _fmt(eda.dataset.row_count)},
                {"metric": "Columns", "value": _fmt(eda.dataset.column_count)},
                {
                    "metric": "Duplicate rows",
                    "value": f"{eda.dataset.duplicate_row_percentage:.2f}%",
                },
                {"metric": "Sampled", "value": _fmt(eda.dataset.sampled)},
                {"metric": "Format", "value": eda.dataset.file_format},
            ],
        )
    )

    target = eda.target
    sections.append(
        ReportSection(
            title="Target",
            rows=[
                {"metric": "Column", "value": target.column},
                {
                    "metric": "Problem type",
                    "value": _fmt(
                        target.inferred_problem_type.value if target.inferred_problem_type else None
                    ),
                },
                {"metric": "Missing", "value": f"{target.missing_percentage:.2f}%"},
                {"metric": "Imbalance ratio", "value": _fmt(target.imbalance_ratio)},
            ],
        )
    )

    sections.append(
        ReportSection(
            title="Columns",
            summary=f"{len(eda.columns)} columns profiled.",
            rows=[
                {
                    "name": c.name,
                    "type": c.semantic_type.value,
                    "missing": f"{c.missing_percentage:.1f}%",
                    "unique": _fmt(c.unique_count),
                }
                for c in eda.columns
            ],
        )
    )

    if eda.target_associations:
        sections.append(
            ReportSection(
                title="Strongest target associations",
                summary="A lone predictor near 1.0 is the loudest sign of a leak (not proof).",
                rows=[
                    {"feature": a.feature, "method": a.method, "score": _fmt(a.score)}
                    for a in eda.target_associations[:20]
                ],
            )
        )

    if leakage and leakage.findings:
        sections.append(
            ReportSection(
                title="Deterministic leakage findings",
                rows=[
                    {
                        "feature": f.feature,
                        "rule": f.rule,
                        "risk": f.risk_level.value,
                        "severity": f.severity.value,
                    }
                    for f in leakage.findings
                ],
            )
        )

    if eda.warnings:
        sections.append(
            ReportSection(
                title="Warnings",
                notes=[f"[{w.severity.value}] {w.message}" for w in eda.warnings],
            )
        )

    sources = [EDA_FILE]
    if leakage and leakage.findings:
        sources.append(LEAKAGE_FILE)
    return Report(
        experiment_id=eda.experiment_id,
        kind=ReportKind.EDA,
        title=f"EDA report — {eda.experiment_id}",
        generated_at=datetime.now(UTC),
        sections=sections,
        source_artifacts=sources,
        narrative=narrative,
        narrative_source=narrative_source,
    )


def _cv_sections(comparison: ComparisonReport) -> list[ReportSection]:
    """The cross-validation and Test_Fold sections (Req 11.2, 11.3).

    Empty in Single_Split_Mode, so single-split reports are unchanged. Every number is copied
    from the comparison's CV summaries and the Test_Evaluation report.
    """
    sections: list[ReportSection] = []
    info = comparison.cross_validation
    if info is not None:
        # Scheme / K / Test_Fold size.
        overview_rows = [
            {"metric": "CV scheme", "value": info.scheme.value},
            {"metric": "Folds (K)", "value": _fmt(info.folds)},
        ]
        if info.test_row_count is not None:
            overview_rows.append({"metric": "Test_Fold rows", "value": _fmt(info.test_row_count)})
        sections.append(
            ReportSection(
                title="Cross-validation",
                summary=(
                    "CV estimate (mean of K folds); the Refit_Model itself is not measured. "
                    "Ranking is by mean CV score."
                ),
                rows=overview_rows,
            )
        )
        # Per-Run mean ± std and min/max per metric, plus per-fold primary values and the flag.
        per_run_rows: list[dict[str, str]] = []
        for entry in comparison.models:
            cv = entry.cv
            if cv is None:
                continue
            fold_values = ", ".join(_fmt(v) for v in cv.primary_fold_values)
            for metric_name, agg in cv.metrics.items():
                per_run_rows.append(
                    {
                        "model": entry.display_name,
                        "metric": metric_name,
                        "mean ± std": f"{_fmt(agg.mean)} ± {_fmt(agg.std)}",
                        "min": _fmt(agg.min),
                        "max": _fmt(agg.max),
                        "folds": _fmt(agg.fold_count),
                    }
                )
            per_run_rows.append(
                {
                    "model": entry.display_name,
                    "metric": f"{comparison.primary_metric} (per fold)",
                    "mean ± std": fold_values or "—",
                    "min": "",
                    "max": "",
                    "folds": "unstable" if cv.instability_flag else "stable",
                }
            )
        if per_run_rows:
            sections.append(
                ReportSection(
                    title="Cross-validation — per-model fold metrics",
                    summary=(
                        "Mean ± standard deviation and min/max of every metric across folds, "
                        "with the per-fold primary-metric values and the instability flag."
                    ),
                    rows=per_run_rows,
                )
            )
    # The Test_Evaluation is reported in its own section and never affects the ranking (Req 11.2).
    test = comparison.test_evaluation
    if test is not None:
        test_rows: list[dict[str, str]] = []
        for run in test.runs:
            base = {"model": run.model_name, "status": run.status}
            if run.status == "completed" and run.metrics:
                for metric_name, value in run.metrics.items():
                    test_rows.append({**base, "metric": metric_name, "value": _fmt(value)})
            else:
                test_rows.append(
                    {**base, "metric": "—", "value": run.message or run.error_code or "—"}
                )
        sections.append(
            ReportSection(
                title="Test evaluation",
                summary=(
                    f"{test.test_row_count} untouched test rows scored once with the "
                    f"{test.model_source.replace('_', ' ')}; "
                    "never used for ranking, tuning or thresholds."
                ),
                rows=test_rows,
            )
        )
    return sections


def build_comparison_report(
    *,
    summary: ExperimentSummary,
    narrative: str | None = None,
    narrative_source: str | None = None,
) -> Report:
    comparison: ComparisonReport | None = summary.comparison
    sections: list[ReportSection] = []
    config = summary.config

    sections.append(
        ReportSection(
            title="Experiment",
            rows=[
                {"metric": "Name", "value": summary.name},
                {"metric": "Status", "value": summary.status.value},
                {"metric": "Target", "value": config.target_column},
                {"metric": "Problem type", "value": config.problem_type.value},
                {"metric": "Primary metric", "value": config.primary_metric},
                {"metric": "Validation fraction", "value": _fmt(config.split.validation_fraction)},
                {"metric": "Random seed", "value": _fmt(config.split.random_seed)},
                {"metric": "Selected features", "value": _fmt(summary.selected_feature_count)},
                {"metric": "Excluded features", "value": _fmt(summary.excluded_feature_count)},
            ],
        )
    )

    if comparison:
        cv_mode = comparison.cross_validation is not None
        headline = (
            f"Best: {comparison.best_model} "
            f"({comparison.primary_metric}={_fmt(comparison.best_score)}). "
            f"{comparison.succeeded_count} succeeded, {comparison.failed_count} failed."
        )
        if cv_mode:
            # The headline is the cross-validated estimate, not a measurement of any single
            # fitted model (Req 11.3).
            headline += (
                " Scores are the CV estimate (mean of K folds); "
                "the Refit_Model itself is not measured."
            )
        sections.append(
            ReportSection(
                title="Model comparison",
                summary=headline,
                rows=[
                    {
                        "rank": _fmt(e.rank),
                        "model": e.display_name,
                        "status": e.status.value,
                        comparison.primary_metric: _fmt(e.primary_score),
                        "features": _fmt(e.feature_count),
                        "seconds": _fmt(e.training_duration_seconds),
                        "note": e.failure_message or "",
                    }
                    for e in comparison.models
                ],
            )
        )
        sections.extend(_cv_sections(comparison))

    if summary.best_model_importance and summary.best_model_importance.entries:
        sections.append(
            ReportSection(
                title="Best model — feature importance",
                summary=f"Method: {summary.best_model_importance.method}.",
                rows=[
                    {"rank": _fmt(e.rank), "feature": e.feature, "importance": _fmt(e.importance)}
                    for e in summary.best_model_importance.entries[:25]
                ],
            )
        )

    sections.append(
        ReportSection(
            title="Feature selection",
            rows=[
                {"role": "selected", "feature": f}
                for f in config.feature_selection.selected_features
            ]
            + [
                {"role": "excluded", "feature": f}
                for f in config.feature_selection.excluded_features
            ],
        )
    )

    if summary.warnings:
        sections.append(
            ReportSection(
                title="Warnings",
                notes=[f"[{w.severity.value}] {w.message}" for w in summary.warnings],
            )
        )

    return Report(
        experiment_id=summary.experiment_id,
        kind=ReportKind.MODEL_COMPARISON,
        title=f"Model comparison — {summary.name}",
        generated_at=datetime.now(UTC),
        sections=sections,
        source_artifacts=[SUMMARY_FILE, COMPARISON_FILE],
        narrative=narrative,
        narrative_source=narrative_source,
    )
