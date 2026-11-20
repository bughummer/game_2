"""Build the ``cross_validation`` reasoning context from the comparison (CV design C7, Req 11.4).

The LLM's only numeric inputs about cross-validation come from here, and every number here is
copied from the deterministic ``ComparisonReport`` — the per-Run CV_Summary and the
Test_Evaluation report. Nothing in this module reads row data, recomputes a statistic or invents
a value; it is a pure projection of the comparison into a small, bounded dictionary.

The result is handed to ``ml_engine.reasoning.numbers.drop_unverified_numbers`` as the set of
numbers the LLM narrative is allowed to cite (Req 11.5).
"""

from __future__ import annotations

from typing import Any

from ml_engine.contracts.comparison import ComparisonReport


def build_cross_validation_context(comparison: ComparisonReport) -> dict[str, Any] | None:
    """Return the ``cross_validation`` context, or ``None`` when the experiment is single-split.

    Single_Split_Mode comparisons carry no ``cross_validation`` info, so no CV context is built
    and the reasoning service leaves the key out (the numeric filter still runs in every mode).
    """
    info = comparison.cross_validation
    if info is None:
        return None

    runs: list[dict[str, Any]] = []
    for entry in comparison.models:
        cv = entry.cv
        if cv is None:
            continue
        primary = cv.metrics.get(comparison.primary_metric)
        run: dict[str, Any] = {
            "model": entry.model_name,
            "folds": cv.folds,
            "succeeded_folds": cv.succeeded_folds,
            "instability_flag": cv.instability_flag,
            "outlier_folds": list(cv.outlier_folds),
            # Only the primary metric's per-fold values and aggregates are exposed as numbers,
            # matching what the narrative is expected to discuss (Req 11.4).
            "primary_fold_values": [v for v in cv.primary_fold_values if v is not None],
        }
        if primary is not None:
            run["primary_mean"] = primary.mean
            run["primary_std"] = primary.std
            run["primary_min"] = primary.min
            run["primary_max"] = primary.max
        runs.append(run)

    context: dict[str, Any] = {
        "scheme": info.scheme.value,
        "folds": info.folds,
        "runs": runs,
    }
    if info.test_row_count is not None:
        context["test_row_count"] = info.test_row_count

    test = comparison.test_evaluation
    if test is not None:
        context["test_evaluation"] = {
            "test_row_count": test.test_row_count,
            "primary_metric": test.primary_metric,
            "runs": [
                {"model": run.model_name, "status": run.status, "metrics": dict(run.metrics)}
                for run in test.runs
            ],
        }
    return context
