"""SDK results writer (free-form-experiments design C2, Req 4.2).

A metric written through this writer is labelled ``engine`` — it was produced by the platform's
deterministic metric functions, so it is comparable with managed results. A script calls
``write_metrics(context, {...})`` on the ``context`` the free-form job passed it; the writer
duck-types ``context.report_engine_metric`` so ``ml_engine`` never imports the job layer.

    from ml_engine.sdk import metrics, results

    def main(context):
        scores = metrics.forecasting_metrics(y_true, y_pred, history=hist)
        results.write_metrics(context, scores)   # labelled 'engine'
"""

from typing import Any


def write_metrics(context: Any, scores: dict[str, float]) -> None:
    """Record each name->value as an engine metric on the free-form context (Req 4.2)."""
    report = getattr(context, "report_engine_metric", None)
    if not callable(report):
        raise TypeError(
            "context does not accept engine metrics; pass the context the free-form job provided"
        )
    for name, value in scores.items():
        report(name, float(value))


def set_baseline(context: Any, baseline: dict[str, float]) -> None:
    """Record the seasonal-naive baseline so a model failing to beat it is visible (Req 5.5)."""
    if hasattr(context, "naive_baseline"):
        context.naive_baseline = {k: float(v) for k, v in baseline.items()}
