"""The CV reasoning context and the numeric-claim filter (CV design C7, Req 11.4, 11.5).

Unit checks of the two pure modules. The broad guarantees are proved by Property 15 in
``test_cv_properties.py``; these pin down the behaviour the design calls out explicitly.
"""

from datetime import UTC, datetime

from ml_engine.contracts.common import MetricDirection, ModelRunStatus, ProblemType
from ml_engine.contracts.comparison import ComparisonReport, ModelComparisonEntry
from ml_engine.contracts.cv import (
    CvComparisonInfo,
    CvRunSummary,
    CvScheme,
    EvaluationMode,
    MetricAggregate,
    TestEvaluationReport,
    TestRunResult,
)
from ml_engine.reasoning.cv_context import build_cross_validation_context
from ml_engine.reasoning.numbers import drop_unverified_numbers

NOW = datetime(2026, 10, 14, tzinfo=UTC)


def _cv_comparison() -> ComparisonReport:
    entry = ModelComparisonEntry(
        model_name="logistic_regression",
        display_name="Logistic Regression",
        status=ModelRunStatus.COMPLETED,
        primary_score=0.82,
        cv=CvRunSummary(
            folds=3,
            succeeded_folds=3,
            metrics={
                "roc_auc": MetricAggregate(mean=0.82, std=0.03, min=0.79, max=0.85, fold_count=3),
            },
            primary_fold_values=[0.79, 0.82, 0.85],
            instability_flag=False,
            outlier_folds=[],
        ),
        test_metrics={"roc_auc": 0.80},
    )
    return ComparisonReport(
        experiment_id="exp-1",
        generated_at=NOW,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        direction=MetricDirection.MAXIMIZE,
        best_model="logistic_regression",
        best_score=0.82,
        succeeded_count=1,
        models=[entry],
        cross_validation=CvComparisonInfo(
            scheme=CvScheme.STRATIFIED_KFOLD, folds=3, test_row_count=80
        ),
        test_evaluation=TestEvaluationReport(
            experiment_id="exp-1",
            generated_at=NOW,
            evaluation_mode=EvaluationMode.CV,
            test_row_count=80,
            primary_metric="roc_auc",
            model_source="refit_model",
            runs=[
                TestRunResult(
                    model_name="logistic_regression",
                    status="completed",
                    metrics={"roc_auc": 0.80},
                )
            ],
        ),
    )


# --- cv_context -------------------------------------------------------------------------------


def test_cv_context_carries_the_summary_numbers():
    context = build_cross_validation_context(_cv_comparison())
    assert context["scheme"] == "stratified_kfold"
    assert context["folds"] == 3
    assert context["test_row_count"] == 80
    run = context["runs"][0]
    assert run["model"] == "logistic_regression"
    assert run["primary_fold_values"] == [0.79, 0.82, 0.85]
    assert run["primary_mean"] == 0.82 and run["primary_std"] == 0.03
    assert context["test_evaluation"]["runs"][0]["metrics"]["roc_auc"] == 0.80


def test_cv_context_is_none_for_single_split():
    comparison = _cv_comparison()
    comparison.cross_validation = None
    assert build_cross_validation_context(comparison) is None


def test_cv_context_contains_only_numbers_from_the_comparison():
    comparison = _cv_comparison()
    context = build_cross_validation_context(comparison)

    def numbers(value):
        if isinstance(value, bool):
            return []
        if isinstance(value, (int, float)):
            return [float(value)]
        if isinstance(value, dict):
            return [n for item in value.values() for n in numbers(item)]
        if isinstance(value, (list, tuple)):
            return [n for item in value for n in numbers(item)]
        return []

    allowed = {0.79, 0.82, 0.85, 0.03, 0.80, 3.0, 80.0}
    for number in numbers(context):
        assert round(number, 6) in {round(v, 6) for v in allowed}, number


# --- drop_unverified_numbers ------------------------------------------------------------------


def test_number_free_sentences_survive():
    context = {"roc_auc": 0.82}
    text = "The model is stable across folds. The spread is narrow."
    assert drop_unverified_numbers(text, context) == text


def test_a_fabricated_metric_is_dropped_but_a_correct_citation_survives():
    context = {"roc_auc_mean": 0.82, "folds": 3}
    text = (
        "Performance is consistent across folds. "
        "The mean ROC AUC is 0.82. "
        "One fold hit 0.97 which looks wrong."
    )
    filtered = drop_unverified_numbers(text, context)
    assert "Performance is consistent across folds." in filtered
    assert "0.82" in filtered
    assert "0.97" not in filtered


def test_precision_is_the_tokens_own():
    context = {"roc_auc": 0.7987}
    # The context number is rounded to the token's decimals: 0.7987 -> 0.80 at two decimals
    # (kept) and -> 0.799 at three (kept), but a token that does not match at its own precision
    # is dropped.
    assert drop_unverified_numbers("Score 0.80 here.", context) == "Score 0.80 here."
    assert drop_unverified_numbers("Score 0.799 here.", context) == "Score 0.799 here."
    assert drop_unverified_numbers("Score 0.70 here.", context) == ""
    assert drop_unverified_numbers("Score 0.798 here.", context) == ""


def test_percentage_tokens_match_either_reading():
    # 12% is accepted against the fraction 0.12 or the written 12.
    assert drop_unverified_numbers("It rose 12%.", {"rate": 0.12}) == "It rose 12%."
    assert drop_unverified_numbers("It rose 12%.", {"count": 12}) == "It rose 12%."
    assert drop_unverified_numbers("It rose 12%.", {"other": 0.99}) == ""
