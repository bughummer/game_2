"""Report documents (requirement 14): a view over artifacts that adds no numbers."""

from datetime import UTC, datetime

from ml_engine.contracts.common import (
    ClassWeighting,
    ExperimentStatus,
    MetricDirection,
    ModelRunStatus,
    ProblemType,
    RequestedProblemType,
    SemanticType,
)
from ml_engine.contracts.comparison import (
    ComparisonReport,
    ExperimentSummary,
    ModelComparisonEntry,
)
from ml_engine.contracts.config import (
    DatasetReference,
    EnvironmentCapture,
    ExperimentConfig,
    FeatureSelection,
    ModelSpec,
    PreprocessingConfig,
    SplitConfig,
)
from ml_engine.contracts.cv import (
    CvComparisonInfo,
    CvRunSummary,
    CvScheme,
    EvaluationMode,
    MetricAggregate,
    TestEvaluationReport,
    TestRunResult,
)
from ml_engine.contracts.eda import (
    ColumnProfile,
    DatasetSummary,
    EdaReport,
    TargetAnalysis,
    TargetAssociation,
)
from ml_engine.contracts.model import FeatureImportance, FeatureImportanceEntry
from ml_engine.contracts.report import ReportKind
from ml_engine.reporting.documents import build_comparison_report, build_eda_report
from ml_engine.reporting.render import render_html

NOW = datetime(2026, 10, 14, tzinfo=UTC)


def _eda() -> EdaReport:
    return EdaReport(
        experiment_id="exp-1",
        generated_at=NOW,
        duration_seconds=0.1,
        dataset=DatasetSummary(
            row_count=400,
            column_count=2,
            duplicate_row_count=0,
            duplicate_row_percentage=0.0,
            memory_usage_bytes=1,
            file_format="csv",
            source_uri="s3://b/d.csv",
        ),
        target=TargetAnalysis(
            column="churned", exists=True, inferred_problem_type=ProblemType.BINARY_CLASSIFICATION
        ),
        columns=[
            ColumnProfile(
                name="charges",
                semantic_type=SemanticType.NUMERIC_CONTINUOUS,
                dtype="float64",
                missing_count=0,
                missing_percentage=0.0,
                unique_count=100,
                unique_percentage=25.0,
                is_constant=False,
                is_high_cardinality=False,
                is_likely_id=False,
            )
        ],
        target_associations=[
            TargetAssociation(feature="charges", method="pearson", score=0.95, non_null_count=400)
        ],
    )


def _summary() -> ExperimentSummary:
    config = ExperimentConfig(
        experiment_id="exp-1",
        name="churn",
        created_at=NOW,
        dataset=DatasetReference(uri="s3://b/d.csv", file_format="csv"),
        target_column="churned",
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        requested_problem_type=RequestedProblemType.AUTO,
        primary_metric="roc_auc",
        feature_selection=FeatureSelection(
            selected_features=["charges"], excluded_features=["churn_copy"]
        ),
        split=SplitConfig(),
        preprocessing=PreprocessingConfig(),
        models=[ModelSpec(name="logistic_regression")],
        class_weighting=ClassWeighting.AUTO,
        environment=EnvironmentCapture(
            python_version="3.14", platform="t", ml_factory_version="2.0.0", captured_at=NOW
        ),
        artifact_prefix="s3://b/ml-factory/experiments/exp-1",
    )
    comparison = ComparisonReport(
        experiment_id="exp-1",
        generated_at=NOW,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="roc_auc",
        direction=MetricDirection.MAXIMIZE,
        best_model="logistic_regression",
        best_score=0.82,
        succeeded_count=1,
        failed_count=0,
        models=[
            ModelComparisonEntry(
                model_name="logistic_regression",
                display_name="Logistic Regression",
                status=ModelRunStatus.COMPLETED,
                primary_score=0.82,
                rank=1,
                feature_count=1,
                training_duration_seconds=1.2,
            )
        ],
    )
    return ExperimentSummary(
        experiment_id="exp-1",
        name="churn",
        generated_at=NOW,
        status=ExperimentStatus.COMPLETED,
        config=config,
        dataset_row_count=400,
        dataset_column_count=2,
        selected_feature_count=1,
        excluded_feature_count=1,
        comparison=comparison,
        best_model_importance=FeatureImportance(
            method="coefficient_magnitude",
            entries=[
                FeatureImportanceEntry(feature="charges", importance=1.0, raw_value=1.0, rank=1)
            ],
        ),
    )


def _cv_summary() -> ExperimentSummary:
    """The single-split summary, re-dressed as a CV_Mode experiment with a Test_Fold."""
    summary = _summary()
    entry = summary.comparison.models[0]
    entry.cv = CvRunSummary(
        folds=3,
        succeeded_folds=3,
        metrics={
            "roc_auc": MetricAggregate(mean=0.82, std=0.03, min=0.79, max=0.85, fold_count=3),
            "f1": MetricAggregate(mean=0.71, std=0.05, min=0.66, max=0.76, fold_count=3),
        },
        primary_fold_values=[0.79, 0.82, 0.85],
        instability_flag=False,
    )
    entry.test_metrics = {"roc_auc": 0.80, "f1": 0.69}
    summary.comparison.cross_validation = CvComparisonInfo(
        scheme=CvScheme.STRATIFIED_KFOLD, folds=3, test_row_count=80
    )
    summary.comparison.test_evaluation = TestEvaluationReport(
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
                metrics={"roc_auc": 0.80, "f1": 0.69},
            )
        ],
    )
    return summary


def test_eda_report_copies_dataset_target_and_associations():
    report = build_eda_report(eda=_eda())
    assert report.kind is ReportKind.EDA
    titles = {s.title for s in report.sections}
    assert {"Dataset", "Target", "Columns", "Strongest target associations"} <= titles
    assoc = next(s for s in report.sections if s.title == "Strongest target associations")
    assert assoc.rows[0]["feature"] == "charges" and assoc.rows[0]["score"] == "0.9500"
    assert any(s.endswith("eda.json") for s in report.source_artifacts)


def test_comparison_report_includes_models_importance_and_selection():
    report = build_comparison_report(summary=_summary())
    assert report.kind is ReportKind.MODEL_COMPARISON
    titles = {s.title for s in report.sections}
    assert {
        "Experiment",
        "Model comparison",
        "Best model — feature importance",
        "Feature selection",
    } <= titles
    comp = next(s for s in report.sections if s.title == "Model comparison")
    assert comp.rows[0]["model"] == "Logistic Regression"
    assert comp.rows[0]["roc_auc"] == "0.8200"


def test_a_narrative_is_recorded_with_its_source_but_adds_no_numbers():
    report = build_comparison_report(
        summary=_summary(),
        narrative="Logistic regression led on ROC AUC.",
        narrative_source="bedrock:model-x",
    )
    assert report.narrative == "Logistic regression led on ROC AUC."
    assert report.narrative_source == "bedrock:model-x"


def test_html_render_is_self_contained_and_escapes():
    report = build_eda_report(eda=_eda())
    html = render_html(report)
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "<style>" in html  # inline styles, no external assets
    assert "charges" in html


def test_html_escapes_dangerous_values():
    eda = _eda()
    eda.columns[0].name = "<script>alert(1)</script>"
    html = render_html(build_eda_report(eda=eda))
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html


# --- cross-validation sections (Req 11.2, 11.3) -----------------------------------------------


def test_comparison_report_adds_cv_and_test_sections():
    report = build_comparison_report(summary=_cv_summary())
    by_title = {s.title: s for s in report.sections}
    assert "Cross-validation" in by_title
    assert "Cross-validation — per-model fold metrics" in by_title
    assert "Test evaluation" in by_title

    overview = by_title["Cross-validation"]
    overview_metrics = {row["metric"]: row["value"] for row in overview.rows}
    assert overview_metrics["CV scheme"] == "stratified_kfold"
    assert overview_metrics["Folds (K)"] == "3"
    assert overview_metrics["Test_Fold rows"] == "80"

    folds = by_title["Cross-validation — per-model fold metrics"]
    roc_row = next(
        r for r in folds.rows if r["model"] == "Logistic Regression" and r["metric"] == "roc_auc"
    )
    assert roc_row["mean ± std"] == "0.8200 ± 0.0300"
    assert roc_row["min"] == "0.7900" and roc_row["max"] == "0.8500"
    per_fold = next(r for r in folds.rows if r["metric"].endswith("(per fold)"))
    assert per_fold["mean ± std"] == "0.7900, 0.8200, 0.8500"
    assert per_fold["folds"] == "stable"

    test = by_title["Test evaluation"]
    assert "80 untouched test rows" in test.summary
    assert "refit model" in test.summary
    test_metrics = {r["metric"]: r["value"] for r in test.rows}
    assert test_metrics["roc_auc"] == "0.8000" and test_metrics["f1"] == "0.6900"


def test_comparison_report_labels_the_cv_estimate():
    report = build_comparison_report(summary=_cv_summary())
    model_section = next(s for s in report.sections if s.title == "Model comparison")
    assert "CV estimate" in model_section.summary
    assert "Refit_Model itself is not measured" in model_section.summary


def test_cv_report_renders_to_html_without_error():
    report = build_comparison_report(summary=_cv_summary())
    html = render_html(report)
    assert html.startswith("<!doctype html>") and html.rstrip().endswith("</html>")
    assert "Cross-validation" in html
    assert "Test evaluation" in html
    assert "0.8200 ± 0.0300" in html


def test_single_split_report_has_no_cv_sections():
    report = build_comparison_report(summary=_summary())
    titles = {s.title for s in report.sections}
    assert "Cross-validation" not in titles
    assert "Test evaluation" not in titles
    model_section = next(s for s in report.sections if s.title == "Model comparison")
    assert "CV estimate" not in model_section.summary
