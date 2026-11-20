"""The leakage investigation builders (requirement 12): warnings from deterministic outputs only.

These tests assert the invariant that defines the capability — association is screening evidence,
never proof — and that the builders compute no new statistic, only translate existing ones.
"""

from datetime import UTC, datetime

import pytest

from ml_engine.contracts.common import (
    LeakageRiskLevel,
    ModelRunStatus,
    ProblemType,
    RecommendedAction,
    Severity,
)
from ml_engine.contracts.eda import EdaReport, TargetAssociation
from ml_engine.contracts.leakage import LeakageFinding, LeakageReport
from ml_engine.contracts.leakage_investigation import (
    Confidence,
    EvidenceKind,
    InvestigationPhase,
)
from ml_engine.contracts.model import FeatureImportance, FeatureImportanceEntry, ModelMetadata
from ml_engine.leakage.investigation import build_post_training, build_pre_training

NOW = datetime(2026, 10, 14, tzinfo=UTC)


def _eda(associations=None, column_names=None) -> EdaReport:
    from ml_engine.contracts.eda import DatasetSummary, TargetAnalysis

    columns = []
    from ml_engine.contracts.common import SemanticType
    from ml_engine.contracts.eda import ColumnProfile

    for name in column_names or []:
        columns.append(
            ColumnProfile(
                name=name,
                semantic_type=SemanticType.NUMERIC_CONTINUOUS,
                dtype="float64",
                missing_count=0,
                missing_percentage=0.0,
                unique_count=100,
                unique_percentage=50.0,
                is_constant=False,
                is_high_cardinality=False,
                is_likely_id=False,
            )
        )
    return EdaReport(
        experiment_id="exp-1",
        generated_at=NOW,
        duration_seconds=0.1,
        dataset=DatasetSummary(
            row_count=200,
            column_count=len(columns),
            duplicate_row_count=0,
            duplicate_row_percentage=0.0,
            memory_usage_bytes=1,
            file_format="csv",
            source_uri="s3://b/d.csv",
        ),
        target=TargetAnalysis(
            column="churned", exists=True, inferred_problem_type=ProblemType.BINARY_CLASSIFICATION
        ),
        columns=columns,
        target_associations=associations or [],
    )


def _leakage(findings=None) -> LeakageReport:
    return LeakageReport(
        experiment_id="exp-1", generated_at=NOW, target_column="churned", findings=findings or []
    )


# --- pre-training -------------------------------------------------------------------------------


def test_a_high_association_warns_but_never_establishes_leakage():
    eda = _eda(
        associations=[
            TargetAssociation(feature="charges", method="pearson", score=0.95, non_null_count=200)
        ]
    )
    report = build_pre_training(
        experiment_id="exp-1", target_column="churned", eda=eda, leakage=_leakage()
    )

    assert len(report.warnings) == 1
    warning = report.warnings[0]
    assert warning.method is EvidenceKind.TARGET_ASSOCIATION
    assert warning.leakage_established is False  # the core invariant (requirement 12.3)
    assert warning.confidence is Confidence.HIGH
    assert warning.evidence["association_score"] == pytest.approx(0.95)


def test_a_deterministic_duplicate_rule_does_establish_leakage():
    finding = LeakageFinding(
        feature="churn_copy",
        rule="duplicate_of_target",
        severity=Severity.CRITICAL,
        risk_level=LeakageRiskLevel.CONFIRMED_DUPLICATE,
        explanation="This column is identical to the target.",
        recommended_action=RecommendedAction.STRONGLY_CONSIDER_EXCLUDING,
        evidence={"match_rate": 1.0},
    )
    report = build_pre_training(
        experiment_id="exp-1", target_column="churned", eda=_eda(), leakage=_leakage([finding])
    )

    warning = report.warning_for("pre_training:churn_copy:rule:duplicate_of_target")
    assert warning is not None
    assert warning.leakage_established is True
    assert warning.method is EvidenceKind.DETERMINISTIC_RULE


def test_a_suspicious_name_is_a_low_confidence_hint_only():
    eda = _eda(column_names=["future_balance", "tenure_months"])
    report = build_pre_training(
        experiment_id="exp-1", target_column="churned", eda=eda, leakage=_leakage()
    )

    names = {w.feature: w for w in report.warnings}
    assert "future_balance" in names
    assert names["future_balance"].method is EvidenceKind.NAME_PATTERN
    assert names["future_balance"].confidence is Confidence.LOW
    assert names["future_balance"].leakage_established is False
    assert "tenure_months" not in names  # an ordinary name is not flagged


def test_an_llm_note_attaches_to_the_relevant_warning():
    eda = _eda(
        associations=[
            TargetAssociation(feature="charges", method="pearson", score=0.95, non_null_count=200)
        ]
    )
    report = build_pre_training(
        experiment_id="exp-1",
        target_column="churned",
        eda=eda,
        leakage=_leakage(),
        llm_notes={"charges": "This looks like a post-billing total."},
    )
    assert report.warnings[0].llm_reasoning == "This looks like a post-billing total."


def test_warning_ids_are_stable_across_rebuilds():
    eda = _eda(
        associations=[
            TargetAssociation(feature="charges", method="pearson", score=0.95, non_null_count=200)
        ]
    )
    first = build_pre_training(
        experiment_id="exp-1", target_column="churned", eda=eda, leakage=_leakage()
    )
    second = build_pre_training(
        experiment_id="exp-1", target_column="churned", eda=eda, leakage=_leakage()
    )
    assert [w.id for w in first.warnings] == [w.id for w in second.warnings]


def test_the_phase_is_tagged_pre_training():
    report = build_pre_training(
        experiment_id="exp-1", target_column="churned", eda=_eda(), leakage=_leakage()
    )
    assert report.phase is InvestigationPhase.PRE_TRAINING


# --- post-training ------------------------------------------------------------------------------


def _model_with_dominant_feature(feature: str, share: float) -> ModelMetadata:
    return ModelMetadata(
        experiment_id="exp-1",
        model_name="xgboost",
        display_name="XGBoost",
        status=ModelRunStatus.COMPLETED,
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        library="xgboost",
        library_version="3.4.1",
        features_used=[feature, "other"],
        feature_count=2,
        training_row_count=160,
        validation_row_count=40,
        training_started_at=NOW,
        training_completed_at=NOW,
        training_duration_seconds=1.0,
        primary_metric="roc_auc",
        primary_score=0.99,
        feature_importance=FeatureImportance(
            method="native",
            entries=[
                FeatureImportanceEntry(feature=feature, importance=share, raw_value=share, rank=1),
                FeatureImportanceEntry(
                    feature="other", importance=1 - share, raw_value=1 - share, rank=2
                ),
            ],
        ),
    )


def test_a_dominant_feature_is_flagged_without_claiming_leakage():
    report = build_post_training(
        experiment_id="exp-1",
        target_column="churned",
        models=[_model_with_dominant_feature("suspicious", 0.92)],
    )
    assert len(report.warnings) == 1
    warning = report.warnings[0]
    assert warning.feature == "suspicious"
    assert warning.method is EvidenceKind.FEATURE_IMPORTANCE
    assert warning.leakage_established is False
    assert warning.evidence["importance_share"] == pytest.approx(0.92)


def test_an_ablation_collapse_establishes_leakage():
    report = build_post_training(
        experiment_id="exp-1",
        target_column="churned",
        models=[],
        ablation_deltas={"suspicious": 0.4},
    )
    warning = report.warning_for("post_training:suspicious:ablation")
    assert warning is not None
    assert warning.leakage_established is True
    assert warning.severity is Severity.CRITICAL


def test_a_small_ablation_delta_is_review_not_leakage():
    report = build_post_training(
        experiment_id="exp-1",
        target_column="churned",
        models=[],
        ablation_deltas={"ordinary": 0.02},
    )
    warning = report.warning_for("post_training:ordinary:ablation")
    assert warning.leakage_established is False
