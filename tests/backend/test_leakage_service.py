"""Business context and the leakage investigation service (requirements 11, 12)."""

from datetime import UTC, datetime

import pytest

from backend.errors import ConflictError, NotFoundError, ValidationError
from backend.repositories import ExperimentRepository
from backend.services.leakage import LeakageService
from ml_engine.contracts.common import (
    LeakageRiskLevel,
    ProblemType,
    RecommendedAction,
    SemanticType,
    Severity,
)
from ml_engine.contracts.config import FeatureSelection
from ml_engine.contracts.eda import (
    ColumnProfile,
    DatasetSummary,
    EdaReport,
    TargetAnalysis,
    TargetAssociation,
)
from ml_engine.contracts.experiment import ExperimentDefinition
from ml_engine.contracts.leakage import LeakageFinding, LeakageReport
from ml_engine.contracts.leakage_investigation import InvestigationPhase, UserDecision
from ml_engine.io import ExperimentLayout, write_model
from tests.support.datasets import ARTIFACT_ROOT

NOW = datetime(2026, 10, 14, tzinfo=UTC)
EXP = "exp-leak01"


@pytest.fixture
def repository(settings, store):
    return ExperimentRepository(settings=settings, store=store)


@pytest.fixture
def service(settings, repository, store):
    return LeakageService(settings=settings, repository=repository, store=store)


@pytest.fixture
def layout():
    return ExperimentLayout.for_experiment(ARTIFACT_ROOT, EXP)


@pytest.fixture
def experiment(repository, store, layout):
    repository.create(
        ExperimentDefinition(
            experiment_id=EXP,
            name="churn",
            created_by="dev@local",
            created_at=NOW,
            dataset={"uri": f"{ARTIFACT_ROOT}/d.csv", "file_format": "csv"},
            target_column="churned",
            artifact_prefix=layout.base,
        )
    )
    # EDA + leakage artifacts, as the profiling stage would have written them.
    write_model(
        store,
        layout.eda,
        EdaReport(
            experiment_id=EXP,
            generated_at=NOW,
            duration_seconds=0.1,
            dataset=DatasetSummary(
                row_count=200,
                column_count=2,
                duplicate_row_count=0,
                duplicate_row_percentage=0.0,
                memory_usage_bytes=1,
                file_format="csv",
                source_uri=f"{ARTIFACT_ROOT}/d.csv",
            ),
            target=TargetAnalysis(
                column="churned",
                exists=True,
                inferred_problem_type=ProblemType.BINARY_CLASSIFICATION,
            ),
            columns=[
                _col("charges"),
                _col("future_flag"),
                _col("tenure"),
            ],
            target_associations=[
                TargetAssociation(
                    feature="charges", method="pearson", score=0.95, non_null_count=200
                )
            ],
        ),
    )
    write_model(
        store,
        layout.leakage,
        LeakageReport(
            experiment_id=EXP,
            generated_at=NOW,
            target_column="churned",
            findings=[
                LeakageFinding(
                    feature="churn_copy",
                    rule="duplicate_of_target",
                    severity=Severity.CRITICAL,
                    risk_level=LeakageRiskLevel.CONFIRMED_DUPLICATE,
                    explanation="Identical to the target.",
                    recommended_action=RecommendedAction.STRONGLY_CONSIDER_EXCLUDING,
                    evidence={"match_rate": 1.0},
                )
            ],
        ),
    )


def _col(name: str) -> ColumnProfile:
    return ColumnProfile(
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


# --- business context ---------------------------------------------------------------------------


def test_business_context_is_stored_in_plain_language(service, experiment):
    context = service.set_business_context(
        EXP,
        objective="Predict which customers will churn next month.",
        prediction_scenario="Score at the start of each billing cycle using data known then.",
        user="dev@local",
    )
    assert context.investigate_leakage is True  # defaults on (requirement 11.3)
    assert context.target_column == "churned"
    assert service.get_business_context(EXP).objective.startswith("Predict")
    # The reasoning context carries only the stated intent, no data.
    assert "objective" in context.as_reasoning_context()


def test_business_context_requires_an_objective_and_scenario(service, experiment):
    with pytest.raises(ValidationError):
        service.set_business_context(EXP, objective="  ", prediction_scenario="x")


# --- investigation phases -----------------------------------------------------------------------


def test_pre_training_phase_gathers_rule_association_and_name_warnings(service, experiment):
    report = service.build_pre_training_phase(EXP)
    features = {w.feature for w in report.warnings}
    established = {w.feature for w in report.warnings if w.leakage_established}
    assert "churn_copy" in established  # the deterministic duplicate rule establishes it
    # association for charges is screening evidence, never established
    charges = [w for w in report.warnings if w.feature == "charges"]
    assert charges and all(not w.leakage_established for w in charges)
    assert "future_flag" in features  # suspicious name
    assert report.phase is InvestigationPhase.PRE_TRAINING


def test_rebuilding_the_phase_keeps_user_decisions(service, experiment):
    service.build_pre_training_phase(EXP)
    service.decide_warning(
        EXP,
        phase=InvestigationPhase.PRE_TRAINING,
        warning_id="pre_training:charges:target_association",
        decision=UserDecision.ACCEPT_RISK,
        user="dev@local",
    )
    rebuilt = service.build_pre_training_phase(EXP)
    warning = rebuilt.warning_for("pre_training:charges:target_association")
    assert warning.user_decision is UserDecision.ACCEPT_RISK  # not discarded on rebuild


def test_pre_training_needs_eda(service, repository, layout):
    repository.create(
        ExperimentDefinition(
            experiment_id="exp-noeda",
            name="x",
            created_at=NOW,
            dataset={"uri": f"{ARTIFACT_ROOT}/d.csv", "file_format": "csv"},
            target_column="y",
            artifact_prefix=ExperimentLayout.for_experiment(ARTIFACT_ROOT, "exp-noeda").base,
        )
    )
    with pytest.raises(ConflictError):
        service.build_pre_training_phase("exp-noeda")


# --- decisions and exclusions -------------------------------------------------------------------


def test_a_decision_is_recorded_and_marks_the_warning_resolved(service, experiment):
    service.build_pre_training_phase(EXP)
    warning_id = "pre_training:churn_copy:rule:duplicate_of_target"
    report = service.decide_warning(
        EXP,
        phase=InvestigationPhase.PRE_TRAINING,
        warning_id=warning_id,
        decision=UserDecision.EXCLUDE,
        user="dev@local",
    )
    warning = report.warning_for(warning_id)
    assert warning.user_decision is UserDecision.EXCLUDE
    assert warning.review_status.value == "resolved"
    assert warning.decided_by == "dev@local"


def test_an_unknown_warning_is_a_not_found(service, experiment):
    service.build_pre_training_phase(EXP)
    with pytest.raises(NotFoundError):
        service.decide_warning(
            EXP,
            phase=InvestigationPhase.PRE_TRAINING,
            warning_id="nope",
            decision=UserDecision.KEEP,
        )


def test_nothing_is_excluded_until_explicitly_applied(service, experiment, store, layout):
    service.build_pre_training_phase(EXP)
    service.decide_warning(
        EXP,
        phase=InvestigationPhase.PRE_TRAINING,
        warning_id="pre_training:churn_copy:rule:duplicate_of_target",
        decision=UserDecision.EXCLUDE,
    )
    # A decision alone does not touch the feature selection (approval vs selection, req 12.6).
    assert service.excluded_features(EXP) == ["churn_copy"]

    write_model(
        store,
        layout.selected_features,
        FeatureSelection(selected_features=["churn_copy", "charges", "tenure"]),
    )
    updated = service.apply_exclusions_to_selection(EXP, user="dev@local")
    assert "churn_copy" not in updated.selected_features
    assert "churn_copy" in updated.excluded_features
    assert "charges" in updated.selected_features


def test_applying_exclusions_needs_a_prior_selection(service, experiment):
    service.build_pre_training_phase(EXP)
    with pytest.raises(ConflictError):
        service.apply_exclusions_to_selection(EXP)


def test_the_two_phases_are_separate_artifacts(service, experiment, store, layout):
    service.build_pre_training_phase(EXP)
    assert store.exists(layout.leakage_pre_training)
    assert not store.exists(layout.leakage_post_training)  # post-training not built yet
