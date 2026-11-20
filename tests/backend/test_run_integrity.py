"""Run integrity and the wired-up analysis: archives on retrain, pinned datasets, status guards,
group-split defaults, pre-flight, governed LLM leak reasoning, and dataset-registration checks.

Everything runs the real stages in-process (InlineOrchestrator) over the in-memory store.
"""

import json

import pytest

from backend.config import Settings
from backend.errors import ConflictError, UpstreamError, ValidationError
from backend.governance import GovernancePolicy
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import (
    AnalysisService,
    DatasetService,
    ExperimentService,
    LeakageService,
    ReasoningService,
)
from ml_engine.contracts.common import ExperimentStatus, ModelRunStatus
from ml_engine.contracts.comparison import ComparisonReport
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.experiment import WorkflowState
from ml_engine.contracts.leakage_investigation import InvestigationPhase
from ml_engine.io import ExperimentLayout, read_model, write_model
from tests.backend.conftest import IMAGE, ROLE
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.inline_orchestrator import InlineOrchestrator

FEATURES = ["tenure_months", "monthly_charges", "contract", "churn_copy"]


class FakeReasoning:
    """Returns a fixed leak judgement — including one about a column that does not exist."""

    model_id = "anthropic.test-model"

    def __init__(self, fail: bool = False) -> None:
        self.prompts: list[str] = []
        self.fail = fail

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        from ml_engine.reasoning import ReasoningUnavailableError

        self.prompts.append(user_prompt)
        if self.fail:
            raise ReasoningUnavailableError("throttled")
        return json.dumps(
            {
                "executive_summary": "churn_copy mirrors the target.",
                "semantic_leakage": [
                    {
                        "feature": "churn_copy",
                        "severity": "critical",
                        "confidence": "high",
                        "reasoning": "It is recorded when the customer has already churned.",
                        "recommended_action": "strongly_consider_excluding",
                        "evidence_basis": ["target_association 1.0", "name"],
                    },
                    {
                        "feature": "invented_column",
                        "severity": "high",
                        "confidence": "low",
                        "reasoning": "made up",
                        "recommended_action": "review",
                        "evidence_basis": [],
                    },
                ],
            }
        )


@pytest.fixture
def settings(price_table_path) -> Settings:
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn=ROLE,
        stage_image_uri=IMAGE,
        price_table_path=price_table_path,
        bedrock_enabled=True,
        bedrock_model_id="anthropic.test-model",
    )


@pytest.fixture
def services(settings, store):
    repository = ExperimentRepository(settings=settings, store=store)
    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    client = FakeReasoning()
    reasoning = ReasoningService(
        settings=settings,
        repository=repository,
        store=store,
        client=client,
        governance=GovernancePolicy.from_settings(settings),
    )
    leakage = LeakageService(
        settings=settings, repository=repository, store=store, reasoning=reasoning
    )
    analysis = AnalysisService(
        settings=settings,
        repository=repository,
        store=store,
        orchestrator=orchestrator,
        experiments=experiments,
    )
    return {
        "experiments": experiments,
        "leakage": leakage,
        "analysis": analysis,
        "client": client,
        "orchestrator": orchestrator,
    }


@pytest.fixture
def experiment(services, store, classification_frame):
    uri = write_dataset(store, "churn.csv", classification_frame)
    record = services["experiments"].create(name="t", dataset_uri=uri, target_column="churned")
    services["experiments"].save_features(record.experiment_id, selected_features=FEATURES)
    return record


def _layout(record) -> ExperimentLayout:
    return ExperimentLayout(base=record.artifact_prefix)


# --- dataset pinning ---------------------------------------------------------------------------


def test_the_dataset_is_pinned_inside_the_experiment(services, store, experiment):
    layout = _layout(experiment)
    assert experiment.dataset.uri == layout.input_dataset("churn.csv")
    assert experiment.dataset.source_uri == f"{APPROVED_PREFIX}/churn.csv"
    pinned = store.read_bytes(experiment.dataset.uri)

    store.write_bytes(f"{APPROVED_PREFIX}/churn.csv", b"overwritten,later\n1,2\n")

    assert store.read_bytes(experiment.dataset.uri) == pinned  # training reads the pinned copy


# --- retrain archives the previous run ---------------------------------------------------------


def test_a_model_that_fails_on_retrain_is_not_reported_with_old_metrics(
    services, store, experiment
):
    experiments = services["experiments"]
    exp_id = experiment.experiment_id
    experiments.start_training(
        exp_id, TrainingRequest(models=["logistic_regression", "random_forest"])
    )
    first = experiments.comparison(exp_id)
    assert first.succeeded_count == 2

    # Run 2: random_forest is given an impossible parameter and fails.
    experiments.start_training(
        exp_id,
        TrainingRequest(
            models=["logistic_regression", "random_forest"],
            model_parameters={"random_forest": {"n_estimators": -5}},
        ),
    )

    status = {m.model_name: m.status for m in experiments.training_status(exp_id).models}
    assert status["random_forest"] == ModelRunStatus.FAILED.value
    second = experiments.comparison(exp_id)
    assert second.failed_count == 1 and second.succeeded_count == 1

    (run,) = experiments.list_runs(exp_id)
    assert run.best_model in {"logistic_regression", "random_forest"}
    archived = read_model(
        store,
        f"{_layout(experiment).run_archive(run.run_id)}/comparison/comparison.json",
        ComparisonReport,
    )
    assert archived.succeeded_count == 2  # the earlier result is kept, not lost


def test_rerun_reuses_the_frozen_config_unchanged(services, store, experiment):
    experiments = services["experiments"]
    exp_id = experiment.experiment_id
    experiments.start_training(exp_id, TrainingRequest(models=["logistic_regression"]))
    layout = _layout(experiment)
    before = read_model(store, layout.experiment_config, ExperimentConfig)
    first = experiments.comparison(exp_id).best_score

    experiments.rerun_training(exp_id)

    assert read_model(store, layout.experiment_config, ExperimentConfig) == before
    assert experiments.comparison(exp_id).best_score == pytest.approx(first, abs=1e-6)
    assert len(experiments.list_runs(exp_id)) == 1


# --- status guards -----------------------------------------------------------------------------


def test_the_selection_cannot_change_while_training_runs(services, store, experiment):
    layout = _layout(experiment)
    from datetime import UTC, datetime

    write_model(
        store,
        layout.workflow_state,
        WorkflowState(
            experiment_id=experiment.experiment_id,
            updated_at=datetime.now(UTC),
            status=ExperimentStatus.TRAINING,
        ),
    )
    with pytest.raises(ConflictError):
        services["experiments"].save_features(
            experiment.experiment_id, selected_features=["tenure_months"]
        )
    with pytest.raises(ConflictError):
        services["leakage"].apply_exclusions_to_selection(experiment.experiment_id)
    # The status still belongs to the workflow, so a second run cannot be started over it.
    assert services["experiments"].get(experiment.experiment_id).status is ExperimentStatus.TRAINING


# --- split defaults ----------------------------------------------------------------------------


def test_an_entity_id_in_the_business_context_makes_the_split_grouped(services, store, experiment):
    exp_id = experiment.experiment_id
    services["leakage"].set_business_context(
        exp_id,
        objective="predict churn",
        prediction_scenario="at month end, using what is known then",
        entity_id_column="customer_id",
    )
    services["experiments"].start_training(exp_id, TrainingRequest(models=["logistic_regression"]))
    config = read_model(store, _layout(experiment).experiment_config, ExperimentConfig)
    assert config.split.strategy == "group" and config.split.group_column == "customer_id"


def test_an_unknown_split_column_is_refused(services, experiment):
    with pytest.raises(ValidationError):
        services["experiments"].start_training(
            experiment.experiment_id,
            TrainingRequest(models=["logistic_regression"], time_column="no_such_column"),
        )


# --- leak reasoning, governed ------------------------------------------------------------------


def test_pre_training_investigation_attaches_governed_llm_suspicion(services, experiment):
    exp_id = experiment.experiment_id
    services["leakage"].set_business_context(
        exp_id, objective="predict churn", prediction_scenario="at contract renewal"
    )

    report = services["leakage"].build_pre_training_phase(exp_id)

    assert "llm_reasoning:used" in report.sources
    noted = [w for w in report.warnings if w.feature == "churn_copy" and w.llm_reasoning]
    assert noted and "LLM suspicion, high confidence" in noted[0].llm_reasoning
    # The LLM note never establishes leakage; only structural proof (an exact duplicate of the
    # target, a deterministic rule) may.
    assert all(
        w.leakage_established is False for w in noted if w.method.value != "deterministic_rule"
    )
    assert not any(w.feature == "invented_column" for w in report.warnings)  # hallucination
    prompt = services["client"].prompts[-1]
    assert "at contract renewal" in prompt  # the business context reaches the reasoning
    assert "C00001" not in prompt  # no row values


def test_bedrock_failure_still_builds_the_deterministic_phase(services, experiment):
    services["client"].fail = True
    report = services["leakage"].build_pre_training_phase(experiment.experiment_id)
    assert "llm_reasoning:unavailable" in report.sources
    assert report.warnings  # deterministic warnings are still there


def test_interpretation_errors_are_structured(services, experiment):
    services["client"].fail = True
    with pytest.raises(UpstreamError):
        services["leakage"]._reasoning.interpret(experiment.experiment_id)


# --- post-training: permutation + ablation feed the phase --------------------------------------


def test_post_training_uses_permutation_and_ablation(services, store, experiment):
    exp_id = experiment.experiment_id
    services["experiments"].start_training(exp_id, TrainingRequest(models=["logistic_regression"]))
    started = services["analysis"].start_ablation(exp_id, features=["churn_copy", "contract"])
    assert started["model_name"] == "logistic_regression"
    assert services["analysis"].ablation(exp_id)["status"] == "completed"

    report = services["leakage"].build_post_training_phase(exp_id)

    methods = {(w.feature, w.method.value) for w in report.warnings}
    assert ("churn_copy", "permutation_importance") in methods
    assert ("churn_copy", "ablation") in methods
    assert report.phase is InvestigationPhase.POST_TRAINING


# --- pre-flight --------------------------------------------------------------------------------


def test_preflight_lists_unresolved_high_severity_warnings(services, experiment):
    exp_id = experiment.experiment_id
    services["leakage"].build_pre_training_phase(exp_id, use_llm=False)

    check = services["analysis"].preflight(exp_id)

    assert check.ready  # advisory: nothing blocks training here
    assert any(w["feature"] == "churn_copy" for w in check.unresolved_leakage_warnings)
    assert any("business context" in w for w in check.warnings)


def test_delete_stops_running_work(services, experiment):
    services["experiments"].delete(experiment.experiment_id)  # inline: nothing left running
    assert services["experiments"].get(experiment.experiment_id).deleted


# --- dataset registration ----------------------------------------------------------------------


@pytest.fixture
def datasets(settings, store):
    return DatasetService(settings=settings, store=store)


def test_register_refuses_unapproved_locations(datasets, store):
    store.write_bytes("s3://someone-elses-bucket/secret.csv", b"a,b\n1,2\n")
    with pytest.raises(ValidationError):
        datasets.register("s3://someone-elses-bucket/secret.csv")


def test_a_semicolon_csv_is_refused_with_the_reason(datasets, store):
    store.write_bytes(f"{APPROVED_PREFIX}/euro.csv", b"a;b;c\n1;2;3\n4;5;6\n")
    with pytest.raises(ValidationError) as error:
        datasets.register(f"{APPROVED_PREFIX}/euro.csv")
    assert "semicolon" in error.value.message


def test_a_non_utf8_csv_is_refused_with_the_reason(datasets, store):
    store.write_bytes(f"{APPROVED_PREFIX}/latin.csv", "name,city\nJosé,Málaga\n".encode("cp1252"))
    with pytest.raises(ValidationError) as error:
        datasets.register(f"{APPROVED_PREFIX}/latin.csv")
    assert "UTF-8" in error.value.message


def test_a_comma_csv_still_registers(datasets, store):
    store.write_bytes(f"{APPROVED_PREFIX}/ok.csv", b"a,b\n1,2\n3,4\n")
    registered = datasets.register(f"{APPROVED_PREFIX}/ok.csv")
    assert [c.name for c in registered.columns] == ["a", "b"]
