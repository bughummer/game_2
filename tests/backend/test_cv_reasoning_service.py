"""The reasoning service in CV_Mode (CV design C7, Req 11.4, 11.5, D14).

Over the inline orchestrator and the in-memory store: a CV experiment's governed context carries
the ``cross_validation`` block, and the LLM narrative is filtered so a fabricated metric value is
dropped while a correctly cited number and number-free prose survive.
"""

import json

import pytest

from backend.config import Settings
from backend.governance import GovernancePolicy
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import ExperimentService, ReasoningService
from ml_engine.contracts.comparison import ExperimentSummary
from ml_engine.contracts.eda import EdaReport
from ml_engine.io import ExperimentLayout, read_model_if_exists
from tests.backend.conftest import IMAGE, ROLE
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.inline_orchestrator import InlineOrchestrator

FEATURES = ["tenure_months", "monthly_charges", "contract"]


class NarrativeClient:
    """Returns a narrative that cites one correct and one fabricated number."""

    model_id = "anthropic.test-model"

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def invoke(self, system_prompt: str, user_prompt: str) -> str:
        self.prompts.append(user_prompt)
        return json.dumps(
            {
                "executive_summary": (
                    "The models were compared with cross-validation. "
                    "The best run scored 0.99999 on every fold. "
                    "Cross-validation keeps the estimate honest."
                ),
                "model_behaviour_narrative": "Performance was consistent across the folds.",
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
def harness(settings, store, classification_frame):
    repository = ExperimentRepository(settings=settings, store=store)
    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    client = NarrativeClient()
    reasoning = ReasoningService(
        settings=settings,
        repository=repository,
        store=store,
        client=client,
        governance=GovernancePolicy.from_settings(settings),
    )
    uri = write_dataset(store, "churn.csv", classification_frame)
    record = experiments.create(name="t", dataset_uri=uri, target_column="churned")
    experiments.save_features(record.experiment_id, selected_features=FEATURES)
    experiments.start_training(
        record.experiment_id,
        TrainingRequest(models=["logistic_regression"], evaluation_mode="cv", cv_folds=3),
    )
    return {
        "experiments": experiments,
        "reasoning": reasoning,
        "client": client,
        "store": store,
        "record": record,
    }


def _layout(record) -> ExperimentLayout:
    return ExperimentLayout(base=record.artifact_prefix)


def test_governed_context_carries_cross_validation(harness):
    layout = _layout(harness["record"])
    store = harness["store"]
    reasoning = harness["reasoning"]
    eda = read_model_if_exists(store, layout.eda, EdaReport)
    summary = read_model_if_exists(store, layout.summary, ExperimentSummary)

    context = reasoning.governed_context(layout, eda=eda, summary=summary)

    assert "cross_validation" in context
    cv = context["cross_validation"]
    assert cv["folds"] == 3 and cv["scheme"]
    assert cv["runs"] and cv["runs"][0]["model"] == "logistic_regression"


def test_fabricated_number_is_filtered_from_the_stored_narrative(harness):
    report = harness["reasoning"].interpret(harness["record"].experiment_id)

    # The fabricated 0.99999 citation is gone; number-free prose survives.
    assert "0.99999" not in report.executive_summary
    assert "Cross-validation keeps the estimate honest." in report.executive_summary
    assert "The models were compared with cross-validation." in report.executive_summary
    assert report.model_behaviour_narrative == "Performance was consistent across the folds."

    # And the stored artifact is the filtered one.
    layout = _layout(harness["record"])
    stored = read_model_if_exists(harness["store"], layout.reasoning, type(report))
    assert stored is not None and "0.99999" not in stored.executive_summary
