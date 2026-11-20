"""Data-context Q&A gate + AI drafting (free-form-experiments design C5, Property 8)."""

import pytest

from backend.config import Settings
from backend.errors import ConflictError
from backend.repositories import ExperimentRepository
from backend.services import ExperimentService, FreeFormService
from ml_engine.contracts.free_form import (
    DATA_CONTEXT_QUESTIONS,
    DataContextAnswer,
    DataContextAnswers,
    DraftSession,
)
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

TARGET = "churned"


def _setup(store, client=None):
    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf-sagemaker-job",
    )
    repository = ExperimentRepository(settings=settings, store=store)
    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    free_form = FreeFormService(
        settings=settings,
        repository=repository,
        store=store,
        orchestrator=orchestrator,
        experiments=experiments,
        client=client,
    )
    return experiments, free_form


def _ready(experiments, store):
    uri = write_dataset(store, "data.csv", golden_frame())
    return experiments.create(name="ff", dataset_uri=uri, target_column=TARGET), uri


def _answers() -> DataContextAnswers:
    return DataContextAnswers(
        answers=[
            DataContextAnswer(
                question_id=q.question_id, question=q.question, status="answered", answer="yes"
            )
            for q in DATA_CONTEXT_QUESTIONS
        ]
    )


def test_data_context_questions_are_offered():
    store = InMemoryObjectStore()
    experiments, free_form = _setup(store)
    record, uri = _ready(experiments, store)
    result = free_form.request_data_context(record.experiment_id, uri)
    assert len(result["questions"]) == len(DATA_CONTEXT_QUESTIONS)
    assert result["defaults"] is None


def test_drafting_is_refused_without_data_context():
    store = InMemoryObjectStore()
    experiments, free_form = _setup(store)
    record, uri = _ready(experiments, store)
    with pytest.raises(ConflictError) as excinfo:
        free_form.draft_script(
            record.experiment_id,
            dataset_uri=uri,
            description="try a new method",
            data_context=DataContextAnswers(),
        )
    assert "DATA_CONTEXT_REQUIRED" in str(excinfo.value) or "data-context" in str(excinfo.value)


def test_draft_is_produced_and_answers_stored_and_defaulted():
    store = InMemoryObjectStore()
    experiments, free_form = _setup(store)
    record, uri = _ready(experiments, store)
    result = free_form.draft_script(
        record.experiment_id,
        dataset_uri=uri,
        description="forecast monthly charges",
        data_context=_answers(),
    )
    assert result["revision"]["revision"] == 1
    assert result["revision"]["authored_by"] == "ai"
    # The answers are stored on the draft session and saved as dataset defaults.
    layout = ExperimentLayout(base=record.artifact_prefix)
    session = read_model(store, layout.free_form_draft(result["draft_id"]), DraftSession)
    assert session.data_context.is_complete
    defaults = free_form.request_data_context(record.experiment_id, uri)["defaults"]
    assert defaults is not None


def test_draft_then_run_end_to_end():
    store = InMemoryObjectStore()
    experiments, free_form = _setup(store)
    record, uri = _ready(experiments, store)
    # Draft (gated on data context), then approve+run the drafted script's content.
    draft = free_form.draft_script(
        record.experiment_id,
        dataset_uri=uri,
        description="baseline",
        data_context=_answers(),
    )
    script = draft["revision"]["content"]
    status = free_form.start_run(record.experiment_id, script=script, dataset_uri=uri)
    assert status["status"] == "completed"
    assert status["result"]["tier"] == "free_form"


def test_revisions_are_retained():
    store = InMemoryObjectStore()
    experiments, free_form = _setup(store)
    record, uri = _ready(experiments, store)
    first = free_form.draft_script(
        record.experiment_id, dataset_uri=uri, description="v1", data_context=_answers()
    )
    second = free_form.draft_script(
        record.experiment_id,
        dataset_uri=uri,
        description="v1",
        data_context=_answers(),
        draft_id=first["draft_id"],
        instruction="use a rolling window of 14",
    )
    assert second["revision_count"] == 2
    assert second["revision"]["instruction"] == "use a rolling window of 14"
