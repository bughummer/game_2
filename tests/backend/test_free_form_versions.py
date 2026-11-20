"""Free-form model-version enumeration + validation availability (free-form design C7, Property 5)."""

import pytest

from backend.config import Settings
from backend.errors import ConflictError
from backend.repositories import ExperimentRepository
from backend.services import ExperimentService, FreeFormService, ModelVersionService
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

TARGET = "churned"

# A script that trains a trivial model and exposes a predict entry point.
MODEL_SCRIPT = """
class _Const:
    def predict(self, frame):
        return [0] * len(frame)


def main(context):
    model = _Const()
    context.model = model
    context.predict = model.predict
    context.report_metric("trained", 1.0)


def predict(frame):
    return [0] * len(frame)
"""

NO_MODEL_SCRIPT = """
def main(context):
    context.report_metric("done", 1.0)
"""


def _setup(store):
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
    )
    model_versions = ModelVersionService(settings=settings, repository=repository, store=store)
    return experiments, free_form, model_versions, orchestrator


def _ready(experiments, store):
    uri = write_dataset(store, "data.csv", golden_frame())
    return experiments.create(name="ff", dataset_uri=uri, target_column=TARGET), uri


def test_a_free_form_model_is_enumerated_as_a_version():
    store = InMemoryObjectStore()
    experiments, free_form, model_versions, orchestrator = _setup(store)
    record, uri = _ready(experiments, store)
    status = free_form.start_run(
        record.experiment_id,
        script=MODEL_SCRIPT,
        dataset_uri=uri,
        predict_entry_point="predict",
    )
    assert not orchestrator.failures, orchestrator.failures
    assert status["result"]["produced_model"] is True
    versions = model_versions.list_versions(record.experiment_id)
    free_form_versions = [v for v in versions if v.source == "free_form"]
    assert len(free_form_versions) == 1
    assert free_form_versions[0].version_id.startswith("free_form:")


def test_validation_is_unavailable_for_a_run_without_a_model():
    store = InMemoryObjectStore()
    experiments, free_form, _mv, _orchestrator = _setup(store)
    record, uri = _ready(experiments, store)
    status = free_form.start_run(record.experiment_id, script=NO_MODEL_SCRIPT, dataset_uri=uri)
    assert status["result"]["produced_model"] is False
    with pytest.raises(ConflictError) as excinfo:
        free_form.validate_run_model(
            record.experiment_id, status["run_id"], dataset_uri=uri, validation=None
        )
    assert "unavailable" in str(excinfo.value).lower()
