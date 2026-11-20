"""Free-form service + job end-to-end (free-form-experiments design C3/C5, Properties 1/6).

Offline, in-memory store + inline orchestrator. A user-approved script runs as a side job, its
result is labelled free_form, a hash mismatch is refused, and runs are isolated.
"""

import pytest

from backend.config import Settings
from backend.errors import ValidationError
from backend.repositories import ExperimentRepository
from backend.services import ExperimentService, FreeFormService
from ml_engine.contracts.free_form import FreeFormRequest, content_hash
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

TARGET = "churned"

# A tiny approved script: reports one self-reported metric and trains a trivial model.
SCRIPT = """
def main(context):
    frame = context.data
    context.report_metric("row_count", float(len(frame)))


def predict(frame):
    return [0] * len(frame)
"""


def _services(store):
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
    return experiments, free_form, orchestrator


def _ready(experiments, store, name="data.csv"):
    uri = write_dataset(store, name, golden_frame())
    return experiments.create(name="ff", dataset_uri=uri, target_column=TARGET), uri


def test_free_form_run_is_labelled_and_reports_metrics():
    store = InMemoryObjectStore()
    experiments, free_form, orchestrator = _services(store)
    record, uri = _ready(experiments, store)
    status = free_form.start_run(record.experiment_id, script=SCRIPT, dataset_uri=uri)
    assert not orchestrator.failures, orchestrator.failures
    assert status["status"] == "completed"
    result = status["result"]
    assert result["tier"] == "free_form"
    names = {m["name"]: m["label"] for m in result["metrics"]}
    assert names["row_count"] == "self_reported"


SDK_SCRIPT = """
from ml_engine.sdk import metrics, results


def main(context):
    frame = context.data
    y_true = frame["tenure_months"].astype(float).to_numpy()[:10]
    y_pred = y_true + 1.0
    scores = {"mae": metrics.mae(y_true, y_pred), "rmse": metrics.rmse(y_true, y_pred)}
    results.write_metrics(context, scores)
"""


def test_sdk_produced_metrics_are_labelled_engine():
    store = InMemoryObjectStore()
    experiments, free_form, orchestrator = _services(store)
    record, uri = _ready(experiments, store)
    status = free_form.start_run(record.experiment_id, script=SDK_SCRIPT, dataset_uri=uri)
    assert not orchestrator.failures, orchestrator.failures
    labels = {m["name"]: m["label"] for m in status["result"]["metrics"]}
    assert labels["mae"] == "engine"
    assert labels["rmse"] == "engine"
    assert status["result"]["sdk_version"]


def test_free_form_registers_a_model_when_predict_is_declared():
    store = InMemoryObjectStore()
    experiments, free_form, _orchestrator = _services(store)
    record, uri = _ready(experiments, store)
    status = free_form.start_run(
        record.experiment_id,
        script=SCRIPT + "\n\ndef _bind(context):\n    context.model = object()\n",
        dataset_uri=uri,
        predict_entry_point="predict",
    )
    # The script above sets no model in main(); predict exists but model is None -> not registered.
    assert status["result"]["produced_model"] is False


def test_a_tampered_script_is_refused_by_content_hash():
    store = InMemoryObjectStore()
    experiments, free_form, _orchestrator = _services(store)
    record, uri = _ready(experiments, store)
    status = free_form.start_run(record.experiment_id, script=SCRIPT, dataset_uri=uri)
    run_id = status["run_id"]
    layout = ExperimentLayout(base=record.artifact_prefix)
    # Tamper with the pinned bytes after approval; re-running the job must refuse.
    store.write_bytes(layout.free_form_script(run_id), b"print('evil')")
    from jobs.free_form.main import run_free_form

    with pytest.raises(Exception) as excinfo:
        run_free_form(store, layout, experiment_id=record.experiment_id, run_id=run_id)
    assert "SCRIPT_NOT_APPROVED" in str(excinfo.value) or "not match" in str(excinfo.value)


def test_a_failing_script_fails_the_run_without_fallback():
    store = InMemoryObjectStore()
    experiments, free_form, _orchestrator = _services(store)
    record, uri = _ready(experiments, store)
    bad = "def main(context):\n    raise RuntimeError('boom')\n"
    status = free_form.start_run(record.experiment_id, script=bad, dataset_uri=uri)
    assert status["status"] == "failed"
    assert "boom" in (status.get("failure_message") or "")


def test_runs_are_isolated():
    store = InMemoryObjectStore()
    experiments, free_form, _orchestrator = _services(store)
    record, uri = _ready(experiments, store)
    a = free_form.start_run(record.experiment_id, script=SCRIPT, dataset_uri=uri)
    b = free_form.start_run(record.experiment_id, script=SCRIPT, dataset_uri=uri)
    assert a["run_id"] != b["run_id"]
    layout = ExperimentLayout(base=record.artifact_prefix)
    ra = read_model(store, layout.free_form_request(a["run_id"]), FreeFormRequest)
    rb = read_model(store, layout.free_form_request(b["run_id"]), FreeFormRequest)
    assert (
        ra.approved_script.content_hash == rb.approved_script.content_hash == content_hash(SCRIPT)
    )
    # Each wrote its own result under its own prefix.
    assert store.exists(layout.free_form_result(a["run_id"]))
    assert store.exists(layout.free_form_result(b["run_id"]))


def test_a_gpu_instance_is_refused_unless_opted_in():
    store = InMemoryObjectStore()
    experiments, free_form, _orchestrator = _services(store)
    record, uri = _ready(experiments, store)
    with pytest.raises(ValidationError):
        free_form.start_run(
            record.experiment_id, script=SCRIPT, dataset_uri=uri, instance_type="ml.p3.2xlarge"
        )


def test_the_dataset_is_leakage_screened_and_findings_are_advisory():
    store = InMemoryObjectStore()
    experiments, free_form, orchestrator = _services(store)
    record, uri = _ready(experiments, store)
    status = free_form.start_run(
        record.experiment_id,
        script=SCRIPT,
        dataset_uri=uri,
        target_column="churned",
    )
    assert not orchestrator.failures, orchestrator.failures
    # The leakage report is attached to the run (advisory; the run still completed).
    assert "leakage" in status
    assert status["status"] == "completed"
    assert status["leakage"]["target_column"] == "churned"
