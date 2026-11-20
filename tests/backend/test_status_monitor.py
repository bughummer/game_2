"""The status monitor drives the workflow and records what SageMaker reports (requirement 6).

It writes an observation per in-flight job, calls the orchestrator to advance the workflow, and
diagnoses a job the moment it fails — all without any chat history, and without any cost/Spot
accounting (that is the MLOps team's domain).
"""

import pytest

from backend.tracking import StatusMonitor
from ml_engine.contracts.execution import ExecutionRecord, JobState
from ml_engine.contracts.experiment import WorkflowState
from ml_engine.contracts.tracking import FailureCategory, FailureDiagnosis, JobObservation
from ml_engine.contracts.tracking import NormalizedStatus as S
from ml_engine.io import ExperimentLayout, read_model, write_model
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.experiments import make_config, make_record
from tests.support.fake_sagemaker import FakeLogs

EXP = "exp-mon123"
DATASET = "s3://mlf-test-approved/curated/churn.csv"
MODELS = ["logistic_regression", "xgboost"]


@pytest.fixture
def layout():
    return ExperimentLayout.for_experiment(ARTIFACT_ROOT, EXP)


@pytest.fixture
def record(layout):
    return make_record(layout, EXP, DATASET)


@pytest.fixture
def frozen(store, layout):
    write_model(store, layout.experiment_config, make_config(layout, EXP, DATASET, MODELS))


@pytest.fixture
def logs():
    return FakeLogs()


@pytest.fixture
def monitor(orchestrator, store, sagemaker, logs):
    return StatusMonitor(orchestrator, store, sagemaker, logs_client=logs)


def _obs(store, layout, execution_id, job_name) -> JobObservation:
    return read_model(store, layout.observation(execution_id, job_name), JobObservation)


def test_monitor_records_an_observation_and_advances_eda(
    monitor, orchestrator, sagemaker, store, layout, record
):
    handle = orchestrator.start_eda(record)
    job = read_model(store, layout.execution(handle.execution_id), ExecutionRecord).jobs[0]
    sagemaker.set_status(job.job_name, "Completed")

    after = monitor.on_event(handle.execution_id, job.job_name)

    assert after.jobs[0].state is JobState.COMPLETED
    observation = _obs(store, layout, handle.execution_id, job.job_name)
    assert observation.normalized_status is S.COMPLETED
    assert observation.source == "event"
    assert read_model(store, layout.workflow_state, WorkflowState).status.value == "FEATURE_REVIEW"


def test_waiting_for_capacity_is_recorded_while_in_progress(
    monitor, orchestrator, sagemaker, store, layout, record
):
    handle = orchestrator.start_eda(record)
    execution = orchestrator.advance(handle.execution_id)
    job = execution.jobs[0]
    sagemaker.set_status(job.job_name, "InProgress", SecondaryStatus="Pending")

    monitor.poll(handle.execution_id)

    # Processing jobs have no secondary status; InProgress reads as Preparing.
    assert _obs(store, layout, handle.execution_id, job.job_name).normalized_status is S.PREPARING


def test_a_failed_job_is_diagnosed_with_bounded_redacted_logs(
    monitor, orchestrator, sagemaker, store, layout, record, logs, frozen
):
    handle = orchestrator.start_training(record)
    execution = orchestrator.advance(handle.execution_id)  # submit preparation
    preparation = execution.jobs_for("preparation")[0]
    sagemaker.set_status(
        preparation.job_name,
        "Failed",
        FailureReason="AccessDenied on s3:GetObject for user dan@corp.com",
    )
    logs.set_events(preparation.job_name, ["boto3 ... AccessDenied", "secret AKIAIOSFODNN7EXAMPLE"])

    monitor.on_event(handle.execution_id, preparation.job_name)

    diagnosis = read_model(
        store, layout.diagnosis(handle.execution_id, preparation.job_name), FailureDiagnosis
    )
    assert diagnosis.category is FailureCategory.S3_ACCESS
    assert all("dan@corp.com" not in line for line in diagnosis.evidence + diagnosis.log_excerpt)
    assert all("AKIAIOSFODNN7EXAMPLE" not in line for line in diagnosis.log_excerpt)
    observation = _obs(store, layout, handle.execution_id, preparation.job_name)
    assert observation.normalized_status is S.FAILED


def test_a_diagnosis_is_written_once_not_on_every_poll(
    monitor, orchestrator, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)
    execution = orchestrator.advance(handle.execution_id)
    preparation = execution.jobs_for("preparation")[0]
    sagemaker.set_status(preparation.job_name, "Failed", FailureReason="AlgorithmError")

    monitor.on_event(handle.execution_id, preparation.job_name)
    first = read_model(
        store, layout.diagnosis(handle.execution_id, preparation.job_name), FailureDiagnosis
    ).diagnosed_at
    monitor.poll(handle.execution_id)
    second = read_model(
        store, layout.diagnosis(handle.execution_id, preparation.job_name), FailureDiagnosis
    ).diagnosed_at

    assert first == second  # not re-diagnosed: the job was already FAILED before the second tick


def test_monitor_drives_a_full_training_run_to_completion(
    monitor, orchestrator, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)

    def complete(stage):
        execution = read_model(store, layout.execution(handle.execution_id), ExecutionRecord)
        for job in execution.jobs_for(stage):
            sagemaker.set_status(job.job_name, "Completed")

    complete("preparation")
    monitor.poll(handle.execution_id)  # preparation done → training submitted
    complete("training")
    monitor.poll(handle.execution_id)  # training done → evaluation submitted
    complete("evaluation")
    after = monitor.poll(handle.execution_id)

    assert after.status.value == "SUCCEEDED"
    observations = store.list_uris(layout.observations_prefix(handle.execution_id))
    assert len(observations) == len(after.jobs)  # every job has a recorded observation


def test_missing_execution_is_handled(monitor):
    assert monitor.poll("exp-nope.eda.0a1b2c3d") is None


def test_no_cost_or_spot_artifacts_are_written(
    monitor, orchestrator, sagemaker, store, layout, record, frozen
):
    """Phase 4 records status and diagnosis only; no spend ledger (enforcement off)."""
    orchestrator.start_training(record)
    assert store.list_uris(layout.spend_prefix) == []
