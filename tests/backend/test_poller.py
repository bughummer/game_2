"""The built-in poller is what moves a workflow from stage to stage.

Without it nothing calls ``StatusMonitor.poll`` in production, so these tests drive whole
workflows through ``ExecutionPoller.tick`` only — never by calling ``advance`` directly.
"""

import pytest

from backend.tracking import ExecutionPoller, StatusMonitor
from ml_engine.contracts.common import ExperimentStatus
from ml_engine.contracts.execution import ExecutionRecord, ExecutionStatus
from ml_engine.contracts.experiment import WorkflowState
from ml_engine.io import ExperimentLayout, read_model, write_model
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.experiments import make_config, make_record

EXP = "exp-poll123"
DATASET = "s3://mlf-test-approved/curated/churn.csv"
MODELS = ["logistic_regression", "xgboost"]


@pytest.fixture
def layout():
    return ExperimentLayout.for_experiment(ARTIFACT_ROOT, EXP)


@pytest.fixture
def record(layout):
    return make_record(layout, EXP, DATASET)


@pytest.fixture
def monitor(orchestrator, store, sagemaker):
    return StatusMonitor(orchestrator, store, sagemaker)


@pytest.fixture
def poller(monitor, store, orchestrator):
    poller = ExecutionPoller(monitor, store, ARTIFACT_ROOT)
    orchestrator.on_started(poller.register)
    return poller


def _execution(store, layout, execution_id) -> ExecutionRecord:
    return read_model(store, layout.execution(execution_id), ExecutionRecord)


def _workflow(store, layout) -> WorkflowState:
    return read_model(store, layout.workflow_state, WorkflowState)


def _complete(sagemaker, execution, stage):
    for job in execution.jobs_for(stage):
        sagemaker.set_status(job.job_name, "Completed")


def test_a_started_workflow_is_registered_immediately(poller, orchestrator, record):
    handle = orchestrator.start_eda(record)

    assert poller.active == {handle.execution_id}


def test_polling_advances_eda_to_feature_review_and_then_forgets_it(
    poller, orchestrator, sagemaker, store, layout, record
):
    handle = orchestrator.start_eda(record)
    poller.tick()
    assert _workflow(store, layout).status is ExperimentStatus.EDA_RUNNING

    _complete(sagemaker, _execution(store, layout, handle.execution_id), "profiling")
    poller.tick()

    assert _workflow(store, layout).status is ExperimentStatus.FEATURE_REVIEW
    assert poller.active == set()


def test_polling_alone_drives_training_through_every_stage(
    poller, orchestrator, sagemaker, store, layout, record
):
    write_model(store, layout.experiment_config, make_config(layout, EXP, DATASET, MODELS))
    handle = orchestrator.start_training(record)
    execution_id = handle.execution_id

    _complete(sagemaker, _execution(store, layout, execution_id), "preparation")
    poller.tick()
    assert _workflow(store, layout).status is ExperimentStatus.TRAINING
    assert len(sagemaker.names("create_training_job")) == len(MODELS)

    _complete(sagemaker, _execution(store, layout, execution_id), "training")
    poller.tick()
    assert _workflow(store, layout).status is ExperimentStatus.EVALUATING

    _complete(sagemaker, _execution(store, layout, execution_id), "evaluation")
    poller.tick()
    assert _execution(store, layout, execution_id).status is ExecutionStatus.SUCCEEDED
    assert poller.active == set()


def test_discovery_finds_running_work_after_a_restart(
    orchestrator, monitor, sagemaker, store, layout, record
):
    """A fresh poller (new process) picks up executions started before it existed."""
    running = orchestrator.start_eda(record)
    other_layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, "exp-done456")
    finished = orchestrator.start_eda(make_record(other_layout, "exp-done456", DATASET))
    _complete(sagemaker, _execution(store, other_layout, finished.execution_id), "profiling")
    orchestrator.advance(finished.execution_id)

    fresh = ExecutionPoller(monitor, store, ARTIFACT_ROOT)
    assert fresh.discover() == 1
    assert fresh.active == {running.execution_id}


def test_discovery_ignores_observation_files(poller, orchestrator, store, layout, record):
    handle = orchestrator.start_eda(record)
    poller.tick()  # writes an observation under executions/<id>/observations/

    fresh = ExecutionPoller(poller._monitor, store, ARTIFACT_ROOT)
    assert fresh.discover() == 1
    assert fresh.active == {handle.execution_id}


def test_one_failing_execution_does_not_stop_the_others(store):
    class Monitor:
        def __init__(self):
            self.calls = []

        def poll(self, execution_id):
            self.calls.append(execution_id)
            if execution_id == "bad":
                raise RuntimeError("throttled")
            return None

    monitor = Monitor()
    poller = ExecutionPoller(monitor, store, ARTIFACT_ROOT)
    poller.register("bad")
    poller.register("good")

    assert poller.tick() == 1
    assert sorted(monitor.calls) == ["bad", "good"]
    assert poller.active == {"bad"}  # kept for retry; "good" finished (poll returned None)


def test_the_background_thread_polls_and_stops(monitor, orchestrator, store, record):
    poller = ExecutionPoller(monitor, store, ARTIFACT_ROOT, interval_seconds=0.01)
    orchestrator.on_started(poller.register)
    orchestrator.start_eda(record)

    thread = poller.start()
    poller.stop(timeout=2)

    assert not thread.is_alive()
