"""SageMakerOrchestrator against a fake SageMaker that validates requests with botocore's model.

Covers: exact request shapes, Spot defaults, checkpoint config only for resumable models,
non-blocking start, stage chaining, per-model failure isolation, explicit-only On-Demand
fallback, guard refusals before any AWS call, idempotent resubmission, stop, and "unknown"
status handling. Design Properties 9 and 11.
"""

import re

import pytest

from backend.orchestration import GuardViolation, OrchestrationError, SageMakerOrchestrator
from backend.orchestration.sagemaker import job_name, parse_execution_id
from backend.pricing import PriceTable
from ml_engine.contracts.common import ExperimentStatus
from ml_engine.contracts.execution import (
    ExecutionRecord,
    ExecutionStatus,
    ExperimentCompute,
    JobState,
    SpendReservation,
)
from ml_engine.contracts.experiment import WorkflowState
from ml_engine.contracts.model import ModelFailure
from ml_engine.io import ExperimentLayout, read_model, write_model
from tests.backend.conftest import IMAGE, NOW, ROLE
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.experiments import cv_split, make_config, make_record
from tests.support.fake_sagemaker import ClientError

EXP = "exp-abc123"
DATASET = "s3://mlf-test-approved/curated/churn.csv"
MODELS = ["logistic_regression", "random_forest", "xgboost", "elastic_net"]
JOB_NAME = re.compile(r"^[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}$")


@pytest.fixture
def layout() -> ExperimentLayout:
    return ExperimentLayout.for_experiment(ARTIFACT_ROOT, EXP)


@pytest.fixture
def record(layout):
    return make_record(layout, EXP, DATASET)


@pytest.fixture
def frozen(store, layout):
    write_model(store, layout.experiment_config, make_config(layout, EXP, DATASET, MODELS))


def _execution(store, layout, handle) -> ExecutionRecord:
    return read_model(store, layout.execution(handle.execution_id), ExecutionRecord)


def _workflow(store, layout) -> WorkflowState:
    return read_model(store, layout.workflow_state, WorkflowState)


def _complete(sagemaker, execution, stage):
    for job in execution.jobs_for(stage):
        sagemaker.set_status(job.job_name, "Completed")


# --- EDA ---------------------------------------------------------------------------------------


def test_start_eda_submits_profiling_and_returns_at_once(
    orchestrator, sagemaker, store, layout, record
):
    handle = orchestrator.start_eda(record)

    assert handle.started and handle.execution_id.startswith(f"{EXP}.eda.")
    ((operation, request),) = sagemaker.calls  # one call, no polling: non-blocking (Property 9)
    assert operation == "create_processing_job"
    app = request["AppSpecification"]
    assert app["ImageUri"] == IMAGE
    assert app["ContainerEntrypoint"] == ["python", "-m", "jobs.profiling.main"]
    assert app["ContainerArguments"] == [
        "--experiment-id",
        EXP,
        "--artifact-base",
        layout.base,
        "--dataset-uri",
        DATASET,
        "--target-column",
        "churned",
        "--file-format",
        "csv",
    ]
    assert request["RoleArn"] == ROLE
    assert request["ProcessingResources"]["ClusterConfig"]["InstanceType"] == "ml.m5.large"
    assert request["Environment"]["ML_FACTORY_S3_SSE"] == "AES256"
    assert {"Key": "ml-factory:experiment", "Value": EXP} in request["Tags"]
    assert _workflow(store, layout).status is ExperimentStatus.EDA_RUNNING


def test_eda_completion_hands_the_experiment_to_feature_review(
    orchestrator, sagemaker, store, layout, record
):
    handle = orchestrator.start_eda(record)
    _complete(sagemaker, _execution(store, layout, handle), "profiling")

    execution = orchestrator.advance(handle.execution_id)

    assert execution.status is ExecutionStatus.SUCCEEDED
    assert _workflow(store, layout).status is ExperimentStatus.FEATURE_REVIEW


def test_eda_failure_is_recorded_with_aws_reason(orchestrator, sagemaker, store, layout, record):
    handle = orchestrator.start_eda(record)
    job = _execution(store, layout, handle).jobs[0]
    sagemaker.set_status(job.job_name, "Failed", FailureReason="AlgorithmError: exit code 1")

    execution = orchestrator.advance(handle.execution_id)

    assert execution.status is ExecutionStatus.FAILED
    state = _workflow(store, layout)
    assert state.status is ExperimentStatus.FAILED
    assert state.failure_message == "AlgorithmError: exit code 1"


# --- training: freezing, chaining, Spot --------------------------------------------------------


def test_training_needs_a_frozen_configuration(orchestrator, sagemaker, record):
    with pytest.raises(OrchestrationError) as error:
        orchestrator.start_training(record)
    assert error.value.code == "EXPERIMENT_CONFIG_MISSING"
    assert sagemaker.calls == []


def test_training_starts_with_preparation_only(
    orchestrator, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)

    assert sagemaker.names("create_processing_job") == [
        _execution(store, layout, handle).jobs[0].job_name
    ]
    assert sagemaker.names("create_training_job") == []
    execution = _execution(store, layout, handle)
    assert [j.stage for j in execution.jobs] == ["preparation", *["training"] * 4, "evaluation"]
    assert all(j.state is JobState.PENDING_SUBMISSION for j in execution.jobs[1:])
    assert _workflow(store, layout).status is ExperimentStatus.PREPARING


def test_models_train_in_parallel_on_spot_after_preparation(
    orchestrator, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")

    orchestrator.advance(handle.execution_id)

    jobs = _execution(store, layout, handle).jobs_for("training")
    assert sorted(sagemaker.names("create_training_job")) == sorted(j.job_name for j in jobs)
    assert {j.model_name for j in jobs} == set(MODELS)
    for job in jobs:
        request = sagemaker.created[job.job_name]
        args = request["AlgorithmSpecification"]["ContainerArguments"]
        assert args[args.index("--model-name") + 1] == job.model_name
        assert request["EnableManagedSpotTraining"] is True
        assert request["StoppingCondition"] == {
            "MaxRuntimeInSeconds": 3600,
            "MaxWaitTimeInSeconds": 3600,
        }
        assert request["AlgorithmSpecification"]["ContainerEntrypoint"] == [
            "python",
            "-m",
            "jobs.training.main",
        ]
        assert request["OutputDataConfig"]["S3OutputPath"] == layout.sagemaker_output
        assert request["ResourceConfig"]["InstanceType"] == "ml.m5.large"
        assert "RetryStrategy" not in request
    assert _workflow(store, layout).status is ExperimentStatus.TRAINING


def test_the_resumable_model_set_is_empty():
    """Stock-image migration (Req 5.1): no model resumes; the constant is kept but empty."""
    from ml_engine.contracts.execution import RESUMABLE_MODELS

    assert frozenset() == RESUMABLE_MODELS


def test_no_job_sets_a_checkpoint(orchestrator, sagemaker, store, layout, record, frozen):
    """Stock-image migration (Req 5.2): Spot checkpoint/resume is dropped, so no training job —
    for any model, even a would-be-resumable xgboost/catboost — sets a CheckpointConfig or gets
    a --checkpoint-dir argument. Managed Spot (the discount) is unchanged."""
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)

    execution = _execution(store, layout, handle)
    for job in execution.jobs_for("training"):
        request = sagemaker.created[job.job_name]
        args = request["AlgorithmSpecification"]["ContainerArguments"]
        assert "CheckpointConfig" not in request
        assert "--checkpoint-dir" not in args
        assert job.checkpoint_uri is None
        assert job.resumable is False
        # Managed Spot is untouched.
        assert request["EnableManagedSpotTraining"] is True


def test_training_records_the_resolved_image(
    orchestrator, sagemaker, store, layout, record, frozen
):
    """The execution records the resolver's output; here the configured override (Req 1.4).

    Every submitted job runs that image, and later stages (side jobs) reuse it.
    """
    handle = orchestrator.start_training(record)
    execution = _execution(store, layout, handle)

    assert execution.image_uri == IMAGE
    prep = execution.jobs_for("preparation")[0]
    assert sagemaker.created[prep.job_name]["AppSpecification"]["ImageUri"] == IMAGE

    _complete(sagemaker, execution, "preparation")
    orchestrator.advance(handle.execution_id)
    for job in _execution(store, layout, handle).jobs_for("training"):
        request = sagemaker.created[job.job_name]
        assert request["AlgorithmSpecification"]["TrainingImage"] == IMAGE
        assert request["Environment"]["ML_FACTORY_STAGE_IMAGE_URI"] == IMAGE


def test_an_unset_image_is_resolved_from_the_stock_lookup(
    sagemaker, store, record, frozen, price_table_path, monkeypatch
):
    """With no configured override, the orchestrator resolves the stock image and records it."""
    import sys
    import types

    from backend.config import Settings
    from tests.backend.conftest import NOW, ROLE

    resolved_uri = "123.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.7-1"
    fake = types.ModuleType("sagemaker")
    image_uris = types.ModuleType("sagemaker.image_uris")
    image_uris.retrieve = lambda **_kwargs: resolved_uri
    fake.image_uris = image_uris
    monkeypatch.setitem(sys.modules, "sagemaker", fake)
    monkeypatch.setitem(sys.modules, "sagemaker.image_uris", image_uris)

    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn=ROLE,
        price_table_path=price_table_path,
    )
    orchestrator = SageMakerOrchestrator(settings, store, sagemaker, None, clock=lambda: NOW)
    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, EXP)

    handle = orchestrator.start_training(record)
    execution = _execution(store, layout, handle)

    assert execution.image_uri == resolved_uri
    prep = execution.jobs_for("preparation")[0]
    assert sagemaker.created[prep.job_name]["AppSpecification"]["ImageUri"] == resolved_uri


def test_one_failed_model_does_not_stop_the_others(
    orchestrator, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)
    execution = _execution(store, layout, handle)
    for job in execution.jobs_for("training"):
        if job.model_name == "random_forest":
            sagemaker.set_status(job.job_name, "Failed", FailureReason="OutOfMemoryError")
        else:
            sagemaker.set_status(job.job_name, "Completed")

    execution = orchestrator.advance(handle.execution_id)

    assert execution.status is ExecutionStatus.RUNNING
    assert len(sagemaker.names("create_processing_job")) == 2  # preparation + evaluation
    failure = read_model(store, layout.model_failure("random_forest"), ModelFailure)
    assert failure.error_code == "SAGEMAKER_JOB_FAILED"
    assert failure.message == "OutOfMemoryError"
    assert _workflow(store, layout).status is ExperimentStatus.EVALUATING


def test_a_stage_written_failure_is_not_overwritten(
    orchestrator, sagemaker, store, layout, record, frozen
):
    """The stage's own failure.json is more specific than SageMaker's; keep it."""
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)
    from jobs.training.main import write_model_failure

    write_model_failure(
        store,
        layout,
        experiment_id=EXP,
        model_name="xgboost",
        code="MODEL_TRAINING_FAILED",
        message="specific",
    )
    for job in _execution(store, layout, handle).jobs_for("training"):
        sagemaker.set_status(job.job_name, "Failed" if job.model_name == "xgboost" else "Completed")

    orchestrator.advance(handle.execution_id)

    assert read_model(store, layout.model_failure("xgboost"), ModelFailure).message == "specific"


def test_evaluation_completion_finishes_the_execution(
    orchestrator, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)
    _complete(sagemaker, _execution(store, layout, handle), "training")
    orchestrator.advance(handle.execution_id)
    _complete(sagemaker, _execution(store, layout, handle), "evaluation")

    assert orchestrator.advance(handle.execution_id).status is ExecutionStatus.SUCCEEDED
    assert orchestrator.describe(handle.execution_id)["status"] == "SUCCEEDED"


def test_advance_is_idempotent(orchestrator, sagemaker, store, layout, record, frozen):
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)
    calls = len(sagemaker.calls)

    orchestrator.advance(handle.execution_id)
    orchestrator.advance(handle.execution_id)

    creates = [c for c in sagemaker.calls[calls:] if c[0].startswith("create_")]
    assert creates == []


def test_a_resubmitted_job_that_already_exists_is_not_an_error(
    orchestrator, sagemaker, store, layout, record
):
    handle = orchestrator.start_eda(record)
    execution = _execution(store, layout, handle)
    job = execution.jobs[0]
    job.state = JobState.PENDING_SUBMISSION  # as if the process died between create and record
    plan = orchestrator._plan_from(execution, job)
    orchestrator._submit(execution, layout, plan)  # SageMaker answers ResourceInUse

    assert _execution(store, layout, handle).jobs[0].state is JobState.IN_PROGRESS


# --- Spot capacity and the On-Demand opt-in ----------------------------------------------------


def _capacity_failure(orchestrator, sagemaker, store, layout, record):
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)
    for job in _execution(store, layout, handle).jobs_for("training"):
        if job.model_name == "xgboost":
            sagemaker.set_status(
                job.job_name,
                "Failed",
                SecondaryStatus="MaxWaitTimeExceeded",
                FailureReason="Spot capacity not available within MaxWaitTimeInSeconds",
            )
        else:
            sagemaker.set_status(job.job_name, "Completed")
    return handle, orchestrator.advance(handle.execution_id)


def test_spot_capacity_failure_fails_and_reports_by_default(
    orchestrator, sagemaker, store, layout, record, frozen
):
    _capacity_failure(orchestrator, sagemaker, store, layout, record)

    on_demand = [
        r
        for op, r in sagemaker.calls
        if op == "create_training_job" and not r["EnableManagedSpotTraining"]
    ]
    assert on_demand == []
    failure = read_model(store, layout.model_failure("xgboost"), ModelFailure)
    assert "MaxWaitTimeExceeded" in failure.message


def test_on_demand_fallback_happens_only_when_opted_in(
    orchestrator, sagemaker, store, layout, record, frozen
):
    write_model(
        store, layout.compute, ExperimentCompute(experiment_id=EXP, allow_on_demand_fallback=True)
    )

    handle, execution = _capacity_failure(orchestrator, sagemaker, store, layout, record)

    fallback = [j for j in execution.jobs if j.fallback_of]
    assert len(fallback) == 1
    request = sagemaker.created[fallback[0].job_name]
    assert request["EnableManagedSpotTraining"] is False
    assert "MaxWaitTimeInSeconds" not in request["StoppingCondition"]
    assert "CheckpointConfig" not in request
    assert fallback[0].job_name.endswith("-od")
    # and never twice
    sagemaker.set_status(
        fallback[0].job_name, "Failed", FailureReason="InsufficientInstanceCapacity"
    )
    execution = orchestrator.advance(handle.execution_id)
    assert len([j for j in execution.jobs if j.fallback_of]) == 1


def test_a_successful_fallback_leaves_no_failure_for_the_model(
    orchestrator, sagemaker, store, layout, record, frozen
):
    """The replaced Spot job must not be recorded as the model's failure."""
    write_model(
        store, layout.compute, ExperimentCompute(experiment_id=EXP, allow_on_demand_fallback=True)
    )
    handle, execution = _capacity_failure(orchestrator, sagemaker, store, layout, record)
    (fallback,) = [j for j in execution.jobs if j.fallback_of]

    orchestrator.advance(handle.execution_id)  # fallback still running
    assert not store.exists(layout.model_failure("xgboost"))

    sagemaker.set_status(fallback.job_name, "Completed")
    execution = orchestrator.advance(handle.execution_id)

    assert not store.exists(layout.model_failure("xgboost"))
    assert execution.jobs_for("evaluation")[0].state is JobState.IN_PROGRESS


def test_a_failed_fallback_is_recorded_as_the_model_failure(
    orchestrator, sagemaker, store, layout, record, frozen
):
    write_model(
        store, layout.compute, ExperimentCompute(experiment_id=EXP, allow_on_demand_fallback=True)
    )
    handle, execution = _capacity_failure(orchestrator, sagemaker, store, layout, record)
    (fallback,) = [j for j in execution.jobs if j.fallback_of]

    sagemaker.set_status(fallback.job_name, "Failed", FailureReason="AlgorithmError: boom")
    orchestrator.advance(handle.execution_id)

    failure = read_model(store, layout.model_failure("xgboost"), ModelFailure)
    assert failure.details["job_name"] == fallback.job_name


def test_a_non_capacity_failure_never_falls_back(
    orchestrator, sagemaker, store, layout, record, frozen
):
    write_model(
        store, layout.compute, ExperimentCompute(experiment_id=EXP, allow_on_demand_fallback=True)
    )
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)
    for job in _execution(store, layout, handle).jobs_for("training"):
        sagemaker.set_status(job.job_name, "Failed", FailureReason="AlgorithmError: ValueError")

    execution = orchestrator.advance(handle.execution_id)

    assert not any(j.fallback_of for j in execution.jobs)


# --- guards: refused before any AWS call (Property 11) -----------------------------------------


def test_over_budget_workflows_are_refused_before_submission(
    budget_orchestrator, budget_settings, sagemaker, store, layout, record, frozen
):
    budget_settings.budget_per_experiment_usd = 0.10  # one m5.large hour is more than this
    with pytest.raises(GuardViolation) as error:
        budget_orchestrator.start_training(record)
    assert error.value.code == "EXPERIMENT_BUDGET_EXCEEDED"
    assert sagemaker.calls == []
    assert store.list_uris(layout.executions_prefix) == []
    assert store.list_uris(layout.spend_prefix) == []


def test_the_monthly_ceiling_counts_every_experiment(
    budget_orchestrator, budget_settings, sagemaker
):
    """Four EDA runs at one m5.large-hour each fit under $0.50; the fifth, in another
    experiment, does not — the monthly ledger spans experiments."""
    budget_settings.budget_per_month_usd = 0.5

    def eda(experiment_id):
        layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, experiment_id)
        return budget_orchestrator.start_eda(make_record(layout, experiment_id, DATASET))

    for index in range(4):
        eda(f"exp-{index}")
    with pytest.raises(GuardViolation) as error:
        eda("exp-5")
    assert error.value.code == "MONTHLY_BUDGET_EXCEEDED"
    assert len(sagemaker.calls) == 4


def test_spend_reservations_are_recorded_with_provenance(
    budget_orchestrator, store, layout, record, frozen
):
    handle = budget_orchestrator.start_training(record)
    execution = _execution(store, layout, handle)

    entries = [
        read_model(store, uri, SpendReservation) for uri in store.list_uris(layout.spend_prefix)
    ]
    assert len(entries) == len(execution.jobs) == 6
    assert sum(e.reserved_usd for e in entries) == pytest.approx(execution.reserved_total_usd)
    monthly = store.list_uris(
        ExperimentLayout.monthly_spend_prefix(ARTIFACT_ROOT, NOW.strftime("%Y-%m"))
    )
    assert len(monthly) == 6
    assert all(e.price_table_version == "test" and "not the bill" in e.label for e in entries)


def test_with_enforcement_off_no_price_is_needed_and_nothing_is_reserved(
    orchestrator, store, layout, record, frozen
):
    """The production default: MLOps owns cost, so ML Factory prices and reserves nothing."""
    handle = orchestrator.start_training(record)
    execution = _execution(store, layout, handle)
    assert execution.reserved_total_usd is None
    assert all(job.reserved_usd is None for job in execution.jobs)
    assert store.list_uris(layout.spend_prefix) == []


def test_enforcement_on_without_a_price_table_is_a_configuration_error(
    budget_settings, store, sagemaker
):
    with pytest.raises(OrchestrationError) as error:
        SageMakerOrchestrator(budget_settings, store, sagemaker, None)
    assert error.value.code == "PRICE_TABLE_REQUIRED"


def test_an_unverified_price_table_refuses_everything(budget_settings, store, sagemaker, record):
    from backend.config import DEFAULT_PRICE_TABLE

    orchestrator = SageMakerOrchestrator(
        budget_settings, store, sagemaker, PriceTable.load(DEFAULT_PRICE_TABLE)
    )
    with pytest.raises(GuardViolation) as error:
        orchestrator.start_eda(record)
    assert error.value.code == "PRICE_TABLE_UNVERIFIED"
    assert sagemaker.calls == []


def test_an_instance_above_the_ceiling_is_refused(
    orchestrator, sagemaker, store, layout, record, frozen
):
    write_model(
        store,
        layout.compute,
        ExperimentCompute(experiment_id=EXP, training_instance_type="ml.m5.12xlarge"),
    )
    with pytest.raises(GuardViolation) as error:
        orchestrator.start_training(record)
    assert error.value.code == "INSTANCE_NOT_ALLOWED"
    assert sagemaker.calls == []


def test_more_models_than_the_parallel_limit_train_in_waves(
    orchestrator, settings, sagemaker, store, layout, record, frozen
):
    settings.max_parallel_training_jobs = 2
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")

    orchestrator.advance(handle.execution_id)
    first_wave = sagemaker.names("create_training_job")
    assert len(first_wave) == 2  # never more than the limit at once

    orchestrator.advance(handle.execution_id)  # nothing finished: nothing new
    assert len(sagemaker.names("create_training_job")) == 2

    for name in first_wave:
        sagemaker.set_status(name, "Completed")
    orchestrator.advance(handle.execution_id)
    assert len(sagemaker.names("create_training_job")) == 4
    assert sagemaker.names("create_processing_job")[-1].endswith("preparation")  # no eval yet


def test_the_platform_wide_cap_counts_other_experiments_jobs(
    orchestrator, settings, sagemaker, store, layout, record, frozen
):
    settings.max_platform_training_jobs = 3
    sagemaker.external_training_jobs = [
        {"TrainingJobName": "mlf-other-train-x", "TrainingJobStatus": "InProgress"}
    ] * 2
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")

    orchestrator.advance(handle.execution_id)

    assert len(sagemaker.names("create_training_job")) == 1


def test_a_runaway_model_list_is_refused(orchestrator, settings, sagemaker, record, frozen):
    settings.max_models_per_run = 2
    with pytest.raises(GuardViolation) as error:
        orchestrator.start_training(record)
    assert error.value.code == "TOO_MANY_MODELS"
    assert sagemaker.calls == []


# --- status honesty, stop, naming -------------------------------------------------------------


def test_an_unreadable_status_is_never_invented(orchestrator, sagemaker, store, layout, record):
    handle = orchestrator.start_eda(record)
    job = _execution(store, layout, handle).jobs[0]
    sagemaker.status.pop(job.job_name)
    sagemaker.created.pop(job.job_name)  # describe now raises

    execution = orchestrator.advance(handle.execution_id)

    assert execution.jobs[0].state is JobState.IN_PROGRESS  # last known, not guessed
    assert execution.status is ExecutionStatus.RUNNING


def test_an_unrecognised_aws_status_becomes_unknown(orchestrator, sagemaker, store, layout, record):
    handle = orchestrator.start_eda(record)
    sagemaker.set_status(_execution(store, layout, handle).jobs[0].job_name, "SomethingNew")

    assert orchestrator.advance(handle.execution_id).jobs[0].state is JobState.UNKNOWN


def test_stop_stops_running_jobs_and_submits_nothing_more(
    orchestrator, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)

    execution = orchestrator.stop(handle.execution_id)

    assert [op for op, _ in sagemaker.calls] == ["create_processing_job", "stop_processing_job"]
    assert execution.status is ExecutionStatus.STOPPED
    assert all(j.state in {JobState.STOPPING, JobState.STOPPED} for j in execution.jobs)
    _complete(sagemaker, execution, "preparation")
    orchestrator.advance(handle.execution_id)
    assert sagemaker.names("create_training_job") == []


def test_a_stopped_execution_is_observed_until_aws_confirms(
    orchestrator, sagemaker, store, layout, record, frozen
):
    from ml_engine.contracts.execution import is_settled

    handle = orchestrator.start_training(record)
    execution = orchestrator.stop(handle.execution_id)
    assert not is_settled(execution)  # preparation is still Stopping

    sagemaker.set_status(execution.jobs[0].job_name, "Stopped")
    execution = orchestrator.advance(handle.execution_id)

    assert execution.jobs[0].state is JobState.STOPPED
    assert is_settled(execution)
    assert sagemaker.names("create_training_job") == []


def test_a_refused_stop_does_not_abandon_the_other_jobs(
    orchestrator, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)
    jobs = _execution(store, layout, handle).jobs_for("training")
    sagemaker.fail_stop[jobs[0].job_name] = ClientError("ValidationException", "already ended")

    execution = orchestrator.stop(handle.execution_id)

    stopped = {
        kwargs["TrainingJobName"] for op, kwargs in sagemaker.calls if op == "stop_training_job"
    }
    assert stopped == {j.job_name for j in jobs}  # every job was asked, despite one refusal
    assert execution.status is ExecutionStatus.STOPPED


def test_a_submission_error_is_recorded_and_raised(orchestrator, sagemaker, store, layout, record):
    sagemaker.fail_create = {}
    original = orchestrator._submit

    def failing(execution, layout_, plan):
        sagemaker.fail_create[plan.name] = ClientError("AccessDeniedException", "not allowed")
        return original(execution, layout_, plan)

    orchestrator._submit = failing
    with pytest.raises(ClientError):
        orchestrator.start_eda(record)
    (uri,) = store.list_uris(layout.executions_prefix)
    assert read_model(store, uri, ExecutionRecord).jobs[0].state is JobState.FAILED


def _refuse_every_create(sagemaker, orchestrator):
    original = orchestrator._submit

    def failing(execution, layout_, plan):
        sagemaker.fail_create[plan.name] = ClientError("AccessDeniedException", "not allowed")
        return original(execution, layout_, plan)

    orchestrator._submit = failing
    return original


def test_a_refused_first_submission_closes_the_execution_and_fails_the_workflow(
    orchestrator, sagemaker, store, layout, record, frozen
):
    _refuse_every_create(sagemaker, orchestrator)
    with pytest.raises(ClientError):
        orchestrator.start_training(record)

    (uri,) = store.list_uris(layout.executions_prefix)
    execution = read_model(store, uri, ExecutionRecord)
    assert execution.status is ExecutionStatus.FAILED
    assert execution.failure_code == "SUBMISSION_FAILED"
    assert execution.jobs_for("preparation")[0].state is JobState.FAILED
    assert all(j.state is JobState.STOPPED for j in execution.jobs_for("training"))
    workflow = _workflow(store, layout)
    assert workflow.status is ExperimentStatus.FAILED
    assert workflow.failure_code == "SUBMISSION_FAILED"


def test_training_can_be_retried_after_a_refused_submission(
    orchestrator, sagemaker, store, layout, record, frozen
):
    original = _refuse_every_create(sagemaker, orchestrator)
    with pytest.raises(ClientError):
        orchestrator.start_training(record)

    sagemaker.fail_create.clear()
    orchestrator._submit = original
    handle = orchestrator.start_training(record)

    assert _execution(store, layout, handle).status is ExecutionStatus.RUNNING
    assert _workflow(store, layout).status is ExperimentStatus.PREPARING


def test_a_refused_submission_does_not_notify_start_listeners(orchestrator, sagemaker, record):
    started = []
    orchestrator.on_started(started.append)
    _refuse_every_create(sagemaker, orchestrator)

    with pytest.raises(ClientError):
        orchestrator.start_eda(record)

    assert started == []


@pytest.mark.parametrize(
    ("experiment_id", "model"),
    [("exp-abc123", "logistic_regression"), ("exp-" + "x" * 60, "catboost_regressor"), ("e", None)],
)
def test_job_names_are_valid_unique_and_deterministic(experiment_id, model):
    name = job_name(experiment_id, "deadbeef", "train", model)
    assert JOB_NAME.match(name) and len(name) <= 63
    assert name == job_name(experiment_id, "deadbeef", "train", model)
    assert name != job_name(experiment_id, "cafebabe", "train", model)


def test_execution_ids_round_trip():
    assert parse_execution_id("exp-1.training.0a1b2c3d")[0] == "exp-1"
    with pytest.raises(OrchestrationError):
        parse_execution_id("nonsense")


# --- derived-feature profiling (side job) ------------------------------------------------------


def test_derived_profiling_runs_as_a_processing_job_without_touching_status(
    orchestrator, sagemaker, store, layout, record
):
    orchestrator.start_eda(record)
    before = _workflow(store, layout)

    handle = orchestrator.start_derived_profiling(record)

    (name,) = [n for n in sagemaker.names("create_processing_job") if "derived" in n]
    request = sagemaker.created[name]
    assert request["AppSpecification"]["ContainerEntrypoint"] == [
        "python",
        "-m",
        "jobs.derived_profiling.main",
    ]
    assert JOB_NAME.match(name)
    assert parse_execution_id(handle.execution_id)[0] == EXP
    assert _workflow(store, layout) == before


def test_a_failed_derived_profiling_job_does_not_fail_the_experiment(
    orchestrator, sagemaker, store, layout, record
):
    orchestrator.start_eda(record)
    before = _workflow(store, layout)
    handle = orchestrator.start_derived_profiling(record)
    job = _execution(store, layout, handle).jobs[0]
    sagemaker.set_status(job.job_name, "Failed", FailureReason="AlgorithmError: boom")

    execution = orchestrator.advance(handle.execution_id)

    assert execution.status is ExecutionStatus.FAILED
    assert execution.failure_code == "DERIVED_PROFILING_FAILED"
    assert _workflow(store, layout) == before


def test_a_completed_derived_profiling_job_succeeds(orchestrator, sagemaker, store, layout, record):
    handle = orchestrator.start_derived_profiling(record)
    _complete(sagemaker, _execution(store, layout, handle), "derived_profiling")

    assert orchestrator.advance(handle.execution_id).status is ExecutionStatus.SUCCEEDED


def test_holdout_evaluation_runs_as_a_side_job(orchestrator, sagemaker, store, layout, record):
    orchestrator.start_eda(record)
    before = _workflow(store, layout)

    handle = orchestrator.start_holdout_evaluation(record, "eval-abc")

    (name,) = [n for n in sagemaker.names("create_processing_job") if "holdout" in n]
    app = sagemaker.created[name]["AppSpecification"]
    assert app["ContainerEntrypoint"] == ["python", "-m", "jobs.holdout_evaluation.main"]
    assert app["ContainerArguments"][-2:] == ["--evaluation-id", "eval-abc"]
    assert JOB_NAME.match(name)
    assert _workflow(store, layout) == before

    job = _execution(store, layout, handle).jobs[0]
    sagemaker.set_status(job.job_name, "Failed", FailureReason="AlgorithmError")
    assert orchestrator.advance(handle.execution_id).status is ExecutionStatus.FAILED
    assert _workflow(store, layout) == before  # the experiment itself is unaffected


# --- the environment a run uses ----------------------------------------------------------------

NEW_IMAGE = "123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-factory-stage@sha256:" + "b" * 64


def _image_of(sagemaker, name):
    request = sagemaker.created[name]
    if "AlgorithmSpecification" in request:
        return request["AlgorithmSpecification"]["TrainingImage"]
    return request["AppSpecification"]["ImageUri"]


def _finish_training(orchestrator, sagemaker, store, layout, handle):
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)
    _complete(sagemaker, _execution(store, layout, handle), "training")
    orchestrator.advance(handle.execution_id)
    _complete(sagemaker, _execution(store, layout, handle), "evaluation")
    return orchestrator.advance(handle.execution_id)


def test_a_run_is_pinned_to_the_image_it_started_on(
    orchestrator, settings, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)
    assert _execution(store, layout, handle).image_uri == IMAGE

    settings.stage_image_uri = NEW_IMAGE  # e.g. the server restarted with a new digest
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)

    for job in _execution(store, layout, handle).jobs_for("training"):
        assert _image_of(sagemaker, job.job_name) == IMAGE
        env = sagemaker.created[job.job_name]["Environment"]
        assert env["ML_FACTORY_STAGE_IMAGE_URI"] == IMAGE


def test_holdout_evaluation_runs_on_the_image_the_models_were_trained_on(
    orchestrator, settings, sagemaker, store, layout, record, frozen
):
    handle = orchestrator.start_training(record)
    assert _finish_training(orchestrator, sagemaker, store, layout, handle).status is (
        ExecutionStatus.SUCCEEDED
    )
    settings.stage_image_uri = NEW_IMAGE  # the platform was upgraded since

    evaluation = orchestrator.start_holdout_evaluation(record, "eval-1")

    (job,) = _execution(store, layout, evaluation).jobs
    assert _image_of(sagemaker, job.job_name) == IMAGE
    assert _execution(store, layout, evaluation).image_uri == IMAGE


def test_without_a_recorded_training_image_holdout_uses_the_current_one(
    orchestrator, settings, sagemaker, store, layout, record
):
    settings.stage_image_uri = NEW_IMAGE

    evaluation = orchestrator.start_holdout_evaluation(record, "eval-1")

    (job,) = _execution(store, layout, evaluation).jobs
    assert _image_of(sagemaker, job.job_name) == NEW_IMAGE


# --- cross-validation: runtime, budget and job granularity (CV design C9, Req 9) --------------

K = 3  # (K + 1) = 4 fits per Run
CV_RUNTIME = 3600 * (K + 1)  # training_max_runtime_seconds x (K + 1)


@pytest.fixture
def frozen_cv(store, layout):
    write_model(
        store,
        layout.experiment_config,
        make_config(layout, EXP, DATASET, MODELS, split=cv_split(folds=K)),
    )


def test_cv_submits_one_training_job_per_model_not_k_jobs(
    orchestrator, sagemaker, store, layout, record, frozen_cv
):
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)

    jobs = _execution(store, layout, handle).jobs_for("training")
    # One training job per Run — folds run sequentially inside it (design D3), never K jobs.
    assert len(jobs) == len(MODELS)
    assert {j.model_name for j in jobs} == set(MODELS)
    assert len(sagemaker.names("create_training_job")) == len(MODELS)


def test_cv_training_runtime_scales_by_k_plus_one_with_spot_on(
    orchestrator, sagemaker, store, layout, record, frozen_cv
):
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)

    for job in _execution(store, layout, handle).jobs_for("training"):
        request = sagemaker.created[job.job_name]
        assert request["EnableManagedSpotTraining"] is True
        stopping = request["StoppingCondition"]
        assert stopping["MaxRuntimeInSeconds"] == CV_RUNTIME
        assert stopping["MaxWaitTimeInSeconds"] >= CV_RUNTIME
        assert job.compute.max_runtime_seconds == CV_RUNTIME


def test_cv_training_runtime_honours_an_explicit_ceiling(
    orchestrator, settings, sagemaker, store, layout, record, frozen_cv
):
    settings.cv_training_max_runtime_seconds = 9000
    settings.spot_max_wait_seconds = 3600  # the wait must still rise to at least the runtime
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)

    for job in _execution(store, layout, handle).jobs_for("training"):
        stopping = sagemaker.created[job.job_name]["StoppingCondition"]
        assert stopping["MaxRuntimeInSeconds"] == 9000
        assert stopping["MaxWaitTimeInSeconds"] == 9000


def test_cv_reservation_is_rate_times_cv_runtime_times_instances(
    budget_orchestrator, store, layout, record, frozen_cv
):
    from backend.pricing import PriceTable, reserved_cost_usd
    from ml_engine.contracts.execution import JobKind

    handle = budget_orchestrator.start_training(record)
    execution = _execution(store, layout, handle)
    rate = PriceTable.load(budget_orchestrator._settings.price_table_path).rate(
        "us-east-1", JobKind.TRAINING, "ml.m5.large"
    )
    expected = reserved_cost_usd(rate, instance_count=1, max_billable_seconds=CV_RUNTIME)
    for job in execution.jobs_for("training"):
        assert job.reserved_usd == pytest.approx(expected)
        # (K + 1) scaling: the CV reservation is (K + 1) x the single-split reservation.
        single = reserved_cost_usd(rate, instance_count=1, max_billable_seconds=3600)
        assert job.reserved_usd == pytest.approx(single * (K + 1))


def test_cv_over_budget_is_refused_before_submission_with_estimate_and_ceiling(
    budget_orchestrator, budget_settings, sagemaker, store, layout, record, frozen_cv
):
    budget_settings.budget_per_experiment_usd = 0.10  # far below 4 Runs x 4 fits of m5.large
    with pytest.raises(GuardViolation) as error:
        budget_orchestrator.start_training(record)
    assert error.value.code == "EXPERIMENT_BUDGET_EXCEEDED"
    assert "$" in error.value.message  # names the estimate and the ceiling
    assert sagemaker.calls == []
    assert store.list_uris(layout.executions_prefix) == []
    assert store.list_uris(layout.spend_prefix) == []


def test_cv_runtime_above_the_sagemaker_cap_is_refused(
    orchestrator, settings, sagemaker, store, layout, record
):
    # 10 folds x a 3-day per-fit runtime = 33 days > SageMaker's 28-day training limit.
    settings.training_max_runtime_seconds = 3 * 24 * 3600
    write_model(
        store,
        layout.experiment_config,
        make_config(layout, EXP, DATASET, MODELS, split=cv_split(folds=10)),
    )
    with pytest.raises(OrchestrationError) as error:
        orchestrator.start_training(record)
    assert error.value.code == "CV_RUNTIME_TOO_LONG"
    assert sagemaker.calls == []


def test_cv_on_demand_fallback_keeps_the_cv_runtime(
    orchestrator, sagemaker, store, layout, record, frozen_cv
):
    from ml_engine.contracts.execution import ExperimentCompute

    write_model(
        store, layout.compute, ExperimentCompute(experiment_id=EXP, allow_on_demand_fallback=True)
    )
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")
    orchestrator.advance(handle.execution_id)
    for job in _execution(store, layout, handle).jobs_for("training"):
        if job.model_name == "xgboost":
            sagemaker.set_status(
                job.job_name,
                "Failed",
                SecondaryStatus="MaxWaitTimeExceeded",
                FailureReason="Spot capacity not available",
            )
        else:
            sagemaker.set_status(job.job_name, "Completed")
    execution = orchestrator.advance(handle.execution_id)

    (fallback,) = [j for j in execution.jobs if j.fallback_of]
    request = sagemaker.created[fallback.job_name]
    assert request["EnableManagedSpotTraining"] is False
    assert request["StoppingCondition"]["MaxRuntimeInSeconds"] == CV_RUNTIME


def test_cv_waves_and_platform_cap_still_hold(
    orchestrator, settings, sagemaker, store, layout, record, frozen_cv
):
    settings.max_parallel_training_jobs = 2
    handle = orchestrator.start_training(record)
    _complete(sagemaker, _execution(store, layout, handle), "preparation")

    orchestrator.advance(handle.execution_id)
    first_wave = sagemaker.names("create_training_job")
    assert len(first_wave) == 2  # never more than the limit, as in single-split mode
    for name in first_wave:
        assert sagemaker.created[name]["StoppingCondition"]["MaxRuntimeInSeconds"] == CV_RUNTIME
        sagemaker.set_status(name, "Completed")
    orchestrator.advance(handle.execution_id)
    assert len(sagemaker.names("create_training_job")) == 4


def test_a_cv_preparation_failure_submits_no_training_job(
    orchestrator, sagemaker, store, layout, record, frozen_cv
):
    handle = orchestrator.start_training(record)
    prep = _execution(store, layout, handle).jobs_for("preparation")[0]
    sagemaker.set_status(prep.job_name, "Failed", FailureReason="SPLIT_FAILED: Test_Fold too small")

    execution = orchestrator.advance(handle.execution_id)

    assert execution.status is ExecutionStatus.FAILED
    assert sagemaker.names("create_training_job") == []
