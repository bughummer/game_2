"""The smoke harness step path, offline (ml-assistant task 12.7; Req 17.3, 17.4, 17.6).

The real container, ``AgentService``, loop and tools run against the in-memory store. SageMaker
is the request-validating fake; when the harness advances the poller, each submitted job runs
its real stage on the in-memory store first, so EDA, training and validation produce real
artifacts. The model is a ``ScriptedFakeConverseClient`` whose later turns are built when they
are requested, because they need the experiment and dataset ids created on the way.
"""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace
from urllib.parse import urlparse

import pytest

from backend.agent.service import LocalFileInput
from backend.auth import Principal
from backend.config import Settings
from backend.container import build_container
from jobs._common.experiment_state import ArtifactExperimentStateWriter
from jobs.evaluation.main import run_evaluation
from jobs.preprocessing.main import run_preparation
from jobs.profiling.main import run_profiling
from jobs.training.main import run_training
from jobs.validation.main import run_validation
from ml_engine.contracts.agent import SessionStatus
from ml_engine.io import ExperimentLayout
from scripts import agent_smoke
from tests.backend.conftest import IMAGE, ROLE
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.fake_sagemaker import FakeLogs, FakeSageMaker
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.scripted_converse import (
    ScriptedFakeConverseClient,
    ScriptStep,
    text_turn,
    tool_turn,
)

OWNER = agent_smoke.SMOKE_OWNER


class FakeS3:
    """Signs nothing; the URL carries the bucket and key the test's PUT writes to."""

    def __init__(self, objects: InMemoryObjectStore) -> None:
        self._objects = objects

    def generate_presigned_url(self, operation, Params, ExpiresIn):  # noqa: N803
        return f"https://{Params['Bucket']}.s3.example/{Params['Key']}?X-Amz-Signature=sig"

    def get_object(self, Bucket, Key, Range=None):  # noqa: N803
        data = self._objects.read_bytes(f"s3://{Bucket}/{Key}")
        if Range:
            start, end = Range.removeprefix("bytes=").split("-")
            data = data[int(start) : int(end) + 1]
        return {"Body": BytesIO(data)}


class StageRunningSageMaker(FakeSageMaker):
    """Completes each in-progress job by running its real stage on the in-memory store."""

    def __init__(self, store: InMemoryObjectStore) -> None:
        super().__init__()
        self._store = store

    def run_pending(self) -> None:
        for job, request in list(self.created.items()):
            if self.status[job]["status"] != "InProgress":
                continue
            tags = {t["Key"]: t["Value"] for t in request["Tags"]}
            spec = request.get("AppSpecification") or request["AlgorithmSpecification"]
            raw = spec["ContainerArguments"]
            args = {raw[i].lstrip("-").replace("-", "_"): raw[i + 1] for i in range(0, len(raw), 2)}
            try:
                _run_stage(self._store, tags["ml-factory:stage"], args)
            except Exception as exc:
                self.set_status(job, "Failed", FailureReason=str(exc)[:200])
            else:
                self.set_status(job, "Completed")


def _run_stage(store, stage: str, args: dict[str, str]) -> None:
    layout = ExperimentLayout(base=args["artifact_base"])
    eid = args["experiment_id"]
    if stage == "profiling":
        run_profiling(
            store, layout, experiment_id=eid, dataset_uri=args["dataset_uri"],
            target_column=args["target_column"], file_format=args.get("file_format"),
        )
    elif stage == "preparation":
        run_preparation(store, layout, experiment_id=eid)
    elif stage == "training":
        run_training(store, layout, experiment_id=eid, model_name=args["model_name"])
    elif stage == "evaluation":
        run_evaluation(
            store, layout, experiment_id=eid,
            state_writer=ArtifactExperimentStateWriter(store, layout),
        )
    elif stage == "validation":
        run_validation(store, layout, experiment_id=eid, validation_id=args["validation_id"])
    else:
        raise AssertionError(f"unexpected stage {stage}")


def _file_id(path) -> str:
    return LocalFileInput.model_validate(agent_smoke.local_file_entry(path)).to_ref().file_id


@pytest.fixture
def env(tmp_path, price_table_path):
    base = Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn=ROLE,
        stage_image_uri=IMAGE,
        price_table_path=price_table_path,
        allowed_instance_types=["ml.m5.xlarge", "ml.c5.large", "ml.m5.large"],
    )
    settings = agent_smoke.smoke_settings(base)
    store = InMemoryObjectStore()
    sagemaker = StageRunningSageMaker(store)
    files = agent_smoke.write_smoke_datasets(tmp_path)  # same bytes the harness writes again
    train_id = _file_id(files[agent_smoke.SMOKE_TRAIN_FILE])
    validation_id = _file_id(files[agent_smoke.SMOKE_VALIDATION_FILE])
    state = SimpleNamespace(service=None, sid=None)

    def session():
        return state.service.store.load(state.sid).session

    def experiment_id():
        return session().experiment_ids[0]

    def dataset_id(name):
        return next(d.dataset_id for d in session().datasets if d.filename == name)

    def lazy(build):
        return ScriptStep(build)  # built at request time; see LazyConverse

    script = [
        ScriptStep(tool_turn(("t-plan", "submit_plan", {"steps": ["upload", "train", "report"]}))),
        ScriptStep(tool_turn(("t-up1", "request_local_upload", {"file_id": train_id}))),
        lazy(lambda: tool_turn(("t-exp", "create_experiment", {
            "name": "smoke", "dataset_id": dataset_id(agent_smoke.SMOKE_TRAIN_FILE),
            "target_column": agent_smoke.SMOKE_TARGET,
        }))),
        lazy(lambda: tool_turn(("t-feat", "decide_features", {
            "experiment_id": experiment_id(), "selected_features": list(agent_smoke.SMOKE_FEATURES),
        }))),
        lazy(lambda: tool_turn(("t-train", "start_training", {
            "experiment_id": experiment_id(), "models": ["logistic_regression"],
        }))),
        ScriptStep(tool_turn(("t-up2", "request_local_upload", {"file_id": validation_id}))),
        lazy(lambda: tool_turn(("t-val", "start_validation", {
            "experiment_id": experiment_id(),
            "dataset_id": dataset_id(agent_smoke.SMOKE_VALIDATION_FILE),
        }))),
        ScriptStep(text_turn("The smoke run finished: the model was trained and validated.")),
    ]
    converse = LazyConverse(steps=script)
    container = build_container(
        settings, store=store, sagemaker_client=sagemaker, s3_client=FakeS3(store),
        logs_client=FakeLogs(), converse_client=converse,
    )
    check, sid = agent_smoke.check_session_round_trip(settings, store, lambda: store, converse)
    assert check.passed, check.detail
    service = agent_smoke.build_step_service(settings, container, converse)
    state.service, state.sid = service, sid

    def put(ticket, body):
        url = urlparse(ticket.upload_url)
        store.write_bytes(f"s3://{url.hostname.split('.')[0]}{url.path}", body)

    def tick():
        sagemaker.run_pending()
        container.poller.tick()

    yield SimpleNamespace(
        service=service, sid=sid, sagemaker=sagemaker, settings=settings, converse=converse,
        folder=tmp_path, put=put, tick=tick, session=session, container=container,
    )
    container.agent._runner.shutdown()


class LazyConverse(ScriptedFakeConverseClient):
    """A scripted client whose steps may be zero-argument builders of the turn."""

    def converse(self, **kwargs):
        turn = super().converse(**kwargs)
        return turn() if callable(turn) else turn


def _run(env, **drive):
    return agent_smoke.run_step_path(
        env.service, Principal(user=OWNER), env.sid, env.folder,
        put=env.put, tick=env.tick, sleep=lambda _: None, **drive,
    )


def test_smoke_settings_pick_the_smallest_allowed_instance_on_spot(env):
    assert env.settings.training_instance_type == "ml.c5.large"
    assert env.settings.spot_enabled is True
    assert env.settings.agent_limit_max_estimated_spend_usd <= 5


def test_synthetic_datasets_stay_within_the_bounds(tmp_path):
    files = agent_smoke.write_smoke_datasets(tmp_path)
    for path in files.values():
        lines = path.read_text(encoding="utf-8").splitlines()
        assert lines[0].split(",")[-1] == agent_smoke.SMOKE_TARGET  # labelled
        assert len(lines) - 1 <= agent_smoke.SMOKE_MAX_ROWS
        assert path.stat().st_size <= agent_smoke.SMOKE_MAX_BYTES


def test_the_step_path_passes_offline_in_order(env):
    results = _run(env)

    assert [r.name for r in results] == [name for name, _ in agent_smoke.SMOKE_STEPS]
    assert all(r.passed for r in results), agent_smoke.format_report(
        agent_smoke.SmokeReport(checks=results, session_id=env.sid)
    )
    s = env.session()
    assert s.mode.value == "supervised"
    assert s.status is SessionStatus.COMPLETED
    assert s.approvals and {a.status.value for a in s.approvals} == {"executed"}
    trained = [kw for op, kw in env.sagemaker.calls if op == "create_training_job"]
    assert trained and all(
        kw["ResourceConfig"]["InstanceType"] == "ml.c5.large" and kw["EnableManagedSpotTraining"]
        for kw in trained
    )
    assert env.converse.remaining == 0


def test_reaching_the_spend_maximum_stops_jobs_and_leaves_the_session_resumable(env):
    def tick():
        # While the EDA job runs, the spend counter reaches the maximum before the job ends.
        with env.service.store.transaction(env.sid) as s:
            if s.status is SessionStatus.WAITING_FOR_JOB:
                s.counters.estimated_spend_usd = s.limits.max_estimated_spend_usd

    results = agent_smoke.run_step_path(
        env.service, Principal(user=OWNER), env.sid, env.folder,
        put=env.put, tick=tick, sleep=lambda _: None,
    )

    spend = results[-1]
    assert spend.name == "spend_limit" and not spend.passed
    assert "max_estimated_spend_usd" in spend.detail
    assert next(r.name for r in results if not r.passed) == "eda"
    stopped = [kw["ProcessingJobName"] for op, kw in env.sagemaker.calls
               if op == "stop_processing_job"]
    assert stopped, "the running EDA job was not stopped"
    s = env.session()
    assert s.status is SessionStatus.WAITING_FOR_JOB  # not cancelled: it can be resumed
    assert not [kw for op, kw in env.sagemaker.calls if op == "create_training_job"]
