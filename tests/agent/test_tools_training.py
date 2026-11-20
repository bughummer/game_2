"""Training, evaluation and run-status tools (ml-assistant task 10.3, design C5 catalogue).

The tool calls run against the offline container: in-memory store and the fake SageMaker behind
the real ``SageMakerOrchestrator``, so every job goes through the orchestrator's guards. EDA and
feature selection are done first by an inline service on the same store.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from backend.agent import catalogue, tools_training
from backend.agent.tools import AwaitedJobSpec, ToolContext
from backend.auth import Principal
from backend.config import Settings
from backend.container import build_container
from backend.errors import ConflictError
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import ExperimentService
from ml_engine.contracts.agent import DatasetRef, Session, ToolClass
from tests.backend.conftest import IMAGE, ROLE
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.fake_sagemaker import FakeLogs, FakeSageMaker
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

OWNER = "alice"
FEATURES = ["tenure_months", "monthly_charges", "contract"]

COSTLY = {
    "start_training": ("training", "start_training", "training"),
    "rerun_training": ("training", "rerun_training", "training"),
    "start_validation": ("validation", "start_validation", "processing"),
    "start_holdout_evaluation": ("holdout", "start_holdout_evaluation", "processing"),
    "start_ablation": ("ablation", "start_ablation", "processing"),
}
READ = {
    "preflight_check",
    "get_training_status",
    "get_execution_status",
    "get_comparison",
    "get_threshold_analysis",
    "list_model_versions",
    "get_validation",
    "get_holdout_evaluation",
    "get_ablation",
}


# --- registration --------------------------------------------------------------------------


def test_registers_the_training_catalogue_with_its_flags():
    registry = catalogue.build_registry(SimpleNamespace())
    names = set(registry.names())
    assert names >= set(COSTLY) | READ
    for name, (kind, audit, estimator) in COSTLY.items():
        spec = registry.get(name)
        assert spec.tool_class is ToolClass.COSTLY, name
        assert spec.job_extractor is not None, name
        assert spec.flags.audit_action == audit, name
        assert spec.flags.estimator == estimator, name
        assert spec.flags.experiment_arg == "experiment_id", name
        assert spec.flags.changes_config == (name == "start_training"), name
        assert spec.job_extractor({"experiment_id": "e1", "execution_id": "x1"}) == [
            AwaitedJobSpec("x1", "e1", kind)
        ]
        assert spec.job_extractor({"experiment_id": "e1", "execution_id": None}) == []
    for name in READ:
        spec = registry.get(name)
        assert spec.tool_class is ToolClass.READ, name
        assert spec.job_extractor is None, name
        assert spec.flags.audit_action is None, name
        assert spec.flags.verbatim == (name in {"get_comparison", "get_validation"}), name
    assert registry.get("start_validation").flags.dataset_args == ("dataset_id",)
    assert registry.get("start_holdout_evaluation").flags.dataset_args == ("dataset_id",)
    # The routes expose no instance or Spot option, so no tool does.
    module_tools = set(COSTLY) | READ
    assert all(not registry.get(n).flags.instance_args for n in module_tools)
    for name in module_tools:
        schema = registry.get(name).input_model.model_json_schema()
        assert not any("instance" in p or "spot" in p for p in schema.get("properties", {}))
    assert not any(word in n for n in names for word in ("promote", "delete", "download"))


def test_start_training_args_are_the_route_body_fields():
    fields = set(tools_training.StartTrainingArgs.model_fields) - {"experiment_id"}
    assert fields <= set(TrainingRequest.model_fields)
    with pytest.raises(ValueError):
        tools_training.StartTrainingArgs.model_validate(
            {"experiment_id": "e1", "training_instance_type": "ml.m5.large"}
        )


# --- against the offline container ---------------------------------------------------------


@pytest.fixture
def env(price_table_path, classification_frame):
    store = InMemoryObjectStore()
    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn=ROLE,
        stage_image_uri=IMAGE,
        price_table_path=price_table_path,
    )
    inline = ExperimentService(
        settings=settings,
        repository=ExperimentRepository(settings=settings, store=store),
        store=store,
        orchestrator=InlineOrchestrator(store),
    )
    uri = write_dataset(store, "churn.csv", classification_frame)
    record = inline.create(name="t", dataset_uri=uri, target_column="churned", user=OWNER)
    inline.save_features(record.experiment_id, selected_features=FEATURES)

    sagemaker = FakeSageMaker()
    container = build_container(
        settings,
        store=store,
        sagemaker_client=sagemaker,
        s3_client=object(),
        logs_client=FakeLogs(),
    )
    ref = DatasetRef(
        dataset_id="ds_churn",
        uri=uri,
        filename="churn.csv",
        sha256=None,
        size_bytes=1,
        file_format="csv",
        columns=[],
        row_count=None,
    )
    session = Session.model_construct(
        session_id="s1", owner=OWNER, experiment_ids=[record.experiment_id], datasets=[ref]
    )
    ctx = ToolContext(session=session, principal=Principal(user=OWNER), services=container)
    yield SimpleNamespace(
        ctx=ctx, container=container, sagemaker=sagemaker, experiment_id=record.experiment_id
    )
    container.agent._runner.shutdown()


def test_start_training_composes_the_route_request_and_reports_its_job(env):
    args = tools_training.StartTrainingArgs(
        experiment_id=env.experiment_id,
        models=["logistic_regression"],
        evaluation_mode="cv",
        cv_folds=3,
    )
    out = tools_training.start_training(env.ctx, args)

    data = out.data
    assert data["experiment_id"] == env.experiment_id
    assert data["execution_id"]
    assert data["requested_models"] == ["logistic_regression"]
    assert (data["evaluation_mode"], data["cv_folds"]) == ("cv", 3)
    assert data["cv_scheme"]
    assert data["class_weighting"] == "auto"
    assert "preflight_warnings" in data and "unresolved_leakage_warnings" in data
    # Same audit detail keys as routes/training.py.
    assert set(out.audit_detail) == {
        "models",
        "evaluation_mode",
        "cv_folds",
        "cv_scheme",
        "test_fraction",
    }
    assert out.audit_detail["models"] == "logistic_regression"
    # The job went through the real orchestrator to the fake SageMaker.
    assert env.sagemaker.calls
    spec = catalogue.build_registry(env.container).get("start_training")
    assert spec.job_extractor(data) == [
        AwaitedJobSpec(data["execution_id"], env.experiment_id, "training")
    ]

    status = tools_training.get_execution_status(
        env.ctx,
        tools_training.ExecutionStatusArgs(
            experiment_id=env.experiment_id, execution_id=data["execution_id"]
        ),
    )
    assert status.data["execution_id"] == data["execution_id"]
    training = tools_training.get_training_status(
        env.ctx, tools_training.ExperimentArgs(experiment_id=env.experiment_id)
    )
    assert training.data["experiment_id"] == env.experiment_id


def test_read_tools_before_training(env):
    exp = tools_training.ExperimentArgs(experiment_id=env.experiment_id)
    preflight = tools_training.preflight_check(env.ctx, exp)
    assert preflight.data["experiment_id"] == env.experiment_id
    assert "ready" in preflight.data
    assert tools_training.get_ablation(env.ctx, exp).data == {"status": "not_started_or_running"}
    versions = tools_training.list_model_versions(env.ctx, exp)
    assert versions.data == {"experiment_id": env.experiment_id, "count": 0, "model_versions": []}


def test_evaluation_tools_refuse_through_the_service_before_training(env):
    # The shell has already swapped the dataset id for its URI.
    uri = env.ctx.session.datasets[0].uri
    with pytest.raises(ConflictError):
        tools_training.start_holdout_evaluation(
            env.ctx,
            tools_training.StartHoldoutArgs(experiment_id=env.experiment_id, dataset_id=uri),
        )
    with pytest.raises(ConflictError):
        tools_training.start_validation(
            env.ctx,
            tools_training.StartValidationArgs(experiment_id=env.experiment_id, dataset_id=uri),
        )
    assert env.sagemaker.calls == []


def test_redact_drops_storage_uris_at_any_depth():
    data = {
        "a": "s3://bucket/key",
        "b": {"model_uri": "s3://x", "name": "m"},
        "c": ["s3://y", "keep"],
        "image_uri": "123.dkr.ecr/img@sha256:1",
    }
    assert tools_training._redact(data) == {
        "b": {"name": "m"},
        "c": ["keep"],
        "image_uri": "123.dkr.ecr/img@sha256:1",
    }
