"""Experiment, leakage and business-context tools against the in-memory container (task 10.1).

The tools run through the real ``ToolExecutor`` and ``ToolInputs`` over a container built on the
in-memory store with the inline orchestrator, so EDA artifacts exist and audit entries are real.
"""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError as PydanticValidationError

from backend.agent import tools_experiments
from backend.agent.execute import CODE_INPUT_REJECTED, ToolExecutor
from backend.agent.inputs import ToolInputs
from backend.agent.tools import AwaitedJobSpec, ToolRegistry
from backend.config import Settings
from backend.container import build_container
from ml_engine.contracts.agent import DatasetRef, Session, ToolClass
from tests.backend.conftest import IMAGE, ROLE
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.fake_sagemaker import FakeLogs, FakeSageMaker
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

OWNER = "alice"
DATASET_ID = "ds_churn"

EXPECTED = {
    "create_experiment": (ToolClass.COSTLY, "create_experiment"),
    "get_leakage_findings": (ToolClass.READ, None),
    "get_feature_review": (ToolClass.READ, None),
    "set_business_context": (ToolClass.MUTATING, "set_business_context"),
    "get_business_context": (ToolClass.READ, None),
    "investigate_leakage": (ToolClass.MUTATING, "investigate_leakage"),
    "get_leakage_investigation": (ToolClass.READ, None),
}


class _NoRejectionAudit:
    def __init__(self) -> None:
        self.records = []

    def write(self, session, record):
        self.records.append(record)
        return "agent-audit"


@pytest.fixture
def store() -> InMemoryObjectStore:
    return InMemoryObjectStore()


@pytest.fixture
def container(store, price_table_path):
    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn=ROLE,
        stage_image_uri=IMAGE,
        price_table_path=price_table_path,
        dev_identity=OWNER,
    )
    built = build_container(
        settings,
        store=store,
        orchestrator=InlineOrchestrator(store),
        sagemaker_client=FakeSageMaker(),
        s3_client=object(),
        logs_client=FakeLogs(),
    )
    yield built
    agent = getattr(built, "agent", None)
    if agent is not None:
        agent._runner.shutdown()


@pytest.fixture
def harness(container, store, classification_frame):
    uri = write_dataset(store, "churn.csv", classification_frame)
    ref = DatasetRef(
        dataset_id=DATASET_ID,
        uri=uri,
        filename="churn.csv",
        sha256=None,
        size_bytes=len(store.read_bytes(uri)),
        file_format="csv",
        columns=[],
        row_count=None,
    )
    session = Session.model_construct(
        session_id="s1", owner=OWNER, experiment_ids=[], datasets=[ref], audit_seq=0
    )
    registry = ToolRegistry()
    tools_experiments.register(registry, container)
    audit = _NoRejectionAudit()
    inputs = ToolInputs(audit=audit, authorizer=container.authorizer, allowed_instance_types=[])
    executor = ToolExecutor(inputs=inputs, services=container)

    def run(name, arguments):
        return executor.execute(session, registry.get(name), arguments)

    return run, session, registry, uri, audit


def _audit_entry(store, ref):
    return json.loads(store.read_bytes(ref).decode("utf-8"))


def _create(run, session):
    result = run(
        "create_experiment",
        {"name": "churn", "dataset_id": DATASET_ID, "target_column": "churned"},
    )
    assert result.ok, result.error
    session.experiment_ids.extend(result.output.experiment_ids)
    return result


# --- registration --------------------------------------------------------------------------


def test_registers_each_tool_with_its_class_flags_and_audit_action():
    registry = ToolRegistry()
    tools_experiments.register(registry, object())
    assert set(registry.names()) == set(EXPECTED)
    for name, (tool_class, action) in EXPECTED.items():
        spec = registry.get(name)
        assert spec.tool_class is tool_class, name
        assert spec.flags.audit_action == action, name
        assert not spec.flags.changes_config, name
    create = registry.get("create_experiment")
    assert create.flags.dataset_args == ("dataset_id",)
    assert create.flags.experiment_arg is None
    assert create.flags.estimator == "processing"
    assert create.job_extractor is tools_experiments.eda_jobs
    for name in set(EXPECTED) - {"create_experiment"}:
        assert registry.get(name).flags.experiment_arg == "experiment_id", name
    for model in (tools_experiments.CreateExperimentArgs, tools_experiments.SetBusinessContextArgs):
        assert model.model_json_schema()["additionalProperties"] is False


def test_identity_cannot_be_passed_as_an_argument():
    with pytest.raises(PydanticValidationError):
        tools_experiments.CreateExperimentArgs.model_validate(
            {"name": "n", "dataset_id": DATASET_ID, "target_column": "y", "user": "bob"}
        )
    with pytest.raises(PydanticValidationError):
        tools_experiments.InvestigateLeakageArgs.model_validate(
            {"experiment_id": "e1", "user": "bob"}
        )


# --- create_experiment ---------------------------------------------------------------------


def test_create_experiment_resolves_dataset_id_returns_ids_and_audits_like_the_route(
    harness, container, store
):
    run, session, _registry, uri, _audit = harness
    result = _create(run, session)
    out = result.output
    experiment_id = out.data["experiment_id"]

    assert out.experiment_ids == (experiment_id,)
    assert out.data["dataset_id"] == DATASET_ID
    assert "s3://" not in repr(out.data) and "artifact_prefix" not in out.data
    record = container.experiments.get(experiment_id)
    assert record.created_by == OWNER
    assert record.target_column == "churned"

    expected_job = AwaitedJobSpec(
        execution_id=record.eda_execution_arn, experiment_id=experiment_id, kind="eda"
    )
    assert out.awaited == (expected_job,)
    assert tools_experiments.eda_jobs(out.data) == [expected_job]

    entry = _audit_entry(store, result.service_audit_ref)
    assert entry["action"] == "create_experiment"
    assert entry["actor"] == OWNER
    assert entry["detail"] == {"dataset_uri": uri, "target_column": "churned"}


def test_create_experiment_rejects_an_unknown_dataset_id_without_creating(harness, container):
    run, _session, _registry, _uri, audit = harness
    result = run(
        "create_experiment",
        {"name": "churn", "dataset_id": "ds_unknown", "target_column": "churned"},
    )
    assert result.error.code == CODE_INPUT_REJECTED
    assert container.experiments.list(created_by=OWNER) == []
    assert [r.detail["reason"] for r in audit.records] == ["unknown_dataset"]


def test_eda_jobs_is_empty_without_an_execution_id():
    assert tools_experiments.eda_jobs({"experiment_id": "e1", "eda_execution_id": None}) == []


# --- reads and mutations on an existing experiment -----------------------------------------


def test_leakage_and_feature_review_reads_match_the_service(harness, container):
    run, session, _registry, _uri, _audit = harness
    experiment_id = _create(run, session).output.data["experiment_id"]

    findings = run("get_leakage_findings", {"experiment_id": experiment_id})
    assert findings.ok, findings.error
    report = container.experiments.leakage(experiment_id)
    assert findings.output.data["target_column"] == report.target_column
    assert len(findings.output.data["findings"]) == len(report.findings)
    assert findings.service_audit_ref is None

    review = run("get_feature_review", {"experiment_id": experiment_id})
    assert review.ok, review.error
    assert review.output.data == container.experiments.feature_review(experiment_id).model_dump(
        mode="json"
    )


def test_business_context_round_trip_and_audit(harness, store):
    run, session, _registry, _uri, _audit = harness
    experiment_id = _create(run, session).output.data["experiment_id"]

    empty = run("get_business_context", {"experiment_id": experiment_id})
    assert empty.output.data == {"experiment_id": experiment_id, "business_context": None}

    result = run(
        "set_business_context",
        {
            "experiment_id": experiment_id,
            "objective": "Reduce churn",
            "prediction_scenario": "Score customers monthly",
            "temporal_columns": ["signup_date"],
        },
    )
    assert result.ok, result.error
    assert result.output.data["created_by"] == OWNER
    entry = _audit_entry(store, result.service_audit_ref)
    assert (entry["action"], entry["detail"]) == ("set_business_context", {})

    stored = run("get_business_context", {"experiment_id": experiment_id})
    assert stored.output.data["business_context"]["objective"] == "Reduce churn"
    assert stored.output.data["business_context"]["temporal_columns"] == ["signup_date"]


def test_investigate_leakage_builds_the_pre_training_phase_and_audits_warnings(
    harness, container, store
):
    run, session, _registry, _uri, _audit = harness
    experiment_id = _create(run, session).output.data["experiment_id"]

    result = run("investigate_leakage", {"experiment_id": experiment_id, "use_llm": False})
    assert result.ok, result.error
    warnings = result.output.data["warnings"]
    entry = _audit_entry(store, result.service_audit_ref)
    assert entry["action"] == "investigate_leakage"
    assert entry["detail"] == {"warnings": str(len(warnings))}

    pre = run("get_leakage_investigation", {"experiment_id": experiment_id})
    assert pre.ok, pre.error
    assert pre.output.data["phase"] == "pre_training"
    assert len(pre.output.data["report"]["warnings"]) == len(warnings)
    post = run(
        "get_leakage_investigation", {"experiment_id": experiment_id, "phase": "post_training"}
    )
    assert post.output.data["report"] is None


def test_experiment_tools_refuse_an_unlinked_experiment(harness):
    run, session, _registry, _uri, _audit = harness
    experiment_id = _create(run, session).output.data["experiment_id"]
    session.experiment_ids.clear()
    result = run("investigate_leakage", {"experiment_id": experiment_id})
    assert result.error.code == CODE_INPUT_REJECTED
