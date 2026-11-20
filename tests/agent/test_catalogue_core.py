"""Unit tests for the catalogue and the core tools (task 4.11)."""

import sys
import types
from types import SimpleNamespace

import pytest
from pydantic import ValidationError as PydanticValidationError

from backend.agent import catalogue, tools_core
from backend.agent.tools import ToolContext
from backend.auth import Principal
from backend.config import Settings
from backend.services.datasets import DatasetService
from ml_engine.contracts.agent import DatasetRef, Session, ToolClass
from tests.support.datasets import ARTIFACT_ROOT

CORE_TOOLS = {
    "submit_plan",
    "ask_user",
    "list_datasets",
    "list_experiments",
    "get_experiment",
    "get_eda_summary",
}


def make_session(datasets=()) -> Session:
    return Session.model_construct(
        session_id="s1", owner="alice", experiment_ids=[], datasets=list(datasets)
    )


def make_ctx(services, session=None) -> ToolContext:
    session = session or make_session()
    return ToolContext(session=session, principal=Principal(user=session.owner), services=services)


def record(experiment_id: str, owner: str):
    return SimpleNamespace(
        experiment_id=experiment_id,
        name=f"name-{experiment_id}",
        status=SimpleNamespace(value="created"),
        current_stage="eda",
        target_column="churned",
        created_by=owner,
        best_model=None,
        best_score=None,
        primary_metric=None,
        artifact_prefix=f"{ARTIFACT_ROOT}/ml-factory/experiments/{experiment_id}",
    )


class RecordingExperiments:
    def __init__(self, records, eda=None) -> None:
        self.records = records
        self.eda_report = eda
        self.list_calls = []

    def list(self, *, limit=None, created_by=None):
        self.list_calls.append({"limit": limit, "created_by": created_by})
        return [r for r in self.records if created_by is None or r.created_by == created_by]

    def get(self, experiment_id):
        return next(r for r in self.records if r.experiment_id == experiment_id)

    def eda(self, experiment_id):
        return self.eda_report


# --- catalogue -----------------------------------------------------------------------------


def test_build_registry_discovers_tools_core_and_classifies_each_once():
    assert "backend.agent.tools_core" in catalogue.tool_module_names()
    registry = catalogue.build_registry(SimpleNamespace())
    assert set(registry.names()) >= CORE_TOOLS
    for name in CORE_TOOLS:
        spec = registry.get(name)
        assert spec.tool_class == ToolClass.READ
        assert spec.flags.control == (name in {"submit_plan", "ask_user"})
    assert registry.get("get_experiment").flags.experiment_arg == "experiment_id"
    assert registry.get("get_eda_summary").flags.experiment_arg == "experiment_id"
    names = {d.name for d in registry.definitions()}
    assert names >= CORE_TOOLS


def test_module_without_register_is_a_catalogue_error(monkeypatch):
    module = types.ModuleType("backend.agent.tools_broken_for_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(catalogue, "tool_module_names", lambda: (module.__name__,))
    with pytest.raises(catalogue.CatalogueError, match="tools_broken_for_test"):
        catalogue.build_registry(SimpleNamespace())


# --- control tools -------------------------------------------------------------------------


def test_submit_plan_accepts_1_to_10_steps_of_at_most_200_chars():
    args = tools_core.SubmitPlanArgs.model_validate({"steps": ["x" * 200] * 10})
    out = tools_core.submit_plan(make_ctx(SimpleNamespace()), args)
    assert out.data["step_count"] == 10
    assert tools_core.SubmitPlanArgs.model_validate({"steps": ["  a  "]}).steps == ["a"]


@pytest.mark.parametrize(
    "steps",
    [[], ["x"] * 11, ["x" * 201], ["   "]],
    ids=["none", "eleven", "too-long", "blank"],
)
def test_submit_plan_rejects_bad_plans(steps):
    with pytest.raises(PydanticValidationError):
        tools_core.SubmitPlanArgs.model_validate({"steps": steps})


def test_ask_user_takes_exactly_one_question():
    args = tools_core.AskUserArgs.model_validate({"question": "Which column is the target?"})
    assert tools_core.ask_user(make_ctx(SimpleNamespace()), args).data == {
        "question": "Which column is the target?"
    }
    for bad in ({"question": ["a?", "b?"]}, {"question": ""}, {"question": "a?", "extra": "b?"}):
        with pytest.raises(PydanticValidationError):
            tools_core.AskUserArgs.model_validate(bad)


# --- read tools ----------------------------------------------------------------------------


@pytest.fixture
def datasets(store, price_table_path) -> DatasetService:
    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/r",
        stage_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/x@sha256:" + "a" * 64,
        price_table_path=price_table_path,
    )
    return DatasetService(settings=settings, store=store)


def test_list_datasets_returns_ids_not_uris_and_links_new_refs(datasets):
    churn = datasets.ingest_bytes(filename="churn.csv", payload=b"a,b\n1,2\n")
    other = datasets.ingest_bytes(filename="other.csv", payload=b"c,d\n3,4\n")
    datasets.ingest_bytes(filename="tabbed.tsv", payload=b"a\tb\n1\t2\n")
    known = DatasetRef(
        dataset_id="ds_known",
        uri=churn.dataset_uri,
        filename="churn.csv",
        sha256=None,
        size_bytes=churn.size_bytes,
        file_format="csv",
        columns=[],
        row_count=None,
    )
    ctx = make_ctx(SimpleNamespace(datasets=datasets), make_session([known]))

    out = tools_core.list_datasets(ctx, tools_core.ListDatasetsArgs())

    assert "s3://" not in repr(out.data)
    ids = {d["filename"]: d["dataset_id"] for d in out.data["datasets"]}
    assert ids["churn.csv"] == "ds_known"
    assert ids["other.csv"] == tools_core.dataset_id_for(other.dataset_uri)
    assert out.data["skipped_unsupported_format"] == 1
    assert [(r.dataset_id, r.uri) for r in out.datasets] == [(ids["other.csv"], other.dataset_uri)]


def test_list_experiments_is_owner_filtered_and_links_ids():
    experiments = RecordingExperiments([record("e1", "alice"), record("e2", "bob")])
    ctx = make_ctx(SimpleNamespace(experiments=experiments))

    out = tools_core.list_experiments(ctx, tools_core.ListExperimentsArgs(limit=5))

    assert experiments.list_calls == [{"limit": 5, "created_by": "alice"}]
    assert out.experiment_ids == ("e1",)
    assert [e["experiment_id"] for e in out.data["experiments"]] == ["e1"]
    assert "artifact_prefix" not in out.data["experiments"][0]
    with pytest.raises(PydanticValidationError):
        tools_core.ListExperimentsArgs.model_validate({"user": "bob"})


def test_get_experiment_wraps_service_get_without_prefix():
    ctx = make_ctx(SimpleNamespace(experiments=RecordingExperiments([record("e1", "alice")])))
    out = tools_core.get_experiment(ctx, tools_core.ExperimentArgs(experiment_id="e1"))
    assert out.data["experiment_id"] == "e1"
    assert "artifact_prefix" not in out.data


def test_get_eda_summary_has_no_values_or_uris(classification_eda):
    ctx = make_ctx(SimpleNamespace(experiments=RecordingExperiments([], eda=classification_eda)))
    out = tools_core.get_eda_summary(ctx, tools_core.ExperimentArgs(experiment_id="exp-test"))
    text = repr(out.data)
    assert "memory://" not in text and "source_uri" not in text
    assert all("sample_values" not in c and "categorical" not in c for c in out.data["columns"])
    assert "customer provided" not in text  # free-text values never appear
    assert out.data["dataset"]["row_count"] == classification_eda.dataset.row_count
    assert len(out.data["columns"]) == len(classification_eda.columns)
