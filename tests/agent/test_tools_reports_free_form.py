"""Unit tests for the report and free-form tools (task 10.4).

Offline: in-memory store, inline orchestrator, the real services.
"""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError as PydanticValidationError

from backend.agent import catalogue, tools_free_form, tools_reports
from backend.agent.tools import ToolContext
from backend.auth import Principal
from backend.config import Settings
from backend.errors import NotFoundError
from backend.repositories import ExperimentRepository
from backend.services import ExperimentService, FreeFormService, ModelVersionService
from backend.services.reporting import ReportingService
from ml_engine.contracts.agent import Session, ToolClass
from ml_engine.contracts.execution import WorkflowKind
from ml_engine.contracts.free_form import FreeFormRequest, content_hash
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.golden import golden_frame
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

REPORT_TOOLS = {"generate_report", "get_report", "interpret_results"}
FREE_FORM_TOOLS = {
    "draft_free_form_script",
    "start_free_form_run",
    "get_free_form_run",
    "list_free_form_runs",
    "compare_free_form",
}

ANSWERS = [
    {"question_id": "time_ordered", "status": "answered", "answer": "yes, signup_date"},
    {"question_id": "entity_repeats", "status": "dont_know"},
]


def _services():
    store = InMemoryObjectStore()
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
    services = SimpleNamespace(
        store=store,
        experiments=experiments,
        free_form=free_form,
        model_versions=ModelVersionService(settings=settings, repository=repository, store=store),
        reporting=ReportingService(settings=settings, repository=repository, store=store),
    )
    uri = write_dataset(store, "data.csv", golden_frame())
    record = experiments.create(name="ff", dataset_uri=uri, target_column="churned")
    return services, record, uri


def _ctx(services, experiment_id) -> ToolContext:
    session = Session.model_construct(
        session_id="s1", owner="alice", experiment_ids=[experiment_id], datasets=[]
    )
    return ToolContext(session=session, principal=Principal(user="alice"), services=services)


def _draft(ctx, experiment_id, uri, **extra):
    # The shell replaces dataset_id with the session's URI before the handler runs.
    args = tools_free_form.DraftScriptArgs.model_validate(
        {
            "experiment_id": experiment_id,
            "dataset_id": "ds_x",
            "description": "Forecast monthly churn.",
            "data_context": ANSWERS,
            **extra,
        }
    ).model_copy(update={"dataset_id": uri})
    return tools_free_form.draft_free_form_script(ctx, args)


# --- registration --------------------------------------------------------------------------


def test_catalogue_registers_report_and_free_form_tools_with_their_flags():
    registry = catalogue.build_registry(SimpleNamespace())
    assert set(registry.names()) >= REPORT_TOOLS | FREE_FORM_TOOLS
    classes = {n: registry.get(n).tool_class for n in REPORT_TOOLS | FREE_FORM_TOOLS}
    assert classes["generate_report"] is ToolClass.MUTATING
    assert classes["interpret_results"] is ToolClass.MUTATING
    assert classes["draft_free_form_script"] is ToolClass.MUTATING
    assert classes["start_free_form_run"] is ToolClass.COSTLY
    for name in ("get_report", "get_free_form_run", "list_free_form_runs", "compare_free_form"):
        assert classes[name] is ToolClass.READ

    assert registry.get("interpret_results").flags.ledger_source is False
    assert registry.get("get_report").flags.verbatim
    assert registry.get("compare_free_form").flags.verbatim

    start = registry.get("start_free_form_run")
    assert start.flags.always_approve
    assert start.flags.estimator == "free_form"
    assert start.flags.audit_action == "start_free_form"
    assert start.protected is not None and start.job_extractor is not None
    assert set(start.input_model.model_fields) == {
        "experiment_id",
        "draft_id",
        "revision",
        "requirements",
        "use_spot",
    }
    assert registry.get("draft_free_form_script").flags.dataset_args == ("dataset_id",)


def test_start_args_never_take_script_text():
    with pytest.raises(PydanticValidationError):
        tools_free_form.StartRunArgs.model_validate(
            {"experiment_id": "e", "draft_id": "draft-a", "revision": 1, "script": "x"}
        )
    with pytest.raises(PydanticValidationError):
        tools_free_form.StartRunArgs.model_validate(
            {"experiment_id": "e", "draft_id": "../x", "revision": 1}
        )


# --- free-form -----------------------------------------------------------------------------


def test_draft_then_start_runs_the_stored_bytes_and_labels_pass_through():
    services, record, uri = _services()
    ctx = _ctx(services, record.experiment_id)

    drafted = _draft(ctx, record.experiment_id, uri)
    draft_id = drafted.data["draft_id"]
    stored = drafted.data["revision"]
    assert drafted.audit_detail == {"draft_id": draft_id, "revision": "1"}
    assert stored["content_hash"] == content_hash(stored["content"])

    args = tools_free_form.StartRunArgs(
        experiment_id=record.experiment_id,
        draft_id=draft_id,
        revision=1,
        requirements=["numpy"],
        use_spot=True,
    )
    preview = tools_free_form.free_form_script_for_approval(ctx, args)
    assert preview["content"] == stored["content"]
    assert preview["content_hash"] == stored["content_hash"]
    assert tools_free_form.start_run_protected(ctx, args) is True

    out = tools_free_form.start_free_form_run(ctx, args)

    assert out.data["status"] == "completed"
    assert out.data["content_hash"] == stored["content_hash"]
    assert out.data["result"]["metrics"][0]["label"] == "self_reported"
    assert "memory://" not in repr(out.data) and "_uri" not in repr(out.data)
    assert out.audit_detail["content_hash"] == stored["content_hash"]
    assert set(out.audit_detail) == {"run_id", "content_hash", "dataset_uri"}
    jobs = tools_free_form.free_form_job_extractor(out.data)
    assert [(j.experiment_id, j.kind) for j in jobs] == [
        (record.experiment_id, WorkflowKind.FREE_FORM.value)
    ]

    layout = ExperimentLayout(base=record.artifact_prefix)
    request = read_model(
        services.store, layout.free_form_request(out.data["run_id"]), FreeFormRequest
    )
    assert request.approved_script.content_hash == stored["content_hash"]
    assert request.approved_script.approved_by == "alice"
    assert request.compute.use_spot is True
    assert request.requirements.packages == ["numpy"]

    run_args = tools_free_form.RunArgs(experiment_id=record.experiment_id)
    got = tools_free_form.get_free_form_run(ctx, run_args)
    assert got.data["run_id"] == out.data["run_id"]
    listed = tools_free_form.list_free_form_runs(
        ctx, tools_free_form.ExperimentArgs(experiment_id=record.experiment_id)
    )
    assert [r["run_id"] for r in listed.data["runs"]] == [out.data["run_id"]]
    compared = tools_free_form.compare_free_form(
        ctx, tools_free_form.ExperimentArgs(experiment_id=record.experiment_id)
    )
    entries = [e for e in compared.data["entries"] if e["tier"] == "free_form"]
    assert entries and all(e["comparable"] is (e["metric_label"] == "engine") for e in entries)


def test_unknown_draft_revision_is_not_found_and_nothing_runs():
    services, record, uri = _services()
    ctx = _ctx(services, record.experiment_id)
    draft_id = _draft(ctx, record.experiment_id, uri).data["draft_id"]
    for bad in (
        {"draft_id": draft_id, "revision": 2},
        {"draft_id": "draft-missing", "revision": 1},
    ):
        args = tools_free_form.StartRunArgs(experiment_id=record.experiment_id, **bad)
        with pytest.raises(NotFoundError):
            tools_free_form.start_free_form_run(ctx, args)
    assert services.free_form.list_runs(record.experiment_id) == []


def test_draft_revision_accessor_returns_each_retained_revision():
    services, record, uri = _services()
    ctx = _ctx(services, record.experiment_id)
    draft_id = _draft(ctx, record.experiment_id, uri).data["draft_id"]
    _draft(ctx, record.experiment_id, uri, draft_id=draft_id, instruction="Use a lag feature.")
    draft, second = services.free_form.draft_revision(record.experiment_id, draft_id, 2)
    _, first = services.free_form.draft_revision(record.experiment_id, draft_id, 1)
    assert draft.dataset_uri == uri
    assert (first.revision, second.revision) == (1, 2)
    assert second.instruction == "Use a lag feature."


# --- reports -------------------------------------------------------------------------------


def test_generate_report_returns_artifact_refs_not_uris_and_get_report_reads_it():
    services, record, _uri = _services()
    ctx = _ctx(services, record.experiment_id)
    args = tools_reports.ReportArgs(experiment_id=record.experiment_id, kind="eda")

    assert tools_reports.get_report(ctx, args).data["available"] is False
    out = tools_reports.generate_report(ctx, args)

    assert "://" not in repr(out.data)
    assert out.data["json_ref"].endswith(".json") and out.data["html_ref"].endswith(".html")
    assert out.audit_detail == {"kind": "eda"}
    report = tools_reports.get_report(ctx, args).data
    assert report["kind"] == "eda" and report["experiment_id"] == record.experiment_id


def test_artifact_ref_is_relative_to_the_prefix():
    assert tools_reports.artifact_ref("memory://b/e1", "memory://b/e1/reports/x.json") == (
        "reports/x.json"
    )
    assert tools_reports.artifact_ref("memory://b/e1", "memory://other/y.html") == "y.html"
    assert tools_reports.artifact_ref("memory://b/e1", None) is None
