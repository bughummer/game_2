"""Core tools: the control tools and the first Read tools (design C5 catalogue; Req 1.3, 1.5,
3.1, 3.2, 3.5).

Control tools (Read class, ``flags.control``):

* ``submit_plan`` takes 1 to 10 steps of at most 200 characters each and returns them. The loop
  records the plan and shows it to the user.
* ``ask_user`` takes one question and returns it. The loop sets ``waiting_for_user``.

Read tools, each wrapping the service call the REST route and the MCP tool make:

* ``list_datasets``: ``DatasetService.list_uploads``. Returns dataset ids, never URIs, and hands
  the new ``DatasetRef`` entries to the loop through ``ToolOutput.datasets``.
* ``list_experiments``: ``ExperimentService.list(created_by=owner)``. Always owner-filtered; the
  returned ids are linked to the session through ``ToolOutput.experiment_ids``.
* ``get_experiment``: ``ExperimentService.get``.
* ``get_eda_summary``: ``ExperimentService.eda``, without sample values or category values.

Handlers use ``ctx.services`` and ``ctx.principal``; identity never comes from an argument.
"""

import hashlib
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.agent.tools import ToolContext, ToolFlags, ToolOutput, ToolRegistry
from backend.api.views import experiment_summary
from ml_engine.contracts.agent import DatasetRef, ToolClass

MAX_PLAN_STEPS = 10
MAX_STEP_CHARS = 200
MAX_QUESTION_CHARS = 500
MAX_LIST_LIMIT = 100
MAX_EDA_ASSOCIATIONS = 20

#: Formats a session ``DatasetRef`` can hold. Other uploads are counted but not listed.
_LINKABLE_FORMATS = frozenset({"csv", "parquet"})

PlanStepText = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_STEP_CHARS)
]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubmitPlanArgs(_Args):
    steps: list[PlanStepText] = Field(
        min_length=1,
        max_length=MAX_PLAN_STEPS,
        description="1 to 10 plan steps, each at most 200 characters.",
    )


class AskUserArgs(_Args):
    question: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS)
    ] = Field(description="One question naming the missing information.")


class ListDatasetsArgs(_Args):
    limit: int = Field(default=50, ge=1, le=MAX_LIST_LIMIT)


class ListExperimentsArgs(_Args):
    limit: int = Field(default=50, ge=1, le=MAX_LIST_LIMIT)


class ExperimentArgs(_Args):
    experiment_id: str = Field(min_length=1, max_length=200)


def dataset_id_for(uri: str) -> str:
    """Stable id for an uploaded object, so the model never handles its URI."""
    return "ds_" + hashlib.sha256(uri.encode("utf-8")).hexdigest()[:16]


def _summary(record: Any) -> dict[str, Any]:
    """The route's experiment summary without the storage prefix."""
    data = experiment_summary(record)
    data.pop("artifact_prefix", None)
    return data


# --- control tools -------------------------------------------------------------------------


def submit_plan(_ctx: ToolContext, args: SubmitPlanArgs) -> ToolOutput:
    return ToolOutput(data={"steps": list(args.steps), "step_count": len(args.steps)})


def ask_user(_ctx: ToolContext, args: AskUserArgs) -> ToolOutput:
    return ToolOutput(data={"question": args.question})


# --- read tools ----------------------------------------------------------------------------


def list_datasets(ctx: ToolContext, args: ListDatasetsArgs) -> ToolOutput:
    listing = ctx.services.datasets.list_uploads(limit=args.limit)
    known = {d.uri: d for d in ctx.session.datasets}
    items: list[dict[str, Any]] = []
    new_refs: list[DatasetRef] = []
    skipped = 0
    for summary in listing.datasets:
        if summary.file_format not in _LINKABLE_FORMATS:
            skipped += 1
            continue
        existing = known.get(summary.dataset_uri)
        dataset_id = existing.dataset_id if existing else dataset_id_for(summary.dataset_uri)
        if existing is None:
            ref = DatasetRef(
                dataset_id=dataset_id,
                uri=summary.dataset_uri,
                filename=summary.filename,
                sha256=None,
                size_bytes=summary.size_bytes,
                file_format=summary.file_format,
                columns=[],
                row_count=None,
            )
            new_refs.append(ref)
            known[summary.dataset_uri] = ref
        items.append(
            {
                "dataset_id": dataset_id,
                "filename": summary.filename,
                "file_format": summary.file_format,
                "size_bytes": summary.size_bytes,
                "last_modified": (
                    summary.last_modified.isoformat() if summary.last_modified else None
                ),
            }
        )
    data: dict[str, Any] = {"count": len(items), "datasets": items}
    if skipped:
        data["skipped_unsupported_format"] = skipped
    return ToolOutput(data=data, datasets=tuple(new_refs))


def list_experiments(ctx: ToolContext, args: ListExperimentsArgs) -> ToolOutput:
    records = ctx.services.experiments.list(limit=args.limit, created_by=ctx.principal.user)
    summaries = [_summary(r) for r in records]
    return ToolOutput(
        data={"count": len(summaries), "experiments": summaries},
        experiment_ids=tuple(r.experiment_id for r in records),
    )


def get_experiment(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    return ToolOutput(data=_summary(ctx.services.experiments.get(args.experiment_id)))


def get_eda_summary(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    report = ctx.services.experiments.eda(args.experiment_id)
    dataset = report.dataset
    columns = [
        column.model_dump(mode="json", exclude={"sample_values", "categorical"})
        | {
            "top_value_count": (
                len(column.categorical.top_values) if column.categorical is not None else None
            )
        }
        for column in report.columns
    ]
    target = report.target.model_dump(mode="json", exclude={"class_distribution", "positive_class"})
    if report.target.class_distribution is not None:
        target["class_count"] = len(report.target.class_distribution)
    data = {
        "experiment_id": report.experiment_id,
        "generated_at": report.generated_at.isoformat(),
        "dataset": dataset.model_dump(mode="json", exclude={"source_uri"}),
        "target": target,
        "columns": columns,
        "warnings": [
            {"rule": w.rule, "severity": w.severity.value, "message": w.message}
            for w in report.warnings
        ],
        "target_associations": [
            a.model_dump(mode="json") for a in report.target_associations[:MAX_EDA_ASSOCIATIONS]
        ],
    }
    return ToolOutput(data=data)


# --- registration --------------------------------------------------------------------------

_CONTROL = ToolFlags(control=True, ledger_source=False)
_EXPERIMENT_READ = ToolFlags(experiment_arg="experiment_id")


def register(registry: ToolRegistry, services: Any) -> None:
    del services  # handlers use ctx.services at call time
    registry.register(
        "submit_plan",
        classes=[ToolClass.READ],
        description=(
            "Submit the step plan for the goal: 1 to 10 short steps, each at most 200 "
            "characters. Must be the first call."
        ),
        input_model=SubmitPlanArgs,
        handler=submit_plan,
        flags=_CONTROL,
    )
    registry.register(
        "ask_user",
        classes=[ToolClass.READ],
        description="Ask the user one question naming the information that is missing.",
        input_model=AskUserArgs,
        handler=ask_user,
        flags=_CONTROL,
    )
    registry.register(
        "list_datasets",
        classes=[ToolClass.READ],
        description=(
            "List uploaded datasets, newest first, by dataset id, file name and size. "
            "Use the dataset id in other tools."
        ),
        input_model=ListDatasetsArgs,
        handler=list_datasets,
    )
    registry.register(
        "list_experiments",
        classes=[ToolClass.READ],
        description="List your experiments, newest first, with their status.",
        input_model=ListExperimentsArgs,
        handler=list_experiments,
    )
    registry.register(
        "get_experiment",
        classes=[ToolClass.READ],
        description="Get one experiment's status, target and best model.",
        input_model=ExperimentArgs,
        handler=get_experiment,
        flags=_EXPERIMENT_READ,
    )
    registry.register(
        "get_eda_summary",
        classes=[ToolClass.READ],
        description=(
            "Get the EDA summary for an experiment: row and column counts, column profiles, "
            "target analysis, warnings and target associations."
        ),
        input_model=ExperimentArgs,
        handler=get_eda_summary,
        flags=_EXPERIMENT_READ,
    )
