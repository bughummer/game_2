"""Report and interpretation tools (design C5 catalogue; Req 3.1, 10.7).

* ``generate_report`` (Mutating): ``ReportingService.build_eda_report`` or
  ``build_comparison_report``, as ``routes/reports.py`` chooses by ``kind``. Returns artifact refs
  (paths relative to the experiment's artifact prefix), never URIs.
* ``get_report`` (Read, ``verbatim``): ``ReportingService.get_report``.
* ``interpret_results`` (Mutating, ``ledger_source=False``): ``ReasoningService.interpret``. Its
  prose does not ground other prose.

The free-form tools live in ``tools_free_form.py``.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.agent.tools import ToolContext, ToolFlags, ToolOutput, ToolRegistry
from ml_engine.contracts.agent import ToolClass
from ml_engine.contracts.report import ReportKind

MAX_QUESTION_CHARS = 1000

ReportKindArg = Literal["eda", "model_comparison"]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentArgs(_Args):
    experiment_id: str = Field(min_length=1, max_length=200)


class ReportArgs(ExperimentArgs):
    kind: ReportKindArg = Field(default="model_comparison", description="eda or model_comparison.")


class InterpretArgs(ExperimentArgs):
    question: (
        Annotated[
            str,
            StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUESTION_CHARS),
        ]
        | None
    ) = Field(default=None, description="Optional question to focus the interpretation on.")


def artifact_ref(prefix: str, uri: str | None) -> str | None:
    """The artifact's path relative to the experiment prefix, so the model never sees a URI.

    A URI outside the prefix gives only its last path segment.
    """
    if uri is None:
        return None
    base = prefix.rstrip("/") + "/"
    if uri.startswith(base):
        return uri[len(base) :]
    return uri.rsplit("/", 1)[-1]


# --- handlers ------------------------------------------------------------------------------


def generate_report(ctx: ToolContext, args: ReportArgs) -> ToolOutput:
    reporting = ctx.services.reporting
    if args.kind == "eda":
        artifacts = reporting.build_eda_report(args.experiment_id)
    else:
        artifacts = reporting.build_comparison_report(args.experiment_id)
    prefix = ctx.services.experiments.get(args.experiment_id).artifact_prefix
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "kind": artifacts.kind.value,
            "json_ref": artifact_ref(prefix, artifacts.json_uri),
            "html_ref": artifact_ref(prefix, artifacts.html_uri),
            "pdf_ref": artifact_ref(prefix, artifacts.pdf_uri),
            "pdf_available": artifacts.pdf_available,
        },
        audit_detail={"kind": args.kind},
    )


def get_report(ctx: ToolContext, args: ReportArgs) -> ToolOutput:
    report = ctx.services.reporting.get_report(args.experiment_id, ReportKind(args.kind))
    if report is None:
        return ToolOutput(
            data={"experiment_id": args.experiment_id, "kind": args.kind, "available": False}
        )
    return ToolOutput(data=report.model_dump(mode="json"))


def interpret_results(ctx: ToolContext, args: InterpretArgs) -> ToolOutput:
    report = ctx.services.reasoning.interpret(args.experiment_id, question=args.question)
    return ToolOutput(data=report.model_dump(mode="json"))


# --- registration --------------------------------------------------------------------------


def register(registry: ToolRegistry, services: Any) -> None:
    del services  # handlers use ctx.services at call time
    registry.register(
        "generate_report",
        classes=[ToolClass.MUTATING],
        description=(
            "Generate a downloadable report (kind eda or model_comparison) as JSON and HTML, "
            "and PDF when available. Returns artifact references, not the report body."
        ),
        input_model=ReportArgs,
        handler=generate_report,
        flags=ToolFlags(experiment_arg="experiment_id", audit_action="generate_report"),
    )
    registry.register(
        "get_report",
        classes=[ToolClass.READ],
        description="Get a generated report document (kind eda or model_comparison).",
        input_model=ReportArgs,
        handler=get_report,
        flags=ToolFlags(experiment_arg="experiment_id", verbatim=True),
    )
    registry.register(
        "interpret_results",
        classes=[ToolClass.MUTATING],
        description=(
            "Interpret a finished experiment (or its EDA before training): summary, suspected "
            "leaks, feature interpretations, data-quality concerns and follow-up experiments. "
            "Advisory prose; quote numbers from the other tools, not from this one."
        ),
        input_model=InterpretArgs,
        handler=interpret_results,
        flags=ToolFlags(
            experiment_arg="experiment_id",
            ledger_source=False,
            audit_action="interpret_results",
        ),
    )
