"""Experiment, leakage and business-context tools (design C5 catalogue; Req 3.1, 3.2, 3.4, 3.5,
5.8).

Each tool calls the service method the REST route (``routes/experiments.py``,
``routes/features.py``) or MCP tool makes, with the same arguments. Mutating and Costly tools set
``flags.audit_action`` and ``ToolOutput.audit_detail`` to the action name and detail keys the
route's ``audit_mutation`` writes.

* ``create_experiment`` (Costly): ``ExperimentService.create``, which submits the EDA job. Its
  ``dataset_id`` argument is resolved to the URI by the shell; the result names the dataset id
  and the EDA execution id, never the URI.
* ``get_leakage_findings`` (Read, advisory): ``ExperimentService.leakage``.
* ``get_feature_review`` (Read): ``ExperimentService.feature_review``.
* ``set_business_context`` (Mutating): ``LeakageService.set_business_context``.
* ``get_business_context`` (Read): ``LeakageService.get_business_context``.
* ``investigate_leakage`` (Mutating, advisory): ``LeakageService.build_pre_training_phase``.
* ``get_leakage_investigation`` (Read): ``LeakageService.get_phase``.

None of these tools changes the feature selection.
"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.agent.tools import AwaitedJobSpec, ToolContext, ToolFlags, ToolOutput, ToolRegistry
from backend.agent.tools_core import _summary, dataset_id_for
from ml_engine.contracts.agent import ToolClass
from ml_engine.contracts.common import RequestedProblemType
from ml_engine.contracts.execution import WorkflowKind
from ml_engine.contracts.leakage_investigation import InvestigationPhase

MAX_NAME_CHARS = 200
MAX_TEXT_CHARS = 2000
MAX_COLUMN_CHARS = 200
MAX_TEMPORAL_COLUMNS = 100

ExperimentId = Annotated[str, StringConstraints(min_length=1, max_length=200)]
ColumnName = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_COLUMN_CHARS)
]
ShortText = Annotated[str, StringConstraints(max_length=MAX_TEXT_CHARS)]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateExperimentArgs(_Args):
    name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NAME_CHARS)
    ]
    dataset_id: str = Field(
        min_length=1, description="A dataset id from list_datasets or an approved upload."
    )
    target_column: ColumnName
    problem_type: RequestedProblemType = RequestedProblemType.AUTO


class ExperimentArgs(_Args):
    experiment_id: ExperimentId


class SetBusinessContextArgs(_Args):
    experiment_id: ExperimentId
    objective: Annotated[str, StringConstraints(min_length=1, max_length=MAX_TEXT_CHARS)]
    prediction_scenario: Annotated[str, StringConstraints(min_length=1, max_length=MAX_TEXT_CHARS)]
    scoring_trigger: ShortText | None = None
    prediction_horizon: ShortText | None = None
    entity_id_column: ColumnName | None = None
    temporal_columns: list[ColumnName] = Field(
        default_factory=list, max_length=MAX_TEMPORAL_COLUMNS
    )
    investigate_leakage: bool = True


class InvestigateLeakageArgs(_Args):
    experiment_id: ExperimentId
    use_llm: bool = True


class LeakageInvestigationArgs(_Args):
    experiment_id: ExperimentId
    phase: InvestigationPhase = InvestigationPhase.PRE_TRAINING


# --- handlers ------------------------------------------------------------------------------


def _dataset_id(ctx: ToolContext, uri: str) -> str:
    for ref in ctx.session.datasets:
        if ref.uri == uri:
            return ref.dataset_id
    return dataset_id_for(uri)


def _file_format(ctx: ToolContext, uri: str) -> str | None:
    for ref in ctx.session.datasets:
        if ref.uri == uri:
            return ref.file_format
    return None


def create_experiment(ctx: ToolContext, args: CreateExperimentArgs) -> ToolOutput:
    # The shell has replaced the dataset id with the session's URI.
    uri = args.dataset_id
    record = ctx.services.experiments.create(
        name=args.name,
        dataset_uri=uri,
        target_column=args.target_column,
        problem_type=args.problem_type,
        file_format=_file_format(ctx, uri),
        user=ctx.principal.user,
    )
    data = _summary(record) | {
        "dataset_id": _dataset_id(ctx, uri),
        "eda_execution_id": record.eda_execution_arn,
    }
    return ToolOutput(
        data=data,
        experiment_ids=(record.experiment_id,),
        awaited=tuple(eda_jobs(data)),
        audit_detail={"dataset_uri": uri, "target_column": args.target_column},
    )


def eda_jobs(data: dict) -> list[AwaitedJobSpec]:
    """``job_extractor`` for ``create_experiment``: the EDA execution it started."""
    execution_id = data.get("eda_execution_id")
    experiment_id = data.get("experiment_id")
    if not execution_id or not experiment_id:
        return []
    return [
        AwaitedJobSpec(
            execution_id=str(execution_id),
            experiment_id=str(experiment_id),
            kind=WorkflowKind.EDA.value,
        )
    ]


def get_leakage_findings(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    report = ctx.services.experiments.leakage(args.experiment_id)
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "target_column": report.target_column,
            "findings": [f.model_dump(mode="json") for f in report.findings],
            "feature_risks": [r.model_dump(mode="json") for r in report.feature_risks],
        }
    )


def get_feature_review(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    return ToolOutput(
        data=ctx.services.experiments.feature_review(args.experiment_id).model_dump(mode="json")
    )


def set_business_context(ctx: ToolContext, args: SetBusinessContextArgs) -> ToolOutput:
    context = ctx.services.leakage.set_business_context(
        args.experiment_id,
        objective=args.objective,
        prediction_scenario=args.prediction_scenario,
        scoring_trigger=args.scoring_trigger,
        prediction_horizon=args.prediction_horizon,
        entity_id_column=args.entity_id_column,
        temporal_columns=list(args.temporal_columns),
        investigate_leakage=args.investigate_leakage,
        user=ctx.principal.user,
    )
    return ToolOutput(data=context.model_dump(mode="json"))


def get_business_context(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    context = ctx.services.leakage.get_business_context(args.experiment_id)
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "business_context": context.model_dump(mode="json") if context else None,
        }
    )


def investigate_leakage(ctx: ToolContext, args: InvestigateLeakageArgs) -> ToolOutput:
    report = ctx.services.leakage.build_pre_training_phase(args.experiment_id, use_llm=args.use_llm)
    return ToolOutput(
        data=report.model_dump(mode="json"),
        audit_detail={"warnings": str(len(report.warnings))},
    )


def get_leakage_investigation(ctx: ToolContext, args: LeakageInvestigationArgs) -> ToolOutput:
    report = ctx.services.leakage.get_phase(args.experiment_id, args.phase)
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "phase": args.phase.value,
            "report": report.model_dump(mode="json") if report else None,
        }
    )


# --- registration --------------------------------------------------------------------------

_EXPERIMENT_READ = ToolFlags(experiment_arg="experiment_id")
_ADVISORY_READ = ToolFlags(experiment_arg="experiment_id", advisory=True)


def register(registry: ToolRegistry, services: Any) -> None:
    del services  # handlers use ctx.services at call time
    registry.register(
        "create_experiment",
        classes=[ToolClass.COSTLY],
        description=(
            "Create an experiment from a dataset id and a target column, and start EDA "
            "(one processing job). Returns the experiment id and the EDA execution id."
        ),
        input_model=CreateExperimentArgs,
        handler=create_experiment,
        flags=ToolFlags(
            dataset_args=("dataset_id",),
            estimator="processing",
            audit_action="create_experiment",
        ),
        job_extractor=eda_jobs,
    )
    registry.register(
        "get_leakage_findings",
        classes=[ToolClass.READ],
        description=(
            "Get the advisory leakage findings and per-feature leakage risks from EDA. "
            "Advisory only: it does not change the feature set."
        ),
        input_model=ExperimentArgs,
        handler=get_leakage_findings,
        flags=_ADVISORY_READ,
    )
    registry.register(
        "get_feature_review",
        classes=[ToolClass.READ],
        description=(
            "Get the feature review table: profile, leakage verdict and recommendation per column."
        ),
        input_model=ExperimentArgs,
        handler=get_feature_review,
        flags=_EXPERIMENT_READ,
    )
    registry.register(
        "set_business_context",
        classes=[ToolClass.MUTATING],
        description=(
            "Set the experiment's business context in plain language: objective and prediction "
            "scenario, optionally scoring trigger, horizon, entity id column and temporal "
            "columns. Leakage investigation stays on unless investigate_leakage is false."
        ),
        input_model=SetBusinessContextArgs,
        handler=set_business_context,
        flags=ToolFlags(experiment_arg="experiment_id", audit_action="set_business_context"),
    )
    registry.register(
        "get_business_context",
        classes=[ToolClass.READ],
        description="Get the experiment's business context, or null when none is set.",
        input_model=ExperimentArgs,
        handler=get_business_context,
        flags=_EXPERIMENT_READ,
    )
    registry.register(
        "investigate_leakage",
        classes=[ToolClass.MUTATING],
        description=(
            "Build or rebuild the pre-training leakage investigation (rules, target association, "
            "name patterns, and LLM suspicion when enabled). Advisory: it records warnings and "
            "keeps earlier decisions, and never changes the feature set. Needs EDA complete."
        ),
        input_model=InvestigateLeakageArgs,
        handler=investigate_leakage,
        flags=ToolFlags(
            experiment_arg="experiment_id", audit_action="investigate_leakage", advisory=True
        ),
    )
    registry.register(
        "get_leakage_investigation",
        classes=[ToolClass.READ],
        description="Get a leakage investigation phase (pre_training or post_training), or null.",
        input_model=LeakageInvestigationArgs,
        handler=get_leakage_investigation,
        flags=_ADVISORY_READ,
    )
