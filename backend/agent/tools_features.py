"""Feature tools: derived-feature proposals and the feature decision (design C5 catalogue;
Req 3.1, 5.7, 5.8).

Each tool wraps the same service call as ``backend/api/routes/features.py`` and the MCP tool of
the same name, and names the same experiment audit action with the same detail keys:

* ``propose_features`` (Mutating, advisory): ``FeatureProposalService.generate``. Writes
  candidates only; approvals and the feature selection are untouched.
* ``get_feature_proposals`` (Read): ``FeatureProposalService.stored``.
* ``define_derived_features`` (Costly, ``changes_config``): ``FeatureProposalService.define``.
  With ``approve`` true it approves the accepted features and starts a profiling job.
* ``approve_features`` (Costly, ``changes_config``): ``FeatureProposalService.approve``. Starts a
  profiling job when something is approved.
* ``decide_features`` (Mutating, ``changes_config``): ``ExperimentService.save_features``.
  ``protected`` unless the selection keeps every candidate column; any error while evaluating
  that means protected (Req 5.7, 5.8).

Leakage and proposal tools never change the feature set: only ``decide_features`` writes the
selection. Ownership and the experiment link are checked by the shell (``experiment_arg``).
"""

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from backend.agent.tools import (
    AwaitedJobSpec,
    ToolContext,
    ToolFlags,
    ToolOutput,
    ToolRegistry,
)
from backend.schemas import ApproveProposalsRequest, ProposeFeaturesRequest
from ml_engine.contracts.agent import ToolClass
from ml_engine.contracts.execution import WorkflowKind
from ml_engine.contracts.proposals import FeatureProposal

MAX_HINT_CHARS = 2000
MAX_DEFINED_FEATURES = 20
MAX_FEATURE_NAMES = 2000
MAX_NAME_CHARS = 200
MAX_REASON_CHARS = 500

#: Estimator key for the derived-feature profiling job (``backend/agent/estimate.py``).
PROFILING_ESTIMATOR = "processing"

ExperimentId = Annotated[str, StringConstraints(min_length=1, max_length=200)]
FeatureName = Annotated[str, StringConstraints(min_length=1, max_length=MAX_NAME_CHARS)]
ReasonText = Annotated[str, StringConstraints(max_length=MAX_REASON_CHARS)]


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentArgs(_Args):
    experiment_id: ExperimentId


class ProposeFeaturesArgs(_Args):
    experiment_id: ExperimentId
    target_definition: str | None = Field(
        default=None,
        max_length=MAX_HINT_CHARS,
        description="What the target means in business terms.",
    )
    prediction_timing: str | None = Field(
        default=None,
        max_length=MAX_HINT_CHARS,
        description="When the prediction is made relative to the outcome.",
    )


class DefineDerivedFeaturesArgs(_Args):
    experiment_id: ExperimentId
    features: list[FeatureProposal] = Field(
        min_length=1,
        max_length=MAX_DEFINED_FEATURES,
        description="Derived feature specifications, one operation each.",
    )
    approve: bool = Field(
        default=True, description="Approve the accepted features, which starts profiling."
    )


class ApproveFeaturesArgs(_Args):
    experiment_id: ExperimentId
    approved_names: list[FeatureName] = Field(
        max_length=MAX_FEATURE_NAMES,
        description="Names of proposed features to build. An empty list means none.",
    )


class DecideFeaturesArgs(_Args):
    experiment_id: ExperimentId
    selected_features: list[FeatureName] = Field(
        min_length=1,
        max_length=MAX_FEATURE_NAMES,
        description="The features to keep. Every other candidate column is excluded.",
    )
    exclusion_reasons: dict[FeatureName, ReasonText] | None = Field(
        default=None, description="Optional reason per excluded feature."
    )


# --- handlers ------------------------------------------------------------------------------


def propose_features(ctx: ToolContext, args: ProposeFeaturesArgs) -> ToolOutput:
    report = ctx.services.proposals.generate(
        args.experiment_id,
        ProposeFeaturesRequest(
            target_definition=args.target_definition, prediction_timing=args.prediction_timing
        ),
    )
    return ToolOutput(
        data=report.model_dump(mode="json"),
        audit_detail={"candidates": str(len(report.candidates))},
    )


def get_feature_proposals(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    report = ctx.services.proposals.stored(args.experiment_id)
    if report is None:
        return ToolOutput(data={"experiment_id": args.experiment_id, "proposals": None})
    return ToolOutput(data=report.model_dump(mode="json"))


def define_derived_features(ctx: ToolContext, args: DefineDerivedFeaturesArgs) -> ToolOutput:
    result = ctx.services.proposals.define(
        args.experiment_id, list(args.features), approve=args.approve, user=ctx.principal.user
    )
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "added": result.added,
            "approved_names": result.report.approved_names,
            "rejected": [r.model_dump(mode="json") for r in result.rejected],
            "profiling_execution_id": result.profiling_execution_id,
        },
        awaited=_profiling_jobs(args.experiment_id, result.profiling_execution_id),
        audit_detail={"added": ",".join(result.added), "refused": str(len(result.rejected))},
    )


def approve_features(ctx: ToolContext, args: ApproveFeaturesArgs) -> ToolOutput:
    result = ctx.services.proposals.approve(
        args.experiment_id,
        ApproveProposalsRequest(approved_names=list(args.approved_names)),
        user=ctx.principal.user,
    )
    report = result.report
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "approved_names": report.approved_names,
            "candidate_count": len(report.candidates),
            "profiling_execution_id": result.profiling_execution_id,
        },
        awaited=_profiling_jobs(args.experiment_id, result.profiling_execution_id),
        audit_detail={"approved": str(len(report.approved_names))},
    )


def decide_features(ctx: ToolContext, args: DecideFeaturesArgs) -> ToolOutput:
    selection = ctx.services.experiments.save_features(
        args.experiment_id,
        selected_features=list(args.selected_features),
        exclusion_reasons=args.exclusion_reasons,
        user=ctx.principal.user,
    )
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "selected_count": len(selection.selected_features),
            "excluded_count": len(selection.excluded_features),
        },
        audit_detail={"selected": str(len(selection.selected_features))},
    )


# --- protection and jobs -------------------------------------------------------------------


def decide_features_protected(ctx: ToolContext, args: DecideFeaturesArgs) -> bool:
    """True unless the selection keeps every candidate column. Fails closed.

    The candidates are the feature review's rows: every dataset column except the target, plus
    every approved derived feature. This is the set ``save_features`` excludes from.
    """
    try:
        review = ctx.services.experiments.feature_review(args.experiment_id)
        candidates = {item.feature for item in review.features}
        if not candidates:
            return True
        return not candidates.issubset(set(args.selected_features))
    except Exception:
        return True


def _profiling_jobs(experiment_id: str, execution_id: str | None) -> tuple[AwaitedJobSpec, ...]:
    if not execution_id:
        return ()
    return (
        AwaitedJobSpec(
            execution_id=execution_id,
            experiment_id=experiment_id,
            kind=WorkflowKind.DERIVED_PROFILING.value,
        ),
    )


def profiling_job_extractor(data: dict[str, Any]) -> list[AwaitedJobSpec]:
    """The profiling job a define or approve result started, if any."""
    execution_id = data.get("profiling_execution_id")
    experiment_id = data.get("experiment_id")
    if not isinstance(execution_id, str) or not isinstance(experiment_id, str):
        return []
    return list(_profiling_jobs(experiment_id, execution_id))


# --- registration --------------------------------------------------------------------------


def register(registry: ToolRegistry, services: Any) -> None:
    del services  # handlers use ctx.services at call time
    registry.register(
        "propose_features",
        classes=[ToolClass.MUTATING],
        description=(
            "Propose derived features (ratio, difference, date_difference, map_categories, "
            "is_missing) for an experiment. Advisory: candidates must be approved before use. "
            "Needs EDA complete and the reasoning model enabled."
        ),
        input_model=ProposeFeaturesArgs,
        handler=propose_features,
        flags=ToolFlags(
            experiment_arg="experiment_id", audit_action="propose_features", advisory=True
        ),
    )
    registry.register(
        "get_feature_proposals",
        classes=[ToolClass.READ],
        description="Get the derived-feature proposals: candidates, refusals and approvals.",
        input_model=ExperimentArgs,
        handler=get_feature_proposals,
        flags=ToolFlags(experiment_arg="experiment_id", advisory=True),
    )
    registry.register(
        "define_derived_features",
        classes=[ToolClass.COSTLY],
        description=(
            "Define derived features exactly as the user describes them. Each feature is one "
            "operation: ratio (numerator/denominator), difference (left-right), "
            "date_difference (start, end, unit days|months|years), map_categories (column, "
            "mapping, default) or is_missing (column). Use only dataset columns, never the "
            "target. Ask the user whether every input is known at prediction time and why the "
            "feature should help. Accepted features are approved by default, which starts a "
            "profiling job; refused ones come back with the reason."
        ),
        input_model=DefineDerivedFeaturesArgs,
        handler=define_derived_features,
        flags=ToolFlags(
            changes_config=True,
            experiment_arg="experiment_id",
            estimator=PROFILING_ESTIMATOR,
            audit_action="define_derived_features",
        ),
        job_extractor=profiling_job_extractor,
    )
    registry.register(
        "approve_features",
        classes=[ToolClass.COSTLY],
        description=(
            "Approve which proposed derived features to build. An empty list means none. "
            "Starts a profiling job; the stats appear in the feature review when it finishes."
        ),
        input_model=ApproveFeaturesArgs,
        handler=approve_features,
        flags=ToolFlags(
            changes_config=True,
            experiment_arg="experiment_id",
            estimator=PROFILING_ESTIMATOR,
            audit_action="approve_features",
        ),
        job_extractor=profiling_job_extractor,
    )
    registry.register(
        "decide_features",
        classes=[ToolClass.MUTATING],
        description=(
            "Save the feature keep/exclude decision. selected_features is the exact list to "
            "keep; every other candidate column is excluded. Excluding any column needs the "
            "user's approval."
        ),
        input_model=DecideFeaturesArgs,
        handler=decide_features,
        flags=ToolFlags(
            changes_config=True, experiment_arg="experiment_id", audit_action="decide_features"
        ),
        protected=decide_features_protected,
    )
