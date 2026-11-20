"""Training, evaluation and run-status tools (design C5 catalogue; Req 3.1, 3.5, 3.7, 6.3).

Each tool wraps the service call the REST route makes:

* Costly: ``start_training`` (``changes_config``), ``rerun_training``, ``start_validation``,
  ``start_holdout_evaluation`` and ``start_ablation``. Each starts a job and has a
  ``job_extractor`` that reports its ``execution_id`` so the loop can wait for it.
* Read: ``preflight_check``, ``get_training_status``, ``get_execution_status``,
  ``get_comparison`` (verbatim), ``get_threshold_analysis``, ``list_model_versions``,
  ``get_validation`` (verbatim), ``get_holdout_evaluation`` and ``get_ablation``.

``start_training`` builds the ``TrainingRequest`` from the same fields, with the same defaults, as
``backend/api/routes/training.py``. The routes expose no instance-type or Spot option, so neither
do these tools: every job goes through the orchestrator's guards (Managed Spot included).
Validation and holdout take a session ``dataset_id``; the shell swaps it for the URI.

No promote, delete or download tool is registered here (Req 3.5 to 3.7). Values that are storage
URIs are dropped from what the model sees.
"""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.agent.tools import AwaitedJobSpec, ToolContext, ToolFlags, ToolOutput, ToolRegistry
from backend.api.views import (
    grid_point_count,
    holdout_summary,
    resolved_ensembles,
    resolved_evaluation,
    resolved_treatment,
)
from backend.schemas import EnsembleRequest, TrainingRequest
from ml_engine.contracts.agent import ToolClass
from ml_engine.contracts.common import ClassWeighting
from ml_engine.contracts.execution import WorkflowKind
from ml_engine.contracts.treatment import OutlierDetector, OutlierHandling

_ID = Field(min_length=1, max_length=200)
_STORAGE_SCHEME = "s3://"

ParamValue = float | int | str | bool | None


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ExperimentArgs(_Args):
    experiment_id: str = _ID


class StartTrainingArgs(_Args):
    """The route's request body, field for field."""

    experiment_id: str = _ID
    models: list[str] = Field(default_factory=list, description="Empty means every model.")
    primary_metric: str | None = None
    problem_type: str | None = None
    class_weighting: ClassWeighting = ClassWeighting.AUTO
    model_parameters: dict[str, dict[str, ParamValue]] = Field(default_factory=dict)
    split_strategy: str | None = Field(
        default=None, description="random | stratified_random | group | temporal."
    )
    group_column: str | None = None
    time_column: str | None = None
    evaluation_mode: Literal["single_split", "cv"] | None = None
    cv_folds: int | None = None
    test_fold: bool = False
    test_fraction: float | None = Field(default=None, gt=0.0, lt=0.5)
    outlier_detector: OutlierDetector | None = None
    outlier_handling: OutlierHandling | None = None
    zscore_threshold: float | None = Field(default=None, gt=0.0)
    iqr_multiplier: float | None = Field(default=None, gt=0.0)
    isolation_contamination: float | None = Field(default=None, gt=0.0, lt=0.5)
    ensembles: list[EnsembleRequest] = Field(default_factory=list)


class ExecutionStatusArgs(ExperimentArgs):
    execution_id: str | None = Field(default=None, min_length=1, max_length=300)


class ThresholdArgs(ExperimentArgs):
    model_name: str | None = Field(default=None, min_length=1, max_length=200)


class StartValidationArgs(ExperimentArgs):
    dataset_id: str = Field(min_length=1, max_length=200, description="A session dataset id.")
    version_ids: list[str] = Field(
        default_factory=list, description="Model versions to validate; empty means all."
    )


class ValidationArgs(ExperimentArgs):
    validation_id: str | None = Field(default=None, min_length=1, max_length=200)


class StartHoldoutArgs(ExperimentArgs):
    dataset_id: str = Field(min_length=1, max_length=200, description="A session dataset id.")


class HoldoutArgs(ExperimentArgs):
    evaluation_id: str | None = Field(default=None, min_length=1, max_length=200)


class StartAblationArgs(ExperimentArgs):
    features: list[str] | None = Field(
        default=None, description="Features to drop one at a time; omit to use the candidates."
    )
    model_name: str | None = Field(default=None, min_length=1, max_length=200)


# --- helpers -------------------------------------------------------------------------------


def _redact(value: Any) -> Any:
    """Drop every entry whose value is a storage URI, at any depth."""
    if isinstance(value, dict):
        return {
            k: _redact(v)
            for k, v in value.items()
            if not (isinstance(v, str) and v.startswith(_STORAGE_SCHEME))
        }
    if isinstance(value, list):
        return [
            _redact(v) for v in value if not (isinstance(v, str) and v.startswith(_STORAGE_SCHEME))
        ]
    return value


def _file_format(ctx: ToolContext, uri: str) -> str | None:
    """The format recorded for the session dataset now held in ``uri``."""
    for ref in getattr(ctx.session, "datasets", ()) or ():
        if ref.uri == uri:
            return ref.file_format
    return None


def _job_extractor(kind: WorkflowKind):
    def extract(data: dict) -> list[AwaitedJobSpec]:
        execution_id = data.get("execution_id")
        experiment_id = data.get("experiment_id")
        if not isinstance(execution_id, str) or not execution_id:
            return []
        if not isinstance(experiment_id, str) or not experiment_id:
            return []
        return [AwaitedJobSpec(execution_id, experiment_id, kind.value)]

    return extract


# --- costly tools --------------------------------------------------------------------------


def start_training(ctx: ToolContext, args: StartTrainingArgs) -> ToolOutput:
    services = ctx.services
    experiment_id = args.experiment_id
    preflight = services.analysis.preflight(experiment_id)
    updated = services.experiments.start_training(
        experiment_id,
        TrainingRequest(
            models=args.models or [],
            primary_metric=args.primary_metric,
            problem_type=args.problem_type,
            class_weighting=ClassWeighting(args.class_weighting),
            model_parameters=args.model_parameters or {},
            split_strategy=args.split_strategy,
            group_column=args.group_column,
            time_column=args.time_column,
            evaluation_mode=args.evaluation_mode,
            cv_folds=args.cv_folds,
            test_fold=args.test_fold,
            test_fraction=args.test_fraction,
            outlier_detector=args.outlier_detector,
            outlier_handling=args.outlier_handling,
            zscore_threshold=args.zscore_threshold,
            iqr_multiplier=args.iqr_multiplier,
            isolation_contamination=args.isolation_contamination,
            ensembles=args.ensembles or [],
        ),
        user=ctx.principal.user,
    )
    prefix = updated.artifact_prefix
    resolved = resolved_evaluation(services.store, prefix)
    data = {
        "experiment_id": experiment_id,
        "status": updated.status.value,
        "execution_id": updated.training_execution_arn,
        "requested_models": updated.requested_models,
        "class_weighting": ClassWeighting(args.class_weighting).value,
        "model_parameters": args.model_parameters,
        "evaluation_mode": resolved["evaluation_mode"],
        "cv_folds": resolved["cv_folds"],
        "cv_scheme": resolved["cv_scheme"],
        "test_fraction": resolved["test_fraction"],
        "grid_points": grid_point_count(services.store, prefix),
        "treatment": resolved_treatment(services.store, prefix),
        "ensembles": resolved_ensembles(services.store, prefix),
        "preflight_warnings": preflight.warnings,
        "unresolved_leakage_warnings": preflight.unresolved_leakage_warnings,
    }
    detail = {
        "models": ",".join(updated.requested_models),
        "evaluation_mode": resolved["evaluation_mode"],
        "cv_folds": str(resolved["cv_folds"]),
        "cv_scheme": str(resolved["cv_scheme"]),
        "test_fraction": str(resolved["test_fraction"]),
    }
    return ToolOutput(data=data, audit_detail=detail)


def rerun_training(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    updated = ctx.services.experiments.rerun_training(args.experiment_id, user=ctx.principal.user)
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "status": updated.status.value,
            "execution_id": updated.training_execution_arn,
        }
    )


def start_validation(ctx: ToolContext, args: StartValidationArgs) -> ToolOutput:
    uri = args.dataset_id  # the shell has replaced the dataset id with its URI
    status = ctx.services.validation.start_validation(
        args.experiment_id,
        dataset_uri=uri,
        version_ids=args.version_ids,
        file_format=_file_format(ctx, uri),
        user=ctx.principal.user,
    )
    detail = {
        "validation_id": status["validation_id"],
        "dataset_uri": uri,
        "version_ids": ",".join(args.version_ids or []),
    }
    data = {"experiment_id": args.experiment_id, **_redact(status)}
    return ToolOutput(data=data, audit_detail=detail)


def start_holdout_evaluation(ctx: ToolContext, args: StartHoldoutArgs) -> ToolOutput:
    uri = args.dataset_id  # the shell has replaced the dataset id with its URI
    status = ctx.services.experiments.start_holdout_evaluation(
        args.experiment_id,
        dataset_uri=uri,
        file_format=_file_format(ctx, uri),
        user=ctx.principal.user,
    )
    detail = {"evaluation_id": status.evaluation_id, "dataset_uri": uri}
    data = {
        "experiment_id": args.experiment_id,
        "execution_id": status.execution_id,
        **_redact(holdout_summary(status)),
    }
    return ToolOutput(data=data, audit_detail=detail)


def start_ablation(ctx: ToolContext, args: StartAblationArgs) -> ToolOutput:
    started = ctx.services.analysis.start_ablation(
        args.experiment_id, features=args.features, model_name=args.model_name
    )
    return ToolOutput(
        data=_redact(started), audit_detail={"features": ",".join(started["features"])}
    )


# --- read tools ----------------------------------------------------------------------------


def preflight_check(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    return ToolOutput(
        data=ctx.services.analysis.preflight(args.experiment_id).model_dump(mode="json")
    )


def get_training_status(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    status = ctx.services.experiments.training_status(args.experiment_id)
    return ToolOutput(data=status.model_dump(mode="json"))


def get_execution_status(ctx: ToolContext, args: ExecutionStatusArgs) -> ToolOutput:
    view = ctx.services.analysis.execution_status(args.experiment_id, args.execution_id)
    return ToolOutput(data=view.model_dump(mode="json"))


def get_comparison(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    report = ctx.services.experiments.comparison(args.experiment_id)
    return ToolOutput(data=_redact(report.model_dump(mode="json")))


def get_threshold_analysis(ctx: ToolContext, args: ThresholdArgs) -> ToolOutput:
    analyses = ctx.services.experiments.thresholds(args.experiment_id, args.model_name)
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "models": [_redact(a.model_dump(mode="json")) for a in analyses],
        }
    )


def list_model_versions(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    versions = ctx.services.model_versions.list_versions(args.experiment_id)
    return ToolOutput(
        data={
            "experiment_id": args.experiment_id,
            "count": len(versions),
            "model_versions": [_redact(v.model_dump(mode="json")) for v in versions],
        }
    )


def get_validation(ctx: ToolContext, args: ValidationArgs) -> ToolOutput:
    result = ctx.services.validation.validation(args.experiment_id, args.validation_id)
    return ToolOutput(data=_redact(result))


def get_holdout_evaluation(ctx: ToolContext, args: HoldoutArgs) -> ToolOutput:
    status = ctx.services.experiments.holdout_evaluation(args.experiment_id, args.evaluation_id)
    return ToolOutput(data=_redact(holdout_summary(status)))


def get_ablation(ctx: ToolContext, args: ExperimentArgs) -> ToolOutput:
    return ToolOutput(data=_redact(ctx.services.analysis.ablation(args.experiment_id)))


# --- registration --------------------------------------------------------------------------

_EXPERIMENT = "experiment_id"
_READ = ToolFlags(experiment_arg=_EXPERIMENT)
_VERBATIM = ToolFlags(experiment_arg=_EXPERIMENT, verbatim=True)


def register(registry: ToolRegistry, services: Any) -> None:
    del services  # handlers use ctx.services at call time

    def costly(name, description, model, handler, kind, *, audit, estimator, **flags):
        registry.register(
            name,
            classes=[ToolClass.COSTLY],
            description=description,
            input_model=model,
            handler=handler,
            flags=ToolFlags(
                experiment_arg=_EXPERIMENT, estimator=estimator, audit_action=audit, **flags
            ),
            job_extractor=_job_extractor(kind),
        )

    def read(name, description, model, handler, flags=_READ):
        registry.register(
            name,
            classes=[ToolClass.READ],
            description=description,
            input_model=model,
            handler=handler,
            flags=flags,
        )

    read(
        "preflight_check",
        "Check an experiment before training: blockers, warnings, unresolved leakage warnings "
        "and the suggested split. Only blockers stop training.",
        ExperimentArgs,
        preflight_check,
    )
    costly(
        "start_training",
        "Start training on SageMaker from the experiment's approved features. Choose models, "
        "metric, class weighting, split, evaluation mode (single_split or cv), Test_Fold, "
        "outlier treatment and ensembles. Returns the execution id and the resolved settings.",
        StartTrainingArgs,
        start_training,
        WorkflowKind.TRAINING,
        audit="start_training",
        estimator="training",
        changes_config=True,
    )
    costly(
        "rerun_training",
        "Train again from the experiment's frozen configuration, unchanged.",
        ExperimentArgs,
        rerun_training,
        WorkflowKind.TRAINING,
        audit="rerun_training",
        estimator="training",
    )
    read(
        "get_training_status",
        "Get per-model training progress, the best model and its score.",
        ExperimentArgs,
        get_training_status,
    )
    read(
        "get_execution_status",
        "Get the status of one execution (latest when no id is given), with its jobs and any "
        "failure diagnosis.",
        ExecutionStatusArgs,
        get_execution_status,
    )
    read(
        "get_comparison",
        "Get the model comparison table for the latest training run.",
        ExperimentArgs,
        get_comparison,
        ToolFlags(experiment_arg=_EXPERIMENT, verbatim=True, advisory=True),
    )
    read(
        "get_threshold_analysis",
        "Get the decision-threshold analysis for one model or all models.",
        ThresholdArgs,
        get_threshold_analysis,
    )
    read(
        "list_model_versions",
        "List the experiment's model versions, current run first.",
        ExperimentArgs,
        list_model_versions,
    )
    costly(
        "start_validation",
        "Validate model versions (empty means all) on a labeled session dataset. Starts a job.",
        StartValidationArgs,
        start_validation,
        WorkflowKind.VALIDATION,
        audit="start_validation",
        estimator="processing",
        dataset_args=("dataset_id",),
    )
    read(
        "get_validation",
        "Get one validation (latest when no id is given) and its report when finished.",
        ValidationArgs,
        get_validation,
        _VERBATIM,
    )
    costly(
        "start_holdout_evaluation",
        "Score the trained models on another session dataset with the target column. Starts a job.",
        StartHoldoutArgs,
        start_holdout_evaluation,
        WorkflowKind.HOLDOUT_EVALUATION,
        audit="start_holdout_evaluation",
        estimator="processing",
        dataset_args=("dataset_id",),
    )
    read(
        "get_holdout_evaluation",
        "Get one holdout evaluation (latest when no id is given) and its headline numbers.",
        HoldoutArgs,
        get_holdout_evaluation,
    )
    costly(
        "start_ablation",
        "Retrain one model (best by default) without each chosen feature in turn. Starts a job.",
        StartAblationArgs,
        start_ablation,
        WorkflowKind.ABLATION,
        audit="start_ablation",
        estimator="processing",
    )
    read(
        "get_ablation",
        "Get the ablation result: completed with its report, failed, or not started/running.",
        ExperimentArgs,
        get_ablation,
    )
