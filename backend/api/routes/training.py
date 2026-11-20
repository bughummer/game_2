"""Training and run-control routes (control-ui-api design C3).

Thin wrappers over ``ExperimentService`` and ``AnalysisService`` mirroring the matching MCP tools.
``start_training`` echoes the resolved evaluation decision read back from the frozen configuration
(Req 1.7); all guards run in the service, so the API adds none and bypasses none (Req 6.5).
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import audit_mutation, principal, require_owner
from backend.api.errors import json_ok
from backend.api.request import json_body
from backend.api.views import (
    grid_point_count,
    resolved_ensembles,
    resolved_evaluation,
    resolved_treatment,
)
from backend.container import AppContainer
from backend.schemas import TrainingRequest
from ml_engine.contracts.common import ClassWeighting


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    return [
        ("POST", "/experiments/{experiment_id}/training", _start_training),
        ("GET", "/experiments/{experiment_id}/training-status", _training_status),
        ("GET", "/experiments/{experiment_id}/preflight", _preflight),
        ("POST", "/experiments/{experiment_id}/training/stop", _stop_training),
        ("GET", "/experiments/{experiment_id}/execution-status", _execution_status),
        ("POST", "/experiments/{experiment_id}/training/rerun", _rerun_training),
        ("GET", "/experiments/{experiment_id}/runs", _list_runs),
        ("GET", "/experiments/{experiment_id}/threshold-analysis", _threshold_analysis),
    ]


def _start_training(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        preflight = container.analysis.preflight(experiment_id)
        class_weighting = body.get("class_weighting", "auto")
        model_parameters = body.get("model_parameters") or {}
        updated = container.experiments.start_training(
            experiment_id,
            TrainingRequest(
                models=body.get("models") or [],
                primary_metric=body.get("primary_metric"),
                problem_type=body.get("problem_type"),
                class_weighting=ClassWeighting(class_weighting),
                model_parameters=model_parameters,
                split_strategy=body.get("split_strategy"),
                group_column=body.get("group_column"),
                time_column=body.get("time_column"),
                evaluation_mode=body.get("evaluation_mode"),
                cv_folds=body.get("cv_folds"),
                test_fold=body.get("test_fold", False),
                test_fraction=body.get("test_fraction"),
                outlier_detector=body.get("outlier_detector"),
                outlier_handling=body.get("outlier_handling"),
                zscore_threshold=body.get("zscore_threshold"),
                iqr_multiplier=body.get("iqr_multiplier"),
                isolation_contamination=body.get("isolation_contamination"),
                ensembles=body.get("ensembles") or [],
            ),
            user=who.user,
        )
        resolved = resolved_evaluation(container.store, updated.artifact_prefix)
        audit_mutation(
            container,
            experiment_id,
            "start_training",
            who.user,
            models=",".join(updated.requested_models),
            evaluation_mode=resolved["evaluation_mode"],
            cv_folds=str(resolved["cv_folds"]),
            cv_scheme=str(resolved["cv_scheme"]),
            test_fraction=str(resolved["test_fraction"]),
        )
        return json_ok(
            {
                "experiment_id": experiment_id,
                "status": updated.status.value,
                "execution_id": updated.training_execution_arn,
                "requested_models": updated.requested_models,
                "class_weighting": class_weighting,
                "model_parameters": model_parameters,
                "evaluation_mode": resolved["evaluation_mode"],
                "cv_folds": resolved["cv_folds"],
                "cv_scheme": resolved["cv_scheme"],
                "test_fraction": resolved["test_fraction"],
                "grid_points": grid_point_count(container.store, updated.artifact_prefix),
                "treatment": resolved_treatment(container.store, updated.artifact_prefix),
                "ensembles": resolved_ensembles(container.store, updated.artifact_prefix),
                "preflight_warnings": preflight.warnings,
                "unresolved_leakage_warnings": preflight.unresolved_leakage_warnings,
            }
        )

    return handler


def _training_status(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        return json_ok(container.experiments.training_status(experiment_id).model_dump(mode="json"))

    return handler


def _preflight(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        return json_ok(container.analysis.preflight(experiment_id).model_dump(mode="json"))

    return handler


def _stop_training(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        execution_id = container.experiments.stop_training(experiment_id)
        audit_mutation(
            container, experiment_id, "stop_training", who.user, execution_id=execution_id
        )
        return json_ok(
            {"experiment_id": experiment_id, "execution_id": execution_id, "status": "stopping"}
        )

    return handler


def _execution_status(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        execution_id = request.query_params.get("execution_id")
        return json_ok(
            container.analysis.execution_status(experiment_id, execution_id).model_dump(mode="json")
        )

    return handler


def _rerun_training(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        updated = container.experiments.rerun_training(experiment_id, user=who.user)
        audit_mutation(container, experiment_id, "rerun_training", who.user)
        return json_ok(
            {
                "experiment_id": experiment_id,
                "status": updated.status.value,
                "execution_id": updated.training_execution_arn,
            }
        )

    return handler


def _list_runs(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        runs = container.experiments.list_runs(experiment_id)
        return json_ok(
            {"experiment_id": experiment_id, "runs": [r.model_dump(mode="json") for r in runs]}
        )

    return handler


def _threshold_analysis(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        model_name = request.query_params.get("model_name")
        analyses = container.experiments.thresholds(experiment_id, model_name)
        return json_ok(
            {
                "experiment_id": experiment_id,
                "models": [a.model_dump(mode="json") for a in analyses],
            }
        )

    return handler
