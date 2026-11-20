"""Experiment and business-context routes (control-ui-api design C3).

Thin wrappers over ``ExperimentService`` and ``LeakageService`` — each calls exactly the service
method the matching MCP tool calls, so a UI and a chat agent drive one experiment identically.
Every mutation resolves the caller, enforces ownership and audits; reads do neither.
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import audit_mutation, principal, require_owner
from backend.api.errors import json_ok
from backend.api.request import json_body, query_flag, query_int
from backend.api.views import experiment_summary
from backend.container import AppContainer
from ml_engine.contracts.common import RequestedProblemType


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    return [
        ("POST", "/experiments", _create),
        ("GET", "/experiments", _list),
        ("GET", "/experiments/{experiment_id}", _get),
        ("DELETE", "/experiments/{experiment_id}", _delete),
        ("POST", "/experiments/{experiment_id}/business-context", _set_business_context),
        ("GET", "/experiments/{experiment_id}/business-context", _get_business_context),
    ]


def _create(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        body = await json_body(request)
        who = principal(container, request)
        experiment = container.experiments.create(
            name=body.get("name"),
            dataset_uri=body.get("dataset_uri"),
            target_column=body.get("target_column"),
            problem_type=RequestedProblemType(body.get("problem_type", "auto")),
            file_format=body.get("file_format"),
            user=who.user,
        )
        audit_mutation(
            container,
            experiment.experiment_id,
            "create_experiment",
            who.user,
            dataset_uri=body.get("dataset_uri"),
            target_column=body.get("target_column"),
        )
        return json_ok(experiment_summary(experiment))

    return handler


def _get(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        return json_ok(experiment_summary(container.experiments.get(experiment_id)))

    return handler


def _list(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        who = principal(container, request)
        created_by = who.user if query_flag(request, "mine_only") else None
        records = container.experiments.list(
            limit=query_int(request, "limit", 50), created_by=created_by
        )
        return json_ok(
            {"count": len(records), "experiments": [experiment_summary(r) for r in records]}
        )

    return handler


def _delete(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        audit_mutation(container, experiment_id, "delete_experiment", who.user)
        stopped = container.experiments.delete(experiment_id)
        return json_ok(
            {
                "experiment_id": experiment_id,
                "status": "deleted",
                "stopped_executions": stopped,
            }
        )

    return handler


def _set_business_context(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        context = container.leakage.set_business_context(
            experiment_id,
            objective=body.get("objective", ""),
            prediction_scenario=body.get("prediction_scenario", ""),
            scoring_trigger=body.get("scoring_trigger"),
            prediction_horizon=body.get("prediction_horizon"),
            entity_id_column=body.get("entity_id_column"),
            temporal_columns=body.get("temporal_columns") or [],
            investigate_leakage=body.get("investigate_leakage", True),
            user=who.user,
        )
        audit_mutation(container, experiment_id, "set_business_context", who.user)
        return json_ok(context.model_dump(mode="json"))

    return handler


def _get_business_context(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        context = container.leakage.get_business_context(experiment_id)
        return json_ok(context.model_dump(mode="json") if context else None)

    return handler
