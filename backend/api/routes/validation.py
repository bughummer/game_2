"""Validation routes (control-ui-api design C5).

Start a validation of one Model_Version, or all of them, on an uploaded labeled dataset, and read
the result. Thin wrappers over ``ValidationService``; the job keeps labels for metrics and a
threshold analysis computed on the validation labels.
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import audit_mutation, principal, require_owner
from backend.api.errors import json_ok
from backend.api.request import json_body
from backend.container import AppContainer


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    return [
        ("POST", "/experiments/{experiment_id}/validations", _start_validation),
        ("GET", "/experiments/{experiment_id}/validations", _get_validation),
    ]


def _start_validation(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        status = container.validation.start_validation(
            experiment_id,
            dataset_uri=body["dataset_uri"],
            version_ids=body.get("version_ids"),
            file_format=body.get("file_format"),
            user=who.user,
        )
        audit_mutation(
            container,
            experiment_id,
            "start_validation",
            who.user,
            validation_id=status["validation_id"],
            dataset_uri=body["dataset_uri"],
            version_ids=",".join(body.get("version_ids") or []),
        )
        return json_ok(status)

    return handler


def _get_validation(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        validation_id = request.query_params.get("validation_id")
        return json_ok(container.validation.validation(experiment_id, validation_id))

    return handler
