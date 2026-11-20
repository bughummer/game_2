"""Free-form experiment routes (free-form-experiments design C9).

Start a user-approved free-form run and read its status. Thin wrappers over ``FreeFormService``;
ownership is enforced and every mutation is audited. The platform runs the exact approved bytes and
labels results honestly.
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import audit_mutation, principal, require_owner
from backend.api.errors import json_ok
from backend.api.request import json_body
from backend.container import AppContainer
from ml_engine.contracts.free_form import DataContextAnswer, DataContextAnswers


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    return [
        ("POST", "/experiments/{experiment_id}/free-form", _start_run),
        ("GET", "/experiments/{experiment_id}/free-form", _get_run),
        ("GET", "/experiments/{experiment_id}/free-form/runs", _list_runs),
        ("GET", "/experiments/{experiment_id}/free-form/comparison", _compare),
        ("POST", "/experiments/{experiment_id}/free-form/validate", _validate),
        ("GET", "/experiments/{experiment_id}/free-form/data-context", _data_context),
        ("POST", "/experiments/{experiment_id}/free-form/draft", _draft),
    ]


def _start_run(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        status = container.free_form.start_run(
            experiment_id,
            script=body["script"],
            dataset_uri=body["dataset_uri"],
            entry_function=body.get("entry_function", "main"),
            entry_module=body.get("entry_module", "experiment"),
            predict_entry_point=body.get("predict_entry_point"),
            requirements=body.get("requirements") or [],
            instance_type=body.get("instance_type"),
            use_spot=body.get("use_spot", False),
            file_format=body.get("file_format"),
            target_column=body.get("target_column"),
            declared_feature_columns=body.get("declared_feature_columns") or [],
            declared_external_urls=body.get("declared_external_urls") or [],
            acknowledged_high_severity_leakage=body.get(
                "acknowledged_high_severity_leakage", False
            ),
            user=who.user,
        )
        audit_mutation(
            container,
            experiment_id,
            "start_free_form",
            who.user,
            run_id=status["run_id"],
            content_hash=status["content_hash"],
            dataset_uri=body["dataset_uri"],
        )
        return json_ok(status)

    return handler


def _get_run(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        run_id = request.query_params.get("run_id")
        return json_ok(container.free_form.run_status(experiment_id, run_id))

    return handler


def _list_runs(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        return json_ok({"runs": container.free_form.list_runs(experiment_id)})

    return handler


def _compare(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        return json_ok(
            container.free_form.compare(experiment_id, model_versions=container.model_versions)
        )

    return handler


def _data_context(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        dataset_uri = request.query_params["dataset_uri"]
        return json_ok(container.free_form.request_data_context(experiment_id, dataset_uri))

    return handler


def _draft(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        answers = DataContextAnswers(
            answers=[DataContextAnswer(**a) for a in body.get("data_context") or []]
        )
        result = container.free_form.draft_script(
            experiment_id,
            dataset_uri=body["dataset_uri"],
            description=body.get("description", ""),
            data_context=answers,
            draft_id=body.get("draft_id"),
            instruction=body.get("instruction"),
        )
        audit_mutation(
            container,
            experiment_id,
            "draft_free_form_script",
            who.user,
            draft_id=result["draft_id"],
            revision=str(result["revision"]["revision"]),
        )
        return json_ok(result)

    return handler


def _validate(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        status = container.free_form.validate_run_model(
            experiment_id,
            body["run_id"],
            dataset_uri=body["dataset_uri"],
            file_format=body.get("file_format"),
            validation=container.validation,
            user=who.user,
        )
        audit_mutation(
            container,
            experiment_id,
            "validate_free_form",
            who.user,
            run_id=body["run_id"],
            dataset_uri=body["dataset_uri"],
        )
        return json_ok(status)

    return handler
