"""Model-version routes (control-ui-api design C4).

Thin reads over ``ModelVersionService``: list every Model_Version of an experiment (current run
first, then archived runs, newest first) and fetch one by its stable id. No mutation — the store is
a read model over per-run storage that already exists.
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.errors import json_ok
from backend.container import AppContainer


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    return [
        ("GET", "/experiments/{experiment_id}/model-versions", _list_versions),
        ("GET", "/experiments/{experiment_id}/model-versions/{version_id}", _get_version),
    ]


def _list_versions(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        versions = container.model_versions.list_versions(experiment_id)
        return json_ok(
            {
                "experiment_id": experiment_id,
                "count": len(versions),
                "model_versions": [v.model_dump(mode="json") for v in versions],
            }
        )

    return handler


def _get_version(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        version_id = request.path_params["version_id"]
        version = container.model_versions.get_version(experiment_id, version_id)
        return json_ok(version.model_dump(mode="json"))

    return handler
