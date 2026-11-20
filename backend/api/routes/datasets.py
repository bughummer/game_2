"""Dataset ingestion and preview routes (control-ui-api design C3).

Thin wrappers over ``DatasetService`` mirroring the MCP's dataset tools. Upload-ticket, register and
list only resolve the caller (no experiment is involved, so no ownership check or audit — matching
the MCP). The experiment-scoped preview is bounded exactly as the MCP's ``preview_dataset`` is
(requirement 9.5 / 3.3): capped rows and columns, never a full dataset.
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import principal
from backend.api.errors import json_ok
from backend.api.request import json_body, query_int
from backend.api.views import PREVIEW_COLUMN_CAP, PREVIEW_ROW_CAP
from backend.container import AppContainer


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    return [
        ("POST", "/datasets/upload-ticket", _upload_ticket),
        ("POST", "/datasets/register", _register),
        ("GET", "/datasets", _list),
        ("GET", "/experiments/{experiment_id}/dataset-preview", _preview),
    ]


def _upload_ticket(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        body = await json_body(request)
        principal(container, request)
        ticket = container.datasets.create_upload_ticket(body.get("filename"))
        return json_ok(ticket.model_dump(mode="json"))

    return handler


def _register(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        body = await json_body(request)
        principal(container, request)
        registered = container.datasets.register(body.get("dataset_uri"))
        return json_ok(registered.model_dump(mode="json"))

    return handler


def _list(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        return json_ok(
            container.datasets.list_uploads(limit=query_int(request, "limit", 100)).model_dump(
                mode="json"
            )
        )

    return handler


def _preview(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        record = container.experiments.get(experiment_id)
        registered = container.datasets.register(record.dataset.uri)
        columns = registered.columns[:PREVIEW_COLUMN_CAP]
        return json_ok(
            {
                "dataset_uri": record.dataset.uri,
                "file_format": registered.file_format,
                "size_bytes": registered.size_bytes,
                "columns": [c.model_dump(mode="json") for c in columns],
                "columns_truncated": len(registered.columns) > PREVIEW_COLUMN_CAP,
                "preview_row_cap": PREVIEW_ROW_CAP,
            }
        )

    return handler
