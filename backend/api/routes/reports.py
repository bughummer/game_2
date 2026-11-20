"""Report generation and download-link routes (control-ui-api design C3).

Thin wrappers over ``ReportingService`` and ``DatasetService`` mirroring the matching MCP tools. A
download link is a mutation (it issues a short-lived credential) and is scoped to the experiment:
a URI outside the experiment's prefix is refused.
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import audit_mutation, principal, require_owner
from backend.api.errors import json_ok
from backend.api.request import json_body
from backend.container import AppContainer
from backend.errors import ValidationError
from ml_engine.contracts.report import ReportKind


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    return [
        ("POST", "/experiments/{experiment_id}/reports", _generate_report),
        ("GET", "/experiments/{experiment_id}/reports", _get_report),
        ("POST", "/experiments/{experiment_id}/download-link", _download_link),
    ]


def _generate_report(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        kind = body.get("kind", "model_comparison")
        if kind == "eda":
            artifacts = container.reporting.build_eda_report(experiment_id)
        else:
            artifacts = container.reporting.build_comparison_report(experiment_id)
        audit_mutation(container, experiment_id, "generate_report", who.user, kind=kind)
        return json_ok(
            {
                "experiment_id": experiment_id,
                "kind": artifacts.kind.value,
                "json_uri": artifacts.json_uri,
                "html_uri": artifacts.html_uri,
                "pdf_uri": artifacts.pdf_uri,
                "pdf_available": artifacts.pdf_available,
            }
        )

    return handler


def _get_report(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        kind = request.query_params.get("kind", "model_comparison")
        report = container.reporting.get_report(experiment_id, ReportKind(kind))
        return json_ok(report.model_dump(mode="json") if report else None)

    return handler


def _download_link(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        uri = body["uri"]
        prefix = container.experiments.get(experiment_id).artifact_prefix
        if not (uri == prefix or uri.startswith(prefix.rstrip("/") + "/")):
            raise ValidationError("That artifact does not belong to this experiment.")
        link = container.datasets.create_download_url(uri)
        audit_mutation(container, experiment_id, "get_download_link", who.user, uri=uri)
        return json_ok(link)

    return handler
