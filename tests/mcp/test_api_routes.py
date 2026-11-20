"""Experiment, dataset, EDA and feature REST routes (control-ui-api task 2).

The API is a peer of the MCP on the same HTTP app over the same container. No socket is opened —
routes are resolved on the app and their endpoints called directly with a minimal ASGI scope, the
same pattern ``test_api_skeleton`` and the MCP's /health test use. The routes are thin wrappers over
the very ``backend.services`` methods the MCP tools call, so these tests assert the shared behaviour:
summaries and counts, ownership (403), missing resource (404) and bounded previews (Req 1.3-1.5, 3.3).
"""

import json

import pandas as pd
import pytest
from starlette.requests import Request

from backend.errors import AuthorizationError
from mcp_server.app import build_app
from tests.support.datasets import APPROVED_PREFIX


@pytest.fixture
def dataset_uri(s3_client, classification_frame: pd.DataFrame) -> str:
    key = "curated/churn.csv"
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2], Key=key, Body=classification_frame.to_csv(index=False)
    )
    return f"{APPROVED_PREFIX}/churn.csv"


def _scope(method: str, path: str, *, body: bytes = b"", query: bytes = b"") -> dict:
    scope = {
        "type": "http",
        "method": method,
        "headers": [(b"content-type", b"application/json")] if body else [],
        "query_string": query,
        "path_params": {},
    }
    return scope


def _request(method: str, *, body: dict | None = None, query: bytes = b"") -> Request:
    raw = json.dumps(body).encode() if body is not None else b""
    request = Request(_scope(method, "", body=raw, query=query))
    if raw:

        async def receive() -> dict:
            return {"type": "http.request", "body": raw, "more_body": False}

        request._receive = receive
    return request


async def _call(app, method: str, path: str, *, body=None, query=b"", path_params=None):
    """Resolve a route by its Starlette path and call its endpoint with a crafted request."""
    http_app = app.http_app()
    route = next(
        r
        for r in http_app.routes
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set())
    )
    request = _request(method, body=body, query=query)
    request.scope["path_params"] = path_params or {}
    response = await route.endpoint(request)
    return response.status_code, json.loads(bytes(response.body))


async def _create_experiment(app, dataset_uri, *, name="churn", target="churned"):
    status, body = await _call(
        app,
        "POST",
        "/api/experiments",
        body={"name": name, "dataset_uri": dataset_uri, "target_column": target},
    )
    return status, body


# --- experiments ------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_create_returns_a_summary_and_is_audited(container, dataset_uri):
    app = build_app(container)
    status, body = await _create_experiment(app, dataset_uri)
    assert status == 200
    assert body["name"] == "churn"
    assert body["target_column"] == "churned"
    assert body["created_by"] == container.settings.dev_identity
    assert body["experiment_id"]

    # The mutation is audited under the experiment's own prefix (Req 1.4 parity with the MCP).
    prefix = container.experiments.get(body["experiment_id"]).artifact_prefix
    events = list(container.store.list_uris(f"{prefix.rstrip('/')}/audit"))
    actions = {json.loads(container.store.read_bytes(uri))["action"] for uri in events}
    assert "create_experiment" in actions


@pytest.mark.anyio
async def test_get_and_list_experiments(container, dataset_uri):
    app = build_app(container)
    _, created = await _create_experiment(app, dataset_uri)
    experiment_id = created["experiment_id"]

    status, got = await _call(
        app, "GET", "/api/experiments/{experiment_id}", path_params={"experiment_id": experiment_id}
    )
    assert status == 200
    assert got["experiment_id"] == experiment_id

    status, listed = await _call(app, "GET", "/api/experiments")
    assert status == 200
    assert listed["count"] == 1
    assert [e["experiment_id"] for e in listed["experiments"]] == [experiment_id]


@pytest.mark.anyio
async def test_delete_experiment(container, dataset_uri):
    app = build_app(container)
    _, created = await _create_experiment(app, dataset_uri)
    experiment_id = created["experiment_id"]

    status, body = await _call(
        app,
        "DELETE",
        "/api/experiments/{experiment_id}",
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    assert body["experiment_id"] == experiment_id
    assert body["status"] == "deleted"
    assert isinstance(body["stopped_executions"], list)


@pytest.mark.anyio
async def test_set_and_get_business_context(container, dataset_uri):
    app = build_app(container)
    _, created = await _create_experiment(app, dataset_uri)
    experiment_id = created["experiment_id"]

    status, context = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/business-context",
        body={
            "objective": "Predict churn before renewal.",
            "prediction_scenario": "Scored nightly on active accounts.",
        },
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    assert context["objective"] == "Predict churn before renewal."

    status, got = await _call(
        app,
        "GET",
        "/api/experiments/{experiment_id}/business-context",
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    assert got["prediction_scenario"] == "Scored nightly on active accounts."


# --- datasets ---------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_dataset_upload_ticket_register_and_list(container, dataset_uri):
    app = build_app(container)

    status, ticket = await _call(
        app, "POST", "/api/datasets/upload-ticket", body={"filename": "churn.csv"}
    )
    assert status == 200
    assert ticket["upload_url"].startswith("https://")
    assert ticket["method"] == "PUT"

    status, registered = await _call(
        app, "POST", "/api/datasets/register", body={"dataset_uri": dataset_uri}
    )
    assert status == 200
    assert registered["file_format"] == "csv"
    assert any(c["name"] == "churned" for c in registered["columns"])

    status, listed = await _call(app, "GET", "/api/datasets")
    assert status == 200
    assert "datasets" in listed and "count" in listed


@pytest.mark.anyio
async def test_dataset_preview_is_bounded(inline_container, dataset_uri):
    app = build_app(inline_container)
    _, created = await _create_experiment(app, dataset_uri)
    experiment_id = created["experiment_id"]

    status, preview = await _call(
        app,
        "GET",
        "/api/experiments/{experiment_id}/dataset-preview",
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    assert preview["preview_row_cap"] <= 20
    assert len(preview["columns"]) <= 100
    assert preview["columns_truncated"] is False


# --- EDA / features (real artifacts) ----------------------------------------------------------


@pytest.mark.anyio
async def test_eda_and_feature_review_read_real_artifacts(inline_container, dataset_uri):
    app = build_app(inline_container)
    _, created = await _create_experiment(app, dataset_uri)
    experiment_id = created["experiment_id"]

    status, eda = await _call(
        app,
        "GET",
        "/api/experiments/{experiment_id}/eda",
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    assert eda["dataset"]["row_count"] > 0

    status, review = await _call(
        app,
        "GET",
        "/api/experiments/{experiment_id}/feature-review",
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    assert review["target_column"] == "churned"
    assert review["features"]


@pytest.mark.anyio
async def test_save_features_returns_counts(inline_container, dataset_uri):
    app = build_app(inline_container)
    _, created = await _create_experiment(app, dataset_uri)
    experiment_id = created["experiment_id"]

    _, review = await _call(
        app,
        "GET",
        "/api/experiments/{experiment_id}/feature-review",
        path_params={"experiment_id": experiment_id},
    )
    selected = [f["feature"] for f in review["features"] if f["selected"]]

    status, body = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/features",
        body={"selected_features": selected},
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    assert body["experiment_id"] == experiment_id
    assert body["selected_count"] == len(selected)
    assert body["excluded_count"] >= 0


# --- error and ownership mapping --------------------------------------------------------------


@pytest.mark.anyio
async def test_a_read_on_a_missing_experiment_is_404(container):
    app = build_app(container)
    status, body = await _call(
        app, "GET", "/api/experiments/{experiment_id}", path_params={"experiment_id": "exp-nope"}
    )
    assert status == 404
    assert body["code"] == "NOT_FOUND"


@pytest.mark.anyio
async def test_an_ownership_failure_is_403(container, dataset_uri, monkeypatch):
    app = build_app(container)
    _, created = await _create_experiment(app, dataset_uri)
    experiment_id = created["experiment_id"]

    # A mutation must call require_owner before acting; make it refuse and assert the 403 maps.
    def deny(_principal, _experiment_id):
        raise AuthorizationError("This experiment belongs to another user.")

    monkeypatch.setattr(container.authorizer, "require_owner", deny)
    status, body = await _call(
        app,
        "DELETE",
        "/api/experiments/{experiment_id}",
        path_params={"experiment_id": experiment_id},
    )
    assert status == 403
    assert body["code"] == "NOT_AUTHORIZED"
