"""Training, analysis, report and download REST routes (control-ui-api task 3).

Drives a real create -> EDA -> decide features -> train flow with the in-process inline container,
then reads the analysis routes, so the routes are exercised against real artifacts. The no-socket
harness mirrors ``test_api_routes``.
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


def _request(method: str, *, body: dict | None = None) -> Request:
    raw = json.dumps(body).encode() if body is not None else b""
    request = Request(
        {
            "type": "http",
            "method": method,
            "headers": [(b"content-type", b"application/json")] if raw else [],
            "query_string": b"",
            "path_params": {},
        }
    )
    if raw:

        async def receive() -> dict:
            return {"type": "http.request", "body": raw, "more_body": False}

        request._receive = receive
    return request


async def _call(app, method, path, *, body=None, query=b"", path_params=None):
    http_app = app.http_app()
    route = next(
        r
        for r in http_app.routes
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set())
    )
    request = _request(method, body=body)
    request.scope["path_params"] = path_params or {}
    request.scope["query_string"] = query
    response = await route.endpoint(request)
    return response.status_code, json.loads(bytes(response.body))


async def _trained_experiment(app):
    """Create, profile, decide features and train one experiment end to end."""
    _, created = await _call(
        app,
        "POST",
        "/api/experiments",
        body={"name": "churn", "dataset_uri": _DATASET_URI[0], "target_column": "churned"},
    )
    experiment_id = created["experiment_id"]
    pp = {"experiment_id": experiment_id}
    _, review = await _call(
        app, "GET", "/api/experiments/{experiment_id}/feature-review", path_params=pp
    )
    selected = [f["feature"] for f in review["features"] if f["selected"]]
    await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/features",
        body={"selected_features": selected},
        path_params=pp,
    )
    status, started = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/training",
        body={"models": ["logistic_regression", "random_forest"]},
        path_params=pp,
    )
    return experiment_id, status, started


# A tiny module-level holder so _trained_experiment can see the dataset uri fixture value.
_DATASET_URI: list[str] = [""]


@pytest.fixture(autouse=True)
def _bind_dataset_uri(dataset_uri):
    _DATASET_URI[0] = dataset_uri
    yield
    _DATASET_URI[0] = ""


# --- training -------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_start_training_echoes_resolved_evaluation_and_is_audited(inline_container):
    app = build_app(inline_container)
    experiment_id, status, started = await _trained_experiment(app)
    assert status == 200
    assert started["execution_id"]
    # Resolved from the frozen config, not the request: single split by default.
    assert started["evaluation_mode"] == "single_split"
    assert started["cv_folds"] is None
    assert set(started["requested_models"]) == {"logistic_regression", "random_forest"}

    prefix = inline_container.experiments.get(experiment_id).artifact_prefix
    actions = {
        json.loads(inline_container.store.read_bytes(uri))["action"]
        for uri in inline_container.store.list_uris(f"{prefix.rstrip('/')}/audit")
    }
    assert "start_training" in actions


@pytest.mark.anyio
async def test_start_training_in_cv_mode_echoes_folds_and_scheme(inline_container):
    app = build_app(inline_container)
    _, created = await _call(
        app,
        "POST",
        "/api/experiments",
        body={"name": "cv", "dataset_uri": _DATASET_URI[0], "target_column": "churned"},
    )
    experiment_id = created["experiment_id"]
    pp = {"experiment_id": experiment_id}
    _, review = await _call(
        app, "GET", "/api/experiments/{experiment_id}/feature-review", path_params=pp
    )
    selected = [f["feature"] for f in review["features"] if f["selected"]]
    await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/features",
        body={"selected_features": selected},
        path_params=pp,
    )
    status, started = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/training",
        body={"models": ["logistic_regression"], "evaluation_mode": "cv", "cv_folds": 3},
        path_params=pp,
    )
    assert status == 200
    assert started["evaluation_mode"] == "cv"
    assert started["cv_folds"] == 3
    assert started["cv_scheme"]


@pytest.mark.anyio
async def test_training_status_and_preflight_are_readable(inline_container):
    app = build_app(inline_container)
    experiment_id, _, _ = await _trained_experiment(app)
    pp = {"experiment_id": experiment_id}

    status, ts = await _call(
        app, "GET", "/api/experiments/{experiment_id}/training-status", path_params=pp
    )
    assert status == 200
    assert ts["experiment_id"] == experiment_id

    status, pf = await _call(
        app, "GET", "/api/experiments/{experiment_id}/preflight", path_params=pp
    )
    assert status == 200
    assert "ready" in pf


@pytest.mark.anyio
async def test_runs_listing_is_readable(inline_container):
    app = build_app(inline_container)
    experiment_id, _, _ = await _trained_experiment(app)
    status, runs = await _call(
        app,
        "GET",
        "/api/experiments/{experiment_id}/runs",
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    assert runs["experiment_id"] == experiment_id
    assert isinstance(runs["runs"], list)


# --- analysis -------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_comparison_and_threshold_and_permutation_after_training(inline_container):
    app = build_app(inline_container)
    experiment_id, _, _ = await _trained_experiment(app)
    pp = {"experiment_id": experiment_id}

    status, comparison = await _call(
        app, "GET", "/api/experiments/{experiment_id}/comparison", path_params=pp
    )
    assert status == 200
    assert comparison["experiment_id"] == experiment_id
    assert comparison["models"]

    status, thresholds = await _call(
        app, "GET", "/api/experiments/{experiment_id}/threshold-analysis", path_params=pp
    )
    assert status == 200
    assert thresholds["models"]  # binary classification -> threshold analyses present

    status, perm = await _call(
        app, "GET", "/api/experiments/{experiment_id}/permutation-importance", path_params=pp
    )
    assert status == 200
    for model in perm["models"]:
        assert len(model["entries"]) <= 25  # bounded


@pytest.mark.anyio
async def test_ablation_start_returns_features_and_is_audited(inline_container):
    app = build_app(inline_container)
    experiment_id, _, _ = await _trained_experiment(app)
    pp = {"experiment_id": experiment_id}

    status, started = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/ablation",
        body={"model_name": "logistic_regression", "features": ["tenure_months"]},
        path_params=pp,
    )
    assert status == 200
    assert "features" in started

    prefix = inline_container.experiments.get(experiment_id).artifact_prefix
    actions = {
        json.loads(inline_container.store.read_bytes(uri))["action"]
        for uri in inline_container.store.list_uris(f"{prefix.rstrip('/')}/audit")
    }
    assert "start_ablation" in actions


# --- reports / downloads --------------------------------------------------------------------


@pytest.mark.anyio
async def test_generate_report_returns_artifact_refs(inline_container):
    app = build_app(inline_container)
    experiment_id, _, _ = await _trained_experiment(app)
    status, report = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/reports",
        body={"kind": "model_comparison"},
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    assert report["json_uri"]
    assert report["kind"] == "model_comparison"


@pytest.mark.anyio
async def test_download_link_rejects_a_foreign_uri(inline_container):
    app = build_app(inline_container)
    experiment_id, _, _ = await _trained_experiment(app)
    status, body = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/download-link",
        body={"uri": "s3://some-other-bucket/elsewhere/x.csv"},
        path_params={"experiment_id": experiment_id},
    )
    assert status == 400
    assert body["code"] == "VALIDATION_ERROR"


# --- ownership ------------------------------------------------------------------------------


@pytest.mark.anyio
async def test_stop_training_ownership_failure_is_403(inline_container, monkeypatch):
    app = build_app(inline_container)
    experiment_id, _, _ = await _trained_experiment(app)

    def deny(_principal, _experiment_id):
        raise AuthorizationError("This experiment belongs to another user.")

    monkeypatch.setattr(inline_container.authorizer, "require_owner", deny)
    status, body = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/training/stop",
        path_params={"experiment_id": experiment_id},
    )
    assert status == 403
    assert body["code"] == "NOT_AUTHORIZED"
