"""Validation on a labeled dataset (control-ui-api task 6).

Drives a real train, then validates the model versions on an uploaded dataset via the inline
container so the validation job actually runs. Asserts labels are kept for metrics, the threshold
analysis is computed on the validation labels, all versions are scored, the labels-absent path
produces predictions only, and a validation does not change training state.
"""

import json

import pandas as pd
import pytest
from starlette.requests import Request

from mcp_server.app import build_app
from tests.support.datasets import APPROVED_PREFIX

MODELS = ["logistic_regression", "random_forest"]


@pytest.fixture
def dataset_uri(s3_client, classification_frame: pd.DataFrame) -> str:
    key = "curated/churn.csv"
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2], Key=key, Body=classification_frame.to_csv(index=False)
    )
    return f"{APPROVED_PREFIX}/churn.csv"


@pytest.fixture
def unlabeled_uri(s3_client, classification_frame: pd.DataFrame) -> str:
    key = "curated/churn_unlabeled.csv"
    frame = classification_frame.drop(columns=["churned"])
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2], Key=key, Body=frame.to_csv(index=False)
    )
    return f"{APPROVED_PREFIX}/churn_unlabeled.csv"


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


async def _train(app, dataset_uri):
    _, created = await _call(
        app,
        "POST",
        "/api/experiments",
        body={"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
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
    await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/training",
        body={"models": MODELS},
        path_params=pp,
    )
    return experiment_id


@pytest.mark.anyio
async def test_validate_all_versions_keeps_labels_for_metrics(inline_container, dataset_uri):
    app = build_app(inline_container)
    experiment_id = await _train(app, dataset_uri)
    pp = {"experiment_id": experiment_id}

    status, result = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/validations",
        body={"dataset_uri": dataset_uri},  # no version_ids -> all versions
        path_params=pp,
    )
    assert status == 200
    assert result["status"] == "completed"
    report = result["report"]
    assert report["labels_present"] is True
    # One result per model version; metrics present because labels were kept.
    names = {r["model_name"] for r in report["results"]}
    assert names == set(MODELS)
    for r in report["results"]:
        assert r["status"] == "completed"
        assert r["metrics_all_rows"] is not None
        # Binary classification: a fresh threshold analysis computed on the validation labels.
        assert r["threshold_analysis"] is not None
        assert r["threshold_analysis"]["curve"]


@pytest.mark.anyio
async def test_validate_a_single_version(inline_container, dataset_uri):
    app = build_app(inline_container)
    experiment_id = await _train(app, dataset_uri)
    pp = {"experiment_id": experiment_id}

    _, versions = await _call(
        app, "GET", "/api/experiments/{experiment_id}/model-versions", path_params=pp
    )
    one = versions["model_versions"][0]["version_id"]

    _, result = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/validations",
        body={"dataset_uri": dataset_uri, "version_ids": [one]},
        path_params=pp,
    )
    report = result["report"]
    assert [r["version_id"] for r in report["results"]] == [one]


@pytest.mark.anyio
async def test_validation_without_labels_produces_predictions_only(
    inline_container, dataset_uri, unlabeled_uri
):
    app = build_app(inline_container)
    experiment_id = await _train(app, dataset_uri)
    pp = {"experiment_id": experiment_id}

    _, result = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/validations",
        body={"dataset_uri": unlabeled_uri},
        path_params=pp,
    )
    report = result["report"]
    assert report["labels_present"] is False
    for r in report["results"]:
        assert r["status"] == "completed"
        assert r["metrics_all_rows"] is None  # no labels -> no metrics
        assert r["prediction_summary"]  # but predictions are summarised


@pytest.mark.anyio
async def test_validation_does_not_change_training_state(inline_container, dataset_uri):
    app = build_app(inline_container)
    experiment_id = await _train(app, dataset_uri)
    pp = {"experiment_id": experiment_id}

    _, before = await _call(app, "GET", "/api/experiments/{experiment_id}", path_params=pp)
    await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/validations",
        body={"dataset_uri": dataset_uri},
        path_params=pp,
    )
    _, after = await _call(app, "GET", "/api/experiments/{experiment_id}", path_params=pp)
    assert after["status"] == before["status"]
    assert after["current_stage"] == before["current_stage"]


@pytest.mark.anyio
async def test_get_validation_is_readable_and_audited(inline_container, dataset_uri):
    app = build_app(inline_container)
    experiment_id = await _train(app, dataset_uri)
    pp = {"experiment_id": experiment_id}

    await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/validations",
        body={"dataset_uri": dataset_uri},
        path_params=pp,
    )
    status, latest = await _call(
        app, "GET", "/api/experiments/{experiment_id}/validations", path_params=pp
    )
    assert status == 200
    assert latest["status"] == "completed"

    prefix = inline_container.experiments.get(experiment_id).artifact_prefix
    actions = {
        json.loads(inline_container.store.read_bytes(uri))["action"]
        for uri in inline_container.store.list_uris(f"{prefix.rstrip('/')}/audit")
    }
    assert "start_validation" in actions
