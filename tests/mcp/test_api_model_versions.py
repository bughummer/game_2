"""Model_Version store and routes (control-ui-api task 5).

Drives a real train + retrain with the inline container, then asserts versions are listed across
the current and archived runs, that a retrain adds versions without losing the earlier ones, and
that each version resolves to loadable artifacts.
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


async def _call(app, method, path, *, body=None, path_params=None):
    http_app = app.http_app()
    route = next(
        r
        for r in http_app.routes
        if getattr(r, "path", None) == path and method in getattr(r, "methods", set())
    )
    request = _request(method, body=body)
    request.scope["path_params"] = path_params or {}
    response = await route.endpoint(request)
    return response.status_code, json.loads(bytes(response.body))


async def _train(app, dataset_uri, *, name="churn"):
    _, created = await _call(
        app,
        "POST",
        "/api/experiments",
        body={"name": name, "dataset_uri": dataset_uri, "target_column": "churned"},
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
async def test_versions_list_the_current_run_models(inline_container, dataset_uri):
    app = build_app(inline_container)
    experiment_id = await _train(app, dataset_uri)

    status, body = await _call(
        app,
        "GET",
        "/api/experiments/{experiment_id}/model-versions",
        path_params={"experiment_id": experiment_id},
    )
    assert status == 200
    names = {v["model_name"] for v in body["model_versions"]}
    assert names == set(MODELS)
    # Current-run versions use the 'current:' id prefix and carry no run_id.
    for v in body["model_versions"]:
        assert v["version_id"].startswith("current:")
        assert v["run_id"] is None
        assert v["artifact_ref"]["model_uri"].endswith("/model.joblib")


@pytest.mark.anyio
async def test_retrain_adds_versions_without_losing_the_earlier_ones(inline_container, dataset_uri):
    app = build_app(inline_container)
    experiment_id = await _train(app, dataset_uri)
    pp = {"experiment_id": experiment_id}

    # Retrain (archives the previous run, then trains again).
    await _call(app, "POST", "/api/experiments/{experiment_id}/training/rerun", path_params=pp)

    _, body = await _call(
        app, "GET", "/api/experiments/{experiment_id}/model-versions", path_params=pp
    )
    versions = body["model_versions"]
    # Current run's models plus the archived run's models — nothing lost.
    current = [v for v in versions if v["run_id"] is None]
    archived = [v for v in versions if v["run_id"] is not None]
    assert {v["model_name"] for v in current} == set(MODELS)
    assert {v["model_name"] for v in archived} == set(MODELS)
    # Archived versions point into runs/<run_id>/, not the live prefix.
    for v in archived:
        assert f"/runs/{v['run_id']}/" in v["artifact_ref"]["model_uri"]


@pytest.mark.anyio
async def test_each_version_is_independently_loadable(inline_container, dataset_uri):
    app = build_app(inline_container)
    experiment_id = await _train(app, dataset_uri)
    pp = {"experiment_id": experiment_id}

    _, body = await _call(
        app, "GET", "/api/experiments/{experiment_id}/model-versions", path_params=pp
    )
    store = inline_container.store
    for v in body["model_versions"]:
        ref = v["artifact_ref"]
        assert store.exists(ref["model_uri"]), ref["model_uri"]
        assert store.exists(ref["metadata_uri"]), ref["metadata_uri"]
        assert store.exists(ref["preprocessor_uri"]), ref["preprocessor_uri"]

        # get_version resolves the same handle.
        status, one = await _call(
            app,
            "GET",
            "/api/experiments/{experiment_id}/model-versions/{version_id}",
            path_params={"experiment_id": experiment_id, "version_id": v["version_id"]},
        )
        assert status == 200
        assert one["version_id"] == v["version_id"]


@pytest.mark.anyio
async def test_a_missing_version_is_404(inline_container, dataset_uri):
    app = build_app(inline_container)
    experiment_id = await _train(app, dataset_uri)
    status, body = await _call(
        app,
        "GET",
        "/api/experiments/{experiment_id}/model-versions/{version_id}",
        path_params={"experiment_id": experiment_id, "version_id": "current:nope"},
    )
    assert status == 404
    assert body["code"] == "NOT_FOUND"
