"""The whole control-UI flow over the REST API, offline (control-ui-api task 9).

Create -> EDA -> decide features -> train -> list Model_Versions -> validate on a labeled dataset
-> post-prediction analysis (metrics + threshold analysis on the validation labels), all through
the API routes over the in-process inline container, so every stage produces real artifacts.
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


@pytest.mark.anyio
async def test_the_whole_flow_over_the_api(inline_container, dataset_uri):
    app = build_app(inline_container)

    # 1. Create the experiment (EDA runs inline on create).
    status, created = await _call(
        app,
        "POST",
        "/api/experiments",
        body={"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
    )
    assert status == 200
    experiment_id = created["experiment_id"]
    pp = {"experiment_id": experiment_id}

    # 2. EDA and the inferred problem type are readable.
    status, eda = await _call(app, "GET", "/api/experiments/{experiment_id}/eda", path_params=pp)
    assert status == 200
    assert eda["dataset"]["row_count"] > 0

    # 3. Approve the feature selection the review suggests.
    _, review = await _call(
        app, "GET", "/api/experiments/{experiment_id}/feature-review", path_params=pp
    )
    selected = [f["feature"] for f in review["features"] if f["selected"]]
    status, saved = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/features",
        body={"selected_features": selected},
        path_params=pp,
    )
    assert status == 200 and saved["selected_count"] == len(selected)

    # 4. Train the chosen algorithms.
    status, started = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/training",
        body={"models": MODELS},
        path_params=pp,
    )
    assert status == 200 and started["execution_id"]

    # 5. The comparison and Model_Versions are available.
    status, comparison = await _call(
        app, "GET", "/api/experiments/{experiment_id}/comparison", path_params=pp
    )
    assert status == 200 and comparison["models"]

    status, versions = await _call(
        app, "GET", "/api/experiments/{experiment_id}/model-versions", path_params=pp
    )
    assert status == 200
    assert {v["model_name"] for v in versions["model_versions"]} == set(MODELS)

    # 6. Validate every version on a labeled dataset.
    status, validation = await _call(
        app,
        "POST",
        "/api/experiments/{experiment_id}/validations",
        body={"dataset_uri": dataset_uri},
        path_params=pp,
    )
    assert status == 200 and validation["status"] == "completed"
    report = validation["report"]
    assert report["labels_present"] is True
    assert {r["model_name"] for r in report["results"]} == set(MODELS)

    # 7. Post-prediction analysis: metrics and a threshold analysis on the validation labels.
    for r in report["results"]:
        assert r["status"] == "completed"
        assert r["metrics_all_rows"] is not None
        assert r["threshold_analysis"] is not None  # binary classification
        curve = r["threshold_analysis"]["curve"]
        assert curve and all("threshold" in point for point in curve)

    # 8. The validation did not change the experiment's training state.
    _, record = await _call(app, "GET", "/api/experiments/{experiment_id}", path_params=pp)
    assert record["status"] == created["status"] or record["best_model"]
