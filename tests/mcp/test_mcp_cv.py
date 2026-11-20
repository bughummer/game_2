"""Cross-validation through the MCP tools (CV design C8, Req 1.7, 11.1).

Over the in-process FastMCP client and the inline orchestrator (which runs the real stages), a
CV experiment is started and compared, and a single-split experiment is checked to still carry
no CV keys. No AWS, no network.
"""

import pandas as pd
import pytest
from fastmcp import Client

from tests.support.datasets import APPROVED_PREFIX


@pytest.fixture
def dataset_uri(s3_client, classification_frame: pd.DataFrame) -> str:
    key = "curated/churn.csv"
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2],
        Key=key,
        Body=classification_frame.to_csv(index=False),
    )
    return f"{APPROVED_PREFIX}/churn.csv"


async def _ready_experiment(client, dataset_uri: str) -> tuple[str, list[str]]:
    experiment_id = (
        await client.call_tool(
            "create_experiment",
            {"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
        )
    ).data["experiment_id"]
    review = (await client.call_tool("get_feature_review", {"experiment_id": experiment_id})).data
    selected = [f["feature"] for f in review["features"] if f["selected"]]
    await client.call_tool(
        "decide_features", {"experiment_id": experiment_id, "selected_features": selected}
    )
    return experiment_id, selected


@pytest.mark.anyio
async def test_start_training_echoes_cv_mode_k_and_scheme(inline_mcp, dataset_uri):
    async with Client(inline_mcp) as client:
        experiment_id, _ = await _ready_experiment(client, dataset_uri)

        started = (
            await client.call_tool(
                "start_training",
                {
                    "experiment_id": experiment_id,
                    "models": ["logistic_regression"],
                    "evaluation_mode": "cv",
                    "cv_folds": 3,
                    "test_fraction": 0.2,
                },
            )
        ).data

        # The response echoes what was frozen, not just what was asked for.
        assert started["evaluation_mode"] == "cv"
        assert started["cv_folds"] == 3
        assert started["cv_scheme"] in {
            "stratified_kfold",
            "kfold",
            "group_kfold",
            "forward_chaining",
        }
        assert started["test_fraction"] == 0.2


@pytest.mark.anyio
async def test_get_comparison_exposes_cv_summaries_and_test_metrics(inline_mcp, dataset_uri):
    async with Client(inline_mcp) as client:
        experiment_id, _ = await _ready_experiment(client, dataset_uri)
        await client.call_tool(
            "start_training",
            {
                "experiment_id": experiment_id,
                "models": ["logistic_regression"],
                "evaluation_mode": "cv",
                "cv_folds": 3,
                "test_fraction": 0.2,
            },
        )

        comparison = (
            await client.call_tool("get_comparison", {"experiment_id": experiment_id})
        ).data

        # Top-level CV info and a separate test-evaluation block are present and not truncated.
        assert comparison["cross_validation"]["folds"] == 3
        assert comparison["cross_validation"]["scheme"]
        assert comparison["test_evaluation"]["test_row_count"] > 0

        entry = next(m for m in comparison["models"] if m["model_name"] == "logistic_regression")
        cv = entry["cv"]
        assert cv["folds"] == 3
        assert len(cv["primary_fold_values"]) == 3
        assert "instability_flag" in cv
        # Mean ± std per metric survived the dump.
        assert comparison["primary_metric"] in cv["metrics"]
        agg = cv["metrics"][comparison["primary_metric"]]
        assert {"mean", "std", "min", "max", "fold_count"} <= set(agg)
        # Test metrics are carried separately per run.
        assert entry["test_metrics"]


@pytest.mark.anyio
async def test_single_split_shows_no_cv_keys(inline_mcp, dataset_uri):
    async with Client(inline_mcp) as client:
        experiment_id, _ = await _ready_experiment(client, dataset_uri)

        started = (
            await client.call_tool(
                "start_training",
                {"experiment_id": experiment_id, "models": ["logistic_regression"]},
            )
        ).data
        assert started["evaluation_mode"] == "single_split"
        assert started["cv_folds"] is None
        assert started["cv_scheme"] is None
        assert started["test_fraction"] is None

        comparison = (
            await client.call_tool("get_comparison", {"experiment_id": experiment_id})
        ).data
        assert comparison["best_model"] == "logistic_regression"
        # The additive CV keys are omitted at their defaults, so a single-split report is clean.
        assert "cross_validation" not in comparison
        assert "test_evaluation" not in comparison
        for entry in comparison["models"]:
            assert "cv" not in entry
            assert "test_metrics" not in entry
