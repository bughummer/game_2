"""The business-context and leakage-investigation MCP tools (requirements 11, 12)."""

import pandas as pd
import pytest
from fastmcp import Client

from tests.support.datasets import APPROVED_PREFIX


@pytest.fixture
def dataset_uri(s3_client):
    """A dataset with a planted duplicate-of-target leak and a strong lone predictor."""
    rows = 200
    frame = pd.DataFrame(
        {
            "tenure": [i % 24 for i in range(rows)],
            "charges": [float(i % 5) for i in range(rows)],
            "churn_copy": [i % 2 for i in range(rows)],
            "future_flag": [i % 2 for i in range(rows)],
            "churned": [i % 2 for i in range(rows)],
        }
    )
    key = "curated/leaky.csv"
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2], Key=key, Body=frame.to_csv(index=False)
    )
    return f"{APPROVED_PREFIX}/leaky.csv"


@pytest.mark.anyio
async def test_business_context_and_leakage_flow_through_tools(inline_mcp, dataset_uri):
    async with Client(inline_mcp) as client:
        created = (
            await client.call_tool(
                "create_experiment",
                {"name": "leaky", "dataset_uri": dataset_uri, "target_column": "churned"},
            )
        ).data
        experiment_id = created["experiment_id"]

        context = (
            await client.call_tool(
                "set_business_context",
                {
                    "experiment_id": experiment_id,
                    "objective": "Predict churn next month.",
                    "prediction_scenario": "Score at billing time using only data known then.",
                },
            )
        ).data
        assert context["investigate_leakage"] is True
        args = {"experiment_id": experiment_id}
        assert (await client.call_tool("get_business_context", args)).data == context

        report = (
            await client.call_tool("investigate_leakage", {"experiment_id": experiment_id})
        ).data
        assert (await client.call_tool("get_leakage_investigation", args)).data == report
        warnings = report["warnings"]
        # The exact-duplicate rule establishes leakage for churn_copy.
        established = [
            w for w in warnings if w["feature"] == "churn_copy" and w["leakage_established"]
        ]
        assert established
        # Association is screening evidence only: never established.
        assert any(
            w["method"] == "target_association" and w["leakage_established"] is False
            for w in warnings
        )

        # Decide and apply — approval and selection stay distinct.
        warning_id = established[0]["id"]
        decided = (
            await client.call_tool(
                "decide_leakage_warning",
                {"experiment_id": experiment_id, "warning_id": warning_id, "decision": "exclude"},
            )
        ).data
        assert decided["user_decision"] == "exclude"

        review = (
            await client.call_tool("get_feature_review", {"experiment_id": experiment_id})
        ).data
        selected = [f["feature"] for f in review["features"] if f["selected"]]
        await client.call_tool(
            "decide_features", {"experiment_id": experiment_id, "selected_features": selected}
        )
        applied = (
            await client.call_tool("apply_leakage_exclusions", {"experiment_id": experiment_id})
        ).data
        assert applied["excluded_count"] >= 1


@pytest.mark.anyio
async def test_mutations_are_audited(inline_mcp, inline_container, dataset_uri):
    from backend.audit import AuditEvent
    from ml_engine.io import ExperimentLayout, read_model
    from tests.support.datasets import ARTIFACT_ROOT

    async with Client(inline_mcp) as client:
        created = (
            await client.call_tool(
                "create_experiment",
                {"name": "leaky", "dataset_uri": dataset_uri, "target_column": "churned"},
            )
        ).data
        experiment_id = created["experiment_id"]
        await client.call_tool(
            "set_business_context",
            {"experiment_id": experiment_id, "objective": "o", "prediction_scenario": "s"},
        )

    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, experiment_id)
    events = [
        read_model(inline_container.store, uri, AuditEvent)
        for uri in inline_container.store.list_uris(layout.path("audit") + "/")
    ]
    actions = {e.action for e in events}
    assert {"create_experiment", "set_business_context"} <= actions
    assert all(e.actor == "dev@local" for e in events)


@pytest.mark.anyio
async def test_the_leakage_tools_are_discoverable_and_annotated(mcp):
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "set_business_context" in tools and "investigate_leakage" in tools
    assert tools["get_business_context"].annotations.readOnlyHint is True
    assert tools["get_leakage_investigation"].annotations.readOnlyHint is True
    # mutations are not marked read-only
    assert (
        tools["decide_leakage_warning"].annotations.readOnlyHint
        if tools["decide_leakage_warning"].annotations
        else None
    ) is not True
