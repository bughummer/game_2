"""Report generation through the MCP tools (requirement 14): JSON/HTML always, PDF when present."""

import pandas as pd
import pytest
from fastmcp import Client

from ml_engine.contracts.report import ReportKind
from ml_engine.io import ExperimentLayout
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT


@pytest.fixture
def dataset_uri(s3_client):
    rows = 300
    frame = pd.DataFrame(
        {
            "tenure": [i % 24 for i in range(rows)],
            "charges": [40.0 + (i % 50) for i in range(rows)],
            "contract": ["monthly" if i % 2 else "annual" for i in range(rows)],
            "churned": [i % 2 for i in range(rows)],
        }
    )
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2],
        Key="curated/churn.csv",
        Body=frame.to_csv(index=False),
    )
    return f"{APPROVED_PREFIX}/churn.csv"


@pytest.mark.anyio
async def test_generate_and_fetch_reports_end_to_end(inline_mcp, inline_container, dataset_uri):
    async with Client(inline_mcp) as client:
        created = (
            await client.call_tool(
                "create_experiment",
                {"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
            )
        ).data
        experiment_id = created["experiment_id"]

        # EDA report is available right after EDA (which ran in-process).
        eda_report = (
            await client.call_tool(
                "generate_report", {"experiment_id": experiment_id, "kind": "eda"}
            )
        ).data
        assert eda_report["kind"] == "eda"
        assert eda_report["json_uri"].endswith("reports/eda.json")
        assert eda_report["html_uri"].endswith("reports/eda.html")
        # PDF depends on WeasyPrint; the flag reports availability honestly.
        assert eda_report["pdf_available"] in (True, False)

        # Run training so the comparison report has a summary to build from.
        review = (
            await client.call_tool("get_feature_review", {"experiment_id": experiment_id})
        ).data
        selected = [f["feature"] for f in review["features"] if f["selected"]]
        await client.call_tool(
            "decide_features", {"experiment_id": experiment_id, "selected_features": selected}
        )
        await client.call_tool(
            "start_training", {"experiment_id": experiment_id, "models": ["logistic_regression"]}
        )

        comparison = (
            await client.call_tool(
                "generate_report", {"experiment_id": experiment_id, "kind": "model_comparison"}
            )
        ).data
        assert comparison["kind"] == "model_comparison"

        fetched = (
            await client.call_tool(
                "get_report", {"experiment_id": experiment_id, "kind": "model_comparison"}
            )
        ).data
        assert fetched["kind"] == "model_comparison"
        titles = {s["title"] for s in fetched["sections"]}
        assert "Model comparison" in titles

    # Artifacts actually landed in S3 (JSON + HTML at least).
    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, experiment_id)
    assert inline_container.store.exists(layout.report_document("eda", "json"))
    assert inline_container.store.exists(layout.report_document("eda", "html"))
    assert inline_container.store.exists(layout.report_document("model_comparison", "json"))


@pytest.mark.anyio
async def test_a_comparison_report_before_training_is_a_clean_error(mcp, container, dataset_uri):
    created = container.experiments.create(
        name="x", dataset_uri=dataset_uri, target_column="churned", user="dev@local"
    )
    from fastmcp.exceptions import ToolError

    async with Client(mcp) as client:
        with pytest.raises(ToolError) as error:
            await client.call_tool(
                "generate_report",
                {"experiment_id": created.experiment_id, "kind": "model_comparison"},
            )
    assert "[ARTIFACT_NOT_READY]" in str(error.value)


@pytest.mark.anyio
async def test_report_tools_are_discoverable(mcp):
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert "generate_report" in tools
    assert tools["get_report"].annotations.readOnlyHint is True
    assert (
        tools["generate_report"].annotations.readOnlyHint
        if tools["generate_report"].annotations
        else None
    ) is not True
    _ = ReportKind  # the kind enum is the tool's contract
