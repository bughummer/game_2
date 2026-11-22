"""The MCP server over an in-process client (requirement 9): discovery, validation, auth, async,
bounded outputs, structured errors, and a full chat-style experiment flow — all with no AWS.
"""

import pandas as pd
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from tests.mcp.conftest import DEV_USER
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT

READ_ONLY = {
    "get_experiment",
    "list_experiments",
    "list_datasets",
    "preview_dataset",
    "get_leakage_findings",
    "get_feature_review",
    "get_training_status",
    "get_comparison",
    "get_business_context",
    "get_leakage_investigation",
    "get_report",
    "get_feature_proposals",
    "get_threshold_analysis",
    "get_holdout_evaluation",
    "get_environment",
    "preflight_check",
    "get_execution_status",
    "list_runs",
    "compare_experiments",
    "get_permutation_importance",
    "get_ablation",
    "get_batch_scoring",
    "resolve_stage_image",
}
MUTATIONS = {
    "investigate_leakage_post_training",
    "stop_training",
    "rerun_training",
    "interpret_results",
    "start_ablation",
    "start_batch_scoring",
    "get_download_link",
    "request_dataset_upload",
    "register_dataset",
    "create_experiment",
    "delete_experiment",
    "decide_features",
    "start_training",
    "set_business_context",
    "investigate_leakage",
    "decide_leakage_warning",
    "apply_leakage_exclusions",
    "generate_report",
    "propose_features_tool",
    "approve_features",
    "define_derived_features",
    "start_holdout_evaluation",
}


@pytest.fixture
def dataset_uri(s3_client, classification_frame: pd.DataFrame) -> str:
    key = "curated/churn.csv"
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2], Key=key, Body=classification_frame.to_csv(index=False)
    )
    return f"{APPROVED_PREFIX}/churn.csv"


async def _call(mcp, tool, args=None):
    async with Client(mcp) as client:
        result = await client.call_tool(tool, args or {})
        return result.data


# --- discovery and annotations ----------------------------------------------------------------


@pytest.mark.anyio
async def test_tool_discovery_lists_every_tool(mcp):
    async with Client(mcp) as client:
        names = {t.name for t in await client.list_tools()}
    assert names == READ_ONLY | MUTATIONS


@pytest.mark.anyio
async def test_read_only_tools_are_annotated_and_mutations_are_not(mcp):
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    for name in READ_ONLY:
        assert tools[name].annotations and tools[name].annotations.readOnlyHint is True, name
    for name in MUTATIONS:
        hint = tools[name].annotations.readOnlyHint if tools[name].annotations else None
        assert hint is not True, name


# --- validation and structured errors ---------------------------------------------------------


@pytest.mark.anyio
async def test_an_unknown_experiment_is_a_structured_error(mcp):
    with pytest.raises(ToolError) as error:
        await _call(mcp, "get_experiment", {"experiment_id": "exp-nope"})
    assert "[NOT_FOUND]" in str(error.value)


@pytest.mark.anyio
async def test_an_unapproved_dataset_is_rejected(mcp):
    with pytest.raises(ToolError) as error:
        await _call(
            mcp,
            "create_experiment",
            {"name": "x", "dataset_uri": "s3://somewhere-else/data.csv", "target_column": "y"},
        )
    assert "[VALIDATION_ERROR]" in str(error.value)


@pytest.mark.anyio
async def test_a_local_path_is_rejected_not_treated_as_a_key(mcp):
    with pytest.raises(ToolError) as error:
        await _call(
            mcp,
            "create_experiment",
            {"name": "x", "dataset_uri": "/etc/passwd", "target_column": "y"},
        )
    assert "[VALIDATION_ERROR]" in str(error.value)


# --- bounded outputs (no full dataset through the LLM) -----------------------------------------


@pytest.mark.anyio
async def test_upload_returns_a_presigned_url_not_bytes(mcp):
    data = await _call(mcp, "request_dataset_upload", {"filename": "churn.csv"})
    assert data["dataset_uri"].startswith(f"{ARTIFACT_ROOT}/ml-factory/uploads/")
    assert data["upload_url"].startswith("https://")
    assert data["method"] == "PUT"
    assert "x-amz-server-side-encryption" in data["headers"]


@pytest.mark.anyio
async def test_preview_is_capped(mcp, dataset_uri):
    created = await _call(
        mcp,
        "create_experiment",
        {"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
    )
    data = await _call(
        mcp, "preview_dataset", {"experiment_id": created["experiment_id"], "rows": 10_000}
    )
    assert data["preview_row_cap"] <= 20
    assert len(data["columns"]) <= 100


# --- authorization -----------------------------------------------------------------------------


@pytest.mark.anyio
async def test_a_mutation_on_another_users_experiment_is_refused(mcp, container, dataset_uri):
    created = container.experiments.create(
        name="theirs", dataset_uri=dataset_uri, target_column="churned", user="someone-else"
    )
    with pytest.raises(ToolError) as error:
        await _call(mcp, "delete_experiment", {"experiment_id": created.experiment_id})
    assert "[NOT_AUTHORIZED]" in str(error.value)


# --- the full chat-style flow, async and non-blocking ------------------------------------------


@pytest.mark.anyio
async def test_a_whole_experiment_runs_through_the_tools(inline_mcp, dataset_uri):
    """Create → EDA → review → decide → train → status, all through MCP tools, no AWS.

    The in-process orchestrator runs the real stages, so get_feature_review reads a real
    eda.json and start_training produces a real comparison.
    """
    async with Client(inline_mcp) as client:
        created = (
            await client.call_tool(
                "create_experiment",
                {"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
            )
        ).data
        experiment_id = created["experiment_id"]
        assert created["created_by"] == DEV_USER
        assert created["status"] == "FEATURE_REVIEW"  # EDA ran in-process

        review = (
            await client.call_tool("get_feature_review", {"experiment_id": experiment_id})
        ).data
        assert review["features"] and review["target_column"] == "churned"
        selected = [f["feature"] for f in review["features"] if f["selected"]] or [
            review["features"][0]["feature"]
        ]

        decided = (
            await client.call_tool(
                "decide_features", {"experiment_id": experiment_id, "selected_features": selected}
            )
        ).data
        assert decided["selected_count"] == len(selected)

        started = (
            await client.call_tool(
                "start_training",
                {"experiment_id": experiment_id, "models": ["logistic_regression"]},
            )
        ).data
        assert started["execution_id"] and started["requested_models"] == ["logistic_regression"]

        comparison = (
            await client.call_tool("get_comparison", {"experiment_id": experiment_id})
        ).data
        assert comparison["best_model"] == "logistic_regression"


@pytest.mark.anyio
async def test_a_user_defined_feature_goes_from_chat_to_training(inline_mcp, dataset_uri):
    """The agent sends plain JSON specs; they are validated, approved, profiled and trained on."""
    async with Client(inline_mcp) as client:
        experiment_id = (
            await client.call_tool(
                "create_experiment",
                {"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
            )
        ).data["experiment_id"]

        defined = (
            await client.call_tool(
                "define_derived_features",
                {
                    "experiment_id": experiment_id,
                    "features": [
                        {
                            "op": "ratio",
                            "name": "charges_per_tenure",
                            "numerator": "monthly_charges",
                            "denominator": "tenure_months",
                            "rationale": "spend per month of tenure",
                            "available_at_prediction_time": True,
                        },
                        {
                            "op": "is_missing",
                            "name": "leaky_flag",
                            "column": "churned",
                            "rationale": "reads the target",
                            "available_at_prediction_time": True,
                        },
                    ],
                },
            )
        ).data
        assert defined["added"] == ["charges_per_tenure"]
        assert defined["approved_names"] == ["charges_per_tenure"]
        assert [r["reason"] for r in defined["rejected"]] == ["target_reference"]

        args = {"experiment_id": experiment_id}
        proposals = (await client.call_tool("get_feature_proposals", args)).data
        assert proposals["approved_names"] == defined["approved_names"]
        approved = (await client.call_tool("approve_features", {**args, "approved_names": defined["approved_names"]})).data
        assert approved["approved_names"] == defined["approved_names"]
        review = (
            await client.call_tool("get_feature_review", args)
        ).data
        row = next(f for f in review["features"] if f["feature"] == "charges_per_tenure")
        assert row["origin"] == "derived" and row["selected"] is True
        assert row["dtype"] != "pending"  # the in-process profiling job already ran

        selected = [f["feature"] for f in review["features"] if f["selected"]]
        await client.call_tool(
            "decide_features", {"experiment_id": experiment_id, "selected_features": selected}
        )
        await client.call_tool(
            "start_training", {"experiment_id": experiment_id, "models": ["logistic_regression"]}
        )
        comparison = (
            await client.call_tool("get_comparison", {"experiment_id": experiment_id})
        ).data
        assert comparison["best_model"] == "logistic_regression"


@pytest.mark.anyio
async def test_a_malformed_feature_spec_is_refused_by_the_schema(inline_mcp, dataset_uri):
    async with Client(inline_mcp) as client:
        experiment_id = (
            await client.call_tool(
                "create_experiment",
                {"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
            )
        ).data["experiment_id"]
        with pytest.raises(ToolError):
            await client.call_tool(
                "define_derived_features",
                {
                    "experiment_id": experiment_id,
                    "features": [{"op": "log", "name": "x", "column": "monthly_charges"}],
                },
            )


@pytest.mark.anyio
async def test_training_options_and_threshold_analysis_through_the_tools(inline_mcp, dataset_uri):
    async with Client(inline_mcp) as client:
        experiment_id = (
            await client.call_tool(
                "create_experiment",
                {"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
            )
        ).data["experiment_id"]
        review = (
            await client.call_tool("get_feature_review", {"experiment_id": experiment_id})
        ).data
        selected = [f["feature"] for f in review["features"] if f["selected"]]
        await client.call_tool(
            "decide_features", {"experiment_id": experiment_id, "selected_features": selected}
        )

        with pytest.raises(ToolError, match="not in this training run"):
            await client.call_tool(
                "start_training",
                {
                    "experiment_id": experiment_id,
                    "models": ["logistic_regression"],
                    "model_parameters": {"xgboost": {"scale_pos_weight": 4}},
                },
            )

        started = (
            await client.call_tool(
                "start_training",
                {
                    "experiment_id": experiment_id,
                    "models": ["logistic_regression", "xgboost"],
                    "class_weighting": "auto",
                    "model_parameters": {"xgboost": {"scale_pos_weight": 4}},
                },
            )
        ).data
        assert started["class_weighting"] == "auto"
        assert started["model_parameters"] == {"xgboost": {"scale_pos_weight": 4}}

        thresholds = (
            await client.call_tool("get_threshold_analysis", {"experiment_id": experiment_id})
        ).data
        by_model = {m["model_name"]: m for m in thresholds["models"]}
        assert set(by_model) == {"logistic_regression", "xgboost"}
        for analysis in by_model.values():
            assert len(analysis["curve"]) == 19
            assert analysis["best_f1"]["f1"] >= analysis["default"]["f1"]

        only = (
            await client.call_tool(
                "get_threshold_analysis",
                {"experiment_id": experiment_id, "model_name": "xgboost"},
            )
        ).data
        assert [m["model_name"] for m in only["models"]] == ["xgboost"]


@pytest.mark.anyio
async def test_class_weighting_rejects_unknown_values(inline_mcp, dataset_uri):
    async with Client(inline_mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "start_training", {"experiment_id": "exp-x", "class_weighting": "smote"}
            )


async def _trained_experiment(client, dataset_uri) -> str:
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
    await client.call_tool(
        "start_training", {"experiment_id": experiment_id, "models": ["logistic_regression"]}
    )
    return experiment_id


@pytest.mark.anyio
async def test_holdout_evaluation_through_the_tools(
    inline_mcp, dataset_uri, s3_client, classification_frame
):
    fresh = classification_frame.copy()
    fresh["monthly_charges"] = (fresh["monthly_charges"] + 0.37).fillna(999.37)
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2],
        Key="curated/fresh.csv",
        Body=fresh.to_csv(index=False),
    )
    fresh_uri = f"{APPROVED_PREFIX}/fresh.csv"

    async with Client(inline_mcp) as client:
        experiment_id = await _trained_experiment(client, dataset_uri)
        with pytest.raises(ToolError, match="not been evaluated"):
            await client.call_tool("get_holdout_evaluation", {"experiment_id": experiment_id})

        # The training file itself: flagged as seen.
        on_training = (
            await client.call_tool(
                "start_holdout_evaluation",
                {"experiment_id": experiment_id, "dataset_uri": dataset_uri},
            )
        ).data
        assert on_training["status"] == "completed"
        assert on_training["overlap"]["overlap_fraction"] > 0.95
        assert on_training["models"][0]["metrics_unseen_rows"] is None
        assert "evaluation_rows_seen_in_training" in {w["rule"] for w in on_training["warnings"]}

        # A fresh file: scored on unseen rows, with the validation threshold re-checked.
        on_fresh = (
            await client.call_tool(
                "start_holdout_evaluation",
                {"experiment_id": experiment_id, "dataset_uri": fresh_uri},
            )
        ).data
        model = on_fresh["models"][0]
        assert on_fresh["overlap"]["unseen_rows"] == on_fresh["row_count"]
        assert model["unseen_rows_score"] is not None
        assert model["at_validation_best_threshold"]["f1"] is not None

        latest = (
            await client.call_tool("get_holdout_evaluation", {"experiment_id": experiment_id})
        ).data
        assert latest["evaluation_id"] == on_fresh["evaluation_id"]
        assert latest["other_evaluation_ids"] == [on_training["evaluation_id"]]


@pytest.mark.anyio
async def test_holdout_evaluation_needs_training_and_an_approved_location(inline_mcp, dataset_uri):
    async with Client(inline_mcp) as client:
        experiment_id = (
            await client.call_tool(
                "create_experiment",
                {"name": "churn", "dataset_uri": dataset_uri, "target_column": "churned"},
            )
        ).data["experiment_id"]
        with pytest.raises(ToolError, match="Train the experiment"):
            await client.call_tool(
                "start_holdout_evaluation",
                {"experiment_id": experiment_id, "dataset_uri": dataset_uri},
            )

        await client.call_tool(
            "decide_features",
            {"experiment_id": experiment_id, "selected_features": ["tenure_months"]},
        )
        await client.call_tool(
            "start_training", {"experiment_id": experiment_id, "models": ["logistic_regression"]}
        )
        with pytest.raises(ToolError, match="not approved"):
            await client.call_tool(
                "start_holdout_evaluation",
                {"experiment_id": experiment_id, "dataset_uri": "s3://some-other-bucket/x.csv"},
            )


@pytest.mark.anyio
async def test_the_training_environment_is_available_through_the_tools(inline_mcp, dataset_uri):
    import numpy

    async with Client(inline_mcp) as client:
        experiment_id = await _trained_experiment(client, dataset_uri)

        environment = (
            await client.call_tool("get_environment", {"experiment_id": experiment_id})
        ).data

        assert environment["key_packages"]["numpy"] == numpy.__version__
        assert f"numpy=={numpy.__version__}" in environment["requirements_txt"].splitlines()
        assert environment["requirements_uri"].endswith("/environment/requirements.txt")
        assert environment["requirements_truncated"] is False


@pytest.mark.anyio
async def test_post_training_analysis_through_the_tools(
    inline_mcp, dataset_uri, s3_client, classification_frame
):
    """Pre-flight → train → permutation → ablation → post-training leakage → scoring → download
    → rerun → runs → compare, all through MCP with no AWS."""
    to_score = classification_frame.drop(columns=["churned"]).head(10)
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2],
        Key="curated/to_score.csv",
        Body=to_score.to_csv(index=False),
    )
    async with Client(inline_mcp) as client:
        experiment_id = await _trained_experiment(client, dataset_uri)

        check = (await client.call_tool("preflight_check", {"experiment_id": experiment_id})).data
        assert check["ready"] is True

        permutation = (
            await client.call_tool("get_permutation_importance", {"experiment_id": experiment_id})
        ).data
        assert permutation["models"][0]["entries"]

        ablation = (
            await client.call_tool(
                "start_ablation", {"experiment_id": experiment_id, "features": ["contract"]}
            )
        ).data
        assert ablation["features"] == ["contract"]
        result = (await client.call_tool("get_ablation", {"experiment_id": experiment_id})).data
        assert result["status"] == "completed"

        post = (
            await client.call_tool(
                "investigate_leakage_post_training", {"experiment_id": experiment_id}
            )
        ).data
        assert post["phase"] == "post_training"

        scoring = (
            await client.call_tool(
                "start_batch_scoring",
                {
                    "experiment_id": experiment_id,
                    "dataset_uri": f"{APPROVED_PREFIX}/to_score.csv",
                    "passthrough_columns": ["customer_id"],
                },
            )
        ).data
        assert scoring["status"] == "completed" and scoring["row_count"] == 10
        assert (await client.call_tool("get_batch_scoring", {"experiment_id": experiment_id})).data == scoring
        link = (
            await client.call_tool(
                "get_download_link",
                {"experiment_id": experiment_id, "uri": scoring["predictions_uri"]},
            )
        ).data
        assert link["download_url"].startswith("https://")
        with pytest.raises(ToolError, match="does not belong"):
            await client.call_tool(
                "get_download_link",
                {"experiment_id": experiment_id, "uri": f"{ARTIFACT_ROOT}/elsewhere/x.csv"},
            )

        await client.call_tool("rerun_training", {"experiment_id": experiment_id})
        runs = (await client.call_tool("list_runs", {"experiment_id": experiment_id})).data
        assert len(runs["runs"]) == 1

        other = await _trained_experiment(client, dataset_uri)
        compared = (
            await client.call_tool(
                "compare_experiments", {"experiment_ids": [experiment_id, other]}
            )
        ).data
        assert [r["experiment_id"] for r in compared["experiments"]] == [experiment_id, other]


@pytest.mark.anyio
async def test_execution_status_and_stop_through_the_tools(mcp, container, dataset_uri):
    """Against the fake SageMaker: per-job status is reported, and training can be stopped."""
    async with Client(mcp) as client:
        created = (
            await client.call_tool(
                "create_experiment",
                {"name": "s", "dataset_uri": dataset_uri, "target_column": "churned"},
            )
        ).data
        experiment_id = created["experiment_id"]
        status = (
            await client.call_tool("get_execution_status", {"experiment_id": experiment_id})
        ).data
        assert status["kind"] == "eda"
        assert status["jobs"][0]["stage"] == "profiling"
        with pytest.raises(ToolError, match="No training is running"):
            await client.call_tool("stop_training", {"experiment_id": experiment_id})


@pytest.mark.anyio
async def test_pre_agent_surface_matches_baseline(mcp, dataset_uri, surface_baseline):
    from backend.api.app import _collect_routes

    baseline = surface_baseline
    assert baseline["result_keys"].keys() == baseline["parameters"].keys()
    assert (
        sorted(
            f"{method} {path}"
            for method, path, _ in _collect_routes()
            if not path.startswith("/agent/")
        )
        == baseline["routes"]
    )
    arguments = {
        "list_experiments": {},
        "list_datasets": {},
        "register_dataset": {"dataset_uri": dataset_uri},
        "request_dataset_upload": {"filename": "training.csv"},
    }
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert {
            tool.name: [
                key if key in tool.inputSchema.get("required", []) else key + "?"
                for key in tool.inputSchema["properties"]
            ]
            for tool in tools
        } == baseline["parameters"]
        for tool, args in arguments.items():
            await client.call_tool(tool, args)


@pytest.mark.anyio
async def test_existing_tool_response_contracts(
    inline_mcp, inline_container, dataset_uri, monkeypatch
):
    from datetime import UTC, datetime

    from ml_engine.contracts.proposals import FeatureProposalReport
    from ml_engine.contracts.reasoning import ReasoningReport

    async with Client(inline_mcp) as client:
        created = (
            await client.call_tool(
                "create_experiment",
                {
                    "name": "contracts",
                    "dataset_uri": dataset_uri,
                    "target_column": "churned",
                },
            )
        ).data
        eid = created["experiment_id"]
        args = {"experiment_id": eid}
        assert (await client.call_tool("get_experiment", args)).data == created
        for name in ("get_leakage_findings", "get_training_status"):
            assert (await client.call_tool(name, args)).data["experiment_id"] == eid
        assert (await client.call_tool("resolve_stage_image", {})).data["source"] == "config"

        # Supply the model-generated artifacts at the service boundary. The real MCP
        # wrappers still serialize the contracts, resolve identity, and write the audits.
        now = datetime.now(UTC)
        proposals = FeatureProposalReport(experiment_id=eid, generated_at=now)
        reasoning = ReasoningReport(
            experiment_id=eid, generated_at=now, model_id="offline", executive_summary="Checked."
        )
        monkeypatch.setattr(inline_container.proposals, "generate", lambda *a, **kw: proposals)
        monkeypatch.setattr(inline_container.reasoning, "interpret", lambda *a, **kw: reasoning)
        for name, report in (
            ("propose_features_tool", proposals),
            ("interpret_results", reasoning),
        ):
            result = await client.call_tool(name, args)
            assert result.data == report.model_dump(mode="json")
        monkeypatch.setattr(inline_container.experiments, "stop_training", lambda eid: "execution")
        assert (await client.call_tool("stop_training", args)).data["execution_id"] == "execution"
        assert (await client.call_tool("delete_experiment", args)).data["status"] == "deleted"
