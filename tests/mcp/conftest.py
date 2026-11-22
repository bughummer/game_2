"""Fixtures for the MCP server: a container over the in-memory store and the fake SageMaker,
plus an in-process FastMCP client so every tool is exercised with no AWS and no network.
"""

import json
from pathlib import Path

import boto3
import pytest
from fastmcp import Client
from moto import mock_aws

from backend.config import Settings
from backend.container import build_container
from mcp_server import build_mcp_server
from tests.backend.conftest import IMAGE, ROLE
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT
from tests.support.fake_sagemaker import FakeLogs, FakeSageMaker

DEV_USER = "dev@local"
BUCKET = ARTIFACT_ROOT.removeprefix("s3://")


@pytest.fixture
def anyio_backend() -> str:
    """Run the async MCP-client tests on asyncio only (no trio dependency)."""
    return "asyncio"


@pytest.fixture
def settings(price_table_path) -> Settings:
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn=ROLE,
        stage_image_uri=IMAGE,
        price_table_path=price_table_path,
        dev_identity=DEV_USER,
    )


@pytest.fixture
def s3_client(monkeypatch):
    """Real (moto) S3 so pre-signed URL generation and the store share one backend."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        client.create_bucket(Bucket=APPROVED_PREFIX.split("/")[2])
        yield client


@pytest.fixture
def sagemaker() -> FakeSageMaker:
    return FakeSageMaker()


@pytest.fixture
def container(settings, s3_client, sagemaker):
    """Container whose orchestrator is the fake SageMaker (tests request shapes + boundaries)."""
    return build_container(
        settings,
        sagemaker_client=sagemaker,
        s3_client=s3_client,
        logs_client=FakeLogs(),
    )


@pytest.fixture(scope="session")
def surface_baseline():
    return json.loads((Path(__file__).parents[1] / "fixtures" / "pre_agent_surface.json").read_text())


@pytest.fixture
def check_tool_results(monkeypatch, surface_baseline):
    """Check response keys on the successful calls made by the offline workflows."""
    call_tool = Client.call_tool

    async def checked(client, name, *args, **kwargs):
        result = await call_tool(client, name, *args, **kwargs)
        if result.data is None:
            assert name in {
                "get_feature_proposals",
                "get_leakage_investigation",
                "get_business_context",
                "get_report",
            }
        else:
            assert isinstance(result.data, dict), name
            assert sorted(result.data) in surface_baseline["result_keys"][name], name
        return result

    monkeypatch.setattr(Client, "call_tool", checked)


@pytest.fixture
def mcp(container, check_tool_results):
    return build_mcp_server(container)


@pytest.fixture
def inline_container(settings, s3_client, sagemaker):
    """Container whose orchestrator runs the real stages in-process, so artifacts actually exist.

    Used for the end-to-end chat flow, where ``get_feature_review`` must read a real eda.json.
    """
    from ml_engine.io import S3ObjectStore
    from tests.support.inline_orchestrator import InlineOrchestrator

    store = S3ObjectStore(s3_client, server_side_encryption=settings.s3_sse)
    return build_container(
        settings,
        store=store,
        orchestrator=InlineOrchestrator(store),
        sagemaker_client=sagemaker,
        s3_client=s3_client,
        logs_client=FakeLogs(),
    )


@pytest.fixture
def inline_mcp(inline_container, check_tool_results):
    return build_mcp_server(inline_container)
