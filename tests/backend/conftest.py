"""Fixtures for the control plane: settings, a fake SageMaker, orchestrators.

``price_table_path`` lives in the root conftest so the MCP suite shares it.
"""

from datetime import UTC, datetime

import pytest

from backend.config import Settings
from backend.orchestration import SageMakerOrchestrator
from backend.pricing import PriceTable
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.fake_sagemaker import FakeSageMaker

ROLE = "arn:aws:iam::123456789012:role/mlf-sagemaker-job"
IMAGE = "123456789012.dkr.ecr.us-east-1.amazonaws.com/ml-factory-stage@sha256:" + "a" * 64
NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


@pytest.fixture
def settings(price_table_path) -> Settings:
    """Budget enforcement OFF by default — the production default (MLOps owns cost)."""
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn=ROLE,
        stage_image_uri=IMAGE,
        price_table_path=price_table_path,
    )


@pytest.fixture
def budget_settings(price_table_path) -> Settings:
    """Budget enforcement ON, for the pricing/guard tests."""
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn=ROLE,
        stage_image_uri=IMAGE,
        price_table_path=price_table_path,
        enforce_budget=True,
    )


@pytest.fixture
def sagemaker() -> FakeSageMaker:
    return FakeSageMaker()


def make_orchestrator(settings, store, sagemaker) -> SageMakerOrchestrator:
    prices = PriceTable.load(settings.price_table_path) if settings.enforce_budget else None
    return SageMakerOrchestrator(settings, store, sagemaker, prices, clock=lambda: NOW)


@pytest.fixture
def orchestrator(settings, store, sagemaker) -> SageMakerOrchestrator:
    return make_orchestrator(settings, store, sagemaker)


@pytest.fixture
def budget_orchestrator(budget_settings, store, sagemaker) -> SageMakerOrchestrator:
    return make_orchestrator(budget_settings, store, sagemaker)
