"""Shared fixtures.

Everything runs against the real contracts and the real stage code, offline. Storage is the
test-only ``InMemoryObjectStore`` addressed by ``s3://`` URIs — the same URIs the production
``S3ObjectStore`` uses, held to the same behaviour by the store contract suite. No network, no
AWS account.

Adapted from old_repo: the directory-backed store, ``file://`` roots, the backend container and
the queue settings were removed. The data fixtures are inherited unchanged.
"""

import os

import numpy as np
import pandas as pd
import pytest

from ml_engine.contracts.common import ProblemType
from ml_engine.io import ExperimentLayout
from ml_engine.profiling import profile_dataset
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.in_memory_store import InMemoryObjectStore


def pytest_collection_modifyitems(config, items):
    """Real-AWS tests run only when explicitly asked for (requirement 3.7)."""
    if os.environ.get("ML_FACTORY_AWS_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="real AWS test; set ML_FACTORY_AWS_TESTS=1 to run")
    for item in items:
        if "aws" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _no_ambient_aws(request, monkeypatch):
    """A developer's own AWS setup must never decide what a default-suite test sees or touches.

    Credentials are pointed at nothing so an accidental real client fails fast instead of
    reaching an account. Opt-in real-AWS tests keep the developer's configured credentials.
    """
    if "aws" in request.keywords:
        return
    for name in (
        "AWS_PROFILE",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", "/nonexistent/aws-config")
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", "/nonexistent/aws-credentials")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def artifact_root() -> str:
    return ARTIFACT_ROOT


@pytest.fixture
def approved_prefix() -> str:
    """An approved location a dataset may be referenced from without being uploaded."""
    return APPROVED_PREFIX


@pytest.fixture
def store() -> InMemoryObjectStore:
    return InMemoryObjectStore()


@pytest.fixture
def layout(artifact_root: str) -> ExperimentLayout:
    return ExperimentLayout.for_experiment(artifact_root, "exp-test")


@pytest.fixture
def classification_frame() -> pd.DataFrame:
    """A small dataset carrying every problem the platform is meant to detect."""
    rng = np.random.default_rng(42)
    rows = 400
    tenure = rng.integers(1, 60, rows)
    charges = rng.normal(70, 20, rows)
    contract = rng.choice(["monthly", "annual"], rows, p=[0.6, 0.4])
    risk = 0.05 * charges - 0.04 * tenure + np.where(contract == "monthly", 1.5, -1.0)
    target = (risk + rng.normal(0, 1.0, rows) > 2.0).astype(int)
    frame = pd.DataFrame(
        {
            "customer_id": [f"C{i:05d}" for i in range(rows)],
            "tenure_months": tenure,
            "monthly_charges": np.round(charges, 2),
            "contract": contract,
            "is_business": rng.choice([True, False], rows),
            "signup_date": pd.date_range("2021-01-01", periods=rows, freq="D"),
            "region_code": "EU",
            "notes": rng.choice(
                [
                    "customer provided a long free text comment about the service quality",
                    "the technician visit was handled professionally and on time",
                    "billing was unclear and the price increase was not communicated",
                ],
                rows,
            ),
            "churn_copy": target,
            "churned": target,
        }
    )
    frame.loc[rng.random(rows) < 0.4, "monthly_charges"] = np.nan
    return frame


@pytest.fixture
def regression_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = 300
    area = rng.normal(90, 25, rows).clip(20, 200)
    rooms = np.maximum(1, (area / 30).round())
    district = rng.choice(["centre", "north"], rows)
    price = 1000 * area + 5000 * rooms + rng.normal(0, 15000, rows)
    return pd.DataFrame(
        {
            "area_sqm": np.round(area, 1),
            "rooms": rooms,
            "district": district,
            "price": np.round(price, 2),
        }
    )


@pytest.fixture
def classification_eda(classification_frame: pd.DataFrame):
    return profile_dataset(
        classification_frame,
        experiment_id="exp-test",
        target_column="churned",
        source_uri="memory://classification",
        file_format="csv",
    )


@pytest.fixture
def dataset_csv(store, classification_frame: pd.DataFrame) -> str:
    return write_dataset(store, "churn.csv", classification_frame)


@pytest.fixture
def regression_csv(store, regression_frame: pd.DataFrame) -> str:
    return write_dataset(store, "prices.csv", regression_frame)


@pytest.fixture
def binary_problem() -> ProblemType:
    return ProblemType.BINARY_CLASSIFICATION


@pytest.fixture
def price_table_path(tmp_path):
    """The shipped SageMaker price table, marked verified so budget-enforcement tests do not
    depend on its real review state. Used by the backend and MCP suites."""
    import json

    from backend.config import DEFAULT_PRICE_TABLE

    raw = json.loads(DEFAULT_PRICE_TABLE.read_text(encoding="utf-8"))
    raw["verified"] = True
    raw["version"] = "test"
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path
