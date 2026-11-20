"""Unit tests for the cost estimator (design C12; Req 4.5, 5.5, 9.6)."""

import json

import pytest
from pydantic import BaseModel

from backend.agent.estimate import NO_ESTIMATE, CostEstimator
from backend.agent.tools import ToolContext
from backend.config import Settings
from ml_engine.contracts.execution import ComputeSpec

CTX = ToolContext(session=None, principal=None, services=None)
RATE = 0.5  # USD per hour, both kinds


def _table(path, *, verified=True):
    rates = {"ml.m5.large": RATE, "ml.m5.xlarge": RATE * 2}
    path.write_text(
        json.dumps(
            {
                "version": "test-1",
                "source": "unit-test",
                "retrieved_on": "2026-01-01",
                "verified": verified,
                "regions": {"us-east-1": {"training": rates, "processing": rates}},
            }
        ),
        encoding="utf-8",
    )
    return path


def _settings(path, **overrides) -> Settings:
    return Settings(
        price_table_path=path,
        processing_instance_type="ml.m5.large",
        training_instance_type="ml.m5.large",
        processing_max_runtime_seconds=3600,
        training_max_runtime_seconds=7200,
        **overrides,
    )


class NoArgs(BaseModel):
    pass


class TrainArgs(BaseModel):
    models: list[str] | None = None
    evaluation_mode: str | None = None
    cv_folds: int | None = None


class FreeFormArgs(BaseModel):
    compute: ComputeSpec | None = None


@pytest.fixture
def table(tmp_path):
    return _table(tmp_path / "prices.json")


def test_processing_estimate_loads_table_with_budget_off(table):
    settings = _settings(table)
    assert settings.enforce_budget is False
    est = CostEstimator(settings, None).estimate("x", NoArgs(), CTX, estimator="processing")
    assert est.usd == pytest.approx(RATE)  # 1 job x 1 hour
    assert (est.price_source, est.price_version) == ("unit-test", "test-1")
    assert "upper bound" in est.basis


def test_training_counts_models_and_cv_runtime(table):
    estimator = CostEstimator(_settings(table), None)
    two = estimator.estimate("t", TrainArgs(models=["a", "b"]), CTX, estimator="training")
    # preparation 1h + 2 x training 2h + evaluation 1h = 6 instance-hours
    assert two.usd == pytest.approx(6 * RATE)
    cv = estimator.estimate(
        "t", TrainArgs(models=["a"], evaluation_mode="cv", cv_folds=3), CTX, estimator="training"
    )
    # preparation 1h + 1 x training (2h x (3 + 1)) + evaluation 1h = 10 instance-hours
    assert cv.usd == pytest.approx(10 * RATE)


def test_training_defaults_to_max_models_per_run(table):
    est = CostEstimator(_settings(table, max_models_per_run=3), None).estimate(
        "t", TrainArgs(), CTX, estimator="training"
    )
    assert est.usd == pytest.approx((2 + 3 * 2) * RATE)


def test_free_form_uses_requested_compute(table):
    compute = ComputeSpec(instance_type="ml.m5.xlarge", max_runtime_seconds=1800)
    est = CostEstimator(_settings(table), None).estimate(
        "f", FreeFormArgs(compute=compute), CTX, estimator="free_form"
    )
    assert est.usd == pytest.approx(RATE * 2 * 0.5)


def test_no_estimator_key(table):
    est = CostEstimator(_settings(table), None).estimate("x", NoArgs(), CTX)
    assert est.usd is None and NO_ESTIMATE in est.basis


def test_missing_table(tmp_path):
    est = CostEstimator(_settings(tmp_path / "absent.json"), None).estimate(
        "x", NoArgs(), CTX, estimator="processing"
    )
    assert est.usd is None and NO_ESTIMATE in est.basis
    assert est.price_source is None


def test_reserve_refusal_gives_no_estimate(tmp_path):
    unverified = _table(tmp_path / "p.json", verified=False)
    est = CostEstimator(_settings(unverified), None).estimate(
        "x", NoArgs(), CTX, estimator="processing"
    )
    assert est.usd is None and "PRICE_TABLE_UNVERIFIED" in est.basis
    assert est.price_version == "test-1"
    unknown = ComputeSpec(instance_type="ml.c5.large", max_runtime_seconds=60)
    est = CostEstimator(_settings(_table(tmp_path / "q.json")), None).estimate(
        "f", FreeFormArgs(compute=unknown), CTX, estimator="free_form"
    )
    assert est.usd is None and "PRICE_UNKNOWN" in est.basis
