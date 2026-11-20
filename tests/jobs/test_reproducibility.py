"""Reproducibility (design Property 5, requirement 3.6).

Default suite: the same frozen configuration, run twice in-process through the real stages,
gives the same primary metric for every model — the in-process counterpart of the opt-in
two-SageMaker-runs test. Tolerance: 1e-6 absolute or 0.1% relative (handover decision 2).
"""

import math

import pytest

from ml_engine.contracts.comparison import ComparisonReport
from ml_engine.io import ExperimentLayout, read_model, write_model
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.experiments import make_config, make_record
from tests.support.inline_orchestrator import InlineOrchestrator

ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-3
MODELS = ["logistic_regression", "random_forest", "xgboost"]


def within_tolerance(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=ABS_TOLERANCE) or math.isclose(a, b, rel_tol=REL_TOLERANCE)


def _run(store, dataset_csv, experiment_id) -> dict[str, float]:
    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, experiment_id)
    record = make_record(layout, experiment_id, dataset_csv)
    orchestrator = InlineOrchestrator(store)
    orchestrator.start_eda(record)
    write_model(
        store, layout.experiment_config, make_config(layout, experiment_id, dataset_csv, MODELS)
    )
    orchestrator.start_training(record)
    assert orchestrator.failures == []
    comparison = read_model(store, layout.comparison, ComparisonReport)
    return {row.model_name: row.primary_score for row in comparison.models}


def test_the_same_frozen_config_reproduces_every_primary_metric(store, dataset_csv):
    first = _run(store, dataset_csv, "exp-run-a")
    second = _run(store, dataset_csv, "exp-run-b")

    assert set(first) == set(MODELS)
    for model in MODELS:
        assert first[model] is not None
        assert within_tolerance(first[model], second[model]), (model, first[model], second[model])


@pytest.mark.parametrize(
    ("a", "b", "ok"),
    [(0.9, 0.9 + 5e-7, True), (100.0, 100.05, True), (0.9, 0.91, False), (0.0, 1e-5, False)],
)
def test_the_documented_tolerance(a, b, ok):
    assert within_tolerance(a, b) is ok


@pytest.mark.aws
def test_two_sagemaker_runs_reproduce_within_tolerance():  # pragma: no cover - opt-in
    pytest.skip("Needs the deployed stage image and a verified price table (Phase 3 rollout).")
