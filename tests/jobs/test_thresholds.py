"""Decision-threshold analysis: the pure computation and its place in the training job."""

import numpy as np
import pytest

from jobs.preprocessing.main import run_preparation
from jobs.profiling.main import run_profiling
from jobs.training.main import run_training
from ml_engine.contracts.common import ClassWeighting
from ml_engine.contracts.model import ModelMetadata
from ml_engine.contracts.thresholds import ThresholdAnalysis
from ml_engine.evaluation.thresholds import GRID, threshold_analysis
from ml_engine.io import read_model, write_model
from tests.support.experiments import make_config

EXP = "exp-test"


def _analyse(truth, scores):
    return threshold_analysis(
        np.asarray(truth), np.asarray(scores), experiment_id=EXP, model_name="m", positive_class="1"
    )


# --- the pure function ------------------------------------------------------------------------


def test_perfect_separation_finds_a_threshold_with_f1_of_one():
    analysis = _analyse([0, 0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.35, 0.4, 0.45])

    assert analysis.best_f1.f1 == 1.0
    assert 0.3 < analysis.best_f1.threshold <= 0.35
    # Everything scores below 0.5, so the default threshold predicts no positives at all.
    assert analysis.default.recall == 0.0 and analysis.default.predicted_positive_rate == 0.0


def test_the_best_threshold_is_never_worse_than_the_default_or_the_grid():
    rng = np.random.default_rng(0)
    truth = rng.integers(0, 2, 500)
    scores = np.clip(truth * 0.3 + rng.normal(0.35, 0.2, 500), 0, 1)

    analysis = _analyse(truth, scores)

    assert analysis.best_f1.f1 >= analysis.default.f1
    assert analysis.best_f1.f1 >= max(point.f1 for point in analysis.curve)
    assert [point.threshold for point in analysis.curve] == list(GRID)
    assert analysis.positive_rate == pytest.approx(truth.mean(), abs=1e-6)


def test_recall_falls_as_the_threshold_rises():
    rng = np.random.default_rng(1)
    truth = rng.integers(0, 2, 300)
    scores = rng.random(300)

    recalls = [point.recall for point in _analyse(truth, scores).curve]

    assert recalls == sorted(recalls, reverse=True)


@pytest.mark.parametrize("truth", [[], [1, 1, 1], [0, 0]])
def test_no_analysis_without_both_classes(truth):
    assert _analyse(truth, [0.5] * len(truth)) is None


def test_non_finite_scores_are_ignored():
    analysis = _analyse([0, 1, 0, 1], [0.1, 0.9, float("nan"), 0.8])

    assert analysis.validation_row_count == 3


# --- inside the training job ------------------------------------------------------------------


@pytest.fixture
def prepared(store, layout, dataset_csv):
    def prepare(models, *, weighting=ClassWeighting.AUTO, params=None):
        run_profiling(
            store, layout, experiment_id=EXP, dataset_uri=dataset_csv, target_column="churned"
        )
        config = make_config(layout, EXP, dataset_csv, models)
        config.class_weighting = weighting
        for spec in config.models:
            spec.params = dict((params or {}).get(spec.name, {}))
        write_model(store, layout.experiment_config, config)
        run_preparation(store, layout, experiment_id=EXP)

    return prepare


def test_training_writes_a_threshold_analysis_consistent_with_the_headline_metrics(
    store, layout, prepared
):
    prepared(["logistic_regression"])

    metadata = run_training(store, layout, experiment_id=EXP, model_name="logistic_regression")

    analysis = read_model(store, layout.model_thresholds("logistic_regression"), ThresholdAnalysis)
    headline = metadata.metrics.values
    assert analysis.validation_row_count == metadata.validation_row_count
    # The 0.5 point is the same computation as the headline precision/recall/F1.
    assert analysis.default.precision == pytest.approx(headline["precision"], abs=0.01)
    assert analysis.default.recall == pytest.approx(headline["recall"], abs=0.01)
    assert analysis.default.f1 == pytest.approx(headline["f1"], abs=0.01)
    assert analysis.positive_rate == pytest.approx(headline["positive_class_rate"], abs=1e-6)
    assert analysis.best_f1.f1 >= analysis.default.f1


def test_class_weighting_none_trains_unweighted(store, layout, prepared):
    prepared(["logistic_regression", "xgboost"], weighting=ClassWeighting.NONE)

    for name in ("logistic_regression", "xgboost"):
        run_training(store, layout, experiment_id=EXP, model_name=name)

    logistic = read_model(store, layout.model_metadata("logistic_regression"), ModelMetadata)
    xgboost = read_model(store, layout.model_metadata("xgboost"), ModelMetadata)
    assert logistic.hyperparameters.get("class_weight") in (None, "None")
    assert "scale_pos_weight" not in xgboost.hyperparameters


def test_a_user_scale_pos_weight_replaces_the_automatic_one(store, layout, prepared):
    prepared(["xgboost"], params={"xgboost": {"scale_pos_weight": 7.5}})

    run_training(store, layout, experiment_id=EXP, model_name="xgboost")

    xgboost = read_model(store, layout.model_metadata("xgboost"), ModelMetadata)
    assert xgboost.hyperparameters["scale_pos_weight"] == 7.5
