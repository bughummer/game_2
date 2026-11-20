"""The ML Factory 2.0 analysis stages, in-process: permutation importance, ablation, scoring."""

import io
from datetime import UTC, datetime

import pandas as pd
import pytest

from jobs.ablation.main import run_ablation
from jobs.preprocessing.main import run_preparation
from jobs.profiling.main import run_profiling
from jobs.scoring.main import run_scoring
from jobs.training.main import run_training
from ml_engine.contracts.explain import PermutationImportance
from ml_engine.contracts.model import ModelMetadata
from ml_engine.contracts.scoring import ScoringRequest
from ml_engine.io import read_model, write_model
from tests.jobs.test_jobs import _config
from tests.support.datasets import write_dataset

LEAKY = ["tenure_months", "monthly_charges", "contract", "churn_copy"]


@pytest.fixture
def trained(store, layout, dataset_csv):
    run_profiling(
        store, layout, experiment_id="exp-test", dataset_uri=dataset_csv, target_column="churned"
    )
    config = _config(dataset_csv, layout, ["logistic_regression"])
    config.feature_selection.selected_features = LEAKY
    write_model(store, layout.experiment_config, config)
    run_preparation(store, layout, experiment_id="exp-test")
    run_training(store, layout, experiment_id="exp-test", model_name="logistic_regression")
    return config


def test_training_writes_permutation_importance_that_finds_the_leak(store, layout, trained):
    result = read_model(
        store, layout.model_permutation("logistic_regression"), PermutationImportance
    )
    metadata = read_model(store, layout.model_metadata("logistic_regression"), ModelMetadata)
    assert result.baseline_score == pytest.approx(metadata.primary_score, abs=1e-6)
    assert result.entries[0].feature == "churn_copy"
    assert result.entries[0].share > 0.6


def test_ablation_retrains_without_each_feature_on_identical_folds(store, layout, trained):
    report = run_ablation(
        store,
        layout,
        experiment_id="exp-test",
        model_name="logistic_regression",
        features=["churn_copy", "contract", "not_a_used_feature"],
    )
    metadata = read_model(store, layout.model_metadata("logistic_regression"), ModelMetadata)
    assert report.baseline_score == pytest.approx(metadata.primary_score, abs=1e-6)
    by_feature = {e.feature: e for e in report.entries}
    assert set(by_feature) == {"churn_copy", "contract"}  # unused features are skipped
    assert by_feature["churn_copy"].drop > by_feature["contract"].drop
    assert by_feature["churn_copy"].drop > 0.03  # a perfect leak: AUC 1.0 -> ~0.93 without it
    assert store.exists(layout.ablation)


def test_batch_scoring_writes_predictions_without_needing_the_target(
    store, layout, trained, classification_frame
):
    unseen = classification_frame.drop(columns=["churned"]).head(25)
    uri = write_dataset(store, "to_score.csv", unseen)
    write_model(
        store,
        layout.scoring_request("score-1"),
        ScoringRequest(
            scoring_id="score-1",
            experiment_id="exp-test",
            model_name="logistic_regression",
            dataset_uri=uri,
            file_format="csv",
            passthrough_columns=["customer_id", "missing_col"],
            requested_at=datetime.now(UTC),
        ),
    )

    report = run_scoring(store, layout, experiment_id="exp-test", scoring_id="score-1")

    predictions = pd.read_csv(io.BytesIO(store.read_bytes(report.predictions_uri)))
    assert len(predictions) == 25 == report.row_count
    assert list(predictions.columns[:2]) == ["customer_id", "prediction"]
    assert any(c.startswith("proba_") for c in predictions.columns)
    assert predictions["customer_id"].tolist() == unseen["customer_id"].tolist()
    assert any(w.rule == "passthrough_column_missing" for w in report.warnings)
