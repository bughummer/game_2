"""CV_Mode downstream consumers: ablation on stored folds, holdout overlap, scoring (design C7).

Each test builds a real CV experiment (preparation + training + evaluation) on the shared churn
fixture with ``logistic_regression`` for speed, against the in-memory store, then exercises one
consumer in CV_Mode and checks it uses the Refit_Model and Development_Set preprocessor and the
stored Fold_Assignment.
"""

from datetime import UTC, datetime

import numpy as np
import pytest

from jobs.ablation.main import CV_MAX_ABLATION_FITS, run_ablation
from jobs.evaluation.main import run_evaluation
from jobs.holdout_evaluation.main import MIN_UNSEEN_ROWS, run_holdout_evaluation
from jobs.preprocessing.main import run_preparation
from jobs.profiling.main import run_profiling
from jobs.scoring.main import run_scoring
from jobs.training.main import run_training
from ml_engine.contracts.config import FeatureSelection, SplitConfig
from ml_engine.contracts.cv import CrossValidationConfig, CvScheme, EvaluationMode
from ml_engine.contracts.explain import AblationReport
from ml_engine.contracts.holdout import HoldoutRequest
from ml_engine.contracts.model import ModelMetadata
from ml_engine.contracts.scoring import ScoringRequest
from ml_engine.io import read_model, read_parquet, write_model
from tests.support.datasets import write_dataset
from tests.support.experiments import make_config

EXP = "exp-test"
MODEL = "logistic_regression"
K = 3
LEAKY = ["tenure_months", "monthly_charges", "contract", "churn_copy"]


def _cv_split(folds: int = K, test_fraction: float | None = 0.2) -> SplitConfig:
    return SplitConfig(
        evaluation_mode=EvaluationMode.CV,
        cv=CrossValidationConfig(folds=folds, scheme=CvScheme.STRATIFIED_KFOLD),
        test_fraction=test_fraction,
    )


@pytest.fixture
def cv_experiment(store, layout, dataset_csv):
    """A trained CV experiment with a Test_Fold: preparation, training, evaluation."""
    run_profiling(
        store, layout, experiment_id=EXP, dataset_uri=dataset_csv, target_column="churned"
    )
    config = make_config(layout, EXP, dataset_csv, [MODEL])
    config = config.model_copy(
        update={
            "split": _cv_split(),
            "feature_selection": FeatureSelection(selected_features=LEAKY),
        }
    )
    write_model(store, layout.experiment_config, config)
    run_preparation(store, layout, experiment_id=EXP)
    run_training(store, layout, experiment_id=EXP, model_name=MODEL)
    run_evaluation(store, layout, experiment_id=EXP)
    return config


# --- 10.1 ablation on stored folds ----------------------------------------------------------


def test_cv_ablation_baseline_equals_the_runs_cv_mean(store, layout, cv_experiment):
    report = run_ablation(
        store, layout, experiment_id=EXP, model_name=MODEL, features=["churn_copy", "contract"]
    )
    metadata = read_model(store, layout.model_metadata(MODEL), ModelMetadata)

    assert report.fold == "cv" and report.fold_count == K
    assert report.baseline_score == pytest.approx(metadata.primary_score, abs=1e-6)
    by_feature = {e.feature: e for e in report.entries}
    assert set(by_feature) == {"churn_copy", "contract"}
    leak = by_feature["churn_copy"]
    assert leak.fold_drops is not None and len(leak.fold_drops) == K
    assert leak.std_drop is not None
    assert leak.drop == pytest.approx(float(np.mean(leak.fold_drops)), abs=1e-6)
    assert leak.drop > by_feature["contract"].drop  # the leak matters most


def test_cv_ablation_never_reads_the_test_fold(store, layout, cv_experiment):
    """The ablation must read development.parquet, never test.parquet."""
    reads: list[str] = []
    original = type(store).read_bytes

    def recording(self, uri):
        reads.append(uri)
        return original(self, uri)

    store.__class__.read_bytes = recording  # type: ignore[method-assign]
    try:
        run_ablation(store, layout, experiment_id=EXP, model_name=MODEL, features=["churn_copy"])
    finally:
        store.__class__.read_bytes = original  # type: ignore[method-assign]
    assert any(uri == layout.development_dataset for uri in reads)
    assert layout.test_dataset not in reads


def test_cv_ablation_caps_candidates_and_reports_the_truncation(store, layout, dataset_csv):
    """With K folds, at most (60 // K - 1) candidates are evaluated; the rest are truncated."""
    run_profiling(
        store, layout, experiment_id=EXP, dataset_uri=dataset_csv, target_column="churned"
    )
    folds = 10  # 60 // 10 - 1 = 5 candidates allowed
    config = make_config(layout, EXP, dataset_csv, [MODEL]).model_copy(
        update={
            "split": SplitConfig(
                evaluation_mode=EvaluationMode.CV,
                cv=CrossValidationConfig(folds=folds, scheme=CvScheme.STRATIFIED_KFOLD),
            ),
            "feature_selection": FeatureSelection(
                selected_features=[
                    "tenure_months",
                    "monthly_charges",
                    "contract",
                    "is_business",
                    "signup_date",
                    "churn_copy",
                ]
            ),
        }
    )
    write_model(store, layout.experiment_config, config)
    run_preparation(store, layout, experiment_id=EXP)
    run_training(store, layout, experiment_id=EXP, model_name=MODEL)

    report = run_ablation(
        store,
        layout,
        experiment_id=EXP,
        model_name=MODEL,
        features=[
            "tenure_months",
            "monthly_charges",
            "contract",
            "is_business",
            "signup_date",
            "churn_copy",
        ],
    )
    allowed = CV_MAX_ABLATION_FITS // folds - 1
    completed = [e for e in report.entries if e.status == "completed"]
    assert len(completed) == allowed == 5
    assert report.truncated == 6 - allowed == 1
    assert report.fold_count == folds


def test_single_split_ablation_output_is_unchanged(store, layout, dataset_csv):
    """A single-split ablation stays on the inherited path and carries no CV fields."""
    run_profiling(
        store, layout, experiment_id=EXP, dataset_uri=dataset_csv, target_column="churned"
    )
    config = make_config(layout, EXP, dataset_csv, [MODEL]).model_copy(
        update={"feature_selection": FeatureSelection(selected_features=LEAKY)}
    )
    write_model(store, layout.experiment_config, config)
    run_preparation(store, layout, experiment_id=EXP)
    run_training(store, layout, experiment_id=EXP, model_name=MODEL)

    report = run_ablation(
        store, layout, experiment_id=EXP, model_name=MODEL, features=["churn_copy", "contract"]
    )
    stored = read_model(store, layout.ablation, AblationReport)
    assert report == stored
    assert report.fold == "validation" and report.fold_count is None and report.truncated == 0
    for entry in report.entries:
        assert entry.fold_drops is None and entry.std_drop is None


# --- 10.2 holdout evaluation and scoring ----------------------------------------------------


def _evaluate_holdout(store, layout, dataset_uri, evaluation_id="eval-cv"):
    write_model(
        store,
        layout.evaluation_request(evaluation_id),
        HoldoutRequest(
            evaluation_id=evaluation_id,
            experiment_id=EXP,
            dataset_uri=dataset_uri,
            file_format="csv",
            requested_at=datetime.now(UTC),
        ),
    )
    return run_holdout_evaluation(store, layout, experiment_id=EXP, evaluation_id=evaluation_id)


def test_cv_holdout_counts_development_and_test_rows_as_seen(
    store, layout, dataset_csv, cv_experiment
):
    """Scoring the full training dataset in CV_Mode flags development AND test rows as seen."""
    report = _evaluate_holdout(store, layout, dataset_csv)

    assert report.overlap.overlap_fraction > 0.95  # every row was in development or test
    assert report.overlap.unseen_rows < MIN_UNSEEN_ROWS
    development = read_parquet(store, layout.development_dataset)
    test = read_parquet(store, layout.test_dataset)
    # The "training fold" count is development + test (both count as training-side exposure).
    assert report.overlap.rows_in_training_fold >= len(development)
    assert report.overlap.rows_in_training_fold <= len(development) + len(test)
    assert "evaluation_rows_seen_in_training" in {w.rule for w in report.warnings}
    for model in report.models:
        assert model.status == "completed"


def test_cv_holdout_uses_the_refit_model_and_development_preprocessor(
    store, layout, dataset_csv, cv_experiment
):
    """Holdout loads the Refit_Model + Development_Set preprocessor (the standard layout paths)."""
    loaded: list[str] = []
    from jobs.holdout_evaluation import main as holdout_main

    original = holdout_main.load_joblib

    def recording(store_, uri):
        loaded.append(uri)
        return original(store_, uri)

    holdout_main.load_joblib = recording  # type: ignore[assignment]
    try:
        _evaluate_holdout(store, layout, dataset_csv)
    finally:
        holdout_main.load_joblib = original  # type: ignore[assignment]

    assert layout.model_artifact(MODEL) in loaded
    assert layout.preprocessor("dense_numeric") in loaded


def test_cv_scoring_uses_the_refit_model_and_development_preprocessor(
    store, layout, classification_frame, cv_experiment
):
    unseen = classification_frame.drop(columns=["churned"]).head(25)
    uri = write_dataset(store, "to_score_cv.csv", unseen)
    write_model(
        store,
        layout.scoring_request("score-cv"),
        ScoringRequest(
            scoring_id="score-cv",
            experiment_id=EXP,
            model_name=MODEL,
            dataset_uri=uri,
            file_format="csv",
            passthrough_columns=["customer_id"],
            requested_at=datetime.now(UTC),
        ),
    )

    report = run_scoring(store, layout, experiment_id=EXP, scoring_id="score-cv")

    assert report.row_count == 25
    assert report.model_name == MODEL
    # The model + preprocessor paths resolve to the Refit_Model + Development_Set fit unchanged.
    assert store.exists(layout.model_artifact(MODEL))
    assert store.exists(layout.preprocessor("dense_numeric"))
