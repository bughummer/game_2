"""CV_Mode downstream services: post-training leakage, experiment comparison notes, archiving.

These run the real stages in-process (InlineOrchestrator) over the in-memory store, as in
``test_run_integrity.py``. CV experiments use ``logistic_regression`` for speed and the leaky
``churn_copy`` column so the suspicious-performance rule fires on a near-perfect CV mean.
"""

import pytest

from backend.config import Settings
from backend.repositories import ExperimentRepository
from backend.schemas import TrainingRequest
from backend.services import AnalysisService, ExperimentService, LeakageService
from ml_engine.contracts.comparison import ComparisonReport
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.cv import EvaluationMode
from ml_engine.contracts.runs import ArchivedRun
from ml_engine.io import ExperimentLayout, read_model
from tests.support.datasets import APPROVED_PREFIX, ARTIFACT_ROOT, write_dataset
from tests.support.inline_orchestrator import InlineOrchestrator

LEAKY = ["tenure_months", "monthly_charges", "contract", "churn_copy"]
CLEAN = ["tenure_months", "monthly_charges", "contract"]


@pytest.fixture
def settings(price_table_path) -> Settings:
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        allowed_dataset_prefixes=[APPROVED_PREFIX],
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf",
        stage_image_uri="123456789012.dkr.ecr.us-east-1.amazonaws.com/img@sha256:" + "a" * 64,
        price_table_path=price_table_path,
    )


@pytest.fixture
def services(settings, store):
    repository = ExperimentRepository(settings=settings, store=store)
    orchestrator = InlineOrchestrator(store)
    experiments = ExperimentService(
        settings=settings, repository=repository, store=store, orchestrator=orchestrator
    )
    leakage = LeakageService(settings=settings, repository=repository, store=store)
    analysis = AnalysisService(
        settings=settings,
        repository=repository,
        store=store,
        orchestrator=orchestrator,
        experiments=experiments,
    )
    return {"experiments": experiments, "leakage": leakage, "analysis": analysis}


def _ready(services, store, classification_frame, features, name="churn.csv"):
    uri = write_dataset(store, name, classification_frame)
    record = services["experiments"].create(name="t", dataset_uri=uri, target_column="churned")
    services["experiments"].save_features(record.experiment_id, selected_features=features)
    return record


def _layout(record) -> ExperimentLayout:
    return ExperimentLayout(base=record.artifact_prefix)


# --- 10.3 post-training leakage uses the CV_Summary ------------------------------------------


def test_cv_post_training_flags_suspicious_cv_mean_and_names_the_best_fold(
    services, store, classification_frame
):
    record = _ready(services, store, classification_frame, LEAKY)
    exp_id = record.experiment_id
    services["experiments"].start_training(
        exp_id, TrainingRequest(models=["logistic_regression"], evaluation_mode="cv", cv_folds=3)
    )
    comparison = services["experiments"].comparison(exp_id)
    assert comparison.cross_validation is not None
    best = next(e for e in comparison.models if e.model_name == "logistic_regression")
    assert best.cv is not None and best.cv.metrics["roc_auc"].mean > 0.999

    report = services["leakage"].build_post_training_phase(exp_id)

    suspicious = [w for w in report.warnings if w.method.value == "suspicious_performance"]
    assert suspicious, "a near-perfect CV mean must raise a suspicious-performance warning"
    warning = suspicious[0]
    assert warning.feature == "logistic_regression"
    assert warning.evidence["cv_mean"] >= 0.999
    assert warning.evidence["cv_max_fold"] == pytest.approx(best.cv.metrics["roc_auc"].max)
    assert f"{best.cv.metrics['roc_auc'].max:.4f}" in warning.summary
    assert "suspicious_performance" in report.sources


def test_single_split_post_training_has_no_cv_suspicious_warning(
    services, store, classification_frame
):
    record = _ready(services, store, classification_frame, LEAKY)
    exp_id = record.experiment_id
    services["experiments"].start_training(exp_id, TrainingRequest(models=["logistic_regression"]))

    report = services["leakage"].build_post_training_phase(exp_id)

    assert not [w for w in report.warnings if w.method.value == "suspicious_performance"]


# --- 10.3 experiment comparison notes --------------------------------------------------------


def test_compare_experiments_reports_mode_and_notes_a_mode_mismatch(
    services, store, classification_frame
):
    cv_record = _ready(services, store, classification_frame, CLEAN, name="cv.csv")
    services["experiments"].start_training(
        cv_record.experiment_id,
        TrainingRequest(models=["logistic_regression"], evaluation_mode="cv", cv_folds=3),
    )
    single_record = _ready(services, store, classification_frame, CLEAN, name="single.csv")
    services["experiments"].start_training(
        single_record.experiment_id, TrainingRequest(models=["logistic_regression"])
    )

    result = services["analysis"].compare_experiments(
        [cv_record.experiment_id, single_record.experiment_id]
    )

    by_id = {r["experiment_id"]: r for r in result["experiments"]}
    assert by_id[cv_record.experiment_id]["evaluation_mode"] == "cv"
    assert by_id[cv_record.experiment_id]["cv_folds"] == 3
    assert by_id[single_record.experiment_id]["evaluation_mode"] == "single_split"
    assert by_id[single_record.experiment_id]["cv_folds"] is None
    assert any("measured differently" in note for note in result["comparability_notes"])


# --- 10.4 retrain archiving carries the CV fields --------------------------------------------


def test_a_cv_run_with_a_test_fold_is_archived_together_with_the_cv_fields(
    services, store, classification_frame
):
    record = _ready(services, store, classification_frame, LEAKY)
    exp_id = record.experiment_id
    layout = _layout(record)
    services["experiments"].start_training(
        exp_id,
        TrainingRequest(
            models=["logistic_regression"],
            evaluation_mode="cv",
            cv_folds=3,
            test_fraction=0.2,
        ),
    )
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    assert config.split.evaluation_mode is EvaluationMode.CV

    # Every CV-bearing artifact exists before the archive.
    for uri in (
        layout.fold_assignment,
        layout.fold_preprocessor(0, "dense_numeric"),
        layout.model_fold_metrics("logistic_regression"),
        layout.model_oof("logistic_regression"),
        layout.comparison,
        layout.test_evaluation,
        layout.development_dataset,
        layout.test_dataset,
        layout.model_artifact("logistic_regression"),
    ):
        assert store.exists(uri), uri

    # Retrain: the previous run is archived and cleared first.
    services["experiments"].start_training(
        exp_id,
        TrainingRequest(
            models=["logistic_regression"],
            evaluation_mode="cv",
            cv_folds=3,
            test_fraction=0.2,
        ),
    )

    runs = services["experiments"].list_runs(exp_id)
    assert len(runs) == 1
    manifest = runs[0]
    assert manifest.evaluation_mode is EvaluationMode.CV
    assert manifest.cv_folds == 3
    assert manifest.test_fraction == 0.2

    archive = layout.run_archive(manifest.run_id)
    for relative in (
        "validation/fold_assignment.json",
        "preprocessing/folds/fold_0/preprocessor_dense_numeric.joblib",
        "models/logistic_regression/cv/fold_metrics.json",
        "models/logistic_regression/cv/oof.json",
        "comparison/comparison.json",
        "comparison/test_evaluation.json",
        "datasets/development.parquet",
        "datasets/test.parquet",
        "models/logistic_regression/model.joblib",
    ):
        assert store.exists(f"{archive}/{relative}"), relative

    # The archived CV comparison carries the CV_Summary.
    archived_comparison = read_model(
        store, f"{archive}/comparison/comparison.json", ComparisonReport
    )
    assert archived_comparison.cross_validation is not None

    # The manifest parses as an ArchivedRun with the CV fields present in the JSON.
    stored = read_model(store, layout.run_manifest(manifest.run_id), ArchivedRun)
    assert stored.evaluation_mode is EvaluationMode.CV and stored.cv_folds == 3
