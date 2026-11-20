"""End-to-end training-time data treatment (training-data-treatment design C3/C4, Testing Strategy).

Offline, with the in-memory store and the inline flow: single-split and CV preparation + training
with drop/cap, the applied summary recorded, held-out frames untreated, the fixed order (outlier
handling before scaling), and the byte-identical 'none' path.
"""

import numpy as np

from jobs.preprocessing.main import run_preparation
from jobs.training.cv import run_cv_training
from jobs.training.main import run_training
from ml_engine.contracts.config import ModelSpec, SplitConfig
from ml_engine.contracts.cv import CrossValidationConfig, CvScheme, EvaluationMode, FoldAssignment
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.contracts.treatment import OutlierDetector, OutlierHandling, TreatmentConfig
from ml_engine.io import load_joblib, read_model, read_parquet, write_model
from tests.support.experiments import make_config
from tests.support.golden import collect_masked, read_golden, run_single_split_experiment
from tests.support.in_memory_store import InMemoryObjectStore

EXPERIMENT_ID = "exp-test"
MODELS = ["logistic_regression", "random_forest"]


def _treatment(detector: OutlierDetector, handling: OutlierHandling) -> TreatmentConfig:
    return TreatmentConfig(outlier_detector=detector, outlier_handling=handling)


def _write_config(store, layout, dataset_uri, *, split=None, treatment=None):
    config = make_config(layout, EXPERIMENT_ID, dataset_uri, MODELS)
    update = {}
    if split is not None:
        update["split"] = split
    if treatment is not None:
        update["treatment"] = treatment
    config = config.model_copy(update=update)
    write_model(store, layout.experiment_config, config)
    return config


def _cv_split(folds: int = 3, test_fraction=None) -> SplitConfig:
    return SplitConfig(
        evaluation_mode=EvaluationMode.CV,
        cv=CrossValidationConfig(folds=folds, scheme=CvScheme.STRATIFIED_KFOLD),
        test_fraction=test_fraction,
    )


# --- single split -----------------------------------------------------------------------------


def test_single_split_drop_writes_treated_train_and_records_summary(store, layout, dataset_csv):
    _write_config(
        store,
        layout,
        dataset_csv,
        treatment=_treatment(OutlierDetector.IQR, OutlierHandling.DROP),
    )
    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    assert store.exists(layout.outlier_treatment)
    assert len(report.applied_treatment) == 1
    summary = report.applied_treatment[0]
    assert summary.detector is OutlierDetector.IQR
    assert summary.handling is OutlierHandling.DROP
    assert summary.fitted_on == "train"

    train = read_parquet(store, layout.train_dataset)
    validation = read_parquet(store, layout.validation_dataset)
    # drop removes rows from the training frame; validation is untouched.
    assert len(train) == report.split.train_row_count - summary.rows_dropped
    assert len(validation) == report.split.validation_row_count


def test_single_split_cap_preserves_training_row_count(store, layout, dataset_csv):
    _write_config(
        store,
        layout,
        dataset_csv,
        treatment=_treatment(OutlierDetector.IQR, OutlierHandling.CAP),
    )
    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)
    train = read_parquet(store, layout.train_dataset)
    assert len(train) == report.split.train_row_count
    assert report.applied_treatment[0].rows_capped >= 0


def test_single_split_flag_changes_no_rows(store, layout, dataset_csv):
    _write_config(
        store,
        layout,
        dataset_csv,
        treatment=_treatment(OutlierDetector.ZSCORE, OutlierHandling.FLAG),
    )
    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)
    train = read_parquet(store, layout.train_dataset)
    assert len(train) == report.split.train_row_count
    assert report.applied_treatment[0].rows_flagged >= 0


def test_single_split_trains_end_to_end_with_treatment(store, layout, dataset_csv):
    _write_config(
        store,
        layout,
        dataset_csv,
        treatment=_treatment(OutlierDetector.IQR, OutlierHandling.DROP),
    )
    run_preparation(store, layout, experiment_id=EXPERIMENT_ID)
    metadata = run_training(
        store, layout, experiment_id=EXPERIMENT_ID, model_name="logistic_regression"
    )
    assert metadata.primary_score is not None
    # every validation row is scored (none dropped)
    report = read_model(store, layout.preparation, PreparationReport)
    assert metadata.validation_row_count == report.split.validation_row_count


# --- fixed order: outlier handling before scaling ---------------------------------------------


def test_outlier_handling_precedes_scaling(store, layout, dataset_csv):
    # With drop, the scaler's n_samples_seen_ equals the post-drop training row count: scaling
    # sees post-outlier data, confirming the fixed order (Req 8.5).
    _write_config(
        store,
        layout,
        dataset_csv,
        treatment=_treatment(OutlierDetector.IQR, OutlierHandling.DROP),
    )
    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)
    dropped = report.applied_treatment[0].rows_dropped
    pipeline = load_joblib(store, layout.preprocessor("dense_numeric"))
    numeric = {n: t for n, t, _ in pipeline._transformer.transformers_}["numeric"]
    seen = np.atleast_1d(numeric.named_steps["scale"].n_samples_seen_)
    assert (seen == report.split.train_row_count - dropped).all()


# --- CV ---------------------------------------------------------------------------------------


def test_cv_saves_a_treatment_per_fold_and_records_summaries(store, layout, dataset_csv):
    _write_config(
        store,
        layout,
        dataset_csv,
        split=_cv_split(folds=3),
        treatment=_treatment(OutlierDetector.IQR, OutlierHandling.DROP),
    )
    report = run_preparation(store, layout, experiment_id=EXPERIMENT_ID)

    for fold in range(3):
        assert store.exists(layout.fold_outlier_treatment(fold)), fold
    # development-level treatment for the refit, plus the untreated development.parquet
    assert store.exists(layout.outlier_treatment)
    assert store.exists(layout.development_dataset)
    # one summary per fold + one for the Development_Set fit
    fitted = {s.fitted_on for s in report.applied_treatment}
    assert fitted == {"fold_0_train", "fold_1_train", "fold_2_train", "development"}


def test_cv_trains_end_to_end_with_treatment(store, layout, dataset_csv):
    spec = ModelSpec(name="logistic_regression")
    config = _write_config(
        store,
        layout,
        dataset_csv,
        split=_cv_split(folds=3),
        treatment=_treatment(OutlierDetector.IQR, OutlierHandling.DROP),
    )
    run_preparation(store, layout, experiment_id=EXPERIMENT_ID)
    metadata = run_cv_training(
        store,
        layout,
        experiment_id=EXPERIMENT_ID,
        model_name="logistic_regression",
        config=config,
        spec=spec,
        plugin_factory=None,
        checkpoint_dir=None,
    )
    assert metadata.cross_validation is not None
    # every CV validation row is scored out-of-fold (none removed)
    assignment = read_model(store, layout.fold_assignment, FoldAssignment)
    assert metadata.validation_row_count == assignment.development_row_count


# --- none is byte-identical to today ----------------------------------------------------------


def test_none_treatment_is_byte_identical_golden():
    store = InMemoryObjectStore()
    layout = run_single_split_experiment(store)
    assert collect_masked(store, layout) == read_golden()
