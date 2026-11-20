"""Preparation with a Fold_Assignment: CV_Mode and the Test_Fold (CV design C4).

The artifact set per mode, the development data as written (row-id order and digest), the job
being the only writer of the assignment, derived features computed before rows are assigned,
failure mapping, and the inherited single split staying untouched.
"""

import sys

import pandas as pd
import pytest

from jobs._common.runtime import JobError
from jobs.evaluation.main import run_evaluation
from jobs.preprocessing import cv as preparation_cv
from jobs.preprocessing.main import main as preparation_main
from jobs.preprocessing.main import run_preparation
from jobs.training.main import run_training
from ml_engine.contracts.config import SplitConfig
from ml_engine.contracts.cv import (
    TEST,
    CrossValidationConfig,
    CvScheme,
    EvaluationMode,
    FoldAssignment,
)
from ml_engine.contracts.model import ModelFailure
from ml_engine.contracts.proposals import RatioProposal
from ml_engine.io import load_dataset, load_joblib, read_model, read_parquet, write_model
from ml_engine.preprocessing import folds as fold_preprocessing
from ml_engine.splitting.folds import (
    dataset_fingerprint,
    development_rows,
    fold_rows,
    test_rows,
)
from tests.support.experiments import make_config
from tests.support.golden import collect_masked, read_golden, run_single_split_experiment
from tests.support.in_memory_store import InMemoryObjectStore

EXPERIMENT_ID = "exp-test"
MODELS = ["logistic_regression", "random_forest"]  # both use dense_numeric
STRATEGIES = ["dense_numeric"]


def _cv_split(folds: int = 3, test_fraction: float | None = 0.2, **extra) -> SplitConfig:
    scheme = extra.pop("scheme", CvScheme.STRATIFIED_KFOLD)
    return SplitConfig(
        evaluation_mode=EvaluationMode.CV,
        cv=CrossValidationConfig(folds=folds, scheme=scheme),
        test_fraction=test_fraction,
        **extra,
    )


def _write_config(store, layout, dataset_uri, split: SplitConfig, models=MODELS, **update):
    config = make_config(layout, EXPERIMENT_ID, dataset_uri, models)
    config = config.model_copy(update={"split": split, **update})
    write_model(store, layout.experiment_config, config)
    return config


def _prepare(store, layout):
    return run_preparation(store, layout, experiment_id=EXPERIMENT_ID)


# --- artifact set per mode ------------------------------------------------------------------


def test_cv_with_a_test_fold_writes_the_cv_artifact_set(store, layout, dataset_csv):
    _write_config(store, layout, dataset_csv, _cv_split(folds=3, test_fraction=0.2))

    report = _prepare(store, layout)

    assignment = read_model(store, layout.fold_assignment, FoldAssignment)
    assert assignment.evaluation_mode is EvaluationMode.CV and assignment.folds == 3
    for uri in (layout.development_dataset, layout.test_dataset, layout.preparation):
        assert store.exists(uri), uri
    for strategy in STRATEGIES:
        assert store.exists(layout.preprocessor(strategy))
        for fold in range(3):
            assert store.exists(layout.fold_preprocessor(fold, strategy))
        assert not store.exists(layout.fold_preprocessor(3, strategy))
    assert not store.exists(layout.train_dataset)
    assert not store.exists(layout.validation_dataset)

    development = assignment.development_row_count
    assert report.cross_validation is not None and report.cross_validation.folds == 3
    assert report.cross_validation.development_row_count == development
    assert [f.fold_index for f in report.cross_validation.fold_rows] == [0, 1, 2]
    assert sum(f.validation_row_count for f in report.cross_validation.fold_rows) == development
    assert report.test_fold is not None
    assert report.test_fold.row_count == assignment.test_row_count == 80
    assert report.test_fold.development_row_count == development
    # The CV SplitSummary mapping (design C4).
    assert report.split.strategy == "stratified_kfold" and report.split.stratified
    assert report.split.train_row_count == development
    assert report.split.validation_row_count == development
    assert report.split.validation_fraction_actual == 1.0
    assert set(report.preprocessing) == set(STRATEGIES)
    assert {m.fitted_on for m in report.preprocessing.values()} == {"development"}
    counts = report.cross_validation.fold_output_feature_counts
    assert set(counts) == set(STRATEGIES) and all(len(v) == 3 for v in counts.values())


def test_fold_preprocessors_are_fitted_on_their_training_rows(store, layout, dataset_csv):
    _write_config(store, layout, dataset_csv, _cv_split(folds=3))
    _prepare(store, layout)
    assignment = read_model(store, layout.fold_assignment, FoldAssignment)

    for fold in range(3):
        pipeline = load_joblib(store, layout.fold_preprocessor(fold, "dense_numeric"))
        assert fold_preprocessing.pipeline_metadata(pipeline).fitted_on == f"fold_{fold}_train"
        numeric = {n: t for n, t, _ in pipeline._transformer.transformers_}["numeric"]
        assert numeric.named_steps["scale"].n_samples_seen_ == len(fold_rows(assignment, fold)[0])
    full = load_joblib(store, layout.preprocessor("dense_numeric"))
    numeric = {n: t for n, t, _ in full._transformer.transformers_}["numeric"]
    assert numeric.named_steps["scale"].n_samples_seen_ == assignment.development_row_count


def test_cv_without_a_test_fold_writes_no_test_dataset(store, layout, dataset_csv):
    _write_config(store, layout, dataset_csv, _cv_split(folds=4, test_fraction=None))

    report = _prepare(store, layout)

    assignment = read_model(store, layout.fold_assignment, FoldAssignment)
    assert TEST not in assignment.assignments and assignment.development_row_count == 400
    assert not store.exists(layout.test_dataset)
    assert store.exists(layout.development_dataset)
    assert report.test_fold is None and report.cross_validation.folds == 4


def test_single_split_with_a_test_fold_writes_the_inherited_split_of_the_development_set(
    store, layout, dataset_csv
):
    _write_config(store, layout, dataset_csv, SplitConfig(test_fraction=0.25))

    report = _prepare(store, layout)

    assignment = read_model(store, layout.fold_assignment, FoldAssignment)
    assert assignment.evaluation_mode is EvaluationMode.SINGLE_SPLIT and assignment.folds == 0
    assert assignment.scheme is None
    train = read_parquet(store, layout.train_dataset)
    validation = read_parquet(store, layout.validation_dataset)
    test = read_parquet(store, layout.test_dataset)
    assert len(test) == assignment.test_row_count == 100
    assert len(train) + len(validation) == assignment.development_row_count == 300
    assert len(validation) == len(fold_rows(assignment, 0)[1])
    for strategy in STRATEGIES:
        assert store.exists(layout.preprocessor(strategy))
        assert not store.exists(layout.fold_preprocessor(0, strategy))
    assert not store.exists(layout.development_dataset)
    assert report.cross_validation is None
    assert report.test_fold is not None and report.test_fold.fraction == 0.25
    assert report.split.train_row_count == len(train)
    assert {m.fitted_on for m in report.preprocessing.values()} == {"train"}


# --- development data as written ------------------------------------------------------------


def test_development_parquet_is_in_row_id_order_and_matches_the_digest(store, layout, dataset_csv):
    _write_config(store, layout, dataset_csv, _cv_split(folds=3, test_fraction=0.2))
    report = _prepare(store, layout)
    assignment = read_model(store, layout.fold_assignment, FoldAssignment)

    written = read_parquet(store, layout.development_dataset)
    assert len(written) == assignment.development_row_count
    assert dataset_fingerprint(written, list(written.columns)) == assignment.development_digest
    assert list(written.columns) == [*report.usable_features, "churned"]

    usable = load_dataset(store, dataset_csv, "csv").frame  # no target is missing
    expected = usable.iloc[development_rows(assignment)][list(written.columns)]
    pd.testing.assert_frame_equal(
        written.astype(str), expected.reset_index(drop=True).astype(str), check_dtype=False
    )
    test = read_parquet(store, layout.test_dataset)
    test_ids = test_rows(assignment)
    pd.testing.assert_frame_equal(
        test.astype(str),
        usable.iloc[test_ids][list(test.columns)].reset_index(drop=True).astype(str),
        check_dtype=False,
    )
    assert assignment.dataset_fingerprint == dataset_fingerprint(
        usable, [*report.usable_features, "churned"]
    )


# --- single writer --------------------------------------------------------------------------


class WriteRecordingStore(InMemoryObjectStore):
    """Records which job module wrote each object."""

    def __init__(self) -> None:
        super().__init__()
        self.writers: dict[str, list[str]] = {}

    def write_bytes(self, uri: str, data: bytes) -> None:
        frame = sys._getframe(1)
        writer = "<unknown>"
        while frame is not None:
            module = frame.f_globals.get("__name__", "")
            if module.startswith("jobs.") and not module.startswith("jobs._common"):
                writer = module
                break
            frame = frame.f_back
        self.writers.setdefault(uri, []).append(writer)
        super().write_bytes(uri, data)


def test_only_the_preparation_job_writes_the_fold_assignment(layout, classification_frame):
    from tests.support.datasets import write_dataset

    store = WriteRecordingStore()
    uri = write_dataset(store, "churn.csv", classification_frame)
    _write_config(
        store, layout, uri, SplitConfig(test_fraction=0.2), models=["logistic_regression"]
    )

    _prepare(store, layout)
    written = store.read_bytes(layout.fold_assignment)
    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name="logistic_regression")
    run_evaluation(store, layout, experiment_id=EXPERIMENT_ID)

    assert store.exists(layout.comparison)
    assert store.writers[layout.fold_assignment] == ["jobs.preprocessing.cv"]
    assert store.read_bytes(layout.fold_assignment) == written


# --- derived features -----------------------------------------------------------------------


def test_derived_features_are_computed_before_rows_are_assigned(
    store, layout, dataset_csv, monkeypatch
):
    ratio = RatioProposal(
        name="charges_per_tenure_month",
        numerator="monthly_charges",
        denominator="tenure_months",
        rationale="spend normalised by relationship length",
        available_at_prediction_time=True,
    )
    _write_config(store, layout, dataset_csv, _cv_split(folds=3), derived_features=[ratio])
    seen: list[list[str]] = []
    real_plan_folds = preparation_cv.plan_folds

    def spy(frame, *args, **kwargs):
        seen.append(list(frame.columns))
        return real_plan_folds(frame, *args, **kwargs)

    monkeypatch.setattr(preparation_cv, "plan_folds", spy)

    report = _prepare(store, layout)

    assert seen and "charges_per_tenure_month" in seen[0]
    assert "charges_per_tenure_month" in report.usable_features
    assert "charges_per_tenure_month" in read_parquet(store, layout.development_dataset).columns
    pipeline = load_joblib(store, layout.fold_preprocessor(0, "dense_numeric"))
    assert "charges_per_tenure_month" in pipeline.plan.usable


# --- failures -------------------------------------------------------------------------------


def test_a_test_fold_too_small_fails_with_split_failed_through_the_cli(
    store, layout, dataset_csv, monkeypatch
):
    import jobs._common.runtime as runtime

    _write_config(store, layout, dataset_csv, _cv_split(folds=3, test_fraction=0.002))
    monkeypatch.setattr(runtime, "build_store", lambda: store)

    code = preparation_main(["--experiment-id", EXPERIMENT_ID, "--artifact-base", layout.base])

    assert code == 1
    failure = read_model(store, layout.path("validation/failure.json"), ModelFailure)
    assert failure.error_code == "SPLIT_FAILED"
    assert failure.details["reason"] == "TEST_FOLD_TOO_SMALL"
    assert failure.details["minimum"] == "2"
    assert all(isinstance(value, str) for value in failure.details.values())
    assert not store.exists(layout.fold_assignment)
    assert not store.exists(layout.preparation)


def test_fewer_groups_than_folds_fails_with_split_failed(store, layout, dataset_csv):
    split = _cv_split(
        folds=3,
        test_fraction=None,
        scheme=CvScheme.GROUP_KFOLD,
        strategy="group",
        group_column="contract",  # two distinct values
    )
    _write_config(store, layout, dataset_csv, split)

    with pytest.raises(JobError) as error:
        _prepare(store, layout)

    assert error.value.code == "SPLIT_FAILED"
    assert error.value.details["reason"] == "GROUPS_FEWER_THAN_FOLDS"
    assert error.value.details["group_count"] == "2"
    assert error.value.details["folds"] == "3"
    assert not store.exists(layout.fold_assignment)


def test_a_fold_fit_failure_is_preprocessing_failed_with_the_fold_index(
    store, layout, dataset_csv, monkeypatch
):
    _write_config(store, layout, dataset_csv, _cv_split(folds=3))
    real_fit = fold_preprocessing.fit_pipelines

    def failing(*args, fitted_on="train", **kwargs):
        if fitted_on == "fold_1_train":
            raise ValueError("cannot fit")
        return real_fit(*args, fitted_on=fitted_on, **kwargs)

    monkeypatch.setattr(fold_preprocessing, "fit_pipelines", failing)

    with pytest.raises(JobError) as error:
        _prepare(store, layout)

    assert error.value.code == "PREPROCESSING_FAILED"
    assert error.value.details["fold_index"] == "1"
    assert error.value.details["strategy"] == "dense_numeric"
    assert not store.exists(layout.fold_assignment)


# --- the inherited single split -------------------------------------------------------------


def test_single_split_without_a_test_fold_never_leaves_the_inherited_path(
    store, layout, dataset_csv, monkeypatch
):
    def refuse(*_args, **_kwargs):
        raise AssertionError("single split without a Test_Fold must not dispatch")

    monkeypatch.setattr(preparation_cv, "prepare_with_folds", refuse)
    _write_config(store, layout, dataset_csv, SplitConfig())

    report = _prepare(store, layout)

    assert report.cross_validation is None and report.test_fold is None
    assert store.exists(layout.train_dataset) and store.exists(layout.validation_dataset)
    for uri in (layout.fold_assignment, layout.development_dataset, layout.test_dataset):
        assert not store.exists(uri)


def test_single_split_artifacts_stay_byte_identical_to_the_golden_fixtures():
    store = InMemoryObjectStore()
    layout = run_single_split_experiment(store)
    assert collect_masked(store, layout) == read_golden()
    assert not store.exists(layout.fold_assignment)
