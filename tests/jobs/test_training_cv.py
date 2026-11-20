"""CV_Mode training: fold loop, OOF, refit, resume and failure isolation (CV design C5).

Each test runs the real preparation (task 4) on the shared churn fixture with a CV config, then
the training job, against the in-memory store.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jobs._common.runtime import JobError
from jobs.preprocessing.main import run_preparation
from jobs.training import cv as training_cv
from jobs.training.main import main as training_main
from jobs.training.main import run_training
from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.config import ModelSpec, SplitConfig
from ml_engine.contracts.cv import (
    CrossValidationConfig,
    CvScheme,
    EvaluationMode,
    FoldAssignment,
    FoldMetrics,
    FoldResult,
    OofPredictions,
)
from ml_engine.contracts.explain import PermutationImportance
from ml_engine.contracts.model import ModelFailure, ModelMetadata
from ml_engine.contracts.thresholds import ThresholdAnalysis
from ml_engine.evaluation import evaluate
from ml_engine.evaluation.thresholds import threshold_analysis
from ml_engine.io import read_model, read_parquet, write_model, write_parquet
from ml_engine.models import get_plugin
from ml_engine.splitting.folds import development_rows, fold_rows
from tests.support.experiments import make_config, make_record
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.inline_orchestrator import InlineOrchestrator

EXPERIMENT_ID = "exp-test"
MODEL = "logistic_regression"
K = 3


def _cv_split(folds: int = K, test_fraction: float | None = 0.2) -> SplitConfig:
    return SplitConfig(
        evaluation_mode=EvaluationMode.CV,
        cv=CrossValidationConfig(folds=folds, scheme=CvScheme.STRATIFIED_KFOLD),
        test_fraction=test_fraction,
    )


def _prepare(store, layout, dataset_uri, models=(MODEL,), split=None, **update):
    config = make_config(layout, EXPERIMENT_ID, dataset_uri, list(models))
    config = config.model_copy(update={"split": split or _cv_split(), **update})
    write_model(store, layout.experiment_config, config)
    run_preparation(store, layout, experiment_id=EXPERIMENT_ID)
    return config


def _assignment(store, layout) -> FoldAssignment:
    return read_model(store, layout.fold_assignment, FoldAssignment)


class Recording:
    """Delegates to a real plugin and records every build (params) and fit (row count)."""

    def __init__(self, plugin, log: list, fail_on_fit: int | None = None) -> None:
        self._plugin = plugin
        self._log = log
        self._fail_on_fit = fail_on_fit

    def __getattr__(self, name):
        return getattr(self._plugin, name)

    def build(self, params, context):
        self._log.append(("build", dict(params)))
        return self._plugin.build(params, context)

    def fit(self, estimator, x, y, context):
        fits = sum(1 for kind, _ in self._log if kind == "fit")
        self._log.append(("fit", len(x)))
        if self._fail_on_fit is not None and fits == self._fail_on_fit:
            raise RuntimeError("injected fit failure")
        return self._plugin.fit(estimator, x, y, context)


def _recording_factory(log: list, fail_on_fit: int | None = None):
    return lambda name: Recording(get_plugin(name), log, fail_on_fit)


def _fits(log):
    return [value for kind, value in log if kind == "fit"]


def _builds(log):
    return [value for kind, value in log if kind == "build"]


# --- fits, params, metrics ------------------------------------------------------------------


def test_k_plus_one_fits_with_identical_params_and_the_cv_artifact_set(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv)
    assignment = _assignment(store, layout)
    log: list = []

    metadata = run_training(
        store,
        layout,
        experiment_id=EXPERIMENT_ID,
        model_name=MODEL,
        plugin_factory=_recording_factory(log),
    )

    expected = [len(fold_rows(assignment, fold)[0]) for fold in range(K)]
    assert _fits(log) == [*expected, assignment.development_row_count]  # K folds + refit
    builds = _builds(log)
    assert len(builds) == K + 1 and all(params == builds[0] for params in builds)

    for fold in range(K):
        assert store.exists(layout.model_fold_result(MODEL, fold))
        assert store.exists(layout.model_fold_oof(MODEL, fold))
    for uri in (
        layout.model_fold_metrics(MODEL),
        layout.model_oof(MODEL),
        layout.model_oof_payload(MODEL),
        layout.model_thresholds(MODEL),
        layout.model_permutation(MODEL),
        layout.model_artifact(MODEL),
        layout.model_metadata(MODEL),
    ):
        assert store.exists(uri), uri

    stored = read_model(store, layout.model_metadata(MODEL), ModelMetadata)
    assert stored == metadata
    fold_metrics = read_model(store, layout.model_fold_metrics(MODEL), FoldMetrics)
    assert [f.fold_index for f in fold_metrics.folds] == list(range(K))
    primary = [f.metrics.values["roc_auc"] for f in fold_metrics.folds]
    record = metadata.cross_validation
    assert record is not None and record.performance_source == "cv_estimate"
    assert record.folds == K and record.primary_fold_values == primary
    assert record.fold_metrics_uri == layout.model_fold_metrics(MODEL)
    assert record.oof_uri == layout.model_oof(MODEL)
    assert not record.refit_iterations.applicable
    assert metadata.primary_score == pytest.approx(float(np.mean(primary)), abs=1e-12)
    for name in fold_metrics.folds[0].metrics.values:
        values = [f.metrics.values[name] for f in fold_metrics.folds]
        assert metadata.metrics.values[name] == pytest.approx(np.mean(values), abs=1e-12)
    cells = np.sum([f.metrics.confusion_matrix.matrix for f in fold_metrics.folds], axis=0)
    assert metadata.metrics.confusion_matrix.matrix == cells.tolist()
    assert metadata.train_metrics is not None  # the refit's in-sample metrics
    assert metadata.training_row_count == assignment.development_row_count
    assert metadata.validation_row_count == assignment.development_row_count  # every row OOF
    assert metadata.preprocessing.fitted_on == "development"
    assert metadata.artifacts.model_uri == layout.model_artifact(MODEL)
    assert metadata.artifacts.preprocessor_uri == layout.preprocessor("dense_numeric")
    assert all(f.job_name is None and not f.resumed for f in fold_metrics.folds)


def test_each_fold_metric_equals_evaluate_on_that_fold(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv)
    assignment = _assignment(store, layout)
    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)

    development = read_parquet(store, layout.development_dataset)
    development.index = development_rows(assignment)
    for fold in range(K):
        result = read_model(store, layout.model_fold_result(MODEL, fold), FoldResult)
        piece = read_parquet(store, layout.model_fold_oof(MODEL, fold))
        _, validation = fold_rows(assignment, fold)
        assert piece["row_id"].tolist() == validation
        assert result.validation_row_count == len(validation)
        y_true = development.loc[validation, "churned"].astype("string")
        assert piece["y_true"].tolist() == y_true.tolist()
        expected = evaluate(
            ProblemType.BINARY_CLASSIFICATION,
            y_true.reset_index(drop=True),
            piece["y_pred"].to_numpy(),
            y_proba=piece[["proba_0", "proba_1"]].to_numpy(),
            class_labels=["0", "1"],
            positive_class="1",
        )
        assert result.metrics.values == expected.values
        assert result.primary_score == expected.values["roc_auc"]


def test_oof_covers_every_validated_row_and_the_manifest_hashes_the_payload(
    store, layout, dataset_csv
):
    _prepare(store, layout, dataset_csv)
    assignment = _assignment(store, layout)
    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)

    manifest = read_model(store, layout.model_oof(MODEL), OofPredictions)
    payload = store.read_bytes(layout.model_oof_payload(MODEL))
    assert manifest.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert manifest.payload_uri == layout.model_oof_payload(MODEL)
    assert manifest.class_labels == ["0", "1"] and manifest.positive_class == "1"
    oof = read_parquet(store, layout.model_oof_payload(MODEL))
    assert manifest.columns == list(oof.columns)
    assert manifest.row_count == len(oof)
    assert oof["row_id"].tolist() == development_rows(assignment)  # no Test_Fold row
    assert set(oof["row_id"]).isdisjoint(
        r for r, v in zip(assignment.row_ids, assignment.assignments, strict=True) if v < 0
    )


def test_thresholds_equal_threshold_analysis_on_the_oof_scores(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv)
    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)

    stored = read_model(store, layout.model_thresholds(MODEL), ThresholdAnalysis)
    oof = read_parquet(store, layout.model_oof_payload(MODEL))
    expected = threshold_analysis(
        (oof["y_true"] == "1").to_numpy().astype(int),
        oof["proba_1"].to_numpy(),
        experiment_id=EXPERIMENT_ID,
        model_name=MODEL,
        positive_class="1",
    )
    assert stored.source == "oof"
    assert stored.validation_row_count == len(oof)
    ignored = {"generated_at", "source", "note"}
    assert stored.model_dump(exclude=ignored) == expected.model_dump(exclude=ignored)


def test_permutation_importance_is_the_fold_mean_and_std(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv)
    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)

    folds = read_model(store, layout.model_fold_metrics(MODEL), FoldMetrics).folds
    stored = read_model(store, layout.model_permutation(MODEL), PermutationImportance)
    assert stored.fold == "cv" and stored.fold_count == K
    assert stored.fold_baselines == [f.permutation_baseline for f in folds]
    assert stored.baseline_score == pytest.approx(
        np.mean([f.permutation_baseline for f in folds]), abs=1e-6
    )
    for entry in stored.entries:
        drops = [e.mean_drop for f in folds for e in f.permutation if e.feature == entry.feature]
        assert len(drops) == K
        assert entry.mean_drop == pytest.approx(np.mean(drops), abs=1e-6)
        assert entry.std_drop == pytest.approx(np.std(drops, ddof=1), abs=1e-6)


def test_a_failed_fold_permutation_is_excluded_from_the_aggregate(
    store, layout, dataset_csv, monkeypatch
):
    _prepare(store, layout, dataset_csv)
    from ml_engine.evaluation import permutation

    real = permutation.permutation_importance
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("permutation broke")
        return real(*args, **kwargs)

    monkeypatch.setattr(permutation, "permutation_importance", flaky)
    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)

    folds = read_model(store, layout.model_fold_metrics(MODEL), FoldMetrics).folds
    assert folds[1].permutation == [] and folds[1].permutation_baseline is None
    stored = read_model(store, layout.model_permutation(MODEL), PermutationImportance)
    assert stored.fold_count == K - 1 and len(stored.fold_baselines) == K - 1


# --- failure isolation ----------------------------------------------------------------------


def test_a_failed_fold_fails_only_its_run_and_records_the_fold(
    store, layout, dataset_csv, monkeypatch
):
    models = [MODEL, "random_forest"]
    _prepare(store, layout, dataset_csv, models=models)
    log: list = []

    def factory(name):
        if name == MODEL:
            return Recording(get_plugin(name), log, fail_on_fit=2)  # fold 2
        return get_plugin(name)

    monkeypatch.setattr(training_cv, "get_plugin", factory)
    orchestrator = InlineOrchestrator(store)
    orchestrator.start_training(make_record(layout, EXPERIMENT_ID, dataset_csv))

    failure = read_model(store, layout.model_failure(MODEL), ModelFailure)
    assert failure.error_code == "CV_FOLD_FAILED"
    assert failure.details == {"failed_fit": "fold_2", "fold_index": "2", "model": MODEL}
    assert "injected fit failure" in failure.message
    assert not store.exists(layout.model_metadata(MODEL))
    assert not store.exists(layout.model_fold_result(MODEL, 2))
    assert store.exists(layout.model_fold_result(MODEL, 1))  # kept for diagnosis

    other = read_model(store, layout.model_metadata("random_forest"), ModelMetadata)
    assert other.cross_validation is not None and other.primary_score is not None
    assert not store.exists(layout.model_failure("random_forest"))
    assert store.exists(layout.comparison)


def test_a_failed_refit_is_recorded_by_the_cli_with_failed_fit(
    store, layout, dataset_csv, monkeypatch
):
    from jobs._common import runtime

    _prepare(store, layout, dataset_csv)
    log: list = []
    monkeypatch.setattr(
        training_cv, "get_plugin", lambda name: Recording(get_plugin(name), log, fail_on_fit=K)
    )
    monkeypatch.setattr(runtime, "build_store", lambda: store)

    code = training_main(
        ["--experiment-id", EXPERIMENT_ID, "--artifact-base", layout.base, "--model-name", MODEL]
    )

    assert code == 1
    failure = read_model(store, layout.model_failure(MODEL), ModelFailure)
    assert failure.error_code == "CV_REFIT_FAILED"
    assert failure.details == {"failed_fit": "refit", "model": MODEL}
    assert store.exists(layout.model_fold_metrics(MODEL))  # fold results kept
    assert not store.exists(layout.model_metadata(MODEL))
    assert not store.exists(layout.model_artifact(MODEL))


def test_missing_and_mismatched_fold_assignment(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv)
    development = read_parquet(store, layout.development_dataset)

    write_parquet(store, layout.development_dataset, development.iloc[:-1])
    with pytest.raises(JobError) as error:
        run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)
    assert error.value.code == "FOLD_ASSIGNMENT_MISMATCH"

    tampered = development.copy()
    tampered.loc[0, "tenure_months"] = tampered.loc[0, "tenure_months"] + 1
    write_parquet(store, layout.development_dataset, tampered)
    with pytest.raises(JobError) as error:
        run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)
    assert error.value.code == "FOLD_ASSIGNMENT_MISMATCH"

    store.delete(layout.fold_assignment)
    with pytest.raises(JobError) as error:
        run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)
    assert error.value.code == "FOLD_ASSIGNMENT_MISSING"
    assert not store.exists(layout.model_fold_result(MODEL, 0))


# --- writes only under models/{m}/ ----------------------------------------------------------


class WriteRecordingStore(InMemoryObjectStore):
    def __init__(self) -> None:
        super().__init__()
        self.written: list[str] = []

    def write_bytes(self, uri: str, data: bytes) -> None:
        self.written.append(uri)
        super().write_bytes(uri, data)


def test_training_writes_only_under_its_own_model_prefix(layout, classification_frame):
    from tests.support.datasets import write_dataset

    store = WriteRecordingStore()
    uri = write_dataset(store, "churn.csv", classification_frame)
    _prepare(store, layout, uri, models=[MODEL, "random_forest"])
    store.written.clear()

    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)

    own = layout.model_dir(MODEL) + "/"
    shared = {layout.environment, layout.requirements}  # the inherited job-environment record
    outside = [uri for uri in store.written if not uri.startswith(own) and uri not in shared]
    assert outside == []
    model_writes = [uri for uri in store.written if uri.startswith(own)]
    assert model_writes[-1] == layout.model_metadata(MODEL)  # written last: marks success


# --- resume ---------------------------------------------------------------------------------


_VOLATILE = {"resumed", "started_at", "completed_at", "duration_seconds", "job_name"}


def test_a_restart_fits_only_the_remaining_folds_and_gives_equal_fold_metrics(
    store, layout, dataset_csv, monkeypatch
):
    _prepare(store, layout, dataset_csv)
    real_train = training_cv.train_model
    calls = {"n": 0}

    def crash_on_fold_2(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise SystemExit("Spot interruption")  # not an Exception: the process dies
        return real_train(*args, **kwargs)

    monkeypatch.setattr(training_cv, "train_model", crash_on_fold_2)
    with pytest.raises(SystemExit):
        run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)
    assert store.exists(layout.model_fold_result(MODEL, 1))
    assert not store.exists(layout.model_fold_result(MODEL, 2))
    monkeypatch.setattr(training_cv, "train_model", real_train)

    log: list = []
    run_training(
        store,
        layout,
        experiment_id=EXPERIMENT_ID,
        model_name=MODEL,
        plugin_factory=_recording_factory(log),
    )
    assignment = _assignment(store, layout)
    assert _fits(log) == [len(fold_rows(assignment, 2)[0]), assignment.development_row_count]
    resumed = read_model(store, layout.model_fold_metrics(MODEL), FoldMetrics)
    assert [f.resumed for f in resumed.folds] == [True, True, False]

    # A clean run on the same assignment gives the same fold metrics.
    for fold in range(K):
        store.delete(layout.model_fold_result(MODEL, fold))
        store.delete(layout.model_fold_oof(MODEL, fold))
    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)
    clean = read_model(store, layout.model_fold_metrics(MODEL), FoldMetrics)
    assert [f.model_dump(exclude=_VOLATILE) for f in resumed.folds] == [
        f.model_dump(exclude=_VOLATILE) for f in clean.folds
    ]


def test_a_persisted_fold_with_other_params_is_refitted(store, layout, dataset_csv):
    config = _prepare(store, layout, dataset_csv)
    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)

    changed = config.model_copy(update={"models": [ModelSpec(name=MODEL, params={"C": 0.5})]})
    write_model(store, layout.experiment_config, changed)
    log: list = []
    run_training(
        store,
        layout,
        experiment_id=EXPERIMENT_ID,
        model_name=MODEL,
        plugin_factory=_recording_factory(log),
    )
    assert len(_fits(log)) == K + 1
    folds = read_model(store, layout.model_fold_metrics(MODEL), FoldMetrics).folds
    assert not any(f.resumed for f in folds)


# --- checkpoints ----------------------------------------------------------------------------


def test_every_fit_gets_its_own_checkpoint_sub_directory(
    store, layout, dataset_csv, monkeypatch, tmp_path
):
    from jobs.training import checkpointing

    _prepare(store, layout, dataset_csv)
    directories: list[Path] = []

    def recording_factory(directory):
        directories.append(Path(directory))
        return get_plugin

    monkeypatch.setattr(checkpointing, "checkpointing_factory", recording_factory)
    run_training(
        store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL, checkpoint_dir=tmp_path
    )
    assert directories == [tmp_path / f"fold_{i}" for i in range(K)] + [tmp_path / "refit"]


def test_cv_training_writes_no_checkpoints(store, layout, dataset_csv, tmp_path):
    """Stock-image migration (decision B4): checkpoint/resume is dropped, so even a would-be-
    resumable model like xgboost writes no checkpoint files, in any fold or at the root — the
    checkpointing_factory is a no-op. CV fold-level persistence (fold_metrics/oof) is unaffected."""
    _prepare(store, layout, dataset_csv, models=["xgboost"])

    run_training(
        store, layout, experiment_id=EXPERIMENT_ID, model_name="xgboost", checkpoint_dir=tmp_path
    )

    # No checkpoint artifacts anywhere under the checkpoint dir.
    assert not list(tmp_path.rglob("xgb_*"))
    # Training still produced its CV artifacts (fold-level persistence is independent).
    assert store.exists(layout.model_fold_metrics("xgboost"))
    assert store.exists(layout.model_oof("xgboost"))


# --- refit iterations -----------------------------------------------------------------------


class EarlyStopping(Recording):
    """A fake early-stopping plugin: reports a best iteration per fold fit."""

    BEST = [5, 9, 7]

    def best_iteration(self, estimator):
        fits = len(_fits(self._log))
        return self.BEST[fits - 1] if fits <= len(self.BEST) else None

    def fixed_iteration_params(self, iterations):
        return {"max_iter": iterations * 10}


def test_refit_uses_the_maximum_best_iteration_of_the_folds(store, layout, dataset_csv):
    _prepare(store, layout, dataset_csv)
    log: list = []

    metadata = run_training(
        store,
        layout,
        experiment_id=EXPERIMENT_ID,
        model_name=MODEL,
        plugin_factory=lambda name: EarlyStopping(get_plugin(name), log),
    )

    iterations = metadata.cross_validation.refit_iterations
    assert iterations.applicable and iterations.per_fold == [5, 9, 7] and iterations.chosen == 9
    builds = _builds(log)
    assert [b["max_iter"] for b in builds] == [1000, 1000, 1000, 90]
    folds = read_model(store, layout.model_fold_metrics(MODEL), FoldMetrics).folds
    assert [f.best_iteration for f in folds] == [5, 9, 7]


def test_refit_iterations_helper():
    not_applicable = training_cv.refit_iterations([None, None])
    assert not not_applicable.applicable and not_applicable.chosen is None
    assert not_applicable.per_fold == []
    chosen = training_cv.refit_iterations([12, None, 30])
    assert chosen.applicable and chosen.per_fold == [12, 30] and chosen.chosen == 30


# --- regression and the inherited path ------------------------------------------------------


def test_regression_cv_writes_numeric_oof_and_no_thresholds(store, layout, regression_csv):
    from ml_engine.contracts.config import FeatureSelection

    split = SplitConfig(
        strategy="random",
        evaluation_mode=EvaluationMode.CV,
        cv=CrossValidationConfig(folds=K, scheme=CvScheme.KFOLD),
    )
    _prepare(
        store,
        layout,
        regression_csv,
        models=["elastic_net"],
        split=split,
        target_column="price",
        problem_type=ProblemType.REGRESSION,
        primary_metric="rmse",
        feature_selection=FeatureSelection(selected_features=["area_sqm", "rooms", "district"]),
    )

    metadata = run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name="elastic_net")

    oof = read_parquet(store, layout.model_oof_payload("elastic_net"))
    assert list(oof.columns) == ["row_id", "fold_index", "y_true", "y_pred"]
    assert pd.api.types.is_float_dtype(oof["y_pred"]) and len(oof) == 300
    assert not store.exists(layout.model_thresholds("elastic_net"))
    folds = read_model(store, layout.model_fold_metrics("elastic_net"), FoldMetrics).folds
    assert metadata.primary_score == pytest.approx(
        np.mean([f.metrics.values["rmse"] for f in folds]), abs=1e-12
    )
    assert metadata.metrics.confusion_matrix is None


def test_single_split_with_a_test_fold_stays_on_the_inherited_path(
    store, layout, dataset_csv, monkeypatch
):
    def refuse(*_args, **_kwargs):
        raise AssertionError("single split must not dispatch to CV training")

    monkeypatch.setattr(training_cv, "run_cv_training", refuse)
    _prepare(store, layout, dataset_csv, split=SplitConfig(test_fraction=0.2))

    metadata = run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)

    assert metadata.cross_validation is None
    assert not store.exists(layout.model_cv_dir(MODEL))


def test_the_job_name_is_recorded_per_fold(store, layout, dataset_csv, monkeypatch):
    _prepare(store, layout, dataset_csv)
    monkeypatch.setenv("TRAINING_JOB_NAME", "mlf-exp-test-lr-1")

    run_training(store, layout, experiment_id=EXPERIMENT_ID, model_name=MODEL)

    folds = read_model(store, layout.model_fold_metrics(MODEL), FoldMetrics).folds
    assert {f.job_name for f in folds} == {"mlf-exp-test-lr-1"}
