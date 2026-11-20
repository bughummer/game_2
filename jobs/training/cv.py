"""CV_Mode training: fold loop, out-of-fold assembly, refit (ML Factory 2.0, CV design C5).

``run_training`` (adapted) dispatches here when the frozen config is in CV_Mode, passing the
same ``checkpoint_dir`` it was given. Folds run sequentially inside the one training job; each
finished fold is persisted immediately (``fold_{i}.json`` is the completion marker) so a Spot
restart skips completed folds (Req 9.4, D3). After the folds the Refit_Model is trained on the
whole Development_Set and ``metadata.json`` is written last.

Only paths under ``models/{m}/`` are written (Req 6.6). The module reuses the deterministic
``train_model`` core unchanged on every fit, the same ``validation_scorer`` the single-split
path uses for permutation, and the inherited threshold/permutation engines.
"""

import hashlib
import json
import logging
import statistics
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jobs._common.runtime import JobError
from ml_engine.contracts.common import (
    MetricDirection,
    ProblemType,
    Severity,
    WarningCategory,
)
from ml_engine.contracts.config import ExperimentConfig, ModelSpec
from ml_engine.contracts.cv import (
    FoldAssignment,
    FoldMetrics,
    FoldResult,
    OofPredictions,
    RefitIterations,
    RunCvRecord,
)
from ml_engine.contracts.explain import PermutationEntry, PermutationImportance
from ml_engine.contracts.metrics import ConfusionMatrix, MetricSet
from ml_engine.contracts.model import ModelArtifacts, ModelMetadata
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    load_joblib,
    read_model_if_exists,
    read_parquet,
    save_joblib,
    write_model,
    write_parquet,
)
from ml_engine.models import (
    ModelNotAvailableError,
    ModelPlugin,
    UnknownModelError,
    get_plugin,
    is_ensemble_name,
    resolve_plugin,
)
from ml_engine.preprocessing.outliers import OutlierTreatment
from ml_engine.splitting.folds import (
    dataset_fingerprint,
    development_rows,
    fold_rows,
)
from ml_engine.training import TrainingError, train_model

LOGGER = logging.getLogger("ml_factory.jobs.training.cv")

OOF_FOLD_MISSING_CLASS_PROBA = 0.0

#: Columns of an OOF slice / payload.
ROW_ID = "row_id"
FOLD_INDEX = "fold_index"
Y_TRUE = "y_true"
Y_PRED = "y_pred"


# --------------------------------------------------------------------------------------------
# 5.1 OOF assembly and refit-iteration helpers (pure)
# --------------------------------------------------------------------------------------------


class OofAssemblyError(ValueError):
    """``assemble_oof`` rejects slices that overlap, miss a row or carry a foreign row."""


def assemble_oof(slices: list[pd.DataFrame], assignment: FoldAssignment) -> pd.DataFrame:
    """Merge per-fold prediction slices into one out-of-fold frame (Property 9).

    Each slice is a frame carrying at least ``row_id`` and ``fold_index``; its rows must be
    exactly the validation rows of that fold. The result holds one prediction per row validated
    in a CV_Fold, tagged with that fold's index, ordered by ``row_id``; no Test_Fold or
    TRAIN_ONLY row appears. Slices that overlap, miss a validated row, or carry a foreign row
    are rejected.
    """
    expected: dict[int, int] = {}
    for fold in range(assignment.folds):
        _, validation_ids = fold_rows(assignment, fold)
        for row_id in validation_ids:
            expected[row_id] = fold

    seen: dict[int, int] = {}
    pieces: list[pd.DataFrame] = []
    for piece in slices:
        row_ids = [int(value) for value in piece[ROW_ID]]
        folds = [int(value) for value in piece[FOLD_INDEX]]
        if len(set(row_ids)) != len(row_ids):
            raise OofAssemblyError("a slice repeats a row id")
        for row_id, fold in zip(row_ids, folds, strict=True):
            if row_id not in expected:
                raise OofAssemblyError(
                    f"row {row_id} is not a CV validation row but has a prediction"
                )
            if expected[row_id] != fold:
                raise OofAssemblyError(
                    f"row {row_id} belongs to fold {expected[row_id]}, not fold {fold}"
                )
            if row_id in seen:
                raise OofAssemblyError(f"row {row_id} is predicted by two folds")
            seen[row_id] = fold
        if len(piece):
            pieces.append(piece)

    missing = sorted(set(expected) - set(seen))
    if missing:
        raise OofAssemblyError(
            f"{len(missing)} CV validation row(s) have no out-of-fold prediction "
            f"(first: {missing[:5]})"
        )

    if pieces:
        oof = pd.concat(pieces, ignore_index=True)
        oof = oof.sort_values(ROW_ID, kind="mergesort").reset_index(drop=True)
    else:
        oof = pd.DataFrame({ROW_ID: pd.Series(dtype="int64"), FOLD_INDEX: pd.Series(dtype="int64")})
    return oof


def refit_iterations(per_fold_best: list[int | None]) -> RefitIterations:
    """D6: the chosen refit iteration count is the max of the per-fold best iterations.

    When no fold reports a best iteration (the current plugins never early-stop) the record is
    marked not applicable.
    """
    reported = [value for value in per_fold_best if value is not None]
    if not reported:
        return RefitIterations(applicable=False)
    return RefitIterations(applicable=True, per_fold=list(reported), chosen=max(reported))


# --------------------------------------------------------------------------------------------
# Digests for resume identity
# --------------------------------------------------------------------------------------------


def assignment_digest(assignment: FoldAssignment) -> str:
    """sha256 of the assignment JSON (minus ``generated_at``), the fold identity of a run."""
    payload = assignment.model_dump(mode="json", exclude={"generated_at"})
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def params_digest(spec: ModelSpec) -> str:
    """sha256 of the model's hyperparameter overrides, part of the fold identity."""
    canonical = json.dumps(dict(spec.params), sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _payload_sha256(store: ObjectStore, uri: str) -> str:
    return hashlib.sha256(store.read_bytes(uri)).hexdigest()


# --------------------------------------------------------------------------------------------
# 5.3 / 5.4 Fold loop, refit and metadata
# --------------------------------------------------------------------------------------------


def run_cv_training(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
    config: ExperimentConfig,
    spec: ModelSpec,
    plugin_factory: Callable[[str], ModelPlugin] | None,
    checkpoint_dir: Path | None,
) -> ModelMetadata:
    """Train one Run in CV_Mode: K folds, then the Refit_Model. Returns its metadata."""
    assignment = _load_assignment(store, layout)
    development = _load_development(store, layout, assignment)
    strategy = _fit_plugin(
        config, model_name, plugin_factory, checkpoint_dir, None
    ).preprocessing_strategy

    digest = assignment_digest(assignment)
    params = params_digest(spec)
    primary_metric = config.primary_metric
    direction = _direction(config.problem_type, primary_metric)

    fold_results: list[FoldResult] = []
    slices: list[pd.DataFrame] = []
    for fold in range(assignment.folds):
        result, slice_ = _run_or_resume_fold(
            store,
            layout,
            fold=fold,
            assignment=assignment,
            development=development,
            config=config,
            spec=spec,
            model_name=model_name,
            strategy=strategy,
            plugin_factory=plugin_factory,
            checkpoint_dir=checkpoint_dir,
            assignment_digest_value=digest,
            params_digest_value=params,
            primary_metric=primary_metric,
            direction=direction,
        )
        fold_results.append(result)
        slices.append(slice_)

    oof_frame, oof_row_count = _assemble_and_write_oof(
        store,
        layout,
        model_name=model_name,
        experiment_id=experiment_id,
        assignment=assignment,
        slices=slices,
        config=config,
    )
    fold_metrics = FoldMetrics(
        experiment_id=experiment_id,
        model_name=model_name,
        scheme=assignment.scheme,  # type: ignore[arg-type]  # CV_Mode always has a scheme
        folds=fold_results,
    )
    write_model(store, layout.model_fold_metrics(model_name), fold_metrics)

    refit_iters = refit_iterations([result.best_iteration for result in fold_results])
    metadata, estimator = _refit(
        store,
        layout,
        experiment_id=experiment_id,
        model_name=model_name,
        config=config,
        spec=spec,
        development=development,
        strategy=strategy,
        plugin_factory=plugin_factory,
        checkpoint_dir=checkpoint_dir,
        refit_iterations=refit_iters,
    )

    metadata = _rewrite_metadata(
        metadata,
        fold_results=fold_results,
        assignment=assignment,
        oof_row_count=oof_row_count,
        development_row_count=len(development),
        fold_metrics_uri=layout.model_fold_metrics(model_name),
        oof_uri=layout.model_oof(model_name),
        refit_iters=refit_iters,
    )

    if config.problem_type is ProblemType.BINARY_CLASSIFICATION:
        _write_oof_thresholds(
            store,
            layout,
            experiment_id=experiment_id,
            model_name=model_name,
            oof_frame=oof_frame,
            config=config,
        )

    _write_permutation_aggregate(
        store,
        layout,
        experiment_id=experiment_id,
        model_name=model_name,
        fold_results=fold_results,
        primary_metric=primary_metric,
        direction=direction,
        random_seed=config.split.random_seed,
    )

    # Grid points carry a '<base>__<hash>' spec name; stamp it so each ranks/lists as its own
    # model version (grid-search design C5). A plain name is unchanged.
    from jobs.training.main import _name_for_spec

    metadata = _name_for_spec(metadata, model_name)
    model_uri = save_joblib(store, layout.model_artifact(model_name), estimator)
    metadata.artifacts = ModelArtifacts(
        model_uri=model_uri,
        metadata_uri=layout.model_metadata(model_name),
        preprocessor_uri=layout.preprocessor(strategy),
    )
    write_model(store, layout.model_metadata(model_name), metadata)  # last: success marker
    LOGGER.info(
        "%s CV completed over %d folds: %s=%s",
        model_name,
        assignment.folds,
        metadata.primary_metric,
        metadata.primary_score,
    )
    return metadata


# --------------------------------------------------------------------------------------------
# Load + verify
# --------------------------------------------------------------------------------------------


def _load_assignment(store: ObjectStore, layout: ExperimentLayout) -> FoldAssignment:
    assignment = read_model_if_exists(store, layout.fold_assignment, FoldAssignment)
    if assignment is None:
        raise JobError(
            "FOLD_ASSIGNMENT_MISSING",
            "The fold assignment is missing; the preparation stage must run first in CV_Mode.",
        )
    return assignment


def _load_development(
    store: ObjectStore, layout: ExperimentLayout, assignment: FoldAssignment
) -> pd.DataFrame:
    try:
        written = read_parquet(store, layout.development_dataset)
    except FileNotFoundError as error:
        raise JobError(
            "FOLD_ASSIGNMENT_MISMATCH",
            "The Development_Set parquet is missing; it must be written by preparation.",
        ) from error
    if len(written) != assignment.development_row_count:
        raise JobError(
            "FOLD_ASSIGNMENT_MISMATCH",
            f"The Development_Set has {len(written)} row(s) but the assignment expects "
            f"{assignment.development_row_count}.",
            {"written": str(len(written)), "expected": str(assignment.development_row_count)},
        )
    digest = dataset_fingerprint(written, list(written.columns))
    if digest != assignment.development_digest:
        raise JobError(
            "FOLD_ASSIGNMENT_MISMATCH",
            "The Development_Set digest does not match the fold assignment; the data the "
            "assignment was built on differs from what training read.",
        )
    # Re-attach row ids positionally: development.parquet is in ascending development row-id order.
    written.index = pd.Index(development_rows(assignment), name="row_id")
    return written


def _apply_fold_treatment(
    store: ObjectStore,
    layout: ExperimentLayout,
    fold: int,
    config: ExperimentConfig,
    train_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the fold's saved OutlierTreatment to its training rows (drop/cap); else unchanged."""
    if config.treatment is None or config.treatment.is_noop:
        return train_frame
    treatment_uri = layout.fold_outlier_treatment(fold)
    if not store.exists(treatment_uri):
        return train_frame
    treatment: OutlierTreatment = load_joblib(store, treatment_uri)
    target_column = config.target_column
    feature_columns = [c for c in train_frame.columns if c != target_column]
    features = train_frame[feature_columns]
    target = train_frame[target_column] if target_column in train_frame.columns else None
    treated_features, treated_target = treatment.apply(features, target)
    treated = treated_features.copy()
    if treated_target is not None:
        treated[target_column] = treated_target
    return treated[train_frame.columns]


def _refit_treated_development(
    store: ObjectStore,
    layout: ExperimentLayout,
    config: ExperimentConfig,
    development: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the development-level OutlierTreatment to the whole Development_Set for the refit.

    The treatment is saved by preparation (fit on the whole Development_Set) alongside the full
    pipeline, which was fit on those same treated rows — so the refit sees the same
    outlier -> impute -> scale order the folds do.
    """
    if config.treatment is None or config.treatment.is_noop:
        return development
    treatment_uri = layout.outlier_treatment
    if not store.exists(treatment_uri):
        return development
    treatment: OutlierTreatment = load_joblib(store, treatment_uri)
    target_column = config.target_column
    feature_columns = [c for c in development.columns if c != target_column]
    features = development[feature_columns]
    target = development[target_column] if target_column in development.columns else None
    treated_features, treated_target = treatment.apply(features, target)
    treated = treated_features.copy()
    if treated_target is not None:
        treated[target_column] = treated_target
    return treated[development.columns]


# --------------------------------------------------------------------------------------------
# One fold
# --------------------------------------------------------------------------------------------


def _run_or_resume_fold(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    fold: int,
    assignment: FoldAssignment,
    development: pd.DataFrame,
    config: ExperimentConfig,
    spec: ModelSpec,
    model_name: str,
    strategy: str,
    plugin_factory: Callable[[str], ModelPlugin] | None,
    checkpoint_dir: Path | None,
    assignment_digest_value: str,
    params_digest_value: str,
    primary_metric: str,
    direction: MetricDirection,
) -> tuple[FoldResult, pd.DataFrame]:
    resumed = _resume_fold(
        store,
        layout,
        fold=fold,
        model_name=model_name,
        assignment_digest_value=assignment_digest_value,
        params_digest_value=params_digest_value,
    )
    if resumed is not None:
        return resumed

    try:
        return _train_fold(
            store,
            layout,
            fold=fold,
            assignment=assignment,
            development=development,
            config=config,
            spec=spec,
            model_name=model_name,
            strategy=strategy,
            plugin_factory=plugin_factory,
            checkpoint_dir=checkpoint_dir,
            assignment_digest_value=assignment_digest_value,
            params_digest_value=params_digest_value,
            primary_metric=primary_metric,
            direction=direction,
        )
    except JobError:
        raise
    except Exception as error:
        raise JobError(
            "CV_FOLD_FAILED",
            f"Fold {fold} of '{model_name}' failed: {error}",
            {"failed_fit": f"fold_{fold}", "fold_index": str(fold), "model": model_name},
        ) from error


def _resume_fold(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    fold: int,
    model_name: str,
    assignment_digest_value: str,
    params_digest_value: str,
) -> tuple[FoldResult, pd.DataFrame] | None:
    """Load fold ``i`` when its marker matches the current run and its OOF slice exists."""
    existing = read_model_if_exists(store, layout.model_fold_result(model_name, fold), FoldResult)
    if existing is None:
        return None
    if (
        existing.assignment_digest != assignment_digest_value
        or existing.params_digest != params_digest_value
    ):
        return None
    oof_uri = layout.model_fold_oof(model_name, fold)
    if not store.exists(oof_uri):
        return None
    slice_frame = read_parquet(store, oof_uri)
    result = existing.model_copy(update={"resumed": True})
    LOGGER.info("Resuming fold %d of %s from a previous attempt", fold, model_name)
    return result, slice_frame


def _train_fold(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    fold: int,
    assignment: FoldAssignment,
    development: pd.DataFrame,
    config: ExperimentConfig,
    spec: ModelSpec,
    model_name: str,
    strategy: str,
    plugin_factory: Callable[[str], ModelPlugin] | None,
    checkpoint_dir: Path | None,
    assignment_digest_value: str,
    params_digest_value: str,
    primary_metric: str,
    direction: MetricDirection,
) -> tuple[FoldResult, pd.DataFrame]:
    started = datetime.now(UTC)
    train_ids, validation_ids = fold_rows(assignment, fold)
    train_frame = development.loc[train_ids]
    validation_frame = development.loc[validation_ids]
    # Apply this fold's outlier treatment to its training rows before transform/fit; validation
    # rows are left untreated and every one is scored (training-data-treatment design C4).
    train_frame = _apply_fold_treatment(store, layout, fold, config, train_frame)
    pipeline = load_joblib(store, layout.fold_preprocessor(fold, strategy))
    plugin = _fit_plugin(config, model_name, plugin_factory, checkpoint_dir, f"fold_{fold}")

    metadata, estimator = train_model(
        plugin,
        experiment_id=config.experiment_id,
        problem_type=config.problem_type,
        pipeline=pipeline,
        train_frame=train_frame,
        validation_frame=validation_frame,
        target_column=config.target_column,
        param_overrides=dict(spec.params),
        primary_metric=primary_metric,
        random_seed=config.split.random_seed,
        class_weighting=config.class_weighting,
        compute_train_metrics=False,
    )

    warnings: list[AnalysisWarning] = []
    slice_frame = _oof_slice(
        plugin,
        estimator,
        pipeline,
        validation_frame,
        validation_ids,
        config=config,
        development=development,
        fold=fold,
        warnings=warnings,
    )
    entries, baseline = _fold_permutation(
        plugin,
        estimator,
        pipeline,
        train_frame,
        validation_frame,
        config=config,
        primary_metric=primary_metric,
        direction=direction,
    )

    finished = datetime.now(UTC)
    result = FoldResult(
        model_name=model_name,
        fold_index=fold,
        assignment_digest=assignment_digest_value,
        params_digest=params_digest_value,
        train_row_count=len(train_ids),
        validation_row_count=len(validation_ids),
        metrics=metadata.metrics,
        primary_score=metadata.primary_score,
        best_iteration=_best_iteration(plugin, estimator),
        permutation=entries,
        permutation_baseline=baseline,
        started_at=started,
        completed_at=finished,
        duration_seconds=round((finished - started).total_seconds(), 4),
        job_name=_training_job_name(),
        resumed=False,
        warnings=warnings,
    )
    # Slice first, then the marker: a restart that sees the marker also sees the slice.
    write_parquet(store, layout.model_fold_oof(model_name, fold), slice_frame)
    write_model(store, layout.model_fold_result(model_name, fold), result)
    return result, slice_frame


def _oof_slice(
    plugin: ModelPlugin,
    estimator: Any,
    pipeline: Any,
    validation_frame: pd.DataFrame,
    validation_ids: list[int],
    *,
    config: ExperimentConfig,
    development: pd.DataFrame,
    fold: int,
    warnings: list[AnalysisWarning],
) -> pd.DataFrame:
    """One fold's OOF predictions, aligned to the Development_Set class labels (Req 6.7)."""
    target = config.target_column
    x_validation = pipeline.transform(validation_frame)
    predictions = plugin.predict(estimator, x_validation)
    classification = config.problem_type.is_classification
    y_true = validation_frame[target]
    y_pred = np.asarray(predictions)
    columns: dict[str, Any] = {
        ROW_ID: [int(r) for r in validation_ids],
        FOLD_INDEX: fold,
        Y_TRUE: y_true.astype("string").to_numpy() if classification else y_true.to_numpy(),
        Y_PRED: y_pred.astype(str) if classification else y_pred,
    }
    if classification:
        dev_labels = sorted(development[target].astype("string").unique().tolist())
        fold_labels = plugin.class_labels(estimator) or []
        proba = plugin.predict_proba(estimator, x_validation)
        matrix = np.asarray(proba, dtype="float64") if proba is not None else None
        missing: list[str] = []
        for label in dev_labels:
            if matrix is not None and label in fold_labels:
                columns[f"proba_{label}"] = matrix[:, fold_labels.index(label)]
            else:
                columns[f"proba_{label}"] = np.full(
                    len(validation_frame), OOF_FOLD_MISSING_CLASS_PROBA
                )
                if label not in fold_labels:
                    missing.append(label)
        if missing:
            warnings.append(
                AnalysisWarning(
                    rule="cv_fold_missing_class",
                    category=WarningCategory.TARGET,
                    severity=Severity.MEDIUM,
                    message=(
                        f"Fold {fold} had no training rows for class(es) {', '.join(missing)}; "
                        "their out-of-fold probability is 0.0."
                    ),
                    details={"fold_index": str(fold), "classes": ", ".join(missing)},
                )
            )
    return pd.DataFrame(columns)


def _fold_permutation(
    plugin: ModelPlugin,
    estimator: Any,
    pipeline: Any,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    *,
    config: ExperimentConfig,
    primary_metric: str,
    direction: MetricDirection,
) -> tuple[list[PermutationEntry], float | None]:
    """Per-fold permutation importance via ``validation_scorer`` on the fold model (D9)."""
    from jobs.training.main import validation_scorer
    from ml_engine.evaluation.permutation import permutation_importance

    try:
        features = [c for c in validation_frame.columns if c != config.target_column]
        result = permutation_importance(
            validation_frame,
            features,
            validation_scorer(plugin, estimator, pipeline, train_frame, config, primary_metric),
            experiment_id=config.experiment_id,
            model_name=plugin.name,
            primary_metric=primary_metric,
            direction=direction,
            random_seed=config.split.random_seed,
        )
    except Exception:  # best effort, per the inherited pattern
        LOGGER.exception("Per-fold permutation importance failed; continuing")
        return [], None
    if result is None:
        return [], None
    return list(result.entries), result.baseline_score


# --------------------------------------------------------------------------------------------
# OOF assembly + write
# --------------------------------------------------------------------------------------------


def _assemble_and_write_oof(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    model_name: str,
    experiment_id: str,
    assignment: FoldAssignment,
    slices: list[pd.DataFrame],
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, int]:
    try:
        oof_frame = assemble_oof(slices, assignment)
    except OofAssemblyError as error:
        raise JobError(
            "CV_FOLD_FAILED",
            f"Out-of-fold predictions for '{model_name}' are not well formed: {error}",
            {"failed_fit": "oof", "model": model_name},
        ) from error

    payload_uri = layout.model_oof_payload(model_name)
    write_parquet(store, payload_uri, oof_frame)
    class_labels = None
    positive_class = None
    if config.problem_type.is_classification:
        class_labels = [c[len("proba_") :] for c in oof_frame.columns if c.startswith("proba_")]
    if config.problem_type is ProblemType.BINARY_CLASSIFICATION:
        positive_class = _positive_class_from_development(store, layout, config)
    row_count = len(oof_frame)
    manifest = OofPredictions(
        experiment_id=experiment_id,
        model_name=model_name,
        row_count=row_count,
        payload_uri=payload_uri,
        columns=list(oof_frame.columns),
        class_labels=class_labels,
        positive_class=positive_class,
        payload_sha256=_payload_sha256(store, payload_uri),
    )
    write_model(store, layout.model_oof(model_name), manifest)
    return oof_frame, row_count


# --------------------------------------------------------------------------------------------
# Refit + metadata
# --------------------------------------------------------------------------------------------


def _refit(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
    config: ExperimentConfig,
    spec: ModelSpec,
    development: pd.DataFrame,
    strategy: str,
    plugin_factory: Callable[[str], ModelPlugin] | None,
    checkpoint_dir: Path | None,
    refit_iterations: RefitIterations,
) -> tuple[ModelMetadata, Any]:
    from ml_engine.preprocessing.folds import pipeline_metadata

    pipeline = load_joblib(store, layout.preprocessor(strategy))
    plugin = _fit_plugin(config, model_name, plugin_factory, checkpoint_dir, "refit")
    # The Refit_Model is the deployed model; treat the whole Development_Set it trains on. A fresh
    # fit on all development rows is leak-safe (the refit uses them all) and deterministic.
    development = _refit_treated_development(store, layout, config, development)
    overrides = dict(spec.params)
    # D6: an early-stopping plugin refits for the chosen (max) iteration count of the folds.
    fixed = getattr(plugin, "fixed_iteration_params", None)
    if refit_iterations.applicable and refit_iterations.chosen is not None and callable(fixed):
        overrides.update(fixed(refit_iterations.chosen))
    try:
        metadata, estimator = train_model(
            plugin,
            experiment_id=experiment_id,
            problem_type=config.problem_type,
            pipeline=pipeline,
            train_frame=development,
            validation_frame=development,
            target_column=config.target_column,
            param_overrides=overrides,
            primary_metric=config.primary_metric,
            random_seed=config.split.random_seed,
            class_weighting=config.class_weighting,
            compute_train_metrics=False,
        )
        # The Development_Set fit scope lives in the pipeline's cv_fitted_on attribute.
        metadata.preprocessing = pipeline_metadata(pipeline)
        return metadata, estimator
    except (TrainingError, ModelNotAvailableError) as error:
        raise JobError(
            "CV_REFIT_FAILED",
            f"The Refit_Model for '{model_name}' could not be trained: {error}",
            {"failed_fit": "refit", "model": model_name},
        ) from error


def _rewrite_metadata(
    metadata: ModelMetadata,
    *,
    fold_results: list[FoldResult],
    assignment: FoldAssignment,
    oof_row_count: int,
    development_row_count: int,
    fold_metrics_uri: str,
    oof_uri: str,
    refit_iters: RefitIterations,
) -> ModelMetadata:
    """D12: ``metrics`` becomes the CV mean, the refit's in-sample metrics become train_metrics."""
    refit_in_sample = metadata.metrics
    mean_values = _mean_per_metric(fold_results)
    confusion = _sum_confusion(fold_results)
    warnings = _collected_warnings(fold_results)
    primary_fold_values = [result.primary_score for result in fold_results]
    primary_present = [value for value in primary_fold_values if value is not None]
    primary_mean = statistics.fmean(primary_present) if primary_present else None

    cv_metrics = MetricSet(values=mean_values, confusion_matrix=confusion, warnings=warnings)
    record = RunCvRecord(
        scheme=assignment.scheme,  # type: ignore[arg-type]
        folds=assignment.folds,
        primary_fold_values=primary_fold_values,
        fold_metrics_uri=fold_metrics_uri,
        oof_uri=oof_uri,
        refit_iterations=refit_iters,
    )
    return metadata.model_copy(
        update={
            "metrics": cv_metrics,
            "train_metrics": refit_in_sample,
            "primary_score": primary_mean,
            "validation_row_count": oof_row_count,
            "training_row_count": development_row_count,
            "cross_validation": record,
        }
    )


def _mean_per_metric(fold_results: list[FoldResult]) -> dict[str, float]:
    collected: dict[str, list[float]] = {}
    for result in fold_results:
        for name, value in result.metrics.values.items():
            collected.setdefault(name, []).append(value)
    return {name: statistics.fmean(values) for name, values in collected.items()}


def _sum_confusion(fold_results: list[FoldResult]) -> ConfusionMatrix | None:
    matrices = [r.metrics.confusion_matrix for r in fold_results if r.metrics.confusion_matrix]
    if not matrices:
        return None
    labels = matrices[0].labels
    if any(matrix.labels != labels for matrix in matrices):
        return None  # incomparable fold label sets; leave the confusion matrix out
    total = np.zeros((len(labels), len(labels)), dtype=int)
    for matrix in matrices:
        total += np.asarray(matrix.matrix, dtype=int)
    return ConfusionMatrix(labels=labels, matrix=[[int(c) for c in row] for row in total])


def _collected_warnings(fold_results: list[FoldResult]) -> list[AnalysisWarning]:
    warnings: list[AnalysisWarning] = []
    for result in fold_results:
        warnings.extend(result.warnings)
        warnings.extend(result.metrics.warnings)
    return warnings


# --------------------------------------------------------------------------------------------
# Thresholds + permutation aggregate
# --------------------------------------------------------------------------------------------


def _write_oof_thresholds(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
    oof_frame: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """Binary thresholds from pooled OOF positive-class scores, ``source="oof"`` (Req 10.1)."""
    from ml_engine.evaluation.thresholds import threshold_analysis

    try:
        positive = _positive_class_from_development(store, layout, config)
        if positive is None or f"proba_{positive}" not in oof_frame.columns:
            return
        truth = (oof_frame["y_true"].astype("string") == positive).to_numpy().astype(int)
        scores = oof_frame[f"proba_{positive}"].to_numpy(dtype="float64")
        analysis = threshold_analysis(
            truth,
            scores,
            experiment_id=experiment_id,
            model_name=model_name,
            positive_class=positive,
        )
        if analysis is not None:
            write_model(
                store,
                layout.model_thresholds(model_name),
                analysis.model_copy(update={"source": "oof"}),
            )
    except Exception:
        LOGGER.exception("OOF threshold analysis failed for %s; continuing", model_name)


def _write_permutation_aggregate(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
    fold_results: list[FoldResult],
    primary_metric: str,
    direction: MetricDirection,
    random_seed: int,
) -> None:
    """Combine per-fold permutation entries into one fold mean ± std aggregate (Req 10.2)."""
    try:
        per_feature: dict[str, list[float]] = {}
        baselines: list[float] = []
        for result in fold_results:
            if result.permutation_baseline is not None:
                baselines.append(result.permutation_baseline)
            for entry in result.permutation:
                per_feature.setdefault(entry.feature, []).append(entry.mean_drop)
        if not per_feature:
            return
        entries: list[PermutationEntry] = []
        means: dict[str, float] = {}
        for feature, drops in per_feature.items():
            mean_drop = statistics.fmean(drops)
            means[feature] = mean_drop
            std_drop = statistics.stdev(drops) if len(drops) > 1 else 0.0
            entries.append(
                PermutationEntry(
                    feature=feature,
                    mean_drop=round(mean_drop, 6),
                    std_drop=round(std_drop, 6),
                    share=0.0,
                )
            )
        positive_total = sum(max(0.0, value) for value in means.values())
        if positive_total > 0:
            for entry in entries:
                entry.share = round(max(0.0, means[entry.feature]) / positive_total, 6)
        entries.sort(key=lambda entry: entry.mean_drop, reverse=True)
        baseline = round(statistics.fmean(baselines), 6) if baselines else 0.0
        contributing = sum(1 for result in fold_results if result.permutation_baseline is not None)
        importance = PermutationImportance(
            experiment_id=experiment_id,
            model_name=model_name,
            generated_at=datetime.now(UTC),
            primary_metric=primary_metric,
            direction=direction,
            baseline_score=baseline,
            repeats=3,
            random_seed=random_seed,
            fold="cv",
            entries=entries,
            fold_count=contributing,
            fold_baselines=[round(value, 6) for value in baselines],
        )
        write_model(store, layout.model_permutation(model_name), importance)
    except Exception:
        LOGGER.exception("Permutation aggregate failed for %s; continuing", model_name)


# --------------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------------


def _fit_plugin(
    config: ExperimentConfig,
    model_name: str,
    plugin_factory: Callable[[str], ModelPlugin] | None,
    checkpoint_dir: Path | None,
    sub_directory: str | None,
) -> ModelPlugin:
    """The plugin for one fit, each fit with its own checkpoint sub-directory.

    An ensemble name resolves to an ``EnsemblePlugin`` built from the frozen spec (no
    checkpointing). Otherwise, when a ``checkpoint_dir`` is set, the per-fit plugin is built by
    ``checkpointing_factory`` rooted at ``checkpoint_dir / sub_directory`` so a native checkpoint
    never crosses fits; else the provided ``plugin_factory`` (or ``get_plugin``) is used.
    """
    if is_ensemble_name(model_name):
        try:
            return resolve_plugin(config, model_name)
        except UnknownModelError as error:
            raise JobError("UNKNOWN_MODEL", str(error)) from error
    factory: Callable[[str], ModelPlugin] | None = plugin_factory
    if checkpoint_dir is not None and sub_directory is not None:
        from jobs.training import checkpointing

        factory = checkpointing.checkpointing_factory(checkpoint_dir / sub_directory)
    try:
        return (factory or get_plugin)(model_name)
    except UnknownModelError as error:
        raise JobError("UNKNOWN_MODEL", str(error)) from error


def _direction(problem_type: ProblemType, primary_metric: str) -> MetricDirection:
    from ml_engine.evaluation.metrics import metric_direction

    return metric_direction(problem_type, primary_metric)


def _best_iteration(plugin: ModelPlugin, estimator: Any) -> int | None:
    """A plugin's early-stopping best iteration, if it reports one (none do today, D6).

    A plugin may expose ``best_iteration(estimator)``; otherwise the sklearn/XGBoost
    ``best_iteration_``/``best_iteration`` attribute is read if present.
    """
    reporter = getattr(plugin, "best_iteration", None)
    if callable(reporter):
        value = reporter(estimator)
        return int(value) if isinstance(value, (int, np.integer)) else None
    inner = getattr(estimator, "estimator", estimator)
    value = getattr(inner, "best_iteration_", None)
    if value is None:
        value = getattr(inner, "best_iteration", None)
    return int(value) if isinstance(value, (int, np.integer)) else None


def _training_job_name() -> str | None:
    from jobs._common.runtime import training_job_name

    return training_job_name()


def _positive_class_from_development(
    store: ObjectStore, layout: ExperimentLayout, config: ExperimentConfig
) -> str | None:
    from ml_engine.training.runner import _positive_class, build_training_context

    development = read_parquet(store, layout.development_dataset)
    labels = development[config.target_column].astype("string")
    context = build_training_context(
        problem_type=config.problem_type,
        y_train=labels,
        feature_count=0,
        random_seed=config.split.random_seed,
        class_weighting=config.class_weighting,
        categorical_features=[],
    )
    return _positive_class(sorted(labels.unique().tolist()), context)
