"""Validation job: score chosen model versions on an uploaded labeled dataset (control-ui-api C5).

A SageMaker Processing job that generalises holdout evaluation. It loads the frozen config, derives
the run's features the same way, and for each requested Model_Version loads that version's model and
preprocessing (from the current run or an archived run) and scores it. When the dataset carries the
label column, the label is dropped from the model input and kept for metrics and a fresh threshold
analysis computed on the validation labels; when it is absent, only predictions and their
distribution are produced. Nothing is re-fit, so the dataset cannot influence a model.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from jobs._common.environment import capture
from jobs._common.runtime import JobError, base_parser, run_entrypoint

# Reuse the holdout job's derivation, overlap and environment helpers verbatim — validation is
# holdout generalised to target specific versions and to recompute thresholds on these labels.
from jobs.holdout_evaluation.main import (
    MIN_UNSEEN_ROWS,
    _derive,
    _environment_warnings,
    _overlap,
    _overlap_warnings,
    _seen_and_training_frames,
)
from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.environment import RuntimeEnvironment
from ml_engine.contracts.model import ModelMetadata
from ml_engine.contracts.model_version import CURRENT_RUN, ArtifactRef, ModelVersion
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.contracts.validation import (
    ValidationModelResult,
    ValidationReport,
    ValidationRequest,
)
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.evaluation import evaluate
from ml_engine.evaluation.evaluator import _positive_scores
from ml_engine.evaluation.thresholds import threshold_analysis
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    load_dataset,
    load_joblib,
    read_model,
    read_model_if_exists,
    write_model,
)
from ml_engine.models import get_plugin
from ml_engine.training.runner import _positive_class, build_training_context

LOGGER = logging.getLogger("ml_factory.jobs.validation")


def run_validation(
    store: ObjectStore, layout: ExperimentLayout, *, experiment_id: str, validation_id: str
) -> ValidationReport:
    request = read_model(store, layout.validation_request(validation_id), ValidationRequest)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    preparation = read_model_or_fail(store, layout)
    target = config.target_column
    warnings: list[AnalysisWarning] = []

    LOGGER.info("Loading validation dataset %s", request.dataset_uri)
    frame = load_dataset(store, request.dataset_uri, request.file_format).frame

    labels_present = target in frame.columns
    rows_before = len(frame)
    if labels_present:
        frame = frame[frame[target].notna()].reset_index(drop=True)
        if frame.empty:
            raise JobError("NO_USABLE_ROWS", "No rows with a target value remain in the dataset.")

    frame = _derive(frame, config, warnings)
    required = list(preparation.usable_features)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise JobError(
            "FEATURE_COLUMNS_MISSING",
            "The validation dataset lacks columns the models were trained on: "
            + ", ".join(missing[:50]),
            {"missing_columns": ",".join(missing[:50])},
        )

    # Overlap and the training-label frame need the seen folds; labels for metrics come from the
    # dataset itself.
    seen_frames, train = _seen_and_training_frames(store, layout)
    overlap, unseen = (
        _overlap(frame, seen_frames, [*required, target])
        if labels_present
        else (
            None,
            np.ones(len(frame), dtype=bool),
        )
    )
    if overlap is not None:
        warnings.extend(_overlap_warnings(overlap))

    versions = _resolve_versions(store, layout, experiment_id, request.version_ids)
    if not versions:
        raise JobError("NO_MODEL_VERSIONS", "No model version was found to validate.")

    y_train = train[target].astype("string") if target in train.columns else None
    results = [
        _validate_version(store, config, version, frame, unseen, y_train, labels_present)
        for version in versions
    ]

    runtime = capture(experiment_id, "validation")
    trained_in = read_model_if_exists(store, layout.environment, RuntimeEnvironment)
    differences = trained_in.differences(runtime) if trained_in is not None else {}
    warnings.extend(_environment_warnings(trained_in, differences))

    report = ValidationReport(
        validation_id=validation_id,
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        problem_type=config.problem_type,
        target_column=target,
        dataset_uri=request.dataset_uri,
        file_format=request.file_format,
        labels_present=labels_present,
        row_count=len(frame),
        rows_dropped_missing_target=rows_before - len(frame) if labels_present else 0,
        overlap=overlap,
        results=results,
        image_uri=runtime.image_uri,
        training_image_uri=trained_in.image_uri if trained_in else None,
        warnings=warnings,
    )
    write_model(store, layout.validation_report(validation_id), report)
    LOGGER.info("Wrote %s", layout.validation_report(validation_id))
    return report


def read_model_or_fail(store: ObjectStore, layout: ExperimentLayout) -> PreparationReport:
    preparation = read_model_if_exists(store, layout.preparation, PreparationReport)
    if preparation is None:
        raise JobError("NOT_TRAINED", "This experiment has not been trained yet.")
    return preparation


def _resolve_versions(
    store: ObjectStore, layout: ExperimentLayout, experiment_id: str, version_ids: list[str]
) -> list[ModelVersion]:
    """The requested Model_Versions, or every version when none is named.

    Enumerated here with only ``ml_engine`` + the store (a job must not import the control plane):
    the current run's ``models/<name>/metadata.json`` and each archived run's copy, exactly the set
    the ModelVersionService lists. Each version's artifact URIs are resolved against its own
    prefix, so an archived version points into ``runs/<run_id>/``.
    """
    from ml_engine.contracts.runs import ArchivedRun

    wanted = set(version_ids)

    def want(version: ModelVersion) -> bool:
        return not wanted or version.version_id in wanted

    versions: list[ModelVersion] = []
    for metadata in _metadata_in(store, layout):
        version = _version_of(experiment_id, metadata, layout, run_id=None)
        if want(version):
            versions.append(version)

    runs = sorted(
        (
            read_model(store, uri, ArchivedRun)
            for uri in store.list_uris(layout.runs_prefix)
            if uri.endswith("/run.json") and uri.count("/") == layout.runs_prefix.count("/") + 1
        ),
        key=lambda r: r.archived_at,
        reverse=True,
    )
    for run in runs:
        run_layout = ExperimentLayout(base=layout.run_archive(run.run_id))
        for metadata in _metadata_in(store, run_layout):
            version = _version_of(
                experiment_id, metadata, run_layout, run_id=run.run_id, source="retrain"
            )
            if want(version):
                versions.append(version)

    # Free-form models (free-form-experiments Req 7.1): enumerate each run's registered model so a
    # free_form:<run_id>:<name> version can be validated through this same flow.
    for version in _free_form_versions(store, layout, experiment_id):
        if want(version):
            versions.append(version)
    return versions


def _free_form_versions(
    store: ObjectStore, layout: ExperimentLayout, experiment_id: str
) -> list[ModelVersion]:
    prefix = layout.free_form_prefix
    run_ids = {
        uri[len(prefix) :].split("/", 1)[0]
        for uri in store.list_uris(prefix)
        if "/" in uri[len(prefix) :]
    }
    versions: list[ModelVersion] = []
    for run_id in sorted(run_ids):
        for uri in store.list_uris(layout.free_form_models_prefix(run_id)):
            if not uri.endswith("/metadata.json"):
                continue
            metadata = read_model_if_exists(store, uri, ModelMetadata)
            if metadata is None:
                continue
            versions.append(
                ModelVersion(
                    version_id=f"free_form:{run_id}:{metadata.model_name}",
                    experiment_id=experiment_id,
                    model_name=metadata.model_name,
                    display_name=metadata.display_name,
                    library=metadata.library,
                    problem_type=metadata.problem_type,
                    hyperparameters=dict(metadata.hyperparameters),
                    primary_metric=metadata.primary_metric,
                    primary_score=metadata.primary_score,
                    metrics=metadata.metrics,
                    created_at=metadata.training_completed_at or metadata.training_started_at,
                    source="free_form",
                    run_id=run_id,
                    artifact_ref=ArtifactRef(
                        model_uri=layout.free_form_model_artifact(run_id, metadata.model_name),
                        metadata_uri=layout.free_form_model_metadata(run_id, metadata.model_name),
                    ),
                )
            )
    return versions


def _metadata_in(store: ObjectStore, layout: ExperimentLayout) -> list[ModelMetadata]:
    prefix = layout.models_prefix + "/"
    found: list[ModelMetadata] = []
    for uri in store.list_uris(prefix):
        if not uri.endswith("/metadata.json") or uri.count("/") != prefix.count("/") + 1:
            continue
        metadata = read_model_if_exists(store, uri, ModelMetadata)
        if metadata is not None:
            found.append(metadata)
    return sorted(found, key=lambda m: m.model_name)


def _version_of(
    experiment_id: str,
    metadata: ModelMetadata,
    layout: ExperimentLayout,
    *,
    run_id: str | None,
    source: str = "training",
) -> ModelVersion:
    prefix = run_id or CURRENT_RUN
    strategy = metadata.preprocessing.strategy if metadata.preprocessing else None
    return ModelVersion(
        version_id=f"{prefix}:{metadata.model_name}",
        experiment_id=experiment_id,
        model_name=metadata.model_name,
        display_name=metadata.display_name,
        library=metadata.library,
        problem_type=metadata.problem_type,
        hyperparameters=dict(metadata.hyperparameters),
        primary_metric=metadata.primary_metric,
        primary_score=metadata.primary_score,
        metrics=metadata.metrics,
        created_at=metadata.training_completed_at or metadata.training_started_at,
        source=source,
        run_id=run_id,
        artifact_ref=ArtifactRef(
            model_uri=layout.model_artifact(metadata.model_name),
            metadata_uri=layout.model_metadata(metadata.model_name),
            preprocessor_uri=layout.preprocessor(strategy) if strategy else None,
            preprocessing_strategy=strategy,
        ),
    )


def _validate_version(
    store: ObjectStore,
    config: ExperimentConfig,
    version: ModelVersion,
    frame: pd.DataFrame,
    unseen: np.ndarray,
    y_train: pd.Series | None,
    labels_present: bool,
) -> ValidationModelResult:
    base = {
        "version_id": version.version_id,
        "model_name": version.model_name,
        "run_id": version.run_id,
        "primary_metric": version.primary_metric or config.primary_metric,
    }
    try:
        plugin = get_plugin(version.model_name)
        ref = version.artifact_ref
        pipeline = load_joblib(store, ref.preprocessor_uri)
        estimator = load_joblib(store, ref.model_uri)
        x = pipeline.transform(frame)
        predictions = plugin.predict(estimator, x)
        probabilities = plugin.predict_proba(estimator, x)
    except Exception as error:
        LOGGER.exception("Validating %s failed", version.version_id)
        return ValidationModelResult(
            **base, status="failed", failure_message=f"{type(error).__name__}: {error}"[:500]
        )

    problem_type = config.problem_type
    labels = plugin.class_labels(estimator) if problem_type.is_classification else None
    summary = _prediction_summary(problem_type, predictions)

    if not labels_present:
        return ValidationModelResult(**base, status="completed", prediction_summary=summary)

    y = frame[config.target_column]
    positive = None
    if problem_type is ProblemType.BINARY_CLASSIFICATION:
        y = y.astype("string")
        context = build_training_context(
            problem_type=problem_type,
            y_train=y_train if y_train is not None else y,
            feature_count=x.shape[1],
            random_seed=config.split.random_seed,
            class_weighting=config.class_weighting,
            categorical_features=[],
        )
        positive = _positive_class(labels, context)
    elif problem_type.is_classification:
        y = y.astype("string")

    def score(mask: np.ndarray):
        return evaluate(
            problem_type,
            y[mask].reset_index(drop=True),
            np.asarray(predictions)[mask],
            y_proba=None if probabilities is None else np.asarray(probabilities)[mask],
            class_labels=labels,
            positive_class=positive,
        )

    everything = np.ones(len(frame), dtype=bool)
    all_rows = score(everything)
    unseen_rows = score(unseen) if unseen.sum() >= MIN_UNSEEN_ROWS else None
    metric = base["primary_metric"]

    analysis = None
    at_best = None
    threshold_rows = None
    if positive is not None:
        scores = _positive_scores(probabilities, labels or [], positive)
        if scores is not None:
            mask = unseen if unseen_rows is not None else everything
            threshold_rows = "unseen" if unseen_rows is not None else "all"
            truth = (y == positive).to_numpy().astype(int)
            # A fresh threshold analysis on THIS dataset's labels (Req 9.2, 9.3).
            analysis = threshold_analysis(
                truth[mask],
                scores[mask],
                experiment_id=config.experiment_id,
                model_name=version.model_name,
                positive_class=positive,
            )
            if analysis is not None:
                at_best = analysis.best_f1

    return ValidationModelResult(
        **base,
        status="completed",
        metrics_all_rows=all_rows,
        metrics_unseen_rows=unseen_rows,
        all_rows_score=all_rows.get(metric),
        unseen_rows_score=unseen_rows.get(metric) if unseen_rows else None,
        threshold_analysis=analysis,
        at_validation_best_threshold=at_best,
        threshold_rows=threshold_rows,
        prediction_summary=summary,
    )


def _prediction_summary(problem_type: ProblemType, predictions: np.ndarray) -> dict[str, float]:
    values = np.asarray(predictions)
    if problem_type.is_classification:
        labels, counts = np.unique(values.astype("str"), return_counts=True)
        total = int(counts.sum()) or 1
        return {
            str(label): round(int(count) / total, 6)
            for label, count in zip(labels, counts, strict=False)
        }
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return {}
    return {
        "mean": round(float(numeric.mean()), 6),
        "min": round(float(numeric.min()), 6),
        "max": round(float(numeric.max()), 6),
    }


# --- entrypoint ---------------------------------------------------------------------------------


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_validation(
        store, layout, experiment_id=args.experiment_id, validation_id=args.validation_id
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory validation job (score model versions on a labeled file)")
    parser.prog = "validation-job"
    parser.add_argument("--validation-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda args, layout: layout.validation_failure(args.validation_id),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
