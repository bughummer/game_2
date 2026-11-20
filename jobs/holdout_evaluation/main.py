"""Holdout evaluation job: score already-trained models on a separate dataset.

Runs as a SageMaker Processing job after training. It loads the experiment's frozen config, its
fitted preprocessing pipelines and every trained model, applies them unchanged to the given file
(the same derived features are computed the same way), and writes
``evaluations/<id>/holdout.json``. Nothing is re-fitted, so the file cannot influence a model.

Rows that also appear in the training or validation fold are counted, and metrics are reported
on all rows and on the unseen rows only: scoring rows a model was trained on inflates every
metric, which is what happens when the "full dataset" includes the training data.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from jobs._common.environment import capture
from jobs._common.runtime import JobError, base_parser, run_entrypoint
from ml_engine.contracts.common import ProblemType, Severity, WarningCategory
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.environment import RESULT_CRITICAL_PACKAGES, RuntimeEnvironment
from ml_engine.contracts.holdout import (
    HoldoutEvaluationReport,
    HoldoutModelResult,
    HoldoutRequest,
    RowOverlap,
)
from ml_engine.contracts.model import ModelMetadata
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.contracts.thresholds import ThresholdAnalysis
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.evaluation import evaluate
from ml_engine.evaluation.evaluator import _positive_scores
from ml_engine.evaluation.thresholds import point_at
from ml_engine.features import apply_proposals, column_types_from_frame, validate_proposals
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    load_dataset,
    load_joblib,
    read_model,
    read_model_if_exists,
    read_parquet,
    write_model,
)
from ml_engine.models import get_plugin
from ml_engine.training.runner import _positive_class, build_training_context

LOGGER = logging.getLogger("ml_factory.jobs.holdout_evaluation")

#: Below this many unseen rows, unseen-only metrics are too noisy to report.
MIN_UNSEEN_ROWS = 30
#: Above this overlap share the all-rows metrics are flagged as substantially inflated.
HIGH_OVERLAP = 0.05


def run_holdout_evaluation(
    store: ObjectStore, layout: ExperimentLayout, *, experiment_id: str, evaluation_id: str
) -> HoldoutEvaluationReport:
    request = read_model(store, layout.evaluation_request(evaluation_id), HoldoutRequest)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    preparation = read_model_if_exists(store, layout.preparation, PreparationReport)
    if preparation is None:
        raise JobError("NOT_TRAINED", "This experiment has not been trained yet.")
    target = config.target_column
    warnings: list[AnalysisWarning] = []

    LOGGER.info("Loading evaluation dataset %s", request.dataset_uri)
    frame = load_dataset(store, request.dataset_uri, request.file_format).frame
    if target not in frame.columns:
        raise JobError(
            "TARGET_COLUMN_MISSING",
            f"The evaluation dataset has no target column '{target}'.",
        )
    rows_before = len(frame)
    frame = frame[frame[target].notna()].reset_index(drop=True)
    if frame.empty:
        raise JobError("NO_USABLE_ROWS", "No rows with a target value remain in the dataset.")

    frame = _derive(frame, config, warnings)
    required = list(preparation.usable_features)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise JobError(
            "FEATURE_COLUMNS_MISSING",
            "The evaluation dataset lacks columns the models were trained on: "
            + ", ".join(missing[:50]),
            {"missing_columns": ",".join(missing[:50])},
        )

    seen_frames, train = _seen_and_training_frames(store, layout)
    overlap, unseen = _overlap(frame, seen_frames, [*required, target])
    warnings.extend(_overlap_warnings(overlap))

    y_train = train[target].astype("string")
    is_binary = config.problem_type is ProblemType.BINARY_CLASSIFICATION
    results = []
    positive_rates: tuple[float | None, float | None] = (None, None)
    for spec in config.enabled_models_specs():
        metadata = read_model_if_exists(store, layout.model_metadata(spec.name), ModelMetadata)
        if metadata is None:
            continue  # this model did not train; nothing to evaluate
        result, rates = _evaluate_model(store, layout, config, metadata, frame, unseen, y_train)
        results.append(result)
        if rates[0] is not None:
            positive_rates = rates
    if not results:
        raise JobError("NO_TRAINED_MODELS", "No trained model was found for this experiment.")

    if is_binary and None not in positive_rates:
        here, trained = positive_rates
        if abs(here - trained) > 0.05:
            warnings.append(
                AnalysisWarning(
                    rule="target_rate_differs_from_training",
                    category=WarningCategory.TARGET,
                    severity=Severity.MEDIUM,
                    message=(
                        f"The positive class is {here:.1%} of this dataset but {trained:.1%} of "
                        "the training fold. Precision and PR AUC move with the base rate."
                    ),
                    details={"evaluation_rate": here, "training_rate": trained},
                )
            )

    runtime = capture(experiment_id, "holdout_evaluation")
    trained_in = read_model_if_exists(store, layout.environment, RuntimeEnvironment)
    differences = trained_in.differences(runtime) if trained_in is not None else {}
    warnings.extend(_environment_warnings(trained_in, differences))

    report = HoldoutEvaluationReport(
        evaluation_id=evaluation_id,
        image_uri=runtime.image_uri,
        training_image_uri=trained_in.image_uri if trained_in else None,
        environment_differences=differences,
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        problem_type=config.problem_type,
        target_column=target,
        dataset_uri=request.dataset_uri,
        file_format=request.file_format,
        row_count=len(frame),
        rows_dropped_missing_target=rows_before - len(frame),
        target_rate=positive_rates[0],
        training_target_rate=positive_rates[1],
        overlap=overlap,
        models=results,
        warnings=warnings,
    )
    write_model(store, layout.evaluation_report(evaluation_id), report)
    LOGGER.info("Wrote %s", layout.evaluation_report(evaluation_id))
    return report


# --- steps --------------------------------------------------------------------------------------


def _derive(
    frame: pd.DataFrame, config: ExperimentConfig, warnings: list[AnalysisWarning]
) -> pd.DataFrame:
    """Compute the run's derived features exactly as preparation did (row-wise, no fitting)."""
    if not config.derived_features:
        return frame
    accepted, rejected = validate_proposals(
        config.derived_features, column_types_from_frame(frame), target_column=config.target_column
    )
    for refusal in rejected:
        warnings.append(
            AnalysisWarning(
                rule="derived_feature_not_computable",
                category=WarningCategory.FEATURE,
                severity=Severity.HIGH,
                message=f"Derived feature '{refusal.name}' cannot be built here: {refusal.message}",
                column=refusal.name,
            )
        )
    frame, _derived = apply_proposals(frame, accepted)
    return frame


def _row_keys(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    """A per-row fingerprint that ignores dtype differences between CSV and Parquet."""
    normalized = pd.DataFrame(index=frame.index)
    for column in columns:
        values = frame[column]
        numeric = pd.to_numeric(values, errors="coerce")
        if values.notna().sum() and numeric.notna().sum() == values.notna().sum():
            normalized[column] = numeric.round(9).astype("string").fillna("<NA>")
        else:
            normalized[column] = values.astype("string").fillna("<NA>")
    return pd.util.hash_pandas_object(normalized, index=False)


def _seen_and_training_frames(
    store: ObjectStore, layout: ExperimentLayout
) -> tuple[list[tuple[str, pd.DataFrame]], pd.DataFrame]:
    """The frames whose rows count as "seen in training", plus the training-label frame.

    Single_Split_Mode reads the train and validation folds. CV_Mode reads ``development.parquet``
    (the Development_Set the Refit_Model was fit on) in place of them. In both modes, a Test_Fold
    (``test.parquet``) is included as "seen" when present, so evaluation rows that duplicate the
    held-out test set are flagged too (design C7, Req 10.4). The training-label frame is the
    training fold in single split and the Development_Set in CV_Mode.
    """
    seen: list[tuple[str, pd.DataFrame]] = []
    if store.exists(layout.development_dataset):
        development = read_parquet(store, layout.development_dataset)
        seen.append(("training_fold", development))
        training = development
    else:
        train = read_parquet(store, layout.train_dataset)
        validation = read_parquet(store, layout.validation_dataset)
        seen.append(("training_fold", train))
        seen.append(("validation_fold", validation))
        training = train
    if store.exists(layout.test_dataset):
        seen.append(("test_fold", read_parquet(store, layout.test_dataset)))
    return seen, training


def _overlap(
    frame: pd.DataFrame, seen_frames: list[tuple[str, pd.DataFrame]], columns: list[str]
) -> tuple[RowOverlap, np.ndarray]:
    keys = _row_keys(frame, columns)
    in_training = np.zeros(len(frame), dtype=bool)  # training/development + test = "seen"
    in_validation = np.zeros(len(frame), dtype=bool)
    for role, source in seen_frames:
        hit = keys.isin(set(_row_keys(source, columns))).to_numpy()
        if role == "validation_fold":
            in_validation |= hit
        else:  # training_fold, development and test_fold all count as training-side exposure
            in_training |= hit
    unseen = ~(in_training | in_validation)
    overlap = RowOverlap(
        compared_columns=columns,
        rows_in_training_fold=int(in_training.sum()),
        rows_in_validation_fold=int(in_validation.sum()),
        unseen_rows=int(unseen.sum()),
        overlap_fraction=round(float(1 - unseen.mean()), 6),
    )
    return overlap, unseen


def _environment_warnings(
    trained_in: RuntimeEnvironment | None, differences: dict[str, str]
) -> list[AnalysisWarning]:
    if trained_in is None:
        return [
            AnalysisWarning(
                rule="training_environment_unknown",
                category=WarningCategory.MODEL,
                severity=Severity.MEDIUM,
                message=(
                    "The training environment was not recorded for this experiment (trained "
                    "before environments were captured), so it cannot be confirmed that the "
                    "models are scored with the library versions they were built with."
                ),
            )
        ]
    critical = {
        name: change
        for name, change in differences.items()
        if name in RESULT_CRITICAL_PACKAGES or name == "python"
    }
    if not critical:
        return []
    return [
        AnalysisWarning(
            rule="environment_differs_from_training",
            category=WarningCategory.MODEL,
            severity=Severity.HIGH,
            message=(
                "These libraries differ from the ones the models were trained with, so scores "
                "may differ from what the training environment would produce: "
                + ", ".join(f"{name} {change}" for name, change in critical.items())
            ),
            details=critical,
        )
    ]


def _overlap_warnings(overlap: RowOverlap) -> list[AnalysisWarning]:
    seen = overlap.rows_in_training_fold + overlap.rows_in_validation_fold
    if seen == 0:
        return []
    warnings = [
        AnalysisWarning(
            rule="evaluation_rows_seen_in_training",
            category=WarningCategory.LEAKAGE,
            severity=(
                Severity.HIGH if overlap.overlap_fraction > HIGH_OVERLAP else Severity.MEDIUM
            ),
            message=(
                f"{overlap.overlap_fraction:.1%} of the rows also appear in the training or "
                "validation data, so the all-rows metrics are inflated. Use the unseen-rows "
                "metrics."
            ),
            details={
                "rows_in_training_fold": overlap.rows_in_training_fold,
                "rows_in_validation_fold": overlap.rows_in_validation_fold,
            },
        )
    ]
    if overlap.unseen_rows < MIN_UNSEEN_ROWS:
        warnings.append(
            AnalysisWarning(
                rule="too_few_unseen_rows",
                category=WarningCategory.METRIC,
                severity=Severity.HIGH,
                message=(
                    f"Only {overlap.unseen_rows} row(s) were not seen in training, too few for "
                    "unseen-rows metrics. This dataset cannot measure performance on new data."
                ),
            )
        )
    return warnings


def _evaluate_model(
    store: ObjectStore,
    layout: ExperimentLayout,
    config: ExperimentConfig,
    metadata: ModelMetadata,
    frame: pd.DataFrame,
    unseen: np.ndarray,
    y_train: pd.Series,
) -> tuple[HoldoutModelResult, tuple[float | None, float | None]]:
    base = {
        "model_name": metadata.model_name,
        "display_name": metadata.display_name,
        "primary_metric": metadata.primary_metric or config.primary_metric,
        "validation_score": metadata.primary_score,
    }
    try:
        plugin = get_plugin(metadata.model_name)
        pipeline = load_joblib(store, layout.preprocessor(plugin.preprocessing_strategy))
        estimator = load_joblib(store, layout.model_artifact(metadata.model_name))
        x = pipeline.transform(frame)
        predictions = plugin.predict(estimator, x)
        probabilities = plugin.predict_proba(estimator, x)
    except Exception as error:
        LOGGER.exception("Scoring %s failed", metadata.model_name)
        return (
            HoldoutModelResult(
                **base, status="failed", failure_message=f"{type(error).__name__}: {error}"[:500]
            ),
            (None, None),
        )

    problem_type = config.problem_type
    y = frame[config.target_column]
    labels = plugin.class_labels(estimator) if problem_type.is_classification else None
    positive = None
    if problem_type is ProblemType.BINARY_CLASSIFICATION:
        y = y.astype("string")
        context = build_training_context(
            problem_type=problem_type,
            y_train=y_train,
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

    threshold_point, threshold_rows = None, None
    rates: tuple[float | None, float | None] = (None, None)
    if positive is not None:
        truth = (y == positive).to_numpy().astype(int)
        rates = (
            round(float(truth.mean()), 6),
            round(float((y_train == positive).mean()), 6),
        )
        analysis = read_model_if_exists(
            store, layout.model_thresholds(metadata.model_name), ThresholdAnalysis
        )
        scores = _positive_scores(probabilities, labels or [], positive)
        if analysis is not None and scores is not None:
            mask = unseen if unseen_rows is not None else everything
            threshold_point = point_at(truth[mask], scores[mask], analysis.best_f1.threshold)
            threshold_rows = "unseen" if unseen_rows is not None else "all"

    return (
        HoldoutModelResult(
            **base,
            status="completed",
            all_rows_score=all_rows.get(metric),
            unseen_rows_score=unseen_rows.get(metric) if unseen_rows else None,
            metrics_all_rows=all_rows,
            metrics_unseen_rows=unseen_rows,
            at_validation_best_threshold=threshold_point,
            threshold_rows=threshold_rows,
        ),
        rates,
    )


# --- entrypoint ---------------------------------------------------------------------------------


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_holdout_evaluation(
        store, layout, experiment_id=args.experiment_id, evaluation_id=args.evaluation_id
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory holdout evaluation job (score trained models on a file)")
    parser.prog = "holdout-evaluation-job"
    parser.add_argument("--evaluation-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda args, layout: layout.evaluation_failure(args.evaluation_id),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
