"""Batch scoring job: apply one trained model to a new file and write predictions to S3.

Runs as a SageMaker Processing job (a side workflow); no endpoint is created. The fitted
preprocessing pipeline and model are applied unchanged, the approved derived features are
computed exactly as preparation computed them, and nothing is re-fitted. The target column is
not needed (and is ignored when present).

Output: ``scoring/<id>/predictions.csv`` — the passthrough columns, ``prediction`` and, for
classifiers with probabilities, one ``proba_<class>`` column per class — and ``scoring.json``.
"""

import argparse
import logging
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from jobs._common.environment import capture
from jobs._common.runtime import JobError, base_parser, run_entrypoint
from ml_engine.contracts.common import Severity, WarningCategory
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.environment import RuntimeEnvironment
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.contracts.scoring import ScoringReport, ScoringRequest
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.features import apply_proposals, column_types_from_frame, validate_proposals
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

LOGGER = logging.getLogger("ml_factory.jobs.scoring")


def run_scoring(
    store: ObjectStore, layout: ExperimentLayout, *, experiment_id: str, scoring_id: str
) -> ScoringReport:
    request = read_model(store, layout.scoring_request(scoring_id), ScoringRequest)
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    preparation = read_model_if_exists(store, layout.preparation, PreparationReport)
    if preparation is None or not store.exists(layout.model_artifact(request.model_name)):
        raise JobError("NOT_TRAINED", f"Model '{request.model_name}' has no trained artifact.")
    warnings: list[AnalysisWarning] = []

    frame = load_dataset(store, request.dataset_uri, request.file_format).frame
    if frame.empty:
        raise JobError("NO_ROWS", "The file to score has no rows.")
    frame = _derive(frame, config, warnings)
    missing = [c for c in preparation.usable_features if c not in frame.columns]
    if missing:
        raise JobError(
            "FEATURE_COLUMNS_MISSING",
            "The file lacks columns the model was trained on: " + ", ".join(missing[:50]),
        )
    passthrough = [c for c in request.passthrough_columns if c in frame.columns]
    for absent in sorted(set(request.passthrough_columns) - set(passthrough)):
        warnings.append(
            AnalysisWarning(
                rule="passthrough_column_missing",
                category=WarningCategory.SCHEMA,
                severity=Severity.LOW,
                message=f"Passthrough column '{absent}' is not in the file; it was skipped.",
                column=absent,
            )
        )

    plugin = get_plugin(request.model_name)
    pipeline = load_joblib(store, layout.preprocessor(plugin.preprocessing_strategy))
    estimator = load_joblib(store, layout.model_artifact(request.model_name))
    features = frame[list(preparation.usable_features)]
    x = pipeline.transform(features)
    predictions = np.asarray(plugin.predict(estimator, x))
    probabilities = plugin.predict_proba(estimator, x)

    output = frame[passthrough].copy() if passthrough else pd.DataFrame(index=frame.index)
    output["prediction"] = predictions
    prediction_columns = ["prediction"]
    labels = plugin.class_labels(estimator) if config.problem_type.is_classification else None
    if probabilities is not None and labels:
        for index, label in enumerate(labels):
            name = f"proba_{label}"
            output[name] = np.round(np.asarray(probabilities)[:, index], 6)
            prediction_columns.append(name)

    with tempfile.TemporaryDirectory(prefix="mlf-scoring-") as tmp:
        local = Path(tmp) / "predictions.csv"
        output.to_csv(local, index=False)
        store.upload(local, layout.scoring_predictions(scoring_id))

    runtime = capture(experiment_id, "scoring")
    trained_in = read_model_if_exists(store, layout.environment, RuntimeEnvironment)
    if trained_in is not None and trained_in.differences(runtime):
        warnings.append(
            AnalysisWarning(
                rule="environment_differs_from_training",
                category=WarningCategory.MODEL,
                severity=Severity.MEDIUM,
                message=(
                    "The scoring environment differs from the training environment: "
                    + ", ".join(
                        f"{k} {v}" for k, v in list(trained_in.differences(runtime).items())[:10]
                    )
                ),
            )
        )

    report = ScoringReport(
        scoring_id=scoring_id,
        experiment_id=experiment_id,
        model_name=request.model_name,
        generated_at=datetime.now(UTC),
        dataset_uri=request.dataset_uri,
        predictions_uri=layout.scoring_predictions(scoring_id),
        row_count=len(output),
        prediction_columns=prediction_columns,
        passthrough_columns=passthrough,
        prediction_summary=_summarize(predictions, config.problem_type.is_classification),
        image_uri=runtime.image_uri,
        training_image_uri=trained_in.image_uri if trained_in else None,
        warnings=warnings,
    )
    write_model(store, layout.scoring_report(scoring_id), report)
    return report


def _derive(
    frame: pd.DataFrame, config: ExperimentConfig, warnings: list[AnalysisWarning]
) -> pd.DataFrame:
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
    frame, _ = apply_proposals(frame, accepted)
    return frame


def _summarize(predictions: np.ndarray, classification: bool) -> dict[str, float]:
    if classification:
        counts = pd.Series(predictions).astype("string").value_counts(normalize=True)
        return {f"share_{k}": round(float(v), 6) for k, v in counts.head(20).items()}
    values = pd.to_numeric(pd.Series(predictions), errors="coerce").dropna()
    if values.empty:
        return {}
    return {
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std(ddof=0)), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
    }


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_scoring(store, layout, experiment_id=args.experiment_id, scoring_id=args.scoring_id)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory batch scoring job (apply a trained model to a file)")
    parser.prog = "scoring-job"
    parser.add_argument("--scoring-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda args, layout: layout.scoring_failure(args.scoring_id),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
