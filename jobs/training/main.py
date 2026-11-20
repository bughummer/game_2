"""Training job: fit exactly one model.

Runs as a SageMaker Training job, one job per model, so a single model failure is isolated
to its own branch of the state machine and never fails the experiment.
"""

import argparse
import logging
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from jobs._common.runtime import JobError, base_parser, run_entrypoint
from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.model import ModelArtifacts, ModelFailure, ModelMetadata
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    load_joblib,
    read_parquet,
    save_joblib,
    write_model,
)
from ml_engine.models import (
    GRID_POINT_SEPARATOR,
    ModelNotAvailableError,
    ModelPlugin,
    UnknownModelError,
    get_plugin,
    is_ensemble_name,
    resolve_plugin,
)
from ml_engine.training import TrainingError, train_model

LOGGER = logging.getLogger("ml_factory.jobs.training")


def _name_for_spec(metadata, model_name: str):
    """Stamp the configured spec name (the grid-point id) onto the metadata.

    ``train_model`` names a model by its plugin, but a grid point's spec name is
    ``<base>__<hash>``. Overriding here makes each point rank and list as its own model version;
    a plain model name has no separator and keeps its ``display_name``.
    """
    if GRID_POINT_SEPARATOR not in model_name:
        return metadata
    display = f"{metadata.display_name} ({model_name.split(GRID_POINT_SEPARATOR, 1)[1]})"
    return metadata.model_copy(update={"model_name": model_name, "display_name": display})


def run_training(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
    plugin_factory: Callable[[str], ModelPlugin] | None = None,
    checkpoint_dir: Path | None = None,
) -> ModelMetadata:
    """Train one configured model and write its artifact and metadata.

    ``plugin_factory`` (ML Factory 2.0) lets the SageMaker entrypoint wrap the plugin with Spot
    checkpointing; ``None`` is the inherited behaviour, ``get_plugin``. ``checkpoint_dir`` is
    used only in CV_Mode, where every fit gets its own checkpoint sub-directory.
    """
    config = read_model_config(store, layout)
    spec = next((m for m in config.enabled_models_specs() if m.name == model_name), None)
    if spec is None:
        raise JobError(
            "MODEL_NOT_CONFIGURED",
            f"Model '{model_name}' is not part of this experiment's configuration.",
        )

    from ml_engine.contracts.cv import EvaluationMode

    if config.split.evaluation_mode is EvaluationMode.CV:  # ML Factory 2.0: CV design C5
        from jobs.training.cv import run_cv_training

        cv_metadata = run_cv_training(
            store,
            layout,
            experiment_id=experiment_id,
            model_name=model_name,
            config=config,
            spec=spec,
            plugin_factory=plugin_factory,
            checkpoint_dir=checkpoint_dir,
        )
        _record_environment(store, layout, experiment_id)
        return cv_metadata

    try:
        # An ensemble name resolves to an EnsemblePlugin built from the frozen spec; every other
        # name uses the checkpointing factory (or the plain registry lookup).
        if is_ensemble_name(model_name):
            plugin = resolve_plugin(config, model_name)
        else:
            plugin = (plugin_factory or get_plugin)(model_name)
    except UnknownModelError as error:
        raise JobError("UNKNOWN_MODEL", str(error)) from error

    try:
        pipeline = load_joblib(store, layout.preprocessor(plugin.preprocessing_strategy))
    except FileNotFoundError as error:
        raise JobError(
            "PREPROCESSING_ARTIFACT_MISSING",
            "The fitted preprocessing pipeline is missing. The preparation stage must run first.",
        ) from error

    train_frame = read_parquet(store, layout.train_dataset)
    validation_frame = read_parquet(store, layout.validation_dataset)
    LOGGER.info(
        "Training %s on %d rows (%d validation rows)",
        model_name,
        len(train_frame),
        len(validation_frame),
    )

    try:
        metadata, estimator = train_model(
            plugin,
            experiment_id=experiment_id,
            problem_type=config.problem_type,
            pipeline=pipeline,
            train_frame=train_frame,
            validation_frame=validation_frame,
            target_column=config.target_column,
            param_overrides=dict(spec.params),
            primary_metric=config.primary_metric,
            random_seed=config.split.random_seed,
            class_weighting=config.class_weighting,
        )
    except (TrainingError, ModelNotAvailableError) as error:
        raise JobError("MODEL_TRAINING_FAILED", str(error), {"model": model_name}) from error

    if config.problem_type is ProblemType.BINARY_CLASSIFICATION:
        _write_threshold_analysis(
            store,
            layout,
            experiment_id=experiment_id,
            plugin=plugin,
            estimator=estimator,
            pipeline=pipeline,
            train_frame=train_frame,
            validation_frame=validation_frame,
            config=config,
        )

    _write_permutation_importance(
        store,
        layout,
        experiment_id=experiment_id,
        plugin=plugin,
        estimator=estimator,
        pipeline=pipeline,
        train_frame=train_frame,
        validation_frame=validation_frame,
        config=config,
        primary_metric=metadata.primary_metric or config.primary_metric,
    )

    # A grid point is a ModelSpec whose name is '<base>__<hash>'; the metadata must carry that
    # point id (not the base plugin name) so each point ranks and lists as its own model version
    # (grid-search design C5). A plain model keeps its name unchanged.
    metadata = _name_for_spec(metadata, model_name)
    model_uri = save_joblib(store, layout.model_artifact(model_name), estimator)
    metadata.artifacts = ModelArtifacts(
        model_uri=model_uri,
        metadata_uri=layout.model_metadata(model_name),
        preprocessor_uri=layout.preprocessor(plugin.preprocessing_strategy),
    )
    write_model(store, layout.model_metadata(model_name), metadata)
    _record_environment(store, layout, experiment_id)
    LOGGER.info("%s completed: %s=%s", model_name, metadata.primary_metric, metadata.primary_score)
    return metadata


def _record_environment(store: ObjectStore, layout: ExperimentLayout, experiment_id: str) -> None:
    try:  # ML Factory 2.0: environment.json + requirements.txt; never fails the model
        from jobs._common.environment import write_training_environment

        write_training_environment(store, layout, experiment_id)
    except Exception:
        LOGGER.exception("Could not record the training environment")


def _write_threshold_analysis(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    plugin: ModelPlugin,
    estimator: Any,
    pipeline: Any,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: ExperimentConfig,
) -> None:
    """ML Factory 2.0: precision/recall/F1 across thresholds on the validation fold.

    Uses the runner's own positive-class and score selection, so these numbers line up with the
    headline metrics at 0.5. Best effort: a failure here is logged and never fails the model.
    """
    from ml_engine.evaluation.evaluator import _positive_scores
    from ml_engine.evaluation.thresholds import threshold_analysis
    from ml_engine.training.runner import _positive_class, build_training_context

    try:
        x_validation = pipeline.transform(validation_frame)
        probabilities = plugin.predict_proba(estimator, x_validation)
        labels = plugin.class_labels(estimator)
        if probabilities is None or not labels:
            return
        context = build_training_context(
            problem_type=config.problem_type,
            y_train=train_frame[config.target_column].astype("string"),
            feature_count=x_validation.shape[1],
            random_seed=config.split.random_seed,
            class_weighting=config.class_weighting,
            categorical_features=[],
        )
        positive = _positive_class(labels, context)
        scores = _positive_scores(probabilities, labels, positive)
        if positive is None or scores is None:
            return
        truth = (validation_frame[config.target_column].astype("string") == positive).to_numpy()
        analysis = threshold_analysis(
            truth.astype(int),
            scores,
            experiment_id=experiment_id,
            model_name=plugin.name,
            positive_class=positive,
        )
        if analysis is not None:
            write_model(store, layout.model_thresholds(plugin.name), analysis)
    except Exception:
        LOGGER.exception("Threshold analysis failed for %s; continuing", plugin.name)


def validation_scorer(
    plugin: ModelPlugin,
    estimator: Any,
    pipeline: Any,
    train_frame: pd.DataFrame,
    config: ExperimentConfig,
    primary_metric: str,
) -> Callable[[pd.DataFrame], float | None]:
    """Score a raw frame (feature columns + target) the way ``train_model`` scores validation.

    Same positive-class rule, same label handling, same ``evaluate`` call — so the baseline of
    a permutation or ablation equals the model's reported validation score.
    """
    from ml_engine.evaluation import evaluate
    from ml_engine.training.runner import _positive_class, build_training_context

    problem_type = config.problem_type
    target = config.target_column
    labels = plugin.class_labels(estimator) if problem_type.is_classification else None
    positive = None
    if problem_type is ProblemType.BINARY_CLASSIFICATION:
        context = build_training_context(
            problem_type=problem_type,
            y_train=train_frame[target].astype("string"),
            feature_count=len(pipeline.feature_names),
            random_seed=config.split.random_seed,
            class_weighting=config.class_weighting,
            categorical_features=[],
        )
        positive = _positive_class(labels, context)

    def score(frame: pd.DataFrame) -> float | None:
        y = frame[target]
        if problem_type.is_classification:
            y = y.astype("string")
        x = pipeline.transform(frame)
        metrics = evaluate(
            problem_type,
            y,
            plugin.predict(estimator, x),
            y_proba=plugin.predict_proba(estimator, x),
            class_labels=labels,
            positive_class=positive,
        )
        return metrics.get(primary_metric)

    return score


def _write_permutation_importance(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    plugin: ModelPlugin,
    estimator: Any,
    pipeline: Any,
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    config: ExperimentConfig,
    primary_metric: str,
) -> None:
    """ML Factory 2.0: permutation importance per input column. Never fails the model."""
    from ml_engine.evaluation.metrics import metric_direction
    from ml_engine.evaluation.permutation import permutation_importance

    try:
        features = [c for c in validation_frame.columns if c != config.target_column]
        result = permutation_importance(
            validation_frame,
            features,
            validation_scorer(plugin, estimator, pipeline, train_frame, config, primary_metric),
            experiment_id=experiment_id,
            model_name=plugin.name,
            primary_metric=primary_metric,
            direction=metric_direction(config.problem_type, primary_metric),
            random_seed=config.split.random_seed,
        )
        if result is not None:
            write_model(store, layout.model_permutation(plugin.name), result)
    except Exception:
        LOGGER.exception("Permutation importance failed for %s; continuing", plugin.name)


def read_model_config(store: ObjectStore, layout: ExperimentLayout) -> ExperimentConfig:
    from ml_engine.io import read_model

    try:
        return read_model(store, layout.experiment_config, ExperimentConfig)
    except FileNotFoundError as error:
        raise JobError(
            "EXPERIMENT_CONFIG_MISSING",
            "The experiment configuration artifact is missing; training cannot start.",
        ) from error


def write_model_failure(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
    code: str,
    message: str,
    details: dict[str, str] | None = None,
) -> None:
    """Per-model failure record, read by the evaluation job when it builds the comparison.

    ``details`` (ML Factory 2.0) carries e.g. CV's ``failed_fit``; omitted, the record is the
    inherited one.
    """
    write_model(
        store,
        layout.model_failure(model_name),
        ModelFailure(
            experiment_id=experiment_id,
            model_name=model_name,
            error_code=code,
            message=message,
            failed_at=datetime.now(UTC),
            details=dict(details or {}),
        ),
    )


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    factory = None
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if checkpoint_dir is not None:
        from jobs.training.checkpointing import checkpointing_factory

        factory = checkpointing_factory(checkpoint_dir)
    run_training(
        store,
        layout,
        experiment_id=args.experiment_id,
        model_name=args.model_name,
        plugin_factory=factory,
        checkpoint_dir=checkpoint_dir,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory training job (one model per job)")
    parser.prog = "training-job"
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Spot checkpoint directory (SageMaker syncs /opt/ml/checkpoints to S3). Only "
        "XGBoost and CatBoost checkpoint; other models restart on interruption.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda args, layout: layout.model_failure(args.model_name),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
