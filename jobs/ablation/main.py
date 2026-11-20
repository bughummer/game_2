"""Ablation job: retrain one model without each candidate feature, on the identical folds.

Runs as a SageMaker Processing job after training (a side workflow). For a leak, the decisive
question is "how good is the model *without* this column?" — this job answers it directly:

1. read the frozen config and the train/validation folds preparation wrote (identical folds);
2. retrain the chosen model with every usable feature → the baseline, in this same job, so the
   baseline and every ablation share one procedure, one seed and one environment;
3. for each candidate: re-plan the columns without it, refit preprocessing on the training fold
   only, retrain with the same hyperparameters, score on the validation fold.

In CV_Mode (design C7) the same comparison runs on the Experiment's stored Fold_Assignment:
the baseline and every candidate are evaluated on every CV_Fold (preprocessing refit on each
fold's training rows from ``development.parquet``), and ``score_without``/``baseline_score``
become the fold means with per-fold ``fold_drops``, their sample ``std_drop`` and the fold
count recorded. Candidates are capped so ``(candidates + 1) * K <= 60`` and the truncation is
reported. The Test_Fold is never read (requirements 10.3, 3.6).

Writes ``leakage/ablation.json``. The holdout test set is never touched (requirement 13.5).
"""

import argparse
import logging
import statistics
import sys
from datetime import UTC, datetime

import pandas as pd

from jobs._common.runtime import JobError, base_parser, run_entrypoint
from ml_engine.contracts.common import MetricDirection
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.cv import EvaluationMode, FoldAssignment
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.explain import AblationEntry, AblationReport
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.evaluation.metrics import metric_direction
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    read_model,
    read_model_if_exists,
    read_parquet,
    write_model,
)
from ml_engine.models import get_plugin
from ml_engine.preprocessing import FeaturePipeline, plan_columns
from ml_engine.preprocessing.folds import fit_pipelines
from ml_engine.splitting.folds import development_rows, fold_rows
from ml_engine.training import train_model

LOGGER = logging.getLogger("ml_factory.jobs.ablation")

MAX_ABLATION_FEATURES = 10
#: Fold-fit ceiling: (candidates + 1) * K must stay at or below this (design C7, Req 10.3).
CV_MAX_ABLATION_FITS = 60


def run_ablation(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
    features: list[str],
) -> AblationReport:
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    preparation = read_model_if_exists(store, layout.preparation, PreparationReport)
    if preparation is None:
        raise JobError("NOT_TRAINED", "Ablation needs a completed preparation stage.")
    spec = next((m for m in config.enabled_models_specs() if m.name == model_name), None)
    if spec is None:
        raise JobError("MODEL_NOT_CONFIGURED", f"'{model_name}' was not trained in this run.")
    candidates = [f for f in dict.fromkeys(features) if f in preparation.usable_features]
    if not candidates:
        raise JobError(
            "NO_ABLATION_FEATURES",
            "None of the requested features was used by the models: " + ", ".join(features[:20]),
        )
    candidates = candidates[:MAX_ABLATION_FEATURES]

    eda = read_model_if_exists(store, layout.eda, EdaReport)
    plugin = get_plugin(model_name)
    usable = list(preparation.usable_features)
    direction = metric_direction(config.problem_type, config.primary_metric)

    if config.split.evaluation_mode is EvaluationMode.CV:
        return _run_cv_ablation(
            store,
            layout,
            experiment_id=experiment_id,
            model_name=model_name,
            config=config,
            spec=spec,
            eda=eda,
            plugin=plugin,
            usable=usable,
            candidates=candidates,
            direction=direction,
        )

    train = read_parquet(store, layout.train_dataset)
    validation = read_parquet(store, layout.validation_dataset)

    baseline = _score_without(None, usable, train, validation, config, eda, plugin, spec.params)
    if baseline is None:
        raise JobError("BASELINE_UNSCORABLE", "The baseline model produced no primary metric.")
    LOGGER.info("Baseline %s=%s", config.primary_metric, baseline)

    entries: list[AblationEntry] = []
    for feature in candidates:
        try:
            score = _score_without(
                feature, usable, train, validation, config, eda, plugin, spec.params
            )
        except Exception as error:  # one feature failing never loses the others
            LOGGER.exception("Ablation without %s failed", feature)
            entries.append(
                AblationEntry(
                    feature=feature, status="failed", message=f"{type(error).__name__}: {error}"
                )
            )
            continue
        drop = None
        if score is not None:
            raw = baseline - score
            drop = round(raw if direction is MetricDirection.MAXIMIZE else -raw, 6)
        entries.append(
            AblationEntry(
                feature=feature,
                score_without=None if score is None else round(score, 6),
                drop=drop,
                status="completed" if score is not None else "failed",
                message=None if score is not None else "The primary metric was not computable.",
            )
        )

    report = AblationReport(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        model_name=model_name,
        primary_metric=config.primary_metric,
        direction=direction,
        baseline_score=round(baseline, 6),
        entries=entries,
    )
    write_model(store, layout.ablation, report)
    return report


def _run_cv_ablation(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    model_name: str,
    config: ExperimentConfig,
    spec,
    eda: EdaReport | None,
    plugin,
    usable: list[str],
    candidates: list[str],
    direction: MetricDirection,
) -> AblationReport:
    """Ablation on the stored Fold_Assignment: baseline and each candidate on every CV_Fold.

    Preprocessing is refit on each fold's training rows (never the Test_Fold), the model is
    retrained with the Run's hyperparameters on the fold's training rows and scored on the
    fold's validation rows, and the per-candidate score is the mean across folds.
    """
    assignment = read_model_if_exists(store, layout.fold_assignment, FoldAssignment)
    if assignment is None or assignment.evaluation_mode is not EvaluationMode.CV:
        raise JobError(
            "FOLD_ASSIGNMENT_MISSING",
            "CV_Mode ablation needs the fold assignment written by preparation.",
        )
    folds = assignment.folds
    # (candidates + 1) * K <= 60: the baseline counts as one "candidate" (Req 10.3).
    max_candidates = max(0, CV_MAX_ABLATION_FITS // folds - 1)
    truncated = max(0, len(candidates) - max_candidates)
    if truncated:
        LOGGER.info(
            "Capping ablation candidates from %d to %d so (candidates + 1) x K=%d stays <= %d",
            len(candidates),
            max_candidates,
            folds,
            CV_MAX_ABLATION_FITS,
        )
        candidates = candidates[:max_candidates]

    development = read_parquet(store, layout.development_dataset)  # never the Test_Fold
    development.index = pd.Index(development_rows(assignment), name="row_id")

    baseline_drops = _cv_fold_scores(
        None, usable, development, assignment, config, eda, plugin, spec.params
    )
    baseline_values = [score for score in baseline_drops if score is not None]
    if not baseline_values:
        raise JobError("BASELINE_UNSCORABLE", "The baseline model produced no primary metric.")
    baseline_mean = statistics.fmean(baseline_values)
    LOGGER.info("CV baseline %s mean=%s over %d folds", config.primary_metric, baseline_mean, folds)

    entries: list[AblationEntry] = []
    for feature in candidates:
        try:
            scores = _cv_fold_scores(
                feature, usable, development, assignment, config, eda, plugin, spec.params
            )
        except Exception as error:  # one feature failing never loses the others
            LOGGER.exception("CV ablation without %s failed", feature)
            entries.append(
                AblationEntry(
                    feature=feature, status="failed", message=f"{type(error).__name__}: {error}"
                )
            )
            continue
        present = [s for s in scores if s is not None]
        if not present:
            entries.append(
                AblationEntry(
                    feature=feature,
                    status="failed",
                    message="The primary metric was not computable on any fold.",
                )
            )
            continue
        # Per-fold drop, oriented so positive always means worse without the feature.
        fold_drops = [
            round(_oriented(baseline - score, direction), 6)
            for baseline, score in zip(baseline_drops, scores, strict=True)
            if baseline is not None and score is not None
        ]
        score_mean = statistics.fmean(present)
        entries.append(
            AblationEntry(
                feature=feature,
                score_without=round(score_mean, 6),
                drop=round(statistics.fmean(fold_drops), 6),
                status="completed",
                fold_drops=fold_drops,
                std_drop=(round(statistics.stdev(fold_drops), 6) if len(fold_drops) > 1 else 0.0),
            )
        )

    report = AblationReport(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        model_name=model_name,
        primary_metric=config.primary_metric,
        direction=direction,
        baseline_score=round(baseline_mean, 6),
        entries=entries,
        fold="cv",
        fold_count=folds,
        truncated=truncated,
        note=(
            "Each feature was removed and the model retrained on every CV fold's training rows "
            "(the identical stored folds). The holdout test set, if any, was not used."
        ),
    )
    write_model(store, layout.ablation, report)
    return report


def _oriented(raw: float, direction: MetricDirection) -> float:
    """Orient a (baseline minus without) difference so positive always means worse without it."""
    return raw if direction is MetricDirection.MAXIMIZE else -raw


def _cv_fold_scores(
    feature: str | None,
    usable: list[str],
    development: pd.DataFrame,
    assignment: FoldAssignment,
    config: ExperimentConfig,
    eda: EdaReport | None,
    plugin,
    params: dict,
) -> list[float | None]:
    """Primary score on each CV_Fold, refitting preprocessing on that fold's training rows."""
    keep = [f for f in usable if f != feature]
    columns = [*keep, config.target_column]
    plan = plan_columns(development[columns], keep, config.preprocessing, eda)
    scores: list[float | None] = []
    for fold in range(assignment.folds):
        train_ids, validation_ids = fold_rows(assignment, fold)
        train_frame = development.loc[train_ids, columns]
        validation_frame = development.loc[validation_ids, columns]
        if plan.is_empty:
            scores.append(None)
            continue
        pipelines = fit_pipelines(
            plan,
            config.preprocessing,
            [plugin.preprocessing_strategy],
            train_frame,
            config.target_column,
        )
        metadata, _ = train_model(
            plugin,
            experiment_id=config.experiment_id,
            problem_type=config.problem_type,
            pipeline=pipelines[plugin.preprocessing_strategy],
            train_frame=train_frame,
            validation_frame=validation_frame,
            target_column=config.target_column,
            param_overrides=dict(params),
            primary_metric=config.primary_metric,
            random_seed=config.split.random_seed,
            class_weighting=config.class_weighting,
            compute_train_metrics=False,
        )
        scores.append(metadata.primary_score)
    return scores


def _score_without(
    feature: str | None,
    usable: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    config: ExperimentConfig,
    eda: EdaReport | None,
    plugin,
    params: dict,
) -> float | None:
    """Refit preprocessing on the training fold without ``feature``, retrain, score validation."""
    keep = [f for f in usable if f != feature]
    columns = [*keep, config.target_column]
    train_frame = train[columns]
    validation_frame = validation[columns]
    plan = plan_columns(train_frame, keep, config.preprocessing, eda)
    if plan.is_empty:
        return None
    pipeline = FeaturePipeline(plan, config.preprocessing, strategy=plugin.preprocessing_strategy)
    pipeline.fit(train_frame, train_frame[config.target_column])
    metadata, _ = train_model(
        plugin,
        experiment_id=config.experiment_id,
        problem_type=config.problem_type,
        pipeline=pipeline,
        train_frame=train_frame,
        validation_frame=validation_frame,
        target_column=config.target_column,
        param_overrides=dict(params),
        primary_metric=config.primary_metric,
        random_seed=config.split.random_seed,
        class_weighting=config.class_weighting,
    )
    return metadata.primary_score


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_ablation(
        store,
        layout,
        experiment_id=args.experiment_id,
        model_name=args.model_name,
        features=[f for f in args.features.split(",") if f],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory ablation job (retrain without each candidate feature)")
    parser.prog = "ablation-job"
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--features", required=True, help="Comma-separated feature names.")
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda _args, layout: layout.ablation_failure,
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
