"""Preparation with a Fold_Assignment: CV_Mode and/or a Test_Fold (CV design C4).

Called from ``jobs/preprocessing/main.py::run_preparation`` after the inherited target drop and
``_derive_features`` — so derived features exist before any row is assigned (Req 5.6) — and only
when the frozen split is in CV_Mode or carries a ``test_fraction``. Single_Split_Mode without a
Test_Fold never reaches this module.

Row ids are positions in the usable frame (after the target drop and derivation). The Test_Fold
is carved first; the column plan is computed on the Development_Set only (design D13). This job is
the only writer of ``validation/fold_assignment.json`` (Req 4.8).

CV_Mode writes ``datasets/development.parquet`` (ascending row id, ``index=False``), the
Development_Set preprocessors at the inherited ``preprocessing/preprocessor_{s}.joblib`` and one
set per fold at ``preprocessing/folds/fold_{i}/preprocessor_{s}.joblib``. Single_Split_Mode with a
Test_Fold writes the inherited ``train``/``validation`` parquet (a split of the Development_Set)
and inherited preprocessors fitted on its training side. Both write ``datasets/test.parquet``.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd

from jobs._common.runtime import JobError
from jobs.preprocessing.main import _serializable, required_strategies, treat_training_frame
from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.cv import (
    TRAIN_ONLY,
    CvPreparationSummary,
    CvScheme,
    EvaluationMode,
    FoldAssignment,
    FoldRowCounts,
    TestFoldSummary,
)
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.model import PreprocessingMetadata
from ml_engine.contracts.preparation import PreparationReport, SplitSummary
from ml_engine.contracts.proposals import DerivedFeature, RejectedProposal
from ml_engine.contracts.treatment import AppliedTreatment
from ml_engine.contracts.warnings import AnalysisWarning, sort_warnings
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    read_parquet,
    save_joblib,
    write_model,
    write_parquet,
)
from ml_engine.preprocessing import ColumnPlan, FeaturePipeline, PreprocessingError, plan_columns
from ml_engine.preprocessing.folds import (
    FoldPreprocessingError,
    fit_fold_pipelines,
    fit_pipelines,
    pipeline_metadata,
)
from ml_engine.splitting.folds import (
    FoldPlan,
    FoldPlanningError,
    carve_test_fold,
    dataset_fingerprint,
    fold_rows,
    plan_folds,
    test_rows,
)
from ml_engine.splitting.strategies import SplittingError

LOGGER = logging.getLogger("ml_factory.jobs.preparation.cv")


@dataclass(slots=True)
class UsableFrame:
    """What the inherited preparation steps produced before the split."""

    frame: pd.DataFrame
    rows_before: int
    rows_dropped: int
    requested: list[str]
    derived: list[DerivedFeature] = field(default_factory=list)
    rejected: list[RejectedProposal] = field(default_factory=list)
    derived_leakage: LeakageReport | None = None
    warnings: list[AnalysisWarning] = field(default_factory=list)


def uses_fold_assignment(config: ExperimentConfig) -> bool:
    """CV_Mode or a Test_Fold: the only cases that leave the inherited preparation path."""
    split = config.split
    return split.evaluation_mode is EvaluationMode.CV or split.test_fraction is not None


def prepare_with_folds(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
    config: ExperimentConfig,
    eda: EdaReport | None,
    usable: UsableFrame,
) -> PreparationReport:
    target_column = config.target_column
    frame = usable.frame.reset_index(drop=True)  # row id = position in the usable frame
    target = frame[target_column]
    warnings = list(usable.warnings)

    carve = _split_failed_on_error(
        lambda: carve_test_fold(frame, target, config.problem_type, config.split)
    )
    development_ids = carve.development_rows
    plan = plan_columns(frame.iloc[development_ids], usable.requested, config.preprocessing, eda)
    warnings.extend(plan.warnings)
    if plan.is_empty:
        raise JobError(
            "NO_USABLE_FEATURES",
            "No selected feature survived preprocessing. Dropped: "
            + ", ".join(f"{k} ({v})" for k, v in plan.dropped.items()),
        )
    columns = [*plan.usable, target_column]
    fingerprint = dataset_fingerprint(frame, columns)
    cv_mode = config.split.evaluation_mode is EvaluationMode.CV
    strategies = required_strategies(config)

    development_digest: str | None = None
    if cv_mode:
        development = frame.loc[development_ids, columns]  # ascending row id
        write_parquet(store, layout.development_dataset, _serializable(development))
        # The digest of the rows exactly as written, so training can verify what it reads.
        written = read_parquet(store, layout.development_dataset)
        development_digest = dataset_fingerprint(written, list(written.columns))
    else:
        development_digest = dataset_fingerprint(frame.loc[development_ids, columns], columns)

    fold_plan = _split_failed_on_error(
        lambda: plan_folds(
            frame,
            target,
            config.problem_type,
            config.split,
            fingerprint=fingerprint,
            experiment_id=experiment_id,
            development_digest=development_digest,
        )
    )
    assignment = fold_plan.assignment
    warnings.extend(fold_plan.warnings)

    if cv_mode:
        split_summary, cv_summary, preprocessing, applied_treatment = _prepare_cv(
            store, layout, config, plan, strategies, development, assignment, fold_plan.warnings
        )
    else:
        split_summary, preprocessing, applied_treatment = _prepare_single(
            store, layout, config, plan, strategies, frame, columns, fold_plan
        )
        cv_summary = None

    test_summary = None
    if assignment.test_row_count:
        test_ids = test_rows(assignment)  # ascending row id
        test_frame = frame.loc[test_ids, columns]
        write_parquet(store, layout.test_dataset, _serializable(test_frame))
        test_summary = TestFoldSummary(
            fraction=float(config.split.test_fraction or 0.0),
            row_count=len(test_ids),
            development_row_count=assignment.development_row_count,
            class_distribution=_distribution(config.problem_type, test_frame[target_column]),
        )

    write_model(store, layout.fold_assignment, assignment)

    report = PreparationReport(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        problem_type=config.problem_type,
        target_column=target_column,
        requested_features=usable.requested,
        usable_features=plan.usable,
        derived_features=usable.derived,
        rejected_proposals=usable.rejected,
        derived_leakage=usable.derived_leakage,
        dropped_features=plan.dropped,
        rows_before=usable.rows_before,
        rows_after=len(frame),
        rows_dropped_missing_target=usable.rows_dropped,
        split=split_summary,
        preprocessing=preprocessing,
        warnings=sort_warnings(warnings),
        cross_validation=cv_summary,
        test_fold=test_summary,
        applied_treatment=applied_treatment,
    )
    write_model(store, layout.preparation, report)
    LOGGER.info(
        "Prepared %s: %d Development_Set rows, %d Test_Fold rows",
        "CV folds" if cv_mode else "a single split",
        assignment.development_row_count,
        assignment.test_row_count,
    )
    return report


def _prepare_cv(
    store: ObjectStore,
    layout: ExperimentLayout,
    config: ExperimentConfig,
    plan: ColumnPlan,
    strategies: list[str],
    development: pd.DataFrame,
    assignment: FoldAssignment,
    planning_warnings: list[AnalysisWarning],
) -> tuple[
    SplitSummary, CvPreparationSummary, dict[str, PreprocessingMetadata], list[AppliedTreatment]
]:
    target_column = config.target_column
    try:
        per_fold, full, fold_treatments, development_treatment, applied_treatment = (
            fit_fold_pipelines(
                plan,
                config.preprocessing,
                strategies,
                development,
                assignment,
                target_column,
                treatment_config=config.treatment,
                random_seed=config.split.random_seed,
            )
        )
    except FoldPreprocessingError as error:
        details = {"strategy": error.strategy}
        if error.fold_index is not None:
            details["fold_index"] = str(error.fold_index)
        raise JobError("PREPROCESSING_FAILED", str(error), details) from error

    for fold, pipelines in enumerate(per_fold):
        for strategy, pipeline in pipelines.items():
            save_joblib(store, layout.fold_preprocessor(fold, strategy), pipeline)
    # Persist each fold's outlier treatment so training can apply it at train time; the shared
    # development.parquet stays untreated (each fold's drop/cap differs).
    for fold, treatment in enumerate(fold_treatments):
        if treatment is not None:
            save_joblib(store, layout.fold_outlier_treatment(fold), treatment)
    # The development-level treatment: the Refit_Model applies it to the whole Development_Set.
    if development_treatment is not None:
        save_joblib(store, layout.outlier_treatment, development_treatment)
    for strategy, pipeline in full.items():
        save_joblib(store, layout.preprocessor(strategy), pipeline)

    labels = development[target_column]
    fold_counts: list[FoldRowCounts] = []
    validated: list[int] = []
    for fold in range(assignment.folds):
        train_ids, validation_ids = fold_rows(assignment, fold)
        validated.extend(validation_ids)
        fold_counts.append(
            FoldRowCounts(
                fold_index=fold,
                train_row_count=len(train_ids),
                validation_row_count=len(validation_ids),
                train_class_distribution=_distribution(config.problem_type, labels.loc[train_ids]),
                validation_class_distribution=_distribution(
                    config.problem_type, labels.loc[validation_ids]
                ),
            )
        )
    assert assignment.scheme is not None  # CV_Mode
    development_rows = len(development)
    # The rare-class fallback to shuffled K-fold means the folds are not stratified.
    stratified = assignment.scheme is CvScheme.STRATIFIED_KFOLD and not any(
        warning.rule == "cv_stratification_not_possible" for warning in planning_warnings
    )
    split_summary = SplitSummary(
        strategy=assignment.scheme.value,
        train_row_count=development_rows,
        validation_row_count=len(validated),
        validation_fraction_actual=len(validated) / development_rows if development_rows else 0.0,
        random_seed=assignment.random_seed,
        stratified=stratified,
        train_class_distribution=_distribution(config.problem_type, labels),
        validation_class_distribution=_distribution(config.problem_type, labels.loc[validated]),
    )
    cv_summary = CvPreparationSummary(
        scheme=assignment.scheme,
        folds=assignment.folds,
        development_row_count=development_rows,
        train_only_row_count=sum(1 for value in assignment.assignments if value == TRAIN_ONLY),
        fold_rows=fold_counts,
        fold_output_feature_counts={
            strategy: [len(pipelines[strategy].feature_names) for pipelines in per_fold]
            for strategy in strategies
        },
    )
    preprocessing = {strategy: pipeline_metadata(full[strategy]) for strategy in strategies}
    return split_summary, cv_summary, preprocessing, applied_treatment


def _prepare_single(
    store: ObjectStore,
    layout: ExperimentLayout,
    config: ExperimentConfig,
    plan: ColumnPlan,
    strategies: list[str],
    frame: pd.DataFrame,
    columns: list[str],
    fold_plan: FoldPlan,
) -> tuple[SplitSummary, dict[str, PreprocessingMetadata], list[AppliedTreatment]]:
    """Single_Split_Mode with a Test_Fold: the inherited split, of the Development_Set only."""
    split = fold_plan.single_split
    assert split is not None  # plan_folds always splits the Development_Set in this mode
    train_frame = frame.loc[split.train_index, columns]
    validation_frame = frame.loc[split.validation_index, columns]

    # Outlier handling on the training rows only, before the pipeline fit (same as main.py).
    train_frame, treatment, treatment_summary = treat_training_frame(
        config, train_frame, config.target_column, fitted_on="train"
    )
    applied_treatment = [treatment_summary] if treatment_summary is not None else []
    if treatment is not None:
        save_joblib(store, layout.outlier_treatment, treatment)

    preprocessing: dict[str, PreprocessingMetadata] = {}
    for strategy in strategies:
        try:
            pipeline: FeaturePipeline = fit_pipelines(
                plan, config.preprocessing, [strategy], train_frame, config.target_column
            )[strategy]
        except PreprocessingError as error:
            raise JobError("PREPROCESSING_FAILED", str(error)) from error
        save_joblib(store, layout.preprocessor(strategy), pipeline)
        preprocessing[strategy] = pipeline_metadata(pipeline)

    write_parquet(store, layout.train_dataset, _serializable(train_frame))
    write_parquet(store, layout.validation_dataset, _serializable(validation_frame))
    return split.summary, preprocessing, applied_treatment


def _distribution(problem_type: ProblemType, labels: pd.Series) -> dict[str, int] | None:
    if problem_type is ProblemType.REGRESSION:
        return None
    return {str(k): int(v) for k, v in labels.astype("string").value_counts().items()}


def _split_failed_on_error[T](build: Callable[[], T]) -> T:
    """Fold planning failures become ``SPLIT_FAILED`` with the reason and its numbers."""
    try:
        return build()
    except FoldPlanningError as error:
        details = {"reason": error.reason.value}
        details.update({key: str(value) for key, value in error.details.items()})
        raise JobError("SPLIT_FAILED", error.message, details) from error
    except SplittingError as error:
        raise JobError("SPLIT_FAILED", str(error)) from error
