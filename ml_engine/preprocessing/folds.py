"""Fold-aware preprocessing fits (ML Factory 2.0, cross-validation design C3).

Rows are selected *before* ``FeaturePipeline.fit`` is called, so a CV_Fold's pipeline never sees
that fold's validation rows and no pipeline ever sees a Test_Fold row (Req 3.6, 5.1-5.4). The
column plan is shared by every fit (computed once on the Development_Set, design D13); only the
fitted statistics (imputation values, scaling, encodings) differ per fold.

``FeaturePipeline.metadata()`` (inherited, unchanged) always reports ``fitted_on="train"``. The
pipelines fitted here carry their real fit scope in the ``cv_fitted_on`` attribute
(``"fold_{i}_train"`` or ``"development"``), and :func:`pipeline_metadata` returns the
inherited metadata with ``fitted_on`` replaced by it. Callers building a report use
:func:`pipeline_metadata` instead of ``pipeline.metadata()``.
"""

from collections.abc import Sequence

import pandas as pd

from ml_engine.contracts.config import PreprocessingConfig
from ml_engine.contracts.cv import FoldAssignment
from ml_engine.contracts.model import PreprocessingMetadata
from ml_engine.contracts.treatment import AppliedTreatment, TreatmentConfig
from ml_engine.preprocessing.outliers import OutlierTreatment
from ml_engine.preprocessing.pipeline import FeaturePipeline, PreprocessingError
from ml_engine.preprocessing.planning import ColumnPlan
from ml_engine.splitting.folds import development_rows, fold_rows

#: Instance attribute recording what a pipeline was fitted on.
FITTED_ON_ATTRIBUTE = "cv_fitted_on"
DEVELOPMENT_FIT = "development"


def fold_fit_label(fold: int) -> str:
    return f"fold_{fold}_train"


class FoldPreprocessingError(PreprocessingError):
    """A pipeline could not be fitted on one CV_Fold's training rows (or the Development_Set)."""

    def __init__(self, message: str, *, fold_index: int | None, strategy: str) -> None:
        super().__init__(message)
        self.fold_index = fold_index
        self.strategy = strategy


def fit_pipelines(
    plan: ColumnPlan,
    config: PreprocessingConfig,
    strategies: Sequence[str],
    frame: pd.DataFrame,
    target_column: str,
    *,
    fitted_on: str = "train",
) -> dict[str, FeaturePipeline]:
    """One fitted pipeline per strategy, each fitted on exactly the rows of ``frame``."""
    pipelines: dict[str, FeaturePipeline] = {}
    target = frame[target_column] if target_column in frame.columns else None
    for strategy in strategies:
        pipeline = FeaturePipeline(plan, config, strategy=strategy)  # type: ignore[arg-type]
        pipeline.fit(frame, target)
        setattr(pipeline, FITTED_ON_ATTRIBUTE, fitted_on)
        pipelines[strategy] = pipeline
    return pipelines


def fit_fold_pipelines(
    plan: ColumnPlan,
    config: PreprocessingConfig,
    strategies: Sequence[str],
    development: pd.DataFrame,
    assignment: FoldAssignment,
    target_column: str,
    *,
    treatment_config: TreatmentConfig | None = None,
    random_seed: int = 42,
) -> tuple[
    list[dict[str, FeaturePipeline]],
    dict[str, FeaturePipeline],
    list[OutlierTreatment | None],
    OutlierTreatment | None,
    list[AppliedTreatment],
]:
    """Per CV_Fold: fit on that fold's training rows only. Plus one fit on the Development_Set.

    ``development`` is indexed by row id (the ids of ``assignment``). It may also hold Test_Fold
    rows: only the rows the assignment names for a fit are ever passed to ``fit``.

    When ``treatment_config`` is set (and not ``none``), an ``OutlierTreatment`` is fit on each
    fold's training rows and applied (drop/cap) *before* that fold's pipeline is fit, so the
    pipeline sees post-outlier data (training-data-treatment design C3). The per-fold treatments
    and applied summaries are returned so the caller can persist them and apply them at train time
    (the shared ``development.parquet`` stays untreated). A development-level treatment is fit on
    the whole Development_Set and the full pipeline is fit on those treated rows, so the Refit_Model
    (which trains on the whole Development_Set) sees the same outlier -> impute -> scale order; that
    development treatment is returned to be persisted and applied at refit time.
    """
    missing = sorted(set(development_rows(assignment)) - set(development.index))
    if missing:
        raise ValueError(
            f"{len(missing)} Development_Set row id(s) are not in the frame index "
            f"(first: {missing[:5]}); index the frame by row id"
        )
    treating = treatment_config is not None and not treatment_config.is_noop
    per_fold: list[dict[str, FeaturePipeline]] = []
    fold_treatments: list[OutlierTreatment | None] = []
    applied: list[AppliedTreatment] = []
    for fold in range(assignment.folds):
        train_ids, _validation_ids = fold_rows(assignment, fold)
        fold_frame = development.loc[train_ids]
        treatment: OutlierTreatment | None = None
        if treating:
            fold_frame, treatment, summary = _treat_fold(
                treatment_config, fold_frame, target_column, fold, random_seed
            )
            applied.append(summary)
        fold_treatments.append(treatment)
        per_fold.append(
            _fit_scope(
                plan,
                config,
                strategies,
                fold_frame,
                target_column,
                fitted_on=fold_fit_label(fold),
                fold_index=fold,
            )
        )
    development_frame = development.loc[development_rows(assignment)]
    development_treatment: OutlierTreatment | None = None
    if treating:
        development_frame, development_treatment, dev_summary = _treat_frame(
            treatment_config, development_frame, target_column, DEVELOPMENT_FIT, random_seed
        )
        applied.append(dev_summary)
    full = _fit_scope(
        plan,
        config,
        strategies,
        development_frame,
        target_column,
        fitted_on=DEVELOPMENT_FIT,
        fold_index=None,
    )
    return per_fold, full, fold_treatments, development_treatment, applied


def _treat_fold(
    treatment_config: TreatmentConfig,
    fold_frame: pd.DataFrame,
    target_column: str,
    fold: int,
    random_seed: int,
) -> tuple[pd.DataFrame, OutlierTreatment, AppliedTreatment]:
    """Fit + apply the fold's outlier treatment to its training rows, before the pipeline fit."""
    return _treat_frame(
        treatment_config, fold_frame, target_column, fold_fit_label(fold), random_seed
    )


def _treat_frame(
    treatment_config: TreatmentConfig,
    frame: pd.DataFrame,
    target_column: str,
    fitted_on: str,
    random_seed: int,
) -> tuple[pd.DataFrame, OutlierTreatment, AppliedTreatment]:
    """Fit + apply an outlier treatment to a set of training rows, before any pipeline fit."""
    feature_columns = [c for c in frame.columns if c != target_column]
    features = frame[feature_columns]
    target = frame[target_column] if target_column in frame.columns else None
    treatment = OutlierTreatment(treatment_config, random_seed=random_seed).fit(features, target)
    summary = treatment.report(features, fitted_on=fitted_on)
    treated_features, treated_target = treatment.apply(features, target)
    treated = treated_features.copy()
    if treated_target is not None:
        treated[target_column] = treated_target
    return treated[frame.columns], treatment, summary


def pipeline_metadata(pipeline: FeaturePipeline) -> PreprocessingMetadata:
    """The inherited metadata with ``fitted_on`` set to what the pipeline was fitted on."""
    metadata = pipeline.metadata()
    fitted_on = getattr(pipeline, FITTED_ON_ATTRIBUTE, None)
    if fitted_on is None or fitted_on == metadata.fitted_on:
        return metadata
    return metadata.model_copy(update={"fitted_on": fitted_on})


def _fit_scope(
    plan: ColumnPlan,
    config: PreprocessingConfig,
    strategies: Sequence[str],
    rows: pd.DataFrame,
    target_column: str,
    *,
    fitted_on: str,
    fold_index: int | None,
) -> dict[str, FeaturePipeline]:
    pipelines: dict[str, FeaturePipeline] = {}
    for strategy in strategies:
        scope = "the Development_Set" if fold_index is None else f"fold {fold_index}"
        try:
            pipelines.update(
                fit_pipelines(plan, config, [strategy], rows, target_column, fitted_on=fitted_on)
            )
        except Exception as error:  # surfaced as PREPROCESSING_FAILED with the fold index
            raise FoldPreprocessingError(
                f"Fitting '{strategy}' preprocessing on {scope} ({len(rows)} rows) failed: {error}",
                fold_index=fold_index,
                strategy=strategy,
            ) from error
    return pipelines
