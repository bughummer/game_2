"""Preparation job: validate the configuration, split the data and fit preprocessing.

Runs as a SageMaker Processing job between EDA and training. It is the only place where a
learned transform is fitted, and it fits exclusively on the training fold — which is what
guarantees that no model in the experiment sees validation information.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime

import pandas as pd

from jobs._common.runtime import JobError, base_parser, run_entrypoint
from ml_engine.contracts.common import LeakageRiskLevel, Severity, WarningCategory
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.cv import EvaluationMode
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.contracts.proposals import DerivedFeature, RejectedProposal
from ml_engine.contracts.treatment import AppliedTreatment
from ml_engine.contracts.warnings import AnalysisWarning, sort_warnings
from ml_engine.features import (
    apply_proposals,
    column_types_from_frame,
    derivation_warnings,
    derive_and_screen,
    validate_proposals,
)
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    load_dataset,
    read_model,
    read_model_if_exists,
    save_joblib,
    write_model,
    write_parquet,
)
from ml_engine.models import resolve_plugin
from ml_engine.preprocessing import FeaturePipeline, PreprocessingError, plan_columns
from ml_engine.preprocessing.outliers import OutlierTreatment, OutlierTreatmentError
from ml_engine.splitting import SplittingError, split_dataset

LOGGER = logging.getLogger("ml_factory.jobs.preparation")


def required_strategies(config: ExperimentConfig) -> list[str]:
    """The distinct preprocessing strategies the configured models need."""
    strategies: list[str] = []
    for spec in config.enabled_models_specs():
        # An ensemble name resolves to its EnsemblePlugin (dense_numeric); plain/grid names to
        # their base plugin (ensembles-and-learners design C4).
        strategy = resolve_plugin(config, spec.name).preprocessing_strategy
        if strategy not in strategies:
            strategies.append(strategy)
    return strategies or ["dense_numeric"]


def treat_training_frame(
    config: ExperimentConfig,
    train_frame: pd.DataFrame,
    target_column: str,
    *,
    fitted_on: str,
) -> tuple[pd.DataFrame, OutlierTreatment | None, AppliedTreatment | None]:
    """Fit + apply the approved outlier treatment to a training frame (training-data-treatment).

    Returns the treated frame, the fitted ``OutlierTreatment`` (to persist), and the applied
    summary. When no treatment is configured (or it is ``none``), returns the frame unchanged and
    ``(None, None)`` so the chain is byte-for-byte today's chain.
    """
    treatment_config = config.treatment
    if treatment_config is None or treatment_config.is_noop:
        return train_frame, None, None
    feature_columns = [c for c in train_frame.columns if c != target_column]
    features = train_frame[feature_columns]
    target = train_frame[target_column] if target_column in train_frame.columns else None
    treatment = OutlierTreatment(treatment_config, random_seed=config.split.random_seed).fit(
        features, target
    )
    summary = treatment.report(features, fitted_on=fitted_on)
    try:
        treated_features, treated_target = treatment.apply(features, target)
    except OutlierTreatmentError as error:
        raise JobError(error.code, str(error)) from error
    treated = treated_features.copy()
    if treated_target is not None:
        treated[target_column] = treated_target
    return treated[train_frame.columns], treatment, summary


def run_preparation(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    experiment_id: str,
) -> PreparationReport:
    """Produce the training and validation folds plus every fitted preprocessing pipeline."""
    config = read_model(store, layout.experiment_config, ExperimentConfig)
    eda = read_model_if_exists(store, layout.eda, EdaReport)
    warnings: list[AnalysisWarning] = []

    LOGGER.info("Loading dataset %s", config.dataset.uri)
    dataset = load_dataset(store, config.dataset.uri, config.dataset.file_format)
    frame = dataset.frame
    rows_before = len(frame)

    if config.target_column not in frame.columns:
        raise JobError(
            "TARGET_COLUMN_MISSING",
            f"Target column '{config.target_column}' is not present in the dataset.",
        )

    frame = frame[frame[config.target_column].notna()]
    rows_dropped = rows_before - len(frame)
    if rows_dropped:
        warnings.append(
            AnalysisWarning(
                rule="rows_dropped_missing_target",
                category=WarningCategory.TARGET,
                severity=Severity.MEDIUM
                if rows_dropped / max(rows_before, 1) > 0.05
                else Severity.LOW,
                message=f"{rows_dropped} row(s) without a target value were dropped.",
                column=config.target_column,
                details={"dropped_rows": rows_dropped},
            )
        )
    if frame.empty:
        raise JobError("NO_USABLE_ROWS", "No rows remain after dropping rows without a target.")

    frame, derived, rejected, derived_leakage = _derive_features(
        frame, config, experiment_id=experiment_id, warnings=warnings
    )

    requested = list(config.feature_selection.selected_features)
    # An approved proposal is already a decision to use the feature; requiring it to be listed
    # again would let a derived column be computed and then silently never reach the model.
    requested.extend(feature.name for feature in derived if feature.name not in requested)

    # --- ML Factory 2.0: CV_Mode or a Test_Fold (cross-validation design C4). A single split
    # without a Test_Fold never takes this branch; the inherited body below is unchanged.
    if config.split.evaluation_mode is EvaluationMode.CV or config.split.test_fraction is not None:
        from jobs.preprocessing.cv import UsableFrame, prepare_with_folds

        return prepare_with_folds(
            store,
            layout,
            experiment_id=experiment_id,
            config=config,
            eda=eda,
            usable=UsableFrame(
                frame=frame,
                rows_before=rows_before,
                rows_dropped=rows_dropped,
                requested=requested,
                derived=derived,
                rejected=rejected,
                derived_leakage=derived_leakage,
                warnings=warnings,
            ),
        )

    plan = plan_columns(frame, requested, config.preprocessing, eda)
    warnings.extend(plan.warnings)
    if plan.is_empty:
        raise JobError(
            "NO_USABLE_FEATURES",
            "No selected feature survived preprocessing. Dropped: "
            + ", ".join(f"{k} ({v})" for k, v in plan.dropped.items()),
        )

    try:
        split = split_dataset(frame, frame[config.target_column], config.problem_type, config.split)
    except SplittingError as error:
        raise JobError("SPLIT_FAILED", str(error)) from error
    warnings.extend(split.warnings)

    columns = [*plan.usable, config.target_column]
    train_frame = frame.loc[split.train_index, columns]
    validation_frame = frame.loc[split.validation_index, columns]

    # --- ML Factory 2.0: outlier handling runs on the training rows only, before the pipeline
    # fits, so the imputer/scaler see post-outlier data. Validation rows are never treated. ---
    train_frame, treatment, treatment_summary = treat_training_frame(
        config, train_frame, config.target_column, fitted_on="train"
    )
    applied_treatment = [treatment_summary] if treatment_summary is not None else []
    if treatment is not None:
        save_joblib(store, layout.outlier_treatment, treatment)

    preprocessing_metadata = {}
    for strategy in required_strategies(config):
        try:
            pipeline = FeaturePipeline(plan, config.preprocessing, strategy=strategy)
            pipeline.fit(train_frame, train_frame[config.target_column])
        except PreprocessingError as error:
            raise JobError("PREPROCESSING_FAILED", str(error)) from error
        save_joblib(store, layout.preprocessor(strategy), pipeline)
        preprocessing_metadata[strategy] = pipeline.metadata()
        LOGGER.info(
            "Fitted '%s' preprocessing: %d output features",
            strategy,
            preprocessing_metadata[strategy].output_feature_count,
        )

    write_parquet(store, layout.train_dataset, _serializable(train_frame))
    write_parquet(store, layout.validation_dataset, _serializable(validation_frame))

    report = PreparationReport(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        problem_type=config.problem_type,
        target_column=config.target_column,
        requested_features=requested,
        usable_features=plan.usable,
        derived_features=derived,
        rejected_proposals=rejected,
        derived_leakage=derived_leakage,
        dropped_features=plan.dropped,
        rows_before=rows_before,
        rows_after=len(frame),
        rows_dropped_missing_target=rows_dropped,
        split=split.summary,
        preprocessing=preprocessing_metadata,
        warnings=sort_warnings(warnings),
        applied_treatment=applied_treatment,
    )
    write_model(store, layout.preparation, report)
    LOGGER.info(
        "Prepared %d training rows and %d validation rows",
        report.split.train_row_count,
        report.split.validation_row_count,
    )
    return report


def _derive_features(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    *,
    experiment_id: str,
    warnings: list[AnalysisWarning],
) -> tuple[pd.DataFrame, list[DerivedFeature], list[RejectedProposal], LeakageReport | None]:
    """Apply the approved derived-feature specifications, then screen what they produced.

    Derivation happens here rather than in the EDA job because a proposal is approved after the
    profile exists. Every operation is row-wise, so computing before the split cannot move
    information across the fold boundary.

    The specifications are re-validated against the frame as it actually is: a configuration
    frozen against an earlier version of the dataset can name a column that has since been
    renamed, and one that reads the target must never be computed, wherever it came from.
    """
    if not config.derived_features:
        return frame, [], [], None

    accepted, rejected = validate_proposals(
        config.derived_features,
        column_types_from_frame(frame),
        target_column=config.target_column,
    )
    for refusal in rejected:
        warnings.append(
            AnalysisWarning(
                rule="derived_feature_rejected",
                category=WarningCategory.FEATURE,
                severity=Severity.HIGH,
                message=(
                    f"Approved derived feature '{refusal.name}' was not computed: {refusal.message}"
                ),
                column=refusal.name,
                details={"reason": refusal.reason.value},
            )
        )

    try:
        frame, derived, _subset_eda, leakage = derive_and_screen(
            frame,
            accepted,
            experiment_id=experiment_id,
            target_column=config.target_column,
            problem_type=config.problem_type,
            source_uri=config.dataset.uri,
            file_format=config.dataset.file_format,
        )
    except Exception:  # screening must never lose an otherwise prepared experiment
        LOGGER.exception("Deriving or leakage-screening derived features failed")
        frame, derived = apply_proposals(frame, accepted)
        leakage = None

    warnings.extend(derivation_warnings(derived, row_count=len(frame)))
    if not derived:
        return frame, derived, rejected, None

    LOGGER.info("Derived %d feature(s): %s", len(derived), ", ".join(f.name for f in derived))
    warnings.extend(_leakage_warnings(leakage))
    return frame, derived, rejected, leakage


def _leakage_warnings(report: LeakageReport | None) -> list[AnalysisWarning]:
    """Surface the serious derived-feature findings in the ordinary warning stream."""
    if report is None:
        return []
    serious = {LeakageRiskLevel.CONFIRMED_DUPLICATE, LeakageRiskLevel.POTENTIAL_LEAKAGE}
    return [
        AnalysisWarning(
            rule=f"derived_{finding.rule}",
            category=WarningCategory.LEAKAGE,
            severity=finding.severity,
            message=(
                f"Derived feature '{finding.feature}' was flagged by leakage screening: "
                f"{finding.explanation}"
            ),
            column=finding.feature,
            recommended_action=finding.recommended_action,
        )
        for finding in report.findings
        if finding.risk_level in serious
    ]


def _serializable(frame: pd.DataFrame) -> pd.DataFrame:
    """Parquet cannot store mixed-type object columns; cast the stragglers to string."""
    result = frame.copy()
    for column in result.columns:
        if result[column].dtype == object:
            result[column] = result[column].astype("string")
    return result


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_preparation(store, layout, experiment_id=args.experiment_id)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory preparation job (validation, splitting, preprocessing)")
    parser.prog = "preparation-job"
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda _args, layout: layout.path("validation/failure.json"),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
