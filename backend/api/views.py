"""Shared response shapes for the REST API (control-ui-api design C3).

The API returns the same summaries the MCP does, so a UI and a chat agent see one experiment the
same way. ``experiment_summary`` mirrors ``mcp_server/server.py::_experiment_summary`` exactly, and
the preview caps are the same constants the MCP's ``preview_dataset`` enforces (requirement 9.5 /
3.3): a preview is bounded, never a full dataset.
"""

from typing import Any

#: A dataset preview is bounded — the same caps the MCP's ``preview_dataset`` applies.
PREVIEW_ROW_CAP = 20
PREVIEW_COLUMN_CAP = 100
#: The requirements.txt returned inline by the environment view is capped; the full file is a
#: download (same as the MCP's ``get_environment``).
REQUIREMENTS_LINE_CAP = 400


def experiment_summary(record) -> dict[str, Any]:
    """The bounded status view of one experiment (identical to the MCP's summary)."""
    return {
        "experiment_id": record.experiment_id,
        "name": record.name,
        "status": record.status.value,
        "current_stage": record.current_stage,
        "target_column": record.target_column,
        "created_by": record.created_by,
        "best_model": record.best_model,
        "best_score": record.best_score,
        "primary_metric": record.primary_metric,
        "artifact_prefix": record.artifact_prefix,
    }


def requirements_uri(experiments, experiment_id: str) -> str:
    """Where the full requirements.txt lives, for download outside the chat (MCP parity)."""
    from ml_engine.io import ExperimentLayout

    return ExperimentLayout(base=experiments.get(experiment_id).artifact_prefix).requirements


def resolved_evaluation(store, artifact_prefix: str) -> dict[str, Any]:
    """The resolved evaluation decision read back from the frozen ExperimentConfig (Req 1.7).

    The service resolves defaults (mode, K, scheme) into ``config.split`` before any job runs, so
    the API echoes what was frozen rather than what the caller asked for. Identical to the MCP's
    ``_resolved_evaluation``.
    """
    from ml_engine.contracts.config import ExperimentConfig
    from ml_engine.io import ExperimentLayout, read_model_if_exists

    layout = ExperimentLayout(base=artifact_prefix)
    config = read_model_if_exists(store, layout.experiment_config, ExperimentConfig)
    if config is None:
        return {
            "evaluation_mode": "single_split",
            "cv_folds": None,
            "cv_scheme": None,
            "test_fraction": None,
        }
    split = config.split
    cv = split.cv
    return {
        "evaluation_mode": split.evaluation_mode.value,
        "cv_folds": cv.folds if cv is not None else None,
        "cv_scheme": cv.scheme.value if cv is not None else None,
        "test_fraction": split.test_fraction,
    }


def grid_point_count(store, artifact_prefix: str) -> int:
    """How many enabled specs in the frozen config are grid points (name has the separator).

    0 means no grid was used (today's behaviour). Lets a caller show that a grid expanded into N
    points without threading a GridPlan through the orchestrator.
    """
    from ml_engine.contracts.config import ExperimentConfig
    from ml_engine.io import ExperimentLayout, read_model_if_exists
    from ml_engine.models import GRID_POINT_SEPARATOR

    layout = ExperimentLayout(base=artifact_prefix)
    config = read_model_if_exists(store, layout.experiment_config, ExperimentConfig)
    if config is None:
        return 0
    return sum(1 for spec in config.enabled_models_specs() if GRID_POINT_SEPARATOR in spec.name)


def resolved_treatment(store, artifact_prefix: str) -> dict[str, Any] | None:
    """The outlier treatment frozen into the ExperimentConfig, or None when untreated.

    Lets a caller echo what outlier handling was approved without re-deriving it. None means no
    treatment (today's behaviour). Identical shape to the MCP's treatment echo.
    """
    from ml_engine.contracts.config import ExperimentConfig
    from ml_engine.io import ExperimentLayout, read_model_if_exists

    layout = ExperimentLayout(base=artifact_prefix)
    config = read_model_if_exists(store, layout.experiment_config, ExperimentConfig)
    if config is None or config.treatment is None:
        return None
    treatment = config.treatment
    return {
        "outlier_detector": treatment.outlier_detector.value,
        "outlier_handling": treatment.outlier_handling.value,
    }


def resolved_ensembles(store, artifact_prefix: str) -> list[dict[str, Any]]:
    """The ensembles frozen into the ExperimentConfig (type, base learners, rule/meta).

    Empty when no ensemble was defined. Lets a caller echo what was approved without re-deriving
    it. Identical shape to the MCP's ensemble echo.
    """
    from ml_engine.contracts.config import ExperimentConfig
    from ml_engine.io import ExperimentLayout, read_model_if_exists

    layout = ExperimentLayout(base=artifact_prefix)
    config = read_model_if_exists(store, layout.experiment_config, ExperimentConfig)
    if config is None or not config.ensembles:
        return []
    return [
        {
            "type": spec.type.value,
            "ensemble_id": spec.ensemble_id,
            "base_learners": [b.name for b in spec.base_learners],
            "voting_rule": spec.voting_rule.value if spec.type.value == "voting" else None,
            "meta_learner": spec.meta_learner.name if spec.meta_learner is not None else None,
        }
        for spec in config.ensembles
    ]


def holdout_summary(status) -> dict[str, Any]:
    """Bounded per-model headline numbers and overlap for a holdout/validation status.

    Identical to the MCP's ``_holdout_summary`` so the UI and the chat agent see one evaluation
    the same way.
    """
    summary: dict[str, Any] = {
        "evaluation_id": status.evaluation_id,
        "status": status.status,
        "dataset_uri": status.dataset_uri,
        "requested_at": status.requested_at.isoformat(),
        "failure_message": status.failure_message,
        "other_evaluation_ids": status.other_evaluation_ids,
    }
    report = status.report
    if report is None:
        return summary
    summary.update(
        {
            "row_count": report.row_count,
            "rows_dropped_missing_target": report.rows_dropped_missing_target,
            "target_rate": report.target_rate,
            "training_target_rate": report.training_target_rate,
            "overlap": report.overlap.model_dump(mode="json", exclude={"compared_columns"}),
            "image_uri": report.image_uri,
            "training_image_uri": report.training_image_uri,
            "environment_matches_training": (
                None
                if any(w.rule == "training_environment_unknown" for w in report.warnings)
                else not report.environment_differences
            ),
            "environment_differences": dict(list(report.environment_differences.items())[:30]),
            "models": [
                {
                    "model_name": m.model_name,
                    "status": m.status,
                    "primary_metric": m.primary_metric,
                    "validation_score": m.validation_score,
                    "unseen_rows_score": m.unseen_rows_score,
                    "all_rows_score": m.all_rows_score,
                    "metrics_unseen_rows": (
                        m.metrics_unseen_rows.values if m.metrics_unseen_rows else None
                    ),
                    "metrics_all_rows": m.metrics_all_rows.values if m.metrics_all_rows else None,
                    "at_validation_best_threshold": (
                        m.at_validation_best_threshold.model_dump()
                        if m.at_validation_best_threshold
                        else None
                    ),
                    "threshold_rows": m.threshold_rows,
                    "failure_message": m.failure_message,
                }
                for m in report.models
            ],
            "warnings": [
                {"rule": w.rule, "severity": w.severity.value, "message": w.message}
                for w in report.warnings
            ],
        }
    )
    return summary
