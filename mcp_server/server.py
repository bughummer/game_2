"""The FastMCP server. Tools wrap ``backend.services``; nothing here computes or trains.

Boundaries enforced here (requirement 9):
* read-only tools carry ``readOnlyHint``; everything else is a mutation.
* a mutation resolves a ``Principal`` and checks experiment ownership before acting.
* long-running work (``start_training``) returns an execution id at once (Property 9).
* outputs are bounded — summaries, previews and artifact references, never a full dataset
  (requirement 9.5). ``preview_dataset`` caps rows and columns.
* no tool runs an arbitrary AWS CLI/SDK call (requirement 9.6).

AppError is translated to an MCP ``ToolError`` carrying the stable code, so a chat agent gets a
structured, safe failure instead of a stack trace.
"""

from collections.abc import Callable
from functools import wraps
from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations

from backend.aws.images import resolve_stage_image
from backend.container import AppContainer
from backend.errors import AppError
from backend.schemas import ApproveProposalsRequest, ProposeFeaturesRequest, TrainingRequest
from ml_engine.contracts.common import ClassWeighting, RequestedProblemType
from ml_engine.contracts.environment import RESULT_CRITICAL_PACKAGES
from ml_engine.contracts.leakage_investigation import InvestigationPhase, UserDecision
from ml_engine.contracts.proposals import FeatureProposal
from ml_engine.contracts.report import ReportKind

PREVIEW_ROW_CAP = 20
PREVIEW_COLUMN_CAP = 100
REQUIREMENTS_LINE_CAP = 400


def environment_uri(experiments, experiment_id: str) -> str:
    """Where the full requirements.txt lives, for download outside the chat."""
    from ml_engine.io import ExperimentLayout

    return ExperimentLayout(base=experiments.get(experiment_id).artifact_prefix).requirements


def _grid_point_count(store, artifact_prefix: str) -> int:
    """How many enabled specs in the frozen config are grid points (grid-search)."""
    from ml_engine.contracts.config import ExperimentConfig
    from ml_engine.io import ExperimentLayout, read_model_if_exists
    from ml_engine.models import GRID_POINT_SEPARATOR

    layout = ExperimentLayout(base=artifact_prefix)
    config = read_model_if_exists(store, layout.experiment_config, ExperimentConfig)
    if config is None:
        return 0
    return sum(1 for spec in config.enabled_models_specs() if GRID_POINT_SEPARATOR in spec.name)


def _resolved_treatment(store, artifact_prefix: str) -> dict[str, Any] | None:
    """The outlier treatment frozen into the ExperimentConfig, or None when untreated."""
    from ml_engine.contracts.config import ExperimentConfig
    from ml_engine.io import ExperimentLayout, read_model_if_exists

    layout = ExperimentLayout(base=artifact_prefix)
    config = read_model_if_exists(store, layout.experiment_config, ExperimentConfig)
    if config is None or config.treatment is None:
        return None
    return {
        "outlier_detector": config.treatment.outlier_detector.value,
        "outlier_handling": config.treatment.outlier_handling.value,
    }


def _resolved_ensembles(store, artifact_prefix: str) -> list[dict[str, Any]]:
    """The ensembles frozen into the ExperimentConfig (type, base learners, rule/meta)."""
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


def _resolved_evaluation(store, artifact_prefix: str) -> dict[str, Any]:
    """Read the frozen ExperimentConfig and report the resolved evaluation decision.

    The Experiment_Service resolves defaults (mode, K, scheme) into ``config.split`` before any
    job runs, so ``start_training`` echoes what was actually frozen rather than what the caller
    asked for (Req 1.7). Single_Split_Mode leaves cv/scheme/test_fraction unset.
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


def _safe(fn: Callable) -> Callable:
    """Translate an expected AppError into a structured ToolError; let nothing else leak."""

    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except AppError as error:
            raise ToolError(f"[{error.code}] {error.message}") from error

    return wrapper


def build_mcp_server(container: AppContainer, *, name: str = "ml-factory") -> FastMCP:
    mcp = FastMCP(name)
    experiments = container.experiments
    datasets = container.datasets
    analysis = container.analysis
    authorizer = container.authorizer
    audit = container.audit
    read_only = ToolAnnotations(readOnlyHint=True)

    def principal(user: str | None):
        """Resolve who is calling. A raw ``user`` is only a claim; the authorizer decides."""
        return authorizer.principal(user)

    def record(experiment_id: str, action: str, actor: str | None, **detail: str) -> None:
        """Audit one mutation on an existing experiment (requirement 15.4)."""
        try:
            prefix = experiments.get(experiment_id).artifact_prefix
        except AppError:
            return
        audit.record(
            artifact_prefix=prefix,
            experiment_id=experiment_id,
            action=action,
            actor=actor,
            detail={k: str(v) for k, v in detail.items()},
        )

    # --- data ---------------------------------------------------------------------------------
    @mcp.tool(
        description="Create a pre-signed S3 URL to upload a CSV/TSV/Parquet dataset to. Mutation.",
    )
    @_safe
    def request_dataset_upload(filename: str, user: str | None = None) -> dict[str, Any]:
        principal(user)
        return datasets.create_upload_ticket(filename).model_dump(mode="json")

    @mcp.tool(
        description="Register an uploaded dataset and return its columns. Mutation.",
    )
    @_safe
    def register_dataset(dataset_uri: str, user: str | None = None) -> dict[str, Any]:
        principal(user)
        return datasets.register(dataset_uri).model_dump(mode="json")

    @mcp.tool(
        annotations=read_only,
        description=(
            "List datasets already uploaded to this platform, newest first, so one can be "
            "reused in a new experiment without re-uploading. Read-only."
        ),
    )
    @_safe
    def list_datasets(limit: int = 100) -> dict[str, Any]:
        return datasets.list_uploads(limit=limit).model_dump(mode="json")

    @mcp.tool(
        annotations=read_only,
        description="Preview the first rows of an experiment's dataset. Read-only, bounded.",
    )
    @_safe
    def preview_dataset(experiment_id: str, rows: int = 10) -> dict[str, Any]:
        record = experiments.get(experiment_id)
        registered = datasets.register(record.dataset.uri)
        capped = max(1, min(rows, PREVIEW_ROW_CAP))
        columns = registered.columns[:PREVIEW_COLUMN_CAP]
        return {
            "dataset_uri": record.dataset.uri,
            "file_format": registered.file_format,
            "size_bytes": registered.size_bytes,
            "columns": [c.model_dump(mode="json") for c in columns],
            "columns_truncated": len(registered.columns) > PREVIEW_COLUMN_CAP,
            "preview_row_cap": capped,
        }

    # --- experiments --------------------------------------------------------------------------
    @mcp.tool(
        description="Create an experiment from a dataset and target, and start EDA. Mutation."
    )
    @_safe
    def create_experiment(
        name: str,
        dataset_uri: str,
        target_column: str,
        problem_type: str = "auto",
        file_format: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        experiment = experiments.create(
            name=name,
            dataset_uri=dataset_uri,
            target_column=target_column,
            problem_type=RequestedProblemType(problem_type),
            file_format=file_format,
            user=who.user,
        )
        audit.record(
            artifact_prefix=experiment.artifact_prefix,
            experiment_id=experiment.experiment_id,
            action="create_experiment",
            actor=who.user,
            detail={"dataset_uri": dataset_uri, "target_column": target_column},
        )
        return _experiment_summary(experiment)

    @mcp.tool(annotations=read_only, description="Get one experiment's status. Read-only.")
    @_safe
    def get_experiment(experiment_id: str) -> dict[str, Any]:
        return _experiment_summary(experiments.get(experiment_id))

    @mcp.tool(annotations=read_only, description="List experiments, newest first. Read-only.")
    @_safe
    def list_experiments(
        limit: int = 50, mine_only: bool = False, user: str | None = None
    ) -> dict[str, Any]:
        created_by = principal(user).user if mine_only else None
        records = experiments.list(limit=limit, created_by=created_by)
        return {"count": len(records), "experiments": [_experiment_summary(r) for r in records]}

    @mcp.tool(description="Soft-delete an experiment (artifacts are retained). Mutation.")
    @_safe
    def delete_experiment(experiment_id: str, user: str | None = None) -> dict[str, str]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        record(experiment_id, "delete_experiment", who.user)
        stopped = experiments.delete(experiment_id)
        return {"experiment_id": experiment_id, "status": "deleted", "stopped_executions": stopped}

    # --- analysis -----------------------------------------------------------------------------
    @mcp.tool(annotations=read_only, description="Leakage findings for an experiment. Read-only.")
    @_safe
    def get_leakage_findings(experiment_id: str) -> dict[str, Any]:
        report = experiments.leakage(experiment_id)
        return {
            "experiment_id": experiment_id,
            "target_column": report.target_column,
            "findings": [f.model_dump(mode="json") for f in report.findings],
            "feature_risks": [r.model_dump(mode="json") for r in report.feature_risks],
        }

    @mcp.tool(
        annotations=read_only, description="Feature review table with recommendations. Read-only."
    )
    @_safe
    def get_feature_review(experiment_id: str) -> dict[str, Any]:
        return experiments.feature_review(experiment_id).model_dump(mode="json")

    # --- derived-feature proposals (closed-vocabulary, LLM-suggested, human-approved) ---------
    proposals = container.proposals

    @mcp.tool(
        description=(
            "Propose derived features (ratio|difference|date_difference|map_categories|"
            "is_missing) for an experiment. The model suggests typed specifications — never "
            "code — which the backend validates (refusing any that read the target or name a "
            "missing column). Needs Bedrock enabled and EDA complete. Mutation."
        ),
    )
    @_safe
    def propose_features_tool(
        experiment_id: str,
        target_definition: str | None = None,
        prediction_timing: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        report = proposals.generate(
            experiment_id,
            ProposeFeaturesRequest(
                target_definition=target_definition, prediction_timing=prediction_timing
            ),
        )
        record(experiment_id, "propose_features", who.user, candidates=len(report.candidates))
        return report.model_dump(mode="json")

    @mcp.tool(
        description=(
            "Define derived features exactly as the user describes them. Each feature is one "
            "operation: ratio (numerator/denominator), difference (left-right), "
            "date_difference (start,end,unit days|months|years), map_categories (column, "
            "mapping {value: group}, default), or is_missing (column). Use only column names "
            "from the dataset; never the target. Ask the user whether every input is known at "
            "prediction time (available_at_prediction_time) and why the feature should help "
            "(rationale) — do not guess. Before calling, restate the exact specification to the "
            "user. Works without Bedrock. Accepted features are approved by default (they then "
            "go into training unless excluded with decide_features); refused ones come back "
            "with the reason. Mutation."
        ),
    )
    @_safe
    def define_derived_features(
        experiment_id: str,
        features: list[FeatureProposal],
        approve: bool = True,
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        result = proposals.define(experiment_id, features, approve=approve, user=who.user)
        record(
            experiment_id,
            "define_derived_features",
            who.user,
            added=",".join(result.added),
            refused=len(result.rejected),
        )
        return {
            "experiment_id": experiment_id,
            "added": result.added,
            "approved_names": result.report.approved_names,
            "rejected": [r.model_dump(mode="json") for r in result.rejected],
            "profiling_execution_id": result.profiling_execution_id,
        }

    @mcp.tool(
        annotations=read_only,
        description=(
            "Get the derived-feature proposals (candidates, refusals, approvals). Read-only."
        ),
    )
    @_safe
    def get_feature_proposals(experiment_id: str) -> dict[str, Any] | None:
        report = proposals.stored(experiment_id)
        return report.model_dump(mode="json") if report else None

    @mcp.tool(
        description=(
            "Approve which proposed derived features to build. An empty list means 'none'. "
            "Returns at once; a SageMaker job then profiles and leakage-screens them, and their "
            "stats appear in get_feature_review when it finishes. Approved features are built "
            "and used at training unless excluded in decide_features. Mutation."
        ),
    )
    @_safe
    def approve_features(
        experiment_id: str, approved_names: list[str], user: str | None = None
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        result = proposals.approve(
            experiment_id,
            ApproveProposalsRequest(approved_names=approved_names),
            user=who.user,
        )
        report = result.report
        record(experiment_id, "approve_features", who.user, approved=len(report.approved_names))
        return {
            "experiment_id": experiment_id,
            "approved_names": report.approved_names,
            "candidate_count": len(report.candidates),
            # Profiling runs on SageMaker; the stats appear in feature review when it finishes.
            "profiling_execution_id": result.profiling_execution_id,
        }

    @mcp.tool(description="Save the feature keep/exclude decision. Mutation.")
    @_safe
    def decide_features(
        experiment_id: str,
        selected_features: list[str],
        exclusion_reasons: dict[str, str] | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        selection = experiments.save_features(
            experiment_id,
            selected_features=selected_features,
            exclusion_reasons=exclusion_reasons,
            user=who.user,
        )
        record(
            experiment_id, "decide_features", who.user, selected=len(selection.selected_features)
        )
        return {
            "experiment_id": experiment_id,
            "selected_count": len(selection.selected_features),
            "excluded_count": len(selection.excluded_features),
        }

    # --- training & results -------------------------------------------------------------------
    @mcp.tool(
        description=(
            "Start training. Returns immediately with an execution id. Call preflight_check "
            "first and show the user its warnings. split_strategy: 'group' keeps every row of "
            "one entity (group_column, defaulting to the business context's entity id) in one "
            "fold; 'temporal' validates on the latest rows of time_column; otherwise random/"
            "stratified. Retraining archives the previous run (see list_runs). "
            "class_weighting: 'auto' (default) balances classes in every model that supports "
            "it (class_weight='balanced', XGBoost scale_pos_weight=negatives/positives, CatBoost "
            "auto_class_weights); 'none' trains unweighted. model_parameters overrides "
            "hyperparameters per model, e.g. {'xgboost': {'scale_pos_weight': 5}}; an override "
            "replaces the automatic weight for that parameter. Only set parameters the user "
            "asked for. evaluation_mode='cv' runs K-fold cross-validation (cv_folds, default K "
            "from settings) instead of a single split: every model is fitted on each fold, the "
            "headline score becomes the mean across folds and the refit model trains on all "
            "development rows. test_fold=True (or test_fraction in (0, 0.5)) carves out an "
            "untouched test fold scored once at the end, reported separately and never used for "
            "selection. The response echoes the resolved evaluation_mode, cv_folds and cv_scheme "
            "read from the frozen configuration. Mutation."
        )
    )
    @_safe
    def start_training(
        experiment_id: str,
        models: list[str] | None = None,
        primary_metric: str | None = None,
        problem_type: str | None = None,
        class_weighting: Literal["auto", "none"] = "auto",
        model_parameters: dict[str, dict[str, float | int | str | bool | None]] | None = None,
        split_strategy: Literal["random", "stratified_random", "group", "temporal"] | None = None,
        group_column: str | None = None,
        time_column: str | None = None,
        evaluation_mode: Literal["single_split", "cv"] | None = None,
        cv_folds: int | None = None,
        test_fold: bool = False,
        test_fraction: float | None = None,
        parameter_grid: dict[str, dict[str, list[float | int | str | bool]]] | None = None,
        outlier_detector: Literal["none", "iqr", "zscore", "isolation_forest"] | None = None,
        outlier_handling: Literal["flag", "cap", "drop"] | None = None,
        ensembles: list[dict[str, Any]] | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        preflight = analysis.preflight(experiment_id)
        updated = experiments.start_training(
            experiment_id,
            TrainingRequest(
                models=models or [],
                primary_metric=primary_metric,
                problem_type=problem_type,
                class_weighting=ClassWeighting(class_weighting),
                model_parameters=model_parameters or {},
                split_strategy=split_strategy,
                group_column=group_column,
                time_column=time_column,
                evaluation_mode=evaluation_mode,
                cv_folds=cv_folds,
                test_fold=test_fold,
                test_fraction=test_fraction,
                parameter_grid=parameter_grid,
                outlier_detector=outlier_detector,
                outlier_handling=outlier_handling,
                ensembles=ensembles or [],
            ),
            user=who.user,
        )
        # The resolved CV decision lives in the frozen config, not the request: read it back so
        # the chat agent sees what was actually frozen (default mode, default K, resolved scheme).
        resolved = _resolved_evaluation(container.store, updated.artifact_prefix)
        audit.record(
            artifact_prefix=updated.artifact_prefix,
            experiment_id=experiment_id,
            action="start_training",
            actor=who.user,
            detail={
                "models": ",".join(updated.requested_models),
                "evaluation_mode": resolved["evaluation_mode"],
                "cv_folds": str(resolved["cv_folds"]),
                "cv_scheme": str(resolved["cv_scheme"]),
                "test_fraction": str(resolved["test_fraction"]),
            },
        )
        return {
            "experiment_id": experiment_id,
            "status": updated.status.value,
            "execution_id": updated.training_execution_arn,
            "requested_models": updated.requested_models,
            "class_weighting": class_weighting,
            "model_parameters": model_parameters or {},
            # Echo the resolved CV decision read from the frozen ExperimentConfig (Req 1.7).
            "evaluation_mode": resolved["evaluation_mode"],
            "cv_folds": resolved["cv_folds"],
            "cv_scheme": resolved["cv_scheme"],
            "test_fraction": resolved["test_fraction"],
            # Grid search: how many grid points the request expanded into (0 = no grid).
            "grid_points": _grid_point_count(container.store, updated.artifact_prefix),
            # Training-data-treatment: the frozen outlier treatment (None = untreated).
            "treatment": _resolved_treatment(container.store, updated.artifact_prefix),
            # Ensembles: the frozen voting/stacking ensembles ([] = none).
            "ensembles": _resolved_ensembles(container.store, updated.artifact_prefix),
            # Advisory: what the pre-flight check found before this run started.
            "preflight_warnings": preflight.warnings,
            "unresolved_leakage_warnings": preflight.unresolved_leakage_warnings,
        }

    @mcp.tool(annotations=read_only, description="Per-model training status. Read-only.")
    @_safe
    def get_training_status(experiment_id: str) -> dict[str, Any]:
        return experiments.training_status(experiment_id).model_dump(mode="json")

    @mcp.tool(
        annotations=read_only,
        description=(
            "Precision/recall/F1 across decision thresholds for each trained binary classifier, "
            "from the validation fold: metrics at the default 0.5, at the F1-maximising "
            "threshold, and on a 0.05-step grid (use the grid to answer 'what threshold gives "
            "recall >= X'). Read numbers from here; do not compute them. The best threshold is "
            "chosen on the same fold it is scored on, so it is optimistic. Read-only."
        ),
    )
    @_safe
    def get_threshold_analysis(experiment_id: str, model_name: str | None = None) -> dict[str, Any]:
        analyses = experiments.thresholds(experiment_id, model_name)
        return {
            "experiment_id": experiment_id,
            "models": [a.model_dump(mode="json") for a in analyses],
        }

    @mcp.tool(
        description=(
            "Evaluate the trained models on a separate dataset (a test set, a later period, or "
            "the full population) without retraining. dataset_uri is an s3:// URI from the "
            "upload flow (request_dataset_upload + register_dataset) or in an approved "
            "location. The file needs the target column and the columns the models used. "
            "Returns at once with an evaluation_id; poll get_holdout_evaluation. Mutation."
        ),
    )
    @_safe
    def start_holdout_evaluation(
        experiment_id: str,
        dataset_uri: str,
        file_format: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        status = experiments.start_holdout_evaluation(
            experiment_id, dataset_uri=dataset_uri, file_format=file_format, user=who.user
        )
        record(
            experiment_id,
            "start_holdout_evaluation",
            who.user,
            evaluation_id=status.evaluation_id,
            dataset_uri=dataset_uri,
        )
        return _holdout_summary(status)

    @mcp.tool(
        annotations=read_only,
        description=(
            "Result of an evaluation on a separate dataset (latest unless evaluation_id is "
            "given): status, per-model metrics on all rows and on rows NOT seen in training, "
            "how many rows overlap with the training data, and how the validation-chosen "
            "threshold holds up. When rows overlap, report the unseen-rows numbers; the "
            "all-rows numbers are inflated. Read-only."
        ),
    )
    @_safe
    def get_holdout_evaluation(
        experiment_id: str, evaluation_id: str | None = None
    ) -> dict[str, Any]:
        return _holdout_summary(experiments.holdout_evaluation(experiment_id, evaluation_id))

    @mcp.tool(
        annotations=read_only,
        description=(
            "The environment the experiment's models were trained in, captured inside the "
            "training job: stage image digest, Python version, the versions of the libraries "
            "that affect results, and a requirements.txt pinning every installed package. "
            "Use it to reproduce or load the models elsewhere. Read-only."
        ),
    )
    @_safe
    def get_environment(experiment_id: str) -> dict[str, Any]:
        environment, requirements = experiments.environment(experiment_id)
        lines = requirements.splitlines()
        return {
            "experiment_id": experiment_id,
            "image_uri": environment.image_uri,
            "python_version": environment.python_version,
            "platform": environment.platform,
            "captured_at": environment.captured_at.isoformat(),
            "key_packages": {
                name: environment.packages[name]
                for name in RESULT_CRITICAL_PACKAGES
                if name in environment.packages
            },
            "package_count": len(environment.packages),
            "requirements_txt": "\n".join(lines[:REQUIREMENTS_LINE_CAP]),
            "requirements_truncated": len(lines) > REQUIREMENTS_LINE_CAP,
            "requirements_uri": f"{environment_uri(experiments, experiment_id)}",
        }

    @mcp.tool(
        annotations=read_only,
        description=(
            "Model comparison for a finished experiment: ranking, per-model metrics, feature "
            "counts, durations and failures. In cross-validation mode each model also carries "
            "'cv' — the mean ± standard deviation and min/max of every metric across the K "
            "folds, the per-fold primary-metric values and an instability flag — and the "
            "ranking is by mean CV score. When a test fold exists, each model carries separate "
            "'test_metrics' and the report's 'test_evaluation' section holds them; these are "
            "scored once and never affect the ranking. Read-only."
        ),
    )
    @_safe
    def get_comparison(experiment_id: str) -> dict[str, Any]:
        # model_dump(mode="json") returns the full report, including the CV and Test_Fold keys;
        # nothing here truncates per-fold values, instability flags or the test metrics.
        return experiments.comparison(experiment_id).model_dump(mode="json")

    # --- business context & leakage investigation (priority capability) -----------------------
    leakage = container.leakage

    @mcp.tool(
        description=(
            "Set the experiment's business context in plain language (objective + prediction "
            "scenario). Replaces per-column timing questions; leakage investigation defaults on. "
            "Mutation."
        ),
    )
    @_safe
    def set_business_context(
        experiment_id: str,
        objective: str,
        prediction_scenario: str,
        scoring_trigger: str | None = None,
        prediction_horizon: str | None = None,
        entity_id_column: str | None = None,
        temporal_columns: list[str] | None = None,
        investigate_leakage: bool = True,
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        context = leakage.set_business_context(
            experiment_id,
            objective=objective,
            prediction_scenario=prediction_scenario,
            scoring_trigger=scoring_trigger,
            prediction_horizon=prediction_horizon,
            entity_id_column=entity_id_column,
            temporal_columns=temporal_columns or [],
            investigate_leakage=investigate_leakage,
            user=who.user,
        )
        record(experiment_id, "set_business_context", who.user)
        return context.model_dump(mode="json")

    @mcp.tool(
        annotations=read_only, description="Get the experiment's business context. Read-only."
    )
    @_safe
    def get_business_context(experiment_id: str) -> dict[str, Any] | None:
        context = leakage.get_business_context(experiment_id)
        return context.model_dump(mode="json") if context else None

    @mcp.tool(
        description=(
            "Build (or rebuild) the pre-training leakage investigation: deterministic rules, "
            "target association and name patterns, plus — when Bedrock is enabled and the "
            "business context allows it — LLM leak reasoning over the governed context, attached "
            "to each warning as suspicion (never proof). 'sources' says whether the LLM was used. "
            "Earlier decisions are kept. Mutation."
        )
    )
    @_safe
    def investigate_leakage(
        experiment_id: str, use_llm: bool = True, user: str | None = None
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        report = leakage.build_pre_training_phase(experiment_id, use_llm=use_llm)
        record(experiment_id, "investigate_leakage", who.user, warnings=len(report.warnings))
        return report.model_dump(mode="json")

    @mcp.tool(
        description=(
            "Build (or rebuild) the post-training leakage investigation from the trained models: "
            "importance dominance, permutation importance and, if run, ablation results. Kept "
            "separate from the pre-training phase. Mutation."
        )
    )
    @_safe
    def investigate_leakage_post_training(
        experiment_id: str, user: str | None = None
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        report = leakage.build_post_training_phase(experiment_id)
        record(
            experiment_id,
            "investigate_leakage_post_training",
            who.user,
            warnings=len(report.warnings),
        )
        return report.model_dump(mode="json")

    @mcp.tool(
        annotations=read_only,
        description="Get a leakage investigation phase (pre_training|post_training). Read-only.",
    )
    @_safe
    def get_leakage_investigation(
        experiment_id: str, phase: str = "pre_training"
    ) -> dict[str, Any] | None:
        report = leakage.get_phase(experiment_id, InvestigationPhase(phase))
        return report.model_dump(mode="json") if report else None

    @mcp.tool(description="Record your decision on one leakage warning. Mutation.")
    @_safe
    def decide_leakage_warning(
        experiment_id: str,
        warning_id: str,
        decision: str,
        phase: str = "pre_training",
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        report = leakage.decide_warning(
            experiment_id,
            phase=InvestigationPhase(phase),
            warning_id=warning_id,
            decision=UserDecision(decision),
            user=who.user,
        )
        warning = report.warning_for(warning_id)
        record(
            experiment_id,
            "decide_leakage_warning",
            who.user,
            warning_id=warning_id,
            decision=decision,
        )
        return {
            "experiment_id": experiment_id,
            "warning_id": warning_id,
            "user_decision": warning.user_decision.value,
            "review_status": warning.review_status.value,
        }

    @mcp.tool(
        description=(
            "Apply your 'exclude' leakage decisions to the feature selection. Explicit step: "
            "approval and selection stay distinct. Mutation."
        ),
    )
    @_safe
    def apply_leakage_exclusions(experiment_id: str, user: str | None = None) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        selection = leakage.apply_exclusions_to_selection(experiment_id, user=who.user)
        record(experiment_id, "apply_leakage_exclusions", who.user)
        return {
            "experiment_id": experiment_id,
            "selected_count": len(selection.selected_features),
            "excluded_count": len(selection.excluded_features),
        }

    # --- reporting ----------------------------------------------------------------------------
    reporting = container.reporting

    @mcp.tool(
        description=(
            "Generate a downloadable report (eda|model_comparison) as JSON + HTML, and PDF when "
            "available. Returns artifact references, not the report body. Mutation."
        ),
    )
    @_safe
    def generate_report(
        experiment_id: str, kind: str = "model_comparison", user: str | None = None
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        if kind == "eda":
            artifacts = reporting.build_eda_report(experiment_id)
        else:
            artifacts = reporting.build_comparison_report(experiment_id)
        record(experiment_id, "generate_report", who.user, kind=kind)
        return {
            "experiment_id": experiment_id,
            "kind": artifacts.kind.value,
            "json_uri": artifacts.json_uri,
            "html_uri": artifacts.html_uri,
            "pdf_uri": artifacts.pdf_uri,
            "pdf_available": artifacts.pdf_available,
        }

    @mcp.tool(
        annotations=read_only,
        description="Get a generated report document (eda|model_comparison) as JSON. Read-only.",
    )
    @_safe
    def get_report(experiment_id: str, kind: str = "model_comparison") -> dict[str, Any] | None:
        report = reporting.get_report(experiment_id, ReportKind(kind))
        return report.model_dump(mode="json") if report else None

    _register_operations(mcp, container, principal, record, read_only)
    return mcp


def _register_operations(mcp, container: AppContainer, principal, record, read_only) -> None:
    """Run control, diagnostics, interpretation, ablation, scoring, runs and comparison."""
    experiments = container.experiments
    analysis = container.analysis
    authorizer = container.authorizer

    @mcp.tool(
        annotations=read_only,
        description=(
            "Check an experiment before training: blockers (training would fail), warnings "
            "(unresolved high-severity leakage warnings on selected features, EXCLUDE decisions "
            "not applied, missing business context, unprofiled derived features) and a suggested "
            "split. Show the result to the user before start_training. Read-only."
        ),
    )
    @_safe
    def preflight_check(experiment_id: str) -> dict[str, Any]:
        return analysis.preflight(experiment_id).model_dump(mode="json")

    @mcp.tool(description="Stop the experiment's running training workflow. Mutation.")
    @_safe
    def stop_training(experiment_id: str, user: str | None = None) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        execution_id = experiments.stop_training(experiment_id)
        record(experiment_id, "stop_training", who.user, execution_id=execution_id)
        return {"experiment_id": experiment_id, "execution_id": execution_id, "status": "stopping"}

    @mcp.tool(
        annotations=read_only,
        description=(
            "Per-job SageMaker status of an execution (latest training run by default): raw and "
            "normalized status (Waiting for capacity, Training, Interrupted/recovering, Unknown, "
            "…) and, for failed jobs, the deterministic failure diagnosis with a recommended "
            "action. 'Unknown' means AWS evidence is insufficient; do not guess. Read-only."
        ),
    )
    @_safe
    def get_execution_status(experiment_id: str, execution_id: str | None = None) -> dict[str, Any]:
        return analysis.execution_status(experiment_id, execution_id).model_dump(mode="json")

    @mcp.tool(
        description=(
            "Train again from the frozen configuration, unchanged — to check reproducibility. "
            "The previous run is archived first (list_runs). Returns at once. Mutation."
        )
    )
    @_safe
    def rerun_training(experiment_id: str, user: str | None = None) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        updated = experiments.rerun_training(experiment_id, user=who.user)
        record(experiment_id, "rerun_training", who.user)
        return {
            "experiment_id": experiment_id,
            "status": updated.status.value,
            "execution_id": updated.training_execution_arn,
        }

    @mcp.tool(
        annotations=read_only,
        description=(
            "Earlier training runs of this experiment (archived on retrain), newest first: best "
            "model and score, models, features, split. Read-only."
        ),
    )
    @_safe
    def list_runs(experiment_id: str) -> dict[str, Any]:
        runs = experiments.list_runs(experiment_id)
        return {"experiment_id": experiment_id, "runs": [r.model_dump(mode="json") for r in runs]}

    @mcp.tool(
        annotations=read_only,
        description=(
            "Compare 2-5 experiments side by side: dataset, target, metric, best model and score, "
            "split and the features added/removed relative to the first. Includes notes on why "
            "scores may not be comparable. Read numbers from here; do not compute. Read-only."
        ),
    )
    @_safe
    def compare_experiments(experiment_ids: list[str]) -> dict[str, Any]:
        return analysis.compare_experiments(experiment_ids)

    @mcp.tool(
        description=(
            "LLM interpretation of a finished experiment (or of its EDA, before training): "
            "summary, suspected leaks, feature interpretations, data-quality concerns and "
            "proposed follow-up experiments. Grounded only in deterministic artifacts; the "
            "context passes the data-governance gate. Needs Bedrock. Mutation (stores the "
            "report)."
        )
    )
    @_safe
    def interpret_results(
        experiment_id: str, question: str | None = None, user: str | None = None
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        report = container.reasoning.interpret(experiment_id, question=question)
        record(experiment_id, "interpret_results", who.user)
        return report.model_dump(mode="json")

    @mcp.tool(
        annotations=read_only,
        description=(
            "Permutation importance on the validation fold, per input column: the drop in the "
            "primary metric when that column is shuffled. Shows what each model relies on; a "
            "large share is a leak fingerprint, not proof. Read-only."
        ),
    )
    @_safe
    def get_permutation_importance(
        experiment_id: str, model_name: str | None = None
    ) -> dict[str, Any]:
        results = analysis.permutation_importance(experiment_id, model_name)
        return {
            "experiment_id": experiment_id,
            "models": [
                {
                    **r.model_dump(mode="json", exclude={"entries"}),
                    "entries": [e.model_dump(mode="json") for e in r.entries[:25]],
                }
                for r in results
            ],
        }

    @mcp.tool(
        description=(
            "Ablation: retrain one model (best by default) without each candidate feature on "
            "the identical folds and report the metric drop. Candidates default to the post-"
            "training warnings, the top permutation features and high-severity pre-training "
            "warnings (max 10; each is a retrain). Runs on SageMaker; returns at once; poll "
            "get_ablation, then investigate_leakage_post_training. Mutation."
        )
    )
    @_safe
    def start_ablation(
        experiment_id: str,
        features: list[str] | None = None,
        model_name: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        started = analysis.start_ablation(experiment_id, features=features, model_name=model_name)
        record(experiment_id, "start_ablation", who.user, features=",".join(started["features"]))
        return started

    @mcp.tool(annotations=read_only, description="Result of the latest ablation. Read-only.")
    @_safe
    def get_ablation(experiment_id: str) -> dict[str, Any]:
        return analysis.ablation(experiment_id)

    @mcp.tool(
        description=(
            "Batch-score a new file with a trained model (best by default). No endpoint is "
            "created: a SageMaker job writes predictions.csv (passthrough columns, prediction, "
            "class probabilities). The file needs the columns the model used; the target is not "
            "needed. passthrough_columns default to the business context's entity id. Returns at "
            "once; poll get_batch_scoring. Mutation."
        )
    )
    @_safe
    def start_batch_scoring(
        experiment_id: str,
        dataset_uri: str,
        model_name: str | None = None,
        passthrough_columns: list[str] | None = None,
        file_format: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        status = analysis.start_scoring(
            experiment_id,
            dataset_uri=dataset_uri,
            model_name=model_name,
            passthrough_columns=passthrough_columns,
            file_format=file_format,
            user=who.user,
        )
        record(experiment_id, "start_batch_scoring", who.user, scoring_id=status["scoring_id"])
        return status

    @mcp.tool(
        annotations=read_only,
        description=(
            "Status of a batch scoring (latest unless scoring_id is given) with the predictions "
            "URI and a summary of the predictions. Use get_download_link for the file. Read-only."
        ),
    )
    @_safe
    def get_batch_scoring(experiment_id: str, scoring_id: str | None = None) -> dict[str, Any]:
        return analysis.scoring(experiment_id, scoring_id)

    @mcp.tool(
        description=(
            "A short-lived download link for one of this experiment's artifacts (predictions "
            "CSV, report PDF/HTML/JSON, requirements.txt). The URI must come from another tool. "
            "Mutation (issues a credential)."
        )
    )
    @_safe
    def get_download_link(experiment_id: str, uri: str, user: str | None = None) -> dict[str, Any]:
        who = principal(user)
        authorizer.require_owner(who, experiment_id)
        prefix = experiments.get(experiment_id).artifact_prefix
        if not (uri == prefix or uri.startswith(prefix.rstrip("/") + "/")):
            raise ToolError("[VALIDATION_ERROR] That artifact does not belong to this experiment.")
        link = container.datasets.create_download_url(uri)
        record(experiment_id, "get_download_link", who.user, uri=uri)
        return link

    @mcp.tool(
        name="resolve_stage_image",
        annotations=read_only,
        description=(
            "Which stage image a run would use, and where the URI comes from ('config' for an "
            "explicit override, 'aws_lookup' for the resolved AWS prebuilt XGBoost image). "
            "Read-only: resolves the same image start_training would, mutates nothing and starts "
            "nothing."
        ),
    )
    @_safe
    def resolve_stage_image_tool() -> dict[str, str]:
        resolved = resolve_stage_image(container.settings)
        return {"uri": resolved.uri, "source": resolved.source}


def _holdout_summary(status) -> dict[str, Any]:
    """Bounded: per-model headline numbers and the overlap, not confusion-matrix walls."""
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
            # None when the training environment was never recorded (older experiments).
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
                    "metrics_unseen_rows": m.metrics_unseen_rows.values
                    if m.metrics_unseen_rows
                    else None,
                    "metrics_all_rows": m.metrics_all_rows.values if m.metrics_all_rows else None,
                    "at_validation_best_threshold": m.at_validation_best_threshold.model_dump()
                    if m.at_validation_best_threshold
                    else None,
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


def _experiment_summary(record) -> dict[str, Any]:
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
