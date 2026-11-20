"""Experiment use cases: validate, record, start workflows, read artifacts.

The control plane never loads a dataset, fits a transform or trains a model — it submits work to
SageMaker and reads the artifacts the stages write. Ported and trimmed from old_repo: the local
file:// paths and the queue are gone, dataset locations are ``s3://``, and the surface is the one
the MCP exposes. The deterministic helpers (default feature selection, metric resolution, the
model catalogue) are reused from ``ml_engine`` unchanged.
"""

from __future__ import annotations

import logging
import threading
import uuid
from datetime import UTC, datetime

from backend.config import Settings
from backend.errors import ArtifactNotReadyError, ConflictError, NotFoundError, ValidationError
from backend.orchestration.base import ExperimentOrchestrator
from backend.orchestration.guards import resolve_cv_folds
from backend.repositories import ACTIVE_TRAINING_STATUSES, ExperimentRepository
from backend.schemas import (
    FeatureReview,
    FeatureReviewItem,
    HoldoutStatus,
    ModelRunState,
    TrainingRequest,
    TrainingStatus,
)
from ml_engine.contracts.business_context import BusinessContext
from ml_engine.contracts.common import (
    ExperimentStatus,
    LeakageRiskLevel,
    ModelRunStatus,
    ProblemType,
    RequestedProblemType,
)
from ml_engine.contracts.comparison import ComparisonReport, ExperimentSummary
from ml_engine.contracts.config import (
    DatasetReference,
    EnvironmentCapture,
    ExperimentConfig,
    FeatureSelection,
    ModelSpec,
    PreprocessingConfig,
    SplitConfig,
)
from ml_engine.contracts.cv import CrossValidationConfig, EvaluationMode
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.ensemble import (
    BaseLearnerSpec,
    EnsembleSpec,
    EnsembleType,
    MetaLearnerSpec,
    default_meta_learner,
)
from ml_engine.contracts.environment import RuntimeEnvironment
from ml_engine.contracts.execution import ExecutionRecord
from ml_engine.contracts.experiment import ExperimentDefinition, ExperimentRecord
from ml_engine.contracts.grid import GridPlan, GridPoint, ParameterGrid
from ml_engine.contracts.holdout import HoldoutEvaluationReport, HoldoutRequest
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.model import ModelFailure, ModelMetadata
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.contracts.proposals import (
    DerivedProfileReport,
    FeatureProposal,
    FeatureProposalReport,
)
from ml_engine.contracts.requirements import (
    RequirementsError,
    RequirementsSpec,
    requirement_name,
)
from ml_engine.contracts.runs import ArchivedRun
from ml_engine.contracts.thresholds import ThresholdAnalysis
from ml_engine.contracts.treatment import OutlierHandling, TreatmentConfig
from ml_engine.evaluation import resolve_primary_metric
from ml_engine.evaluation.metrics import UnknownMetricError
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    copy_object,
    detect_format,
    is_s3_uri,
    join_uri,
    read_model,
    read_model_if_exists,
    uri_matches_prefix,
    write_model,
)
from ml_engine.io.layout import (
    EXPERIMENT_CONFIG_FILE,
    RUN_OUTPUT_FILES,
    RUN_OUTPUT_PREFIXES,
)
from ml_engine.leakage import default_feature_selection
from ml_engine.models import UnknownModelError, default_model_names, get_plugin, plugins_for
from ml_engine.models.grid import expand_grid
from ml_engine.splitting import available_strategies
from ml_engine.splitting.folds import resolve_cv_scheme
from ml_engine.training.runner import environment_metadata, package_versions

LOGGER = logging.getLogger("ml_factory.services.experiments")

ML_FACTORY_VERSION = "2.0.0"


class ExperimentService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ExperimentRepository,
        store: ObjectStore,
        orchestrator: ExperimentOrchestrator,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store
        self._orchestrator = orchestrator
        # One "start training" at a time per experiment. The status check and the start are
        # separate S3 reads/writes, so without this two concurrent calls (a double click, an
        # agent retry) both pass the check and launch two runs. Holds within one process, which
        # matches the one-server deployment the status poller already requires.
        self._start_locks: dict[str, threading.Lock] = {}
        self._start_locks_guard = threading.Lock()

    def _start_lock(self, experiment_id: str) -> threading.Lock:
        with self._start_locks_guard:
            return self._start_locks.setdefault(experiment_id, threading.Lock())

    # --- lifecycle ----------------------------------------------------------------------------
    def create(
        self,
        *,
        name: str,
        dataset_uri: str,
        target_column: str,
        problem_type: RequestedProblemType = RequestedProblemType.AUTO,
        file_format: str | None = None,
        user: str | None = None,
    ) -> ExperimentRecord:
        """Validate the dataset, record the experiment, start EDA. Returns at once."""
        name = (name or "").strip()
        target_column = (target_column or "").strip()
        if not name:
            raise ValidationError("An experiment name is required.")
        if not target_column:
            raise ValidationError("A target column is required.")
        dataset = self._validate_dataset(dataset_uri, file_format)

        experiment_id = _new_experiment_id()
        layout = ExperimentLayout.for_experiment(self._settings.artifact_root, experiment_id)
        dataset = self._pin_dataset(dataset, layout.input_dir)
        record = self._repository.create(
            ExperimentDefinition(
                experiment_id=experiment_id,
                name=name,
                created_by=user,
                created_at=datetime.now(UTC),
                dataset=dataset,
                target_column=target_column,
                requested_problem_type=problem_type,
                artifact_prefix=layout.base,
            )
        )
        handle = self._orchestrator.start_eda(record)
        LOGGER.info("Started EDA for %s (%s)", experiment_id, handle.execution_id)
        return self._repository.update(experiment_id, eda_execution_arn=handle.execution_id)

    def get(self, experiment_id: str) -> ExperimentRecord:
        return self._repository.get(experiment_id)

    def list(
        self, *, limit: int | None = None, created_by: str | None = None
    ) -> list[ExperimentRecord]:
        return self._repository.list(
            limit=min(
                limit or self._settings.experiment_list_limit, self._settings.experiment_list_limit
            ),
            created_by=created_by,
        )

    def delete(self, experiment_id: str) -> list[str]:
        """Soft delete, after stopping every job still running for it. Returns what was stopped."""
        self._repository.get(experiment_id)
        stopped = self._stop_all(experiment_id)
        self._repository.delete(experiment_id)
        return stopped

    def stop_training(self, experiment_id: str) -> str:
        """Stop the running training workflow. Returns its execution id."""
        record = self._repository.get(experiment_id)
        if record.status not in ACTIVE_TRAINING_STATUSES or not record.training_execution_arn:
            raise ConflictError(
                f"No training is running for this experiment (status {record.status.value})."
            )
        stop = getattr(self._orchestrator, "stop", None)
        if not callable(stop):
            raise ConflictError("This deployment's orchestrator cannot stop executions.")
        stop(record.training_execution_arn)
        return record.training_execution_arn

    def _stop_all(self, experiment_id: str) -> list[str]:
        running = getattr(self._orchestrator, "running_executions", None)
        stop = getattr(self._orchestrator, "stop", None)
        if not callable(running) or not callable(stop):
            return []
        stopped: list[str] = []
        for execution in running(experiment_id):
            try:
                stop(execution.execution_id)
                stopped.append(execution.execution_id)
            except Exception:
                LOGGER.exception("Could not stop %s", execution.execution_id)
        return stopped

    # --- artifacts ----------------------------------------------------------------------------
    def eda(self, experiment_id: str) -> EdaReport:
        return self._artifact(experiment_id, lambda layout: layout.eda, EdaReport, "EDA")

    def leakage(self, experiment_id: str) -> LeakageReport:
        return self._artifact(
            experiment_id, lambda layout: layout.leakage, LeakageReport, "leakage analysis"
        )

    def comparison(self, experiment_id: str) -> ComparisonReport:
        return self._artifact(
            experiment_id, lambda layout: layout.comparison, ComparisonReport, "model comparison"
        )

    # --- evaluation on a separate dataset ------------------------------------------------------
    def start_holdout_evaluation(
        self,
        experiment_id: str,
        *,
        dataset_uri: str,
        file_format: str | None = None,
        user: str | None = None,
    ) -> HoldoutStatus:
        """Score the trained models on another file. Returns at once; a SageMaker job runs it.

        The file must be in an approved location (the upload inbox or an allowed prefix), like
        any dataset. It needs the target column and every column the models were trained on.
        """
        record = self._repository.get(experiment_id)
        if record.status in ACTIVE_TRAINING_STATUSES:
            raise ConflictError("Training is running; evaluate once it has finished.")
        layout = ExperimentLayout(base=record.artifact_prefix)
        if not self._store.exists(layout.comparison):
            raise ConflictError("Train the experiment before evaluating it on another dataset.")
        dataset = self._validate_dataset(dataset_uri, file_format)
        evaluation_id = f"eval-{uuid.uuid4().hex[:12]}"
        dataset = self._pin_dataset(dataset, layout.evaluation_input_dir(evaluation_id))

        request = HoldoutRequest(
            evaluation_id=evaluation_id,
            experiment_id=experiment_id,
            dataset_uri=dataset.uri,
            file_format=dataset.file_format,
            dataset_etag=dataset.etag,
            dataset_version_id=dataset.version_id,
            requested_at=datetime.now(UTC),
            requested_by=user,
        )
        write_model(self._store, layout.evaluation_request(request.evaluation_id), request)
        handle = self._orchestrator.start_holdout_evaluation(record, request.evaluation_id)
        request.execution_id = handle.execution_id
        write_model(self._store, layout.evaluation_request(request.evaluation_id), request)
        return self.holdout_evaluation(experiment_id, request.evaluation_id)

    def holdout_evaluation(
        self, experiment_id: str, evaluation_id: str | None = None
    ) -> HoldoutStatus:
        """One evaluation (the latest when no id is given) and where it stands."""
        record = self.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        requests = sorted(
            (
                read_model(self._store, uri, HoldoutRequest)
                for uri in self._store.list_uris(layout.evaluations_prefix)
                if uri.endswith("/request.json")
            ),
            key=lambda r: r.requested_at,
            reverse=True,
        )
        if not requests:
            raise ArtifactNotReadyError(
                "This experiment has not been evaluated on another dataset."
            )
        if evaluation_id is None:
            request = requests[0]
        else:
            request = next((r for r in requests if r.evaluation_id == evaluation_id), None)
            if request is None:
                raise NotFoundError(f"No evaluation '{evaluation_id}' for this experiment.")

        status, report, failure = "running", None, None
        report = read_model_if_exists(
            self._store, layout.evaluation_report(request.evaluation_id), HoldoutEvaluationReport
        )
        if report is not None:
            status = "completed"
        else:
            job_failure = read_model_if_exists(
                self._store, layout.evaluation_failure(request.evaluation_id), ModelFailure
            )
            execution = (
                read_model_if_exists(
                    self._store, layout.execution(request.execution_id), ExecutionRecord
                )
                if request.execution_id
                else None
            )
            if job_failure is not None:
                status, failure = "failed", job_failure.message
            elif execution is not None and execution.status.value in {"FAILED", "STOPPED"}:
                status, failure = "failed", execution.failure_message
        return HoldoutStatus(
            evaluation_id=request.evaluation_id,
            status=status,
            dataset_uri=request.dataset_uri,
            requested_at=request.requested_at,
            execution_id=request.execution_id,
            report=report,
            failure_message=failure,
            other_evaluation_ids=[
                r.evaluation_id for r in requests if r.evaluation_id != request.evaluation_id
            ],
        )

    def environment(self, experiment_id: str) -> tuple[RuntimeEnvironment, str]:
        """The environment the models were trained in, and its requirements.txt."""
        record = self.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        environment = read_model_if_exists(self._store, layout.environment, RuntimeEnvironment)
        if environment is None:
            raise ArtifactNotReadyError(
                "No training environment is recorded yet. It is written by the training jobs "
                "(experiments trained before this was added have none).",
                {"status": record.status.value},
            )
        requirements = (
            self._store.read_bytes(layout.requirements).decode("utf-8")
            if self._store.exists(layout.requirements)
            else ""
        )
        return environment, requirements

    def thresholds(
        self, experiment_id: str, model_name: str | None = None
    ) -> list[ThresholdAnalysis]:
        """Threshold analyses for the trained models (binary classification only)."""
        record = self.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        names = [model_name] if model_name else sorted(record.requested_models)
        if model_name and model_name not in record.requested_models:
            raise ValidationError(
                f"Model '{model_name}' was not trained in this experiment.",
                {"models": sorted(record.requested_models)},
            )
        found = [
            analysis
            for name in names
            if (
                analysis := read_model_if_exists(
                    self._store, layout.model_thresholds(name), ThresholdAnalysis
                )
            )
        ]
        if not found:
            raise ArtifactNotReadyError(
                "No threshold analysis is available. It is produced for binary classification "
                "models with probabilities once training has finished.",
                {"status": record.status.value},
            )
        return found

    def summary(self, experiment_id: str) -> ExperimentSummary:
        return self._artifact(
            experiment_id, lambda layout: layout.summary, ExperimentSummary, "experiment summary"
        )

    # --- feature review -----------------------------------------------------------------------
    def feature_review(self, experiment_id: str) -> FeatureReview:
        record = self.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        eda = self.eda(experiment_id)
        leakage = read_model_if_exists(self._store, layout.leakage, LeakageReport) or LeakageReport(
            experiment_id=experiment_id,
            generated_at=eda.generated_at,
            target_column=record.target_column,
        )
        stored = read_model_if_exists(self._store, layout.selected_features, FeatureSelection)
        derived_profiles = read_model_if_exists(
            self._store, layout.derived_profiles, DerivedProfileReport
        )
        approved = self._approved_proposals(layout)
        if stored is not None:
            selected = set(stored.selected_features)
            # Approved after the selection was saved: selected until someone excludes it.
            selected |= {p.name for p in approved if p.name not in stored.excluded_features}
        else:
            proposed, _ = default_feature_selection(eda, leakage, record.target_column)
            # An approved derived feature is, by that approval, meant to be used.
            selected = set(proposed) | {p.name for p in approved}

        items: list[FeatureReviewItem] = []
        for profile in eda.columns:
            if profile.name == record.target_column:
                continue
            risk = leakage.risk_for(profile.name)
            items.append(
                FeatureReviewItem(
                    feature=profile.name,
                    selected=profile.name in selected,
                    semantic_type=profile.semantic_type.value,
                    dtype=profile.dtype,
                    missing_percentage=profile.missing_percentage,
                    unique_count=profile.unique_count,
                    leakage_risk=(risk.risk_level if risk else LeakageRiskLevel.NONE).value,
                    recommended_action=(risk.recommended_action.value if risk else "keep"),
                    reasons=list(risk.reasons) if risk else [],
                )
            )
        items.extend(self._derived_items(layout, approved, derived_profiles, selected))
        return FeatureReview(
            experiment_id=experiment_id,
            target_column=record.target_column,
            problem_type=eda.target.inferred_problem_type,
            features=items,
            selected_count=sum(1 for i in items if i.selected),
            excluded_count=sum(1 for i in items if not i.selected),
            decided=stored is not None,
        )

    def save_features(
        self,
        experiment_id: str,
        *,
        selected_features: list[str],
        exclusion_reasons: dict[str, str] | None = None,
        user: str | None = None,
    ) -> FeatureSelection:
        """Persist the user's exact decision. Nothing is auto-added or auto-removed."""
        record = self._repository.get(experiment_id)
        if (
            record.status in ACTIVE_TRAINING_STATUSES
            or record.status is ExperimentStatus.EDA_RUNNING
        ):
            # Setting READY_FOR_TRAINING here while a run is in flight would take the status
            # back from the workflow and let a second run start over the first one's outputs.
            raise ConflictError(
                f"The feature selection cannot change while the experiment is "
                f"{record.status.value}.",
                {"status": record.status.value},
            )
        layout = ExperimentLayout(base=record.artifact_prefix)
        eda = self.eda(experiment_id)
        known = {profile.name for profile in eda.columns}
        # Approved derived features can be kept or excluded like any column.
        known |= {proposal.name for proposal in self._approved_proposals(layout)}

        unknown = [f for f in selected_features if f not in known]
        if unknown:
            raise ValidationError(
                "The selection contains columns that are not in the dataset.",
                {"unknown_features": unknown},
            )
        if record.target_column in selected_features:
            raise ValidationError(
                f"The target column '{record.target_column}' cannot be used as a feature."
            )
        if not selected_features:
            raise ValidationError("Select at least one feature.")

        selected = list(dict.fromkeys(selected_features))
        excluded = [f for f in known if f not in set(selected) and f != record.target_column]
        reasons = exclusion_reasons or {}
        selection = FeatureSelection(
            selected_features=selected,
            excluded_features=excluded,
            exclusion_reasons={k: v for k, v in reasons.items() if k in excluded},
            decided_at=datetime.now(UTC),
            decided_by=user,
        )
        write_model(self._store, layout.selected_features, selection)
        self._repository.update(
            experiment_id,
            status=ExperimentStatus.READY_FOR_TRAINING,
            current_stage="ready_for_training",
        )
        return selection

    # --- training -----------------------------------------------------------------------------
    def start_training(
        self, experiment_id: str, request: TrainingRequest, user: str | None = None
    ) -> ExperimentRecord:
        """Freeze the configuration to S3, then start the SageMaker training workflow."""
        lock = self._start_lock(experiment_id)
        if not lock.acquire(blocking=False):
            raise ConflictError("Training is already being started for this experiment.")
        try:
            return self._start_training(experiment_id, request, user)
        finally:
            lock.release()

    def _start_training(
        self, experiment_id: str, request: TrainingRequest, user: str | None
    ) -> ExperimentRecord:
        record = self._repository.get(experiment_id)
        if record.status in ACTIVE_TRAINING_STATUSES:
            raise ConflictError(
                f"Training is already running for this experiment (status {record.status.value})."
            )
        layout = ExperimentLayout(base=record.artifact_prefix)
        eda = self.eda(experiment_id)

        requested = (
            RequestedProblemType(request.problem_type)
            if request.problem_type
            else record.requested_problem_type
        )
        problem_type = self._resolve_problem_type(requested, eda)

        selection = read_model_if_exists(self._store, layout.selected_features, FeatureSelection)
        if selection is None:
            raise ConflictError("Confirm the feature selection before starting training.")

        try:
            primary_metric = resolve_primary_metric(problem_type, request.primary_metric)
        except UnknownMetricError as error:
            raise ValidationError(str(error)) from error

        base_models = self._resolve_models(request, problem_type)
        # Grid search expands the model set into one spec per grid point (validated first). With no
        # grid this is the base specs unchanged. Done before the split so the fold-count guard
        # sees the full grid-point count (grid-search design C4).
        models = self._apply_grid(base_models, request)
        # Ensembles: validate each approved definition, freeze the specs, and append one ModelSpec
        # per ensemble so the existing per-model machinery trains them. Validated before the split
        # so a rejected ensemble leaves the previous run untouched (ensembles-and-learners C5).
        ensemble_specs = self._resolve_ensembles(request, problem_type)
        models = models + [ModelSpec(name=spec.ensemble_id, params={}) for spec in ensemble_specs]
        # A stacking ensemble's b base learners are each fit K+1 times (OOF + refit), so it costs
        # more than its single ModelSpec; count the surplus (b - 1) toward the fit-count ceiling.
        extra_fit_models = sum(
            max(len(spec.base_learners) - 1, 0)
            for spec in ensemble_specs
            if spec.type is EnsembleType.STACKING
        )
        # Resolve and validate the split (including the CV guards) BEFORE anything is archived
        # or submitted, so a rejected request leaves the previous run untouched. The guard's
        # model_count is len(models) (grid points + ensembles) plus the stacking fit surplus.
        split = self._resolve_split(
            layout, request, eda, problem_type, models, extra_fit_models=extra_fit_models
        )
        # Validate the per-experiment requirements before archiving too (same reason).
        requirements = self._resolve_requirements(request)
        # Resolve and validate the outlier treatment before archiving (same reason).
        treatment = self._resolve_treatment(request)
        self.archive_previous_run(experiment_id, user=user)
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=record.name,
            created_at=record.created_at,
            created_by=user or record.created_by,
            dataset=record.dataset,
            target_column=record.target_column,
            problem_type=problem_type,
            requested_problem_type=requested,
            primary_metric=primary_metric,
            feature_selection=selection,
            # Only what a person approved, minus anything they then excluded in the selection.
            derived_features=[
                proposal
                for proposal in self._approved_proposals(layout)
                if proposal.name not in selection.excluded_features
            ],
            split=split,
            preprocessing=request.preprocessing or PreprocessingConfig(),
            models=models,
            class_weighting=request.class_weighting,
            environment=EnvironmentCapture(
                python_version=environment_metadata()["python_version"],
                platform=environment_metadata()["platform"],
                package_versions=package_versions(),
                ml_factory_version=ML_FACTORY_VERSION,
                captured_at=datetime.now(UTC),
            ),
            artifact_prefix=layout.base,
            requirements=requirements,
            treatment=treatment,
            ensembles=ensemble_specs,
        )
        write_model(self._store, layout.experiment_config, config)
        updated = self._repository.update(
            experiment_id,
            status=ExperimentStatus.READY_FOR_TRAINING,
            current_stage="training_requested",
            primary_metric=primary_metric,
            requested_models=[spec.name for spec in models],
        )
        handle = self._orchestrator.start_training(updated)
        LOGGER.info("Started training for %s (%s)", experiment_id, handle.execution_id)
        return self._repository.update(experiment_id, training_execution_arn=handle.execution_id)

    def training_status(self, experiment_id: str) -> TrainingStatus:
        """Per-model progress, derived from the artifacts each training job writes."""
        record = self.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        states = [
            self._model_state(layout, name, record.status)
            for name in sorted(record.requested_models)
        ]
        return TrainingStatus(
            experiment_id=experiment_id,
            status=record.status.value,
            current_stage=record.current_stage,
            models=states,
            best_model=record.best_model,
            best_score=record.best_score,
            primary_metric=record.primary_metric,
            failure_message=record.failure_message,
            updated_at=record.updated_at,
        )

    # --- helpers ------------------------------------------------------------------------------
    def _approved_proposals(self, layout: ExperimentLayout) -> list[FeatureProposal]:
        report = read_model_if_exists(self._store, layout.feature_proposals, FeatureProposalReport)
        return report.approved if report is not None else []

    def _derived_items(
        self,
        layout: ExperimentLayout,
        approved: list[FeatureProposal],
        profiles: DerivedProfileReport | None,
        selected: set[str],
    ) -> list[FeatureReviewItem]:
        """One review row per approved derived feature.

        Stats come from the derived-profiling job. Until it finishes, the row is shown with
        "pending" stats so the feature can still be kept or excluded.
        """
        preparation = read_model_if_exists(self._store, layout.preparation, PreparationReport)
        built = {f.name for f in preparation.derived_features} if preparation else set()
        items: list[FeatureReviewItem] = []
        for proposal in approved:
            profile = profiles.profile_for(proposal.name) if profiles else None
            status = "built" if proposal.name in built else "approved"
            if profile is None:
                items.append(
                    FeatureReviewItem(
                        feature=proposal.name,
                        selected=proposal.name in selected,
                        semantic_type="pending",
                        dtype="pending",
                        missing_percentage=0.0,
                        unique_count=0,
                        leakage_risk="pending",
                        recommended_action="keep",
                        reasons=[f"{proposal.op.value}: profiling has not finished yet."],
                        origin="derived",
                        status=status,
                    )
                )
                continue
            items.append(
                FeatureReviewItem(
                    feature=proposal.name,
                    selected=proposal.name in selected,
                    semantic_type=profile.semantic_type.value,
                    dtype=profile.dtype,
                    missing_percentage=profile.missing_percentage,
                    unique_count=profile.unique_count,
                    leakage_risk=profile.leakage_risk.value,
                    recommended_action=profile.recommended_action.value,
                    reasons=list(profile.reasons),
                    origin="derived",
                    status=status,
                )
            )
        return items

    def _model_state(
        self, layout: ExperimentLayout, name: str, status: ExperimentStatus
    ) -> ModelRunState:
        metadata = read_model_if_exists(self._store, layout.model_metadata(name), ModelMetadata)
        if metadata is not None:
            return ModelRunState(
                model_name=name,
                display_name=metadata.display_name,
                status=ModelRunStatus.COMPLETED.value,
                primary_score=metadata.primary_score,
            )
        failure = read_model_if_exists(self._store, layout.model_failure(name), ModelFailure)
        if failure is not None:
            return ModelRunState(
                model_name=name,
                display_name=_display_name(name),
                status=ModelRunStatus.FAILED.value,
                failure_message=failure.message,
            )
        pending = (
            ModelRunStatus.RUNNING if status is ExperimentStatus.TRAINING else ModelRunStatus.QUEUED
        )
        return ModelRunState(
            model_name=name, display_name=_display_name(name), status=pending.value
        )

    def _artifact(self, experiment_id, locator, model_type, label):
        record = self.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        artifact = read_model_if_exists(self._store, locator(layout), model_type)
        if artifact is None:
            raise ArtifactNotReadyError(
                f"The {label} is not available yet (status: {record.status.value}).",
                {"status": record.status.value, "stage": record.current_stage},
            )
        return artifact

    def _resolve_problem_type(self, requested: RequestedProblemType, eda: EdaReport) -> ProblemType:
        if requested is not RequestedProblemType.AUTO:
            return ProblemType(requested.value)
        inferred = eda.target.inferred_problem_type
        if inferred is None:
            raise ConflictError(
                "The problem type could not be inferred; choose one explicitly.",
                {"target_column": eda.target.column},
            )
        return inferred

    def _resolve_models(
        self, request: TrainingRequest, problem_type: ProblemType
    ) -> list[ModelSpec]:
        names = request.models or default_model_names(problem_type)
        if not names:
            raise ValidationError(
                f"No model plugin is available for {problem_type.value} in this environment."
            )
        # An override for a model that is not being trained would be silently ignored, and the
        # user would believe it was applied.
        stray = sorted(set(request.model_parameters) - set(names))
        if stray:
            raise ValidationError(
                "Parameters were given for models that are not in this training run.",
                {"models_with_parameters": stray, "models_in_run": list(dict.fromkeys(names))},
            )
        requirement_names = {requirement_name(line) for line in request.requirements}
        specs: list[ModelSpec] = []
        for name in dict.fromkeys(names):
            try:
                plugin = get_plugin(name)
            except UnknownModelError as error:
                raise ValidationError(str(error)) from error
            if not plugin.supports(problem_type):
                raise ValidationError(
                    f"Model '{name}' does not support {problem_type.value}.",
                    {"supported_models": [p.name for p in plugins_for(problem_type)]},
                )
            available, reason = plugin.is_available()
            # Open catalogue (ensembles-and-learners Req 2.3): an installed learner may be absent
            # from the control plane but is accepted when its package is in the approved
            # requirements — it is installed on the stock image at job start (stock-image B10).
            if not available and requirement_name(plugin.library) not in requirement_names:
                raise ValidationError(
                    f"Model '{name}' is not available: {reason}. Add its package "
                    f"('{plugin.library}') to this experiment's requirements to use it.",
                    {"reason": "MODEL_NOT_INSTALLED", "model": name, "package": plugin.library},
                )
            specs.append(ModelSpec(name=name, params=dict(request.model_parameters.get(name, {}))))
        return specs

    def _apply_grid(self, base_specs: list[ModelSpec], request: TrainingRequest) -> list[ModelSpec]:
        """Expand a parameter grid into one ModelSpec per grid point (grid-search design C4).

        With no grid, the base specs are returned unchanged (today's behaviour). Otherwise each
        grid value is validated against the plugin's parameter_space and the non-parameter keys
        are refused, before anything is submitted (Req 1.3, 1.4). A model without a grid entry
        keeps its single default point (Req 1.5).
        """
        if not request.parameter_grid:
            return base_specs

        base_names = [spec.name for spec in base_specs]
        reserved = {"features", "split", "outlier", "outlier_handling", "imputation"}
        offending = sorted(set(request.parameter_grid) & reserved)
        if offending:
            raise ValidationError(
                "A grid may vary model parameters only, not " + ", ".join(offending) + ".",
                {"reason": "GRID_NON_PARAMETER_KEY", "keys": offending},
            )
        stray = sorted(set(request.parameter_grid) - set(base_names))
        if stray:
            raise ValidationError(
                "A grid was given for models not in this run.",
                {"reason": "GRID_MODEL_NOT_IN_RUN", "models": stray},
            )
        # Validate every value against the plugin's declared parameter space.
        for model_name, params in request.parameter_grid.items():
            plugin = get_plugin(model_name)
            for param, values in params.items():
                for value in values:
                    reasons = plugin.validate_grid_value(param, value)
                    if reasons:
                        raise ValidationError(
                            f"{model_name}.{param}={value!r} is invalid: {'; '.join(reasons)}",
                            {
                                "reason": "GRID_VALUE_INVALID",
                                "model": model_name,
                                "parameter": param,
                                "value": value,
                            },
                        )

        grid = ParameterGrid(grid=request.parameter_grid)
        expanded = expand_grid(grid, base_names)
        base_overrides = {spec.name: dict(spec.params) for spec in base_specs}
        specs: list[ModelSpec] = []
        for name in base_names:
            points = expanded.get(name) or []
            if not points:
                # No grid entry for this model: keep its single (default/override) spec.
                specs.append(next(s for s in base_specs if s.name == name))
                continue
            for point in points:
                # A grid point starts from the model's manual overrides, then applies the point.
                params = {**base_overrides.get(name, {}), **point.params}
                specs.append(ModelSpec(name=point.point_id, params=params))
        return specs

    def grid_plan(
        self, base_specs: list[ModelSpec], request: TrainingRequest, fold_count: int
    ) -> GridPlan | None:
        """The Grid_Plan for a request (points + fit count), or None when there is no grid."""
        if not request.parameter_grid:
            return None
        base_names = [spec.name for spec in base_specs]
        expanded = expand_grid(ParameterGrid(grid=request.parameter_grid), base_names)
        points: list[GridPoint] = []
        for name in base_names:
            points.extend(expanded.get(name) or [])
        fits_per_point = fold_count + 1 if fold_count >= 2 else 1
        return GridPlan(
            points=points,
            fold_count=fold_count if fold_count >= 2 else 1,
            fits_per_point=fits_per_point,
            total_fits=len(points) * fits_per_point,
        )

    # --- dataset pinning ----------------------------------------------------------------------
    def _pin_dataset(self, dataset: DatasetReference, destination_dir: str) -> DatasetReference:
        """Copy the dataset under the experiment's own prefix and point every stage at the copy.

        The upload inbox (and any approved prefix) stays writable after registration — a
        pre-signed PUT is valid for its whole TTL — so reading the source directly would let EDA
        and training see different bytes. S3 copies are server-side; nothing passes through the
        control plane. A prefix of Parquet parts is copied part by part.
        """
        if not self._settings.pin_datasets:
            return dataset
        source = dataset.uri
        if source.endswith("/"):
            name = source.rstrip("/").rsplit("/", 1)[-1] or "dataset"
            target = join_uri(destination_dir, name) + "/"
            for member in self._store.list_uris(source):
                if member.endswith("/"):
                    continue
                copy_object(
                    self._store, member, join_uri(target.rstrip("/"), member[len(source) :])
                )
            return dataset.model_copy(update={"uri": target, "source_uri": source})
        filename = dataset.original_filename or source.rsplit("/", 1)[-1]
        target = join_uri(destination_dir, filename)
        copy_object(self._store, source, target)
        copied = self._store.stat(target)
        if copied is None:
            raise ValidationError("The dataset could not be copied into the experiment.")
        if dataset.size_bytes is not None and copied.size_bytes != dataset.size_bytes:
            raise ConflictError(
                "The dataset changed while it was being registered; register it again.",
                {"dataset_uri": source},
            )
        return dataset.model_copy(
            update={
                "uri": target,
                "source_uri": source,
                "etag": copied.etag or dataset.etag,
                "version_id": copied.version_id,
                "size_bytes": copied.size_bytes,
                "last_modified": copied.last_modified,
            }
        )

    # --- runs ---------------------------------------------------------------------------------
    def archive_previous_run(
        self, experiment_id: str, *, user: str | None = None
    ) -> ArchivedRun | None:
        """Move the previous run's outputs to ``runs/<run_id>/`` before a new run starts.

        Without this, a model that succeeded in run 1 and failed in run 2 would still have run
        1's ``metadata.json``, and evaluation would report it as a success of run 2. Only called
        when no workflow is active (``_start_training`` checks), so no job is writing here.
        """
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        uris: list[str] = []
        for prefix in RUN_OUTPUT_PREFIXES:
            uris.extend(self._store.list_uris(layout.path(prefix) + "/"))
        uris.extend(layout.path(name) for name in RUN_OUTPUT_FILES)
        uris = [u for u in dict.fromkeys(uris) if self._store.stat(u) is not None]
        if not uris:
            return None

        config = read_model_if_exists(self._store, layout.experiment_config, ExperimentConfig)
        comparison = read_model_if_exists(self._store, layout.comparison, ComparisonReport)
        execution_id = record.training_execution_arn
        token = execution_id.rsplit(".", 1)[-1] if execution_id else None
        run_id = f"run-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}" + (
            f"-{token}" if token else ""
        )
        archive = layout.run_archive(run_id)
        for uri in uris:
            copy_object(self._store, uri, join_uri(archive, uri[len(layout.base) + 1 :]))
        if self._store.exists(layout.experiment_config):
            copy_object(
                self._store, layout.experiment_config, join_uri(archive, EXPERIMENT_CONFIG_FILE)
            )
        manifest = ArchivedRun(
            run_id=run_id,
            experiment_id=experiment_id,
            archived_at=datetime.now(UTC),
            archived_by=user,
            execution_id=execution_id,
            primary_metric=comparison.primary_metric if comparison else None,
            best_model=comparison.best_model if comparison else None,
            best_score=comparison.best_score if comparison else None,
            models=[m.name for m in config.enabled_models_specs()] if config else [],
            selected_features=list(config.feature_selection.selected_features) if config else [],
            split_strategy=config.split.strategy if config else None,
            dataset_uri=config.dataset.uri if config else None,
            dataset_etag=config.dataset.etag if config else None,
            archived_uris=len(uris),
            # How the archived run measured its models (design C7, Req 10.6).
            evaluation_mode=config.split.evaluation_mode if config else None,
            cv_folds=(config.split.cv.folds if config and config.split.cv is not None else None),
            test_fraction=config.split.test_fraction if config else None,
        )
        write_model(self._store, layout.run_manifest(run_id), manifest)
        for uri in uris:  # delete only after everything was copied and the manifest written
            self._store.delete(uri)
        LOGGER.info("Archived %d object(s) of %s to %s", len(uris), experiment_id, archive)
        return manifest

    def list_runs(self, experiment_id: str) -> list[ArchivedRun]:
        """Archived earlier runs, newest first. The current run is the experiment itself."""
        layout = ExperimentLayout(base=self.get(experiment_id).artifact_prefix)
        runs = [
            read_model(self._store, uri, ArchivedRun)
            for uri in self._store.list_uris(layout.runs_prefix)
            if uri.endswith("/run.json") and uri.count("/") == layout.runs_prefix.count("/") + 1
        ]
        return sorted(runs, key=lambda r: r.archived_at, reverse=True)

    def rerun_training(self, experiment_id: str, user: str | None = None) -> ExperimentRecord:
        """Train again from the frozen configuration, unchanged (reproducibility, Req 3.6).

        The earlier run is archived first, so the two can be compared with ``list_runs``.
        """
        lock = self._start_lock(experiment_id)
        if not lock.acquire(blocking=False):
            raise ConflictError("Training is already being started for this experiment.")
        try:
            record = self._repository.get(experiment_id)
            if record.status in ACTIVE_TRAINING_STATUSES:
                raise ConflictError("Training is already running for this experiment.")
            layout = ExperimentLayout(base=record.artifact_prefix)
            config = read_model_if_exists(self._store, layout.experiment_config, ExperimentConfig)
            if config is None:
                raise ConflictError("This experiment has no frozen configuration to rerun.")
            self.archive_previous_run(experiment_id, user=user)
            write_model(self._store, layout.experiment_config, config)  # unchanged, re-asserted
            updated = self._repository.update(
                experiment_id,
                status=ExperimentStatus.READY_FOR_TRAINING,
                current_stage="training_requested",
                primary_metric=config.primary_metric,
                requested_models=[spec.name for spec in config.enabled_models_specs()],
            )
            handle = self._orchestrator.start_training(updated)
            return self._repository.update(
                experiment_id, training_execution_arn=handle.execution_id
            )
        finally:
            lock.release()

    def _resolve_requirements(self, request: TrainingRequest) -> RequirementsSpec | None:
        """The approved per-experiment pip requirements, validated before anything is submitted.

        Empty -> ``None`` (nothing installed; config byte-identical to a single-image run). Each
        entry must be a well-formed PEP 508 requirement. When ``enforce_package_allowlist`` is on,
        every package must be on ``package_install_allowlist`` (off by default for trusted users).
        """
        lines = [entry.strip() for entry in request.requirements if entry and entry.strip()]
        if not lines:
            return None
        try:
            spec = RequirementsSpec(packages=lines)
        except RequirementsError as error:
            raise ValidationError(str(error), {"reason": "REQUIREMENT_MALFORMED"}) from error
        except ValueError as error:  # pydantic wraps the field validator error
            raise ValidationError(
                f"A requirement is malformed: {error}", {"reason": "REQUIREMENT_MALFORMED"}
            ) from error
        if self._settings.enforce_package_allowlist:
            allowed = {requirement_name(a) for a in self._settings.package_install_allowlist}
            blocked = [p for p in spec.packages if requirement_name(p) not in allowed]
            if blocked:
                raise ValidationError(
                    "These packages are not on the install allowlist: " + ", ".join(blocked),
                    {"reason": "PACKAGE_NOT_ALLOWED", "blocked": blocked},
                )
        return spec

    def _resolve_treatment(self, request: TrainingRequest) -> TreatmentConfig | None:
        """The approved outlier treatment, validated and frozen before anything is submitted.

        Absence (or a 'none' detector) -> ``None``, so the config and the preprocessing chain are
        byte-identical to today. The detector/handling combination is validated here (e.g. 'cap'
        with a multivariate detector is refused) so a malformed request never starts a job.
        """
        detector = request.outlier_detector
        if detector is None or detector.value == "none":
            return None
        fields: dict[str, object] = {"outlier_detector": detector}
        if request.outlier_handling is not None:
            fields["outlier_handling"] = request.outlier_handling
        else:
            fields["outlier_handling"] = OutlierHandling.FLAG
        if request.zscore_threshold is not None:
            fields["zscore_threshold"] = request.zscore_threshold
        if request.iqr_multiplier is not None:
            fields["iqr_multiplier"] = request.iqr_multiplier
        if request.isolation_contamination is not None:
            fields["isolation_contamination"] = request.isolation_contamination
        try:
            return TreatmentConfig(**fields)
        except ValueError as error:  # pydantic wraps the model validator (e.g. cap + isoforest)
            raise ValidationError(
                f"The outlier treatment is invalid: {error}",
                {"reason": "TREATMENT_INVALID"},
            ) from error

    def _resolve_ensembles(
        self, request: TrainingRequest, problem_type: ProblemType
    ) -> list[EnsembleSpec]:
        """Validate and freeze the approved ensembles (ensembles-and-learners C5).

        Empty -> ``[]`` (today's single-learner training). Each ensemble needs >=2 base learners,
        every base/meta learner must be registered and support the problem type, a stacking
        ensemble requires CV_Mode, and an installed learner's package must be in the approved
        requirements. All checked before anything is archived or submitted.
        """
        if not request.ensembles:
            return []
        specs: list[EnsembleSpec] = []
        requirement_names = {requirement_name(line) for line in request.requirements}
        for ensemble in request.ensembles:
            base_learners = [
                BaseLearnerSpec(name=b.name, params=dict(b.params), weight=b.weight)
                for b in ensemble.base_learners
            ]
            meta_learner: MetaLearnerSpec | None = None
            if ensemble.type is EnsembleType.STACKING:
                if request.evaluation_mode != "cv":
                    raise ValidationError(
                        "A stacking ensemble requires cross-validation; set "
                        "evaluation_mode='cv'. It trains its meta-learner on out-of-fold "
                        "predictions, which only CV_Mode can produce.",
                        {"reason": "STACKING_REQUIRES_CV"},
                    )
                if ensemble.meta_learner is not None:
                    meta_learner = MetaLearnerSpec(
                        name=ensemble.meta_learner, params=dict(ensemble.meta_learner_params)
                    )
                else:
                    meta_learner = default_meta_learner(problem_type)
            try:
                spec = EnsembleSpec(
                    type=ensemble.type,
                    base_learners=base_learners,
                    voting_rule=ensemble.voting_rule,
                    meta_learner=meta_learner,
                )
            except ValueError as error:  # <2 base learners
                raise ValidationError(
                    f"Invalid ensemble: {error}", {"reason": "ENSEMBLE_INVALID"}
                ) from error
            self._validate_ensemble_learners(spec, problem_type, requirement_names)
            specs.append(spec)
        return specs

    def _validate_ensemble_learners(
        self,
        spec: EnsembleSpec,
        problem_type: ProblemType,
        requirement_names: set[str],
    ) -> None:
        """Every base/meta learner is registered, supports the problem type, and is installable."""
        learners = [b.name for b in spec.base_learners]
        if spec.meta_learner is not None:
            learners.append(spec.meta_learner.name)
        for learner in learners:
            try:
                plugin = get_plugin(learner)
            except UnknownModelError as error:
                raise ValidationError(
                    f"Ensemble names an unknown learner '{learner}'.",
                    {"reason": "ENSEMBLE_UNKNOWN_LEARNER", "learner": learner},
                ) from error
            if not plugin.supports(problem_type):
                raise ValidationError(
                    f"Learner '{learner}' does not support {problem_type.value}.",
                    {"reason": "ENSEMBLE_LEARNER_PROBLEM_TYPE", "learner": learner},
                )
            available, _reason = plugin.is_available()
            if not available:
                package = requirement_name(plugin.library)
                if package not in requirement_names:
                    raise ValidationError(
                        f"Ensemble learner '{learner}' needs package '{plugin.library}' added to "
                        "this experiment's requirements.",
                        {
                            "reason": "ENSEMBLE_LEARNER_NOT_INSTALLED",
                            "learner": learner,
                            "package": plugin.library,
                        },
                    )

    def _resolve_split(
        self,
        layout: ExperimentLayout,
        request: TrainingRequest,
        eda: EdaReport,
        problem_type: ProblemType,
        models: list[ModelSpec],
        extra_fit_models: int = 0,
    ) -> SplitConfig:
        """The split, defaulting to a group split when the business context names an entity.

        In CV_Mode this also resolves K, runs the fold-count guard (Req 1.3-1.5), resolves the
        CV_Scheme and freezes the configured flag thresholds into ``split.cv`` so the stages
        read them from the config alone. All CV checks happen here, before ``archive_previous_run``
        and before the orchestrator is called.
        """
        columns = {profile.name for profile in eda.columns}
        business = read_model_if_exists(self._store, layout.business_context, BusinessContext)
        group_column = request.group_column
        if group_column is None and business and business.entity_id_column in columns:
            group_column = business.entity_id_column
        strategy = request.split_strategy
        if strategy is None:
            if request.time_column:
                strategy = "temporal"
            elif group_column:
                strategy = "group"
            else:
                strategy = "stratified_random" if problem_type.is_classification else "random"
        if strategy not in available_strategies():
            raise ValidationError(
                f"Unknown split strategy '{strategy}'.", {"available": available_strategies()}
            )
        if strategy == "group" and group_column not in columns:
            raise ValidationError(
                "A group split needs group_column (or a business-context entity id column) "
                "naming a dataset column.",
                {"group_column": group_column},
            )
        if strategy == "temporal" and request.time_column not in columns:
            raise ValidationError(
                "A temporal split needs time_column naming a dataset column.",
                {"time_column": request.time_column},
            )

        # Test_Fold: an explicit fraction implies the fold; the fold without a fraction uses the
        # configured default (design D4).
        test_fraction: float | None = None
        if request.test_fraction is not None:
            test_fraction = request.test_fraction
        elif request.test_fold:
            test_fraction = self._settings.default_test_fraction

        split = SplitConfig(
            strategy=strategy,
            validation_fraction=request.validation_fraction,
            random_seed=request.random_seed,
            group_column=group_column if strategy == "group" else None,
            time_column=request.time_column if strategy == "temporal" else None,
            test_fraction=test_fraction,
        )

        # The fold-count guard runs even in Single_Split_Mode so that cv_folds without CV_Mode is
        # rejected (CV_FOLDS_WITHOUT_CV_MODE); it returns the resolved K only when CV is requested.
        # A stacking ensemble costs more than one model's K+1 fits (its b base learners are each
        # fit K+1 times for OOF + refit); extra_fit_models carries that surplus so the ceiling
        # bounds the true fold-fit count (ensembles-and-learners Req 4.7).
        folds = resolve_cv_folds(
            requested_folds=request.cv_folds,
            evaluation_mode=request.evaluation_mode,
            model_count=len(models) + extra_fit_models,
            default_folds=self._settings.cv_default_folds,
            max_folds=self._settings.cv_max_folds,
            max_fits=self._settings.cv_max_fits,
        )
        if request.evaluation_mode != "cv":
            return split

        scheme = resolve_cv_scheme(problem_type, split)
        return split.model_copy(
            update={
                "evaluation_mode": EvaluationMode.CV,
                "cv": CrossValidationConfig(
                    folds=folds,
                    scheme=scheme,
                    instability_std=self._settings.cv_instability_std,
                    instability_cv=self._settings.cv_instability_cv,
                    outlier_multiplier=self._settings.cv_outlier_multiplier,
                    outlier_min_gap=self._settings.cv_outlier_min_gap,
                ),
            }
        )

    def _validate_dataset(self, uri: str, file_format: str | None) -> DatasetReference:
        candidate = (uri or "").strip()
        if not is_s3_uri(candidate):
            raise ValidationError(
                "The dataset must be an s3:// URI — one returned by the upload step or an "
                "approved location.",
                {"dataset_uri": candidate},
            )
        if ".." in candidate:
            raise ValidationError(
                "A dataset location must not contain '..'.", {"dataset_uri": candidate}
            )
        if not self._dataset_uri_allowed(candidate):
            raise ValidationError(
                "This dataset location is not approved. Upload the file, or ask for the location "
                "to be approved.",
                {"dataset_uri": candidate, "allowed_prefixes": self._allowed_prefixes()},
            )
        resolved = (file_format or detect_format(candidate)).lower()
        if resolved == "unknown":
            raise ValidationError(
                "Could not determine the dataset format. Supported: csv, tsv, parquet.",
                {"dataset_uri": candidate},
            )
        metadata = self._store.stat(candidate) if not candidate.endswith("/") else None
        if metadata is None and not candidate.endswith("/"):
            raise ValidationError(
                "The dataset could not be found, or the platform has no access to it.",
                {"dataset_uri": candidate},
            )
        return DatasetReference(
            uri=candidate,
            file_format=resolved,
            version_id=metadata.version_id if metadata else None,
            etag=metadata.etag if metadata else None,
            size_bytes=metadata.size_bytes if metadata else None,
            last_modified=metadata.last_modified if metadata else None,
            original_filename=candidate.rstrip("/").rsplit("/", 1)[-1],
        )

    def validate_dataset(self, uri: str, file_format: str | None) -> DatasetReference:
        """Public form of the dataset checks, for other services (scoring)."""
        return self._validate_dataset(uri, file_format)

    def pin_dataset(self, dataset: DatasetReference, destination_dir: str) -> DatasetReference:
        return self._pin_dataset(dataset, destination_dir)

    def _allowed_prefixes(self) -> list[str]:
        """The upload inbox is always approved; configured prefixes extend it."""
        inbox = ExperimentLayout.uploads_prefix(self._settings.artifact_root).rstrip("/")
        return [inbox, *self._settings.allowed_dataset_prefixes]

    def _dataset_uri_allowed(self, uri: str) -> bool:
        return any(uri_matches_prefix(uri, prefix) for prefix in self._allowed_prefixes())


def _new_experiment_id() -> str:
    return f"exp-{uuid.uuid4().hex[:12]}"


def _display_name(model_name: str) -> str:
    try:
        return get_plugin(model_name).display_name
    except UnknownModelError:
        return model_name.replace("_", " ").title()
