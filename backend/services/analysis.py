"""Post-training analysis and operations: ablation, batch scoring, permutation importance,
pre-flight checks, execution status, and experiment-to-experiment comparison.

Every number here is read from an artifact a SageMaker job wrote; this service only selects,
validates and starts work. Long-running work (ablation, scoring) returns at once.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from backend.config import Settings
from backend.errors import ArtifactNotReadyError, ConflictError, NotFoundError, ValidationError
from backend.orchestration.base import ExperimentOrchestrator
from backend.repositories import ACTIVE_TRAINING_STATUSES, ExperimentRepository
from backend.schemas import ExecutionStatusView, JobStatusView, PreflightCheck
from backend.services.experiments import ExperimentService
from backend.tracking.status import normalize_processing_status, normalize_training_status
from ml_engine.contracts.business_context import BusinessContext
from ml_engine.contracts.common import SEVERITY_ORDER, ExperimentStatus, Severity
from ml_engine.contracts.comparison import ComparisonReport, ExperimentSummary
from ml_engine.contracts.config import FeatureSelection
from ml_engine.contracts.execution import ExecutionRecord, JobKind
from ml_engine.contracts.explain import AblationReport, PermutationImportance
from ml_engine.contracts.leakage_investigation import LeakageInvestigationReport, UserDecision
from ml_engine.contracts.model import ModelFailure, ModelMetadata
from ml_engine.contracts.proposals import DerivedProfileReport, FeatureProposalReport
from ml_engine.contracts.scoring import ScoringReport, ScoringRequest
from ml_engine.contracts.tracking import FailureDiagnosis
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    read_model,
    read_model_if_exists,
    write_model,
)

LOGGER = logging.getLogger("ml_factory.services.analysis")

ABLATION_DEFAULT_FEATURES = 5
ABLATION_MAX_FEATURES = 10
MAX_COMPARED_EXPERIMENTS = 5
_SERIOUS = SEVERITY_ORDER[Severity.HIGH]


class AnalysisService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ExperimentRepository,
        store: ObjectStore,
        orchestrator: ExperimentOrchestrator,
        experiments: ExperimentService,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store
        self._orchestrator = orchestrator
        self._experiments = experiments

    # --- pre-flight -----------------------------------------------------------------------------
    def preflight(self, experiment_id: str) -> PreflightCheck:
        """Everything worth knowing before training. Only ``blockers`` stop training."""
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        blockers: list[str] = []
        warnings: list[str] = []

        if (
            record.status in ACTIVE_TRAINING_STATUSES
            or record.status is ExperimentStatus.EDA_RUNNING
        ):
            blockers.append(f"The experiment is {record.status.value}; wait for it to finish.")
        if not self._store.exists(layout.eda):
            blockers.append("EDA has not completed.")
        selection = read_model_if_exists(self._store, layout.selected_features, FeatureSelection)
        if selection is None:
            blockers.append("No feature selection has been confirmed (decide_features).")

        current = (
            self._store.stat(record.dataset.uri) if not record.dataset.uri.endswith("/") else None
        )
        if record.dataset.etag and current is not None and current.etag != record.dataset.etag:
            blockers.append(
                "The dataset object changed after the experiment was created; create a new "
                "experiment from the new file."
            )
        if not record.dataset.source_uri and self._settings.pin_datasets:
            warnings.append(
                "The dataset is read from its original location, not a pinned copy; later edits "
                "to that object would change what trains."
            )

        business = read_model_if_exists(self._store, layout.business_context, BusinessContext)
        suggested_split = None
        if business is None:
            warnings.append(
                "No business context is set. Leak reasoning, the default group split and the "
                "interpretation all work better with it (set_business_context)."
            )
        elif business.entity_id_column:
            suggested_split = "group"
        elif business.temporal_columns:
            suggested_split = "temporal"
            warnings.append(
                "Time-related columns were named; consider a temporal split (time_column) so "
                "validation uses later rows than training."
            )

        pre = read_model_if_exists(
            self._store, layout.leakage_pre_training, LeakageInvestigationReport
        )
        unresolved: list[dict[str, str]] = []
        if pre is None:
            warnings.append("The pre-training leakage investigation has not been run.")
        elif selection is not None:
            selected = set(selection.selected_features)
            for warning in pre.warnings:
                if warning.feature not in selected:
                    continue
                if warning.user_decision is UserDecision.EXCLUDE:
                    warnings.append(
                        f"'{warning.feature}' was marked EXCLUDE but is still selected "
                        "(apply_leakage_exclusions)."
                    )
                elif (
                    warning.user_decision is UserDecision.PENDING
                    and SEVERITY_ORDER[warning.severity] >= _SERIOUS
                ):
                    unresolved.append(
                        {
                            "warning_id": warning.id,
                            "feature": warning.feature,
                            "severity": warning.severity.value,
                            "summary": warning.summary,
                        }
                    )
            if unresolved:
                warnings.append(
                    f"{len(unresolved)} high-severity leakage warning(s) on selected features "
                    "have no decision yet."
                )

        proposals = read_model_if_exists(
            self._store, layout.feature_proposals, FeatureProposalReport
        )
        profiles = read_model_if_exists(self._store, layout.derived_profiles, DerivedProfileReport)
        if proposals and proposals.approved and (profiles is None or not profiles.profiles):
            warnings.append(
                "Approved derived features have not been profiled yet; their statistics are "
                "pending in feature review."
            )

        return PreflightCheck(
            experiment_id=experiment_id,
            ready=not blockers,
            blockers=blockers,
            warnings=warnings,
            unresolved_leakage_warnings=unresolved,
            suggested_split=suggested_split,
        )

    # --- ablation --------------------------------------------------------------------------------
    def start_ablation(
        self,
        experiment_id: str,
        *,
        features: list[str] | None = None,
        model_name: str | None = None,
    ) -> dict[str, Any]:
        record, layout, comparison = self._trained(experiment_id)
        model = model_name or comparison.best_model
        metadata = read_model_if_exists(
            self._store, layout.model_metadata(model or ""), ModelMetadata
        )
        if metadata is None:
            raise ValidationError(f"Model '{model}' has no successful result in this run.")
        chosen = list(dict.fromkeys(features or self._ablation_candidates(layout, metadata)))
        if not chosen:
            raise ValidationError("No candidate features were found to ablate; name some.")
        if len(chosen) > ABLATION_MAX_FEATURES:
            raise ValidationError(
                f"At most {ABLATION_MAX_FEATURES} features per ablation (each is a retrain)."
            )
        start = getattr(self._orchestrator, "start_ablation", None)
        if not callable(start):
            raise ConflictError("This deployment's orchestrator cannot run ablations.")
        for uri in (layout.ablation, layout.ablation_failure):  # never show a stale result
            if self._store.exists(uri):
                self._store.delete(uri)
        handle = start(record, model_name=metadata.model_name, features=chosen)
        return {
            "experiment_id": experiment_id,
            "execution_id": handle.execution_id,
            "model_name": metadata.model_name,
            "features": chosen,
        }

    def ablation(self, experiment_id: str) -> dict[str, Any]:
        layout = self._layout(experiment_id)
        report = read_model_if_exists(self._store, layout.ablation, AblationReport)
        if report is not None:
            return {"status": "completed", "report": report.model_dump(mode="json")}
        failure = read_model_if_exists(self._store, layout.ablation_failure, ModelFailure)
        if failure is not None:
            return {"status": "failed", "failure_message": failure.message}
        return {"status": "not_started_or_running"}

    def _ablation_candidates(self, layout: ExperimentLayout, metadata: ModelMetadata) -> list[str]:
        """Post-training warnings first, then the strongest permutation features."""
        candidates: list[str] = []
        post = read_model_if_exists(
            self._store, layout.leakage_post_training, LeakageInvestigationReport
        )
        if post is not None:
            candidates.extend(w.feature for w in post.warnings)
        permutation = read_model_if_exists(
            self._store, layout.model_permutation(metadata.model_name), PermutationImportance
        )
        if permutation is not None:
            candidates.extend(e.feature for e in permutation.entries[:ABLATION_DEFAULT_FEATURES])
        pre = read_model_if_exists(
            self._store, layout.leakage_pre_training, LeakageInvestigationReport
        )
        if pre is not None:
            candidates.extend(
                w.feature for w in pre.warnings if SEVERITY_ORDER[w.severity] >= _SERIOUS
            )
        return list(dict.fromkeys(candidates))[:ABLATION_DEFAULT_FEATURES]

    # --- permutation importance ------------------------------------------------------------------
    def permutation_importance(
        self, experiment_id: str, model_name: str | None = None
    ) -> list[PermutationImportance]:
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        names = [model_name] if model_name else sorted(record.requested_models)
        found = [
            result
            for name in names
            if (
                result := read_model_if_exists(
                    self._store, layout.model_permutation(name), PermutationImportance
                )
            )
        ]
        if not found:
            raise ArtifactNotReadyError(
                "No permutation importance is available; it is written by each training job."
            )
        return found

    # --- batch scoring ---------------------------------------------------------------------------
    def start_scoring(
        self,
        experiment_id: str,
        *,
        dataset_uri: str,
        model_name: str | None = None,
        passthrough_columns: list[str] | None = None,
        file_format: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        record, layout, comparison = self._trained(experiment_id)
        model = model_name or comparison.best_model
        if not model or not self._store.exists(layout.model_artifact(model)):
            raise ValidationError(f"Model '{model}' has no trained artifact in this run.")
        dataset = self._experiments.validate_dataset(dataset_uri, file_format)
        scoring_id = f"score-{uuid.uuid4().hex[:12]}"
        dataset = self._experiments.pin_dataset(dataset, layout.scoring_input_dir(scoring_id))
        if passthrough_columns is None:
            business = read_model_if_exists(self._store, layout.business_context, BusinessContext)
            passthrough_columns = (
                [business.entity_id_column] if business and business.entity_id_column else []
            )
        request = ScoringRequest(
            scoring_id=scoring_id,
            experiment_id=experiment_id,
            model_name=model,
            dataset_uri=dataset.uri,
            source_uri=dataset.source_uri,
            file_format=dataset.file_format,
            passthrough_columns=passthrough_columns,
            requested_at=datetime.now(UTC),
            requested_by=user,
        )
        write_model(self._store, layout.scoring_request(scoring_id), request)
        start = getattr(self._orchestrator, "start_scoring", None)
        if not callable(start):
            raise ConflictError("This deployment's orchestrator cannot run batch scoring.")
        handle = start(record, scoring_id)
        request.execution_id = handle.execution_id
        write_model(self._store, layout.scoring_request(scoring_id), request)
        return self.scoring(experiment_id, scoring_id)

    def scoring(self, experiment_id: str, scoring_id: str | None = None) -> dict[str, Any]:
        layout = self._layout(experiment_id)
        requests = sorted(
            (
                read_model(self._store, uri, ScoringRequest)
                for uri in self._store.list_uris(layout.scoring_prefix)
                if uri.endswith("/request.json")
            ),
            key=lambda r: r.requested_at,
            reverse=True,
        )
        if not requests:
            raise ArtifactNotReadyError("No batch scoring has been requested for this experiment.")
        request = (
            requests[0]
            if scoring_id is None
            else next((r for r in requests if r.scoring_id == scoring_id), None)
        )
        if request is None:
            raise NotFoundError(f"No scoring '{scoring_id}' for this experiment.")
        result: dict[str, Any] = {
            "scoring_id": request.scoring_id,
            "model_name": request.model_name,
            "dataset_uri": request.source_uri or request.dataset_uri,
            "execution_id": request.execution_id,
            "requested_at": request.requested_at.isoformat(),
            "status": "running",
            "other_scoring_ids": [r.scoring_id for r in requests if r is not request],
        }
        report = read_model_if_exists(
            self._store, layout.scoring_report(request.scoring_id), ScoringReport
        )
        if report is not None:
            result.update(
                status="completed",
                predictions_uri=report.predictions_uri,
                row_count=report.row_count,
                prediction_columns=report.prediction_columns,
                prediction_summary=report.prediction_summary,
                warnings=[w.message for w in report.warnings],
            )
            return result
        failure = read_model_if_exists(
            self._store, layout.scoring_failure(request.scoring_id), ModelFailure
        )
        execution = (
            read_model_if_exists(
                self._store, layout.execution(request.execution_id), ExecutionRecord
            )
            if request.execution_id
            else None
        )
        if failure is not None:
            result.update(status="failed", failure_message=failure.message)
        elif execution is not None and execution.status.value in {"FAILED", "STOPPED"}:
            result.update(status="failed", failure_message=execution.failure_message)
        return result

    # --- execution status ------------------------------------------------------------------------
    def execution_status(
        self, experiment_id: str, execution_id: str | None = None
    ) -> ExecutionStatusView:
        """Per-job raw + normalized status and any failure diagnosis (requirements 6, 8)."""
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        execution_id = execution_id or record.training_execution_arn or record.eda_execution_arn
        if not execution_id or not execution_id.startswith(experiment_id + "."):
            raise NotFoundError("No execution with that id belongs to this experiment.")
        execution = read_model_if_exists(
            self._store, layout.execution(execution_id), ExecutionRecord
        )
        if execution is None:
            raise NotFoundError(f"No execution '{execution_id}'.")
        jobs: list[JobStatusView] = []
        for job in execution.jobs:
            normalized = (
                normalize_training_status(job.raw_status, job.secondary_status)
                if job.kind is JobKind.TRAINING
                else normalize_processing_status(job.raw_status)
            )
            diagnosis = read_model_if_exists(
                self._store, layout.diagnosis(execution_id, job.job_name), FailureDiagnosis
            )
            jobs.append(
                JobStatusView(
                    job_name=job.job_name,
                    stage=job.stage,
                    model_name=job.model_name,
                    state=job.state.value,
                    normalized_status=normalized.value if job.raw_status else None,
                    raw_status=job.raw_status,
                    secondary_status=job.secondary_status,
                    failure_reason=diagnosis.failure_reason if diagnosis else job.failure_reason,
                    diagnosis_category=diagnosis.category.value if diagnosis else None,
                    diagnosis_summary=diagnosis.summary if diagnosis else None,
                    recommended_action=diagnosis.recommended_action if diagnosis else None,
                )
            )
        return ExecutionStatusView(
            execution_id=execution.execution_id,
            kind=execution.kind.value,
            status=execution.status.value,
            stage=execution.stage,
            started_at=execution.started_at,
            updated_at=execution.updated_at,
            failure_code=execution.failure_code,
            failure_message=execution.failure_message,
            jobs=jobs,
        )

    # --- experiment comparison -------------------------------------------------------------------
    def compare_experiments(self, experiment_ids: list[str]) -> dict[str, Any]:
        """Side-by-side headline results and what changed, relative to the first experiment."""
        ids = list(dict.fromkeys(experiment_ids))
        if not 2 <= len(ids) <= MAX_COMPARED_EXPERIMENTS:
            raise ValidationError(f"Compare between 2 and {MAX_COMPARED_EXPERIMENTS} experiments.")
        rows: list[dict[str, Any]] = []
        baseline_features: set[str] | None = None
        for experiment_id in ids:
            record = self._repository.get(experiment_id)
            layout = ExperimentLayout(base=record.artifact_prefix)
            summary = read_model_if_exists(self._store, layout.summary, ExperimentSummary)
            row: dict[str, Any] = {
                "experiment_id": experiment_id,
                "name": record.name,
                "status": record.status.value,
                "target_column": record.target_column,
                "dataset_uri": record.dataset.source_uri or record.dataset.uri,
                "dataset_etag": record.dataset.etag,
            }
            if summary is None:
                row["trained"] = False
                rows.append(row)
                continue
            config = summary.config
            features = set(config.feature_selection.selected_features) | {
                d.name for d in config.derived_features
            }
            comparison = summary.comparison
            row.update(
                trained=True,
                problem_type=config.problem_type.value,
                primary_metric=config.primary_metric,
                best_model=comparison.best_model if comparison else None,
                best_score=comparison.best_score if comparison else None,
                models=[m.name for m in config.enabled_models_specs()],
                split_strategy=config.split.strategy,
                validation_fraction=config.split.validation_fraction,
                feature_count=len(features),
                evaluation_mode=config.split.evaluation_mode.value,
                cv_folds=config.split.cv.folds if config.split.cv is not None else None,
                test_fraction=config.split.test_fraction,
            )
            if baseline_features is None:
                baseline_features = features
            else:
                row["features_added_vs_first"] = sorted(features - baseline_features)[:50]
                row["features_removed_vs_first"] = sorted(baseline_features - features)[:50]
            rows.append(row)

        trained = [r for r in rows if r.get("trained")]
        notes: list[str] = []
        if len({r["primary_metric"] for r in trained}) > 1:
            notes.append(
                "The experiments use different primary metrics; scores are not comparable."
            )
        if len({(r["dataset_uri"], r["dataset_etag"]) for r in rows}) > 1:
            notes.append("The experiments use different datasets (or dataset versions).")
        if len({(r.get("split_strategy"), r.get("validation_fraction")) for r in trained}) > 1:
            notes.append("The validation splits differ, so scores are measured on different rows.")
        if len({r.get("evaluation_mode") for r in trained}) > 1:
            notes.append(
                "The experiments mix evaluation modes (cross-validation and a single split), so "
                "their headline scores are measured differently and are not directly comparable."
            )
        if len({r["target_column"] for r in rows}) > 1:
            notes.append("The targets differ; these experiments answer different questions.")
        return {"experiments": rows, "comparability_notes": notes}

    # --- helpers ---------------------------------------------------------------------------------
    def _trained(self, experiment_id: str):
        record = self._repository.get(experiment_id)
        if record.status in ACTIVE_TRAINING_STATUSES:
            raise ConflictError("Training is running; wait for it to finish.")
        layout = ExperimentLayout(base=record.artifact_prefix)
        comparison = read_model_if_exists(self._store, layout.comparison, ComparisonReport)
        if comparison is None:
            raise ConflictError("Train the experiment first.")
        return record, layout, comparison

    def _layout(self, experiment_id: str) -> ExperimentLayout:
        return ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)


__all__ = ["AnalysisService"]
