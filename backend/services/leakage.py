"""Business context and the human-controlled leakage investigation (requirements 11, 12).

The service stores the business context once, builds the pre-training leakage warnings from the
deterministic EDA + leakage artifacts (reusing ``ml_engine.leakage.investigation``), lets a user
record a decision on each warning, and — after training — builds the post-training phase from the
model metadata. The two phases are stored as separate artifacts and never overwrite each other.

Nothing is auto-excluded except the target: a decision only records what the user chose. Applying
an "exclude" decision to the feature selection is a separate, explicit call so approval and
selection stay distinct (requirement 12.6).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from backend.config import Settings
from backend.errors import AppError, ConflictError, NotFoundError, ValidationError
from backend.repositories import ACTIVE_TRAINING_STATUSES, ExperimentRepository
from backend.services.reasoning import ReasoningService
from ml_engine.contracts.business_context import BusinessContext
from ml_engine.contracts.common import RequestedProblemType
from ml_engine.contracts.comparison import ComparisonReport
from ml_engine.contracts.config import FeatureSelection
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.explain import AblationReport, PermutationImportance
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.leakage_investigation import (
    InvestigationPhase,
    LeakageInvestigationReport,
    ReviewStatus,
    UserDecision,
)
from ml_engine.contracts.model import ModelMetadata
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    read_model_if_exists,
    write_model,
)
from ml_engine.leakage.investigation import build_post_training, build_pre_training

LOGGER = logging.getLogger("ml_factory.services.leakage")


class LeakageService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ExperimentRepository,
        store: ObjectStore,
        reasoning: ReasoningService | None = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store
        self._reasoning = reasoning

    # --- business context ---------------------------------------------------------------------
    def set_business_context(
        self,
        experiment_id: str,
        *,
        objective: str,
        prediction_scenario: str,
        scoring_trigger: str | None = None,
        prediction_horizon: str | None = None,
        entity_id_column: str | None = None,
        temporal_columns: list[str] | None = None,
        investigate_leakage: bool = True,
        user: str | None = None,
    ) -> BusinessContext:
        record = self._repository.get(experiment_id)
        if not objective.strip() or not prediction_scenario.strip():
            raise ValidationError("Both an objective and a prediction scenario are required.")
        context = BusinessContext(
            experiment_id=experiment_id,
            created_at=datetime.now(UTC),
            created_by=user or record.created_by,
            objective=objective.strip(),
            prediction_scenario=prediction_scenario.strip(),
            target_column=record.target_column,
            problem_type=record.requested_problem_type or RequestedProblemType.AUTO,
            primary_metric=record.primary_metric,
            scoring_trigger=scoring_trigger,
            prediction_horizon=prediction_horizon,
            entity_id_column=entity_id_column,
            temporal_columns=temporal_columns or [],
            investigate_leakage=investigate_leakage,
        )
        layout = ExperimentLayout(base=record.artifact_prefix)
        write_model(self._store, layout.business_context, context)
        return context

    def get_business_context(self, experiment_id: str) -> BusinessContext | None:
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        return read_model_if_exists(self._store, layout.business_context, BusinessContext)

    # --- investigation phases -----------------------------------------------------------------
    def build_pre_training_phase(
        self,
        experiment_id: str,
        *,
        llm_notes: dict[str, str] | None = None,
        use_llm: bool = True,
    ) -> LeakageInvestigationReport:
        """Build (or rebuild) the pre-training warnings from the deterministic artifacts.

        When Bedrock is enabled and the business context leaves leakage investigation on (the
        default), the four-signal leak reasoning runs first over the governed context and its
        suspicions are attached to the warnings — as suspicion, never as proof. If Bedrock is
        unavailable the deterministic warnings are still built and the report says so.

        Existing user decisions are carried over by warning id, so rebuilding never discards a
        human choice.
        """
        llm_status = "not_requested"
        if llm_notes is None and use_llm and self._reasoning is not None:
            context = self.get_business_context(experiment_id)
            if context is not None and not context.investigate_leakage:
                llm_status = "disabled_by_business_context"
            elif not self._reasoning.enabled:
                llm_status = "bedrock_disabled"
            else:
                try:
                    llm_notes = self._reasoning.leak_notes(experiment_id)
                    llm_status = "used"
                except AppError as error:
                    LOGGER.warning("Leak reasoning unavailable for %s: %s", experiment_id, error)
                    llm_status = "unavailable"
        elif llm_notes is not None:
            llm_status = "used"
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        eda = read_model_if_exists(self._store, layout.eda, EdaReport)
        if eda is None:
            raise ConflictError("EDA has not completed yet, so leakage cannot be investigated.")
        leakage = read_model_if_exists(self._store, layout.leakage, LeakageReport) or LeakageReport(
            experiment_id=experiment_id,
            generated_at=eda.generated_at,
            target_column=record.target_column,
        )
        report = build_pre_training(
            experiment_id=experiment_id,
            target_column=record.target_column,
            eda=eda,
            leakage=leakage,
            llm_notes=llm_notes,
        )
        if llm_status != "not_requested":
            report.sources = sorted({*report.sources, f"llm_reasoning:{llm_status}"})
        self._carry_decisions(layout.leakage_pre_training, report)
        write_model(self._store, layout.leakage_pre_training, report)
        return report

    def build_post_training_phase(
        self, experiment_id: str, *, ablation_deltas: dict[str, float] | None = None
    ) -> LeakageInvestigationReport:
        """Post-training warnings from importance, permutation importance and ablation.

        Ablation drops and permutation shares are read from the artifacts the SageMaker jobs
        wrote (``leakage/ablation.json``, ``models/<name>/permutation.json``) unless given.
        """
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        comparison = read_model_if_exists(self._store, layout.comparison, ComparisonReport)
        if comparison is None:
            raise ConflictError(
                "Training has not completed, so the post-training phase cannot run."
            )
        models = [
            metadata
            for name in {e.model_name for e in comparison.models}
            if (
                metadata := read_model_if_exists(
                    self._store, layout.model_metadata(name), ModelMetadata
                )
            )
        ]
        if ablation_deltas is None:
            ablation = read_model_if_exists(self._store, layout.ablation, AblationReport)
            ablation_deltas = (
                {e.feature: e.drop for e in ablation.entries if e.drop is not None}
                if ablation
                else None
            )
        permutation: dict[str, tuple[float, float]] = {}
        for metadata in models:
            result = read_model_if_exists(
                self._store, layout.model_permutation(metadata.model_name), PermutationImportance
            )
            for entry in result.entries if result else []:
                if entry.share > permutation.get(entry.feature, (0.0, 0.0))[0]:
                    permutation[entry.feature] = (entry.share, entry.mean_drop)
        # CV_Mode: the suspicious-performance rule evaluates the CV mean of the primary metric and
        # names the best fold, read from each Run's CV_Summary on the comparison (Req 10.5).
        cv_summaries = (
            {entry.model_name: entry.cv for entry in comparison.models if entry.cv is not None}
            if comparison.cross_validation is not None
            else None
        )
        report = build_post_training(
            experiment_id=experiment_id,
            target_column=record.target_column,
            models=models,
            ablation_deltas=ablation_deltas,
            permutation_shares=permutation,
            cv_summaries=cv_summaries or None,
            primary_metric=comparison.primary_metric,
        )
        self._carry_decisions(layout.leakage_post_training, report)
        write_model(self._store, layout.leakage_post_training, report)
        return report

    def get_phase(
        self, experiment_id: str, phase: InvestigationPhase
    ) -> LeakageInvestigationReport | None:
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        return read_model_if_exists(
            self._store, self._phase_uri(layout, phase), LeakageInvestigationReport
        )

    # --- user decisions -----------------------------------------------------------------------
    def decide_warning(
        self,
        experiment_id: str,
        *,
        phase: InvestigationPhase,
        warning_id: str,
        decision: UserDecision,
        user: str | None = None,
    ) -> LeakageInvestigationReport:
        """Record the user's choice on one warning. Nothing is auto-excluded (requirement 12.5)."""
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        report = read_model_if_exists(
            self._store, self._phase_uri(layout, phase), LeakageInvestigationReport
        )
        if report is None:
            raise NotFoundError(f"No {phase.value} leakage phase exists for this experiment.")
        warning = report.warning_for(warning_id)
        if warning is None:
            raise NotFoundError(f"No warning '{warning_id}' in the {phase.value} phase.")
        warning.user_decision = decision
        warning.review_status = (
            ReviewStatus.RESOLVED
            if decision is not UserDecision.PENDING
            else ReviewStatus.UNREVIEWED
        )
        warning.decided_by = user
        warning.decided_at = datetime.now(UTC)
        write_model(self._store, self._phase_uri(layout, phase), report)
        return report

    def excluded_features(self, experiment_id: str) -> list[str]:
        """Features the user decided to EXCLUDE across both phases. Advisory, not applied here."""
        excluded: set[str] = set()
        for phase in InvestigationPhase:
            report = self.get_phase(experiment_id, phase)
            if report is None:
                continue
            excluded.update(
                w.feature for w in report.warnings if w.user_decision is UserDecision.EXCLUDE
            )
        return sorted(excluded)

    def apply_exclusions_to_selection(
        self, experiment_id: str, *, user: str | None = None
    ) -> FeatureSelection | None:
        """Explicitly move EXCLUDE decisions into the feature selection.

        A separate call on purpose: a leakage decision is a warning verdict; changing what the
        model trains on is a selection change. Approval and selection stay distinct (req 12.6).
        """
        record = self._repository.get(experiment_id)
        if record.status in ACTIVE_TRAINING_STATUSES:
            raise ConflictError(
                "Training is running; change the feature selection once it has finished."
            )
        layout = ExperimentLayout(base=record.artifact_prefix)
        selection = read_model_if_exists(self._store, layout.selected_features, FeatureSelection)
        if selection is None:
            raise ConflictError("Decide the feature selection before applying leakage exclusions.")
        to_exclude = set(self.excluded_features(experiment_id))
        if not to_exclude:
            return selection
        kept = [f for f in selection.selected_features if f not in to_exclude]
        if not kept:
            raise ValidationError("Excluding these features would leave no features to train on.")
        reasons = dict(selection.exclusion_reasons)
        for feature in to_exclude:
            reasons[feature] = "Excluded by a leakage-investigation decision."
        updated = FeatureSelection(
            selected_features=kept,
            excluded_features=sorted(set(selection.excluded_features) | to_exclude),
            exclusion_reasons=reasons,
            decided_at=datetime.now(UTC),
            decided_by=user,
        )
        write_model(self._store, layout.selected_features, updated)
        return updated

    # --- helpers ------------------------------------------------------------------------------
    def _phase_uri(self, layout: ExperimentLayout, phase: InvestigationPhase) -> str:
        return (
            layout.leakage_pre_training
            if phase is InvestigationPhase.PRE_TRAINING
            else layout.leakage_post_training
        )

    def _carry_decisions(self, uri: str, report: LeakageInvestigationReport) -> None:
        previous = read_model_if_exists(self._store, uri, LeakageInvestigationReport)
        if previous is None:
            return
        prior = {w.id: w for w in previous.warnings}
        for warning in report.warnings:
            earlier = prior.get(warning.id)
            if earlier is not None and earlier.user_decision is not UserDecision.PENDING:
                warning.user_decision = earlier.user_decision
                warning.review_status = earlier.review_status
                warning.decided_by = earlier.decided_by
                warning.decided_at = earlier.decided_at
