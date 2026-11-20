"""Derived-feature proposals: generate, read, approve.

Three responsibilities, kept apart on purpose. A model *suggests*; validation decides what may
be offered; a person decides what is used. The service never promotes a suggestion on its own,
and the approved set is what the preparation job later computes — nothing else.

Ported from old_repo/backend/services/proposals.py for ML Factory 2.0: S3 is the only store, the
EDA/leakage artifacts are read straight from the experiment prefix, and the optional
``prediction_timing``/``target_definition`` hints fall back to the experiment's stored business
context when the caller does not supply them.
"""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from backend.config import Settings
from backend.errors import (
    ArtifactNotReadyError,
    ConflictError,
    FeatureDisabledError,
    UpstreamError,
    ValidationError,
)
from backend.repositories import ExperimentRepository
from backend.schemas import ApproveProposalsRequest, ProposeFeaturesRequest
from ml_engine.contracts.business_context import BusinessContext
from ml_engine.contracts.dictionary import DataDictionary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.contracts.proposals import (
    DerivedProfileReport,
    FeatureProposal,
    FeatureProposalReport,
    ProposalRejection,
    RejectedProposal,
)
from ml_engine.features import column_types_from_eda, validate_proposals
from ml_engine.io import ExperimentLayout, ObjectStore, read_model_if_exists, write_model
from ml_engine.reasoning import (
    ReasoningClient,
    ReasoningOutputError,
    ReasoningUnavailableError,
    propose_features,
)

LOGGER = logging.getLogger("ml_factory.services.proposals")


class FeatureProposalService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ExperimentRepository,
        store: ObjectStore,
        client: ReasoningClient | None,
        orchestrator: Any = None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store
        self._client = client
        self._orchestrator = orchestrator

    @property
    def enabled(self) -> bool:
        return self._settings.bedrock_enabled and self._client is not None

    def generate(
        self, experiment_id: str, request: ProposeFeaturesRequest
    ) -> FeatureProposalReport:
        """Ask the model for specifications and add them to the experiment's proposal set.

        Proposing again accumulates rather than replaces: the earlier candidates and their
        approvals are kept, and the newly suggested ones are appended. A new name that already
        belongs to an existing candidate or to a derived column a run has already built is
        refused with ``ALREADY_PROPOSED`` rather than silently shadowing the first.
        """
        if not self.enabled:
            raise FeatureDisabledError(
                "Feature proposals need Bedrock. Enable it and configure a model id to use them."
            )
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)

        eda = read_model_if_exists(self._store, layout.eda, EdaReport)
        if eda is None:
            raise ConflictError(
                "Feature proposals require the deterministic EDA to have completed first."
            )

        target_definition, prediction_timing = self._hints(layout, request)
        try:
            fresh = propose_features(
                self._client,
                experiment_id=experiment_id,
                eda=eda,
                target_column=record.target_column,
                leakage=read_model_if_exists(self._store, layout.leakage, LeakageReport),
                dictionary=read_model_if_exists(
                    self._store, layout.data_dictionary, DataDictionary
                ),
                target_definition=target_definition,
                prediction_timing=prediction_timing,
            )
        except (ReasoningUnavailableError, ReasoningOutputError) as error:
            raise UpstreamError(str(error)) from error

        existing = read_model_if_exists(
            self._store, layout.feature_proposals, FeatureProposalReport
        )
        report = self._accumulate(existing, fresh, layout)

        write_model(self._store, layout.feature_proposals, report)
        LOGGER.info(
            "Proposal set for %s now holds %d candidate(s); %d refused this round",
            experiment_id,
            len(report.candidates),
            len(fresh.rejected),
        )
        return report

    def define(
        self,
        experiment_id: str,
        specifications: list[FeatureProposal],
        *,
        approve: bool = True,
        user: str | None = None,
    ) -> "DefineResult":
        """Add derived features a person specified exactly. No model involved; Bedrock not needed.

        The specifications go through the same validation as model proposals — closed
        vocabulary, columns must exist with the right types, nothing may read the target, names
        must be new — so a hand-written feature gets no shortcut past the leakage rules.
        Because the person wrote it, ``approve`` (default on) also approves what was accepted,
        alongside anything approved earlier, which starts the profiling job.
        """
        if not specifications:
            raise ValidationError("Specify at least one derived feature.")
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        eda = read_model_if_exists(self._store, layout.eda, EdaReport)
        if eda is None:
            raise ConflictError("Derived features can be defined once EDA has completed.")

        accepted, refused = validate_proposals(
            specifications, column_types_from_eda(eda), target_column=record.target_column
        )
        fresh = FeatureProposalReport(
            experiment_id=experiment_id,
            generated_at=datetime.now(UTC),
            proposed_by=None,  # contract: null means written by hand
            candidates=accepted,
            rejected=refused,
        )
        existing = read_model_if_exists(
            self._store, layout.feature_proposals, FeatureProposalReport
        )
        before_names = {c.name for c in existing.candidates} if existing else set()
        before_refusals = len(existing.rejected) if existing else 0
        report = self._accumulate(existing, fresh, layout)
        write_model(self._store, layout.feature_proposals, report)

        added = [c.name for c in report.candidates if c.name not in before_names]
        rejected = report.rejected[before_refusals:]
        LOGGER.info(
            "User-defined features for %s: %d added, %d refused",
            experiment_id,
            len(added),
            len(rejected),
        )

        profiling = None
        if approve and added:
            approved = list(dict.fromkeys([*report.approved_names, *added]))
            result = self.approve(
                experiment_id, ApproveProposalsRequest(approved_names=approved), user=user
            )
            report, profiling = result.report, result.profiling_execution_id
        return DefineResult(
            report=report, added=added, rejected=rejected, profiling_execution_id=profiling
        )

    def _hints(
        self, layout: ExperimentLayout, request: ProposeFeaturesRequest
    ) -> tuple[str | None, str | None]:
        """Caller hints win; otherwise fall back to the stored business context (design #8)."""
        target_definition = request.target_definition
        prediction_timing = request.prediction_timing
        if target_definition and prediction_timing:
            return target_definition, prediction_timing
        context = read_model_if_exists(self._store, layout.business_context, BusinessContext)
        if context is not None:
            target_definition = target_definition or context.objective
            prediction_timing = prediction_timing or context.prediction_scenario
        return target_definition, prediction_timing

    def _accumulate(
        self,
        existing: FeatureProposalReport | None,
        fresh: FeatureProposalReport,
        layout: ExperimentLayout,
    ) -> FeatureProposalReport:
        """Merge a fresh round of proposals into the experiment's existing set.

        The first round has no existing report, so it simply becomes the set. On a later round
        the earlier candidates, approvals and refusals are preserved, and each new candidate is
        kept only if its name is still free — otherwise it joins the refusals as
        ``ALREADY_PROPOSED``, so re-proposing never shadows an earlier approval.
        """
        if existing is None:
            return fresh

        built = self._built_derived_names(layout)
        claimed = {candidate.name for candidate in existing.candidates} | built

        candidates: list[FeatureProposal] = list(existing.candidates)
        rejected: list[RejectedProposal] = list(existing.rejected)
        rejected.extend(fresh.rejected)
        for candidate in fresh.candidates:
            if candidate.name in claimed:
                rejected.append(
                    RejectedProposal(
                        name=candidate.name,
                        op=candidate.op,
                        reason=ProposalRejection.ALREADY_PROPOSED,
                        message=(
                            f"'{candidate.name}' was already proposed for this experiment; the "
                            "earlier candidate is kept and this duplicate is dropped."
                        ),
                    )
                )
                continue
            claimed.add(candidate.name)
            candidates.append(candidate)

        # The approvals belong to the person who made them, not to the latest model round.
        return existing.model_copy(
            update={
                "generated_at": fresh.generated_at,
                # A hand-written round has no model id; keep the last model's, if any.
                "proposed_by": fresh.proposed_by or existing.proposed_by,
                "candidates": candidates,
                "rejected": rejected,
            }
        )

    def _built_derived_names(self, layout: ExperimentLayout) -> set[str]:
        """Names of derived columns a training run has already materialized, if any."""
        preparation = read_model_if_exists(self._store, layout.preparation, PreparationReport)
        if preparation is None:
            return set()
        return {feature.name for feature in preparation.derived_features}

    def get(self, experiment_id: str) -> FeatureProposalReport:
        report = self.stored(experiment_id)
        if report is None:
            raise ArtifactNotReadyError("No derived features have been proposed yet.")
        return report

    def stored(self, experiment_id: str) -> FeatureProposalReport | None:
        """The report if it exists. Used by training start, where absence is not an error."""
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        return read_model_if_exists(self._store, layout.feature_proposals, FeatureProposalReport)

    def approve(
        self, experiment_id: str, request: ApproveProposalsRequest, user: str | None = None
    ) -> "ApprovalResult":
        """Record exactly which candidates a person approved. Approving none is a valid answer.

        Returns at once. When something was approved, profiling runs as a SageMaker job and its
        results appear in feature review when it finishes.
        """
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)
        report = read_model_if_exists(self._store, layout.feature_proposals, FeatureProposalReport)
        if report is None:
            raise ArtifactNotReadyError("No derived features have been proposed yet.")

        offered = {candidate.name for candidate in report.candidates}
        unknown = [name for name in request.approved_names if name not in offered]
        if unknown:
            raise ValidationError(
                "Only proposals that were offered for this experiment can be approved.",
                {"unknown_proposals": unknown},
            )

        report.approved_names = list(dict.fromkeys(request.approved_names))
        report.decided_at = datetime.now(UTC)
        report.decided_by = user
        write_model(self._store, layout.feature_proposals, report)
        LOGGER.info(
            "Approved %d of %d proposed features for %s",
            len(report.approved_names),
            len(report.candidates),
            experiment_id,
        )

        # Approving is a decision to build. The approved columns are profiled and leakage-
        # screened over the full dataset so feature review can show real statistics — but in a
        # SageMaker Processing job, never here: the control plane does not load datasets.
        # Earlier profiles are cleared first so a changed approval never shows stale rows.
        write_model(
            self._store,
            layout.derived_profiles,
            DerivedProfileReport(
                experiment_id=record.experiment_id,
                generated_at=datetime.now(UTC),
                target_column=record.target_column,
                profiles=[],
            ),
        )
        profiling: str | None = None
        if report.approved and self._orchestrator is not None:
            try:
                profiling = self._orchestrator.start_derived_profiling(record).execution_id
            except Exception:
                # The approval is recorded either way, and preparation computes and screens the
                # features again at training time, so a failed start loses nothing.
                LOGGER.exception("Could not start derived-feature profiling for %s", experiment_id)
        return ApprovalResult(report=report, profiling_execution_id=profiling)


@dataclass(frozen=True, slots=True)
class ApprovalResult:
    """An approval as recorded, and the profiling job it started (None when none was needed or
    the job could not be started)."""

    report: FeatureProposalReport
    profiling_execution_id: str | None


@dataclass(frozen=True, slots=True)
class DefineResult:
    """What a hand-written definition did: which features were added, which were refused and
    why, and the profiling job started when they were approved."""

    report: FeatureProposalReport
    added: list[str]
    rejected: list[RejectedProposal]
    profiling_execution_id: str | None
