"""LLM interpretation and leak reasoning, behind the governance gate (requirements 12, 15.5, 16).

Every Bedrock call made on behalf of an experiment goes through ``_invoke`` here, which:

1. builds the bounded, data-free context from deterministic artifacts (inherited
   ``ml_engine.reasoning.context.build_context``) plus the stored business context;
2. applies ``backend.governance.apply_governance`` — the configured caps on sample values —
   so the governance setting actually controls what reaches Bedrock;
3. validates the response against the ``ReasoningReport`` schema and drops every reference to
   a column that is not in the context (inherited analyzer guarantees).

The inherited ``run_reasoning`` does steps 1 and 3 in one call with no hook for step 2, so this
service composes the same inherited pieces with the gate in between rather than editing the
manifest-protected analyzer.

The LLM narrates; it never computes, ranks, changes a selection or starts a job.
"""

from __future__ import annotations

import logging
from typing import Any

from backend.config import Settings
from backend.errors import ConflictError, FeatureDisabledError, UpstreamError
from backend.governance import GovernancePolicy, apply_governance
from backend.repositories import ExperimentRepository
from ml_engine.contracts.business_context import BusinessContext
from ml_engine.contracts.comparison import ExperimentSummary
from ml_engine.contracts.dictionary import DataDictionary
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.reasoning import ReasoningReport
from ml_engine.io import ExperimentLayout, ObjectStore, read_model_if_exists, write_model
from ml_engine.reasoning import ReasoningClient, ReasoningOutputError, ReasoningUnavailableError
from ml_engine.reasoning.analyzer import _drop_unknown_columns, _validate
from ml_engine.reasoning.context import build_context, referenced_columns
from ml_engine.reasoning.cv_context import build_cross_validation_context
from ml_engine.reasoning.json_io import parse_json_object
from ml_engine.reasoning.numbers import drop_unverified_numbers
from ml_engine.reasoning.prompts import SYSTEM_PROMPT, build_user_prompt

LOGGER = logging.getLogger("ml_factory.services.reasoning")

LEAK_QUESTION = (
    "Focus on leakage. Using the business context (objective and prediction scenario), judge "
    "for each feature whether it would be known at the moment the prediction is made. Report "
    "suspected leaks in 'semantic_leakage' with honest confidence and the evidence basis."
)


def _filter_numeric_claims(report: ReasoningReport, context: dict[str, Any]) -> ReasoningReport:
    """Run ``drop_unverified_numbers`` over the LLM's free-text fields only.

    The deterministic fields the analyzer already governs — feature names, severities, the
    confidence qualifier, enum actions — are left untouched; only prose the model wrote freely
    is filtered, so a fabricated metric value in a sentence is removed while number-free prose
    and correctly cited numbers survive.
    """

    def clean(value: str | None) -> str | None:
        if value is None:
            return None
        return drop_unverified_numbers(value, context)

    def clean_lines(items: list[str]) -> list[str]:
        return [filtered for item in items if (filtered := drop_unverified_numbers(item, context))]

    report.executive_summary = clean(report.executive_summary) or ""
    report.model_behaviour_narrative = clean(report.model_behaviour_narrative)
    report.assumptions = clean_lines(report.assumptions)
    report.limitations = clean_lines(report.limitations)
    for finding in report.semantic_leakage:
        finding.reasoning = drop_unverified_numbers(finding.reasoning, context)
    for interpretation in report.feature_interpretations:
        interpretation.interpretation = drop_unverified_numbers(
            interpretation.interpretation, context
        )
        interpretation.concern = clean(interpretation.concern)
    for concern in report.data_quality_concerns:
        concern.explanation = drop_unverified_numbers(concern.explanation, context)
    for proposal in report.proposed_experiments:
        proposal.hypothesis = drop_unverified_numbers(proposal.hypothesis, context)
        proposal.expected_insight = drop_unverified_numbers(proposal.expected_insight, context)
        proposal.proposed_changes = clean_lines(proposal.proposed_changes)
    return report


class ReasoningService:
    def __init__(
        self,
        *,
        settings: Settings,
        repository: ExperimentRepository,
        store: ObjectStore,
        client: ReasoningClient | None,
        governance: GovernancePolicy,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store
        self._client = client
        self._governance = governance

    @property
    def enabled(self) -> bool:
        return self._settings.bedrock_enabled and self._client is not None

    # --- public -------------------------------------------------------------------------------
    def interpret(self, experiment_id: str, *, question: str | None = None) -> ReasoningReport:
        """Narrate the finished experiment: results, leaks, data quality, follow-ups."""
        layout = self._layout(experiment_id)
        summary = read_model_if_exists(self._store, layout.summary, ExperimentSummary)
        report = self._invoke(layout, experiment_id, summary=summary, question=question)
        write_model(self._store, layout.reasoning, report)
        return report

    def leak_notes(self, experiment_id: str) -> dict[str, str]:
        """Pre-training leak suspicions per feature, as text for the investigation warnings.

        Stored as ``reasoning/leakage_reasoning.json`` for traceability. Each note states the
        model's confidence and evidence basis so a reader can see it is suspicion, not proof.
        """
        layout = self._layout(experiment_id)
        report = self._invoke(layout, experiment_id, summary=None, question=LEAK_QUESTION)
        write_model(self._store, layout.leakage_reasoning, report)
        notes: dict[str, str] = {}
        for finding in report.semantic_leakage:
            basis = "; ".join(finding.evidence_basis) or "not stated"
            notes[finding.feature] = (
                f"[LLM suspicion, {finding.confidence} confidence] {finding.reasoning} "
                f"(evidence basis: {basis})"
            )
        return notes

    def stored(self, experiment_id: str) -> ReasoningReport | None:
        return read_model_if_exists(
            self._store, self._layout(experiment_id).reasoning, ReasoningReport
        )

    # --- internals ----------------------------------------------------------------------------
    def _invoke(
        self,
        layout: ExperimentLayout,
        experiment_id: str,
        *,
        summary: ExperimentSummary | None,
        question: str | None,
    ) -> ReasoningReport:
        if not self.enabled:
            raise FeatureDisabledError(
                "LLM reasoning needs Bedrock. Enable it and configure a model id to use it."
            )
        eda = read_model_if_exists(self._store, layout.eda, EdaReport)
        if eda is None:
            raise ConflictError("EDA has not completed, so there is nothing to reason about.")
        context = self.governed_context(layout, eda=eda, summary=summary)
        try:
            raw = self._client.invoke(SYSTEM_PROMPT, build_user_prompt(context, question))
            payload = parse_json_object(raw)
            report = _validate(payload, experiment_id=experiment_id, model_id=self._client.model_id)
        except (ReasoningUnavailableError, ReasoningOutputError) as error:
            raise UpstreamError(str(error)) from error
        report = _drop_unknown_columns(report, referenced_columns(context))
        # Drop any free-text sentence citing a number that is not in the governed context, at the
        # sentence's stated precision (Req 11.5; applies in all modes per D14). Only the LLM's
        # narrative/summary prose is filtered — deterministic fields are never touched.
        report = _filter_numeric_claims(report, context)
        inputs = ["eda"]
        if "deterministic_leakage_findings" in context:
            inputs.append("leakage")
        if "business_context" in context:
            inputs.append("business_context")
        if "model_results" in context:
            inputs.append("summary")
        report.input_artifacts = inputs
        return report

    def governed_context(
        self,
        layout: ExperimentLayout,
        *,
        eda: EdaReport,
        summary: ExperimentSummary | None,
    ) -> dict[str, Any]:
        """The exact document sent to Bedrock, after the governance gate."""
        business = read_model_if_exists(self._store, layout.business_context, BusinessContext)
        context = build_context(
            eda=eda,
            leakage=read_model_if_exists(self._store, layout.leakage, LeakageReport),
            dictionary=read_model_if_exists(self._store, layout.data_dictionary, DataDictionary),
            summary=summary,
            target_definition=business.objective if business else None,
            prediction_timing=business.prediction_scenario if business else None,
        )
        if business is not None:
            context["business_context"] = business.as_reasoning_context()
        # In CV_Mode the comparison carries CV summaries and a Test_Evaluation; project the
        # numbers the LLM may cite into a bounded, data-free block (Req 11.4). None in
        # single-split mode, so the key is simply absent.
        if summary is not None and summary.comparison is not None:
            cross_validation = build_cross_validation_context(summary.comparison)
            if cross_validation is not None:
                context["cross_validation"] = cross_validation
        return apply_governance(context, self._governance)

    def _layout(self, experiment_id: str) -> ExperimentLayout:
        return ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
