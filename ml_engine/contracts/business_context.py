"""Business context: the prediction problem in plain language, captured once (requirement 11).

This replaces the old, dangerous "when is the prediction made relative to the outcome?" question
(handover decision 8). A user describes the business scenario in their own words; the system does
not ask them to classify each column's availability timing. One record per experiment, stored at
``config/business_context.json`` and reused by feature engineering, leakage analysis and the
reasoning layer — so the same stated intent informs every judgement.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import RequestedProblemType, StrictModel

BUSINESS_CONTEXT_SCHEMA_VERSION = "1.0"


class BusinessContext(StrictModel):
    schema_version: str = BUSINESS_CONTEXT_SCHEMA_VERSION
    experiment_id: str
    created_at: datetime
    created_by: str | None = None

    objective: str = Field(
        description="What the user is trying to predict and why, in plain language."
    )
    prediction_scenario: str = Field(
        description=(
            "The business situation at the moment of prediction, in plain language — e.g. "
            "'predict default within 30 days using what is known when the loan is issued'. "
            "Replaces the per-column timing question (requirement 11.2)."
        )
    )
    target_column: str
    problem_type: RequestedProblemType = RequestedProblemType.AUTO
    primary_metric: str | None = None

    entity_id_column: str | None = Field(
        default=None, description="Column identifying the unit a row describes (customer, loan)."
    )
    scoring_trigger: str | None = Field(
        default=None,
        description="The event that would trigger a prediction, if the user named one.",
    )
    prediction_horizon: str | None = Field(
        default=None, description="How far ahead the prediction looks, in the user's words."
    )
    temporal_columns: list[str] = Field(
        default_factory=list,
        description="Columns the user flagged as time-related. Optional; never required.",
    )

    #: Leakage investigation defaults ON (requirement 11.3): the user is not required to know
    #: each column's availability timing for the system to look for leaks.
    investigate_leakage: bool = True

    def as_reasoning_context(self) -> dict[str, str]:
        """A compact, PII-free dict for the Bedrock prompt. Only the stated intent, no data."""
        fields = {
            "objective": self.objective,
            "prediction_scenario": self.prediction_scenario,
            "scoring_trigger": self.scoring_trigger or "",
            "prediction_horizon": self.prediction_horizon or "",
        }
        return {k: v for k, v in fields.items() if v}
