"""Leakage investigation: the human-controlled warning layer over deterministic findings.

This is the priority capability (requirement 12). It is strictly additive to the inherited
``ml_engine.contracts.leakage`` and ``ml_engine.contracts.reasoning`` — those keep their tested
behaviour; nothing here changes a rule or an association calculation.

A ``LeakageWarning`` is one finding a user acts on: it carries the deterministic method and
evidence, an optional LLM reasoning note (as suspicion, never proof), a severity and confidence,
a recommended investigation, a suggested correction, and the user's own decision and review
status. Findings are gathered in two phases that are persisted *separately* so a later phase
never overwrites an earlier one:

* pre-training  — names, associations, deterministic rules, business-context reasoning
* post-training — importance, suspicious performance, ablation, updated reasoning

Statistical association is distinguished from evidence of temporal/causal leakage throughout:
high correlation alone never sets ``leakage_established`` (requirement 12.3).
"""

from datetime import datetime
from enum import StrEnum

from pydantic import Field

from ml_engine.contracts.common import LeakageRiskLevel, Severity, StrictModel

LEAKAGE_INVESTIGATION_SCHEMA_VERSION = "1.0"


class InvestigationPhase(StrEnum):
    PRE_TRAINING = "pre_training"
    POST_TRAINING = "post_training"


class EvidenceKind(StrEnum):
    """What a warning rests on. Association is evidence of correlation, not of leakage."""

    NAME_PATTERN = "name_pattern"
    TARGET_ASSOCIATION = "target_association"
    DETERMINISTIC_RULE = "deterministic_rule"
    BUSINESS_CONTEXT = "business_context"
    FEATURE_IMPORTANCE = "feature_importance"
    SUSPICIOUS_PERFORMANCE = "suspicious_performance"
    ABLATION = "ablation"
    PERMUTATION_IMPORTANCE = "permutation_importance"


class Confidence(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class UserDecision(StrEnum):
    """What the user chose. ``PENDING`` until they act; nothing is auto-excluded (req 12.5)."""

    PENDING = "pending"
    KEEP = "keep"
    EXCLUDE = "exclude"
    INVESTIGATE = "investigate"
    ACCEPT_RISK = "accept_risk"
    COMPARE_WITH_AND_WITHOUT = "compare_with_and_without"


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    IN_REVIEW = "in_review"
    RESOLVED = "resolved"


class LeakageWarning(StrictModel):
    """One evidence-based, human-controlled leakage warning (requirement 12.4)."""

    id: str = Field(description="Stable id: phase + feature + method, so re-runs don't duplicate.")
    phase: InvestigationPhase
    feature: str
    method: EvidenceKind
    severity: Severity
    risk_level: LeakageRiskLevel
    confidence: Confidence
    #: True only when temporal/causal evidence establishes leakage. Association alone is False.
    leakage_established: bool = False
    evidence: dict[str, float | int | str | bool | None] = Field(
        default_factory=dict, description="The deterministic numbers behind this warning."
    )
    summary: str = Field(description="Plain-language statement of what is suspicious.")
    llm_reasoning: str | None = Field(
        default=None, description="LLM suspicion with its basis; never presented as proven fact."
    )
    recommended_investigation: str | None = None
    suggested_correction: str | None = None
    user_decision: UserDecision = UserDecision.PENDING
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    decided_by: str | None = None
    decided_at: datetime | None = None


class LeakageInvestigationReport(StrictModel):
    """One phase's warnings. Pre- and post-training phases are stored as separate artifacts."""

    schema_version: str = LEAKAGE_INVESTIGATION_SCHEMA_VERSION
    experiment_id: str
    phase: InvestigationPhase
    generated_at: datetime
    target_column: str
    warnings: list[LeakageWarning] = Field(default_factory=list)
    #: Rules/calculations that produced these warnings — traceability, not new computation.
    sources: list[str] = Field(default_factory=list)

    def warning_for(self, warning_id: str) -> LeakageWarning | None:
        return next((w for w in self.warnings if w.id == warning_id), None)

    def pending_count(self) -> int:
        return sum(1 for w in self.warnings if w.user_decision is UserDecision.PENDING)
