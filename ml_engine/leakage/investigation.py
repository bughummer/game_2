"""Build human-controlled leakage warnings from deterministic outputs (requirement 12).

Additive to the inherited leakage engine: this module *reads* an existing ``LeakageReport`` and
``EdaReport`` and translates them into ``LeakageWarning`` objects. It computes no new statistic —
every number it records already came from the deterministic engine. It never changes a rule.

Two phase builders:

* ``build_pre_training`` — name patterns + per-feature target associations + the deterministic
  rules + (optional) business-context LLM reasoning.
* ``build_post_training`` — feature importance + suspicious single-feature dominance + (optional)
  ablation deltas, from the trained models' metadata and comparison.

The association rule is explicit: a high association raises a warning but never sets
``leakage_established``. Only a deterministic rule that proves duplication/temporal leakage, or a
confirmed ablation collapse, may establish leakage.
"""

import re
from datetime import UTC, datetime

from ml_engine.contracts.common import LeakageRiskLevel, Severity
from ml_engine.contracts.cv import CvRunSummary
from ml_engine.contracts.eda import EdaReport, TargetAssociation
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.leakage_investigation import (
    Confidence,
    EvidenceKind,
    InvestigationPhase,
    LeakageInvestigationReport,
    LeakageWarning,
)
from ml_engine.contracts.model import ModelMetadata

#: Association at or above this is worth a warning. It is a screening threshold, not proof.
ASSOCIATION_WARNING = 0.7
ASSOCIATION_STRONG = 0.9
#: A column carrying this share of the total permutation drop dominates the model.
PERMUTATION_DOMINANT_SHARE = 0.6

#: A primary metric this close to perfect is almost always leakage, not a good model. Matches
#: ``ml_engine.reporting.comparison.SUSPICIOUSLY_PERFECT``; in CV_Mode it is checked against the
#: CV mean of the primary metric (design C7, Req 10.5).
SUSPICIOUS_PERFORMANCE = 0.999
#: Primary metrics bounded in [0, 1] where "near 1.0" is meaningful. Mirrors the comparison rule.
_BOUNDED_METRICS = frozenset(
    {"roc_auc", "pr_auc", "accuracy", "f1", "macro_f1", "weighted_f1", "r2"}
)

#: Column-name fragments that often mark a post-outcome or target-derived field. Suspicion only.
# Case-insensitive substring tokens that often mark a post-outcome or target-derived field.
# Matched on the lowercased, token-split name so "future_balance" and "balance_after" both hit.
_SUSPICIOUS_TOKENS = (
    "target",
    "label",
    "outcome",
    "result",
    "prediction",
    "predicted",
    "proba",
    "future",
    "post",
    "after",
    "resolved",
    "settled",
    "repaid",
    "approved",
    "rejected",
    "next",
)


def _name_is_suspicious(name: str) -> bool:
    tokens = set(re.split(r"[^a-z0-9]+", name.lower()))
    return bool(tokens & set(_SUSPICIOUS_TOKENS))


def _warning_id(phase: InvestigationPhase, feature: str, method: EvidenceKind) -> str:
    return f"{phase.value}:{feature}:{method.value}"


def _confidence_from_score(score: float) -> Confidence:
    if score >= ASSOCIATION_STRONG:
        return Confidence.HIGH
    if score >= ASSOCIATION_WARNING:
        return Confidence.MEDIUM
    return Confidence.LOW


def build_pre_training(
    *,
    experiment_id: str,
    target_column: str,
    eda: EdaReport,
    leakage: LeakageReport,
    llm_notes: dict[str, str] | None = None,
) -> LeakageInvestigationReport:
    """Pre-training warnings. ``llm_notes`` maps feature -> suspicion text (optional, Bedrock)."""
    llm_notes = llm_notes or {}
    warnings: list[LeakageWarning] = []
    sources: list[str] = []

    # 1. Deterministic rules — the authoritative signal; these may establish leakage.
    if leakage.findings:
        sources.append("deterministic_rules")
    for finding in leakage.findings:
        established = finding.risk_level is LeakageRiskLevel.CONFIRMED_DUPLICATE
        warnings.append(
            LeakageWarning(
                # The rule is part of the id: a feature may trip several rules, each its own
                # warning, so a non-establishing rule never overwrites an establishing one.
                id=f"{InvestigationPhase.PRE_TRAINING.value}:{finding.feature}:rule:{finding.rule}",
                phase=InvestigationPhase.PRE_TRAINING,
                feature=finding.feature,
                method=EvidenceKind.DETERMINISTIC_RULE,
                severity=finding.severity,
                risk_level=finding.risk_level,
                confidence=Confidence.HIGH if established else Confidence.MEDIUM,
                leakage_established=established,
                evidence=dict(finding.evidence),
                summary=finding.explanation,
                recommended_investigation=(
                    "Confirm this column is recorded only after the outcome, or is a copy of it."
                ),
                suggested_correction=(
                    "Exclude the column, or replace it with a prediction-time-safe version."
                ),
                llm_reasoning=llm_notes.get(finding.feature),
            )
        )

    # 2. Target associations — screening evidence of correlation, never proof of leakage.
    strong = [a for a in eda.target_associations if a.score >= ASSOCIATION_WARNING]
    if strong:
        sources.append("target_association")
    for association in strong:
        if association.feature == target_column:
            continue
        warnings.append(_association_warning(target_column, association, llm_notes))

    # 3. Name patterns — the weakest signal; suspicion only, low confidence.
    named = {w.feature for w in warnings}
    for profile in eda.columns:
        if profile.name == target_column or profile.name in named:
            continue
        if _name_is_suspicious(profile.name):
            warnings.append(
                LeakageWarning(
                    id=_warning_id(
                        InvestigationPhase.PRE_TRAINING, profile.name, EvidenceKind.NAME_PATTERN
                    ),
                    phase=InvestigationPhase.PRE_TRAINING,
                    feature=profile.name,
                    method=EvidenceKind.NAME_PATTERN,
                    severity=Severity.LOW,
                    risk_level=LeakageRiskLevel.REQUIRES_REVIEW,
                    confidence=Confidence.LOW,
                    leakage_established=False,
                    evidence={"column_name": profile.name},
                    summary=(
                        f"The column name '{profile.name}' resembles an outcome or target-derived "
                        "field. This is a naming hint only, not evidence."
                    ),
                    recommended_investigation=(
                        "Check what this column records and when it becomes known."
                    ),
                    llm_reasoning=llm_notes.get(profile.name),
                )
            )
        elif profile.name in llm_notes:
            # The LLM raised a concern the deterministic signals did not — keep it, as suspicion.
            warnings.append(
                LeakageWarning(
                    id=_warning_id(
                        InvestigationPhase.PRE_TRAINING, profile.name, EvidenceKind.BUSINESS_CONTEXT
                    ),
                    phase=InvestigationPhase.PRE_TRAINING,
                    feature=profile.name,
                    method=EvidenceKind.BUSINESS_CONTEXT,
                    severity=Severity.LOW,
                    risk_level=LeakageRiskLevel.REQUIRES_REVIEW,
                    confidence=Confidence.LOW,
                    leakage_established=False,
                    evidence={},
                    summary="Flagged by business-context reasoning for review.",
                    llm_reasoning=llm_notes[profile.name],
                )
            )
            sources.append("business_context")

    return LeakageInvestigationReport(
        experiment_id=experiment_id,
        phase=InvestigationPhase.PRE_TRAINING,
        generated_at=datetime.now(UTC),
        target_column=target_column,
        warnings=warnings,
        sources=sorted(set(sources)),
    )


def _association_warning(
    target_column: str, association: TargetAssociation, llm_notes: dict[str, str]
) -> LeakageWarning:
    strong = association.score >= ASSOCIATION_STRONG
    return LeakageWarning(
        id=_warning_id(
            InvestigationPhase.PRE_TRAINING, association.feature, EvidenceKind.TARGET_ASSOCIATION
        ),
        phase=InvestigationPhase.PRE_TRAINING,
        feature=association.feature,
        method=EvidenceKind.TARGET_ASSOCIATION,
        severity=Severity.HIGH if strong else Severity.MEDIUM,
        risk_level=LeakageRiskLevel.POTENTIAL_LEAKAGE
        if strong
        else LeakageRiskLevel.REQUIRES_REVIEW,
        confidence=_confidence_from_score(association.score),
        leakage_established=False,  # association is never proof (requirement 12.3)
        evidence={
            "association_score": round(association.score, 6),
            "method": association.method,
            "non_null_count": association.non_null_count,
        },
        summary=(
            f"'{association.feature}' alone predicts '{target_column}' very strongly "
            f"({association.method}={association.score:.3f}). A strong lone predictor is the "
            "loudest sign of a leak, but high association is not proof of one."
        ),
        recommended_investigation=(
            "Confirm this column is available at prediction time and not derived from the outcome."
        ),
        suggested_correction="If it is only known after the outcome, exclude it.",
        llm_reasoning=llm_notes.get(association.feature),
    )


def build_post_training(
    *,
    experiment_id: str,
    target_column: str,
    models: list[ModelMetadata],
    ablation_deltas: dict[str, float] | None = None,
    permutation_shares: dict[str, tuple[float, float]] | None = None,
    cv_summaries: dict[str, CvRunSummary] | None = None,
    primary_metric: str | None = None,
) -> LeakageInvestigationReport:
    """Post-training warnings from feature importance, permutation importance and ablation.

    ``permutation_shares`` maps feature -> (share of the summed permutation drop, mean drop),
    the strongest value across models. A single column carrying most of the drop is the same
    fingerprint as importance dominance, measured model-agnostically on raw input columns.

    ``ablation_deltas`` maps feature -> drop in the primary metric when that feature is removed
    (computed on identical validation folds by the caller). A collapse without the feature is
    strong evidence the model leaned on a leak.

    ``cv_summaries`` (CV_Mode only) maps model name -> its CV_Summary. The suspicious-performance
    rule then uses the CV mean of the primary metric and the warning names the best (max) fold
    value, so a model that only looks perfect on one lucky fold is described honestly (Req 10.5).
    """
    warnings: list[LeakageWarning] = []
    sources: list[str] = []

    if cv_summaries and primary_metric in _BOUNDED_METRICS:
        performance = _cv_suspicious_performance(target_column, cv_summaries, primary_metric)
        if performance:
            sources.append("suspicious_performance")
            warnings.extend(performance)

    dominance = _single_feature_dominance(models)
    if dominance:
        sources.append("feature_importance")
    for feature, share in dominance.items():
        if feature == target_column:
            continue
        warnings.append(
            LeakageWarning(
                id=_warning_id(
                    InvestigationPhase.POST_TRAINING, feature, EvidenceKind.FEATURE_IMPORTANCE
                ),
                phase=InvestigationPhase.POST_TRAINING,
                feature=feature,
                method=EvidenceKind.FEATURE_IMPORTANCE,
                severity=Severity.HIGH if share >= 0.8 else Severity.MEDIUM,
                risk_level=LeakageRiskLevel.POTENTIAL_LEAKAGE,
                confidence=Confidence.HIGH if share >= 0.8 else Confidence.MEDIUM,
                leakage_established=False,
                evidence={"importance_share": round(share, 6)},
                summary=(
                    f"'{feature}' carries {share:.0%} of the model's importance. A single feature "
                    "dominating the model this strongly can indicate a leak."
                ),
                recommended_investigation=(
                    "Train without this feature and compare the metric (ablation)."
                ),
                suggested_correction=(
                    "If the metric collapses only when this feature is removed, exclude it."
                ),
            )
        )

    for feature, (share, drop) in (permutation_shares or {}).items():
        if feature == target_column or share < PERMUTATION_DOMINANT_SHARE:
            continue
        sources.append("permutation_importance")
        strong = share >= 0.8
        warnings.append(
            LeakageWarning(
                id=_warning_id(
                    InvestigationPhase.POST_TRAINING, feature, EvidenceKind.PERMUTATION_IMPORTANCE
                ),
                phase=InvestigationPhase.POST_TRAINING,
                feature=feature,
                method=EvidenceKind.PERMUTATION_IMPORTANCE,
                severity=Severity.HIGH if strong else Severity.MEDIUM,
                risk_level=LeakageRiskLevel.POTENTIAL_LEAKAGE,
                confidence=Confidence.HIGH if strong else Confidence.MEDIUM,
                leakage_established=False,
                evidence={"permutation_share": round(share, 6), "mean_metric_drop": drop},
                summary=(
                    f"Shuffling '{feature}' accounts for {share:.0%} of the validation-score drop "
                    "across all columns: the model relies on it far more than on anything else."
                ),
                recommended_investigation=(
                    "Run an ablation (retrain without it) and check when its value is recorded."
                ),
                suggested_correction=(
                    "If it is only known after the outcome, exclude it and retrain."
                ),
            )
        )

    for feature, delta in (ablation_deltas or {}).items():
        sources.append("ablation")
        established = delta >= 0.25  # removing it collapses the metric: strong leak evidence
        warnings.append(
            LeakageWarning(
                id=_warning_id(InvestigationPhase.POST_TRAINING, feature, EvidenceKind.ABLATION),
                phase=InvestigationPhase.POST_TRAINING,
                feature=feature,
                method=EvidenceKind.ABLATION,
                severity=Severity.CRITICAL if established else Severity.MEDIUM,
                risk_level=(
                    LeakageRiskLevel.POTENTIAL_LEAKAGE
                    if established
                    else LeakageRiskLevel.REQUIRES_REVIEW
                ),
                confidence=Confidence.HIGH if established else Confidence.MEDIUM,
                leakage_established=established,
                evidence={"primary_metric_drop_without_feature": round(delta, 6)},
                summary=(
                    f"Removing '{feature}' changes the primary metric by {delta:.3f} on identical "
                    "validation folds."
                ),
                recommended_investigation="Confirm the dependence is legitimate, not a leak.",
                suggested_correction=(
                    "If performance depends almost entirely on this feature, exclude it."
                ),
            )
        )

    return LeakageInvestigationReport(
        experiment_id=experiment_id,
        phase=InvestigationPhase.POST_TRAINING,
        generated_at=datetime.now(UTC),
        target_column=target_column,
        warnings=warnings,
        sources=sorted(set(sources)),
    )


def _cv_suspicious_performance(
    target_column: str, cv_summaries: dict[str, CvRunSummary], primary_metric: str
) -> list[LeakageWarning]:
    """One warning per Run whose CV mean of the primary metric is near-perfect (Req 10.5).

    The warning uses the CV mean (not a single fold) and names the best (max) fold value, so the
    reader sees both the headline estimate and the most optimistic fold.
    """
    warnings: list[LeakageWarning] = []
    for model_name, summary in cv_summaries.items():
        aggregate = summary.metrics.get(primary_metric)
        if aggregate is None or aggregate.mean < SUSPICIOUS_PERFORMANCE:
            continue
        warnings.append(
            LeakageWarning(
                # The Run is the subject, not a feature; the id keys off the model name.
                id=_warning_id(
                    InvestigationPhase.POST_TRAINING,
                    model_name,
                    EvidenceKind.SUSPICIOUS_PERFORMANCE,
                ),
                phase=InvestigationPhase.POST_TRAINING,
                feature=model_name,
                method=EvidenceKind.SUSPICIOUS_PERFORMANCE,
                severity=Severity.HIGH,
                risk_level=LeakageRiskLevel.POTENTIAL_LEAKAGE,
                confidence=Confidence.HIGH,
                leakage_established=False,
                evidence={
                    "cv_mean": round(aggregate.mean, 6),
                    "cv_max_fold": round(aggregate.max, 6),
                    "cv_min_fold": round(aggregate.min, 6),
                    "folds": aggregate.fold_count,
                    "metric": primary_metric,
                },
                summary=(
                    f"'{model_name}' reaches a cross-validated mean {primary_metric}="
                    f"{aggregate.mean:.4f} (best fold {aggregate.max:.4f}) across "
                    f"{aggregate.fold_count} folds. A cross-validated score this close to perfect "
                    f"nearly always means a feature encodes '{target_column}'."
                ),
                recommended_investigation=(
                    "Re-check the leakage findings; a near-perfect CV mean is the loudest sign of "
                    "a feature that encodes the outcome."
                ),
                suggested_correction=(
                    "Confirm every feature is available at prediction time; exclude any that is "
                    "derived from the outcome and retrain."
                ),
            )
        )
    return warnings


def _single_feature_dominance(models: list[ModelMetadata]) -> dict[str, float]:
    """The max importance share each feature reaches across models, where it exceeds a threshold."""
    dominance: dict[str, float] = {}
    for metadata in models:
        importance = metadata.feature_importance
        if importance is None or not importance.entries:
            continue
        top = max(importance.entries, key=lambda e: e.importance)
        if top.importance >= 0.6:
            dominance[top.feature] = max(dominance.get(top.feature, 0.0), top.importance)
    return dominance
