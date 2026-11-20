"""Derive the approved columns and screen them, in one place both callers share.

Two callers need the same thing. The preparation job computes the approved derived columns on
the training frame and screens them for leakage before fitting (``jobs/preprocessing``). The
control plane does the same the moment a person approves a proposal, so feature review can show
populated statistics without waiting for a training run (``backend/services/proposals``).

Both are leak-safe for the same reason: every operation in the vocabulary is row-wise, so a
derived value depends only on its own row, and *nothing is fitted here*. No imputer, scaler or
encoder touches these columns — that still happens only in ``ml_engine/preprocessing`` on the
train fold at training time. Profiling an unfitted, row-wise column early moves no information
across the split (architecture.md AD-14).

Keeping the body here rather than duplicated in the job means the two paths cannot drift: the
subset that is profiled, the way leakage is screened and the columns that are read are defined
once.
"""

from collections.abc import Sequence
from datetime import UTC, datetime

import pandas as pd

from ml_engine.contracts.common import (
    LeakageRiskLevel,
    ProblemType,
    RecommendedAction,
    SemanticType,
)
from ml_engine.contracts.eda import EdaReport
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.contracts.proposals import (
    DerivedColumnProfile,
    DerivedFeature,
    DerivedFeatureStatus,
    DerivedProfileReport,
    FeatureProposal,
)
from ml_engine.features.derive import apply_proposals
from ml_engine.leakage import analyze_leakage
from ml_engine.profiling import profile_dataset


def derive_and_screen(
    frame: pd.DataFrame,
    proposals: Sequence[FeatureProposal],
    *,
    experiment_id: str,
    target_column: str,
    problem_type: ProblemType,
    source_uri: str,
    file_format: str,
) -> tuple[pd.DataFrame, list[DerivedFeature], EdaReport | None, LeakageReport | None]:
    """Apply the (already validated) proposals and profile + leakage-screen what they produced.

    The screening runs over the derived columns and the target only: the original columns were
    profiled by the EDA job already, so re-reading them here would cost the whole dataset for no
    new information. Returns the frame with the derived columns added, the ``DerivedFeature``
    records, and the EDA/leakage reports for the derived subset (both ``None`` when nothing was
    derived).

    This function fits nothing. ``profile_dataset`` and ``analyze_leakage`` are pure, read-only
    computations over the derived subset — the leak-free guarantee holds because the derivation
    itself is row-wise, not because of anything done here.
    """
    frame, derived = apply_proposals(frame, proposals)
    if not derived:
        return frame, derived, None, None

    columns = [feature.name for feature in derived]
    subset = frame[[*columns, target_column]]
    subset_eda = profile_dataset(
        subset,
        experiment_id=experiment_id,
        target_column=target_column,
        source_uri=source_uri,
        file_format=file_format,
    )
    leakage = analyze_leakage(
        subset,
        experiment_id=experiment_id,
        target_column=target_column,
        problem_type=problem_type,
        eda=subset_eda,
    )
    return frame, derived, subset_eda, leakage


def build_derived_profiles(
    derived: Sequence[DerivedFeature],
    subset_eda: EdaReport | None,
    leakage: LeakageReport | None,
    *,
    experiment_id: str,
    target_column: str,
    status: DerivedFeatureStatus = DerivedFeatureStatus.APPROVED,
) -> DerivedProfileReport:
    """Join the derived records with their profile and leakage verdict into one report.

    This is the small contract feature review reads: for each derived column its formula
    sources, the profile statistics the EDA computed over the derived subset, and the leakage
    verdict ``analyze_leakage`` reached — exactly the columns the originals show. The caller
    decides the ``status``; approving stores ``APPROVED`` and a run reads ``BUILT`` from the
    preparation report instead.
    """
    column_profiles = {}
    if subset_eda is not None:
        column_profiles = {profile.name: profile for profile in subset_eda.columns}

    profiles: list[DerivedColumnProfile] = []
    for feature in derived:
        profile = column_profiles.get(feature.name)
        risk = leakage.risk_for(feature.name) if leakage is not None else None
        profiles.append(
            DerivedColumnProfile(
                name=feature.name,
                op=feature.op,
                source_columns=list(feature.source_columns),
                rationale=feature.rationale,
                semantic_type=profile.semantic_type if profile else SemanticType.UNKNOWN,
                dtype=profile.dtype if profile else "",
                missing_percentage=profile.missing_percentage if profile else 0.0,
                unique_count=profile.unique_count if profile else 0,
                unique_percentage=profile.unique_percentage if profile else 0.0,
                is_constant=profile.is_constant if profile else False,
                is_high_cardinality=profile.is_high_cardinality if profile else False,
                leakage_risk=risk.risk_level if risk else LeakageRiskLevel.NONE,
                max_severity=risk.max_severity if risk else None,
                recommended_action=risk.recommended_action if risk else RecommendedAction.KEEP,
                reasons=list(risk.reasons) if risk else [],
                available_at_prediction_time=feature.available_at_prediction_time,
                status=status,
            )
        )

    return DerivedProfileReport(
        experiment_id=experiment_id,
        generated_at=datetime.now(UTC),
        target_column=target_column,
        profiles=profiles,
    )
