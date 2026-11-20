"""Decision-threshold analysis for a binary classifier (ML Factory 2.0).

The headline precision/recall/F1 are measured at a fixed 0.5 cut-off. With class weighting the
predicted probabilities shift, so 0.5 is often not where the model works best. This contract
records, from the validation fold, how precision/recall/F1 move with the threshold and where
F1 peaks — every number computed deterministically, so a reader (or an LLM) only reads it.

A threshold chosen on the validation fold and scored on that same fold is optimistic. The
report says so; confirm the chosen threshold on fresh data before relying on it.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import StrictModel
from ml_engine.contracts.cv import is_unset

THRESHOLD_SCHEMA_VERSION = "1.0"


class ThresholdPoint(StrictModel):
    threshold: float
    precision: float
    recall: float
    f1: float
    predicted_positive_rate: float = Field(
        description="Share of validation rows predicted positive at this threshold."
    )


class ThresholdAnalysis(StrictModel):
    schema_version: str = THRESHOLD_SCHEMA_VERSION
    experiment_id: str
    model_name: str
    generated_at: datetime
    positive_class: str
    validation_row_count: int
    positive_rate: float = Field(description="Share of validation rows in the positive class.")
    default: ThresholdPoint = Field(description="Metrics at the fixed 0.5 threshold.")
    best_f1: ThresholdPoint = Field(description="The threshold that maximises validation F1.")
    curve: list[ThresholdPoint] = Field(
        default_factory=list, description="Metrics on a fixed grid of thresholds (0.05 steps)."
    )
    note: str = (
        "Chosen and scored on the same validation fold, so best_f1 is optimistic. "
        "Confirm on fresh data before using it in production."
    )
    source: str | None = Field(
        default=None,
        exclude_if=is_unset,
        description="'oof' in CV_Mode (out-of-fold scores); unset means the validation fold.",
    )
