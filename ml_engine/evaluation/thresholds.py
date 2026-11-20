"""Precision/recall/F1 as a function of the decision threshold (ML Factory 2.0). Pure: no IO."""

from datetime import UTC, datetime

import numpy as np
from sklearn import metrics as skm

from ml_engine.contracts.thresholds import ThresholdAnalysis, ThresholdPoint

GRID = tuple(round(0.05 * step, 2) for step in range(1, 20))  # 0.05 … 0.95
DEFAULT_THRESHOLD = 0.5


def threshold_analysis(
    y_true: np.ndarray,
    scores: np.ndarray,
    *,
    experiment_id: str,
    model_name: str,
    positive_class: str,
) -> ThresholdAnalysis | None:
    """Analyse ``scores`` (positive-class probabilities) against binary ``y_true`` (0/1).

    Returns None when the analysis is meaningless: no rows, or only one class present.
    """
    truth = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(scores, dtype="float64")
    finite = np.isfinite(probabilities)
    truth, probabilities = truth[finite], probabilities[finite]
    if truth.size == 0 or truth.min() == truth.max():
        return None

    return ThresholdAnalysis(
        experiment_id=experiment_id,
        model_name=model_name,
        generated_at=datetime.now(UTC),
        positive_class=positive_class,
        validation_row_count=int(truth.size),
        positive_rate=round(float(truth.mean()), 6),
        default=_point(truth, probabilities, DEFAULT_THRESHOLD),
        best_f1=_point(truth, probabilities, _best_f1_threshold(truth, probabilities)),
        curve=[_point(truth, probabilities, t) for t in GRID],
    )


def point_at(truth: np.ndarray, probabilities: np.ndarray, threshold: float) -> ThresholdPoint:
    """Precision/recall/F1 of binary ``truth`` (0/1) at one fixed threshold."""
    return _point(
        np.asarray(truth, dtype=int), np.asarray(probabilities, dtype="float64"), threshold
    )


def _best_f1_threshold(truth: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = skm.precision_recall_curve(truth, probabilities)
    # The last precision/recall pair has no threshold (it is the "predict nothing" end).
    precision, recall = precision[:-1], recall[:-1]
    denominator = precision + recall
    f1 = np.divide(
        2 * precision * recall, denominator, out=np.zeros_like(denominator), where=denominator > 0
    )
    return float(thresholds[int(np.argmax(f1))])


def _point(truth: np.ndarray, probabilities: np.ndarray, threshold: float) -> ThresholdPoint:
    predicted = (probabilities >= threshold).astype(int)
    return ThresholdPoint(
        threshold=round(float(threshold), 6),
        precision=round(float(skm.precision_score(truth, predicted, zero_division=0)), 6),
        recall=round(float(skm.recall_score(truth, predicted, zero_division=0)), 6),
        f1=round(float(skm.f1_score(truth, predicted, zero_division=0)), 6),
        predicted_positive_rate=round(float(predicted.mean()), 6),
    )
