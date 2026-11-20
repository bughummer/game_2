"""Permutation importance per input column, on the validation fold (ML Factory 2.0).

Model-agnostic: shuffle one raw input column, push the frame through the *fitted* pipeline and
model, and measure the drop in the primary metric. Measured on raw input columns (before
one-hot encoding or date-part extraction) so the answer is about the columns a person selected.

Pure apart from the injected ``score`` callable; deterministic for a given seed.
"""

from collections.abc import Callable
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from ml_engine.contracts.common import MetricDirection
from ml_engine.contracts.explain import PermutationEntry, PermutationImportance

DEFAULT_REPEATS = 3
MAX_FEATURES = 60


def permutation_importance(
    frame: pd.DataFrame,
    features: list[str],
    score: Callable[[pd.DataFrame], float | None],
    *,
    experiment_id: str,
    model_name: str,
    primary_metric: str,
    direction: MetricDirection,
    repeats: int = DEFAULT_REPEATS,
    random_seed: int = 42,
    max_features: int = MAX_FEATURES,
) -> PermutationImportance | None:
    """Mean metric drop per feature when its values are shuffled. ``None`` if unscorable."""
    baseline = score(frame)
    if baseline is None or not np.isfinite(baseline):
        return None
    measured = [f for f in features if f in frame.columns][:max_features]
    sign = 1.0 if direction is MetricDirection.MAXIMIZE else -1.0
    rng = np.random.default_rng(random_seed)
    drops: dict[str, list[float]] = {}
    for feature in measured:
        values: list[float] = []
        for _ in range(max(1, repeats)):
            shuffled = frame.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            permuted = score(shuffled)
            if permuted is None or not np.isfinite(permuted):
                continue
            values.append(sign * (baseline - permuted))
        if values:
            drops[feature] = values

    positive_total = sum(max(0.0, float(np.mean(v))) for v in drops.values())
    entries = [
        PermutationEntry(
            feature=feature,
            mean_drop=round(float(np.mean(values)), 6),
            std_drop=round(float(np.std(values)), 6),
            share=round(max(0.0, float(np.mean(values))) / positive_total, 6)
            if positive_total > 0
            else 0.0,
        )
        for feature, values in drops.items()
    ]
    entries.sort(key=lambda e: e.mean_drop, reverse=True)
    return PermutationImportance(
        experiment_id=experiment_id,
        model_name=model_name,
        generated_at=datetime.now(UTC),
        primary_metric=primary_metric,
        direction=direction,
        baseline_score=round(float(baseline), 6),
        repeats=max(1, repeats),
        random_seed=random_seed,
        entries=entries,
        truncated_features=max(0, len([f for f in features if f in frame.columns]) - max_features),
    )
