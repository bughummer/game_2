"""Per-feature association with the target.

A single, comparable "how strongly does this column alone predict the target" score per feature.
It exists for one purpose: a feature that predicts the target almost perfectly on its own is the
loudest signal of a data leak, especially when the column came from a feature store someone else
built and nobody can vouch for its prediction-time availability.

The score is deterministic and bounded in [0, 1], where higher means a stronger lone
relationship with the target:

* numeric feature vs numeric target     -> |Pearson correlation|
* numeric feature vs categorical target -> correlation ratio (eta), between-group variance share
* categorical feature vs any target     -> Cramer's V (categorical target) or eta (numeric target)

This is a screening heuristic, not an inference: it never decides anything, it is handed to the
reasoning layer and shown to the user as evidence. It computes nothing that scikit-learn would
not, and it fits no model — so running it before the train/validation split is safe, exactly as
the row-wise derived-feature computation is.
"""

import numpy as np
import pandas as pd
from pandas.api import types as pdt

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.eda import TargetAssociation

#: Columns with more distinct values than this are treated as identifiers for association:
#: a near-unique column trivially "predicts" anything and the score would be meaningless.
MAX_CATEGORICAL_LEVELS = 1000


def target_associations(
    frame: pd.DataFrame,
    *,
    target_column: str,
    problem_type: ProblemType | None,
    columns: list[str] | None = None,
) -> list[TargetAssociation]:
    """Score each feature's lone association with the target, strongest first.

    ``columns`` limits the scoring to a subset (e.g. only derived columns); by default every
    column except the target is scored. Columns that cannot be scored — all-null, constant, or
    effectively an identifier — are skipped rather than reported as a spurious strong signal.
    """
    if target_column not in frame.columns:
        return []
    target = frame[target_column]
    target_numeric = _is_numeric(target) and problem_type is ProblemType.REGRESSION

    candidates = columns if columns is not None else list(frame.columns)
    results: list[TargetAssociation] = []
    for name in candidates:
        if name == target_column or name not in frame.columns:
            continue
        association = _score_one(frame[name], target, target_numeric=target_numeric)
        if association is not None:
            results.append(association)

    results.sort(key=lambda a: a.score, reverse=True)
    return results


def _score_one(
    feature: pd.Series, target: pd.Series, *, target_numeric: bool
) -> TargetAssociation | None:
    paired = pd.concat([feature, target], axis=1).dropna()
    if len(paired) < 3:
        return None
    feature_values = paired.iloc[:, 0]
    target_values = paired.iloc[:, 1]
    if feature_values.nunique() < 2:
        return None  # constant over the shared rows: no relationship to measure

    feature_numeric = _is_numeric(feature_values)
    name = str(feature.name)
    non_null = len(paired)

    if feature_numeric and target_numeric:
        score = _abs_pearson(feature_values, target_values)
        method = "pearson"
    elif feature_numeric and not target_numeric:
        score = _correlation_ratio(categories=target_values, values=feature_values)
        method = "correlation_ratio"
    elif not feature_numeric and target_numeric:
        score = _correlation_ratio(categories=feature_values, values=target_values)
        method = "correlation_ratio"
    else:
        if feature_values.nunique() > MAX_CATEGORICAL_LEVELS:
            return None
        score = _cramers_v(feature_values, target_values)
        method = "cramers_v"

    if score is None or np.isnan(score):
        return None
    return TargetAssociation(
        feature=name,
        method=method,
        score=round(float(min(max(score, 0.0), 1.0)), 6),
        non_null_count=non_null,
    )


def _is_numeric(series: pd.Series) -> bool:
    return pdt.is_numeric_dtype(series) and not pdt.is_bool_dtype(series)


def _abs_pearson(x: pd.Series, y: pd.Series) -> float | None:
    if x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
        return None
    return float(abs(np.corrcoef(x.astype(float), y.astype(float))[0, 1]))


def _correlation_ratio(*, categories: pd.Series, values: pd.Series) -> float | None:
    """Eta: the share of the numeric variable's variance explained by group membership.

    sqrt(between-group variance / total variance). 1 means the category determines the value.
    """
    numeric = values.astype(float)
    grand_mean = numeric.mean()
    total_ss = float(((numeric - grand_mean) ** 2).sum())
    if total_ss == 0:
        return None
    between_ss = 0.0
    for _label, group in numeric.groupby(categories.astype("string"), observed=True):
        count = len(group)
        if count:
            between_ss += count * (group.mean() - grand_mean) ** 2
    return float(np.sqrt(between_ss / total_ss))


def _cramers_v(a: pd.Series, b: pd.Series) -> float | None:
    """Cramer's V: association between two categorical variables, bias-corrected, in [0, 1]."""
    table = pd.crosstab(a.astype("string"), b.astype("string"))
    if table.shape[0] < 2 or table.shape[1] < 2:
        return None
    observed = table.to_numpy(dtype=float)
    n = observed.sum()
    if n == 0:
        return None
    row_sums = observed.sum(axis=1, keepdims=True)
    col_sums = observed.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum(np.where(expected > 0, (observed - expected) ** 2 / expected, 0.0)))
    phi2 = chi2 / n
    r, k = observed.shape
    # Bias correction (Bergsma 2013), floored at 0 so noise does not read as a weak signal.
    phi2_corrected = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denominator = min(r_corrected - 1, k_corrected - 1)
    if denominator <= 0:
        return None
    return float(np.sqrt(phi2_corrected / denominator))
