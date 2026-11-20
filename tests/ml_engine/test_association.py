"""Target-association scoring: the leak-screening signal.

The property that matters: a feature that alone almost determines the target scores near 1.0,
whatever the column/target type combination, and an unrelated feature scores near 0. That is
what makes a runaway score usable as leak evidence.
"""

import numpy as np
import pandas as pd

from ml_engine.contracts.common import ProblemType
from ml_engine.profiling import target_associations


def _rng():
    return np.random.default_rng(0)


def test_a_near_perfect_predictor_scores_near_one_for_a_numeric_target():
    rng = _rng()
    target = rng.normal(size=500)
    frame = pd.DataFrame(
        {
            "leak": target * 2.0 + rng.normal(scale=1e-3, size=500),  # almost the target
            "noise": rng.normal(size=500),
            "target": target,
        }
    )
    scores = {
        a.feature: a.score
        for a in target_associations(
            frame, target_column="target", problem_type=ProblemType.REGRESSION
        )
    }
    assert scores["leak"] > 0.99
    assert scores["noise"] < 0.3
    # strongest-first ordering is what the context relies on
    first = target_associations(frame, target_column="target", problem_type=ProblemType.REGRESSION)[
        0
    ]
    assert first.feature == "leak"
    assert first.method == "pearson"


def test_a_categorical_feature_that_determines_a_categorical_target_scores_high():
    rng = _rng()
    group = rng.choice(["a", "b", "c"], size=600)
    frame = pd.DataFrame(
        {
            "leak": group,  # the label is a deterministic function of this
            "unrelated": rng.choice(["x", "y"], size=600),
            "target": np.where(group == "a", "win", np.where(group == "b", "draw", "loss")),
        }
    )
    scores = {
        a.feature: (a.score, a.method)
        for a in target_associations(
            frame, target_column="target", problem_type=ProblemType.MULTICLASS_CLASSIFICATION
        )
    }
    assert scores["leak"][0] > 0.95
    assert scores["leak"][1] == "cramers_v"
    assert scores["unrelated"][0] < 0.3


def test_a_numeric_feature_separating_the_classes_scores_high_via_correlation_ratio():
    rng = _rng()
    label = rng.integers(0, 2, size=400)
    frame = pd.DataFrame(
        {
            "separates": label * 10.0 + rng.normal(scale=0.1, size=400),
            "target": label,
        }
    )
    [association] = target_associations(
        frame, target_column="target", problem_type=ProblemType.BINARY_CLASSIFICATION
    )
    assert association.feature == "separates"
    assert association.method == "correlation_ratio"
    assert association.score > 0.95


def test_constant_and_empty_columns_are_skipped_not_scored_as_strong():
    frame = pd.DataFrame(
        {
            "constant": [1] * 100,
            "empty": [None] * 100,
            "target": _rng().integers(0, 2, size=100),
        }
    )
    scored = {
        a.feature
        for a in target_associations(
            frame, target_column="target", problem_type=ProblemType.BINARY_CLASSIFICATION
        )
    }
    assert "constant" not in scored
    assert "empty" not in scored


def test_the_target_never_scores_itself():
    frame = pd.DataFrame({"x": _rng().normal(size=50), "target": _rng().normal(size=50)})
    features = {
        a.feature
        for a in target_associations(
            frame, target_column="target", problem_type=ProblemType.REGRESSION
        )
    }
    assert "target" not in features


def test_scoring_can_be_restricted_to_named_columns():
    rng = _rng()
    frame = pd.DataFrame(
        {"a": rng.normal(size=80), "b": rng.normal(size=80), "target": rng.normal(size=80)}
    )
    scored = {
        a.feature
        for a in target_associations(
            frame, target_column="target", problem_type=ProblemType.REGRESSION, columns=["a"]
        )
    }
    assert scored == {"a"}


def test_a_missing_target_column_yields_no_associations():
    frame = pd.DataFrame({"x": [1, 2, 3]})
    assert (
        target_associations(frame, target_column="absent", problem_type=ProblemType.REGRESSION)
        == []
    )


def test_every_score_is_bounded():
    rng = _rng()
    frame = pd.DataFrame(
        {
            "num": rng.normal(size=200),
            "cat": rng.choice(list("abcde"), size=200),
            "target": rng.integers(0, 3, size=200),
        }
    )
    for association in target_associations(
        frame, target_column="target", problem_type=ProblemType.MULTICLASS_CLASSIFICATION
    ):
        assert 0.0 <= association.score <= 1.0
