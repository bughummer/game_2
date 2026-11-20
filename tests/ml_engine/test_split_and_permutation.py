"""ML Factory 2.0 engine additions: group and temporal splits, permutation importance."""

import numpy as np
import pandas as pd
import pytest

from ml_engine.contracts.common import MetricDirection, ProblemType
from ml_engine.contracts.config import SplitConfig
from ml_engine.evaluation.permutation import permutation_importance
from ml_engine.splitting import SplittingError, split_dataset


def _entities(rows: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "customer": rng.integers(0, 60, rows).astype(str),
            "when": pd.date_range("2024-01-01", periods=rows, freq="D")[rng.permutation(rows)],
            "x": rng.normal(size=rows),
            "y": rng.integers(0, 2, rows),
        }
    )


def test_a_group_split_never_puts_one_entity_in_both_folds():
    frame = _entities()
    result = split_dataset(
        frame,
        frame["y"],
        ProblemType.BINARY_CLASSIFICATION,
        SplitConfig(strategy="group", group_column="customer"),
    )
    train = set(frame.loc[result.train_index, "customer"])
    validation = set(frame.loc[result.validation_index, "customer"])
    assert train and validation and not train & validation
    assert len(result.train_index) + len(result.validation_index) == len(frame)
    assert result.summary.strategy == "group"


def test_a_group_split_needs_a_real_column():
    frame = _entities()
    with pytest.raises(SplittingError):
        split_dataset(frame, frame["y"], ProblemType.REGRESSION, SplitConfig(strategy="group"))


def test_a_temporal_split_validates_only_on_later_rows():
    frame = _entities()
    frame.loc[frame.index[:5], "when"] = pd.NaT
    result = split_dataset(
        frame,
        frame["y"],
        ProblemType.BINARY_CLASSIFICATION,
        SplitConfig(strategy="temporal", time_column="when", validation_fraction=0.25),
    )
    latest_train = frame.loc[result.train_index, "when"].max()
    earliest_validation = frame.loc[result.validation_index, "when"].min()
    assert latest_train < earliest_validation
    # Rows with no time are never used for validation, and the user is told.
    assert not set(frame.index[:5]) & set(result.validation_index)
    assert any(w.rule == "time_column_unparseable_values" for w in result.warnings)
    assert len(result.validation_index) == pytest.approx(0.25 * len(frame), abs=2)


def test_permutation_importance_finds_the_column_the_score_depends_on():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"signal": rng.normal(size=200), "noise": rng.normal(size=200)})
    truth = frame["signal"].to_numpy()

    def score(f: pd.DataFrame) -> float:
        return float(np.corrcoef(f["signal"], truth)[0, 1])

    result = permutation_importance(
        frame,
        ["signal", "noise"],
        score,
        experiment_id="e",
        model_name="m",
        primary_metric="r",
        direction=MetricDirection.MAXIMIZE,
    )
    assert result is not None
    top = result.entries[0]
    assert top.feature == "signal" and top.share == pytest.approx(1.0)
    assert result.baseline_score == pytest.approx(1.0)
    noise = next(e for e in result.entries if e.feature == "noise")
    assert noise.mean_drop == pytest.approx(0.0)


def test_permutation_importance_orients_a_minimised_metric():
    frame = pd.DataFrame({"a": np.arange(50.0)})

    def error(f: pd.DataFrame) -> float:  # lower is better; 0 when in order
        return float(np.abs(f["a"].to_numpy() - np.arange(50.0)).mean())

    result = permutation_importance(
        frame,
        ["a"],
        error,
        experiment_id="e",
        model_name="m",
        primary_metric="mae",
        direction=MetricDirection.MINIMIZE,
    )
    assert result.entries[0].mean_drop > 0  # positive = worse without the information
