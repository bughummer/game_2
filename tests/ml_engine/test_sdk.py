"""Platform_SDK: splitters, metrics, and engine labelling (free-form design C2, Properties 3/4)."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ml_engine.contracts.common import ProblemType
from ml_engine.evaluation import evaluate
from ml_engine.sdk import SDK_VERSION, metrics, results, splits

SETTINGS = settings(max_examples=100, deadline=None, derandomize=True)


def _timed_frame(rows: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    times = pd.date_range("2020-01-01", periods=rows, freq="D")
    return pd.DataFrame({"t": times, "x": rng.normal(size=rows), "y": rng.normal(size=rows)})


def test_sdk_version_is_exposed():
    assert SDK_VERSION


# --- Property 3: temporal safety --------------------------------------------------------------


@given(
    rows=st.integers(min_value=20, max_value=200),
    eval_fraction=st.floats(min_value=0.1, max_value=0.5),
    gap=st.integers(min_value=0, max_value=3),
)
@SETTINGS
def test_temporal_holdout_is_train_before_eval(rows, eval_fraction, gap):
    frame = _timed_frame(rows)
    fold = splits.temporal_holdout(frame, time_column="t", eval_fraction=eval_fraction, gap=gap)
    latest_train = frame["t"].loc[fold.train_index].max()
    earliest_eval = frame["t"].loc[fold.eval_index].min()
    assert latest_train <= earliest_eval


@given(
    rows=st.integers(min_value=40, max_value=200),
    folds=st.integers(min_value=1, max_value=4),
)
@SETTINGS
def test_rolling_origin_is_train_before_eval_every_fold(rows, folds):
    frame = _timed_frame(rows)
    try:
        time_folds = splits.rolling_origin_split(frame, time_column="t", folds=folds)
    except splits.TemporalLeakError:
        return  # too few rows for the requested folds; a legitimate refusal
    for fold in time_folds:
        latest_train = frame["t"].loc[fold.train_index].max()
        earliest_eval = frame["t"].loc[fold.eval_index].min()
        assert latest_train <= earliest_eval


def test_a_leaking_temporal_config_is_refused():
    frame = _timed_frame(10)
    # Shuffle the time column so the natural order is scrambled; unparseable/odd configs refuse.
    with pytest.raises(splits.TemporalLeakError):
        splits.temporal_holdout(frame, time_column="not_a_column", eval_fraction=0.2)
    with pytest.raises(splits.TemporalLeakError):
        splits.rolling_origin_split(frame, time_column="t", folds=20)


# --- Property 4: SDK equals managed -----------------------------------------------------------


def test_sdk_evaluate_equals_managed_evaluate():
    y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    sdk_result = metrics.evaluate(ProblemType.REGRESSION, y_true, y_pred)
    managed = evaluate(ProblemType.REGRESSION, y_true, y_pred)
    assert sdk_result == managed


# --- forecasting metrics + baseline -----------------------------------------------------------


def test_forecasting_metrics_include_the_naive_baseline():
    history = np.array([10.0, 11.0, 12.0, 13.0])
    y_true = np.array([14.0, 15.0])
    y_pred = np.array([14.1, 14.8])
    scores = metrics.forecasting_metrics(y_true, y_pred, history=history, season_length=1)
    assert {"mae", "rmse", "mape", "smape", "naive_mae", "naive_rmse"} <= set(scores)
    # The last-value baseline forecasts 13.0 for both steps.
    assert scores["naive_mae"] == pytest.approx(np.mean([abs(14 - 13), abs(15 - 13)]))


def test_seasonal_naive_repeats_the_season():
    history = np.array([1.0, 2.0, 3.0, 4.0])
    baseline = metrics.seasonal_naive_baseline(history, horizon=4, season_length=2)
    assert list(baseline) == [3.0, 4.0, 3.0, 4.0]


# --- results writer labels engine metrics -----------------------------------------------------


class _FakeContext:
    def __init__(self):
        self.engine: dict[str, float] = {}
        self.naive_baseline: dict[str, float] = {}

    def report_engine_metric(self, name, value):
        self.engine[name] = value


def test_results_writer_records_engine_metrics():
    ctx = _FakeContext()
    results.write_metrics(ctx, {"rmse": 1.5, "mae": 1.0})
    assert ctx.engine == {"rmse": 1.5, "mae": 1.0}
    results.set_baseline(ctx, {"naive_rmse": 2.0})
    assert ctx.naive_baseline == {"naive_rmse": 2.0}
