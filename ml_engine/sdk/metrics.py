"""SDK metrics (free-form-experiments design C2, Req 4.2/5.2).

Re-exports the engine's deterministic ``evaluate`` (so a classification/regression metric equals
the managed pipeline's for the same inputs, Req 4.3) and adds forecasting metrics with a
seasonal-naive baseline (Req 5.2, 5.5). All pure.
"""

import numpy as np

# Re-export the managed evaluator unchanged (Req 4.3).
from ml_engine.evaluation import evaluate

__all__ = [
    "evaluate",
    "forecasting_metrics",
    "mae",
    "mape",
    "rmse",
    "seasonal_naive_baseline",
    "smape",
]


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype="float64")
    b = np.asarray(y_pred, dtype="float64")
    if a.shape != b.shape:
        raise ValueError(f"y_true {a.shape} and y_pred {b.shape} differ in shape")
    return a, b


def mae(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def rmse(y_true, y_pred) -> float:
    a, b = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error, ignoring rows where the true value is zero."""
    a, b = _clean(y_true, y_pred)
    nonzero = a != 0
    if not nonzero.any():
        return float("nan")
    return float(np.mean(np.abs((a[nonzero] - b[nonzero]) / a[nonzero])) * 100.0)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE in [0, 200]; 0 where both values are 0."""
    a, b = _clean(y_true, y_pred)
    denom = np.abs(a) + np.abs(b)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom == 0, 0.0, np.abs(a - b) / (denom / 2.0))
    return float(np.mean(ratio) * 100.0)


def seasonal_naive_baseline(history, horizon: int, *, season_length: int = 1):
    """The seasonal-naive forecast: repeat the value from one season earlier (Req 5.5).

    ``history`` is the observed series; the forecast for step ``h`` is ``history[-season_length + h
    % season_length]``. With ``season_length=1`` this is the last-value (random-walk) baseline.
    """
    series = np.asarray(history, dtype="float64")
    if series.size < season_length:
        raise ValueError("history is shorter than one season")
    season = series[-season_length:]
    return np.array([season[h % season_length] for h in range(horizon)], dtype="float64")


def forecasting_metrics(
    y_true, y_pred, *, history=None, season_length: int = 1
) -> dict[str, float]:
    """MAE/RMSE/MAPE/sMAPE plus, when ``history`` is given, the seasonal-naive baseline's errors.

    So a model that fails to beat the naive baseline is visible (Req 5.5): the returned dict carries
    both the model's errors and ``naive_*`` errors on the same evaluation window.
    """
    result = {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
    }
    if history is not None:
        horizon = len(np.asarray(y_pred))
        baseline = seasonal_naive_baseline(history, horizon, season_length=season_length)
        result["naive_mae"] = mae(y_true, baseline)
        result["naive_rmse"] = rmse(y_true, baseline)
    return result
