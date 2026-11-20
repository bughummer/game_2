"""SDK splitters (free-form-experiments design C2, Req 5.1/5.4).

Re-exports the engine's existing splitters and adds time-ordered evaluation: a temporal holdout
and an expanding-window (rolling-origin) split, each guaranteeing all training rows are strictly
earlier than the evaluation rows, with an optional gap. A configuration that would leak a later
observation into training than into evaluation is refused with ``TemporalLeakError``.
"""

from dataclasses import dataclass

import pandas as pd

# Re-export the managed splitters so a script uses exactly the platform's code (Req 4.3).
from ml_engine.splitting import (  # noqa: F401
    RandomSplit,
    SplitResult,
    SplittingError,
    StratifiedRandomSplit,
    split_dataset,
)
from ml_engine.splitting.strategies import GroupSplit, TemporalSplit  # noqa: F401


class TemporalLeakError(ValueError):
    """A time-ordered split would place a later observation in training than in evaluation."""


@dataclass(frozen=True, slots=True)
class TimeFold:
    """One rolling-origin fold: positional train/eval row indices into the time-sorted frame."""

    fold: int
    train_index: list[int]
    eval_index: list[int]


def _ordered_positions(frame: pd.DataFrame, time_column: str) -> list[int]:
    if time_column not in frame.columns:
        raise TemporalLeakError(f"time column '{time_column}' is not in the frame")
    times = pd.to_datetime(frame[time_column], errors="coerce", utc=True)
    if times.isna().any():
        raise TemporalLeakError(f"time column '{time_column}' has unparseable or missing values")
    # Stable sort by time; ties keep input order. Positions index the original frame.
    return list(times.sort_values(kind="stable").index)


def temporal_holdout(
    frame: pd.DataFrame, *, time_column: str, eval_fraction: float = 0.2, gap: int = 0
) -> TimeFold:
    """A single out-of-time split: the earliest rows train, the most recent evaluate.

    ``gap`` rows between the two sides are dropped from both (e.g. to respect a label delay). The
    result always has every training row strictly earlier than every evaluation row.
    """
    if not 0.0 < eval_fraction < 1.0:
        raise TemporalLeakError("eval_fraction must be between 0 and 1")
    order = _ordered_positions(frame, time_column)
    n = len(order)
    eval_size = max(1, round(n * eval_fraction))
    split_at = n - eval_size
    train = order[: max(0, split_at - gap)]
    evaluation = order[split_at:]
    if not train or not evaluation:
        raise TemporalLeakError("the split leaves an empty training or evaluation set")
    _assert_before(frame, time_column, train, evaluation)
    return TimeFold(fold=0, train_index=train, eval_index=evaluation)


def rolling_origin_split(
    frame: pd.DataFrame,
    *,
    time_column: str,
    folds: int = 3,
    eval_size: int | None = None,
    gap: int = 0,
) -> list[TimeFold]:
    """Expanding-window (rolling-origin) CV: each fold trains on all rows before its eval window.

    The training window grows with each fold; every fold's training rows are strictly earlier than
    its evaluation rows (with the optional ``gap``). Refuses a config that cannot be honoured.
    """
    if folds < 1:
        raise TemporalLeakError("folds must be >= 1")
    order = _ordered_positions(frame, time_column)
    n = len(order)
    window = eval_size or max(1, n // (folds + 1))
    if window * folds >= n:
        raise TemporalLeakError(
            f"{folds} folds of eval size {window} need more than the {n} available rows"
        )
    result: list[TimeFold] = []
    first_eval_start = n - window * folds
    for fold in range(folds):
        eval_start = first_eval_start + fold * window
        eval_end = eval_start + window
        train = order[: max(0, eval_start - gap)]
        evaluation = order[eval_start:eval_end]
        if not train or not evaluation:
            raise TemporalLeakError(f"fold {fold} leaves an empty training or evaluation set")
        _assert_before(frame, time_column, train, evaluation)
        result.append(TimeFold(fold=fold, train_index=train, eval_index=evaluation))
    return result


def _assert_before(
    frame: pd.DataFrame, time_column: str, train: list[int], evaluation: list[int]
) -> None:
    """Guarantee the latest training time is <= the earliest evaluation time (Req 5.4)."""
    times = pd.to_datetime(frame[time_column], utc=True)
    latest_train = times.loc[train].max()
    earliest_eval = times.loc[evaluation].min()
    if latest_train > earliest_eval:
        raise TemporalLeakError(
            f"a training row at {latest_train} is later than an evaluation row at "
            f"{earliest_eval}; refusing to leak the future into the past"
        )
