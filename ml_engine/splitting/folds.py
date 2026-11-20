"""Fold_Planner: Test_Fold carve-out and K-fold assignment (ML Factory 2.0, CV design C2).

Pure functions, no IO. Row ids are the integer positions of the rows in the frame passed in
(``0 .. len(frame) - 1``); the caller passes the usable frame, so the assignment covers every
usable row. Determinism comes from ``random_state=split.random_seed`` everywhere and stable
(mergesort) ordering.

The Test_Fold is carved first by reusing the inherited ``split_dataset`` with
``validation_fraction=test_fraction``, so it depends only on the strategy, seed and fraction —
never on K or the evaluation mode. CV folds are then assigned on the Development_Set with the
scheme matching the split strategy (``stratified_kfold``, ``kfold``, ``group_kfold``,
``forward_chaining``). ``strategies.py`` is reused unchanged.
"""

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from ml_engine.contracts.common import (
    ProblemType,
    RecommendedAction,
    Severity,
    WarningCategory,
)
from ml_engine.contracts.config import SplitConfig
from ml_engine.contracts.cv import TEST, TRAIN_ONLY, CvScheme, EvaluationMode, FoldAssignment
from ml_engine.contracts.preparation import SplitSummary
from ml_engine.contracts.warnings import AnalysisWarning
from ml_engine.splitting.strategies import (
    MIN_ROWS_PER_FOLD,
    GroupSplit,
    RandomSplit,
    SplitResult,
    SplittingError,
    StratifiedRandomSplit,
    TemporalSplit,
    resolve_strategy,
    split_dataset,
)

#: The group value given to rows without a group, as ``GroupSplit`` does.
MISSING_GROUP = "<missing>"

_SCHEME_BY_STRATEGY = {
    StratifiedRandomSplit.name: CvScheme.STRATIFIED_KFOLD,
    RandomSplit.name: CvScheme.KFOLD,
    GroupSplit.name: CvScheme.GROUP_KFOLD,
    TemporalSplit.name: CvScheme.FORWARD_CHAINING,
}


class FoldPlanningReason(StrEnum):
    GROUPS_FEWER_THAN_FOLDS = "GROUPS_FEWER_THAN_FOLDS"
    FOLD_TOO_SMALL = "FOLD_TOO_SMALL"
    TEST_FOLD_TOO_SMALL = "TEST_FOLD_TOO_SMALL"
    TEMPORAL_TIES = "TEMPORAL_TIES"


class FoldPlanningError(SplittingError):
    """The folds cannot be built. The preparation job maps it to ``SPLIT_FAILED``."""

    def __init__(
        self, reason: FoldPlanningReason | str, message: str, **details: int | float | str
    ) -> None:
        super().__init__(message)
        self.reason = FoldPlanningReason(reason)
        self.message = message
        self.details: dict[str, int | float | str] = dict(details)


@dataclass(slots=True)
class TestCarveOut:
    """Positions of the Development_Set and the Test_Fold (both ascending)."""

    __test__ = False  # not a pytest test class

    development_rows: list[int]
    test_rows: list[int]
    warnings: list[AnalysisWarning] = field(default_factory=list)
    summary: SplitSummary | None = None  # the carve-out split; None without a Test_Fold


@dataclass(slots=True)
class FoldPlan:
    assignment: FoldAssignment
    warnings: list[AnalysisWarning] = field(default_factory=list)
    carve: TestCarveOut | None = None
    #: Single_Split_Mode only: the inherited split of the Development_Set (indices = row ids).
    single_split: SplitResult | None = None


# --------------------------------------------------------------------------------------------
# Scheme, fingerprint, row helpers
# --------------------------------------------------------------------------------------------


def resolve_cv_scheme(problem_type: ProblemType, split: SplitConfig) -> CvScheme:
    """The CV_Scheme matching the split strategy the inherited resolver would pick."""
    return _SCHEME_BY_STRATEGY[resolve_strategy(problem_type, split).name]


def dataset_fingerprint(frame: pd.DataFrame, columns: list[str]) -> str:
    """sha256 over the column names, the row count and ``hash_pandas_object`` of the rows."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"columns not in the frame: {missing}")
    hasher = hashlib.sha256()
    hasher.update(json.dumps([str(column) for column in columns]).encode("utf-8"))
    hasher.update(str(len(frame)).encode("ascii"))
    if columns and len(frame):
        hashes = pd.util.hash_pandas_object(frame[list(columns)], index=False)
        hasher.update(hashes.to_numpy(dtype="<u8").tobytes())
    return hasher.hexdigest()


def fold_rows(assignment: FoldAssignment, fold: int) -> tuple[list[int], list[int]]:
    """(training row ids, validation row ids) of one fold; Test_Fold rows are in neither.

    ``forward_chaining`` trains on the TRAIN_ONLY rows plus the earlier folds (expanding
    window); every other scheme trains on all other Development_Set rows. In Single_Split_Mode
    fold 0 is the validation partition and the TRAIN_ONLY rows are the training partition.
    """
    limit = assignment.folds if assignment.evaluation_mode is EvaluationMode.CV else 1
    if not 0 <= fold < limit:
        raise ValueError(f"fold {fold} is outside 0..{limit - 1}")
    forward = assignment.scheme is CvScheme.FORWARD_CHAINING
    train: list[int] = []
    validation: list[int] = []
    for row_id, value in zip(assignment.row_ids, assignment.assignments, strict=True):
        if value == TEST:
            continue
        if value == fold:
            validation.append(row_id)
        elif not forward or value == TRAIN_ONLY or value < fold:
            train.append(row_id)
    return train, validation


def development_rows(assignment: FoldAssignment) -> list[int]:
    return [r for r, v in zip(assignment.row_ids, assignment.assignments, strict=True) if v != TEST]


def test_rows(assignment: FoldAssignment) -> list[int]:
    return [r for r, v in zip(assignment.row_ids, assignment.assignments, strict=True) if v == TEST]


test_rows.__test__ = False  # type: ignore[attr-defined]  # not a pytest test when imported


# --------------------------------------------------------------------------------------------
# Internals
# --------------------------------------------------------------------------------------------


def _positional(frame: pd.DataFrame, target: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    if len(frame) != len(target):
        raise ValueError(f"frame ({len(frame)}) and target ({len(target)}) lengths differ")
    return (
        frame.reset_index(drop=True),
        pd.Series(target.to_numpy(), name=target.name, dtype=target.dtype),
    )


def _require_columns(frame: pd.DataFrame, problem_type: ProblemType, split: SplitConfig) -> str:
    """Resolve the strategy and check the column it needs, as the strategies would."""
    strategy = resolve_strategy(problem_type, split).name
    if strategy == GroupSplit.name and (
        not split.group_column or split.group_column not in frame.columns
    ):
        raise SplittingError(
            f"A group split needs group_column to name a dataset column "
            f"(got {split.group_column!r})."
        )
    if strategy == TemporalSplit.name and (
        not split.time_column or split.time_column not in frame.columns
    ):
        raise SplittingError(
            f"A temporal split needs time_column to name a dataset column "
            f"(got {split.time_column!r})."
        )
    return strategy


def parse_times(frame: pd.DataFrame, column: str) -> pd.Series:
    """Parse a time column exactly as ``TemporalSplit`` does (datetime, else numeric)."""
    times = frame[column]
    if not pd.api.types.is_datetime64_any_dtype(times):
        times = pd.to_datetime(times, errors="coerce", utc=True)
        if pd.to_numeric(frame[column], errors="coerce").notna().all():
            times = pd.to_numeric(frame[column], errors="coerce")
    return times


def _too_small(rows: int, folds: int, *, fold_index: int | None = None, part: str) -> None:
    details: dict[str, int | float | str] = {
        "row_count": rows,
        "folds": folds,
        "minimum": MIN_ROWS_PER_FOLD,
        "part": part,
    }
    if fold_index is not None:
        details["fold_index"] = fold_index
    raise FoldPlanningError(
        FoldPlanningReason.FOLD_TOO_SMALL,
        f"The {part} would contain {rows} row(s) with K={folds}; at least "
        f"{MIN_ROWS_PER_FOLD} rows per fold are required.",
        **details,
    )


def _require_rows_for_k(rows: int, folds: int) -> None:
    if rows < folds * MIN_ROWS_PER_FOLD:
        _too_small(rows // folds, folds, part="smallest CV fold")


def _from_splits(splits, size: int) -> np.ndarray:
    values = np.empty(size, dtype=np.int64)
    for index, (_, validation) in enumerate(splits):
        values[validation] = index
    return values


def _kfold(size: int, folds: int, seed: int) -> np.ndarray:
    _require_rows_for_k(size, folds)
    splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
    return _from_splits(splitter.split(np.zeros(size)), size)


def _stratified_kfold(
    labels: pd.Series, folds: int, seed: int
) -> tuple[np.ndarray, list[AnalysisWarning]]:
    size = len(labels)
    _require_rows_for_k(size, folds)
    labels = labels.astype("string")
    counts = labels.value_counts()
    rarest = int(counts.min())
    if rarest < folds:
        rare_class = sorted(str(label) for label, count in counts.items() if count == rarest)[0]
        warning = AnalysisWarning(
            rule="cv_stratification_not_possible",
            category=WarningCategory.TARGET,
            severity=Severity.HIGH,
            message=(
                f"Class '{rare_class}' has {rarest} row(s) in the Development_Set, fewer than "
                f"K={folds}, so stratified K-fold is impossible. Shuffled K-fold was used "
                "instead; per-fold metrics for rare classes will be unreliable."
            ),
            column=str(labels.name) if labels.name is not None else None,
            recommended_action=RecommendedAction.REVIEW,
            details={"class": rare_class, "class_count": rarest, "folds": folds},
        )
        return _kfold(size, folds, seed), [warning]
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return _from_splits(splitter.split(np.zeros(size), labels.to_numpy(dtype=object)), size), []


def _group_kfold(groups: pd.Series, folds: int, seed: int) -> np.ndarray:
    group_count = int(groups.nunique())
    if group_count < folds:
        raise FoldPlanningError(
            FoldPlanningReason.GROUPS_FEWER_THAN_FOLDS,
            f"The Development_Set has {group_count} distinct group value(s), fewer than "
            f"K={folds}; group K-fold needs at least one group per fold.",
            group_count=group_count,
            folds=folds,
        )
    size = len(groups)
    splitter = GroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    split = splitter.split(np.zeros(size), groups=groups.to_numpy(dtype=object))
    return _from_splits(split, size)


def _forward_chaining(times: pd.Series, folds: int) -> np.ndarray:
    """K + 1 contiguous blocks on distinct time values; block 0 and unknown times TRAIN_ONLY."""
    values = np.full(len(times), TRAIN_ONLY, dtype=np.int64)
    known = times.notna().to_numpy()
    codes, uniques = pd.factorize(times[known], sort=True)
    distinct = len(uniques)
    if distinct < folds + 1:
        raise FoldPlanningError(
            FoldPlanningReason.TEMPORAL_TIES,
            f"The Development_Set has {distinct} distinct parseable time value(s); forward "
            f"chaining with K={folds} needs {folds + 1} non-empty time blocks.",
            distinct_times=distinct,
            folds=folds,
            required=folds + 1,
        )
    counts = np.bincount(codes, minlength=distinct)
    cumulative = np.cumsum(counts)
    total = int(cumulative[-1])
    cuts: list[int] = []
    previous = -1
    for block in range(1, folds + 1):
        goal = block * total / (folds + 1)
        nearest = int(np.argmin(np.abs(cumulative[: distinct - 1] - goal)))
        # Every block keeps at least one distinct value: clamp the cut into the feasible range.
        cut = min(max(nearest, previous + 1), distinct - 2 - (folds - block))
        cuts.append(cut)
        previous = cut
    block_of_value = np.searchsorted(np.asarray(cuts), np.arange(distinct), side="left")
    blocks = block_of_value[codes]
    first_block = int((blocks == 0).sum())
    if first_block < MIN_ROWS_PER_FOLD:
        _too_small(first_block, folds, part="first training block")
    values[known] = np.where(blocks == 0, TRAIN_ONLY, blocks - 1)
    return values


def _merge_warnings(*groups: list[AnalysisWarning]) -> list[AnalysisWarning]:
    merged: list[AnalysisWarning] = []
    seen: set[tuple[str, str | None]] = set()
    for group in groups:
        for warning in group:
            key = (warning.rule, warning.column)
            if key not in seen:
                seen.add(key)
                merged.append(warning)
    return merged


def _missing_group_warning(frame: pd.DataFrame, column: str) -> list[AnalysisWarning]:
    missing = int(frame[column].isna().sum())
    if not missing:
        return []
    return [
        AnalysisWarning(
            rule="group_column_missing_values",
            category=WarningCategory.DATA_QUALITY,
            severity=Severity.MEDIUM,
            message=(
                f"{missing} row(s) have no '{column}' value; they were treated as one group and "
                "kept together in a single fold."
            ),
            column=column,
            details={"missing_rows": missing},
        )
    ]


def _unparseable_time_warning(unknown: int, column: str) -> list[AnalysisWarning]:
    if not unknown:
        return []
    return [
        AnalysisWarning(
            rule="time_column_unparseable_values",
            category=WarningCategory.DATA_QUALITY,
            severity=Severity.MEDIUM,
            message=(
                f"{unknown} row(s) have no usable '{column}' value; they are used for training "
                "in every fold and never for validation."
            ),
            column=column,
            details={"rows": unknown},
        )
    ]


# --------------------------------------------------------------------------------------------
# Public planning API
# --------------------------------------------------------------------------------------------


def carve_test_fold(
    frame: pd.DataFrame, target: pd.Series, problem_type: ProblemType, split: SplitConfig
) -> TestCarveOut:
    """Split off the Test_Fold with the inherited strategy (``validation_fraction=test_fraction``).

    Without a ``test_fraction`` every row is in the Development_Set. Reusing ``split_dataset``
    keeps stratification (and its rare-class fallback), group isolation and temporal ordering
    (every Test_Fold time strictly later than every Development_Set time) identical to the
    single split.
    """
    frame, target = _positional(frame, target)
    rows = len(frame)
    if split.test_fraction is None:
        return TestCarveOut(development_rows=list(range(rows)), test_rows=[])
    _require_columns(frame, problem_type, split)
    fraction = split.test_fraction
    expected = round(rows * fraction)
    config = split.model_copy(update={"validation_fraction": fraction})
    details: dict[str, int | float | str] = {
        "minimum": MIN_ROWS_PER_FOLD,
        "fraction": fraction,
        "row_count": rows,
    }
    try:
        result = split_dataset(frame, target, problem_type, config)
    except ValueError as error:  # SplittingError, or scikit-learn refusing the split
        raise FoldPlanningError(
            FoldPlanningReason.TEST_FOLD_TOO_SMALL,
            f"test_fraction={fraction} cannot form a Test_Fold of at least "
            f"{MIN_ROWS_PER_FOLD} rows from {rows} rows (about {expected} rows): {error}",
            test_rows=expected,
            **details,
        ) from error
    test = sorted(int(position) for position in result.validation_index)
    if len(test) < MIN_ROWS_PER_FOLD:
        raise FoldPlanningError(
            FoldPlanningReason.TEST_FOLD_TOO_SMALL,
            f"test_fraction={fraction} yields a Test_Fold of {len(test)} row(s); at least "
            f"{MIN_ROWS_PER_FOLD} are required.",
            test_rows=len(test),
            **details,
        )
    development = sorted(int(position) for position in result.train_index)
    return TestCarveOut(
        development_rows=development,
        test_rows=test,
        warnings=list(result.warnings),
        summary=result.summary,
    )


def plan_folds(
    frame: pd.DataFrame,
    target: pd.Series,
    problem_type: ProblemType,
    split: SplitConfig,
    *,
    fingerprint: str,
    experiment_id: str,
    generated_at: datetime | None = None,
    development_digest: str | None = None,
) -> FoldPlan:
    """Build the Fold_Assignment of the usable ``frame`` (row id = position in ``frame``).

    CV_Mode assigns each Development_Set row to a CV_Fold (or, for ``forward_chaining``, to
    TRAIN_ONLY). Single_Split_Mode records ``folds=0``: TEST, TRAIN_ONLY for the training side
    and 0 for the validation side of the inherited split of the Development_Set.

    ``development_digest`` defaults to the fingerprint of the Development_Set rows over all
    columns; the preparation job may pass the digest of the rows as written instead.
    """
    frame, target = _positional(frame, target)
    rows = len(frame)
    strategy = _require_columns(frame, problem_type, split)
    seed = split.random_seed
    carve = carve_test_fold(frame, target, problem_type, split)
    development = np.asarray(carve.development_rows, dtype=np.int64)
    values = np.full(rows, TEST, dtype=np.int64)
    extra: list[AnalysisWarning] = []
    single: SplitResult | None = None

    if split.evaluation_mode is EvaluationMode.CV:
        assert split.cv is not None  # guaranteed by the SplitConfig validator
        folds, scheme = split.cv.folds, split.cv.scheme
        expected_scheme = _SCHEME_BY_STRATEGY[strategy]
        if scheme is not expected_scheme:
            raise SplittingError(
                f"CV scheme {scheme.value!r} does not match the split strategy {strategy!r} "
                f"(expected {expected_scheme.value!r})."
            )
        if scheme is CvScheme.STRATIFIED_KFOLD:
            assigned, extra = _stratified_kfold(target.iloc[development], folds, seed)
        elif scheme is CvScheme.KFOLD:
            assigned = _kfold(len(development), folds, seed)
        elif scheme is CvScheme.GROUP_KFOLD:
            column = str(split.group_column)
            groups = frame[column].astype("string").fillna(MISSING_GROUP).iloc[development]
            assigned = _group_kfold(groups, folds, seed)
            extra = _missing_group_warning(frame.iloc[development], column)
        else:
            column = str(split.time_column)
            times = parse_times(frame, column).iloc[development]
            assigned = _forward_chaining(times, folds)
            extra = _unparseable_time_warning(int(times.isna().sum()), column)
        values[development] = assigned
        for fold in range(folds):
            count = int((values == fold).sum())
            if count < MIN_ROWS_PER_FOLD:
                _too_small(
                    count, folds, fold_index=fold, part=f"validation partition of fold {fold}"
                )
        mode, scheme_value = EvaluationMode.CV, scheme
    else:
        folds, scheme_value, mode = 0, None, EvaluationMode.SINGLE_SPLIT
        try:
            single = split_dataset(
                frame.iloc[development], target.iloc[development], problem_type, split
            )
        except ValueError as error:  # SplittingError, or scikit-learn refusing the split
            raise FoldPlanningError(
                FoldPlanningReason.FOLD_TOO_SMALL,
                f"The Development_Set ({len(development)} rows) cannot be split into training "
                f"and validation folds: {error}",
                row_count=len(development),
                folds=0,
                minimum=MIN_ROWS_PER_FOLD,
                part="validation fold",
            ) from error
        values[development] = TRAIN_ONLY
        values[np.asarray(single.validation_index, dtype=np.int64)] = 0
        extra = list(single.warnings)

    if development_digest is None:
        development_digest = dataset_fingerprint(frame.iloc[development], list(frame.columns))
    assignment = FoldAssignment(
        experiment_id=experiment_id,
        generated_at=generated_at or datetime.now(UTC),
        evaluation_mode=mode,
        scheme=scheme_value,
        folds=folds,
        test_fraction=split.test_fraction,
        random_seed=seed,
        dataset_fingerprint=fingerprint,
        development_digest=development_digest,
        row_ids=list(range(rows)),
        assignments=values.tolist(),
        group_column=split.group_column if strategy == GroupSplit.name else None,
        time_column=split.time_column if strategy == TemporalSplit.name else None,
    )
    return FoldPlan(
        assignment=assignment,
        warnings=_merge_warnings(carve.warnings, extra),
        carve=carve,
        single_split=single,
    )
