"""Fold_Planner (CV design C2): every scheme and every planning error on hand-built frames."""

import numpy as np
import pandas as pd
import pytest

from ml_engine.contracts.common import ProblemType, Severity
from ml_engine.contracts.config import SplitConfig
from ml_engine.contracts.cv import (
    TEST,
    TRAIN_ONLY,
    CrossValidationConfig,
    CvScheme,
    EvaluationMode,
    FoldAssignment,
)
from ml_engine.splitting import folds
from ml_engine.splitting.folds import (
    FoldPlanningError,
    FoldPlanningReason,
    carve_test_fold,
    dataset_fingerprint,
    development_rows,
    fold_rows,
    plan_folds,
    resolve_cv_scheme,
)
from ml_engine.splitting.strategies import MIN_ROWS_PER_FOLD, SplittingError

BINARY = ProblemType.BINARY_CLASSIFICATION
REGRESSION = ProblemType.REGRESSION


def _frame(rows: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "x": rng.normal(size=rows),
            "label": np.where(np.arange(rows) % 3 == 0, "yes", "no"),
            "amount": rng.normal(size=rows),
            "customer": [f"c{i % 12}" for i in range(rows)],
            "day": [f"2024-01-{(i % 20) + 1:02d}" for i in range(rows)],
        }
    )


def _cv(problem_type: ProblemType, folds_k: int = 3, **split) -> SplitConfig:
    base = SplitConfig(**split)
    scheme = resolve_cv_scheme(problem_type, base)
    return base.model_copy(
        update={
            "evaluation_mode": EvaluationMode.CV,
            "cv": CrossValidationConfig(folds=folds_k, scheme=scheme),
        }
    )


def _plan(frame, split, problem_type=BINARY, target="label"):
    return plan_folds(
        frame, frame[target], problem_type, split, fingerprint="fp", experiment_id="exp"
    )


def _values(plan) -> np.ndarray:
    return np.asarray(plan.assignment.assignments)


# --- scheme resolution and fingerprint ------------------------------------------------------


@pytest.mark.parametrize(
    ("problem_type", "strategy", "stratify", "expected"),
    [
        (BINARY, "stratified_random", True, CvScheme.STRATIFIED_KFOLD),
        (BINARY, "stratified_random", False, CvScheme.KFOLD),
        (REGRESSION, "stratified_random", True, CvScheme.KFOLD),
        (BINARY, "random", True, CvScheme.KFOLD),
        (BINARY, "group", True, CvScheme.GROUP_KFOLD),
        (REGRESSION, "temporal", True, CvScheme.FORWARD_CHAINING),
    ],
)
def test_resolve_cv_scheme_follows_the_split_strategy(problem_type, strategy, stratify, expected):
    split = SplitConfig(strategy=strategy, stratify=stratify)
    assert resolve_cv_scheme(problem_type, split) is expected


def test_resolve_cv_scheme_rejects_unknown_strategy():
    with pytest.raises(SplittingError):
        resolve_cv_scheme(BINARY, SplitConfig(strategy="nope"))


def test_dataset_fingerprint_is_stable_and_sensitive():
    frame = _frame()
    columns = ["x", "label"]
    assert dataset_fingerprint(frame, columns) == dataset_fingerprint(frame.copy(), columns)
    changed = frame.copy()
    changed.loc[5, "x"] = 99.0
    assert dataset_fingerprint(changed, columns) != dataset_fingerprint(frame, columns)
    assert dataset_fingerprint(frame, ["x"]) != dataset_fingerprint(frame, columns)
    # The index is not part of the identity (row ids are positions).
    assert dataset_fingerprint(frame.set_axis(range(100, 160)), columns) == dataset_fingerprint(
        frame, columns
    )
    with pytest.raises(ValueError, match="missing_column"):
        dataset_fingerprint(frame, ["missing_column"])


def test_planning_error_is_a_splitting_error_with_reason_and_details():
    error = FoldPlanningError("FOLD_TOO_SMALL", "too small", row_count=1, folds=3)
    assert isinstance(error, SplittingError)
    assert error.reason is FoldPlanningReason.FOLD_TOO_SMALL
    assert error.details == {"row_count": 1, "folds": 3}


# --- Test_Fold carve-out -------------------------------------------------------------------


def test_no_test_fraction_keeps_every_row_in_development():
    frame = _frame()
    carve = carve_test_fold(frame, frame["label"], BINARY, SplitConfig())
    assert carve.development_rows == list(range(len(frame)))
    assert carve.test_rows == []


def test_stratified_test_fold_preserves_class_proportions():
    frame = _frame(90)
    carve = carve_test_fold(frame, frame["label"], BINARY, SplitConfig(test_fraction=0.2))
    test_labels = frame["label"].iloc[carve.test_rows]
    assert len(carve.test_rows) == 18
    assert int((test_labels == "yes").sum()) == 6  # 30 "yes" x 0.2
    assert sorted(carve.development_rows + carve.test_rows) == list(range(90))


def test_group_test_fold_keeps_groups_whole():
    frame = _frame()
    split = SplitConfig(strategy="group", group_column="customer", test_fraction=0.25)
    carve = carve_test_fold(frame, frame["label"], BINARY, split)
    test_groups = set(frame["customer"].iloc[carve.test_rows])
    development_groups = set(frame["customer"].iloc[carve.development_rows])
    assert test_groups and not test_groups & development_groups


def test_temporal_test_fold_is_strictly_later():
    frame = _frame()
    split = SplitConfig(strategy="temporal", time_column="day", test_fraction=0.2)
    carve = carve_test_fold(frame, frame["amount"], REGRESSION, split)
    assert (
        frame["day"].iloc[carve.test_rows].min() > frame["day"].iloc[carve.development_rows].max()
    )


def test_carve_uses_positions_not_index_labels():
    frame = _frame().set_axis(range(1000, 1060))
    carve = carve_test_fold(frame, frame["label"], BINARY, SplitConfig(test_fraction=0.2))
    assert sorted(carve.development_rows + carve.test_rows) == list(range(60))


@pytest.mark.parametrize(
    ("rows", "fraction", "labels"),
    [
        (10, 0.1, None),  # round(10 x 0.1) = 1 test row
        (12, 0.15, ["a", "b", "c"] * 4),  # 2 test rows for 3 classes: scikit-learn refuses
    ],
)
def test_test_fold_too_small(rows, fraction, labels):
    frame = _frame(rows)
    if labels is not None:
        frame["label"] = labels
    with pytest.raises(FoldPlanningError) as caught:
        carve_test_fold(frame, frame["label"], BINARY, SplitConfig(test_fraction=fraction))
    assert caught.value.reason is FoldPlanningReason.TEST_FOLD_TOO_SMALL
    assert caught.value.details["minimum"] == MIN_ROWS_PER_FOLD
    assert "test_rows" in caught.value.details


def test_missing_group_column_is_a_plain_splitting_error():
    frame = _frame()
    split = _cv(BINARY, strategy="group", group_column="nope")
    with pytest.raises(SplittingError) as caught:
        _plan(frame, split)
    assert not isinstance(caught.value, FoldPlanningError)


def test_scheme_must_match_the_strategy():
    frame = _frame()
    split = SplitConfig(
        strategy="random",
        evaluation_mode=EvaluationMode.CV,
        cv=CrossValidationConfig(folds=3, scheme=CvScheme.GROUP_KFOLD),
    )
    with pytest.raises(SplittingError, match="does not match"):
        _plan(frame, split)


# --- CV schemes ----------------------------------------------------------------------------


def test_stratified_kfold_balances_classes_and_excludes_the_test_fold():
    frame = _frame(90)
    plan = _plan(frame, _cv(BINARY, 3, test_fraction=0.2))
    values = _values(plan)
    assert plan.assignment.scheme is CvScheme.STRATIFIED_KFOLD
    assert plan.warnings == []
    assert int((values == TEST).sum()) == 18
    for label in ("yes", "no"):
        per_fold = [int(((values == f) & (frame["label"] == label)).sum()) for f in range(3)]
        assert max(per_fold) - min(per_fold) <= 1
    assert TRAIN_ONLY not in values


def test_stratified_kfold_falls_back_to_kfold_for_a_rare_class():
    frame = _frame(40)
    frame.loc[[0, 1], "label"] = "rare"
    plan = _plan(frame, _cv(ProblemType.MULTICLASS_CLASSIFICATION, 3))
    (warning,) = plan.warnings
    assert warning.rule == "cv_stratification_not_possible"
    assert warning.severity is Severity.HIGH
    assert warning.details == {"class": "rare", "class_count": 2, "folds": 3}
    assert sorted(set(plan.assignment.assignments)) == [0, 1, 2]


def test_kfold_for_regression_partitions_all_rows():
    frame = _frame(50)
    plan = _plan(frame, _cv(REGRESSION, 4), REGRESSION, "amount")
    counts = np.bincount(_values(plan))
    assert plan.assignment.scheme is CvScheme.KFOLD
    assert sorted(counts.tolist()) == [12, 12, 13, 13]


def test_group_kfold_isolates_groups_and_treats_missing_as_one_group():
    frame = _frame()
    frame.loc[[2, 9, 33, 50], "customer"] = None
    plan = _plan(frame, _cv(BINARY, 4, strategy="group", group_column="customer"))
    values = _values(plan)
    groups = frame["customer"].astype("string").fillna("<missing>")
    for group in groups.unique():
        assert len(set(values[(groups == group).to_numpy()])) == 1
    assert plan.assignment.group_column == "customer"
    assert [w.rule for w in plan.warnings] == ["group_column_missing_values"]


def test_group_kfold_with_test_fold_keeps_groups_out_of_development():
    frame = _frame()
    plan = _plan(
        frame, _cv(BINARY, 3, strategy="group", group_column="customer", test_fraction=0.25)
    )
    values = _values(plan)
    for group in frame["customer"].unique():
        assert len(set(values[(frame["customer"] == group).to_numpy()])) == 1


def test_groups_fewer_than_folds():
    frame = _frame()
    frame["customer"] = ["a", "b", "c"] * 20
    with pytest.raises(FoldPlanningError) as caught:
        _plan(frame, _cv(BINARY, 4, strategy="group", group_column="customer"))
    assert caught.value.reason is FoldPlanningReason.GROUPS_FEWER_THAN_FOLDS
    assert caught.value.details == {"group_count": 3, "folds": 4}


def test_forward_chaining_orders_folds_in_time():
    frame = _frame()
    frame.loc[[4, 17], "day"] = "not a date"
    plan = _plan(
        frame, _cv(REGRESSION, 3, strategy="temporal", time_column="day"), REGRESSION, "amount"
    )
    assignment = plan.assignment
    values = _values(plan)
    times = pd.to_datetime(frame["day"], errors="coerce", utc=True)
    assert values[4] == TRAIN_ONLY and values[17] == TRAIN_ONLY
    assert assignment.time_column == "day"
    for fold in range(3):
        train, validation = fold_rows(assignment, fold)
        assert times.iloc[train].max() < times.iloc[validation].min()
        assert {4, 17} <= set(train)
    # Block 0 (the earliest days) is never validated.
    earliest = times.idxmin()
    assert values[earliest] == TRAIN_ONLY
    assert [w.rule for w in plan.warnings] == ["time_column_unparseable_values"]


def test_forward_chaining_with_numeric_times_and_test_fold():
    frame = _frame()
    frame["tick"] = np.arange(60) // 2  # 30 distinct values, each twice
    split = _cv(REGRESSION, 4, strategy="temporal", time_column="tick", test_fraction=0.2)
    plan = _plan(frame, split, REGRESSION, "amount")
    values = _values(plan)
    ticks = frame["tick"].to_numpy()
    assert ticks[values == TEST].min() > ticks[values != TEST].max()
    for fold in range(4):
        train, validation = fold_rows(plan.assignment, fold)
        assert ticks[train].max() < ticks[validation].min()
        assert not set(np.flatnonzero(values == TEST)) & set(train)


def test_temporal_ties():
    frame = _frame()
    frame["tick"] = [0] * 30 + [1] * 30  # 2 distinct times cannot form K + 1 = 4 blocks
    with pytest.raises(FoldPlanningError) as caught:
        _plan(
            frame, _cv(REGRESSION, 3, strategy="temporal", time_column="tick"), REGRESSION, "amount"
        )
    assert caught.value.reason is FoldPlanningReason.TEMPORAL_TIES
    assert caught.value.details == {"distinct_times": 2, "folds": 3, "required": 4}


def test_fold_too_small_for_kfold():
    frame = _frame(5)
    with pytest.raises(FoldPlanningError) as caught:
        _plan(frame, _cv(REGRESSION, 3), REGRESSION, "amount")
    assert caught.value.reason is FoldPlanningReason.FOLD_TOO_SMALL
    assert caught.value.details["folds"] == 3
    assert caught.value.details["minimum"] == MIN_ROWS_PER_FOLD


def test_fold_too_small_for_group_kfold_with_a_tiny_fold():
    frame = _frame(20)
    frame["customer"] = ["big"] * 17 + ["s1", "s2", "s3"]
    with pytest.raises(FoldPlanningError) as caught:
        _plan(frame, _cv(BINARY, 4, strategy="group", group_column="customer"))
    assert caught.value.reason is FoldPlanningReason.FOLD_TOO_SMALL
    assert caught.value.details["row_count"] == 1


def test_fold_too_small_for_the_first_forward_chaining_block():
    frame = _frame(20)
    frame["tick"] = [0] + [1] * 6 + [2] * 6 + [3] * 7  # block 0 = the single earliest row
    with pytest.raises(FoldPlanningError) as caught:
        _plan(
            frame, _cv(REGRESSION, 3, strategy="temporal", time_column="tick"), REGRESSION, "amount"
        )
    assert caught.value.reason is FoldPlanningReason.FOLD_TOO_SMALL
    assert caught.value.details["part"] == "first training block"


# --- Single_Split_Mode with a Test_Fold, helpers, determinism -------------------------------


def test_single_split_with_test_fold_records_three_partitions():
    frame = _frame(80)
    plan = _plan(frame, SplitConfig(test_fraction=0.2))
    assignment = plan.assignment
    values = _values(plan)
    assert assignment.evaluation_mode is EvaluationMode.SINGLE_SPLIT
    assert assignment.folds == 0 and assignment.scheme is None
    assert set(values) == {TEST, TRAIN_ONLY, 0}
    assert int((values == TEST).sum()) == 16
    train, validation = fold_rows(assignment, 0)
    assert sorted(validation) == sorted(int(i) for i in plan.single_split.validation_index)
    assert sorted(train) == sorted(int(i) for i in plan.single_split.train_index)
    assert len(validation) == round(64 * 0.2)


def test_single_split_without_room_for_validation_is_fold_too_small():
    frame = _frame(8)
    with pytest.raises(FoldPlanningError) as caught:
        _plan(frame, SplitConfig(test_fraction=0.25, validation_fraction=0.1))
    assert caught.value.reason is FoldPlanningReason.FOLD_TOO_SMALL


def test_test_fold_does_not_depend_on_k_or_mode():
    frame = _frame(90)
    reference = folds.test_rows(_plan(frame, SplitConfig(test_fraction=0.2)).assignment)
    for k in (2, 3, 5):
        plan = _plan(frame, _cv(BINARY, k, test_fraction=0.2))
        assert folds.test_rows(plan.assignment) == reference


def test_fold_rows_helpers_and_bounds():
    assignment = FoldAssignment(
        experiment_id="e",
        generated_at="2026-01-01T00:00:00Z",
        evaluation_mode=EvaluationMode.CV,
        scheme=CvScheme.KFOLD,
        folds=2,
        test_fraction=0.2,
        random_seed=1,
        dataset_fingerprint="f",
        development_digest="d",
        row_ids=[0, 1, 2, 3, 4],
        assignments=[TEST, 0, 1, 0, 1],
    )
    assert fold_rows(assignment, 0) == ([2, 4], [1, 3])
    assert development_rows(assignment) == [1, 2, 3, 4]
    assert folds.test_rows(assignment) == [0]
    with pytest.raises(ValueError):
        fold_rows(assignment, 2)


def test_plan_is_deterministic_and_seed_sensitive():
    frame = _frame(90)
    first = _plan(frame, _cv(BINARY, 3, test_fraction=0.2))
    second = _plan(frame, _cv(BINARY, 3, test_fraction=0.2))
    assert first.assignment.model_dump(exclude={"generated_at"}) == second.assignment.model_dump(
        exclude={"generated_at"}
    )
    other = _plan(frame, _cv(BINARY, 3, test_fraction=0.2, random_seed=7))
    assert other.assignment.assignments != first.assignment.assignments


def test_assignment_records_its_inputs():
    frame = _frame()
    split = _cv(BINARY, 3, test_fraction=0.2, random_seed=11)
    assignment = _plan(frame, split).assignment
    assert assignment.dataset_fingerprint == "fp"
    assert assignment.experiment_id == "exp"
    assert (assignment.folds, assignment.test_fraction, assignment.random_seed) == (3, 0.2, 11)
    assert assignment.row_ids == list(range(60))
    assert len(assignment.development_digest) == 64
