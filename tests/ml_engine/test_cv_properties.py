"""Property-based tests of the cross-validation feature (design "Correctness Properties").

Every test runs 100 derandomized examples so the offline suite stays deterministic.
"""

import json
import statistics
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from ml_engine.contracts.common import (
    ClassWeighting,
    MetricDirection,
    ModelRunStatus,
    ProblemType,
    RequestedProblemType,
)
from ml_engine.contracts.comparison import ComparisonReport, ModelComparisonEntry
from ml_engine.contracts.config import (
    DatasetReference,
    EnvironmentCapture,
    ExperimentConfig,
    FeatureSelection,
    ModelSpec,
    PreprocessingConfig,
    SplitConfig,
)
from ml_engine.contracts.cv import (
    TEST,
    TRAIN_ONLY,
    CrossValidationConfig,
    CvScheme,
    EvaluationMode,
    FoldAssignment,
)
from ml_engine.contracts.metrics import MetricSet
from ml_engine.contracts.model import ModelMetadata
from ml_engine.contracts.preparation import PreparationReport, SplitSummary
from ml_engine.preprocessing import ColumnPlan, FeaturePipeline
from ml_engine.preprocessing.folds import (
    DEVELOPMENT_FIT,
    fit_fold_pipelines,
    fold_fit_label,
    pipeline_metadata,
)
from ml_engine.splitting import folds as planner
from ml_engine.splitting.folds import (
    FoldPlan,
    FoldPlanningError,
    FoldPlanningReason,
    fold_rows,
    plan_folds,
    resolve_cv_scheme,
)
from tests.support.golden import GOLDEN_FILES, golden_document

PBT = settings(max_examples=100, deadline=None, derandomize=True)

#: Every key the cross-validation feature adds to an inherited contract.
CV_KEYS = frozenset(
    {
        "evaluation_mode",
        "cv",
        "test_fraction",
        "cross_validation",
        "test_fold",
        "test_metrics",
        "test_evaluation",
    }
)

#: Prefixed so a generated metric or column name can never collide with a CV key.
names = st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=12).map(
    lambda text: f"n_{text}"
)
finite = st.floats(allow_nan=False, allow_infinity=False, width=64)
unit = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
moments = st.datetimes(timezones=st.just(UTC))
metric_values = st.dictionaries(names, finite, max_size=4)

splits = st.builds(
    SplitConfig,
    strategy=st.sampled_from(["stratified_random", "random", "group", "temporal"]),
    validation_fraction=st.floats(min_value=0.01, max_value=0.99),
    random_seed=st.integers(min_value=0, max_value=2**31 - 1),
    stratify=st.booleans(),
    group_column=st.none() | names,
    time_column=st.none() | names,
)

configs = st.builds(
    ExperimentConfig,
    experiment_id=names,
    name=names,
    created_at=moments,
    created_by=st.none() | names,
    dataset=st.builds(DatasetReference, uri=st.just("s3://mlf-test/x.csv"), file_format=names),
    target_column=names,
    problem_type=st.sampled_from(ProblemType),
    requested_problem_type=st.sampled_from(RequestedProblemType),
    primary_metric=names,
    feature_selection=st.builds(FeatureSelection, selected_features=st.lists(names, min_size=1)),
    split=splits,
    preprocessing=st.just(PreprocessingConfig()),
    models=st.lists(st.builds(ModelSpec, name=names), max_size=3),
    class_weighting=st.sampled_from(ClassWeighting),
    environment=st.builds(
        EnvironmentCapture,
        python_version=names,
        platform=names,
        ml_factory_version=names,
        captured_at=moments,
    ),
    artifact_prefix=st.just("s3://mlf-test/prefix"),
)

preparations = st.builds(
    PreparationReport,
    experiment_id=names,
    generated_at=moments,
    problem_type=st.sampled_from(ProblemType),
    target_column=names,
    requested_features=st.lists(names, max_size=4),
    usable_features=st.lists(names, max_size=4),
    rows_before=st.integers(min_value=0, max_value=10_000),
    rows_after=st.integers(min_value=0, max_value=10_000),
    rows_dropped_missing_target=st.integers(min_value=0, max_value=100),
    split=st.builds(
        SplitSummary,
        strategy=names,
        train_row_count=st.integers(min_value=0, max_value=10_000),
        validation_row_count=st.integers(min_value=0, max_value=10_000),
        validation_fraction_actual=unit,
        random_seed=st.integers(min_value=0, max_value=1000),
        stratified=st.booleans(),
    ),
)

metadatas = st.builds(
    ModelMetadata,
    experiment_id=names,
    model_name=names,
    display_name=names,
    status=st.sampled_from(ModelRunStatus),
    problem_type=st.sampled_from(ProblemType),
    library=names,
    library_version=names,
    primary_metric=st.none() | names,
    primary_score=st.none() | finite,
    metrics=st.builds(MetricSet, values=metric_values),
    train_metrics=st.none() | st.builds(MetricSet, values=metric_values),
    training_started_at=st.none() | moments,
)

comparisons = st.builds(
    ComparisonReport,
    experiment_id=names,
    generated_at=moments,
    problem_type=st.sampled_from(ProblemType),
    primary_metric=names,
    direction=st.sampled_from(MetricDirection),
    best_model=st.none() | names,
    best_score=st.none() | finite,
    models=st.lists(
        st.builds(
            ModelComparisonEntry,
            model_name=names,
            display_name=names,
            status=st.sampled_from(ModelRunStatus),
            primary_score=st.none() | finite,
            metrics=metric_values,
            rank=st.none() | st.integers(min_value=1, max_value=20),
        ),
        max_size=4,
    ),
)

contracts = st.one_of(splits, configs, preparations, metadatas, comparisons)


def _keys(value: Any) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for item in value.values() for k in _keys(item)}
    if isinstance(value, list):
        return {k for item in value for k in _keys(item)}
    return set()


def _split_of(model) -> SplitConfig | None:
    if isinstance(model, SplitConfig):
        return model
    if isinstance(model, ExperimentConfig):
        return model.split
    return None


# Feature: cross-validation, Property 14: Backward compatibility of contracts — contracts built
# without the new fields serialise with no new keys, and parsing that JSON yields
# evaluation_mode=single_split, no Test_Fold and all CV fields None.
@PBT
@given(model=contracts)
def test_property_14_pre_feature_contracts_serialise_and_parse_unchanged(model):
    text = model.model_dump_json(indent=2, by_alias=True)  # exactly what write_model writes
    document = json.loads(text)

    # No new keys anywhere, so the JSON is what the pre-feature contract produced.
    assert not (_keys(document) & CV_KEYS)

    parsed = type(model).model_validate_json(text)
    assert parsed == model
    assert parsed.model_dump_json(indent=2, by_alias=True) == text

    split = _split_of(parsed)
    if split is not None:
        assert split.evaluation_mode is EvaluationMode.SINGLE_SPLIT
        assert split.cv is None and split.test_fraction is None
    for field in ("cross_validation", "test_fold", "test_evaluation"):
        assert getattr(parsed, field, None) is None
    for entry in getattr(parsed, "models", []):
        if isinstance(entry, ModelComparisonEntry):
            assert entry.cv is None and entry.test_metrics is None


def test_property_14_golden_fixtures_parse_with_single_split_semantics():
    """The pre-feature fixtures (task 1.1) carry no CV key and parse with defaults."""
    for name in GOLDEN_FILES:
        assert not (_keys(golden_document(name)) & CV_KEYS), name
    config = ExperimentConfig.model_validate(golden_document("experiment_config.json"))
    assert config.split.evaluation_mode is EvaluationMode.SINGLE_SPLIT
    assert config.split.cv is None and config.split.test_fraction is None
    comparison = ComparisonReport.model_validate(golden_document("comparison.json"))
    assert comparison.cross_validation is None and comparison.test_evaluation is None
    assert all(e.cv is None and e.test_metrics is None for e in comparison.models)


# ============================================================================================
# Fold_Planner (design C2): Properties 1-6
# ============================================================================================

KNOWN_REASONS = frozenset(FoldPlanningReason)
_STRATEGY_OF_SCHEME = {
    CvScheme.STRATIFIED_KFOLD: "stratified_random",
    CvScheme.KFOLD: "random",
    CvScheme.GROUP_KFOLD: "group",
    CvScheme.FORWARD_CHAINING: "temporal",
}
_BASE_DAY = date(2024, 1, 1)


@dataclass(frozen=True)
class PlanningInput:
    frame: pd.DataFrame
    problem_type: ProblemType
    split: SplitConfig

    @property
    def target(self) -> pd.Series:
        return self.frame["target"]

    def with_split(self, **update) -> "PlanningInput":
        return PlanningInput(self.frame, self.problem_type, self.split.model_copy(update=update))


def _cv_split(base: SplitConfig, problem_type: ProblemType, folds: int) -> SplitConfig:
    scheme = resolve_cv_scheme(problem_type, base)
    return base.model_copy(
        update={
            "evaluation_mode": EvaluationMode.CV,
            "cv": CrossValidationConfig(folds=folds, scheme=scheme),
        }
    )


@st.composite
def planning_inputs(
    draw,
    schemes: tuple[CvScheme, ...] = tuple(CvScheme),
    modes: tuple[EvaluationMode, ...] = (EvaluationMode.CV, EvaluationMode.SINGLE_SPLIT),
    test_fold: st.SearchStrategy = st.none() | st.floats(min_value=0.05, max_value=0.45),
) -> PlanningInput:
    """Small frames (8-200 rows) with a random class balance, a group column of random
    cardinality with missing values, and a time column with duplicates and unparseable values
    (ISO dates with junk, or numbers with gaps)."""
    rows = draw(st.integers(min_value=8, max_value=200))
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
    scheme = draw(st.sampled_from(schemes))
    mode = draw(st.sampled_from(modes))
    folds = draw(st.integers(min_value=2, max_value=5))

    if scheme is CvScheme.STRATIFIED_KFOLD:
        classification = True
    elif scheme is CvScheme.KFOLD:
        classification = draw(st.booleans())
    else:
        classification = draw(st.booleans())
    classes = draw(st.integers(min_value=2, max_value=4))
    if classification:
        weights = rng.dirichlet(np.full(classes, draw(st.floats(min_value=0.3, max_value=5.0))))
        target = rng.choice([f"c{i}" for i in range(classes)], size=rows, p=weights)
        problem_type = (
            ProblemType.BINARY_CLASSIFICATION
            if classes == 2
            else ProblemType.MULTICLASS_CLASSIFICATION
        )
    else:
        target = rng.normal(size=rows)
        problem_type = ProblemType.REGRESSION

    group_count = draw(st.integers(min_value=1, max_value=40))
    groups = pd.Series([f"g{i}" for i in rng.integers(0, group_count, rows)], dtype=object)
    groups[rng.random(rows) < draw(st.floats(min_value=0.0, max_value=0.3))] = None

    distinct = draw(st.integers(min_value=1, max_value=rows))
    ticks = rng.integers(0, distinct, rows)
    unparseable = rng.random(rows) < draw(st.floats(min_value=0.0, max_value=0.2))
    if draw(st.sampled_from(["date", "number"])) == "date":
        times = pd.Series(
            [(_BASE_DAY + timedelta(days=int(t))).isoformat() for t in ticks], dtype=object
        )
        times[unparseable] = draw(st.sampled_from(["not a date", None]))
    else:
        times = pd.Series(ticks.astype(float))
        times[unparseable] = np.nan

    frame = pd.DataFrame({"x": rng.normal(size=rows), "grp": groups, "ts": times, "target": target})
    strategy = _STRATEGY_OF_SCHEME[scheme]
    if scheme is CvScheme.KFOLD and not classification and draw(st.booleans()):
        strategy = "stratified_random"  # stratified_random on regression is shuffled K-fold too
    base = SplitConfig(
        strategy=strategy,
        validation_fraction=draw(st.floats(min_value=0.1, max_value=0.4)),
        random_seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        group_column="grp",
        time_column="ts",
        test_fraction=draw(test_fold),
    )
    split = _cv_split(base, problem_type, folds) if mode is EvaluationMode.CV else base
    if split.cv is not None:
        assert split.cv.scheme is scheme
    return PlanningInput(frame, problem_type, split)


def _plan(inputs: PlanningInput) -> FoldPlan:
    return plan_folds(
        inputs.frame,
        inputs.target,
        inputs.problem_type,
        inputs.split,
        fingerprint=planner.dataset_fingerprint(inputs.frame, list(inputs.frame.columns)),
        experiment_id="exp-pbt",
    )


def _plan_or_reject(inputs: PlanningInput) -> FoldPlan:
    """Inputs that legitimately cannot be planned are discarded; the error must be known."""
    try:
        return _plan(inputs)
    except FoldPlanningError as error:
        assert error.reason in KNOWN_REASONS
        assert error.details, error.reason
        assume(False)
        raise  # unreachable: assume(False) aborts the example


def _comparable(plan: FoldPlan) -> tuple[dict, list[dict]]:
    return (
        plan.assignment.model_dump(mode="json", exclude={"generated_at"}),
        [warning.model_dump(mode="json") for warning in plan.warnings],
    )


def _test_set(plan: FoldPlan) -> list[int]:
    return planner.test_rows(plan.assignment)


# Feature: cross-validation, Property 1: Fold assignment is deterministic — planning twice on
# the same frame, problem type, strategy, K, Test_Fold fraction and seed gives equal
# FoldAssignments (ignoring generated_at) and equal warnings.
@PBT
@given(inputs=planning_inputs())
def test_property_1_fold_assignment_is_deterministic(inputs):
    first = _plan_or_reject(inputs)
    np.random.seed(12345)  # global RNG state must not matter
    second = _plan(inputs)
    assert _comparable(first) == _comparable(second)


# Feature: cross-validation, Property 2: Fold assignment partitions the usable rows — each row
# is in exactly one of {Test_Fold, TRAIN_ONLY, fold 0..K-1}; TRAIN_ONLY only for
# forward_chaining; folds non-empty; fold training rows are exactly the expected Development_Set
# rows and never Test_Fold rows; the Test_Fold does not depend on K or the mode.
@PBT
@given(inputs=planning_inputs(), other_folds=st.integers(min_value=2, max_value=5))
def test_property_2_fold_assignment_partitions_the_usable_rows(inputs, other_folds):
    plan = _plan_or_reject(inputs)
    assignment = plan.assignment
    rows = len(inputs.frame)
    values = np.asarray(assignment.assignments)

    assert assignment.row_ids == list(range(rows))
    assert len(values) == rows
    cv_mode = assignment.evaluation_mode is EvaluationMode.CV
    k = assignment.folds
    allowed = {TEST, TRAIN_ONLY} | (set(range(k)) if cv_mode else {0})
    assert set(values.tolist()) <= allowed
    test = set(np.flatnonzero(values == TEST).tolist())
    if inputs.split.test_fraction is None:
        assert not test

    if cv_mode:
        forward = assignment.scheme is CvScheme.FORWARD_CHAINING
        if not forward:
            assert TRAIN_ONLY not in values
        sizes = [int((values == fold).sum()) for fold in range(k)]
        assert all(size > 0 for size in sizes)
        if assignment.scheme is CvScheme.KFOLD:  # shuffled K-fold: near-equal folds (Req 2.2)
            assert max(sizes) - min(sizes) <= 1
        development = set(range(rows)) - test
        train_only = set(np.flatnonzero(values == TRAIN_ONLY).tolist())
        for fold in range(k):
            train, validation = fold_rows(assignment, fold)
            assert set(validation) == set(np.flatnonzero(values == fold).tolist())
            if forward:
                earlier = set(np.flatnonzero((values >= 0) & (values < fold)).tolist())
                assert set(train) == train_only | earlier
            else:
                assert set(train) == development - set(validation)
            assert not (set(train) | set(validation)) & test
            assert len(train) == len(set(train)) and len(validation) == len(set(validation))
    else:
        train, validation = fold_rows(assignment, 0)
        assert set(train) | set(validation) | test == set(range(rows))
        assert not set(train) & set(validation)

    # The Test_Fold is carved first: identical for another K and for Single_Split_Mode.
    variants = [
        inputs.with_split(evaluation_mode=EvaluationMode.SINGLE_SPLIT, cv=None),
        PlanningInput(
            inputs.frame,
            inputs.problem_type,
            _cv_split(
                inputs.split.model_copy(
                    update={"evaluation_mode": EvaluationMode.SINGLE_SPLIT, "cv": None}
                ),
                inputs.problem_type,
                other_folds,
            ),
        ),
    ]
    for variant in variants:
        try:
            other = _plan(variant)
        except FoldPlanningError as error:
            # The carve-out is shared, so only the K- or mode-specific step may fail.
            assert error.reason is not FoldPlanningReason.TEST_FOLD_TOO_SMALL
            continue
        assert _test_set(other) == sorted(test)


# Feature: cross-validation, Property 3: Group isolation — under group_kfold all rows sharing a
# group value (missing values form one group) have the same assignment value, so no group is in
# two CV_Folds or in both the Test_Fold and the Development_Set.
@PBT
@given(inputs=planning_inputs(schemes=(CvScheme.GROUP_KFOLD,), modes=(EvaluationMode.CV,)))
def test_property_3_group_isolation(inputs):
    plan = _plan_or_reject(inputs)
    values = pd.Series(plan.assignment.assignments)
    groups = inputs.frame["grp"].astype("string").fillna("<missing>")
    per_group = values.groupby(groups.to_numpy()).nunique()
    assert (per_group == 1).all(), per_group[per_group > 1].to_dict()
    assert plan.assignment.group_column == "grp"


# Feature: cross-validation, Property 4: Temporal ordering — under forward_chaining every fold's
# training times are strictly earlier than its validation times; rows without a parseable time
# are never validated nor in the Test_Fold; every Test_Fold time is strictly later than every
# Development_Set time.
@PBT
@given(inputs=planning_inputs(schemes=(CvScheme.FORWARD_CHAINING,), modes=(EvaluationMode.CV,)))
def test_property_4_temporal_ordering(inputs):
    plan = _plan_or_reject(inputs)
    assignment = plan.assignment
    values = np.asarray(assignment.assignments)
    times = planner.parse_times(inputs.frame, "ts")
    known = times.notna().to_numpy()

    assert (values[~known] == TRAIN_ONLY).all()
    for fold in range(assignment.folds):
        train, validation = fold_rows(assignment, fold)
        assert validation and known[validation].all()
        train_known = [row for row in train if known[row]]
        assert train_known
        assert times.iloc[train_known].max() < times.iloc[validation].min()
    test = values == TEST
    if test.any():
        assert known[test].all()
        assert times[test].min() > times[~test & known].max()


#: Found by Hypothesis: scikit-learn sizes the Test_Fold as ceil(0.1586 x 183) = 30 rows, so
#: proportions are preserved relative to the actual share 30/183, not the requested fraction.
#: Kept as a pinned example of the rounding case.
_P5_TEST_FOLD_COUNTEREXAMPLE = PlanningInput(
    frame=pd.DataFrame({"x": np.arange(183.0), "target": ["a"] * 155 + ["b"] * 14 + ["c"] * 14}),
    problem_type=ProblemType.MULTICLASS_CLASSIFICATION,
    split=_cv_split(
        SplitConfig(test_fraction=0.15864161393205867),
        ProblemType.MULTICLASS_CLASSIFICATION,
        2,
    ),
)


# Feature: cross-validation, Property 5: Stratified folds preserve class proportions — when every
# class has at least K Development_Set rows, each class's count differs by at most one between
# any two CV_Folds (and the rare-class fallback warning is raised exactly when one has fewer).
@PBT
@given(inputs=planning_inputs(schemes=(CvScheme.STRATIFIED_KFOLD,), modes=(EvaluationMode.CV,)))
def test_property_5_stratified_folds_preserve_class_proportions(inputs):
    plan = _plan_or_reject(inputs)
    rules = {warning.rule for warning in plan.warnings}
    values = np.asarray(plan.assignment.assignments)
    labels = inputs.target.astype(str).to_numpy()
    dev_counts = pd.Series(labels[values != TEST]).value_counts()
    k = plan.assignment.folds

    if (dev_counts >= k).all():
        assert "cv_stratification_not_possible" not in rules
        for label in dev_counts.index:
            per_fold = [int(((values == fold) & (labels == label)).sum()) for fold in range(k)]
            assert max(per_fold) - min(per_fold) <= 1, (label, per_fold)
    else:
        assert "cv_stratification_not_possible" in rules


# Feature: cross-validation, Property 5: Stratified folds preserve class proportions — a
# stratified Test_Fold holds each class within one row of (test rows / usable rows) x class count.
@PBT
@example(inputs=_P5_TEST_FOLD_COUNTEREXAMPLE)
@given(
    inputs=planning_inputs(
        schemes=(CvScheme.STRATIFIED_KFOLD,),
        modes=(EvaluationMode.CV,),
        test_fold=st.floats(min_value=0.05, max_value=0.45),
    )
)
def test_property_5_stratified_test_fold_preserves_class_proportions(inputs):
    plan = _plan_or_reject(inputs)
    rules = {warning.rule for warning in plan.warnings}
    assume("stratification_not_possible" not in rules)  # the inherited rare-class fallback
    values = np.asarray(plan.assignment.assignments)
    labels = inputs.target.astype(str).to_numpy()
    fraction = float((values == TEST).sum()) / len(values)  # the actual Test_Fold share
    for label, count in pd.Series(labels).value_counts().items():
        in_test = int(((values == TEST) & (labels == label)).sum())
        assert abs(in_test - fraction * count) <= 1, (label, in_test, fraction, count)


@st.composite
def fold_assignments(draw) -> FoldAssignment:
    mode = draw(st.sampled_from(EvaluationMode))
    if mode is EvaluationMode.CV:
        scheme = draw(st.sampled_from(CvScheme))
        k = draw(st.integers(min_value=2, max_value=10))
        allowed = [TEST, *range(k)] + ([TRAIN_ONLY] if scheme is CvScheme.FORWARD_CHAINING else [])
        required = list(range(k))
    else:
        scheme, k = None, 0
        allowed, required = [TEST, TRAIN_ONLY, 0], []
    extra = draw(st.lists(st.sampled_from(allowed), max_size=60))
    assignments = draw(st.permutations(required + extra))
    row_ids = sorted(
        draw(
            st.sets(
                st.integers(min_value=0, max_value=10_000),
                min_size=len(assignments),
                max_size=len(assignments),
            )
        )
    )
    return FoldAssignment(
        experiment_id=draw(names),
        generated_at=draw(moments),
        evaluation_mode=mode,
        scheme=scheme,
        folds=k,
        test_fraction=draw(st.none() | st.floats(min_value=0.01, max_value=0.49)),
        random_seed=draw(st.integers(min_value=0, max_value=2**31 - 1)),
        dataset_fingerprint=draw(st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)),
        development_digest=draw(st.text(alphabet="0123456789abcdef", min_size=64, max_size=64)),
        row_ids=row_ids,
        assignments=assignments,
        group_column=draw(st.none() | names),
        time_column=draw(st.none() | names),
    )


# Feature: cross-validation, Property 6: Fold assignment serialization round trip — parsing the
# JSON of any valid FoldAssignment gives an equal object with the same fingerprint, scheme, K,
# Test_Fold fraction, seed and schema version.
@PBT
@given(assignment=fold_assignments())
def test_property_6_fold_assignment_round_trip(assignment):
    parsed = FoldAssignment.model_validate_json(assignment.model_dump_json())
    assert parsed == assignment
    assert (
        parsed.dataset_fingerprint,
        parsed.scheme,
        parsed.folds,
        parsed.test_fraction,
        parsed.random_seed,
        parsed.schema_version,
    ) == (
        assignment.dataset_fingerprint,
        assignment.scheme,
        assignment.folds,
        assignment.test_fraction,
        assignment.random_seed,
        assignment.schema_version,
    )


# Feature: cross-validation, Property 6: Fold assignment serialization round trip — also for
# assignments produced by the planner itself.
@PBT
@given(inputs=planning_inputs())
def test_property_6_planned_assignment_round_trip(inputs):
    assignment = _plan_or_reject(inputs).assignment
    parsed = FoldAssignment.model_validate_json(assignment.model_dump_json())
    assert parsed == assignment


# ============================================================================================
# Fold-aware preprocessing (design C3): Properties 7-8
# ============================================================================================


@dataclass(frozen=True)
class PreprocessingInput:
    frame: pd.DataFrame  # every usable row, indexed by row id, Test_Fold rows included
    assignment: FoldAssignment
    plan: ColumnPlan
    config: PreprocessingConfig
    strategy: str


def _with_features(frame: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Numeric with missing values, a categorical with missing values, a boolean-ish column."""
    rows = len(frame)
    frame = frame.copy()
    num = rng.normal(10.0, 3.0, rows)
    num[rng.random(rows) < 0.25] = np.nan
    frame["num"] = num
    cat = pd.Series(rng.choice(["a", "b", "c", "d", "e"], rows, p=[0.4, 0.3, 0.15, 0.1, 0.05]))
    cat[rng.random(rows) < 0.2] = None
    frame["cat"] = cat.astype(object)
    flag = pd.Series(rng.choice(["yes", "no"], rows), dtype=object)
    flag[rng.random(rows) < 0.15] = None
    frame["flag"] = flag
    return frame


@st.composite
def preprocessing_inputs(draw) -> PreprocessingInput:
    inputs = draw(planning_inputs(modes=(EvaluationMode.CV,)))
    plan = _plan_or_reject(inputs)
    rng = np.random.default_rng(draw(st.integers(min_value=0, max_value=2**32 - 1)))
    frame = _with_features(inputs.frame, rng)
    numeric = draw(st.sampled_from([["x", "num"], ["num"], ["x"], []]))
    categorical = draw(st.sampled_from([["cat"], []]))
    boolean = draw(st.sampled_from([["flag"], []]))
    assume(numeric or categorical or boolean)
    config = PreprocessingConfig(
        numeric_imputation=draw(st.sampled_from(["median", "mean", "constant"])),
        categorical_imputation=draw(st.sampled_from(["constant", "most_frequent"])),
        scale_numeric=draw(st.booleans()),
        one_hot_max_categories=draw(st.integers(min_value=2, max_value=6)),
        one_hot_min_frequency=draw(st.none() | st.sampled_from([0.05, 0.2])),
    )
    return PreprocessingInput(
        frame=frame,
        assignment=plan.assignment,
        plan=ColumnPlan(numeric=numeric, categorical=categorical, boolean=boolean),
        config=config,
        strategy=draw(st.sampled_from(["dense_numeric", "native_categorical"])),
    )


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            return ("object-array", value.shape, [_canonical(item) for item in value.tolist()])
        return ("array", str(value.dtype), value.shape, value.tobytes())
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, float | np.floating):
        return ("float", float(value).hex())
    if isinstance(value, int | np.integer | str | bool) or value is None:
        return value
    return ("repr", repr(value))


def fitted_statistics(pipeline: FeaturePipeline) -> dict[str, Any]:
    """Every learned attribute of every fitted step (imputer statistics_, scaler mean_/scale_/
    var_/n_samples_seen_, encoder categories_, ...), plus the output feature names."""
    stats: dict[str, Any] = {"feature_names": pipeline.feature_names}
    for block, transformer, _columns in pipeline._transformer.transformers_:
        if block == "remainder":
            continue
        steps = transformer.steps if hasattr(transformer, "steps") else [("", transformer)]
        for step, estimator in steps:
            for attribute, value in sorted(vars(estimator).items()):
                if attribute.endswith("_") and not attribute.startswith("_"):
                    stats[f"{block}.{step}.{attribute}"] = _canonical(value)
    return stats


def _training_mask(assignment: FoldAssignment, fold: int | None) -> np.ndarray:
    """Training rows computed from the raw values, independently of ``fold_rows``."""
    values = np.asarray(assignment.assignments)
    if fold is None:
        return values != TEST
    if assignment.scheme is CvScheme.FORWARD_CHAINING:
        return (values == TRAIN_ONLY) | ((values >= 0) & (values < fold))
    return (values != TEST) & (values != fold)


def _reference(inputs: PreprocessingInput, mask: np.ndarray) -> FeaturePipeline:
    rows = inputs.frame[mask].reset_index(drop=True)  # a fresh frame holding only these rows
    pipeline = FeaturePipeline(inputs.plan, inputs.config, strategy=inputs.strategy)
    return pipeline.fit(rows, rows["target"])


def _fit_all(inputs: PreprocessingInput, frame: pd.DataFrame | None = None):
    # fit_fold_pipelines returns (per_fold, full, fold_treatments, development_treatment, applied);
    # with no treatment config the treatment entries are None/empty. These properties only check
    # the pipeline fits, so keep the historical (per_fold, full) shape.
    per_fold, full, *_rest = fit_fold_pipelines(
        inputs.plan,
        inputs.config,
        [inputs.strategy],
        inputs.frame if frame is None else frame,
        inputs.assignment,
        "target",
    )
    return per_fold, full


# Feature: cross-validation, Property 7: Preprocessing fit isolation — the fitted statistics of
# fold i's pipeline equal those of a pipeline fitted directly on fold i's training rows alone,
# and the full pipeline equals one fitted on the Development_Set rows alone (Test_Fold rows never
# included).
@PBT
@given(inputs=preprocessing_inputs())
def test_property_7_preprocessing_fit_isolation(inputs):
    per_fold, full = _fit_all(inputs)
    strategy = inputs.strategy
    assert len(per_fold) == inputs.assignment.folds

    for fold, pipelines in enumerate(per_fold):
        mask = _training_mask(inputs.assignment, fold)
        fitted = pipelines[strategy]
        assert fitted_statistics(fitted) == fitted_statistics(_reference(inputs, mask))
        assert pipeline_metadata(fitted).fitted_on == fold_fit_label(fold)
        blocks = {name: t for name, t, _ in fitted._transformer.transformers_}
        numeric = blocks.get("numeric")
        if numeric is not None and "scale" in numeric.named_steps:
            # Scaling follows imputation, so it saw exactly the fold's training rows.
            seen = np.atleast_1d(numeric.named_steps["scale"].n_samples_seen_)
            assert (seen == int(mask.sum())).all()

    development = _training_mask(inputs.assignment, None)
    assert fitted_statistics(full[strategy]) == fitted_statistics(_reference(inputs, development))
    assert pipeline_metadata(full[strategy]).fitted_on == DEVELOPMENT_FIT


def _scramble(frame: pd.DataFrame, rows: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """Arbitrary new feature values (other scales, unseen categories, missing) on ``rows``."""
    changed = frame.copy()
    count = int(rows.sum())
    if not count:
        return changed
    changed.loc[rows, "x"] = rng.normal(1e3, 1e2, count)
    changed.loc[rows, "num"] = np.where(rng.random(count) < 0.5, np.nan, rng.normal(-50, 9, count))
    changed.loc[rows, "cat"] = rng.choice(["zz", "unseen", None], count)
    changed.loc[rows, "flag"] = rng.choice(["yes", "no", None], count)
    return changed


# Feature: cross-validation, Property 8: Validation independence of fold preprocessing —
# replacing the feature values of fold i's validation rows (or of Test_Fold rows) with arbitrary
# values leaves fold i's fitted pipeline statistics unchanged.
@PBT
@given(
    inputs=preprocessing_inputs(),
    fold_choice=st.integers(min_value=0, max_value=100),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_property_8_validation_independence(inputs, fold_choice, seed):
    rng = np.random.default_rng(seed)
    fold = fold_choice % inputs.assignment.folds
    values = np.asarray(inputs.assignment.assignments)
    strategy = inputs.strategy
    per_fold, full = _fit_all(inputs)

    scrambled = _scramble(inputs.frame, (values == fold) | (values == TEST), rng)
    changed_fold, _ = _fit_all(inputs, scrambled)
    assert fitted_statistics(changed_fold[fold][strategy]) == fitted_statistics(
        per_fold[fold][strategy]
    )

    test_only = _scramble(inputs.frame, values == TEST, rng)
    changed_folds, changed_full = _fit_all(inputs, test_only)
    assert fitted_statistics(changed_full[strategy]) == fitted_statistics(full[strategy])
    for index, pipelines in enumerate(changed_folds):
        assert fitted_statistics(pipelines[strategy]) == fitted_statistics(
            per_fold[index][strategy]
        )


# ============================================================================================
# Out-of-fold predictions (design C5): Property 9
# ============================================================================================


@st.composite
def oof_inputs(draw) -> tuple[FoldAssignment, list[pd.DataFrame]]:
    """A CV assignment (with Test_Fold and, for forward chaining, TRAIN_ONLY rows) and one
    prediction slice per fold covering exactly that fold's validation rows, in any order."""
    from jobs.training.cv import FOLD_INDEX, ROW_ID, Y_PRED, Y_TRUE

    scheme = draw(st.sampled_from(CvScheme))
    k = draw(st.integers(min_value=2, max_value=6))
    allowed = [TEST, *range(k)] + ([TRAIN_ONLY] if scheme is CvScheme.FORWARD_CHAINING else [])
    extra = draw(st.lists(st.sampled_from(allowed), max_size=50))
    assignments = draw(st.permutations([*range(k), *extra]))
    row_ids = sorted(
        draw(
            st.sets(
                st.integers(min_value=0, max_value=5_000),
                min_size=len(assignments),
                max_size=len(assignments),
            )
        )
    )
    assignment = FoldAssignment(
        experiment_id="e",
        generated_at=datetime(2026, 1, 1, tzinfo=UTC),
        evaluation_mode=EvaluationMode.CV,
        scheme=scheme,
        folds=k,
        test_fraction=None,
        random_seed=0,
        dataset_fingerprint="0" * 64,
        development_digest="0" * 64,
        row_ids=row_ids,
        assignments=assignments,
    )
    slices = []
    for fold in range(k):
        _, validation = planner.fold_rows(assignment, fold)
        ordered = draw(st.permutations(validation))
        slices.append(
            pd.DataFrame(
                {
                    ROW_ID: pd.Series(ordered, dtype="int64"),
                    FOLD_INDEX: pd.Series([fold] * len(ordered), dtype="int64"),
                    Y_TRUE: pd.Series(["a"] * len(ordered), dtype=object),
                    Y_PRED: pd.Series(["b"] * len(ordered), dtype=object),
                    "proba_a": pd.Series([0.25] * len(ordered), dtype="float64"),
                }
            )
        )
    return assignment, draw(st.permutations(slices))


# Feature: cross-validation, Property 9: Out-of-fold coverage — assemble_oof returns exactly one
# prediction for every row validated in a CV_Fold, tagged with that row's fold index, and none for
# Test_Fold or TRAIN_ONLY rows; slices that overlap, miss a row, or contain a foreign row are
# rejected.
@PBT
@given(
    inputs=oof_inputs(),
    corruption=st.sampled_from(["overlap", "gap", "foreign", "test_row", "wrong_fold"]),
    pick=st.integers(min_value=0, max_value=10_000),
)
def test_property_9_out_of_fold_coverage(inputs, corruption, pick):
    from jobs.training.cv import FOLD_INDEX, ROW_ID, OofAssemblyError, assemble_oof

    assignment, slices = inputs
    oof = assemble_oof(slices, assignment)

    value_of = dict(zip(assignment.row_ids, assignment.assignments, strict=True))
    validated = sorted(row for row, value in value_of.items() if value >= 0)
    assert oof[ROW_ID].tolist() == validated  # exactly once each, ordered by row id
    assert all(
        value_of[row] == fold for row, fold in zip(oof[ROW_ID], oof[FOLD_INDEX], strict=True)
    )
    assert not {row for row, value in value_of.items() if value < 0} & set(oof[ROW_ID])

    # Every corruption is rejected.
    slices = [piece.copy() for piece in slices]
    nonempty = [i for i, piece in enumerate(slices) if len(piece)]
    target = nonempty[pick % len(nonempty)]
    piece = slices[target]
    row = pick % len(piece)
    if corruption == "overlap":
        slices.append(piece.iloc[[row]].copy())
    elif corruption == "gap":
        slices[target] = piece.drop(piece.index[row])
    elif corruption == "foreign":
        foreign = piece.iloc[[row]].copy()
        foreign[ROW_ID] = max(assignment.row_ids) + 1 + pick
        slices.append(foreign)
    elif corruption == "test_row":
        unvalidated = [r for r, v in value_of.items() if v < 0]
        assume(unvalidated)
        extra = piece.iloc[[row]].copy()
        extra[ROW_ID] = unvalidated[pick % len(unvalidated)]
        slices.append(extra)
    else:
        moved = piece.copy()
        moved.loc[moved.index[row], FOLD_INDEX] = (int(moved[FOLD_INDEX].iloc[row]) + 1) % (
            assignment.folds
        )
        slices[target] = moved
    with pytest.raises(OofAssemblyError):
        assemble_oof(slices, assignment)


# ============================================================================================
# CV aggregation, ranking and stability (design C6): Properties 10-12
# ============================================================================================

from ml_engine.contracts.cv import (  # noqa: E402
    CvRunSummary,
    FoldMetrics,
    FoldResult,
    MetricAggregate,
)
from ml_engine.reporting.cv import (  # noqa: E402
    instability_warnings,
    rank_cv,
    summarize_folds,
)


def _fold_result(fold_index: int, values: dict[str, float], primary: str) -> FoldResult:
    moment = datetime(2026, 1, 1, tzinfo=UTC)
    return FoldResult(
        model_name="m",
        fold_index=fold_index,
        assignment_digest="a",
        params_digest="p",
        train_row_count=10,
        validation_row_count=5,
        metrics=MetricSet(values=dict(values)),
        primary_score=values.get(primary),
        started_at=moment,
        completed_at=moment,
        duration_seconds=0.0,
    )


@st.composite
def fold_metrics_inputs(draw) -> tuple[FoldMetrics, str]:
    """A Run's FoldMetrics: K folds, the same metric names on every fold, one of them primary."""
    k = draw(st.integers(min_value=2, max_value=8))
    metric_names = draw(st.lists(names, min_size=1, max_size=4, unique=True))
    primary = draw(st.sampled_from(metric_names))
    bounded = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
    folds = [
        _fold_result(i, {name: draw(bounded) for name in metric_names}, primary) for i in range(k)
    ]
    metrics = FoldMetrics(experiment_id="e", model_name="m", scheme=CvScheme.KFOLD, folds=folds)
    return metrics, primary


# Feature: cross-validation, Property 10: CV aggregates are correct and ordered — for every
# metric, summarize_folds reports the sample std, the fold count, and min <= mean <= max; the
# mean equals statistics.fmean of the fold values (clamped into [min, max]).
@PBT
@given(inputs=fold_metrics_inputs())
def test_property_10_cv_aggregates_are_correct_and_ordered(inputs):
    fold_metrics, primary = inputs
    summary = summarize_folds(fold_metrics, primary)

    assert summary.folds == len(fold_metrics.folds)
    for name, aggregate in summary.metrics.items():
        values = [f.metrics.values[name] for f in fold_metrics.folds]
        assert aggregate.fold_count == len(values)
        assert aggregate.min == pytest.approx(min(values), abs=0, rel=0)
        assert aggregate.max == pytest.approx(max(values), abs=0, rel=0)
        assert aggregate.min <= aggregate.mean <= aggregate.max
        clamped = min(max(statistics.fmean(values), min(values)), max(values))
        assert aggregate.mean == pytest.approx(clamped, abs=1e-9)
        expected_std = statistics.stdev(values) if len(values) > 1 else 0.0
        assert aggregate.std == pytest.approx(expected_std, abs=1e-9)
    assert summary.primary_fold_values == [f.primary_score for f in fold_metrics.folds]


def _entry(model_name: str, mean: float, std: float, primary: str) -> ModelComparisonEntry:
    folds = 3
    return ModelComparisonEntry(
        model_name=model_name,
        display_name=model_name,
        status=ModelRunStatus.COMPLETED,
        primary_score=mean,
        cv=CvRunSummary(
            folds=folds,
            succeeded_folds=folds,
            metrics={
                primary: MetricAggregate(mean=mean, std=std, min=mean, max=mean, fold_count=folds)
            },
            primary_fold_values=[mean] * folds,
        ),
    )


@st.composite
def ranking_inputs(draw):
    """A set of entries with a CV summary (mean, std) and a direction."""
    primary = draw(names)
    direction = draw(st.sampled_from(MetricDirection))
    count = draw(st.integers(min_value=1, max_value=6))
    model_names = draw(st.lists(names, min_size=count, max_size=count, unique=True))
    means = st.floats(min_value=-1000, max_value=1000, allow_nan=False)
    stds = st.floats(min_value=0.0, max_value=100, allow_nan=False)
    entries = [_entry(name, draw(means), draw(stds), primary) for name in model_names]
    return entries, direction, primary


# Feature: cross-validation, Property 11: CV ranking order, independent of test results — rank_cv
# orders rankable Runs by direction-adjusted mean primary metric, then lower std, then model
# name, assigns ranks 1..n, and attaching arbitrary Test_Fold metrics leaves ranks and best
# unchanged (Req 3.8).
@PBT
@given(inputs=ranking_inputs(), test_values=st.lists(finite, max_size=6))
def test_property_11_cv_ranking_order_independent_of_test(inputs, test_values):
    entries, direction, primary = inputs
    best_model, best_score = rank_cv(entries, direction, primary_metric=primary)

    maximize = direction is MetricDirection.MAXIMIZE
    ranked = sorted(
        entries,
        key=lambda e: (
            -e.cv.metrics[primary].mean if maximize else e.cv.metrics[primary].mean,
            e.cv.metrics[primary].std,
            e.model_name,
        ),
    )
    assert [e.model_name for e in ranked] == [
        e.model_name for e in sorted(entries, key=lambda e: e.rank)
    ]
    assert [e.rank for e in sorted(entries, key=lambda e: e.rank)] == list(
        range(1, len(entries) + 1)
    )
    assert best_model == ranked[0].model_name
    assert best_score == pytest.approx(ranked[0].cv.metrics[primary].mean)

    before = [(e.model_name, e.rank) for e in entries]
    for entry, value in zip(entries, test_values, strict=False):
        entry.test_metrics = {primary: value, "accuracy": value}
    again_model, again_score = rank_cv(entries, direction, primary_metric=primary)
    after = [(e.model_name, e.rank) for e in entries]
    assert before == after
    assert (again_model, again_score) == (best_model, best_score)


@st.composite
def instability_inputs(draw):
    """A CvRunSummary with K primary fold values and a threshold config."""
    k = draw(st.integers(min_value=2, max_value=8))
    primary = draw(names)
    values = draw(
        st.lists(
            st.floats(min_value=-10, max_value=10, allow_nan=False),
            min_size=k,
            max_size=k,
        )
    )
    cfg = CrossValidationConfig(
        folds=k,
        scheme=CvScheme.KFOLD,
        instability_std=draw(st.floats(min_value=0.001, max_value=1.0)),
        instability_cv=draw(st.floats(min_value=0.001, max_value=1.0)),
        outlier_multiplier=draw(st.floats(min_value=1.0, max_value=5.0)),
        outlier_min_gap=draw(st.floats(min_value=0.0, max_value=0.5)),
    )
    mean = statistics.fmean(values)
    std = statistics.stdev(values) if len(values) > 1 else 0.0
    summary = CvRunSummary(
        folds=k,
        succeeded_folds=k,
        metrics={
            primary: MetricAggregate(
                mean=mean, std=std, min=min(values), max=max(values), fold_count=k
            )
        },
        primary_fold_values=list(values),
    )
    return summary, cfg, primary, values


# Feature: cross-validation, Property 12: Instability and outlier warnings follow the thresholds —
# the Instability_Flag is set exactly when the primary std exceeds instability_std or the
# coefficient of variation exceeds instability_cv; a fold is flagged an outlier exactly when,
# with K >= 3, its distance from the other folds' mean exceeds max(multiplier x other std, min gap).
@PBT
@given(inputs=instability_inputs())
def test_property_12_instability_and_outlier_follow_thresholds(inputs):
    summary, cfg, primary, values = inputs
    warnings = instability_warnings(summary, cfg, primary_metric=primary, model_name="m")
    rules = [w.rule for w in warnings]

    mean = summary.metrics[primary].mean
    std = summary.metrics[primary].std
    by_std = std > cfg.instability_std
    by_cv = abs(mean) > 1e-12 and std / abs(mean) > cfg.instability_cv
    assert summary.instability_flag == (by_std or by_cv)
    assert ("cv_unstable_model" in rules) == summary.instability_flag

    expected_outliers = []
    if len(values) >= 3:
        for fold, value in enumerate(values):
            others = [v for i, v in enumerate(values) if i != fold]
            others_mean = statistics.fmean(others)
            others_std = statistics.stdev(others)
            threshold = max(cfg.outlier_multiplier * others_std, cfg.outlier_min_gap)
            if abs(value - others_mean) > threshold:
                expected_outliers.append(fold)
    assert summary.outlier_folds == expected_outliers
    assert rules.count("cv_outlier_fold") == len(expected_outliers)


# Feature: cross-validation, Property 12: the design's worked example flags only fold 4, and K=2
# never flags an outlier.
def test_property_12_worked_example_flags_only_fold_four():
    primary = "n_roc"
    cfg = CrossValidationConfig(folds=5, scheme=CvScheme.KFOLD)
    values = [0.80, 0.81, 0.79, 0.98, 0.80]
    summary = CvRunSummary(
        folds=5,
        succeeded_folds=5,
        metrics={
            primary: MetricAggregate(
                mean=statistics.fmean(values),
                std=statistics.stdev(values),
                min=min(values),
                max=max(values),
                fold_count=5,
            )
        },
        primary_fold_values=values,
    )
    instability_warnings(summary, cfg, primary_metric=primary, model_name="m")
    assert summary.outlier_folds == [3]

    k2 = CvRunSummary(
        folds=2,
        succeeded_folds=2,
        metrics={primary: MetricAggregate(mean=0.89, std=0.127, min=0.80, max=0.98, fold_count=2)},
        primary_fold_values=[0.80, 0.98],
    )
    instability_warnings(k2, cfg, primary_metric=primary, model_name="m")
    assert k2.outlier_folds == []


# ============================================================================================
# Reasoning numeric grounding (design C7): Property 15
# ============================================================================================

from ml_engine.contracts.cv import (  # noqa: E402
    CvComparisonInfo,
    TestEvaluationReport,
    TestRunResult,
)
from ml_engine.reasoning.cv_context import build_cross_validation_context  # noqa: E402
from ml_engine.reasoning.numbers import (  # noqa: E402
    _collect_numbers,
    _sentence_is_grounded,
    drop_unverified_numbers,
)

# Values with at most four decimals so a token's stated precision is well defined.
_scores = st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False).map(
    lambda v: round(v, 4)
)


@st.composite
def cv_comparisons(draw) -> ComparisonReport:
    """A CV_Mode comparison with per-Run CV summaries and an optional Test_Evaluation."""
    primary = draw(names)
    k = draw(st.integers(min_value=2, max_value=6))
    model_names = draw(st.lists(names, min_size=1, max_size=3, unique=True))
    has_test = draw(st.booleans())

    entries: list[ModelComparisonEntry] = []
    test_runs: list[TestRunResult] = []
    for model in model_names:
        fold_values = draw(st.lists(_scores, min_size=k, max_size=k))
        metric_names = draw(st.lists(names, min_size=1, max_size=3, unique=True))
        metrics = {}
        for metric in {primary, *metric_names}:
            agg_values = draw(st.lists(_scores, min_size=1, max_size=k))
            metrics[metric] = MetricAggregate(
                mean=round(statistics.fmean(agg_values), 4),
                std=round(statistics.pstdev(agg_values), 4),
                min=min(agg_values),
                max=max(agg_values),
                fold_count=len(agg_values),
            )
        entries.append(
            ModelComparisonEntry(
                model_name=model,
                display_name=model,
                status=ModelRunStatus.COMPLETED,
                primary_score=metrics[primary].mean,
                cv=CvRunSummary(
                    folds=k,
                    succeeded_folds=k,
                    metrics=metrics,
                    primary_fold_values=fold_values,
                    instability_flag=draw(st.booleans()),
                    outlier_folds=draw(st.lists(st.integers(min_value=0, max_value=k - 1))),
                ),
                test_metrics=({metric: draw(_scores) for metric in metrics} if has_test else None),
            )
        )
        if has_test:
            test_runs.append(
                TestRunResult(
                    model_name=model,
                    status="completed",
                    metrics={metric: draw(_scores) for metric in metrics},
                )
            )

    test_rows = draw(st.integers(min_value=1, max_value=500))
    return ComparisonReport(
        experiment_id=draw(names),
        generated_at=draw(moments),
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric=primary,
        direction=MetricDirection.MAXIMIZE,
        best_model=model_names[0],
        best_score=entries[0].primary_score,
        models=entries,
        succeeded_count=len(entries),
        cross_validation=CvComparisonInfo(
            scheme=CvScheme.KFOLD,
            folds=k,
            test_row_count=test_rows if has_test else None,
        ),
        test_evaluation=(
            TestEvaluationReport(
                experiment_id="exp",
                generated_at=datetime.now(UTC),
                evaluation_mode=EvaluationMode.CV,
                test_row_count=test_rows,
                primary_metric=primary,
                model_source="refit_model",
                runs=test_runs,
            )
            if has_test
            else None
        ),
    )


# Feature: cross-validation, Property 15: LLM numbers are grounded in deterministic artifacts —
# every number in the cross_validation reasoning context appears in the comparison or
# test-evaluation artifact.
@PBT
@given(comparison=cv_comparisons())
def test_property_15_context_numbers_come_from_the_artifact(comparison):
    context = build_cross_validation_context(comparison)
    assert context is not None

    # Numbers the comparison/test-evaluation artifact actually holds.
    artifact: list[float] = []
    for entry in comparison.models:
        cv = entry.cv
        if cv is None:
            continue
        artifact.extend(float(v) for v in cv.primary_fold_values)
        for agg in cv.metrics.values():
            artifact.extend([agg.mean, agg.std, agg.min, agg.max, float(agg.fold_count)])
        artifact.extend([float(cv.folds), float(cv.succeeded_folds)])
        artifact.extend(float(f) for f in cv.outlier_folds)
        if entry.test_metrics:
            artifact.extend(float(v) for v in entry.test_metrics.values())
    artifact.append(float(comparison.cross_validation.folds))
    if comparison.cross_validation.test_row_count is not None:
        artifact.append(float(comparison.cross_validation.test_row_count))
    if comparison.test_evaluation is not None:
        artifact.append(float(comparison.test_evaluation.test_row_count))
        for run in comparison.test_evaluation.runs:
            artifact.extend(float(v) for v in run.metrics.values())

    context_numbers: list[float] = []
    _collect_numbers(context, context_numbers)
    artifact_set = {round(v, 9) for v in artifact}
    for number in context_numbers:
        assert round(number, 9) in artifact_set, number


#: A pinned case: "80%" grounds against the fraction 0.8 as well as the written 80.
_PERCENT_READINGS_CONTEXT = {"rate": 0.8, "count": 80.0}


# Feature: cross-validation, Property 15: LLM numbers are grounded in deterministic artifacts —
# drop_unverified_numbers keeps a sentence iff every decimal/percentage token in it matches a
# context number at the sentence's stated precision.
@PBT
@example(numbers=[0.8], token="80%", other="review the data")
@given(
    numbers=st.lists(_scores, min_size=1, max_size=6),
    token=st.text(alphabet="0123456789.%", min_size=1, max_size=8),
    other=st.text(alphabet="abcdefg ", max_size=20),
)
def test_property_15_filter_keeps_iff_tokens_match(numbers, token, other):
    context = {"values": numbers}
    sentence = f"The score was {token} on the fold {other}."
    result = drop_unverified_numbers(sentence, context)
    # The oracle: the sentence survives exactly when the module judges it grounded.
    grounded = _sentence_is_grounded(sentence, numbers)
    assert (result == sentence) == grounded
    assert result in ("", sentence)


# Feature: cross-validation, Property 15: number-free sentences always survive and a fabricated
# number is always dropped, while a correctly cited number survives.
def test_property_15_fabricated_number_dropped_correct_kept():
    context = {"roc_auc": {"mean": 0.8734}, "folds": 5}
    text = (
        "The model is reasonably stable across folds. "
        "Its mean ROC AUC is 0.87. "
        "A fold reached 0.99 which is suspicious."
    )
    filtered = drop_unverified_numbers(text, context)
    assert "reasonably stable across folds" in filtered  # no number: kept
    assert "0.87" in filtered  # 0.8734 rounds to 0.87 at two decimals: kept
    assert "0.99" not in filtered  # fabricated: dropped


def test_property_15_single_split_builds_no_context():
    comparison = ComparisonReport(
        experiment_id="exp",
        generated_at=datetime.now(UTC),
        problem_type=ProblemType.BINARY_CLASSIFICATION,
        primary_metric="n_roc",
        direction=MetricDirection.MAXIMIZE,
    )
    assert build_cross_validation_context(comparison) is None
