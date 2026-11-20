"""Cross-validation contracts (task 1.2, 1.3): new models, additive fields, byte compatibility."""

import json
from datetime import UTC, datetime

import pytest
from pydantic import Field, ValidationError

from ml_engine.contracts.common import StrictModel
from ml_engine.contracts.comparison import (
    ComparisonReport,
    ExperimentSummary,
    ModelComparisonEntry,
)
from ml_engine.contracts.config import ExperimentConfig, SplitConfig
from ml_engine.contracts.cv import (
    CV_SCHEMA_VERSION,
    TEST,
    TRAIN_ONLY,
    CrossValidationConfig,
    CvComparisonInfo,
    CvRunSummary,
    CvScheme,
    EvaluationMode,
    FoldAssignment,
    MetricAggregate,
    TestEvaluationReport,
    TestRunResult,
)
from ml_engine.contracts.explain import AblationEntry, AblationReport, PermutationImportance
from ml_engine.contracts.model import ModelMetadata
from ml_engine.contracts.preparation import PreparationReport
from ml_engine.contracts.runs import ArchivedRun
from ml_engine.contracts.thresholds import ThresholdAnalysis, ThresholdPoint
from ml_engine.io import read_model, write_model
from tests.support.golden import golden_document

NOW = datetime(2026, 10, 26, 12, 0, tzinfo=UTC)
CV5 = CrossValidationConfig(folds=5, scheme=CvScheme.STRATIFIED_KFOLD)

# --- exclude_if on the pinned Pydantic, through write_model -----------------------------------


class _Before(StrictModel):
    name: str
    count: int = 0


class _After(StrictModel):
    name: str
    count: int = 0
    extra: str | None = Field(default=None, exclude_if=lambda value: value is None)
    flags: list[int] = Field(default_factory=list, exclude_if=lambda value: value == [])


def test_exclude_if_keeps_write_model_bytes_identical_at_the_default(store):
    write_model(store, "s3://mlf-test-bucket/before.json", _Before(name="a", count=2))
    write_model(store, "s3://mlf-test-bucket/after.json", _After(name="a", count=2))
    assert store.read_bytes("s3://mlf-test-bucket/after.json") == store.read_bytes(
        "s3://mlf-test-bucket/before.json"
    )


def test_exclude_if_writes_the_field_when_it_is_set(store):
    write_model(store, "s3://mlf-test-bucket/after.json", _After(name="a", extra="x", flags=[1]))
    document = json.loads(store.read_bytes("s3://mlf-test-bucket/after.json"))
    assert document == {"name": "a", "count": 0, "extra": "x", "flags": [1]}
    assert read_model(store, "s3://mlf-test-bucket/after.json", _After) == _After(
        name="a", extra="x", flags=[1]
    )


def test_exclude_if_applies_to_nested_models(store):
    class _Outer(StrictModel):
        inner: _After

    write_model(store, "s3://mlf-test-bucket/outer.json", _Outer(inner=_After(name="a")))
    assert json.loads(store.read_bytes("s3://mlf-test-bucket/outer.json")) == {
        "inner": {"name": "a", "count": 0}
    }


def test_old_json_without_the_new_key_parses_to_the_default(store):
    write_model(store, "s3://mlf-test-bucket/before.json", _Before(name="a"))
    parsed = read_model(store, "s3://mlf-test-bucket/before.json", _After)
    assert parsed.extra is None and parsed.flags == []


def test_strict_models_still_reject_unknown_keys():
    with pytest.raises(ValidationError):
        _Before.model_validate({"name": "a", "unknown": datetime.now(UTC)})


# --- SplitConfig ------------------------------------------------------------------------------


def test_split_config_defaults_to_single_split_and_omits_the_new_keys():
    split = SplitConfig()
    assert split.evaluation_mode is EvaluationMode.SINGLE_SPLIT
    assert split.cv is None and split.test_fraction is None
    assert set(json.loads(split.model_dump_json())) == {
        "strategy",
        "validation_fraction",
        "random_seed",
        "stratify",
        "group_column",
        "time_column",
    }


def test_split_config_in_cv_mode_round_trips_with_the_new_keys():
    split = SplitConfig(evaluation_mode=EvaluationMode.CV, cv=CV5, test_fraction=0.2)
    document = json.loads(split.model_dump_json())
    assert document["evaluation_mode"] == "cv"
    assert document["cv"]["folds"] == 5 and document["test_fraction"] == 0.2
    assert SplitConfig.model_validate_json(split.model_dump_json()) == split


def test_a_test_fold_alone_is_written_in_single_split_mode():
    document = json.loads(SplitConfig(test_fraction=0.1).model_dump_json())
    assert document["test_fraction"] == 0.1 and "evaluation_mode" not in document


def test_cv_mode_requires_a_cv_config():
    with pytest.raises(ValidationError, match="needs a cross-validation"):
        SplitConfig(evaluation_mode=EvaluationMode.CV)


def test_a_cv_config_requires_cv_mode():
    with pytest.raises(ValidationError, match="needs evaluation_mode"):
        SplitConfig(cv=CV5)


@pytest.mark.parametrize("fraction", [0.0, 0.5, 0.75, -0.1])
def test_test_fraction_must_be_in_the_open_interval(fraction):
    with pytest.raises(ValidationError):
        SplitConfig(test_fraction=fraction)


def test_cross_validation_config_needs_at_least_two_folds():
    with pytest.raises(ValidationError):
        CrossValidationConfig(folds=1, scheme=CvScheme.KFOLD)


# --- FoldAssignment ---------------------------------------------------------------------------


def _assignment(**overrides) -> FoldAssignment:
    values = {
        "experiment_id": "exp-1",
        "generated_at": NOW,
        "evaluation_mode": EvaluationMode.CV,
        "scheme": CvScheme.KFOLD,
        "folds": 2,
        "test_fraction": 0.2,
        "random_seed": 42,
        "dataset_fingerprint": "f" * 64,
        "development_digest": "d" * 64,
        "row_ids": [0, 1, 2, 3, 4],
        "assignments": [0, 1, TEST, 0, 1],
    }
    values.update(overrides)
    return FoldAssignment(**values)


def test_fold_assignment_round_trips_and_counts_rows():
    assignment = _assignment()
    assert FoldAssignment.model_validate_json(assignment.model_dump_json()) == assignment
    assert assignment.schema_version == CV_SCHEMA_VERSION
    assert assignment.development_row_count == 4 and assignment.test_row_count == 1


def test_fold_assignment_rejects_misaligned_lists():
    with pytest.raises(ValidationError, match="same length"):
        _assignment(assignments=[0, 1, 0, 1])


@pytest.mark.parametrize("row_ids", [[0, 2, 1, 3, 4], [0, 1, 1, 3, 4]])
def test_fold_assignment_rejects_unordered_or_duplicate_row_ids(row_ids):
    with pytest.raises(ValidationError, match="strictly ascending"):
        _assignment(row_ids=row_ids)


def test_fold_assignment_rejects_a_fold_index_out_of_range():
    with pytest.raises(ValidationError, match="outside"):
        _assignment(assignments=[0, 1, 2, 0, 1])


def test_fold_assignment_rejects_an_empty_fold():
    with pytest.raises(ValidationError, match="without validation rows"):
        _assignment(assignments=[0, 0, TEST, 0, 0])


def test_train_only_rows_are_only_allowed_for_forward_chaining():
    with pytest.raises(ValidationError, match="forward_chaining"):
        _assignment(assignments=[TRAIN_ONLY, 1, TEST, 0, 1])
    chained = _assignment(scheme=CvScheme.FORWARD_CHAINING, assignments=[TRAIN_ONLY, 0, 0, 1, 1])
    assert chained.development_row_count == 5


def test_single_split_assignment_uses_zero_for_validation_and_no_scheme():
    single = _assignment(
        evaluation_mode=EvaluationMode.SINGLE_SPLIT,
        scheme=None,
        folds=0,
        assignments=[TRAIN_ONLY, TRAIN_ONLY, TEST, 0, TRAIN_ONLY],
    )
    assert single.test_row_count == 1
    with pytest.raises(ValidationError, match="Single_Split_Mode"):
        _assignment(evaluation_mode=EvaluationMode.SINGLE_SPLIT, scheme=CvScheme.KFOLD)
    with pytest.raises(ValidationError, match="CV_Mode"):
        _assignment(scheme=None)


# --- additive fields: omitted at the default, written when set --------------------------------


def _keys(model) -> set[str]:
    return set(json.loads(model.model_dump_json()))


def test_comparison_entry_cv_fields_are_written_only_when_set():
    entry = ModelComparisonEntry(model_name="m", display_name="M", status="COMPLETED")
    assert not {"cv", "test_metrics"} & _keys(entry)
    summary = CvRunSummary(
        folds=2,
        succeeded_folds=2,
        metrics={"roc_auc": MetricAggregate(mean=0.8, std=0.01, min=0.79, max=0.81, fold_count=2)},
        primary_fold_values=[0.79, 0.81],
    )
    entry = entry.model_copy(update={"cv": summary, "test_metrics": {"roc_auc": 0.78}})
    assert {"cv", "test_metrics"} <= _keys(entry)


def test_comparison_report_cv_fields_are_written_only_when_set():
    report = ComparisonReport(
        experiment_id="e",
        generated_at=NOW,
        problem_type="binary_classification",
        primary_metric="roc_auc",
        direction="maximize",
    )
    assert not {"cross_validation", "test_evaluation"} & _keys(report)
    test_report = TestEvaluationReport(
        experiment_id="e",
        generated_at=NOW,
        evaluation_mode=EvaluationMode.CV,
        test_row_count=10,
        primary_metric="roc_auc",
        model_source="refit_model",
        runs=[TestRunResult(model_name="m", status="completed", metrics={"roc_auc": 0.7})],
    )
    report = report.model_copy(
        update={
            "cross_validation": CvComparisonInfo(scheme=CvScheme.KFOLD, folds=5),
            "test_evaluation": test_report,
        }
    )
    assert {"cross_validation", "test_evaluation"} <= _keys(report)
    assert ComparisonReport.model_validate_json(report.model_dump_json()) == report


def test_new_file_contracts_omit_their_cv_fields_at_the_default():
    permutation = PermutationImportance(
        experiment_id="e",
        model_name="m",
        generated_at=NOW,
        primary_metric="roc_auc",
        direction="maximize",
        baseline_score=0.8,
        repeats=5,
        random_seed=42,
    )
    assert not {"fold_count", "fold_baselines"} & _keys(permutation)
    assert {"fold_count", "fold_baselines"} <= _keys(
        permutation.model_copy(update={"fold_count": 3, "fold_baselines": [0.8, 0.7, 0.9]})
    )

    entry = AblationEntry(feature="x")
    assert not {"fold_drops", "std_drop"} & _keys(entry)
    assert {"fold_drops", "std_drop"} <= _keys(
        entry.model_copy(update={"fold_drops": [0.1, 0.2], "std_drop": 0.07})
    )
    ablation = AblationReport(
        experiment_id="e",
        generated_at=NOW,
        model_name="m",
        primary_metric="roc_auc",
        direction="maximize",
        baseline_score=0.8,
    )
    assert "fold_count" not in _keys(ablation)
    assert "fold_count" in _keys(ablation.model_copy(update={"fold_count": 5}))

    point = ThresholdPoint(threshold=0.5, precision=1, recall=1, f1=1, predicted_positive_rate=0.5)
    thresholds = ThresholdAnalysis(
        experiment_id="e",
        model_name="m",
        generated_at=NOW,
        positive_class="1",
        validation_row_count=10,
        positive_rate=0.5,
        default=point,
        best_f1=point,
    )
    assert "source" not in _keys(thresholds)
    assert _keys(thresholds.model_copy(update={"source": "oof"})) >= {"source"}

    run = ArchivedRun(run_id="r", experiment_id="e", archived_at=NOW)
    assert not {"evaluation_mode", "cv_folds", "test_fraction"} & _keys(run)
    run = run.model_copy(
        update={"evaluation_mode": EvaluationMode.CV, "cv_folds": 5, "test_fraction": 0.2}
    )
    assert ArchivedRun.model_validate_json(run.model_dump_json()) == run


# --- pre-feature golden fixtures parse with single-split semantics ----------------------------


def test_the_golden_config_parses_as_single_split_without_a_test_fold():
    config = ExperimentConfig.model_validate(golden_document("experiment_config.json"))
    assert config.split.evaluation_mode is EvaluationMode.SINGLE_SPLIT
    assert config.split.cv is None and config.split.test_fraction is None


def test_the_golden_artifacts_parse_with_every_cv_field_unset():
    preparation = PreparationReport.model_validate(golden_document("validation.json"))
    assert preparation.cross_validation is None and preparation.test_fold is None
    metadata = ModelMetadata.model_validate(golden_document("metadata.json"))
    assert metadata.cross_validation is None
    comparison = ComparisonReport.model_validate(golden_document("comparison.json"))
    assert comparison.cross_validation is None and comparison.test_evaluation is None
    assert all(e.cv is None and e.test_metrics is None for e in comparison.models)
    summary = ExperimentSummary.model_validate(golden_document("experiment_summary.json"))
    assert summary.config.split.evaluation_mode is EvaluationMode.SINGLE_SPLIT
    assert summary.comparison is not None and summary.comparison.cross_validation is None
