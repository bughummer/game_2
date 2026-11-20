"""Scoring trained models on a separate dataset: no refit, honest about overlap."""

from datetime import UTC, datetime

import pytest

from jobs._common.runtime import JobError
from jobs.holdout_evaluation.main import MIN_UNSEEN_ROWS, run_holdout_evaluation
from jobs.preprocessing.main import run_preparation
from jobs.profiling.main import run_profiling
from jobs.training.main import run_training
from ml_engine.contracts.holdout import HoldoutRequest
from ml_engine.io import write_model
from tests.support.datasets import write_dataset
from tests.support.experiments import make_config

EXP = "exp-test"
MODELS = ["logistic_regression", "xgboost"]


@pytest.fixture
def trained(store, layout, dataset_csv):
    run_profiling(
        store, layout, experiment_id=EXP, dataset_uri=dataset_csv, target_column="churned"
    )
    write_model(store, layout.experiment_config, make_config(layout, EXP, dataset_csv, MODELS))
    run_preparation(store, layout, experiment_id=EXP)
    for name in MODELS:
        run_training(store, layout, experiment_id=EXP, model_name=name)


def _evaluate(store, layout, dataset_uri, evaluation_id="eval-1"):
    write_model(
        store,
        layout.evaluation_request(evaluation_id),
        HoldoutRequest(
            evaluation_id=evaluation_id,
            experiment_id=EXP,
            dataset_uri=dataset_uri,
            file_format="csv",
            requested_at=datetime.now(UTC),
        ),
    )
    return run_holdout_evaluation(store, layout, experiment_id=EXP, evaluation_id=evaluation_id)


def _fresh(classification_frame):
    """Same population, different rows: every row differs from the training data."""
    frame = classification_frame.copy()
    # The shared frame has missing charges; NaN + 0.37 stays NaN and would leave those rows
    # identical to training rows, so give them a value no training row has.
    frame["monthly_charges"] = (frame["monthly_charges"] + 0.37).fillna(999.37)
    return frame


def test_the_full_training_dataset_is_flagged_as_seen(store, layout, dataset_csv, trained):
    report = _evaluate(store, layout, dataset_csv)

    assert report.overlap.overlap_fraction > 0.95
    assert report.overlap.unseen_rows < MIN_UNSEEN_ROWS
    rules = {w.rule: w.severity.value for w in report.warnings}
    assert rules["evaluation_rows_seen_in_training"] == "high"
    assert "too_few_unseen_rows" in rules
    for model in report.models:
        assert model.status == "completed"
        assert model.metrics_all_rows is not None and model.metrics_unseen_rows is None


def test_a_fresh_dataset_is_scored_on_unseen_rows(store, layout, classification_frame, trained):
    uri = write_dataset(store, "fresh.csv", _fresh(classification_frame))

    report = _evaluate(store, layout, uri)

    assert report.overlap.unseen_rows == report.row_count
    assert not [w for w in report.warnings if w.rule == "evaluation_rows_seen_in_training"]
    assert {m.model_name for m in report.models} == set(MODELS)
    for model in report.models:
        assert model.unseen_rows_score is not None
        assert model.unseen_rows_score == model.all_rows_score  # nothing was seen
        assert model.at_validation_best_threshold is not None
        assert model.threshold_rows == "unseen"
    assert report.target_rate is not None and report.training_target_rate is not None


def test_models_and_pipelines_are_applied_unchanged(store, layout, classification_frame, trained):
    """Nothing is re-fitted or rewritten: every training artifact is byte-identical after."""
    uris = [
        *(layout.model_artifact(name) for name in MODELS),
        *(layout.model_metadata(name) for name in MODELS),
        *(uri for uri in store.list_uris(layout.path("preprocessing")) if uri.endswith(".joblib")),
        layout.train_dataset,
        layout.validation_dataset,
    ]
    before = {uri: store.read_bytes(uri) for uri in uris}

    _evaluate(store, layout, write_dataset(store, "fresh.csv", _fresh(classification_frame)))

    assert {uri: store.read_bytes(uri) for uri in uris} == before


def test_a_missing_model_column_is_reported_by_name(store, layout, classification_frame, trained):
    uri = write_dataset(
        store, "short.csv", _fresh(classification_frame).drop(columns=["tenure_months"])
    )

    with pytest.raises(JobError) as error:
        _evaluate(store, layout, uri)

    assert error.value.code == "FEATURE_COLUMNS_MISSING"
    assert "tenure_months" in error.value.message


def test_a_missing_target_is_refused(store, layout, classification_frame, trained):
    uri = write_dataset(store, "notarget.csv", classification_frame.drop(columns=["churned"]))

    with pytest.raises(JobError) as error:
        _evaluate(store, layout, uri)

    assert error.value.code == "TARGET_COLUMN_MISSING"


def test_extra_columns_and_partial_overlap_are_handled(
    store, layout, classification_frame, dataset_csv, trained
):
    """Half old rows, half new, plus a column the models never used."""
    old = classification_frame.iloc[:100]
    new = _fresh(classification_frame).iloc[100:]
    import pandas as pd

    mixed = pd.concat([old, new]).assign(unused_column="x")

    report = _evaluate(store, layout, write_dataset(store, "mixed.csv", mixed))

    assert report.overlap.unseen_rows == len(new)
    assert 0.2 < report.overlap.overlap_fraction < 0.3
    for model in report.models:
        assert model.metrics_unseen_rows is not None


# --- environment ------------------------------------------------------------------------------


def test_training_records_its_environment_and_a_requirements_file(store, layout, trained):
    import platform as platform_module

    import numpy
    import sklearn

    from ml_engine.contracts.environment import RuntimeEnvironment
    from ml_engine.io import read_model

    environment = read_model(store, layout.environment, RuntimeEnvironment)
    assert environment.python_version == platform_module.python_version()
    assert environment.packages["numpy"] == numpy.__version__
    assert environment.packages["scikit-learn"] == sklearn.__version__

    requirements = store.read_bytes(layout.requirements).decode()
    assert f"numpy=={numpy.__version__}" in requirements.splitlines()
    assert f"scikit-learn=={sklearn.__version__}" in requirements.splitlines()
    assert f"# python {platform_module.python_version()}" in requirements
    # Every non-comment line is an exact pin.
    pins = [line for line in requirements.splitlines() if line and not line.startswith("#")]
    assert pins and all("==" in line for line in pins)


def test_scoring_in_the_training_environment_reports_no_difference(
    store, layout, classification_frame, trained
):
    report = _evaluate(
        store, layout, write_dataset(store, "fresh.csv", _fresh(classification_frame))
    )

    assert report.environment_differences == {}
    rules = {w.rule for w in report.warnings}
    assert "environment_differs_from_training" not in rules
    assert "training_environment_unknown" not in rules


def test_a_library_version_change_is_flagged(store, layout, classification_frame, trained):
    from ml_engine.contracts.environment import RuntimeEnvironment
    from ml_engine.io import read_model

    environment = read_model(store, layout.environment, RuntimeEnvironment)
    environment.packages["scikit-learn"] = "0.0.1"  # as if trained on an older image
    write_model(store, layout.environment, environment)

    report = _evaluate(
        store, layout, write_dataset(store, "fresh.csv", _fresh(classification_frame))
    )

    assert report.environment_differences["scikit-learn"].startswith("0.0.1 -> ")
    warning = next(w for w in report.warnings if w.rule == "environment_differs_from_training")
    assert warning.severity.value == "high" and "scikit-learn" in warning.message


def test_an_unrecorded_training_environment_is_said_plainly(
    store, layout, classification_frame, trained
):
    store._objects.pop(tuple(layout.environment.removeprefix("s3://").split("/", 1)))

    report = _evaluate(
        store, layout, write_dataset(store, "fresh.csv", _fresh(classification_frame))
    )

    assert "training_environment_unknown" in {w.rule for w in report.warnings}
