"""Cross-validation and Test_Fold artifact paths (task 1.5). Every path comes from the layout."""

import pytest

from ml_engine.io import ExperimentLayout
from ml_engine.io.layout import (
    DEVELOPMENT_DATASET_FILE,
    FOLD_ASSIGNMENT_FILE,
    FOLD_PREPROCESSOR_TEMPLATE,
    RUN_OUTPUT_FILES,
    RUN_OUTPUT_PREFIXES,
    TEST_DATASET_FILE,
    TEST_EVALUATION_FILE,
)

BASE = "s3://mlf-test-artifacts/ml-factory/experiments/exp-test"


@pytest.fixture
def layout() -> ExperimentLayout:
    return ExperimentLayout(base=BASE)


def test_fold_assignment(layout):
    assert FOLD_ASSIGNMENT_FILE == "validation/fold_assignment.json"
    assert layout.fold_assignment == f"{BASE}/validation/fold_assignment.json"


def test_development_and_test_datasets(layout):
    assert layout.development_dataset == f"{BASE}/datasets/development.parquet"
    assert layout.test_dataset == f"{BASE}/datasets/test.parquet"


def test_fold_preprocessor(layout):
    assert FOLD_PREPROCESSOR_TEMPLATE.format(fold=0, strategy="s") == (
        "preprocessing/folds/fold_0/preprocessor_s.joblib"
    )
    assert layout.fold_preprocessor(3, "dense_numeric") == (
        f"{BASE}/preprocessing/folds/fold_3/preprocessor_dense_numeric.joblib"
    )
    # The full-Development_Set fit keeps the inherited location (refit, holdout, scoring).
    assert layout.preprocessor("dense_numeric") == (
        f"{BASE}/preprocessing/preprocessor_dense_numeric.joblib"
    )


def test_model_cv_dir(layout):
    assert layout.model_cv_dir("xgboost") == f"{BASE}/models/xgboost/cv"


def test_model_fold_result_and_oof_slice(layout):
    assert layout.model_fold_result("xgboost", 2) == f"{BASE}/models/xgboost/cv/folds/fold_2.json"
    assert layout.model_fold_oof("xgboost", 2) == (
        f"{BASE}/models/xgboost/cv/folds/fold_2_oof.parquet"
    )


def test_model_fold_metrics(layout):
    assert layout.model_fold_metrics("xgboost") == f"{BASE}/models/xgboost/cv/fold_metrics.json"


def test_model_oof_manifest_and_payload(layout):
    assert layout.model_oof("xgboost") == f"{BASE}/models/xgboost/cv/oof.json"
    assert layout.model_oof_payload("xgboost") == f"{BASE}/models/xgboost/cv/oof.parquet"


def test_test_evaluation(layout):
    assert TEST_EVALUATION_FILE == "comparison/test_evaluation.json"
    assert layout.test_evaluation == f"{BASE}/comparison/test_evaluation.json"


def test_every_cv_model_artifact_is_inside_the_model_directory(layout):
    """Each Run writes only under ``models/{m}/`` (requirement 6.6)."""
    model_dir = layout.model_dir("m") + "/"
    paths = [
        layout.model_cv_dir("m"),
        layout.model_fold_result("m", 0),
        layout.model_fold_oof("m", 0),
        layout.model_fold_metrics("m"),
        layout.model_oof("m"),
        layout.model_oof_payload("m"),
    ]
    assert all(path.startswith(model_dir) for path in paths)


def test_a_retrain_archives_every_cv_artifact(layout):
    """Requirement 10.6: everything CV writes is under an archived prefix or file."""
    assert DEVELOPMENT_DATASET_FILE in RUN_OUTPUT_FILES
    assert TEST_DATASET_FILE in RUN_OUTPUT_FILES
    under_prefix = [
        layout.fold_assignment,
        layout.fold_preprocessor(0, "s"),
        layout.model_fold_result("m", 0),
        layout.model_oof_payload("m"),
        layout.test_evaluation,
    ]
    for path in under_prefix:
        relative = path.removeprefix(BASE + "/")
        assert relative.split("/", 1)[0] in RUN_OUTPUT_PREFIXES, relative


def test_the_inherited_run_output_files_are_unchanged_and_first():
    assert RUN_OUTPUT_FILES[-2:] == (DEVELOPMENT_DATASET_FILE, TEST_DATASET_FILE)
    assert len(RUN_OUTPUT_FILES) == 11
