"""Failure diagnosis (requirement 8): deterministic category, bounded and redacted evidence."""

from datetime import UTC, datetime

import pytest

from backend.tracking.diagnostics import (
    MAX_LOG_BYTES,
    MAX_LOG_LINES,
    bound_logs,
    classify,
    diagnose_failure,
    fetch_logs,
    redact,
)
from ml_engine.contracts.model import ModelFailure
from ml_engine.contracts.tracking import FailureCategory
from ml_engine.io import ExperimentLayout, write_model
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.fake_sagemaker import FakeLogs

NOW = datetime(2026, 10, 14, tzinfo=UTC)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("MaxWaitTimeExceeded", FailureCategory.SPOT_WAIT_TIMEOUT),
        ("InsufficientInstanceCapacity for ml.m5.large", FailureCategory.INSUFFICIENT_CAPACITY),
        ("Container killed due to OutOfMemory", FailureCategory.OUT_OF_MEMORY),
        ("ModuleNotFoundError: No module named 'catboost'", FailureCategory.MISSING_DEPENDENCY),
        ("botocore ... AccessDenied when calling s3:GetObject", FailureCategory.S3_ACCESS),
        (
            "User is not authorized to perform sagemaker:CreateTrainingJob",
            FailureCategory.IAM_PERMISSION,
        ),
        ("TARGET_COLUMN_MISSING: churned not present", FailureCategory.INVALID_DATA),
        ("exec format error", FailureCategory.CONTAINER_STARTUP),
        ("AlgorithmError: Traceback ... ValueError", FailureCategory.SCRIPT_EXCEPTION),
        ("some unrecognizable text", FailureCategory.UNKNOWN),
    ],
)
def test_classifier_categories(text, expected):
    category, summary, _ = classify(text)
    assert category is expected
    assert summary


def test_spot_timeout_is_matched_before_generic_capacity():
    """Ordering matters: a wait timeout is more specific than plain insufficient capacity."""
    category, _, _ = classify("MaxWaitTimeExceeded: InsufficientInstanceCapacity")
    assert category is FailureCategory.SPOT_WAIT_TIMEOUT


@pytest.mark.parametrize(
    ("raw", "masked"),
    [
        ("contact ops@corp.com", "<email>"),
        ("key AKIAIOSFODNN7EXAMPLE", "<aws-access-key>"),
        ("arn:aws:iam::123456789012:role/x", "<arn>"),
        ("account 123456789012 denied", "<account-id>"),
        ("from 10.0.12.34 refused", "<ip>"),
        ("digest a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6", "<hex>"),
    ],
)
def test_redaction_masks_identifiers(raw, masked):
    assert masked in redact(raw)
    assert "ops@corp.com" not in redact("ops@corp.com")


def test_logs_are_bounded_by_lines_and_bytes():
    many = [f"line {i}" for i in range(1000)]
    kept, truncated = bound_logs(many)
    assert len(kept) <= MAX_LOG_LINES and truncated
    assert kept[-1] == "line 999"  # the most recent lines are the ones kept

    huge = ["x" * 1000 for _ in range(100)]
    kept, truncated = bound_logs(huge)
    assert sum(len(line.encode()) + 1 for line in kept) <= MAX_LOG_BYTES and truncated


def test_logs_are_redacted_when_bounded():
    kept, _ = bound_logs(["error for user bob@corp.com from 10.1.2.3"])
    assert "<email>" in kept[0] and "<ip>" in kept[0]


def test_diagnose_reads_the_stage_failure_and_records_evidence(store):
    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, "exp-diag")
    write_model(
        store,
        layout.model_failure("xgboost"),
        ModelFailure(
            experiment_id="exp-diag",
            model_name="xgboost",
            error_code="MODEL_TRAINING_FAILED",
            message="ValueError: bad input for user carol@corp.com",
            failed_at=NOW,
        ),
    )

    diagnosis = diagnose_failure(
        store,
        layout,
        execution_id="exp-diag.training.0a1b2c3d",
        job_name="mlf-exp-diag-train-xgboost",
        stage="training",
        model_name="xgboost",
        failure_reason="AlgorithmError: see job logs",
        log_lines=["Traceback (most recent call last):", "ValueError: bad input"],
        now=NOW,
    )

    assert diagnosis.category is FailureCategory.SCRIPT_EXCEPTION
    assert diagnosis.stage_error_code == "MODEL_TRAINING_FAILED"
    assert any("MODEL_TRAINING_FAILED" in line for line in diagnosis.evidence)
    assert all("carol@corp.com" not in line for line in diagnosis.evidence)
    assert diagnosis.log_excerpt and not diagnosis.logs_truncated


def test_a_user_stop_is_not_called_a_failure(store):
    layout = ExperimentLayout.for_experiment(ARTIFACT_ROOT, "exp-stop")
    diagnosis = diagnose_failure(
        store,
        layout,
        execution_id="exp-stop.training.0a1b2c3d",
        job_name="j",
        stage="training",
        model_name="catboost",
        failure_reason=None,
        stopped_by_user=True,
        now=NOW,
    )
    assert diagnosis.category is FailureCategory.STOPPED_BY_USER


def test_fetch_logs_never_raises():
    logs = FakeLogs()
    logs.raise_error = True
    assert fetch_logs(logs, "/aws/sagemaker/TrainingJobs", "job") == []
    logs.raise_error = False
    logs.set_events("job", ["a", "b"])
    assert fetch_logs(logs, "/aws/sagemaker/TrainingJobs", "job") == ["a", "b"]
