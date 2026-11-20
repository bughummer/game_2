"""Status normalization (requirement 6.3, 6.4; Property 10): never invent a status."""

import pytest

from backend.tracking.status import normalize_processing_status, normalize_training_status
from ml_engine.contracts.tracking import NormalizedStatus as S


@pytest.mark.parametrize(
    ("primary", "secondary", "expected"),
    [
        ("InProgress", "Starting", S.PREPARING),
        ("InProgress", "Downloading", S.PREPARING),
        ("InProgress", "Pending", S.QUEUED),
        ("InProgress", "Training", S.TRAINING),
        ("InProgress", "Uploading", S.TRAINING),
        ("InProgress", "Interrupted", S.INTERRUPTED_RECOVERING),
        ("InProgress", "Restarting", S.INTERRUPTED_RECOVERING),
        ("Completed", "Completed", S.COMPLETED),
        ("Failed", "Failed", S.FAILED),
        ("Stopped", "Stopped", S.STOPPED),
        # A Spot wait timeout surfaces as terminal Stopped + MaxWaitTimeExceeded; to the user
        # that is a failure, not a clean stop.
        ("Stopped", "MaxWaitTimeExceeded", S.FAILED),
    ],
)
def test_training_status_mapping(primary, secondary, expected):
    assert normalize_training_status(primary, secondary) is expected


@pytest.mark.parametrize(
    ("primary", "secondary"),
    [("InProgress", None), ("InProgress", "SomeNewSecondary"), (None, None), ("Weird", "Training")],
)
def test_unclear_training_evidence_is_unknown(primary, secondary):
    """When the fields do not clearly say which state the job is in, say UNKNOWN."""
    assert normalize_training_status(primary, secondary) is S.UNKNOWN


@pytest.mark.parametrize(
    ("primary", "expected"),
    [
        ("InProgress", S.PREPARING),
        ("Completed", S.COMPLETED),
        ("Failed", S.FAILED),
        ("Stopped", S.STOPPED),
        ("Stopping", S.STOPPED),
        (None, S.UNKNOWN),
        ("Unexpected", S.UNKNOWN),
    ],
)
def test_processing_status_mapping(primary, expected):
    assert normalize_processing_status(primary) is expected


def test_unknown_is_terminal_aware():
    assert S.COMPLETED.is_terminal and S.FAILED.is_terminal and S.STOPPED.is_terminal
    assert not S.UNKNOWN.is_terminal and not S.TRAINING.is_terminal
