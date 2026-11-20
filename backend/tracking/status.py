"""Map raw SageMaker status to the user-facing status (requirement 6.3, 6.4; Property 10).

Pure functions of the fields AWS returns. The rule throughout: if the evidence does not clearly
say which state a job is in, return ``UNKNOWN`` rather than guess. SageMaker's own vocabulary is
the source — primary ``*JobStatus`` plus, for training, ``SecondaryStatus``.
"""

from ml_engine.contracts.tracking import NormalizedStatus

# SageMaker training SecondaryStatus → our status. Only unambiguous mappings; anything else that
# arrives with primary InProgress stays UNKNOWN (we would be guessing otherwise).
_TRAINING_SECONDARY = {
    "Starting": NormalizedStatus.PREPARING,
    "LaunchingMLInstances": NormalizedStatus.PREPARING,
    "PreparingTrainingStack": NormalizedStatus.PREPARING,
    "Downloading": NormalizedStatus.PREPARING,
    "DownloadingTrainingImage": NormalizedStatus.PREPARING,
    "Pending": NormalizedStatus.QUEUED,
    "Training": NormalizedStatus.TRAINING,
    "Uploading": NormalizedStatus.TRAINING,
    "Interrupted": NormalizedStatus.INTERRUPTED_RECOVERING,
    "Restarting": NormalizedStatus.INTERRUPTED_RECOVERING,
    "MaxWaitTimeExceeded": NormalizedStatus.FAILED,
    "Completed": NormalizedStatus.COMPLETED,
    "Failed": NormalizedStatus.FAILED,
    "Stopped": NormalizedStatus.STOPPED,
    "Stopping": NormalizedStatus.STOPPED,
}

# A SecondaryStatus that means "waiting for Spot capacity" while primary is still InProgress.
_WAITING_SECONDARY = {"Pending", "Starting"}


def normalize_training_status(
    primary: str | None, secondary: str | None = None
) -> NormalizedStatus:
    """Training jobs carry a SecondaryStatus; prefer it, fall back to primary, else UNKNOWN."""
    terminal = _primary_terminal(primary)
    if terminal is not None:
        # A Spot wait that timed out shows terminal Stopped with secondary MaxWaitTimeExceeded;
        # that is a failure to the user, not a clean stop.
        if terminal is NormalizedStatus.STOPPED and secondary == "MaxWaitTimeExceeded":
            return NormalizedStatus.FAILED
        return terminal
    if primary == "InProgress":
        if secondary in _TRAINING_SECONDARY:
            return _TRAINING_SECONDARY[secondary]
        if secondary in _WAITING_SECONDARY:
            return NormalizedStatus.WAITING_FOR_CAPACITY
        return NormalizedStatus.UNKNOWN
    if primary == "Stopping":
        return NormalizedStatus.STOPPED
    return NormalizedStatus.UNKNOWN


def normalize_processing_status(primary: str | None) -> NormalizedStatus:
    """Processing jobs have no SecondaryStatus; map the primary status alone."""
    terminal = _primary_terminal(primary)
    if terminal is not None:
        return terminal
    if primary == "InProgress":
        return NormalizedStatus.PREPARING
    if primary == "Stopping":
        return NormalizedStatus.STOPPED
    return NormalizedStatus.UNKNOWN


def _primary_terminal(primary: str | None) -> NormalizedStatus | None:
    return {
        "Completed": NormalizedStatus.COMPLETED,
        "Failed": NormalizedStatus.FAILED,
        "Stopped": NormalizedStatus.STOPPED,
    }.get(primary or "")
