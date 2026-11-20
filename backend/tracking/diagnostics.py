"""Deterministic failure diagnosis (requirement 8).

A failed job is classified by matching AWS-reported evidence — ``FailureReason``, the stage's own
``failure.json``, and a bounded slice of CloudWatch logs — against an ordered rule list. The
category and the evidence are facts; an LLM may later explain them but never invents a cause
(requirement 8.3). Everything that could reach an LLM is bounded and PII-redacted first
(requirement 8.4).
"""

import re
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from ml_engine.contracts.model import ModelFailure
from ml_engine.contracts.tracking import FailureCategory, FailureDiagnosis
from ml_engine.io import ExperimentLayout, ObjectStore, read_model_if_exists

MAX_LOG_LINES = 200
MAX_LOG_BYTES = 32_768

# --- redaction (applied before any text could reach an LLM) -----------------------------------

_REDACTIONS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<aws-access-key>"),
    (re.compile(r"\bASIA[0-9A-Z]{16}\b"), "<aws-temp-key>"),
    (re.compile(r"arn:aws[\w:/-]*"), "<arn>"),
    (re.compile(r"\b\d{12}\b"), "<account-id>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
    (re.compile(r"\b[0-9a-fA-F]{32,}\b"), "<hex>"),
    (re.compile(r"\b\d{7,}\b"), "<number>"),
]


def redact(text: str) -> str:
    """Mask the obvious identifiers. Conservative: it only ever removes information."""
    for pattern, replacement in _REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


def bound_logs(lines: list[str]) -> tuple[list[str], bool]:
    """Keep the last ``MAX_LOG_LINES`` lines within ``MAX_LOG_BYTES``, redacted; flag truncation."""
    truncated = len(lines) > MAX_LOG_LINES
    kept: list[str] = []
    size = 0
    for line in reversed(lines[-MAX_LOG_LINES:]):
        clean = redact(line.rstrip("\n"))
        encoded = len(clean.encode("utf-8")) + 1
        if size + encoded > MAX_LOG_BYTES:
            truncated = True
            break
        kept.append(clean)
        size += encoded
    kept.reverse()
    return kept, truncated


# --- classification ---------------------------------------------------------------------------
# Ordered: the first rule whose predicate matches wins. Specific before general.

_Rule = tuple[FailureCategory, Callable[[str], bool], str, str | None]


def _has(*needles: str) -> Callable[[str], bool]:
    lowered = [n.lower() for n in needles]
    return lambda text: any(n in text for n in lowered)


_RULES: list[_Rule] = [
    (
        FailureCategory.SPOT_WAIT_TIMEOUT,
        _has("maxwaittimeexceeded", "max wait time", "spot capacity was not available"),
        "The job waited for Spot capacity longer than the configured maximum and was stopped.",
        "Increase MaxWaitTimeInSeconds, try another instance type, or enable the On-Demand "
        "fallback for this experiment.",
    ),
    (
        FailureCategory.INSUFFICIENT_CAPACITY,
        _has("insufficientinstancecapacity", "capacityerror", "insufficient capacity"),
        "AWS had no capacity for the requested instance type.",
        "Retry later, choose another allowed instance type, or enable the On-Demand fallback.",
    ),
    (
        FailureCategory.OUT_OF_MEMORY,
        _has("outofmemory", "out of memory", "oomkilled", "cannot allocate memory", "killed"),
        "The job ran out of memory.",
        "Use a larger allowed instance type or reduce the dataset or model size.",
    ),
    (
        FailureCategory.MISSING_DEPENDENCY,
        _has("modulenotfounderror", "importerror", "no module named"),
        "The container is missing a Python dependency.",
        "Rebuild the stage image with the dependency pinned in pyproject.toml.",
    ),
    (
        FailureCategory.S3_ACCESS,
        # Before the generic IAM rule: an S3 denial also says "AccessDenied", but a missing
        # bucket/key or an s3: action points specifically at storage.
        _has(
            "nosuchbucket",
            "nosuchkey",
            "s3.amazonaws.com",
            "could not download",
            "access denied to s3",
            "s3:getobject",
            "s3:putobject",
            "s3:listbucket",
        ),
        "The job could not read or write an S3 object.",
        "Check the artifact bucket exists and the SageMaker execution role may read and write it.",
    ),
    (
        FailureCategory.IAM_PERMISSION,
        _has("accessdenied", "not authorized", "is not authorized to perform", "assumerole"),
        "The job was denied an AWS permission.",
        "Add the missing action to the SageMaker execution role policy.",
    ),
    (
        FailureCategory.INVALID_DATA,
        _has(
            "target_column_missing",
            "unsupporteddatasetformat",
            "datasetloaderror",
            "could not parse",
        ),
        "The dataset could not be used as configured.",
        "Check the target column name and the file format, then recreate the experiment.",
    ),
    (
        FailureCategory.CONTAINER_STARTUP,
        _has("containererror", "failed to start", "exec format error", "entrypoint", "oci runtime"),
        "The container failed to start.",
        "Check the image entrypoint and that the image was built for linux/amd64.",
    ),
    (
        FailureCategory.SCRIPT_EXCEPTION,
        _has("algorithmerror", "traceback", "exit code 1", "error:", "exception"),
        "The stage script raised an error.",
        "Read the diagnosis evidence and the stage failure code; reproduce the stage locally.",
    ),
]


def classify(text: str) -> tuple[FailureCategory, str, str | None]:
    lowered = text.lower()
    for category, matches, summary, action in _RULES:
        if matches(lowered):
            return category, summary, action
    return (
        FailureCategory.UNKNOWN,
        "The job failed and no recognizable cause was found in the available evidence.",
        "Review the logs; if this recurs, capture the job name for support.",
    )


def diagnose_failure(
    store: ObjectStore,
    layout: ExperimentLayout,
    *,
    execution_id: str,
    job_name: str,
    stage: str,
    failure_reason: str | None,
    model_name: str | None = None,
    log_lines: list[str] | None = None,
    stopped_by_user: bool = False,
    now: datetime | None = None,
) -> FailureDiagnosis:
    """Classify one failed job from its evidence. Pure except for reading the stage failure.json."""
    stage_failure = _read_stage_failure(store, layout, stage, model_name)
    excerpt, truncated = bound_logs(log_lines or [])

    evidence: list[str] = []
    if failure_reason:
        evidence.append(f"FailureReason: {redact(failure_reason)}")
    if stage_failure is not None:
        evidence.append(f"stage {stage_failure.error_code}: {redact(stage_failure.message)}")

    if stopped_by_user:
        category = FailureCategory.STOPPED_BY_USER
        summary = "The job was stopped on request."
        action = None
    else:
        corpus = "\n".join(
            [failure_reason or "", stage_failure.message if stage_failure else "", *excerpt]
        )
        category, summary, action = classify(corpus)

    return FailureDiagnosis(
        execution_id=execution_id,
        job_name=job_name,
        stage=stage,
        model_name=model_name,
        diagnosed_at=now or datetime.now(UTC),
        category=category,
        summary=summary,
        recommended_action=action,
        failure_reason=redact(failure_reason) if failure_reason else None,
        stage_error_code=stage_failure.error_code if stage_failure else None,
        evidence=evidence,
        log_excerpt=excerpt,
        logs_truncated=truncated,
    )


def _read_stage_failure(
    store: ObjectStore, layout: ExperimentLayout, stage: str, model_name: str | None
) -> ModelFailure | None:
    if stage == "training" and model_name:
        uri = layout.model_failure(model_name)
    elif stage in _STAGE_FAILURE_DIR:
        uri = layout.path(f"{_STAGE_FAILURE_DIR[stage]}/failure.json")
    else:
        # A stage whose failure file is per-run (e.g. holdout evaluation) or unknown: fall back
        # to SageMaker's reason and the logs rather than raising.
        return None
    return read_model_if_exists(store, uri, ModelFailure)


_STAGE_FAILURE_DIR = {
    "profiling": "eda",
    "preparation": "validation",
    "evaluation": "comparison",
    "derived_profiling": "derived",
}


def fetch_logs(logs_client: Any, log_group: str, job_name: str) -> list[str]:
    """Best-effort bounded CloudWatch fetch. Any AWS error yields no logs, never an exception.

    SageMaker writes each job to ``<log_group>/<job_name>/...``; we read the most recent events
    across those streams and let ``bound_logs`` cap and redact them.
    """
    try:
        response = logs_client.filter_log_events(
            logGroupName=log_group,
            logStreamNamePrefix=job_name,
            limit=MAX_LOG_LINES,
        )
    except Exception:
        return []
    return [event.get("message", "") for event in response.get("events", [])]
