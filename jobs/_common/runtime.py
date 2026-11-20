"""Shared runtime for the stage entrypoints.

Every stage is a thin shell: parse arguments, build an object store, call a pure function from
``ml_engine``, write contracts back. Business logic never lives in this layer.

Each stage is also runnable on its own from a shell, which is how you reproduce one step of a
failed experiment without re-running the others.
"""

import argparse
import logging
import os
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from ml_engine.contracts.model import ModelFailure
from ml_engine.io import ExperimentLayout, ObjectStore, S3ObjectStore, write_model

LOGGER = logging.getLogger("ml_factory.jobs")


class JobError(RuntimeError):
    """A job failure with a stable, user-safe error code."""

    def __init__(self, code: str, message: str, details: dict[str, str] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}


def configure_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or os.environ.get("ML_FACTORY_LOG_LEVEL", "INFO")).upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


def build_store() -> ObjectStore:
    """The store every stage reads and writes through: always S3.

    One function so no stage constructs a store itself. Credentials and region come from the
    standard AWS chain — inside SageMaker, the job's execution role. boto3 is imported here, not
    at module import, so the stages stay importable (and testable) without the AWS SDK.

    Environment (set by the orchestrator on the job):
      ML_FACTORY_S3_SSE         "AES256" (default) or "aws:kms"
      ML_FACTORY_S3_KMS_KEY_ID  KMS key id/ARN when SSE is aws:kms
    """
    import boto3
    from botocore.config import Config

    client = boto3.client("s3", config=Config(retries={"mode": "standard", "max_attempts": 5}))
    return S3ObjectStore(
        client,
        server_side_encryption=os.environ.get("ML_FACTORY_S3_SSE", "AES256"),
        kms_key_id=os.environ.get("ML_FACTORY_S3_KMS_KEY_ID") or None,
    )


def stage_image_uri() -> str | None:
    """The image this job runs on, as set by the orchestrator (ML Factory 2.0)."""
    return os.environ.get("ML_FACTORY_STAGE_IMAGE_URI") or None


def training_job_name() -> str | None:
    """The SageMaker Training job this process runs in (``TRAINING_JOB_NAME``); ``None`` offline.

    Recorded per CV fold so a fold result can be traced to the job that produced it (CV Req 9.7).
    """
    return os.environ.get("TRAINING_JOB_NAME") or None


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument(
        "--artifact-base",
        required=True,
        help="Base URI of this experiment's artifacts (s3://bucket/ml-factory/experiments/<id>).",
    )
    return parser


def run_entrypoint(
    parser: argparse.ArgumentParser,
    handler: Callable[[argparse.Namespace, ObjectStore, ExperimentLayout], Any],
    argv: Sequence[str] | None = None,
    *,
    failure_uri: Callable[[argparse.Namespace, ExperimentLayout], str] | None = None,
) -> int:
    """Run a stage, converting any failure into a ``failure.json`` artifact and a non-zero exit.

    The exit code is for a shell; the artifact is for the UI, which gets a user-safe reason
    without a stack trace. The worker reads neither — it catches the exception directly — but a
    stage run by hand still behaves like a well-mannered command.
    """
    configure_logging()
    args = parser.parse_args(argv)
    layout = ExperimentLayout(base=args.artifact_base)
    store = build_store()
    job_name = parser.prog

    try:
        handler(args, store, layout)
    except JobError as error:
        LOGGER.exception("%s failed: %s", job_name, error.message)
        _write_failure(store, layout, args, error.code, error.message, error.details, failure_uri)
        return 1
    except Exception as error:
        LOGGER.exception("%s failed unexpectedly", job_name)
        _write_failure(
            store,
            layout,
            args,
            "INTERNAL_JOB_ERROR",
            f"{type(error).__name__}: {error}",
            {},
            failure_uri,
        )
        return 1
    LOGGER.info("%s completed", job_name)
    return 0


def _write_failure(
    store: ObjectStore,
    layout: ExperimentLayout,
    args: argparse.Namespace,
    code: str,
    message: str,
    details: dict[str, str],
    failure_uri: Callable[[argparse.Namespace, ExperimentLayout], str] | None,
) -> None:
    uri = failure_uri(args, layout) if failure_uri else layout.path("failure.json")
    failure = ModelFailure(
        experiment_id=args.experiment_id,
        model_name=getattr(args, "model_name", "") or "experiment",
        error_code=code,
        message=message,
        failed_at=datetime.now(UTC),
        details=details,
    )
    try:
        write_model(store, uri, failure)
    except Exception:
        LOGGER.exception("Could not write the failure artifact to %s", uri)
