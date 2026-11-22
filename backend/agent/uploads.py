"""Local-folder uploads: pre-transfer checks, duplicate reuse and row counting (design C16;
Req 7.6, 7.7, 7.11, 7.12).

``reuse_or_plan`` decides, from the client's manifest alone, what happens to a local file the
user asked to upload. It reads no bytes and touches no storage:

* a file the client never registered in ``session.local_files`` is refused;
* an extension other than ``.csv`` or ``.parquet`` is refused (Req 7.7);
* a size above ``max_dataset_upload_bytes`` is refused before any transfer (Req 7.6);
* a SHA-256 equal to a dataset already in ``session.datasets`` reuses that dataset (Req 7.11);
* anything else becomes an ``UploadPlan`` for the approval gate.

``RowCounter`` counts the rows of a stored dataset: from the Parquet footer, or by a chunked
read of one CSV column, so a large file is never held in memory.

Dataset ids come from ``tools_core.dataset_id_for`` (Req 7.12); this module defines no second
copy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import PurePath
from typing import Any, Literal

import pandas as pd

from backend.agent.tools_core import dataset_id_for
from backend.config import Settings
from backend.errors import ValidationError
from backend.services.datasets import _RangedS3File
from ml_engine.contracts.agent import DatasetRef, LocalFileRef, Session
from ml_engine.io import ExperimentLayout, ObjectStore, parse_s3_uri

__all__ = [
    "SUPPORTED_EXTENSIONS",
    "SUPPORTED_FORMAT_NAMES",
    "RowCounter",
    "UploadPlan",
    "UploadRefusal",
    "UploadReuse",
    "dataset_id_for",
    "format_for",
    "reuse_or_plan",
]

#: The only formats an agent upload accepts. Narrower than ``DatasetService`` on purpose.
SUPPORTED_EXTENSIONS: dict[str, Literal["csv", "parquet"]] = {
    ".csv": "csv",
    ".parquet": "parquet",
}
SUPPORTED_FORMAT_NAMES: tuple[str, ...] = ("CSV", "Parquet")
_SUPPORTED_TEXT = "CSV (.csv) and Parquet (.parquet)"

#: Rows per chunk when counting CSV rows.
CSV_CHUNK_ROWS = 100_000

RefusalReason = Literal["unknown_file", "unsupported_format", "too_large"]


@dataclass(frozen=True)
class UploadRefusal:
    """A structured refusal. Nothing was transferred and nothing was registered."""

    reason: RefusalReason
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    code: str = ValidationError.code

    def to_error(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "reason": self.reason,
            "message": self.message,
            "details": dict(self.details),
        }


@dataclass(frozen=True)
class UploadReuse:
    """The file's content is already registered in this session; reuse it (Req 7.11)."""

    dataset: DatasetRef

    @property
    def dataset_id(self) -> str:
        return self.dataset.dataset_id

    def to_result(self) -> dict[str, Any]:
        return {"reused": True, "dataset_id": self.dataset.dataset_id}


@dataclass(frozen=True)
class UploadPlan:
    """The file passed the manifest checks; it may go to the approval gate."""

    file: LocalFileRef
    file_format: Literal["csv", "parquet"]
    destination_prefix: str

    @property
    def size_bytes(self) -> int:
        return self.file.size_bytes


def format_for(filename: str) -> Literal["csv", "parquet"] | None:
    """The upload format a file name's extension names, or ``None`` if unsupported."""
    return SUPPORTED_EXTENSIONS.get(PurePath(filename.replace("\\", "/")).suffix.lower())


def reuse_or_plan(
    session: Session, file: LocalFileRef, settings: Settings
) -> UploadPlan | UploadRefusal | UploadReuse:
    """Decide from the manifest alone: refuse, reuse an existing dataset, or plan an upload."""
    if not any(f.file_id == file.file_id for f in session.local_files):
        return UploadRefusal(
            reason="unknown_file",
            message="That file is not in the list the client registered. List the folder again.",
            details={"file_id": file.file_id},
        )

    file_format = format_for(file.name)
    if file_format is None:
        return UploadRefusal(
            reason="unsupported_format",
            message=(
                f"The file {file.name!r} is not a supported dataset. Supported formats: "
                f"{_SUPPORTED_TEXT}."
            ),
            details={
                "filename": file.name,
                "extension": PurePath(file.name).suffix.lower(),
                "supported_formats": list(SUPPORTED_FORMAT_NAMES),
            },
        )

    limit = settings.max_dataset_upload_bytes
    if file.size_bytes > limit:
        return UploadRefusal(
            reason="too_large",
            message=(
                f"The file {file.name!r} is {file.size_bytes} bytes, over the upload limit of "
                f"{limit} bytes. Nothing was uploaded."
            ),
            details={"filename": file.name, "size_bytes": file.size_bytes, "max_bytes": limit},
        )

    digest = file.sha256.lower()
    for dataset in session.datasets:
        if dataset.sha256 is not None and dataset.sha256.lower() == digest:
            return UploadReuse(dataset=dataset)

    return UploadPlan(
        file=file,
        file_format=file_format,
        destination_prefix=ExperimentLayout.uploads_prefix(settings.artifact_root),
    )


class RowCounter:
    """Counts the rows of a stored dataset without loading it whole.

    With an S3 client, Parquet reads only the footer through ranged GETs and CSV streams the
    object body. Without one (tests, local stores) the bytes come from the ``ObjectStore``.
    """

    def __init__(self, store: ObjectStore, s3_client: Any = None) -> None:
        self._store = store
        self._s3 = s3_client

    def count(self, uri: str, file_format: str) -> int:
        try:
            if file_format == "parquet":
                return self._count_parquet(uri)
            if file_format == "csv":
                return self._count_csv(uri)
        except ValidationError:
            raise
        except Exception as error:  # any parse failure is a format refusal
            raise ValidationError(
                f"The file could not be read as {file_format}. Supported formats: "
                f"{_SUPPORTED_TEXT}.",
                {
                    "reason": str(error)[:200],
                    "supported_formats": list(SUPPORTED_FORMAT_NAMES),
                },
            ) from error
        raise ValidationError(
            f"Unsupported dataset format {file_format!r}. Supported formats: {_SUPPORTED_TEXT}.",
            {"file_format": file_format, "supported_formats": list(SUPPORTED_FORMAT_NAMES)},
        )

    def _count_parquet(self, uri: str) -> int:
        import pyarrow.parquet as pq

        if self._s3 is not None:
            metadata = self._store.stat(uri)
            if metadata is None:
                raise ValidationError("No uploaded object is at that location.", {"uri": uri})
            bucket, key = parse_s3_uri(uri)
            source: Any = _RangedS3File(self._s3, bucket, key, metadata.size_bytes)
        else:
            source = BytesIO(self._store.read_bytes(uri))
        return int(pq.ParquetFile(source).metadata.num_rows)

    def _count_csv(self, uri: str) -> int:
        if self._s3 is not None:
            bucket, key = parse_s3_uri(uri)
            source: Any = self._s3.get_object(Bucket=bucket, Key=key)["Body"]
        else:
            source = BytesIO(self._store.read_bytes(uri))
        total = 0
        with pd.read_csv(source, usecols=[0], chunksize=CSV_CHUNK_ROWS) as chunks:
            for chunk in chunks:
                total += len(chunk)
        return total
