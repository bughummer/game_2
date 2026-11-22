"""Dataset ingestion — CSV/TSV/Parquet into S3 (requirement 10). Route D only; no DB, no SQL.

Two steps, so a dataset never travels through the MCP/LLM context (requirement 10.3):

1. ``create_upload_ticket`` — a pre-signed S3 PUT the caller uploads the raw bytes to directly.
2. ``register`` — the backend reads a bounded head of the stored object, validates it, and
   returns the column list the user picks a target from.

Validation (size, format, NUL byte, duplicate headers, bounded preview) is ported verbatim from
old_repo; only the storage path changed from local to S3 and the single upload-and-store call
was split into a pre-signed PUT plus a register step.
"""

import csv
import logging
import re
import unicodedata
from datetime import UTC, datetime, timedelta
from io import BytesIO, RawIOBase
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from backend.config import Settings
from backend.errors import ValidationError
from backend.schemas import (
    DatasetColumnPreview,
    DatasetList,
    DatasetSummary,
    RegisteredDataset,
    UploadTicket,
)
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    detect_format,
    parse_s3_uri,
    uri_matches_prefix,
)
from ml_engine.io.dataset import SUPPORTED_FORMATS

#: How much of a CSV/TSV the preview reads: enough for the preview rows of any sane file.
PREVIEW_BYTES = 2 * 1024 * 1024

LOGGER = logging.getLogger("ml_factory.services.datasets")

UPLOAD_FORMATS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".txt": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
}
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_NAME_LENGTH = 120


class DatasetParseError(ValidationError):
    """The uploaded bytes could not be read in the declared format."""


class DatasetService:
    def __init__(self, *, settings: Settings, store: ObjectStore, s3_client: Any = None) -> None:
        self._settings = settings
        self._store = store
        self._s3 = s3_client  # only needed to mint pre-signed URLs

    def create_upload_ticket(self, filename: str) -> UploadTicket:
        """A pre-signed PUT the caller uploads to. The backend never sees the bytes in transit."""
        if self._s3 is None:
            raise ValidationError("Uploads require an S3 client; none is configured.")
        safe_name = _sanitize_filename(filename or "dataset")
        _resolve_format(safe_name)  # reject an unsupported extension before minting a URL
        uri = ExperimentLayout.upload_object(
            self._settings.artifact_root, f"{Path(safe_name).stem}-{uuid4().hex[:8]}", safe_name
        )
        bucket, key = parse_s3_uri(uri)
        headers = {"x-amz-server-side-encryption": self._settings.s3_sse}
        params: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "ServerSideEncryption": self._settings.s3_sse,
        }
        if self._settings.s3_kms_key_id:
            params["SSEKMSKeyId"] = self._settings.s3_kms_key_id
            headers["x-amz-server-side-encryption-aws-kms-key-id"] = self._settings.s3_kms_key_id
        url = self._s3.generate_presigned_url(
            "put_object", Params=params, ExpiresIn=self._settings.presigned_url_ttl_seconds
        )
        return UploadTicket(
            dataset_uri=uri,
            upload_url=url,
            headers=headers,
            expires_at=datetime.now(UTC)
            + timedelta(seconds=self._settings.presigned_url_ttl_seconds),
            max_bytes=self._settings.max_dataset_upload_bytes,
        )

    def register(self, dataset_uri: str) -> RegisteredDataset:
        """Validate an uploaded object and describe its columns. Bounded read, no full load."""
        parse_s3_uri(dataset_uri)  # reject anything that is not a well-formed s3:// URI
        if ".." in dataset_uri or not self._allowed(dataset_uri):
            # Without this, register/preview could read any object the control-plane role can
            # see. Same allow-list experiment creation applies, plus experiments' own copies.
            raise ValidationError(
                "This dataset location is not approved. Upload the file, or ask for the "
                "location to be approved.",
                {"dataset_uri": dataset_uri},
            )
        metadata = self._store.stat(dataset_uri)
        if metadata is None:
            raise ValidationError(
                "No uploaded object is at that location. Upload the file first.",
                {"dataset_uri": dataset_uri},
            )
        if metadata.size_bytes == 0:
            raise ValidationError("The uploaded file is empty.")
        if metadata.size_bytes > self._settings.max_dataset_upload_bytes:
            raise ValidationError(
                "The dataset exceeds the maximum upload size for this platform.",
                {
                    "max_bytes": self._settings.max_dataset_upload_bytes,
                    "size_bytes": metadata.size_bytes,
                },
            )
        filename = _filename_from_uri(dataset_uri)
        file_format = _resolve_format(filename)
        if file_format == "parquet":
            frame = self._parquet_head(dataset_uri, metadata.size_bytes)
            columns, rows, truncated = self._describe(frame, b"", file_format)
        else:
            payload = self._read_bytes_for_preview(dataset_uri, metadata.size_bytes)
            columns, rows, truncated = self._preview(payload, file_format)
        return RegisteredDataset(
            dataset_uri=dataset_uri,
            filename=filename,
            file_format=file_format,
            size_bytes=metadata.size_bytes,
            columns=columns,
            preview_row_count=rows,
            truncated_preview=truncated,
            registered_at=datetime.now(UTC),
        )

    def list_uploads(self, *, limit: int = 100) -> DatasetList:
        """Datasets already uploaded to the inbox, newest first, so they can be reused.

        Lists the upload inbox only — the stable home of a dataset before an experiment owns
        it — and skips anything that is not a supported dataset file (a stray key, a folder
        marker). No bytes are read; this is metadata only.
        """
        inbox = ExperimentLayout.uploads_prefix(self._settings.artifact_root)
        summaries: list[DatasetSummary] = []
        for uri in self._store.list_uris(inbox):
            filename = _filename_from_uri(uri)
            try:
                file_format = _resolve_format(filename)
            except ValidationError:
                continue  # not a dataset object (e.g. a folder placeholder)
            metadata = self._store.stat(uri)
            if metadata is None or metadata.size_bytes == 0:
                continue
            summaries.append(
                DatasetSummary(
                    dataset_uri=uri,
                    filename=filename,
                    file_format=file_format,
                    size_bytes=metadata.size_bytes,
                    last_modified=metadata.last_modified,
                )
            )
        summaries.sort(key=lambda s: (s.last_modified is None, s.last_modified), reverse=True)
        capped = summaries[: max(1, limit)]
        return DatasetList(count=len(capped), datasets=capped)

    def ingest_bytes(self, *, filename: str, payload: bytes) -> RegisteredDataset:
        """Store bytes directly, then register. For an HTTP upload route and for tests."""
        if not payload:
            raise ValidationError("The uploaded file is empty.")
        if len(payload) > self._settings.max_dataset_upload_bytes:
            raise ValidationError(
                "The dataset exceeds the maximum upload size for this platform.",
                {"max_bytes": self._settings.max_dataset_upload_bytes, "size_bytes": len(payload)},
            )
        safe_name = _sanitize_filename(filename or "dataset")
        _resolve_format(safe_name)
        uri = ExperimentLayout.upload_object(
            self._settings.artifact_root, f"{Path(safe_name).stem}-{uuid4().hex[:8]}", safe_name
        )
        self._store.write_bytes(uri, payload)
        LOGGER.info("Stored upload %s (%d bytes)", uri, len(payload))
        return self.register(uri)

    def create_download_url(self, uri: str) -> dict[str, Any]:
        """A short-lived pre-signed GET for an artifact under the experiments namespace only."""
        if self._s3 is None:
            raise ValidationError("Downloads require an S3 client; none is configured.")
        experiments = ExperimentLayout.experiments_prefix(self._settings.artifact_root)
        if ".." in uri or not uri_matches_prefix(uri, experiments.rstrip("/")):
            raise ValidationError("Only experiment artifacts can be downloaded.", {"uri": uri})
        if self._store.stat(uri) is None:
            raise ValidationError("That artifact does not exist (yet).", {"uri": uri})
        bucket, key = parse_s3_uri(uri)
        ttl = self._settings.presigned_url_ttl_seconds
        url = self._s3.generate_presigned_url(
            "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=ttl
        )
        return {
            "uri": uri,
            "download_url": url,
            "expires_at": (datetime.now(UTC) + timedelta(seconds=ttl)).isoformat(),
        }

    # --- preview (bounded) --------------------------------------------------------------------
    def _allowed(self, uri: str) -> bool:
        prefixes = [
            ExperimentLayout.uploads_prefix(self._settings.artifact_root).rstrip("/"),
            ExperimentLayout.experiments_prefix(self._settings.artifact_root).rstrip("/"),
            *self._settings.allowed_dataset_prefixes,
        ]
        return any(uri_matches_prefix(uri, prefix) for prefix in prefixes)

    def _read_bytes_for_preview(self, uri: str, size: int | None = None) -> bytes:
        """The first ``PREVIEW_BYTES`` of the object (a ranged GET on S3), cut at a line end.

        Registering a 500 MB file used to read all of it into the control plane's memory, on
        every preview. A preview needs a few dozen rows, so a bounded head is enough.
        """
        limit = PREVIEW_BYTES
        if self._s3 is not None and size is not None and size > limit:
            bucket, key = parse_s3_uri(uri)
            response = self._s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{limit - 1}")
            payload = response["Body"].read()
        else:
            payload = self._store.read_bytes(uri)
        if len(payload) >= limit and (size is None or size > len(payload)):
            cut = payload.rfind(b"\n")
            payload = payload[: cut + 1] if cut > 0 else payload
        return payload

    def _parquet_head(self, uri: str, size: int) -> pd.DataFrame:
        """First rows of a Parquet file via ranged reads of its footer and first row group."""
        limit = self._settings.dataset_preview_rows
        if self._s3 is not None:
            bucket, key = parse_s3_uri(uri)
            source: Any = _RangedS3File(self._s3, bucket, key, size)
        else:
            source = BytesIO(self._store.read_bytes(uri))
        try:
            return _read_parquet_head(source, limit)
        except Exception as error:
            raise DatasetParseError(
                "The file could not be read as parquet.", {"reason": str(error)[:200]}
            ) from error

    def _preview(
        self, payload: bytes, file_format: str
    ) -> tuple[list[DatasetColumnPreview], int, bool]:
        limit = self._settings.dataset_preview_rows
        if file_format in {"csv", "tsv"}:
            _reject_binary(payload)
            _check_encoding(payload)
            _check_delimiter(payload, file_format)
        try:
            frame = _read_head(payload, file_format, limit)
        except ValidationError:
            raise
        except Exception as error:
            raise DatasetParseError(
                f"The file could not be read as {file_format}. Check the delimiter and encoding.",
                {"reason": str(error)[:200]},
            ) from error
        return self._describe(frame, payload, file_format)

    def _describe(
        self, frame: pd.DataFrame, payload: bytes, file_format: str
    ) -> tuple[list[DatasetColumnPreview], int, bool]:
        limit = self._settings.dataset_preview_rows
        if frame.columns.empty:
            raise ValidationError("The file has no columns.")
        _reject_duplicate_columns(payload, file_format, frame)
        columns = [
            DatasetColumnPreview(
                name=str(name),
                dtype=str(frame[name].dtype),
                non_null_in_sample=int(frame[name].notna().sum()),
                sample_values=[
                    str(value)[:80] for value in frame[name].dropna().unique()[:3].tolist()
                ],
            )
            for name in frame.columns
        ]
        return columns, len(frame), len(frame) >= limit


class _RangedS3File(RawIOBase):
    """A seekable, read-only view of one S3 object that fetches only the ranges read.

    pyarrow reads a Parquet footer from the end and then the row groups it needs, so the
    preview downloads the footer and the first row group instead of the whole file.
    """

    def __init__(self, client: Any, bucket: str, key: str, size: int) -> None:
        self._client, self._bucket, self._key, self._size = client, bucket, key, size
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self._position, 2: self._size}[whence]
        self._position = max(0, min(self._size, base + offset))
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self._position >= self._size:
            return b""
        end = self._size if size is None or size < 0 else min(self._size, self._position + size)
        response = self._client.get_object(
            Bucket=self._bucket, Key=self._key, Range=f"bytes={self._position}-{end - 1}"
        )
        data = response["Body"].read()
        self._position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _check_encoding(payload: bytes) -> None:
    """The stages read text as UTF-8; say so now instead of failing inside a SageMaker job."""
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError as error:
        hint = "Windows-1252/Latin-1" if _decodes(payload, "cp1252") else "an unknown encoding"
        raise ValidationError(
            f"The file is not UTF-8 (it looks like {hint}). Save it as UTF-8 CSV and upload "
            "it again.",
            {"byte_offset": error.start},
        ) from error


def _decodes(payload: bytes, encoding: str) -> bool:
    try:
        payload.decode(encoding)
        return True
    except UnicodeDecodeError:
        return False


_DELIMITER_NAMES = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe"}


def _check_delimiter(payload: bytes, file_format: str) -> None:
    """Refuse a file whose delimiter is not the one its format implies, naming the right one.

    A semicolon CSV read with commas becomes one wide column and every later stage would
    "work" on garbage. The stages only read ',' (csv) and tab (tsv), so detect, and explain.
    """
    expected = "\t" if file_format == "tsv" else ","
    sample = payload[:65_536].decode("utf-8", errors="replace")
    lines = [line for line in sample.splitlines()[:20] if line.strip()]
    if len(lines) < 1:
        return

    def width(delimiter: str) -> int:
        """Columns every sampled line agrees on (the narrowest line)."""
        rows = list(csv.reader(lines, delimiter=delimiter))
        return min((len(row) for row in rows), default=0)

    if width(expected) > 1:
        return
    for candidate in (",", "\t", ";", "|"):
        if candidate != expected and width(candidate) > 1:
            suggestion = (
                "Rename it to .tsv"
                if candidate == "\t"
                else ("Rename it to .csv" if candidate == "," else "Re-export it comma-separated")
            )
            raise ValidationError(
                f"The file is {_DELIMITER_NAMES[candidate]}-separated, but a .{file_format} file "
                f"must be {_DELIMITER_NAMES[expected]}-separated. {suggestion} and upload it "
                "again.",
                {"detected_delimiter": _DELIMITER_NAMES[candidate]},
            )


def _reject_binary(payload: bytes) -> None:
    if b"\x00" in payload[:8192]:
        raise ValidationError(
            "That file is not text. Export it as CSV, TSV or Parquet and upload it again."
        )


def _reject_duplicate_columns(payload: bytes, file_format: str, frame: pd.DataFrame) -> None:
    if file_format in {"csv", "tsv"}:
        delimiter = "\t" if file_format == "tsv" else ","
        header = payload.split(b"\n", 1)[0].decode("utf-8", errors="replace")
        names = next(csv.reader([header], delimiter=delimiter), [])
    else:
        names = [str(name) for name in frame.columns]
    seen: set[str] = set()
    duplicates = sorted({name for name in names if name in seen or seen.add(name)})
    if duplicates:
        raise ValidationError(
            "The file has duplicate column names. Make them unique and upload it again.",
            {"duplicate_columns": duplicates},
        )


def _read_head(payload: bytes, file_format: str, rows: int) -> pd.DataFrame:
    buffer = BytesIO(payload)
    if file_format == "csv":
        return pd.read_csv(buffer, nrows=rows)
    if file_format == "tsv":
        return pd.read_csv(buffer, sep="\t", nrows=rows)
    if file_format == "parquet":
        return _read_parquet_head(buffer, rows)
    raise ValidationError(f"Unsupported dataset format {file_format!r}.")


def _read_parquet_head(buffer: Any, rows: int) -> pd.DataFrame:
    import pyarrow.parquet as pq

    reader = pq.ParquetFile(buffer)
    for batch in reader.iter_batches(batch_size=rows):
        return batch.to_pandas()
    return reader.schema_arrow.empty_table().to_pandas()


def _resolve_format(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    fmt = UPLOAD_FORMATS.get(suffix) or detect_format(filename)
    if fmt not in SUPPORTED_FORMATS:
        raise ValidationError(
            "Unsupported file type. Upload a .csv, .tsv or .parquet file.",
            {"filename": filename, "supported": sorted(set(UPLOAD_FORMATS))},
        )
    return fmt


def _filename_from_uri(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1] or "dataset"


def _sanitize_filename(filename: str) -> str:
    base = Path(filename.replace("\\", "/")).name
    normalized = unicodedata.normalize("NFKD", base).encode("ascii", "ignore").decode("ascii")
    cleaned = _UNSAFE_NAME.sub("-", normalized).strip("-.")
    if not cleaned:
        raise ValidationError(
            "The file name contains no usable characters. Rename the file and try again.",
            {"filename": filename},
        )
    path = Path(cleaned)
    stem = path.stem.strip("-._")[:_MAX_NAME_LENGTH].strip("-._") or "dataset"
    return f"{stem}{path.suffix.lower()}"
