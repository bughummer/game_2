"""Object store boundary.

``ObjectStore`` is the only way ML Factory code touches bytes it did not create. The production
implementation is ``S3ObjectStore`` (``ml_engine/io/s3.py``); tests use an in-memory store that
honours the same contract.

The protocol remains a protocol rather than being collapsed into the one implementation. It is
what keeps every layer above this one from calling boto3 directly — code that reads artifacts is
handed a store, so the boundary stays visible and testable.

Adapted from old_repo: ``LocalObjectStore`` and its ``file://`` path handling were removed. The
Protocol and every helper below are inherited unchanged.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class ObjectNotFoundError(FileNotFoundError):
    """Raised when a requested object does not exist."""


@dataclass(frozen=True, slots=True)
class ObjectMetadata:
    uri: str
    size_bytes: int
    last_modified: datetime | None = None
    etag: str | None = None
    version_id: str | None = None


@runtime_checkable
class ObjectStore(Protocol):
    """Minimal, testable storage contract."""

    def read_bytes(self, uri: str) -> bytes: ...

    def write_bytes(self, uri: str, data: bytes) -> None: ...

    def exists(self, uri: str) -> bool: ...

    def list_uris(self, prefix: str) -> list[str]: ...

    def list_prefixes(self, prefix: str) -> list[str]: ...

    def stat(self, uri: str) -> ObjectMetadata | None: ...

    def download(self, uri: str, destination: Path) -> Path: ...

    def upload(self, source: Path, uri: str) -> None: ...

    def delete(self, uri: str) -> None:
        """Remove one object. Deleting a missing object is not an error (S3 semantics)."""
        ...


def write_model(store: ObjectStore, uri: str, model: BaseModel) -> str:
    """Serialize a contract to JSON and store it. Returns the URI for convenience."""
    payload = model.model_dump_json(indent=2, by_alias=True)
    store.write_bytes(uri, payload.encode("utf-8"))
    return uri


def read_model[ModelT: BaseModel](store: ObjectStore, uri: str, model_type: type[ModelT]) -> ModelT:
    """Read and validate a contract artifact. Raises ``ObjectNotFoundError`` if absent."""
    raw = store.read_bytes(uri)
    return model_type.model_validate_json(raw)


def read_model_if_exists[ModelT: BaseModel](
    store: ObjectStore, uri: str, model_type: type[ModelT]
) -> ModelT | None:
    try:
        return read_model(store, uri, model_type)
    except (ObjectNotFoundError, FileNotFoundError):
        return None


def write_json(store: ObjectStore, uri: str, payload: dict) -> str:
    store.write_bytes(uri, json.dumps(payload, indent=2, default=str).encode("utf-8"))
    return uri


def copy_object(store: ObjectStore, source_uri: str, destination_uri: str) -> str:
    """Copy one object to another location through the store, streaming via a temp file.

    Used to pin an uploaded dataset inside its experiment's own prefix, so the experiment
    carries the exact bytes it was trained on even if the upload inbox is later pruned.
    """
    import tempfile

    if source_uri == destination_uri:
        return destination_uri
    native = getattr(store, "copy", None)
    if callable(native):  # S3: server-side copy, the bytes never leave AWS
        native(source_uri, destination_uri)
        return destination_uri
    with tempfile.TemporaryDirectory(prefix="mlf-copy-") as tmp:
        local = store.download(source_uri, Path(tmp) / "object")
        store.upload(local, destination_uri)
    return destination_uri
