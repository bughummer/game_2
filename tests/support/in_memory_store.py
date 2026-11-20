"""A test-only ``ObjectStore`` with S3 semantics, held in a dict.

It exists so the inherited engine and job tests run offline and fast. It is not shipped: the
production store is always ``S3ObjectStore``. Both are held to the same behaviour by
``tests/ml_engine/test_object_store_contract.py``, so a difference between them is a test
failure, not a surprise in SageMaker.
"""

import shutil
import threading
from datetime import UTC, datetime
from hashlib import md5
from pathlib import Path

from ml_engine.io import ObjectMetadata, ObjectNotFoundError
from ml_engine.io.uri import parse_s3_uri, to_s3_uri


class InMemoryObjectStore:
    def __init__(self) -> None:
        self._objects: dict[tuple[str, str], tuple[bytes, datetime]] = {}
        self._lock = threading.Lock()

    # --- reads ------------------------------------------------------------
    def read_bytes(self, uri: str) -> bytes:
        bucket, key = parse_s3_uri(uri)
        with self._lock:
            entry = self._objects.get((bucket, key))
        if entry is None or not key or key.endswith("/"):
            raise ObjectNotFoundError(f"No object at {uri}")
        return entry[0]

    def exists(self, uri: str) -> bool:
        bucket, key = parse_s3_uri(uri)
        directory = _directory_prefix(key)
        with self._lock:
            return (bucket, key) in self._objects or any(
                b == bucket and k.startswith(directory) for b, k in self._objects
            )

    def stat(self, uri: str) -> ObjectMetadata | None:
        bucket, key = parse_s3_uri(uri)
        with self._lock:
            entry = self._objects.get((bucket, key))
        if entry is None:
            return None
        data, modified = entry
        return ObjectMetadata(
            uri=uri, size_bytes=len(data), last_modified=modified, etag=md5(data).hexdigest()
        )

    def list_uris(self, prefix: str) -> list[str]:
        bucket, key = parse_s3_uri(prefix)
        with self._lock:
            if key and not key.endswith("/") and (bucket, key) in self._objects:
                return [to_s3_uri(bucket, key)]
            directory = _directory_prefix(key)
            return sorted(
                to_s3_uri(b, k) for b, k in self._objects if b == bucket and k.startswith(directory)
            )

    def list_prefixes(self, prefix: str) -> list[str]:
        bucket, key = parse_s3_uri(prefix)
        directory = _directory_prefix(key)
        children: set[str] = set()
        with self._lock:
            for b, k in self._objects:
                if b != bucket or not k.startswith(directory):
                    continue
                rest = k[len(directory) :]
                if "/" in rest:
                    children.add(directory + rest.split("/", 1)[0] + "/")
        return sorted(to_s3_uri(bucket, child) for child in children)

    def download(self, uri: str, destination: Path) -> Path:
        data = self.read_bytes(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        return destination

    # --- writes -----------------------------------------------------------
    def write_bytes(self, uri: str, data: bytes) -> None:
        bucket, key = parse_s3_uri(uri)
        if not key or key.endswith("/"):
            raise ValueError(f"Cannot write an object at a prefix: {uri!r}")
        with self._lock:
            self._objects[(bucket, key)] = (bytes(data), datetime.now(UTC))

    def upload(self, source: Path, uri: str) -> None:
        self.write_bytes(uri, Path(source).read_bytes())

    def delete(self, uri: str) -> None:
        bucket, key = parse_s3_uri(uri)
        if not key or key.endswith("/"):
            raise ValueError(f"Cannot delete a prefix: {uri!r}")
        with self._lock:
            self._objects.pop((bucket, key), None)

    def copy(self, source_uri: str, destination_uri: str) -> None:
        self.write_bytes(destination_uri, self.read_bytes(source_uri))

    # --- test helpers -----------------------------------------------------
    def export_to(self, directory: Path) -> None:
        """Dump every object to disk, for debugging a failing test by hand."""
        shutil.rmtree(directory, ignore_errors=True)
        for (bucket, key), (data, _) in self._objects.items():
            target = directory / bucket / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)


def _directory_prefix(key: str) -> str:
    stripped = key.strip("/")
    return f"{stripped}/" if stripped else ""
