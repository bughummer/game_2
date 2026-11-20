"""S3-backed ``ObjectStore`` — the only production artifact store.

The boto3 S3 client is injected, so this module imports neither boto3 nor botocore and
``ml_engine`` stays installable without the AWS SDK. AWS errors are recognised by their error
code (``exc.response["Error"]["Code"]``), which is how botocore's ``ClientError`` exposes them.

Semantics match the store contract suite (``tests/ml_engine/test_object_store_contract.py``):

* listings return full ``s3://`` URIs, sorted;
* a listing prefix matches at a ``/`` boundary only — ``exp-1`` never matches ``exp-10/...``;
* ``list_prefixes`` returns immediate child prefixes ending in ``/``;
* ``exists`` is true for an object or for a non-empty "directory" prefix;
* a missing object raises ``ObjectNotFoundError`` from ``read_bytes``/``download`` and yields
  ``None`` from ``stat``.
"""

from pathlib import Path
from typing import Any

from ml_engine.io.object_store import ObjectMetadata, ObjectNotFoundError
from ml_engine.io.uri import parse_s3_uri, to_s3_uri

_MISSING_CODES = frozenset({"NoSuchKey", "404", "NotFound", "NoSuchBucket"})
SSE_S3 = "AES256"
SSE_KMS = "aws:kms"


def _error_code(error: Exception) -> str | None:
    response = getattr(error, "response", None)
    if not isinstance(response, dict):
        return None
    return str(response.get("Error", {}).get("Code", "")) or None


def _is_missing(error: Exception) -> bool:
    return _error_code(error) in _MISSING_CODES


class S3ObjectStore:
    """``ObjectStore`` over one S3 client.

    ``server_side_encryption`` is applied to every write: ``"AES256"`` (SSE-S3, the default) or
    ``"aws:kms"`` with an optional ``kms_key_id``.
    """

    def __init__(
        self,
        client: Any,
        *,
        server_side_encryption: str = SSE_S3,
        kms_key_id: str | None = None,
    ) -> None:
        if server_side_encryption not in {SSE_S3, SSE_KMS}:
            raise ValueError(
                f"server_side_encryption must be {SSE_S3!r} or {SSE_KMS!r}, "
                f"got {server_side_encryption!r}"
            )
        if kms_key_id and server_side_encryption != SSE_KMS:
            raise ValueError("kms_key_id requires server_side_encryption='aws:kms'")
        self._client = client
        self._write_args: dict[str, str] = {"ServerSideEncryption": server_side_encryption}
        if kms_key_id:
            self._write_args["SSEKMSKeyId"] = kms_key_id

    # --- reads ------------------------------------------------------------
    def read_bytes(self, uri: str) -> bytes:
        bucket, key = parse_s3_uri(uri)
        try:
            response = self._client.get_object(Bucket=bucket, Key=key)
        except Exception as error:
            if _is_missing(error):
                raise ObjectNotFoundError(f"No object at {uri}") from error
            raise
        return response["Body"].read()

    def exists(self, uri: str) -> bool:
        if self._head(uri) is not None:
            return True
        bucket, key = parse_s3_uri(uri)
        response = self._client.list_objects_v2(
            Bucket=bucket, Prefix=_directory_prefix(key), MaxKeys=1
        )
        return response.get("KeyCount", len(response.get("Contents", []))) > 0

    def stat(self, uri: str) -> ObjectMetadata | None:
        head = self._head(uri)
        if head is None:
            return None
        etag = head.get("ETag")
        return ObjectMetadata(
            uri=uri,
            size_bytes=int(head.get("ContentLength", 0)),
            last_modified=head.get("LastModified"),
            etag=etag.strip('"') if isinstance(etag, str) else None,
            version_id=head.get("VersionId"),
        )

    def list_uris(self, prefix: str) -> list[str]:
        """Every object at or under ``prefix``; a prefix naming an object returns just it."""
        bucket, key = parse_s3_uri(prefix)
        if key and not key.endswith("/") and self._head(prefix) is not None:
            return [to_s3_uri(bucket, key)]
        uris: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=_directory_prefix(key)):
            for item in page.get("Contents", []):
                if not item["Key"].endswith("/"):
                    uris.append(to_s3_uri(bucket, item["Key"]))
        return sorted(uris)

    def list_prefixes(self, prefix: str) -> list[str]:
        """Immediate child "directories" — one entry per experiment, not per object."""
        bucket, key = parse_s3_uri(prefix)
        found: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=_directory_prefix(key), Delimiter="/"):
            for item in page.get("CommonPrefixes", []):
                found.append(to_s3_uri(bucket, item["Prefix"]))
        return sorted(found)

    def download(self, uri: str, destination: Path) -> Path:
        bucket, key = parse_s3_uri(uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(bucket, key, str(destination))
        except Exception as error:
            if _is_missing(error):
                raise ObjectNotFoundError(f"No object at {uri}") from error
            raise
        return destination

    # --- writes -----------------------------------------------------------
    def write_bytes(self, uri: str, data: bytes) -> None:
        bucket, key = parse_s3_uri(uri)
        _require_key(uri, key)
        self._client.put_object(Bucket=bucket, Key=key, Body=data, **self._write_args)

    def upload(self, source: Path, uri: str) -> None:
        bucket, key = parse_s3_uri(uri)
        _require_key(uri, key)
        self._client.upload_file(str(source), bucket, key, ExtraArgs=dict(self._write_args))

    def delete(self, uri: str) -> None:
        bucket, key = parse_s3_uri(uri)
        _require_key(uri, key)
        self._client.delete_object(Bucket=bucket, Key=key)

    def copy(self, source_uri: str, destination_uri: str) -> None:
        """Server-side copy (managed, multipart for large objects), re-encrypted on write."""
        source_bucket, source_key = parse_s3_uri(source_uri)
        bucket, key = parse_s3_uri(destination_uri)
        _require_key(destination_uri, key)
        try:
            self._client.copy(
                {"Bucket": source_bucket, "Key": source_key},
                bucket,
                key,
                ExtraArgs=dict(self._write_args),
            )
        except Exception as error:
            if _is_missing(error):
                raise ObjectNotFoundError(f"No object at {source_uri}") from error
            raise

    # --- internals --------------------------------------------------------
    def _head(self, uri: str) -> dict[str, Any] | None:
        bucket, key = parse_s3_uri(uri)
        if not key or key.endswith("/"):
            return None
        try:
            return self._client.head_object(Bucket=bucket, Key=key)
        except Exception as error:
            if _is_missing(error):
                return None
            raise


def _directory_prefix(key: str) -> str:
    """The listing prefix for a "directory": empty at the bucket root, else ending in ``/``."""
    stripped = key.strip("/")
    return f"{stripped}/" if stripped else ""


def _require_key(uri: str, key: str) -> None:
    if not key or key.endswith("/"):
        raise ValueError(f"Cannot write an object at a prefix: {uri!r}")
