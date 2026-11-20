"""URI parsing and validation.

The only place in the platform that knows how a storage location is spelled. Every artifact and
dataset lives in S3 and is addressed as ``s3://bucket/key``, so every location in a record, a
contract or an API response is an absolute, unambiguous URI.

Adapted from old_repo: the ``file://`` helpers were removed (there is no local store) and
``s3://`` parsing was added. ``join_uri``, ``uri_matches_prefix`` and ``detect_format`` are
inherited unchanged.
"""

import re

S3_SCHEME = "s3://"

# S3 general-purpose bucket naming rules: 3-63 chars, lowercase letters, digits, dots and
# hyphens, starting and ending with a letter or digit.
_BUCKET_PATTERN = re.compile(r"^[a-z0-9][a-z0-9.\-]{1,61}[a-z0-9]$")


class InvalidS3UriError(ValueError):
    """Raised when a string is not a usable ``s3://`` URI."""


def is_s3_uri(value: str) -> bool:
    return value.startswith(S3_SCHEME)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``. The key may be empty (bucket root).

    Anything that is not a well-formed S3 URI is refused rather than guessed at, because a
    mis-parsed location would read or write somewhere the operator did not configure.
    """
    if not isinstance(uri, str) or not is_s3_uri(uri):
        raise InvalidS3UriError(f"Not an s3:// URI: {uri!r}")
    remainder = uri[len(S3_SCHEME) :]
    bucket, _, key = remainder.partition("/")
    if not _BUCKET_PATTERN.match(bucket) or ".." in bucket:
        raise InvalidS3UriError(f"Invalid S3 bucket name in {uri!r}")
    if "\\" in key or "\x00" in key:
        raise InvalidS3UriError(f"Invalid S3 key in {uri!r}")
    return bucket, key


def to_s3_uri(bucket: str, key: str) -> str:
    return f"{S3_SCHEME}{bucket}/{key}" if key else f"{S3_SCHEME}{bucket}"


def normalize_storage_root(value: str) -> str:
    """Validate an artifact root and return it without a trailing slash.

    An empty value stays empty so configuration validation can report it as missing.
    """
    candidate = value.strip().rstrip("/")
    if not candidate:
        return ""
    parse_s3_uri(candidate)
    return candidate


def join_uri(base: str, *parts: str) -> str:
    """Join URI or path components with a single separator."""
    cleaned = [p.strip("/") for p in parts if p]
    if not cleaned:
        return base
    return "/".join([base.rstrip("/"), *cleaned])


def uri_matches_prefix(uri: str, allowed_prefix: str) -> bool:
    """Allow-list check. A prefix only matches at a path boundary, never mid-segment."""
    normalized_prefix = allowed_prefix.rstrip("/")
    if uri == normalized_prefix:
        return True
    return uri.startswith(normalized_prefix + "/")


def detect_format(uri: str) -> str:
    """Infer the dataset file format from the URI suffix.

    A trailing slash means a directory of part files, which the platform reads as parquet.
    """
    if uri.endswith("/"):
        return "parquet"
    lowered = uri.lower()
    for suffix, fmt in (
        (".csv.gz", "csv"),
        (".csv", "csv"),
        (".tsv", "tsv"),
        (".parquet", "parquet"),
        (".pq", "parquet"),
        (".json", "json"),
        (".xlsx", "xlsx"),
        (".xls", "xlsx"),
    ):
        if lowered.endswith(suffix):
            return fmt
    return "unknown"
