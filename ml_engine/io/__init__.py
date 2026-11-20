"""IO boundary: object storage, dataset loading, URI handling and the artifact layout.

This is the only package inside ``ml_engine`` permitted to know about storage backends.
Adapted from old_repo: local-filesystem names removed, S3 names added.
"""

from ml_engine.io.dataset import (
    DatasetLoadError,
    LoadedDataset,
    UnsupportedDatasetFormatError,
    load_dataset,
    load_joblib,
    read_parquet,
    resolve_format,
    save_joblib,
    write_parquet,
)
from ml_engine.io.layout import ExperimentLayout
from ml_engine.io.object_store import (
    ObjectMetadata,
    ObjectNotFoundError,
    ObjectStore,
    copy_object,
    read_model,
    read_model_if_exists,
    write_json,
    write_model,
)
from ml_engine.io.s3 import S3ObjectStore
from ml_engine.io.uri import (
    InvalidS3UriError,
    detect_format,
    is_s3_uri,
    join_uri,
    normalize_storage_root,
    parse_s3_uri,
    to_s3_uri,
    uri_matches_prefix,
)

__all__ = [
    "DatasetLoadError",
    "ExperimentLayout",
    "InvalidS3UriError",
    "LoadedDataset",
    "ObjectMetadata",
    "ObjectNotFoundError",
    "ObjectStore",
    "S3ObjectStore",
    "UnsupportedDatasetFormatError",
    "copy_object",
    "detect_format",
    "is_s3_uri",
    "join_uri",
    "load_dataset",
    "load_joblib",
    "normalize_storage_root",
    "parse_s3_uri",
    "read_model",
    "read_model_if_exists",
    "read_parquet",
    "resolve_format",
    "save_joblib",
    "to_s3_uri",
    "uri_matches_prefix",
    "write_json",
    "write_model",
    "write_parquet",
]
