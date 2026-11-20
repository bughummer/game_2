"""The IO boundary: URI safety, the artifact layout and dataset loading.

Adapted from old_repo: the ``file://``/``LocalObjectStore`` tests were replaced by ``s3://``
validation tests; store behaviour itself is covered by ``test_object_store_contract.py``.
"""

import io

import pandas as pd
import pytest

from ml_engine.contracts.eda import EdaReport
from ml_engine.io import (
    ExperimentLayout,
    InvalidS3UriError,
    ObjectNotFoundError,
    copy_object,
    detect_format,
    load_dataset,
    normalize_storage_root,
    parse_s3_uri,
    read_model,
    read_model_if_exists,
    uri_matches_prefix,
    write_model,
    write_parquet,
)
from ml_engine.io.dataset import UnsupportedDatasetFormatError, resolve_format

ROOT = "s3://mlf-test-artifacts"
DATA = "s3://mlf-test-approved/curated"


def test_prefix_matching_only_matches_path_boundaries():
    """The dataset allow-list is a string comparison, so the boundary rule is the whole check."""
    assert uri_matches_prefix("s3://b/data/x.csv", "s3://b/data") is True
    assert uri_matches_prefix("s3://b/data", "s3://b/data") is True
    assert uri_matches_prefix("s3://b/database/x.csv", "s3://b/data") is False


@pytest.mark.parametrize(
    ("uri", "expected"),
    [("a/b.csv", "csv"), ("a/b.parquet", "parquet"), ("a/b.CSV", "csv"), ("a/b.txt", "unknown")],
)
def test_format_detection(uri, expected):
    assert detect_format(uri) == expected


def test_unsupported_format_is_rejected():
    with pytest.raises(UnsupportedDatasetFormatError):
        resolve_format("data.txt")


def test_layout_matches_the_documented_structure():
    layout = ExperimentLayout.for_experiment(ROOT, "exp-1")
    base = f"{ROOT}/ml-factory/experiments/exp-1"
    assert layout.base == base
    assert layout.eda == f"{base}/eda/eda.json"
    assert layout.leakage == f"{base}/eda/leakage.json"
    assert layout.experiment_config == f"{base}/config/experiment_config.json"
    assert layout.selected_features == f"{base}/config/selected_features.json"
    assert layout.preparation == f"{base}/validation/validation.json"
    assert layout.train_dataset == f"{base}/datasets/train.parquet"
    assert layout.comparison == f"{base}/comparison/comparison.json"
    assert layout.summary == f"{base}/report_data/experiment_summary.json"
    assert layout.model_metadata("xgboost") == f"{base}/models/xgboost/metadata.json"
    assert layout.model_failure("xgboost") == f"{base}/models/xgboost/failure.json"
    assert layout.reasoning == f"{base}/reasoning/reasoning.json"


def test_preprocessor_paths_are_strategy_specific():
    layout = ExperimentLayout.for_experiment(ROOT, "exp-1")
    assert layout.preprocessor("dense_numeric") != layout.preprocessor("native_categorical")


def test_the_experiment_prefix_holds_its_own_dataset():
    layout = ExperimentLayout.for_experiment(ROOT, "exp-1")

    assert layout.eda == f"{ROOT}/ml-factory/experiments/exp-1/eda/eda.json"
    assert (
        layout.input_dataset("churn.csv") == f"{ROOT}/ml-factory/experiments/exp-1/input/churn.csv"
    )


def test_the_upload_inbox_sits_beside_the_experiments():
    assert ExperimentLayout.uploads_prefix(ROOT) == f"{ROOT}/ml-factory/uploads/"
    assert (
        ExperimentLayout.upload_object(ROOT, "churn-ab12", "churn.csv")
        == f"{ROOT}/ml-factory/uploads/churn-ab12/churn.csv"
    )


# --- s3:// addressing ----------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://my-bucket/a/b.csv", ("my-bucket", "a/b.csv")),
        ("s3://my-bucket", ("my-bucket", "")),
        ("s3://my-bucket/", ("my-bucket", "")),
        ("s3://my.bucket.01/prefix/", ("my.bucket.01", "prefix/")),
    ],
)
def test_s3_uris_are_split_into_bucket_and_key(uri, expected):
    assert parse_s3_uri(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "file:///data/artifacts",
        "/data/artifacts",
        "https://my-bucket.s3.amazonaws.com/key",
        "s3://",
        "s3://ab/key",  # too short
        "s3://My-Bucket/key",  # uppercase
        "s3://-bucket/key",
        "s3://bucket-/key",
        "s3://bad..bucket/key",
        "s3://bucket/key\\with\\backslashes",
        "s3://bucket/key\x00nul",
    ],
)
def test_anything_but_a_well_formed_s3_uri_is_refused(uri):
    """A mis-parsed location would read or write somewhere the operator did not configure."""
    with pytest.raises(InvalidS3UriError):
        parse_s3_uri(uri)


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("s3://mlf-artifacts", "s3://mlf-artifacts"),
        ("s3://mlf-artifacts/", "s3://mlf-artifacts"),
        ("  s3://mlf-artifacts/team-a/  ", "s3://mlf-artifacts/team-a"),
        ("", ""),
    ],
)
def test_storage_roots_are_normalized(given, expected):
    assert normalize_storage_root(given) == expected


def test_a_non_s3_storage_root_is_refused():
    with pytest.raises(InvalidS3UriError):
        normalize_storage_root("/data/artifacts")


# --- the store helpers, through the store every test uses -------------------


def test_missing_object_raises(store):
    with pytest.raises(ObjectNotFoundError):
        store.read_bytes(f"{ROOT}/absent.json")
    assert store.stat(f"{ROOT}/absent.json") is None


def test_contracts_round_trip_through_the_store(store, classification_eda):
    uri = f"{ROOT}/eda.json"
    write_model(store, uri, classification_eda)
    restored = read_model(store, uri, EdaReport)
    assert restored.dataset == classification_eda.dataset
    assert len(restored.columns) == len(classification_eda.columns)
    assert read_model_if_exists(store, f"{ROOT}/nope.json", EdaReport) is None


def test_csv_and_parquet_load_identically(store, classification_frame):
    csv_uri = f"{DATA}/data.csv"
    store.write_bytes(csv_uri, classification_frame.to_csv(index=False).encode())
    parquet_uri = f"{DATA}/data.parquet"
    write_parquet(store, parquet_uri, classification_frame)

    from_csv = load_dataset(store, csv_uri)
    from_parquet = load_dataset(store, parquet_uri)
    assert from_csv.frame.shape == from_parquet.frame.shape
    assert list(from_csv.frame.columns) == list(from_parquet.frame.columns)
    assert from_csv.file_format == "csv"
    assert from_parquet.file_format == "parquet"


def test_tsv_is_loaded(store, regression_frame):
    uri = f"{DATA}/prices.tsv"
    store.write_bytes(uri, regression_frame.to_csv(index=False, sep="\t").encode())
    loaded = load_dataset(store, uri)
    assert loaded.file_format == "tsv"
    assert list(loaded.frame.columns) == list(regression_frame.columns)


def test_sampling_is_flagged(store, classification_frame):
    uri = f"{DATA}/data.csv"
    store.write_bytes(uri, classification_frame.to_csv(index=False).encode())
    loaded = load_dataset(store, uri, max_rows=50)
    assert len(loaded.frame) == 50
    assert loaded.sampled is True


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def test_parquet_prefix_of_parts_is_loaded(store):
    store.write_bytes(f"{DATA}/parts/part-0.parquet", _parquet_bytes(pd.DataFrame({"a": [1, 2]})))
    store.write_bytes(f"{DATA}/parts/part-1.parquet", _parquet_bytes(pd.DataFrame({"a": [3, 4]})))
    loaded = load_dataset(store, f"{DATA}/parts/")
    assert len(loaded.frame) == 4


def test_a_sibling_prefix_is_not_part_of_a_parquet_prefix(store):
    """``parts/`` must not pick up ``parts-old/``: S3 prefixes are strings, the store adds the /."""
    store.write_bytes(f"{DATA}/parts/part-0.parquet", _parquet_bytes(pd.DataFrame({"a": [1]})))
    store.write_bytes(f"{DATA}/parts-old/part-0.parquet", _parquet_bytes(pd.DataFrame({"a": [9]})))
    loaded = load_dataset(store, f"{DATA}/parts/")
    assert loaded.frame["a"].tolist() == [1]


def test_copying_an_object_leaves_the_original(store):
    source = f"{ROOT}/ml-factory/uploads/u1/churn.csv"
    destination = f"{ROOT}/ml-factory/experiments/exp-1/input/churn.csv"
    store.write_bytes(source, b"a,b\n1,2\n")

    copy_object(store, source, destination)

    assert store.read_bytes(destination) == b"a,b\n1,2\n"
    assert store.exists(source)


def test_a_dataset_records_its_size(store, regression_frame):
    uri = f"{DATA}/prices.csv"
    store.write_bytes(uri, regression_frame.to_csv(index=False).encode())

    loaded = load_dataset(store, uri)

    assert loaded.file_format == "csv"
    assert list(loaded.frame.columns) == list(regression_frame.columns)
    assert loaded.source_size_bytes > 0
