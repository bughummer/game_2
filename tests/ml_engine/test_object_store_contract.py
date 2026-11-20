"""One behavioural contract for every ObjectStore.

Each test runs against the production ``S3ObjectStore`` (over moto, so no AWS account) and the
test-only ``InMemoryObjectStore`` that the rest of the suite uses. If the two ever disagree, the
in-memory store is lying to the engine tests — that is what this file exists to catch.
Design Property 4: a contract round-trips through S3 byte-identically.
"""

import io
from pathlib import Path

import boto3
import pandas as pd
import pytest
from moto import mock_aws

from ml_engine.contracts.eda import EdaReport
from ml_engine.io import (
    ObjectNotFoundError,
    ObjectStore,
    S3ObjectStore,
    load_dataset,
    read_model,
    write_model,
)
from tests.support.in_memory_store import InMemoryObjectStore

BUCKET = "mlf-contract-test"
ROOT = f"s3://{BUCKET}"


@pytest.fixture
def s3_client(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture(params=["s3", "memory"])
def any_store(request) -> ObjectStore:
    if request.param == "s3":
        return S3ObjectStore(request.getfixturevalue("s3_client"))
    return InMemoryObjectStore()


def test_both_stores_satisfy_the_protocol(any_store):
    assert isinstance(any_store, ObjectStore)


def test_bytes_round_trip(any_store):
    uri = f"{ROOT}/nested/file.bin"
    any_store.write_bytes(uri, b"payload")
    assert any_store.read_bytes(uri) == b"payload"
    assert any_store.exists(uri)
    assert any_store.stat(uri).size_bytes == 7


def test_overwrite_replaces_the_object(any_store):
    uri = f"{ROOT}/a.txt"
    any_store.write_bytes(uri, b"one")
    any_store.write_bytes(uri, b"two")
    assert any_store.read_bytes(uri) == b"two"


def test_a_missing_object_is_reported_not_guessed(any_store, tmp_path):
    uri = f"{ROOT}/absent.json"
    with pytest.raises(ObjectNotFoundError):
        any_store.read_bytes(uri)
    with pytest.raises(ObjectNotFoundError):
        any_store.download(uri, tmp_path / "x")
    assert any_store.stat(uri) is None
    assert any_store.exists(uri) is False


def test_a_prefix_exists_when_something_is_under_it(any_store):
    any_store.write_bytes(f"{ROOT}/exp-1/eda/eda.json", b"{}")
    assert any_store.exists(f"{ROOT}/exp-1")
    assert any_store.exists(f"{ROOT}/exp-1/")
    assert not any_store.exists(f"{ROOT}/exp-2")


def test_listing_returns_full_sorted_uris(any_store):
    for key in ("exp-1/b.json", "exp-1/a.json", "exp-1/sub/c.json"):
        any_store.write_bytes(f"{ROOT}/{key}", b"{}")
    assert any_store.list_uris(f"{ROOT}/exp-1") == [
        f"{ROOT}/exp-1/a.json",
        f"{ROOT}/exp-1/b.json",
        f"{ROOT}/exp-1/sub/c.json",
    ]
    # Listings are addressable as-is: that is what records store.
    assert any_store.read_bytes(any_store.list_uris(f"{ROOT}/exp-1/")[0]) == b"{}"


def test_listing_matches_at_a_path_boundary_only(any_store):
    """``exp-1`` must never pick up ``exp-10``: S3 prefixes are plain strings."""
    any_store.write_bytes(f"{ROOT}/exp-1/a.json", b"{}")
    any_store.write_bytes(f"{ROOT}/exp-10/a.json", b"{}")
    assert any_store.list_uris(f"{ROOT}/exp-1") == [f"{ROOT}/exp-1/a.json"]
    assert any_store.list_prefixes(f"{ROOT}/exp-1") == []


def test_listing_an_object_returns_just_that_object(any_store):
    any_store.write_bytes(f"{ROOT}/data.csv", b"a\n1\n")
    any_store.write_bytes(f"{ROOT}/data.csv.bak", b"old")
    assert any_store.list_uris(f"{ROOT}/data.csv") == [f"{ROOT}/data.csv"]


def test_listing_nothing_is_empty(any_store):
    assert any_store.list_uris(f"{ROOT}/nothing-here/") == []
    assert any_store.list_prefixes(f"{ROOT}/nothing-here/") == []


def test_list_prefixes_returns_immediate_children(any_store):
    for key in ("experiments/exp-1/experiment.json", "experiments/exp-2/eda/eda.json"):
        any_store.write_bytes(f"{ROOT}/{key}", b"{}")
    any_store.write_bytes(f"{ROOT}/experiments/index.json", b"{}")
    assert any_store.list_prefixes(f"{ROOT}/experiments/") == [
        f"{ROOT}/experiments/exp-1/",
        f"{ROOT}/experiments/exp-2/",
    ]
    assert any_store.list_prefixes(ROOT) == [f"{ROOT}/experiments/"]


def test_writing_at_a_prefix_is_refused(any_store, tmp_path):
    source = tmp_path / "f"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match="prefix"):
        any_store.write_bytes(f"{ROOT}/dir/", b"x")
    with pytest.raises(ValueError, match="prefix"):
        any_store.upload(source, ROOT)


def test_upload_and_download_round_trip(any_store, tmp_path):
    source = tmp_path / "in.bin"
    source.write_bytes(b"\x00\x01binary\xff")
    any_store.upload(source, f"{ROOT}/blobs/in.bin")
    target = any_store.download(f"{ROOT}/blobs/in.bin", tmp_path / "out" / "deep" / "out.bin")
    assert target.read_bytes() == source.read_bytes()


def test_delete_removes_one_object_and_tolerates_a_missing_one(any_store):
    keep, gone = f"{ROOT}/d/keep.json", f"{ROOT}/d/gone.json"
    any_store.write_bytes(keep, b"1")
    any_store.write_bytes(gone, b"2")
    any_store.delete(gone)
    any_store.delete(gone)  # S3 semantics: deleting a missing key is not an error
    assert any_store.list_uris(f"{ROOT}/d/") == [keep]


def test_copy_object_copies_bytes_through_either_store(any_store):
    from ml_engine.io import copy_object

    any_store.write_bytes(f"{ROOT}/src/data.csv", b"a,b\n1,2\n")
    copy_object(any_store, f"{ROOT}/src/data.csv", f"{ROOT}/dst/data.csv")
    assert any_store.read_bytes(f"{ROOT}/dst/data.csv") == b"a,b\n1,2\n"
    with pytest.raises(ObjectNotFoundError):
        copy_object(any_store, f"{ROOT}/src/absent.csv", f"{ROOT}/dst/absent.csv")


def test_a_contract_round_trips_byte_identically(any_store, classification_eda):
    uri = f"{ROOT}/eda/eda.json"
    write_model(any_store, uri, classification_eda)
    expected = classification_eda.model_dump_json(indent=2, by_alias=True).encode("utf-8")
    assert any_store.read_bytes(uri) == expected
    assert read_model(any_store, uri, EdaReport) == classification_eda


def test_a_dataset_loads_the_same_through_either_store(any_store, regression_frame):
    text = regression_frame.to_csv(index=False)
    uri = f"{ROOT}/curated/prices.csv"
    any_store.write_bytes(uri, text.encode())
    loaded = load_dataset(any_store, uri)
    pd.testing.assert_frame_equal(loaded.frame, pd.read_csv(io.StringIO(text)))
    assert loaded.source_size_bytes == len(text.encode())


# --- S3-specific behaviour ---------------------------------------------------------


def test_listing_follows_pagination(s3_client):
    """list_objects_v2 returns at most 1000 keys per page; a long experiment must not truncate."""
    store = S3ObjectStore(s3_client)
    for index in range(1005):
        s3_client.put_object(Bucket=BUCKET, Key=f"many/obj-{index:04d}", Body=b"")
    assert len(store.list_uris(f"{ROOT}/many/")) == 1005


def test_every_write_is_encrypted_with_sse_s3_by_default(s3_client, tmp_path):
    store = S3ObjectStore(s3_client)
    store.write_bytes(f"{ROOT}/a.json", b"{}")
    source = tmp_path / "b"
    source.write_bytes(b"b")
    store.upload(source, f"{ROOT}/b.bin")
    for key in ("a.json", "b.bin"):
        assert s3_client.head_object(Bucket=BUCKET, Key=key)["ServerSideEncryption"] == "AES256"


def test_stat_reports_s3_metadata(s3_client):
    store = S3ObjectStore(s3_client)
    store.write_bytes(f"{ROOT}/a.json", b"{}")
    meta = store.stat(f"{ROOT}/a.json")
    assert meta.size_bytes == 2
    assert meta.etag and '"' not in meta.etag
    assert meta.last_modified is not None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"server_side_encryption": "none"}, "server_side_encryption"),
        ({"kms_key_id": "alias/x"}, "kms_key_id"),
    ],
)
def test_invalid_encryption_settings_are_refused(kwargs, message):
    with pytest.raises(ValueError, match=message):
        S3ObjectStore(object(), **kwargs)


def test_kms_settings_are_passed_on_every_write():
    calls = []

    class RecordingClient:
        def put_object(self, **kwargs):
            calls.append(("put", kwargs))

        def upload_file(self, filename, bucket, key, ExtraArgs):  # noqa: N803
            calls.append(("upload", ExtraArgs))

    store = S3ObjectStore(
        RecordingClient(), server_side_encryption="aws:kms", kms_key_id="alias/mlf"
    )
    store.write_bytes(f"{ROOT}/a", b"x")
    store.upload(Path(__file__), f"{ROOT}/b")
    assert calls[0][1]["ServerSideEncryption"] == "aws:kms"
    assert calls[0][1]["SSEKMSKeyId"] == "alias/mlf"
    assert calls[1][1] == {"ServerSideEncryption": "aws:kms", "SSEKMSKeyId": "alias/mlf"}


def test_non_missing_errors_are_not_swallowed():
    """An access-denied must surface; treating it as "missing" would hide an IAM problem."""

    class Denied(Exception):
        response = {"Error": {"Code": "AccessDenied"}}

    class DenyingClient:
        def get_object(self, **_):
            raise Denied()

        def head_object(self, **_):
            raise Denied()

    store = S3ObjectStore(DenyingClient())
    with pytest.raises(Denied):
        store.read_bytes(f"{ROOT}/a")
    with pytest.raises(Denied):
        store.stat(f"{ROOT}/a")
