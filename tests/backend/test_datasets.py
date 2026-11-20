"""Dataset ingestion (requirement 10): validation, pre-signed upload, bounded preview."""

import boto3
import pytest
from moto import mock_aws

from backend.errors import ValidationError
from backend.services.datasets import DatasetService
from tests.support.datasets import ARTIFACT_ROOT

BUCKET = ARTIFACT_ROOT.removeprefix("s3://")


@pytest.fixture
def s3_client(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket=BUCKET)
        yield client


@pytest.fixture
def service(settings, store, s3_client):
    from ml_engine.io import S3ObjectStore

    return DatasetService(settings=settings, store=S3ObjectStore(s3_client), s3_client=s3_client)


def test_upload_ticket_is_a_presigned_put_into_the_upload_inbox(service):
    ticket = service.create_upload_ticket("Churn Data (final).csv")
    assert ticket.dataset_uri.startswith(f"{ARTIFACT_ROOT}/ml-factory/uploads/")
    assert ticket.dataset_uri.endswith("/Churn-Data-final.csv")  # spaces/parens sanitized away
    assert ticket.upload_url.startswith("https://")
    assert ticket.headers["x-amz-server-side-encryption"] == "AES256"


@pytest.mark.parametrize("filename", ["model.pkl", "data.json", "archive.zip", "report.xlsx"])
def test_unsupported_extensions_are_refused(service, filename):
    with pytest.raises(ValidationError):
        service.create_upload_ticket(filename)


def test_register_describes_columns_from_a_bounded_read(service, classification_frame):
    registered = service.ingest_bytes(
        filename="churn.csv", payload=classification_frame.to_csv(index=False).encode()
    )
    names = {c.name for c in registered.columns}
    assert {"churned", "tenure_months", "contract"} <= names
    assert registered.file_format == "csv"
    assert registered.preview_row_count <= service._settings.dataset_preview_rows


def test_an_empty_file_is_refused(service):
    with pytest.raises(ValidationError, match="empty"):
        service.ingest_bytes(filename="x.csv", payload=b"")


def test_a_file_too_large_is_refused(service):
    service._settings.max_dataset_upload_bytes = 10
    with pytest.raises(ValidationError, match="maximum upload size"):
        service.ingest_bytes(filename="x.csv", payload=b"a,b\n1,2\n3,4\n")


def test_binary_disguised_as_csv_is_refused(service):
    with pytest.raises(ValidationError, match="not text"):
        service.ingest_bytes(filename="x.csv", payload=b"PK\x03\x04 binary \x00 content")


def test_duplicate_columns_are_refused(service):
    with pytest.raises(ValidationError, match="duplicate column"):
        service.ingest_bytes(filename="x.csv", payload=b"a,b,a\n1,2,3\n")


def test_register_rejects_a_non_s3_uri(service):
    with pytest.raises(Exception):  # noqa: B017 - InvalidS3UriError or ValidationError
        service.register("/local/path.csv")


def test_register_rejects_a_missing_object(service):
    with pytest.raises(ValidationError, match="No uploaded object"):
        service.register(f"{ARTIFACT_ROOT}/ml-factory/uploads/none/missing.csv")


def test_list_uploads_returns_registered_datasets_for_reuse(service, classification_frame):
    csv = classification_frame.to_csv(index=False).encode()
    first = service.ingest_bytes(filename="alpha.csv", payload=csv)
    second = service.ingest_bytes(filename="beta.csv", payload=csv)

    listing = service.list_uploads()

    uris = {d.dataset_uri for d in listing.datasets}
    assert {first.dataset_uri, second.dataset_uri} <= uris
    assert listing.count == len(listing.datasets)
    assert all(d.size_bytes > 0 for d in listing.datasets)
    assert all(d.file_format == "csv" for d in listing.datasets)


def test_list_uploads_is_empty_when_nothing_was_uploaded(service):
    assert service.list_uploads().datasets == []


def test_list_uploads_respects_the_limit(service, classification_frame):
    csv = classification_frame.to_csv(index=False).encode()
    for name in ("one.csv", "two.csv", "three.csv"):
        service.ingest_bytes(filename=name, payload=csv)

    listing = service.list_uploads(limit=2)

    assert listing.count == 2
