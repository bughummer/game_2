"""Unit tests for ``backend.agent.uploads`` (task 12.1; Req 7.6, 7.7, 7.11, 7.12)."""

from __future__ import annotations

from datetime import UTC, datetime
from io import BytesIO

import pandas as pd
import pytest

from backend.agent.tools_core import dataset_id_for
from backend.agent.uploads import (
    RowCounter,
    UploadPlan,
    UploadRefusal,
    UploadReuse,
    reuse_or_plan,
)
from backend.config import Settings
from backend.errors import ValidationError
from ml_engine.contracts.agent import DatasetRef, LocalFileRef, Session
from tests.support.datasets import ARTIFACT_ROOT
from tests.support.in_memory_store import InMemoryObjectStore

ROLE = "arn:aws:iam::123456789012:role/mlf-sagemaker-job"
LIMIT = 1_000
SHA = "a" * 64


def _settings() -> Settings:
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn=ROLE,
        max_dataset_upload_bytes=LIMIT,
    )


def _file(name: str = "data.csv", size: int = 100, sha: str = SHA) -> LocalFileRef:
    return LocalFileRef(
        file_id="f1", name=name, size_bytes=size, modified_at=datetime.now(UTC), sha256=sha
    )


def _session(file: LocalFileRef, datasets=()) -> Session:
    return Session.model_construct(
        session_id="s1", owner="alice", local_files=[file], datasets=list(datasets)
    )


def _dataset(sha: str) -> DatasetRef:
    uri = f"{ARTIFACT_ROOT}/uploads/old-1234/data.csv"
    return DatasetRef(
        dataset_id=dataset_id_for(uri),
        uri=uri,
        filename="data.csv",
        sha256=sha,
        size_bytes=100,
        file_format="csv",
        columns=[],
        row_count=3,
    )


def test_oversized_file_is_refused_with_size_and_limit_in_bytes():
    file = _file(size=LIMIT + 1)
    outcome = reuse_or_plan(_session(file), file, _settings())
    assert isinstance(outcome, UploadRefusal)
    assert outcome.reason == "too_large"
    assert outcome.details["size_bytes"] == LIMIT + 1
    assert outcome.details["max_bytes"] == LIMIT
    assert f"{LIMIT + 1} bytes" in outcome.message and f"{LIMIT} bytes" in outcome.message


def test_file_at_the_limit_is_planned():
    file = _file(name="data.PARQUET", size=LIMIT)
    outcome = reuse_or_plan(_session(file), file, _settings())
    assert isinstance(outcome, UploadPlan)
    assert outcome.file_format == "parquet"
    assert outcome.destination_prefix.endswith("/uploads/")


@pytest.mark.parametrize("name", ["data.tsv", "data.txt", "data.xlsx", "data"])
def test_unsupported_extension_is_refused_naming_csv_and_parquet(name):
    file = _file(name=name)
    outcome = reuse_or_plan(_session(file), file, _settings())
    assert isinstance(outcome, UploadRefusal)
    assert outcome.reason == "unsupported_format"
    assert outcome.details["supported_formats"] == ["CSV", "Parquet"]
    assert "CSV" in outcome.message and "Parquet" in outcome.message


def test_unregistered_file_is_refused():
    file = _file()
    session = _session(file.model_copy(update={"file_id": "other"}))
    outcome = reuse_or_plan(session, file, _settings())
    assert isinstance(outcome, UploadRefusal) and outcome.reason == "unknown_file"


def test_same_sha256_reuses_the_registered_dataset():
    existing = _dataset(SHA)
    file = _file(name="renamed.csv", sha=SHA.upper())
    outcome = reuse_or_plan(_session(file, [existing]), file, _settings())
    assert isinstance(outcome, UploadReuse)
    assert outcome.to_result() == {"reused": True, "dataset_id": existing.dataset_id}
    assert outcome.dataset_id == dataset_id_for(existing.uri)


def test_different_sha256_plans_a_new_upload():
    file = _file()
    outcome = reuse_or_plan(_session(file, [_dataset("b" * 64)]), file, _settings())
    assert isinstance(outcome, UploadPlan)


def test_row_counter_counts_csv_and_parquet_and_refuses_corrupt_parquet():
    store = InMemoryObjectStore()
    frame = pd.DataFrame({"a": range(7), "b": ["x\ny"] * 7})  # quoted newlines are not rows
    csv_uri = f"{ARTIFACT_ROOT}/uploads/t/d.csv"
    pq_uri = f"{ARTIFACT_ROOT}/uploads/t/d.parquet"
    bad_uri = f"{ARTIFACT_ROOT}/uploads/t/bad.parquet"
    store.write_bytes(csv_uri, frame.to_csv(index=False).encode())
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    store.write_bytes(pq_uri, buffer.getvalue())
    store.write_bytes(bad_uri, b"not parquet")

    counter = RowCounter(store)
    assert counter.count(csv_uri, "csv") == 7
    assert counter.count(pq_uri, "parquet") == 7
    with pytest.raises(ValidationError) as error:
        counter.count(bad_uri, "parquet")
    assert error.value.details["supported_formats"] == ["CSV", "Parquet"]
