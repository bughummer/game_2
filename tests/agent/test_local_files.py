"""Unit tests for ml_factory_cli.local_files (Req 7.1, 7.2)."""

from __future__ import annotations

import hashlib
import os
from datetime import UTC, datetime, timedelta

import pytest

from backend.agent.service import LocalFileInput
from ml_factory_cli.local_files import list_candidates, sha256_file


def _touch(path, data=b"x"):
    path.write_bytes(data)
    return path


def test_filters_extensions_case_insensitively_and_ignores_subdirectories(tmp_path):
    for name in ["b.csv", "A.CSV", "c.Parquet", "notes.txt", "data.csv.bak", "noext"]:
        _touch(tmp_path / name)
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub / "deep.csv")
    (tmp_path / "folder.csv").mkdir()  # a directory with a matching name is not a file

    files, total = list_candidates(tmp_path)

    assert [f["name"] for f in files] == ["A.CSV", "b.csv", "c.Parquet"]
    assert total == 3


def test_entries_match_local_file_input_shape(tmp_path):
    data = b"a,b\n1,2\n"
    p = _touch(tmp_path / "data.csv", data)
    when = datetime(2024, 5, 1, 12, 30, tzinfo=UTC)
    os.utime(p, (when.timestamp(), when.timestamp()))

    (entry,), total = list_candidates(tmp_path)

    assert total == 1
    assert entry == {
        "name": "data.csv",
        "size_bytes": len(data),
        "modified_at": when.isoformat(),
        "sha256": hashlib.sha256(data).hexdigest(),
    }
    parsed = datetime.fromisoformat(entry["modified_at"])
    assert parsed.utcoffset() == timedelta(0)
    LocalFileInput.model_validate(entry)  # accepted by register_local_files as-is


def test_caps_at_100_sorted_and_reports_total(tmp_path):
    for i in range(105):
        _touch(tmp_path / f"f{i:03d}.csv")

    files, total = list_candidates(tmp_path)

    assert total == 105
    assert len(files) == 100
    assert [f["name"] for f in files] == [f"f{i:03d}.csv" for i in range(100)]


def test_single_file_path(tmp_path):
    good = _touch(tmp_path / "one.PARQUET", b"pq")
    bad = _touch(tmp_path / "one.json")
    _touch(tmp_path / "other.csv")

    files, total = list_candidates(good)
    assert total == 1 and [f["name"] for f in files] == ["one.PARQUET"]

    assert list_candidates(bad) == ([], 0)


def test_empty_folder_and_missing_path(tmp_path):
    assert list_candidates(tmp_path) == ([], 0)
    with pytest.raises(FileNotFoundError):
        list_candidates(tmp_path / "missing")


def test_sha256_file_matches_hashlib_across_chunks(tmp_path):
    data = os.urandom(3 * 1024 * 1024 + 17)
    p = _touch(tmp_path / "big.csv", data)
    assert sha256_file(p) == hashlib.sha256(data).hexdigest()
    assert sha256_file(_touch(tmp_path / "empty.csv", b"")) == hashlib.sha256(b"").hexdigest()
