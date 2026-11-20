"""List and hash candidate dataset files on the user's machine (Req 7.1, 7.2).

Each listed file is a dict shaped like ``backend.agent.service.LocalFileInput``
(``name``, ``size_bytes``, ``modified_at``, ``sha256``) so the CLI can pass it to
``register_local_files`` unchanged.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NamedTuple

SUPPORTED_EXTENSIONS = frozenset({".csv", ".parquet"})
MAX_CANDIDATES = 100
_CHUNK_BYTES = 1024 * 1024


class Candidates(NamedTuple):
    files: list[dict[str, Any]]  # at most MAX_CANDIDATES, sorted by name
    total: int  # number of candidates found, before truncation


def sha256_file(path: str | Path) -> str:
    """Hex SHA-256 of the file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while chunk := fh.read(_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _is_candidate(p: Path) -> bool:
    return p.suffix.lower() in SUPPORTED_EXTENSIONS and p.is_file()


def _describe(p: Path) -> dict[str, Any]:
    st = p.stat()
    return {
        "name": p.name,
        "size_bytes": st.st_size,
        "modified_at": datetime.fromtimestamp(st.st_mtime, UTC).isoformat(),
        "sha256": sha256_file(p),
    }


def list_candidates(path: str | Path) -> Candidates:
    """Candidate ``.csv``/``.parquet`` files in a folder (non-recursive), or the named file.

    Raises ``FileNotFoundError`` if ``path`` does not exist.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"no such file or folder: {p}")
    if p.is_file():
        found = [p] if _is_candidate(p) else []
    else:
        found = sorted((c for c in p.iterdir() if _is_candidate(c)), key=lambda c: c.name)
    return Candidates([_describe(c) for c in found[:MAX_CANDIDATES]], len(found))
