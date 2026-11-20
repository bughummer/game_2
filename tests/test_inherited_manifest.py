"""The inherited engine changes only on purpose (requirement 1.5).

A failure here means an inherited file changed (or a new file appeared) without a reviewed
manifest update. If the change is intended, explain it in the change and run ``make manifest``.
"""

import json

import pytest

from scripts.inherited_manifest import ADAPTED, MANIFEST, NEW, ROOT, digest, tracked_files


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_the_manifest_declares_the_same_adapted_and_new_files(manifest):
    assert manifest["adapted"] == ADAPTED
    assert manifest["new"] == NEW


def test_every_engine_file_is_inherited_adapted_or_declared_new(manifest):
    undeclared = sorted(set(tracked_files()) - set(manifest["files"]))
    assert not undeclared, (
        f"files not in the manifest: {undeclared}. Declare new code in "
        "scripts/inherited_manifest.py NEW, then run `make manifest`."
    )


def test_no_inherited_file_was_removed(manifest):
    missing = sorted(r for r in manifest["files"] if not (ROOT / r).is_file())
    assert not missing, f"inherited files removed: {missing}"


def test_no_inherited_file_changed(manifest):
    changed = sorted(
        relative
        for relative, expected in manifest["files"].items()
        if (ROOT / relative).is_file() and digest(ROOT / relative) != expected
    )
    assert not changed, (
        f"inherited files changed: {changed}. The engine is the specification; if this change "
        "is deliberate, justify it in review and run `make manifest`."
    )
