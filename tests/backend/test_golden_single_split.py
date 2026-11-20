"""Single_Split_Mode artifacts stay byte-identical to the pre-feature golden fixtures.

The fixtures under ``tests/fixtures/pre_cv/`` were generated from the code before the
cross-validation feature (``python -m tests.support.golden --write``). A fresh run of the same
experiment, masked the same way, must reproduce them exactly (requirements 7.7, 12.3, 13.6).
"""

import json

import pytest

from tests.support.golden import (
    GOLDEN_FILES,
    collect_masked,
    collect_raw,
    read_golden,
    run_single_split_experiment,
)

#: Keys the cross-validation feature adds. None may appear in a single-split artifact.
CV_KEYS = {"evaluation_mode", "cv", "test_fraction", "cross_validation", "test_fold"}
CV_KEYS |= {"test_metrics", "test_evaluation"}


@pytest.fixture(scope="module")
def fresh_run():
    from tests.support.in_memory_store import InMemoryObjectStore

    store = InMemoryObjectStore()
    layout = run_single_split_experiment(store)
    return store, layout


@pytest.mark.parametrize("name", sorted(GOLDEN_FILES))
def test_single_split_artifact_matches_the_pre_feature_fixture(fresh_run, name):
    store, layout = fresh_run
    assert collect_masked(store, layout)[name] == read_golden()[name]


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for item in value.values() for k in _keys(item)}
    if isinstance(value, list):
        return {k for item in value for k in _keys(item)}
    return set()


@pytest.mark.parametrize("name", sorted(GOLDEN_FILES))
def test_single_split_artifacts_carry_no_cv_keys(fresh_run, name):
    store, layout = fresh_run
    document = json.loads(collect_raw(store, layout)[name])
    assert not (_keys(document) & CV_KEYS)
