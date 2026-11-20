"""Test datasets written into the (in-memory) store under ``s3://`` URIs."""

import pandas as pd

from ml_engine.io import ObjectStore, join_uri

ARTIFACT_ROOT = "s3://mlf-test-artifacts"
APPROVED_PREFIX = "s3://mlf-test-approved/curated"


def write_dataset(store: ObjectStore, name: str, frame: pd.DataFrame) -> str:
    """A CSV sitting in the approved location, referenced by URI rather than uploaded."""
    uri = join_uri(APPROVED_PREFIX, name)
    store.write_bytes(uri, frame.to_csv(index=False).encode("utf-8"))
    return uri
