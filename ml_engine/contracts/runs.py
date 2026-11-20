"""Archived training runs.

Retraining an experiment moves the previous run's outputs (model artifacts, comparison, summary,
post-training leakage, reports) to ``runs/<run_id>/`` before any new job is submitted, so a model
that fails in the new run can never be reported with the old run's metrics. The manifest below
records what was moved and the headline result, so runs can be listed and compared.
"""

from datetime import datetime

from pydantic import Field

from ml_engine.contracts.common import StrictModel
from ml_engine.contracts.cv import EvaluationMode, is_unset

RUN_SCHEMA_VERSION = "1.0"


class ArchivedRun(StrictModel):
    schema_version: str = RUN_SCHEMA_VERSION
    run_id: str
    experiment_id: str
    archived_at: datetime
    archived_by: str | None = None
    execution_id: str | None = None
    primary_metric: str | None = None
    best_model: str | None = None
    best_score: float | None = None
    models: list[str] = Field(default_factory=list)
    selected_features: list[str] = Field(default_factory=list)
    split_strategy: str | None = None
    dataset_uri: str | None = None
    dataset_etag: str | None = None
    archived_uris: int = 0
    # How the archived run measured its models (omitted when unknown, e.g. older manifests).
    evaluation_mode: EvaluationMode | None = Field(default=None, exclude_if=is_unset)
    cv_folds: int | None = Field(default=None, exclude_if=is_unset)
    test_fraction: float | None = Field(default=None, exclude_if=is_unset)
