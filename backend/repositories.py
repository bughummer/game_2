"""Experiment records in S3 — the control plane's "database" (there is no database).

An experiment is three small documents under its own prefix, each with exactly one writer so no
component ever read-modify-writes a shared object:

* ``experiment.json``     the definition — written once, at creation
* ``state/control.json``  user-driven transitions (feature decision, training requested)
* ``state/workflow.json`` execution-driven transitions — written by the orchestrator/evaluation

``ExperimentRecord.compose`` merges the three into the view callers see. Listing is an S3 prefix
scan; ownership and search are applied over the composed records.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from backend.config import Settings
from backend.errors import NotFoundError
from ml_engine.contracts.common import ExperimentStatus
from ml_engine.contracts.experiment import (
    ControlPlaneState,
    ExperimentDefinition,
    ExperimentRecord,
    WorkflowState,
)
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    read_model_if_exists,
    write_model,
)

LOGGER = logging.getLogger("ml_factory.repository")


class ExperimentRepository:
    def __init__(self, *, settings: Settings, store: ObjectStore) -> None:
        self._settings = settings
        self._store = store

    @property
    def _root(self) -> str:
        return self._settings.artifact_root

    def create(self, definition: ExperimentDefinition) -> ExperimentRecord:
        layout = ExperimentLayout(base=definition.artifact_prefix)
        write_model(self._store, layout.definition, definition)
        write_model(
            self._store,
            layout.control_state,
            ControlPlaneState(
                experiment_id=definition.experiment_id,
                updated_at=definition.created_at,
                status_updated_at=definition.created_at,
            ),
        )
        return self.get(definition.experiment_id)

    def get(self, experiment_id: str) -> ExperimentRecord:
        layout = ExperimentLayout.for_experiment(self._root, experiment_id)
        definition = read_model_if_exists(self._store, layout.definition, ExperimentDefinition)
        if definition is None:
            raise NotFoundError(
                f"No experiment '{experiment_id}' exists.", {"experiment_id": experiment_id}
            )
        control = read_model_if_exists(self._store, layout.control_state, ControlPlaneState)
        workflow = read_model_if_exists(self._store, layout.workflow_state, WorkflowState)
        return ExperimentRecord.compose(definition, control, workflow)

    def update(self, experiment_id: str, **changes) -> ExperimentRecord:
        """Patch the control-plane document. Only the control plane's own fields are writable here.

        Setting a status stamps ``status_updated_at`` so a later stale workflow write cannot
        take the status back (the ownership rule in ``ExperimentRecord.compose``).
        """
        layout = ExperimentLayout.for_experiment(self._root, experiment_id)
        control = read_model_if_exists(self._store, layout.control_state, ControlPlaneState)
        if control is None:
            raise NotFoundError(f"No experiment '{experiment_id}' exists.")
        data = control.model_dump()
        data.update(changes)
        data["updated_at"] = datetime.now(UTC)
        if "status" in changes:
            data["status_updated_at"] = data["updated_at"]
        write_model(self._store, layout.control_state, ControlPlaneState(**data))
        return self.get(experiment_id)

    def delete(self, experiment_id: str) -> None:
        """Soft delete: the record drops out of listings, the artifacts stay auditable."""
        self.update(experiment_id, deleted=True)

    def list(
        self, *, limit: int, created_by: str | None = None, include_deleted: bool = False
    ) -> list[ExperimentRecord]:
        records: list[ExperimentRecord] = []
        for prefix in self._store.list_prefixes(ExperimentLayout.experiments_prefix(self._root)):
            experiment_id = prefix.rstrip("/").rsplit("/", 1)[-1]
            try:
                record = self.get(experiment_id)
            except NotFoundError:
                continue
            if record.deleted and not include_deleted:
                continue
            if created_by is not None and record.created_by != created_by:
                continue
            records.append(record)
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[:limit]

    def owner(self, experiment_id: str) -> str | None:
        return self.get(experiment_id).created_by


ACTIVE_TRAINING_STATUSES = frozenset(
    {ExperimentStatus.PREPARING, ExperimentStatus.TRAINING, ExperimentStatus.EVALUATING}
)
