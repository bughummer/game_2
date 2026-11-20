"""Validate model versions on an uploaded labeled dataset (control-ui-api design C5).

The control-plane side of the validation side job: validate a chosen Model_Version, or every
version, on a dataset in an approved location. One SageMaker job loops the versions; the labels
(when present) are kept for metrics and threshold analysis. Holdout evaluation is the special case
"validate the current run's models", so the two paths stay one.
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from backend.config import Settings
from backend.errors import ArtifactNotReadyError, ConflictError, NotFoundError
from backend.repositories import ACTIVE_TRAINING_STATUSES, ExperimentRepository
from ml_engine.contracts.execution import ExecutionRecord
from ml_engine.contracts.model import ModelFailure
from ml_engine.contracts.validation import ValidationReport, ValidationRequest
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    read_model,
    read_model_if_exists,
    write_model,
)

LOGGER = logging.getLogger("ml_factory.services.validation")


class ValidationService:
    """Starts validations and reads their status."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: ExperimentRepository,
        store: ObjectStore,
        orchestrator,
        experiments,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store
        self._orchestrator = orchestrator
        self._experiments = experiments

    def start_validation(
        self,
        experiment_id: str,
        *,
        dataset_uri: str,
        version_ids: list[str] | None = None,
        file_format: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        """Validate the given Model_Versions (empty = all) on ``dataset_uri``. Returns at once."""
        record = self._repository.get(experiment_id)
        if record.status in ACTIVE_TRAINING_STATUSES:
            raise ConflictError("Training is running; validate once it has finished.")
        layout = ExperimentLayout(base=record.artifact_prefix)
        if not self._store.exists(layout.comparison):
            raise ConflictError("Train the experiment before validating its models.")
        dataset = self._experiments.validate_dataset(dataset_uri, file_format)
        validation_id = f"val-{uuid.uuid4().hex[:12]}"
        dataset = self._experiments.pin_dataset(dataset, layout.validation_input_dir(validation_id))
        request = ValidationRequest(
            validation_id=validation_id,
            experiment_id=experiment_id,
            dataset_uri=dataset.uri,
            file_format=dataset.file_format,
            version_ids=list(version_ids or []),
            requested_at=datetime.now(UTC),
            requested_by=user,
        )
        write_model(self._store, layout.validation_request(validation_id), request)
        start = getattr(self._orchestrator, "start_validation", None)
        if not callable(start):
            raise ConflictError("This deployment's orchestrator cannot run validation.")
        handle = start(record, validation_id)
        request.execution_id = handle.execution_id
        write_model(self._store, layout.validation_request(validation_id), request)
        return self.validation(experiment_id, validation_id)

    def validation(self, experiment_id: str, validation_id: str | None = None) -> dict[str, Any]:
        """One validation (latest unless an id is given) and where it stands."""
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        requests = sorted(
            (
                read_model(self._store, uri, ValidationRequest)
                for uri in self._store.list_uris(layout.validations_prefix)
                if uri.endswith("/request.json")
            ),
            key=lambda r: r.requested_at,
            reverse=True,
        )
        if not requests:
            raise ArtifactNotReadyError("No validation has been requested for this experiment.")
        if validation_id is None:
            request = requests[0]
        else:
            request = next((r for r in requests if r.validation_id == validation_id), None)
            if request is None:
                raise NotFoundError(f"No validation '{validation_id}' for this experiment.")

        result: dict[str, Any] = {
            "validation_id": request.validation_id,
            "dataset_uri": request.dataset_uri,
            "version_ids": request.version_ids,
            "execution_id": request.execution_id,
            "requested_at": request.requested_at.isoformat(),
            "status": "running",
            "other_validation_ids": [
                r.validation_id for r in requests if r.validation_id != request.validation_id
            ],
        }
        report = read_model_if_exists(
            self._store, layout.validation_report(request.validation_id), ValidationReport
        )
        if report is not None:
            result["status"] = "completed"
            result["report"] = report.model_dump(mode="json")
            return result
        failure = read_model_if_exists(
            self._store, layout.validation_failure(request.validation_id), ModelFailure
        )
        execution = (
            read_model_if_exists(
                self._store, layout.execution(request.execution_id), ExecutionRecord
            )
            if request.execution_id
            else None
        )
        if failure is not None:
            result.update(status="failed", failure_message=failure.message)
        elif execution is not None and execution.status.value in {"FAILED", "STOPPED"}:
            result.update(status="failed", failure_message=execution.failure_message)
        return result
