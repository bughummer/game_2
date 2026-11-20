"""The Model_Version store (control-ui-api design C4).

A read model over storage that already exists: the current run's ``models/<name>/metadata.json``
and every archived ``runs/<run_id>/models/<name>/metadata.json``. Each trained model becomes a
listable, independently-loadable ``ModelVersion``; nothing is copied or overwritten to produce the
store, so a retrain simply *adds* the archived run's versions (Req 7.2) and a pre-feature experiment
lists its models with no re-training (Req 7.6).
"""

import logging

from backend.config import Settings
from backend.errors import NotFoundError
from backend.repositories import ExperimentRepository
from ml_engine.contracts.model import ModelMetadata
from ml_engine.contracts.model_version import CURRENT_RUN, ArtifactRef, ModelVersion
from ml_engine.contracts.runs import ArchivedRun
from ml_engine.io import ExperimentLayout, ObjectStore, read_model, read_model_if_exists

LOGGER = logging.getLogger("ml_factory.services.model_versions")


class ModelVersionService:
    """Lists and resolves the Model_Versions of an experiment."""

    def __init__(
        self, *, settings: Settings, repository: ExperimentRepository, store: ObjectStore
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store

    def list_versions(self, experiment_id: str) -> list[ModelVersion]:
        """Every Model_Version of the experiment, the current run first then archived runs.

        The current run's models come from ``models/<name>/metadata.json``; archived runs from
        ``runs/<run_id>/models/<name>/metadata.json``. Newest run first; within a run, by model
        name.
        """
        record = self._repository.get(experiment_id)
        base = ExperimentLayout(base=record.artifact_prefix)
        versions: list[ModelVersion] = []

        # Current run (the live experiment prefix).
        for metadata in self._metadata_in(base):
            versions.append(self._version(experiment_id, metadata, base, run_id=None))

        # Archived runs, newest first (ArchivedRun.archived_at).
        for run in self._archived_runs(base):
            run_layout = ExperimentLayout(base=base.run_archive(run.run_id))
            for metadata in self._metadata_in(run_layout):
                versions.append(
                    self._version(
                        experiment_id,
                        metadata,
                        run_layout,
                        run_id=run.run_id,
                        source="retrain",
                    )
                )

        # Free-form runs that exposed a Predict_Entry_Point (ensembles-and-learners/free-form):
        # each is an independently-loadable, validatable Model_Version under free_form/<id>/.
        versions.extend(self._free_form_versions(experiment_id, base))
        return versions

    def _free_form_versions(self, experiment_id: str, base: ExperimentLayout) -> list[ModelVersion]:
        """Model_Versions produced by free-form runs that registered a model (free-form Req 6.3)."""
        versions: list[ModelVersion] = []
        prefix = base.free_form_prefix
        run_ids = {
            uri[len(prefix) :].split("/", 1)[0]
            for uri in self._store.list_uris(prefix)
            if "/" in uri[len(prefix) :]
        }
        for run_id in sorted(run_ids):
            models_prefix = base.free_form_models_prefix(run_id)
            for uri in self._store.list_uris(models_prefix):
                if not uri.endswith("/metadata.json"):
                    continue
                metadata = read_model_if_exists(self._store, uri, ModelMetadata)
                if metadata is None:
                    continue
                versions.append(self._free_form_version(experiment_id, metadata, base, run_id))
        return versions

    def _free_form_version(
        self,
        experiment_id: str,
        metadata: ModelMetadata,
        base: ExperimentLayout,
        run_id: str,
    ) -> ModelVersion:
        name = metadata.model_name
        artifact_ref = ArtifactRef(
            model_uri=base.free_form_model_artifact(run_id, name),
            metadata_uri=base.free_form_model_metadata(run_id, name),
            preprocessor_uri=None,
            preprocessing_strategy=None,
        )
        return ModelVersion(
            version_id=f"free_form:{run_id}:{name}",
            experiment_id=experiment_id,
            model_name=name,
            display_name=metadata.display_name,
            library=metadata.library,
            problem_type=metadata.problem_type,
            hyperparameters=dict(metadata.hyperparameters),
            primary_metric=metadata.primary_metric,
            primary_score=metadata.primary_score,
            metrics=metadata.metrics,
            created_at=metadata.training_completed_at or metadata.training_started_at,
            source="free_form",
            run_id=run_id,
            artifact_ref=artifact_ref,
        )

    def get_version(self, experiment_id: str, version_id: str) -> ModelVersion:
        """One Model_Version by its stable id, or ``NotFoundError``."""
        for version in self.list_versions(experiment_id):
            if version.version_id == version_id:
                return version
        raise NotFoundError(
            f"No model version '{version_id}' in this experiment.",
            {"experiment_id": experiment_id},
        )

    # --- helpers ------------------------------------------------------------------------------
    def _metadata_in(self, layout: ExperimentLayout) -> list[ModelMetadata]:
        """Every ``models/<name>/metadata.json`` directly under ``layout``'s models prefix."""
        prefix = layout.models_prefix + "/"
        found: list[ModelMetadata] = []
        for uri in self._store.list_uris(prefix):
            if not uri.endswith("/metadata.json"):
                continue
            # Only one level deep: models/<name>/metadata.json, not models/<name>/cv/....
            if uri.count("/") != prefix.count("/") + 1:
                continue
            metadata = read_model_if_exists(self._store, uri, ModelMetadata)
            if metadata is not None:
                found.append(metadata)
        return sorted(found, key=lambda m: m.model_name)

    def _archived_runs(self, base: ExperimentLayout) -> list[ArchivedRun]:
        runs = [
            read_model(self._store, uri, ArchivedRun)
            for uri in self._store.list_uris(base.runs_prefix)
            if uri.endswith("/run.json") and uri.count("/") == base.runs_prefix.count("/") + 1
        ]
        return sorted(runs, key=lambda r: r.archived_at, reverse=True)

    def _version(
        self,
        experiment_id: str,
        metadata: ModelMetadata,
        layout: ExperimentLayout,
        *,
        run_id: str | None,
        source: str = "training",
    ) -> ModelVersion:
        prefix = run_id or CURRENT_RUN
        strategy = metadata.preprocessing.strategy if metadata.preprocessing else None
        # Resolve the artifact URIs against THIS layout (the current prefix or the archive), so
        # an archived version points into runs/<run_id>/ rather than the moved-away live paths.
        preprocessor_uri = layout.preprocessor(strategy) if strategy else None
        artifact_ref = ArtifactRef(
            model_uri=layout.model_artifact(metadata.model_name),
            metadata_uri=layout.model_metadata(metadata.model_name),
            preprocessor_uri=preprocessor_uri,
            preprocessing_strategy=strategy,
        )
        return ModelVersion(
            version_id=f"{prefix}:{metadata.model_name}",
            experiment_id=experiment_id,
            model_name=metadata.model_name,
            display_name=metadata.display_name,
            library=metadata.library,
            problem_type=metadata.problem_type,
            hyperparameters=dict(metadata.hyperparameters),
            primary_metric=metadata.primary_metric,
            primary_score=metadata.primary_score,
            metrics=metadata.metrics,
            created_at=metadata.training_completed_at or metadata.training_started_at,
            source=source,
            run_id=run_id,
            artifact_ref=artifact_ref,
        )
