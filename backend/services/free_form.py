"""Free-form experiments — the control-plane side (free-form-experiments design C5).

Mirrors ``ValidationService``: it guards, pins the approved script and the dataset under the run's
own prefix, writes the request, submits the side job through the orchestrator, and reads status.
The platform runs the exact approved bytes (content-hash bound) and labels results honestly; the
AI may draft and the user approves, but the service never lets the AI approve or submit (B12).
"""

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from backend.config import Settings
from backend.errors import ConflictError, NotFoundError, ValidationError
from backend.orchestration.guards import GuardViolation, check_free_form_instance
from backend.repositories import ExperimentRepository
from ml_engine.contracts.execution import ComputeSpec, ExecutionRecord
from ml_engine.contracts.free_form import (
    DATA_CONTEXT_QUESTIONS,
    ApprovedScript,
    DataContextAnswer,
    DataContextAnswers,
    DraftSession,
    FreeFormRequest,
    FreeFormResult,
    ScriptRevision,
    content_hash,
)
from ml_engine.contracts.model import ModelFailure
from ml_engine.contracts.requirements import RequirementsSpec
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    read_model,
    read_model_if_exists,
    write_model,
)

LOGGER = logging.getLogger("ml_factory.services.free_form")


class FreeFormService:
    """Approve a script, start a free-form run, and read its status."""

    def __init__(
        self,
        *,
        settings: Settings,
        repository: ExperimentRepository,
        store: ObjectStore,
        orchestrator,
        experiments,
        client=None,
    ) -> None:
        self._settings = settings
        self._repository = repository
        self._store = store
        self._orchestrator = orchestrator
        self._experiments = experiments
        self._client = client

    # --- data-context Q&A + AI drafting (Req 2, 14) -------------------------------------------
    def request_data_context(self, experiment_id: str, dataset_uri: str) -> dict[str, Any]:
        """The data-context questions the AI must have answered before drafting (Req 14.1).

        Saved answers for the same dataset are returned as defaults (Req 14.6). No raw rows.
        """
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        defaults = read_model_if_exists(
            self._store,
            layout.free_form_data_context_defaults(_dataset_key(dataset_uri)),
            DataContextAnswers,
        )
        return {
            "questions": [q.model_dump(mode="json") for q in DATA_CONTEXT_QUESTIONS],
            "defaults": defaults.model_dump(mode="json") if defaults else None,
        }

    def draft_script(
        self,
        experiment_id: str,
        *,
        dataset_uri: str,
        description: str,
        data_context: DataContextAnswers,
        draft_id: str | None = None,
        instruction: str | None = None,
    ) -> dict[str, Any]:
        """Produce a Script_Draft. Refuses until the data-context questions are answered (Req 14).

        Each revision is retained; the AI may draft and revise but never approves or submits (B12).
        Answers are saved as defaults for the dataset.
        """
        if not data_context.is_complete:
            raise ConflictError(
                "Answer (or explicitly skip) the data-context questions before drafting.",
                {"reason": "DATA_CONTEXT_REQUIRED"},
            )
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        session = self._load_or_new_draft(layout, experiment_id, dataset_uri, draft_id)
        session.data_context = data_context
        previous = session.latest_revision
        source, assumptions = self._generate_draft(description, data_context, previous, instruction)
        revision = ScriptRevision(
            revision=(previous.revision + 1) if previous else 1,
            content=source,
            content_hash=content_hash(source),
            authored_by="ai",
            instruction=instruction,
            assumptions=assumptions,
            created_at=datetime.now(UTC),
        )
        session.revisions.append(revision)
        session.updated_at = datetime.now(UTC)
        write_model(self._store, layout.free_form_draft(session.draft_id), session)
        # Save the answers as defaults for later runs on the same dataset (Req 14.6).
        write_model(
            self._store,
            layout.free_form_data_context_defaults(_dataset_key(dataset_uri)),
            data_context,
        )
        return {
            "draft_id": session.draft_id,
            "revision": revision.model_dump(mode="json"),
            "revision_count": len(session.revisions),
        }

    def draft_revision(
        self, experiment_id: str, draft_id: str, revision: int
    ) -> tuple[DraftSession, ScriptRevision]:
        """One stored Script_Draft revision and its drafting session. Read-only.

        Raises ``NotFoundError`` when the experiment has no such draft or the draft has no such
        revision.
        """
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        session = read_model_if_exists(self._store, layout.free_form_draft(draft_id), DraftSession)
        if session is None or session.experiment_id != experiment_id:
            raise NotFoundError(f"No free-form draft '{draft_id}' for this experiment.")
        found = next((r for r in session.revisions if r.revision == revision), None)
        if found is None:
            raise NotFoundError(f"Draft '{draft_id}' has no revision {revision}.")
        return session, found

    def _load_or_new_draft(
        self, layout: ExperimentLayout, experiment_id: str, dataset_uri: str, draft_id: str | None
    ) -> DraftSession:
        if draft_id is not None:
            existing = read_model_if_exists(
                self._store, layout.free_form_draft(draft_id), DraftSession
            )
            if existing is not None:
                return existing
        new_id = draft_id or f"draft-{uuid.uuid4().hex[:12]}"
        now = datetime.now(UTC)
        return DraftSession(
            draft_id=new_id,
            experiment_id=experiment_id,
            dataset_uri=dataset_uri,
            created_at=now,
            updated_at=now,
        )

    def _generate_draft(
        self,
        description: str,
        data_context: DataContextAnswers,
        previous: ScriptRevision | None,
        instruction: str | None,
    ) -> tuple[str, list[str]]:
        """Ask the LLM for a draft, or fall back to a template when no client is configured.

        The LLM (when present) sees the description + data-context answers only, never raw rows
        (platform invariant). The template encodes the Req-14.4/14.5 behaviour: it states which
        answer drove the split choice and which risks stay open for 'don't know'/skips.
        """
        assumptions = _draft_assumptions(data_context)
        if self._client is not None:
            try:
                text = self._client.draft_free_form_script(
                    description=description,
                    data_context=data_context.model_dump(mode="json"),
                    previous=previous.content if previous else None,
                    instruction=instruction,
                )
                if text:
                    return text, assumptions
            except Exception:  # fall back to the template; the user still reviews and approves
                LOGGER.exception("LLM drafting failed; returning a template draft")
        return _template_script(description, data_context), assumptions

    def start_run(
        self,
        experiment_id: str,
        *,
        script: str,
        dataset_uri: str,
        entry_function: str = "main",
        entry_module: str = "experiment",
        predict_entry_point: str | None = None,
        requirements: list[str] | None = None,
        instance_type: str | None = None,
        use_spot: bool = False,
        file_format: str | None = None,
        target_column: str | None = None,
        declared_feature_columns: list[str] | None = None,
        declared_external_urls: list[str] | None = None,
        data_context: DataContextAnswers | None = None,
        acknowledged_high_severity_leakage: bool = False,
        user: str | None = None,
    ) -> dict[str, Any]:
        """Approve the given script and submit one free-form run. Returns the run status at once.

        The user supplies the exact code to run; approval binds to its content hash (Req 2.4).
        """
        record = self._repository.get(experiment_id)
        layout = ExperimentLayout(base=record.artifact_prefix)

        compute = self._resolve_compute(instance_type, use_spot)
        requirements_spec = self._resolve_requirements(requirements or [])

        run_id = f"ff-{uuid.uuid4().hex[:12]}"
        dataset = self._experiments.validate_dataset(dataset_uri, file_format)
        dataset = self._experiments.pin_dataset(dataset, layout.free_form_input_dir(run_id))

        # Pin the exact approved bytes; the hash the run binds to is computed from them (Req 2.4).
        self._store.write_bytes(layout.free_form_script(run_id), script.encode("utf-8"))
        approved = ApprovedScript(
            content_hash=content_hash(script),
            revision=1,
            entry_module=entry_module,
            entry_function=entry_function,
            predict_entry_point=predict_entry_point,
            approved_by=user,
            approved_at=datetime.now(UTC),
            acknowledged_high_severity_leakage=acknowledged_high_severity_leakage,
        )
        request = FreeFormRequest(
            run_id=run_id,
            experiment_id=experiment_id,
            approved_script=approved,
            dataset_uri=dataset.uri,
            file_format=dataset.file_format,
            requirements=requirements_spec,
            compute=compute,
            target_column=target_column,
            declared_feature_columns=list(declared_feature_columns or []),
            declared_external_urls=list(declared_external_urls or []),
            data_context=data_context or DataContextAnswers(),
            requested_at=datetime.now(UTC),
            requested_by=user,
        )
        write_model(self._store, layout.free_form_request(run_id), request)

        start = getattr(self._orchestrator, "start_free_form", None)
        if not callable(start):
            raise ConflictError("This deployment's orchestrator cannot run free-form experiments.")
        handle = start(record, run_id)
        request.execution_id = handle.execution_id
        write_model(self._store, layout.free_form_request(run_id), request)
        return self.run_status(experiment_id, run_id)

    def _resolve_compute(self, instance_type: str | None, use_spot: bool) -> ComputeSpec:
        instance = instance_type or self._settings.processing_instance_type
        try:
            check_free_form_instance(self._settings, instance)
        except GuardViolation as error:
            raise ValidationError(error.message, {"reason": error.code}) from error
        runtime = self._settings.free_form_max_runtime_seconds
        if use_spot:
            return ComputeSpec(
                instance_type=instance,
                volume_size_gb=self._settings.volume_size_gb,
                max_runtime_seconds=runtime,
                use_spot=True,
                max_wait_seconds=max(runtime, self._settings.spot_max_wait_seconds),
            )
        return ComputeSpec(
            instance_type=instance,
            volume_size_gb=self._settings.volume_size_gb,
            max_runtime_seconds=runtime,
        )

    def _resolve_requirements(self, lines: list[str]) -> RequirementsSpec:
        cleaned = [entry.strip() for entry in lines if entry and entry.strip()]
        if not cleaned:
            return RequirementsSpec()
        try:
            return RequirementsSpec(packages=cleaned)
        except ValueError as error:
            raise ValidationError(
                f"A requirement is malformed: {error}", {"reason": "REQUIREMENT_MALFORMED"}
            ) from error

    def run_status(self, experiment_id: str, run_id: str | None = None) -> dict[str, Any]:
        """One free-form run (latest unless an id is given) and where it stands."""
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        requests = self._requests(layout)
        if not requests:
            raise NotFoundError("No free-form run exists for this experiment.")
        if run_id is None:
            request = requests[0]
        else:
            request = next((r for r in requests if r.run_id == run_id), None)
            if request is None:
                raise NotFoundError(f"No free-form run '{run_id}' for this experiment.")

        result: dict[str, Any] = {
            "run_id": request.run_id,
            "dataset_uri": request.dataset_uri,
            "content_hash": request.approved_script.content_hash,
            "execution_id": request.execution_id,
            "requested_at": request.requested_at.isoformat(),
            "status": "running",
            "other_run_ids": [r.run_id for r in requests if r.run_id != request.run_id],
        }
        # The leakage screen (Req 13) is attached to the run and shown; it is advisory.
        from ml_engine.contracts.leakage import LeakageReport

        leakage = read_model_if_exists(
            self._store, layout.free_form_leakage(request.run_id), LeakageReport
        )
        if leakage is not None:
            result["leakage"] = leakage.model_dump(mode="json")

        report = read_model_if_exists(
            self._store, layout.free_form_result(request.run_id), FreeFormResult
        )
        if report is not None:
            result["status"] = report.status
            result["result"] = report.model_dump(mode="json")
            return result
        failure = read_model_if_exists(
            self._store, layout.free_form_failure(request.run_id), ModelFailure
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

    def list_runs(self, experiment_id: str) -> list[dict[str, Any]]:
        """Every free-form run of the experiment, newest first."""
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        return [self.run_status(experiment_id, r.run_id) for r in self._requests(layout)]

    def compare(self, experiment_id: str, *, model_versions) -> dict[str, Any]:
        """List free-form and managed runs together with their tier + metric labels (Req 8).

        A like-for-like comparison uses engine metrics only (managed model scores, or SDK-produced
        free-form metrics); self-reported numbers are listed but flagged as not comparable. Nothing
        is auto-selected (Req 8.3).
        """
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        managed = [
            {
                "tier": "managed",
                "name": v.model_name,
                "version_id": v.version_id,
                "primary_metric": v.primary_metric,
                "primary_score": v.primary_score,
                "metric_label": "engine",
                "comparable": True,
            }
            for v in model_versions.list_versions(experiment_id)
            if v.source != "free_form"
        ]
        free_form_entries: list[dict[str, Any]] = []
        for request in self._requests(layout):
            result = read_model_if_exists(
                self._store, layout.free_form_result(request.run_id), FreeFormResult
            )
            if result is None:
                continue
            for metric in result.metrics:
                free_form_entries.append(
                    {
                        "tier": "free_form",
                        "name": f"{request.run_id}:{metric.name}",
                        "run_id": request.run_id,
                        "metric": metric.name,
                        "value": metric.value,
                        "metric_label": metric.label.value,
                        # Only engine metrics are comparable with managed scores (Req 8.2).
                        "comparable": metric.label.value == "engine",
                    }
                )
        return {
            "experiment_id": experiment_id,
            "entries": managed + free_form_entries,
            "note": (
                "Like-for-like comparison uses engine metrics only; self-reported free-form "
                "numbers are shown but not comparable."
            ),
        }

    def validate_run_model(
        self,
        experiment_id: str,
        run_id: str,
        *,
        dataset_uri: str,
        file_format: str | None = None,
        validation,
        user: str | None = None,
    ) -> dict[str, Any]:
        """Validate a free-form run's Model_Version through the standard validation flow (Req 7).

        Requires the run to have produced a Predict_Entry_Point model; otherwise validation is
        unavailable for that run (Req 7.4). Engine-computed metrics (Req 7.2).
        """
        layout = ExperimentLayout(base=self._repository.get(experiment_id).artifact_prefix)
        result = read_model_if_exists(self._store, layout.free_form_result(run_id), FreeFormResult)
        if result is None or not result.produced_model or result.model_version_name is None:
            raise ConflictError(
                "This free-form run exposed no predict entry point, so independent validation is "
                "unavailable for it (its results stay self-reported).",
                {"reason": "VALIDATION_UNAVAILABLE", "run_id": run_id},
            )
        version_id = f"free_form:{run_id}:{result.model_version_name}"
        return validation.start_validation(
            experiment_id,
            dataset_uri=dataset_uri,
            version_ids=[version_id],
            file_format=file_format,
            user=user,
        )

    def _requests(self, layout: ExperimentLayout) -> list[FreeFormRequest]:
        requests = [
            read_model(self._store, uri, FreeFormRequest)
            for uri in self._store.list_uris(layout.free_form_prefix)
            if uri.endswith("/request.json")
        ]
        return sorted(requests, key=lambda r: r.requested_at, reverse=True)


def _dataset_key(dataset_uri: str) -> str:
    """A filesystem/S3-safe key for a dataset, used to store its data-context defaults."""
    import hashlib

    return hashlib.sha256(dataset_uri.encode("utf-8")).hexdigest()[:16]


def _answer_for(data_context: DataContextAnswers, question_id: str) -> DataContextAnswer | None:
    return next((a for a in data_context.answers if a.question_id == question_id), None)


def _draft_assumptions(data_context: DataContextAnswers) -> list[str]:
    """State which answer drove each split/feature choice; flag open risks (Req 14.4/5)."""
    assumptions: list[str] = []
    entity = _answer_for(data_context, "entity_repeats")
    if entity and entity.status == "answered" and entity.answer:
        assumptions.append(
            f"Entities repeat ({entity.answer}); the draft uses a group split to keep an entity "
            "in one fold."
        )
    time_ordered = _answer_for(data_context, "time_ordered")
    if time_ordered and time_ordered.status == "answered" and time_ordered.answer:
        assumptions.append(
            f"Rows are time-ordered ({time_ordered.answer}); the draft uses a time-ordered split."
        )
    post_outcome = _answer_for(data_context, "post_outcome_columns")
    if post_outcome and post_outcome.status == "answered" and post_outcome.answer:
        assumptions.append(
            f"Columns recorded after the outcome were named ({post_outcome.answer}); the draft "
            "excludes or flags them."
        )
    # Open risks for don't-know / skipped answers on split- or feature-affecting questions.
    for qid in ("entity_repeats", "time_ordered", "post_outcome_columns", "label_defining_columns"):
        answer = _answer_for(data_context, qid)
        if answer is None or answer.status in {"dont_know", "skipped"}:
            assumptions.append(
                f"Open risk: '{qid}' was not answered, so the related leakage/split risk remains."
            )
    return assumptions


def _template_script(description: str, data_context: DataContextAnswers) -> str:
    """A minimal, SDK-using Script_Draft the user edits or asks the AI to revise (Req 4.5)."""
    time_ordered = _answer_for(data_context, "time_ordered")
    use_temporal = bool(time_ordered and time_ordered.status == "answered")
    split_note = (
        "# Rows are time-ordered: use the SDK temporal split (train strictly before eval)."
        if use_temporal
        else "# Use the SDK stratified/group split as appropriate."
    )
    return (
        f'"""Free-form experiment draft.\n\n{description}\n\n'
        "This draft uses the Platform_SDK so its metrics are engine-computed and comparable.\n"
        'Edit it or ask the AI to revise it; nothing runs until you approve the exact code."""\n\n'
        "from ml_engine.sdk import metrics, results, splits\n\n\n"
        "def main(context):\n"
        f"    {split_note}\n"
        "    frame = context.data\n"
        "    # TODO: fit your model and compute predictions, then report engine metrics:\n"
        "    # results.write_metrics(context, {'rmse': metrics.rmse(y_true, y_pred)})\n"
        "    context.report_metric('rows', float(len(frame)))\n"
    )
