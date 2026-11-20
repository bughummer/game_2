"""An orchestrator that runs the real stages synchronously, in-process, for tests.

The same stage functions SageMaker runs (``jobs/*/main.py``), in the same order the
``SageMakerOrchestrator`` submits them, against whatever store the test provides. A failed model
becomes a ``failure.json`` and the run continues — per-model isolation, as on SageMaker.

It is a test double and deliberately lives outside the application packages. Adapted from
old_repo, where it delegated to the (not ported) ``workflow.runner``.
"""

from datetime import UTC, datetime

from backend.orchestration.base import ExecutionHandle
from jobs._common.experiment_state import ArtifactExperimentStateWriter
from jobs._common.runtime import JobError
from jobs.ablation.main import run_ablation
from jobs.derived_profiling.main import run_derived_profiling
from jobs.evaluation.main import run_evaluation
from jobs.holdout_evaluation.main import run_holdout_evaluation
from jobs.preprocessing.main import run_preparation
from jobs.profiling.main import run_profiling
from jobs.scoring.main import run_scoring
from jobs.training.main import run_training, write_model_failure
from jobs.validation.main import run_validation
from ml_engine.contracts.common import ExperimentStatus
from ml_engine.contracts.config import ExperimentConfig
from ml_engine.contracts.experiment import ExperimentRecord, WorkflowState
from ml_engine.contracts.model import ModelFailure
from ml_engine.io import ExperimentLayout, ObjectStore, read_model, write_model


class InlineOrchestrator:
    name = "inline"

    def __init__(self, store: ObjectStore) -> None:
        self._store = store
        self.failures: list[str] = []
        self.stopped: list[str] = []

    def start_eda(self, record: ExperimentRecord) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        self._mark(layout, record.experiment_id, ExperimentStatus.EDA_RUNNING, "profiling")
        try:
            run_profiling(
                self._store,
                layout,
                experiment_id=record.experiment_id,
                dataset_uri=record.dataset.uri,
                target_column=record.target_column,
                file_format=record.dataset.file_format,
            )
        except Exception as error:
            return self._failed(layout, record.experiment_id, "profiling", error)
        self._mark(layout, record.experiment_id, ExperimentStatus.EDA_COMPLETED, "eda_completed")
        self._mark(layout, record.experiment_id, ExperimentStatus.FEATURE_REVIEW, "feature_review")
        return ExecutionHandle(execution_id=f"inline-eda-{record.experiment_id}")

    def start_training(self, record: ExperimentRecord) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        experiment_id = record.experiment_id
        self._mark(layout, experiment_id, ExperimentStatus.PREPARING, "preparation")
        try:
            run_preparation(self._store, layout, experiment_id=experiment_id)
        except Exception as error:
            return self._failed(layout, experiment_id, "preparation", error)

        config = read_model(self._store, layout.experiment_config, ExperimentConfig)
        self._mark(layout, experiment_id, ExperimentStatus.TRAINING, "training")
        for spec in config.enabled_models_specs():
            try:
                run_training(self._store, layout, experiment_id=experiment_id, model_name=spec.name)
            except Exception as error:
                write_model_failure(
                    self._store,
                    layout,
                    experiment_id=experiment_id,
                    model_name=spec.name,
                    code=error.code if isinstance(error, JobError) else "MODEL_TRAINING_FAILED",
                    message=str(error),
                    details=error.details if isinstance(error, JobError) else None,
                )

        self._mark(layout, experiment_id, ExperimentStatus.EVALUATING, "evaluation")
        try:
            run_evaluation(
                self._store,
                layout,
                experiment_id=experiment_id,
                state_writer=ArtifactExperimentStateWriter(self._store, layout),
            )
        except Exception as error:
            return self._failed(layout, experiment_id, "evaluation", error)
        return ExecutionHandle(execution_id=f"inline-training-{experiment_id}")

    def start_derived_profiling(self, record: ExperimentRecord) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        try:
            run_derived_profiling(self._store, layout, experiment_id=record.experiment_id)
        except Exception as error:  # a side job: never changes the experiment's status
            self.failures.append(f"derived_profiling: {error}")
            return ExecutionHandle(execution_id="inline-derived", detail=str(error))
        return ExecutionHandle(execution_id=f"inline-derived-{record.experiment_id}")

    def start_holdout_evaluation(
        self, record: ExperimentRecord, evaluation_id: str
    ) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        try:
            run_holdout_evaluation(
                self._store,
                layout,
                experiment_id=record.experiment_id,
                evaluation_id=evaluation_id,
            )
        except Exception as error:  # a side job: record the failure like the CLI would
            self.failures.append(f"holdout_evaluation: {error}")
            write_model(
                self._store,
                layout.evaluation_failure(evaluation_id),
                ModelFailure(
                    experiment_id=record.experiment_id,
                    model_name="experiment",
                    error_code=error.code if isinstance(error, JobError) else "STAGE_FAILED",
                    message=error.message if isinstance(error, JobError) else str(error),
                    failed_at=datetime.now(UTC),
                ),
            )
        return ExecutionHandle(execution_id=f"inline-holdout-{evaluation_id}")

    def start_ablation(
        self, record: ExperimentRecord, *, model_name: str, features: list[str]
    ) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        try:
            run_ablation(
                self._store,
                layout,
                experiment_id=record.experiment_id,
                model_name=model_name,
                features=features,
            )
        except Exception as error:
            self.failures.append(f"ablation: {error}")
            self._side_failure(layout.ablation_failure, record.experiment_id, error)
        return ExecutionHandle(execution_id=f"inline-ablation-{record.experiment_id}")

    def start_scoring(self, record: ExperimentRecord, scoring_id: str) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        try:
            run_scoring(
                self._store, layout, experiment_id=record.experiment_id, scoring_id=scoring_id
            )
        except Exception as error:
            self.failures.append(f"scoring: {error}")
            self._side_failure(layout.scoring_failure(scoring_id), record.experiment_id, error)
        return ExecutionHandle(execution_id=f"inline-scoring-{scoring_id}")

    def start_validation(self, record: ExperimentRecord, validation_id: str) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        try:
            run_validation(
                self._store,
                layout,
                experiment_id=record.experiment_id,
                validation_id=validation_id,
            )
        except Exception as error:
            self.failures.append(f"validation: {error}")
            self._side_failure(
                layout.validation_failure(validation_id), record.experiment_id, error
            )
        return ExecutionHandle(execution_id=f"inline-validation-{validation_id}")

    def start_free_form(self, record: ExperimentRecord, run_id: str) -> ExecutionHandle:
        layout = ExperimentLayout(base=record.artifact_prefix)
        try:
            from jobs.free_form.main import run_free_form

            run_free_form(self._store, layout, experiment_id=record.experiment_id, run_id=run_id)
        except Exception as error:
            self.failures.append(f"free_form: {error}")
            self._side_failure(layout.free_form_failure(run_id), record.experiment_id, error)
        return ExecutionHandle(execution_id=f"inline-free-form-{run_id}")

    def stop(self, execution_id: str) -> None:
        self.stopped.append(execution_id)  # everything inline has already finished

    def _side_failure(self, uri: str, experiment_id: str, error: Exception) -> None:
        write_model(
            self._store,
            uri,
            ModelFailure(
                experiment_id=experiment_id,
                model_name="experiment",
                error_code=error.code if isinstance(error, JobError) else "STAGE_FAILED",
                message=error.message if isinstance(error, JobError) else str(error),
                failed_at=datetime.now(UTC),
            ),
        )

    def describe(self, execution_id: str) -> dict[str, str]:
        return {"execution_id": execution_id, "orchestrator": self.name}

    def _mark(self, layout, experiment_id, status, stage) -> None:
        write_model(
            self._store,
            layout.workflow_state,
            WorkflowState(
                experiment_id=experiment_id,
                updated_at=datetime.now(UTC),
                status=status,
                current_stage=stage,
            ),
        )

    def _failed(self, layout, experiment_id, stage, error) -> ExecutionHandle:
        self.failures.append(f"{stage}: {error}")
        write_model(
            self._store,
            layout.workflow_state,
            WorkflowState(
                experiment_id=experiment_id,
                updated_at=datetime.now(UTC),
                status=ExperimentStatus.FAILED,
                current_stage=stage,
                failure_code=error.code if isinstance(error, JobError) else "STAGE_FAILED",
                failure_message=str(error),
            ),
        )
        return ExecutionHandle(execution_id=f"inline-{stage}", detail=str(error))
