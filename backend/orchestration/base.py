"""The orchestration boundary.

The control plane starts workflows and reads status; it never runs ML work itself. The shipped
implementation is ``SageMakerOrchestrator`` (boto3; Step Functions later); the tests substitute
``tests/support/inline_orchestrator.py``, which runs the same stages in-process.

Ported verbatim from old_repo apart from this docstring.
"""

from dataclasses import dataclass
from typing import Protocol

from ml_engine.contracts.experiment import ExperimentRecord


@dataclass(frozen=True, slots=True)
class ExecutionHandle:
    """Identifies a started workflow execution."""

    execution_id: str
    started: bool = True
    detail: str | None = None


class ExperimentOrchestrator(Protocol):
    """Starts the EDA and training workflows for an experiment."""

    name: str

    def start_eda(self, record: ExperimentRecord) -> ExecutionHandle: ...

    def start_training(self, record: ExperimentRecord) -> ExecutionHandle: ...

    def start_derived_profiling(self, record: ExperimentRecord) -> ExecutionHandle:
        """Profile approved derived features off the control plane. Does not change status."""
        ...

    def start_holdout_evaluation(
        self, record: ExperimentRecord, evaluation_id: str
    ) -> ExecutionHandle:
        """Score trained models on a separate dataset. Does not change status."""
        ...

    def start_ablation(
        self, record: ExperimentRecord, *, model_name: str, features: list[str]
    ) -> ExecutionHandle:
        """Retrain one model without each feature on identical folds. Does not change status."""
        ...

    def start_scoring(self, record: ExperimentRecord, scoring_id: str) -> ExecutionHandle:
        """Apply a trained model to a new file (batch scoring). Does not change status."""
        ...

    def start_free_form(self, record: ExperimentRecord, run_id: str) -> ExecutionHandle:
        """Run a user-approved free-form script as a side job. Does not change status."""
        ...

    def stop(self, execution_id: str) -> object:
        """Stop every running job of an execution. Nothing further is submitted."""
        ...

    def describe(self, execution_id: str) -> dict[str, str]: ...
