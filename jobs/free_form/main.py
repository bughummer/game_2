"""Free-form experiment job (free-form-experiments design C3).

Runs a user-approved script as a SageMaker Processing job. The platform does not control what the
script does, so it does not claim the managed guarantees; it executes the exact approved bytes
(content-hash verified), captures everything around the run (pip freeze, image digest, logs), and
labels the results honestly. A metric is ``engine`` only when the script produced it through the
Platform_SDK results writer; otherwise it is ``self_reported``. If the script exposes a
Predict_Entry_Point, a Model_Version-shaped record is written so the model can be validated by the
platform itself.

The script contract: the approved module defines ``def main(context) -> None`` and fills
``context.report_metric(name, value)`` (self-reported) or uses the SDK's results writer (engine).
It may set ``context.model`` (a fitted estimator) and ``context.predict`` (a callable mapping a
feature table to predictions) to register a Model_Version. The script sees ``context.data`` (the
loaded dataframe) and ``context.output_dir`` and may import ``ml_engine.sdk``. It never receives raw
rows through the API or the LLM; the data arrives only here, inside the job.
"""

import argparse
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from jobs._common.runtime import JobError, base_parser, run_entrypoint
from ml_engine.contracts.free_form import (
    FreeFormMetric,
    FreeFormRequest,
    FreeFormResult,
    MetricLabel,
    content_hash,
)
from ml_engine.contracts.leakage import LeakageReport
from ml_engine.io import (
    ExperimentLayout,
    ObjectStore,
    load_dataset,
    read_model,
    save_joblib,
    write_model,
)

LOGGER = logging.getLogger("ml_factory.jobs.free_form")

#: The default model name a free-form Model_Version is saved under.
DEFAULT_MODEL_NAME = "free_form_model"


class FreeFormContext:
    """What an Approved_Script receives. The script fills metrics and may set a model/predict.

    A metric reported via :meth:`report_metric` is ``self_reported``; a metric written through the
    Platform_SDK results writer is ``engine`` (the SDK calls :meth:`report_engine_metric`).
    """

    def __init__(self, data: Any, output_dir: str) -> None:
        self.data = data
        self.output_dir = output_dir
        self._metrics: list[FreeFormMetric] = []
        #: Set by the script to register a Model_Version for independent validation.
        self.model: Any = None
        self.predict: Any = None
        #: A ProblemType the script declares for its model, used when validating it later.
        self.problem_type: Any = None
        self.naive_baseline: dict[str, float] = {}
        self.external_urls: list[str] = []

    def report_metric(self, name: str, value: float) -> None:
        """Record a self-reported metric (the script computed it)."""
        self._metrics.append(
            FreeFormMetric(name=name, value=float(value), label=MetricLabel.SELF_REPORTED)
        )

    def report_engine_metric(self, name: str, value: float) -> None:
        """Record an engine metric (produced through the Platform_SDK). Called by the SDK."""
        self._metrics.append(
            FreeFormMetric(name=name, value=float(value), label=MetricLabel.ENGINE)
        )

    @property
    def metrics(self) -> list[FreeFormMetric]:
        return list(self._metrics)


def run_free_form(
    store: ObjectStore, layout: ExperimentLayout, *, experiment_id: str, run_id: str
) -> FreeFormResult:
    """Execute one Approved_Script and write its labelled result."""
    request = read_model(store, layout.free_form_request(run_id), FreeFormRequest)
    approved = request.approved_script

    source = _load_approved_source(store, layout, run_id, approved.content_hash)
    LOGGER.info("Running free-form script for %s (run %s)", experiment_id, run_id)
    frame = load_dataset(store, request.dataset_uri, request.file_format).frame

    # Leakage screen on the run's inputs (Req 13): deterministic, advisory, attached to the run and
    # never auto-editing. Reuses the managed rules unchanged.
    _screen_leakage(store, layout, request, frame)

    context = FreeFormContext(data=frame, output_dir=layout.free_form_output_dir(run_id))
    context.external_urls = list(request.declared_external_urls)
    namespace = _execute_script(source, approved.entry_module, approved.entry_function, context)

    produced_model = False
    model_version_name: str | None = None
    if approved.predict_entry_point is not None:
        produced_model, model_version_name = _register_model_version(
            store, layout, request, context, namespace
        )

    result = FreeFormResult(
        run_id=run_id,
        experiment_id=experiment_id,
        status="completed",
        generated_at=datetime.now(UTC),
        metrics=context.metrics,
        produced_model=produced_model,
        model_version_name=model_version_name,
        content_hash=approved.content_hash,
        sdk_version=_sdk_version(),
        instance_type=request.compute.instance_type,
        used_spot=request.compute.use_spot,
        naive_baseline=context.naive_baseline,
        declared_external_urls=context.external_urls,
    )
    write_model(store, layout.free_form_result(run_id), result)
    LOGGER.info(
        "Free-form run %s completed: %d metric(s), model=%s",
        run_id,
        len(result.metrics),
        model_version_name or "none",
    )
    return result


def _screen_leakage(
    store: ObjectStore, layout: ExperimentLayout, request: FreeFormRequest, frame: Any
) -> None:
    """Run the managed leakage analysis on the run's inputs and attach the report (Req 13).

    Advisory only: this writes ``free_form/<id>/leakage.json`` and never edits the run. The target
    is the declared target, or a label-for-analysis column when one is given (Req 13.7); with no
    target the analyzer records that the target-dependent checks were skipped (Req 13.6).
    """
    from ml_engine.leakage import analyze_leakage
    from ml_engine.profiling.profiler import profile_dataset

    target = request.target_column or (
        request.label_columns_for_analysis[0] if request.label_columns_for_analysis else None
    )
    try:
        if target is None or target not in frame.columns:
            report = LeakageReport(
                experiment_id=request.experiment_id,
                generated_at=datetime.now(UTC),
                target_column=target or "",
                checks_skipped={
                    "target_dependent": "no target column was declared for this free-form run"
                },
            )
        else:
            eda = profile_dataset(
                frame,
                experiment_id=request.experiment_id,
                target_column=target,
                source_uri=request.dataset_uri,
                file_format=request.file_format,
            )
            problem_type = eda.target.inferred_problem_type
            if problem_type is None:
                report = LeakageReport(
                    experiment_id=request.experiment_id,
                    generated_at=datetime.now(UTC),
                    target_column=target,
                    checks_skipped={"all": "the target cannot support supervised learning"},
                )
            else:
                report = analyze_leakage(
                    frame,
                    experiment_id=request.experiment_id,
                    target_column=target,
                    problem_type=problem_type,
                    eda=eda,
                )
        write_model(store, layout.free_form_leakage(request.run_id), report)
    except Exception:  # the screen is advisory; its failure must not fail the run
        LOGGER.exception("Free-form leakage screen failed; continuing without it")


def _load_approved_source(
    store: ObjectStore, layout: ExperimentLayout, run_id: str, expected_hash: str
) -> str:
    """Load the pinned script bytes and refuse to run anything but the approved content."""
    uri = layout.free_form_script(run_id)
    if not store.exists(uri):
        raise JobError("SCRIPT_MISSING", "The approved script was not pinned for this run.")
    source = store.read_bytes(uri).decode("utf-8")
    if content_hash(source) != expected_hash:
        raise JobError(
            "SCRIPT_NOT_APPROVED",
            "The pinned script does not match the approved content hash; refusing to run.",
        )
    return source


def _execute_script(
    source: str, entry_module: str, entry_function: str, context: FreeFormContext
) -> dict[str, Any]:
    """Execute the script and invoke its entry function. A failure fails the run, no fallback.

    The script runs as a real module registered in ``sys.modules`` under ``entry_module`` so that
    classes/functions it defines can be pickled by reference — joblib needs an importable module
    for a saved estimator. The module registration is removed again afterwards.
    """
    import sys
    import types

    module = types.ModuleType(entry_module)
    namespace = module.__dict__
    namespace["__name__"] = entry_module
    sys.modules[entry_module] = module
    try:
        compiled = compile(source, f"<free_form:{entry_module}>", "exec")
        exec(compiled, namespace)
    except Exception as error:  # a failing script fails the run with the error (Req 1.6)
        sys.modules.pop(entry_module, None)
        raise JobError("SCRIPT_FAILED", f"The script raised during import: {error}") from error
    entry = namespace.get(entry_function)
    if not callable(entry):
        sys.modules.pop(entry_module, None)
        raise JobError(
            "SCRIPT_NO_ENTRY",
            f"The script defines no callable '{entry_function}' to run.",
        )
    try:
        entry(context)
    except Exception as error:
        sys.modules.pop(entry_module, None)
        raise JobError("SCRIPT_FAILED", f"The script raised during run: {error}") from error
    return namespace


def _register_model_version(
    store: ObjectStore,
    layout: ExperimentLayout,
    request: FreeFormRequest,
    context: FreeFormContext,
    namespace: dict[str, Any],
) -> tuple[bool, str | None]:
    """Save a Model_Version-shaped record when the run exposes a usable Predict_Entry_Point."""
    predict = context.predict or namespace.get(request.approved_script.predict_entry_point or "")
    if context.model is None or not callable(predict):
        LOGGER.warning("A predict entry point was declared but no model/predict was produced.")
        return False, None
    # Validate the entry point returns predictions of the expected shape on a tiny probe (Req 7.3).
    try:
        probe = context.data.head(min(5, len(context.data)))
        predictions = predict(probe)
        if len(predictions) != len(probe):
            raise ValueError("predict returned the wrong number of rows")
    except Exception as error:
        raise JobError(
            "PREDICT_ENTRY_POINT_INVALID",
            f"The declared predict entry point did not return valid predictions: {error}",
        ) from error

    name = DEFAULT_MODEL_NAME
    save_joblib(store, layout.free_form_model_artifact(request.run_id, name), context.model)
    from ml_engine.contracts.common import ModelRunStatus
    from ml_engine.contracts.model import ModelArtifacts, ModelMetadata

    metadata = ModelMetadata(
        experiment_id=request.experiment_id,
        model_name=name,
        display_name="Free-form model",
        status=ModelRunStatus.COMPLETED,
        problem_type=_problem_type(context),
        library="free_form",
        library_version=_sdk_version() or "unknown",
        artifacts=ModelArtifacts(
            model_uri=layout.free_form_model_artifact(request.run_id, name),
            metadata_uri=layout.free_form_model_metadata(request.run_id, name),
        ),
    )
    write_model(store, layout.free_form_model_metadata(request.run_id, name), metadata)
    return True, name


def _problem_type(context: FreeFormContext):
    """The problem type the script declares via ``context.problem_type``, else regression.

    A free-form script sets ``context.problem_type`` (a ``ProblemType``) when it wants its model
    validated as a classifier; absent, the model is treated as a regressor for validation.
    """
    from ml_engine.contracts.common import ProblemType

    declared = getattr(context, "problem_type", None)
    if isinstance(declared, ProblemType):
        return declared
    if isinstance(declared, str):
        try:
            return ProblemType(declared)
        except ValueError:
            return ProblemType.REGRESSION
    return ProblemType.REGRESSION


def _sdk_version() -> str | None:
    try:
        from ml_engine.sdk import SDK_VERSION

        return SDK_VERSION
    except Exception:  # the SDK lands in a later task; absence is not fatal
        return None


# --- entrypoint ---------------------------------------------------------------------------------


def _handler(args: argparse.Namespace, store: ObjectStore, layout: ExperimentLayout) -> None:
    run_free_form(store, layout, experiment_id=args.experiment_id, run_id=args.run_id)


def build_parser() -> argparse.ArgumentParser:
    parser = base_parser("ML Factory free-form job (run a user-approved script)")
    parser.prog = "free-form-job"
    parser.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return run_entrypoint(
        build_parser(),
        _handler,
        argv,
        failure_uri=lambda args, layout: layout.free_form_failure(args.run_id),
    )


if __name__ == "__main__":  # pragma: no cover - container entrypoint
    sys.exit(main())
