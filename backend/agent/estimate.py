"""Cost estimator for Costly tool calls (design C12, D12; Req 4.5, 5.5, 9.6).

Builds the same ``PlannedJob`` list the orchestrator would submit, using the orchestrator's own
plan builders, and prices each job with ``guards.reserve``. The result is an upper bound for
compute only, labelled with the price table's source and version. It is never a bill.

The price table is loaded from ``settings.price_table_path`` regardless of ``enforce_budget``.
When the table is missing or unverified, a price is unknown, a plan cannot be built, or the tool
has no estimator, ``usd`` is ``None`` and ``basis`` says why ("no estimate available").

Estimator keys (``ToolFlags.estimator``):

- ``processing``: one Processing job on the experiment's processing instance (EDA, derived
  profiling, validation, holdout evaluation, ablation, scoring).
- ``training``: preparation + N training jobs + evaluation. N is ``len(args.models)`` or
  ``args.model_count`` when given, else ``max_models_per_run``. ``args.evaluation_mode == "cv"``
  with optional ``args.cv_folds`` scales the training runtime as the orchestrator does.
- ``free_form``: one Processing job on ``args.compute`` (a ``ComputeSpec``) when given, else the
  default processing instance.

Optional argument overrides read by every key: ``processing_instance_type`` and
``training_instance_type``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from backend.agent.tools import ToolContext
from backend.config import Settings
from backend.orchestration.guards import GuardViolation, PlannedJob, reserve
from backend.orchestration.sagemaker import JobPlan, OrchestrationError, SageMakerOrchestrator
from backend.pricing import PriceTable
from ml_engine.contracts.config import SplitConfig
from ml_engine.contracts.cv import CrossValidationConfig, CvScheme, EvaluationMode
from ml_engine.contracts.execution import ComputeSpec, ExperimentCompute
from ml_engine.io import ExperimentLayout

NO_ESTIMATE = "no estimate available"
_EXPERIMENT_PLACEHOLDER = "estimate"
_TOKEN = "estimate"


@dataclass(frozen=True, slots=True)
class Estimate:
    usd: float | None
    basis: str
    price_source: str | None
    price_version: str | None


class CostEstimator:
    def __init__(self, settings: Settings, prices: PriceTable | None) -> None:
        self._settings = settings
        self._load_error: str | None = None
        if prices is None:
            try:
                prices = PriceTable.load(settings.price_table_path)
            except (OSError, ValueError) as error:
                self._load_error = f"price table not available: {type(error).__name__}"
        self._prices = prices
        # The planner is used only for its pure plan builders: no store or AWS client is touched.
        self._planner = (
            SageMakerOrchestrator(settings, store=None, sagemaker_client=None, prices=prices)
            if prices is not None
            else None
        )
        self._builders: dict[str, Callable[[BaseModel], list[JobPlan]]] = {
            "processing": self._processing,
            "training": self._training,
            "free_form": self._free_form,
        }

    def estimate(
        self,
        tool: str,
        args: BaseModel,
        ctx: ToolContext,  # noqa: ARG002 - design C12 signature; reserved for context lookups
        *,
        estimator: str | None = None,
    ) -> Estimate:
        """Price a tool call. ``estimator`` is the tool's ``ToolFlags.estimator`` key; without
        one there is no estimate (fail safe)."""
        builder = self._builders.get(estimator) if estimator else None
        if builder is None:
            return self._none(f"{NO_ESTIMATE}: {tool} has no estimator")
        if self._prices is None or self._planner is None:
            return self._none(f"{NO_ESTIMATE}: {self._load_error or 'price table not loaded'}")
        try:
            plans = builder(args)
            reservations = [
                reserve(
                    self._settings,
                    self._prices,
                    PlannedJob(plan.kind, plan.compute, on_demand_fallback=False),
                )
                for plan in plans
            ]
        except (GuardViolation, OrchestrationError) as error:
            return self._labelled(None, f"{NO_ESTIMATE}: {error.code}")
        except ValueError as error:  # invalid compute or split in the arguments
            return self._labelled(None, f"{NO_ESTIMATE}: {type(error).__name__}")
        total = round(sum(r.reserved_usd for r in reservations), 6)
        jobs = ", ".join(
            f"{p.stage} {p.compute.instance_type} x{p.compute.instance_count} "
            f"<= {p.compute.max_runtime_seconds}s"
            for p in plans
        )
        return self._labelled(total, f"upper bound for compute only ({len(plans)} jobs: {jobs})")

    # --- plan builders (delegate to the orchestrator's own) --------------------------------
    def _compute(self, args: BaseModel) -> ExperimentCompute:
        return ExperimentCompute(
            experiment_id=_EXPERIMENT_PLACEHOLDER,
            processing_instance_type=_arg(args, "processing_instance_type"),
            training_instance_type=_arg(args, "training_instance_type"),
        )

    def _processing(self, args: BaseModel) -> list[JobPlan]:
        return [self._processing_plan("processing", args)]

    def _free_form(self, args: BaseModel) -> list[JobPlan]:
        override = _arg(args, "compute")
        if isinstance(override, dict):
            override = ComputeSpec.model_validate(override)
        if not isinstance(override, ComputeSpec):
            override = None
        return [self._processing_plan("free_form", args, override=override)]

    def _training(self, args: BaseModel) -> list[JobPlan]:
        assert self._planner is not None
        compute = self._compute(args)
        models = _arg(args, "models")
        count = len(models) if models else _arg(args, "model_count")
        count = int(count) if count else self._settings.max_models_per_run
        split = self._split(args)
        trainings = [
            self._planner._training_plan(
                _EXPERIMENT_PLACEHOLDER, _TOKEN, f"model{i}", _layout(), compute, split=split
            )
            for i in range(count)
        ]
        return [
            self._processing_plan("preparation", args),
            *trainings,
            self._processing_plan("evaluation", args),
        ]

    def _processing_plan(
        self, stage: str, args: BaseModel, *, override: ComputeSpec | None = None
    ) -> JobPlan:
        assert self._planner is not None
        return self._planner._processing_plan(
            _EXPERIMENT_PLACEHOLDER, _TOKEN, stage, [], self._compute(args), override=override
        )

    def _split(self, args: BaseModel) -> SplitConfig | None:
        if str(_arg(args, "evaluation_mode") or "") != EvaluationMode.CV.value:
            return None
        folds = _arg(args, "cv_folds") or self._settings.cv_default_folds
        return SplitConfig(
            evaluation_mode=EvaluationMode.CV,
            cv=CrossValidationConfig(folds=int(folds), scheme=CvScheme.KFOLD),
        )

    # --- results ------------------------------------------------------------------------------
    def _labelled(self, usd: float | None, basis: str) -> Estimate:
        assert self._prices is not None
        return Estimate(usd, basis, self._prices.source, self._prices.version)

    def _none(self, basis: str) -> Estimate:
        if self._prices is None:
            return Estimate(None, basis, None, None)
        return self._labelled(None, basis)


def _arg(args: BaseModel, name: str) -> Any:
    return getattr(args, name, None)


def _layout() -> ExperimentLayout:
    # Only used to build job arguments, which do not affect the price.
    return ExperimentLayout(base="s3://estimate/estimate")
