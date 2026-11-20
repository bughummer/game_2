"""Pre-submission spend guards (requirements 4.4, 4.5, 4.7; design Property 11).

Pure functions: given what is about to be submitted and what has already been reserved, decide
whether it may go. They never call AWS. A refusal is a ``GuardViolation`` raised *before* any
SageMaker API call, so no job exists to clean up.

The rule is fail-closed: an instance with no known price, or a price table that has not been
verified, blocks submission — an unknown cost is not a zero cost.
"""

from dataclasses import dataclass

from backend.config import GPU_FAMILY_PREFIXES, Settings
from backend.errors import ValidationError
from backend.pricing import PriceTable, Rate, reserved_cost_usd
from ml_engine.contracts.execution import ComputeSpec, JobKind


class GuardViolation(RuntimeError):
    """A submission was refused before reaching AWS. ``code`` is stable and user-safe."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class PlannedJob:
    kind: JobKind
    compute: ComputeSpec
    on_demand_fallback: bool = False


@dataclass(frozen=True, slots=True)
class Reservation:
    rate: Rate
    reserved_usd: float
    max_billable_seconds: int


def check_instance(settings: Settings, instance_type: str) -> None:
    if instance_type.startswith(GPU_FAMILY_PREFIXES):
        raise GuardViolation(
            "INSTANCE_NOT_ALLOWED", f"{instance_type} is an accelerated instance; CPU only."
        )
    if instance_type not in settings.allowed_instance_types:
        raise GuardViolation(
            "INSTANCE_NOT_ALLOWED",
            f"{instance_type} is not allowed. "
            f"Allowed: {', '.join(settings.allowed_instance_types)}.",
        )


def check_free_form_instance(settings: Settings, instance_type: str) -> None:
    """A free-form run may use any allow-listed instance, including GPU when opted in (Req 1.4).

    Unlike the managed ``check_instance`` (CPU-only), a GPU family is permitted when the instance
    is in ``allowed_free_form_instance_types``; everything else must still be in the managed
    allow-list. An unknown instance is refused before any AWS call.
    """
    allowed = set(settings.allowed_instance_types) | set(settings.allowed_free_form_instance_types)
    if instance_type in settings.allowed_free_form_instance_types:
        return  # explicitly opted in (may be GPU)
    if instance_type.startswith(GPU_FAMILY_PREFIXES):
        raise GuardViolation(
            "INSTANCE_NOT_ALLOWED",
            f"{instance_type} is an accelerated instance; add it to "
            "allowed_free_form_instance_types to use it for a free-form run.",
        )
    if instance_type not in allowed:
        raise GuardViolation(
            "INSTANCE_NOT_ALLOWED",
            f"{instance_type} is not allowed. Allowed: {', '.join(sorted(allowed))}.",
        )


def check_spot_policy(settings: Settings, job: PlannedJob, *, allow_on_demand: bool) -> None:
    """Training is Spot unless the experiment explicitly opted into an On-Demand fallback."""
    if job.kind is not JobKind.TRAINING:
        return
    if settings.spot_enabled and not job.compute.use_spot and not job.on_demand_fallback:
        raise GuardViolation(
            "ON_DEMAND_NOT_ALLOWED", "Training must use Managed Spot; On-Demand was requested."
        )
    if job.on_demand_fallback and not allow_on_demand:
        raise GuardViolation(
            "ON_DEMAND_NOT_ALLOWED",
            "On-Demand fallback was not enabled for this experiment; refusing to escalate cost.",
        )


def reserve(settings: Settings, prices: PriceTable, job: PlannedJob) -> Reservation:
    """The upper-bound cost of one job, from the price table. Unknown price → refused."""
    if settings.require_verified_prices and not prices.verified:
        raise GuardViolation(
            "PRICE_TABLE_UNVERIFIED",
            f"The price table ({prices.version}) is not marked verified, so spend cannot be "
            "bounded. Verify infra/pricing/sagemaker_prices.json before submitting jobs.",
        )
    rate = prices.rate(settings.aws_region, job.kind, job.compute.instance_type)
    if rate is None:
        raise GuardViolation(
            "PRICE_UNKNOWN",
            f"No price for {job.compute.instance_type} ({job.kind.value}) in "
            f"{settings.aws_region}; refusing to submit a job whose cost cannot be bounded.",
        )
    seconds = job.compute.max_runtime_seconds
    return Reservation(
        rate=rate,
        reserved_usd=reserved_cost_usd(
            rate, instance_count=job.compute.instance_count, max_billable_seconds=seconds
        ),
        max_billable_seconds=seconds,
    )


def check_budget(
    settings: Settings,
    *,
    new_reservations_usd: float,
    experiment_reserved_usd: float,
    month_reserved_usd: float,
) -> None:
    if experiment_reserved_usd + new_reservations_usd > settings.budget_per_experiment_usd:
        raise GuardViolation(
            "EXPERIMENT_BUDGET_EXCEEDED",
            f"This would reserve ${new_reservations_usd:.2f} on top of "
            f"${experiment_reserved_usd:.2f}, over the per-experiment ceiling of "
            f"${settings.budget_per_experiment_usd:.2f}.",
        )
    if month_reserved_usd + new_reservations_usd > settings.budget_per_month_usd:
        raise GuardViolation(
            "MONTHLY_BUDGET_EXCEEDED",
            f"This would reserve ${new_reservations_usd:.2f} on top of "
            f"${month_reserved_usd:.2f} this month, over the monthly ceiling of "
            f"${settings.budget_per_month_usd:.2f}.",
        )


def check_concurrency_ceiling(settings: Settings, training_jobs: int) -> None:
    """A run may train at most ``max_models_per_run`` models (submitted in waves)."""
    if training_jobs > settings.max_models_per_run:
        raise GuardViolation(
            "TOO_MANY_MODELS",
            f"{training_jobs} models requested; one run may train at most "
            f"{settings.max_models_per_run}.",
        )


def check_concurrency(settings: Settings, training_jobs: int) -> None:
    """One experiment may not fan out wider than the configured parallelism."""
    if training_jobs > settings.max_parallel_training_jobs:
        raise GuardViolation(
            "TOO_MANY_PARALLEL_JOBS",
            f"{training_jobs} training jobs requested; the limit is "
            f"{settings.max_parallel_training_jobs}.",
        )


def resolve_cv_folds(
    *,
    requested_folds: int | None,
    evaluation_mode: str | None,
    model_count: int,
    default_folds: int,
    max_folds: int,
    max_fits: int,
) -> int:
    """Validate a CV fold request and return the resolved K (design C1, Req 1.3-1.5).

    Pure: no IO, no AWS. Raises ``ValidationError`` with a stable ``reason`` in ``details`` so
    nothing is archived or submitted when the request is rejected. The caller runs this before
    ``archive_previous_run``. Property 13 exercises this function directly.

    - ``requested_folds`` set without CV_Mode -> ``CV_FOLDS_WITHOUT_CV_MODE``.
    - resolved K outside ``[2, max_folds]`` -> ``CV_FOLDS_OUT_OF_RANGE``.
    - ``model_count x (K + 1) > max_fits`` -> ``CV_FIT_LIMIT_EXCEEDED``.
    """
    is_cv = evaluation_mode == "cv"
    if requested_folds is not None and not is_cv:
        raise ValidationError(
            "cv_folds was given without evaluation_mode='cv'; set the evaluation mode to 'cv' "
            "to request cross-validation.",
            {"reason": "CV_FOLDS_WITHOUT_CV_MODE", "cv_folds": requested_folds},
        )
    if not is_cv:
        # Single_Split_Mode: no K to resolve and no fit ceiling to apply.
        return default_folds
    folds = default_folds if requested_folds is None else requested_folds
    if folds < 2 or folds > max_folds:
        raise ValidationError(
            f"K={folds} is outside the allowed range of 2 to {max_folds} folds.",
            {
                "reason": "CV_FOLDS_OUT_OF_RANGE",
                "requested": folds,
                "allowed_min": 2,
                "allowed_max": max_folds,
            },
        )
    requested_fits = model_count * (folds + 1)
    if requested_fits > max_fits:
        raise ValidationError(
            f"{model_count} model(s) x (K + 1) = {requested_fits} fold fits exceed the ceiling "
            f"of {max_fits} fits per experiment. Lower K or train fewer models.",
            {
                "reason": "CV_FIT_LIMIT_EXCEEDED",
                "requested_fits": requested_fits,
                "allowed_fits": max_fits,
            },
        )
    return folds
