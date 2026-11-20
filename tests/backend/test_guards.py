"""Pre-submission spend guards (design Property 11): refused before AWS is ever called."""

import pytest

from backend.orchestration.guards import (
    GuardViolation,
    PlannedJob,
    check_budget,
    check_concurrency,
    check_instance,
    check_spot_policy,
    reserve,
)
from backend.pricing import PriceTable
from ml_engine.contracts.execution import ComputeSpec, JobKind


def _spot(instance="ml.m5.large", runtime=3600) -> ComputeSpec:
    return ComputeSpec(
        instance_type=instance, max_runtime_seconds=runtime, use_spot=True, max_wait_seconds=runtime
    )


def _on_demand(instance="ml.m5.large", runtime=3600) -> ComputeSpec:
    return ComputeSpec(instance_type=instance, max_runtime_seconds=runtime)


@pytest.mark.parametrize(
    "instance", ["ml.m5.4xlarge", "ml.c5.large", "ml.g5.xlarge", "ml.p3.2xlarge"]
)
def test_instances_outside_the_ceiling_are_refused(settings, instance):
    with pytest.raises(GuardViolation) as error:
        check_instance(settings, instance)
    assert error.value.code == "INSTANCE_NOT_ALLOWED"


def test_allowed_instances_pass(settings):
    for instance in settings.allowed_instance_types:
        check_instance(settings, instance)


def test_on_demand_training_is_refused_without_opt_in(settings):
    with pytest.raises(GuardViolation) as error:
        check_spot_policy(
            settings, PlannedJob(JobKind.TRAINING, _on_demand()), allow_on_demand=False
        )
    assert error.value.code == "ON_DEMAND_NOT_ALLOWED"


def test_an_on_demand_fallback_needs_the_experiment_opt_in(settings):
    fallback = PlannedJob(JobKind.TRAINING, _on_demand(), on_demand_fallback=True)
    with pytest.raises(GuardViolation):
        check_spot_policy(settings, fallback, allow_on_demand=False)
    check_spot_policy(settings, fallback, allow_on_demand=True)


def test_processing_jobs_are_on_demand_by_design(settings):
    """Processing jobs cannot use Managed Spot; the Spot rule applies to training only."""
    check_spot_policy(settings, PlannedJob(JobKind.PROCESSING, _on_demand()), allow_on_demand=False)


def test_spot_training_passes(settings):
    check_spot_policy(settings, PlannedJob(JobKind.TRAINING, _spot()), allow_on_demand=False)


def test_an_unverified_price_table_blocks_submission(settings, price_table_path):
    from backend.config import DEFAULT_PRICE_TABLE

    with pytest.raises(GuardViolation) as error:
        reserve(
            settings, PriceTable.load(DEFAULT_PRICE_TABLE), PlannedJob(JobKind.TRAINING, _spot())
        )
    assert error.value.code == "PRICE_TABLE_UNVERIFIED"


def test_an_unknown_price_blocks_submission(settings, price_table_path):
    settings.aws_region = "eu-central-1"
    with pytest.raises(GuardViolation) as error:
        reserve(settings, PriceTable.load(price_table_path), PlannedJob(JobKind.TRAINING, _spot()))
    assert error.value.code == "PRICE_UNKNOWN"


def test_a_reservation_is_the_full_runtime_at_on_demand_rate(settings, price_table_path):
    prices = PriceTable.load(price_table_path)
    reservation = reserve(settings, prices, PlannedJob(JobKind.TRAINING, _spot(runtime=7200)))
    rate = prices.rate("us-east-1", JobKind.TRAINING, "ml.m5.large")
    assert reservation.reserved_usd == pytest.approx(rate.hourly_usd * 2)
    assert reservation.max_billable_seconds == 7200


def test_budget_ceilings_are_enforced(settings):
    settings.budget_per_experiment_usd = 1.0
    settings.budget_per_month_usd = 10.0
    check_budget(
        settings, new_reservations_usd=0.5, experiment_reserved_usd=0.5, month_reserved_usd=0
    )
    with pytest.raises(GuardViolation) as error:
        check_budget(
            settings, new_reservations_usd=0.51, experiment_reserved_usd=0.5, month_reserved_usd=0
        )
    assert error.value.code == "EXPERIMENT_BUDGET_EXCEEDED"
    with pytest.raises(GuardViolation) as error:
        check_budget(
            settings, new_reservations_usd=0.5, experiment_reserved_usd=0, month_reserved_usd=9.6
        )
    assert error.value.code == "MONTHLY_BUDGET_EXCEEDED"


def test_parallelism_is_capped(settings):
    check_concurrency(settings, settings.max_parallel_training_jobs)
    with pytest.raises(GuardViolation) as error:
        check_concurrency(settings, settings.max_parallel_training_jobs + 1)
    assert error.value.code == "TOO_MANY_PARALLEL_JOBS"


def test_a_spot_job_must_wait_at_least_its_runtime():
    with pytest.raises(ValueError, match="max_wait_seconds"):
        ComputeSpec(
            instance_type="ml.m5.large",
            max_runtime_seconds=3600,
            use_spot=True,
            max_wait_seconds=60,
        )
    with pytest.raises(ValueError, match="Spot jobs only"):
        ComputeSpec(instance_type="ml.m5.large", max_runtime_seconds=3600, max_wait_seconds=3600)
