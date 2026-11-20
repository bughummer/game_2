"""Control-plane settings and the static price table."""

import json

import pytest
from pydantic import ValidationError

from backend.config import DEFAULT_PRICE_TABLE, Settings
from backend.pricing import PriceTable, PriceTableError, reserved_cost_usd
from ml_engine.contracts.execution import JobKind
from ml_engine.io import InvalidS3UriError


def test_a_complete_configuration_has_no_problems(settings):
    assert settings.problems() == []


def test_budget_enforcement_is_off_by_default(settings):
    """MLOps owns cost; ML Factory does not price or cap jobs unless told to."""
    assert settings.enforce_budget is False


def test_a_missing_price_table_only_matters_when_enforcing(settings, tmp_path):
    settings.price_table_path = tmp_path / "absent.json"
    assert settings.problems() == []  # not enforcing: no price table needed
    settings.enforce_budget = True
    assert any("price table not found" in p for p in settings.problems())


def test_missing_essentials_are_all_reported(price_table_path):
    problems = Settings(price_table_path=price_table_path).problems()
    assert any("ARTIFACT_ROOT" in p for p in problems)
    assert any("EXECUTION_ROLE_ARN" in p for p in problems)


def test_a_missing_stage_image_is_no_longer_a_problem(price_table_path):
    """The stock-image migration makes stage_image_uri optional; the resolver supplies it."""
    from tests.support.datasets import ARTIFACT_ROOT

    settings = Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn="arn:aws:iam::123456789012:role/mlf-sagemaker-job",
        price_table_path=price_table_path,
    )
    assert settings.stage_image_uri == ""
    assert not any("STAGE_IMAGE_URI" in p for p in settings.problems())


def test_a_local_artifact_root_is_refused():
    with pytest.raises((ValidationError, InvalidS3UriError)):
        Settings(artifact_root="/data/artifacts")


def test_gpu_instances_cannot_be_allowed():
    with pytest.raises(ValidationError, match="CPU only"):
        Settings(allowed_instance_types=["ml.m5.large", "ml.g5.xlarge"])


def test_default_instance_must_be_inside_the_ceiling(settings):
    settings.training_instance_type = "ml.m5.4xlarge"
    assert any("outside allowed_instance_types" in p for p in settings.problems())


def test_spot_wait_shorter_than_runtime_is_a_problem(settings):
    settings.spot_max_wait_seconds = 600
    assert any("SPOT_MAX_WAIT" in p for p in settings.problems())


def test_settings_come_from_the_environment(monkeypatch, price_table_path):
    monkeypatch.setenv("ML_FACTORY_ARTIFACT_ROOT", "s3://from-env-bucket/")
    monkeypatch.setenv("ML_FACTORY_BUDGET_PER_EXPERIMENT_USD", "12.5")
    loaded = Settings(price_table_path=price_table_path)
    assert loaded.artifact_root == "s3://from-env-bucket"
    assert loaded.budget_per_experiment_usd == 12.5


# --- cross-validation settings ------------------------------------------------------------


def test_cv_settings_have_the_documented_defaults(settings):
    assert settings.cv_default_folds == 5
    assert settings.cv_max_folds == 10
    assert settings.cv_max_fits == 60
    assert settings.cv_training_max_runtime_seconds is None
    assert settings.default_test_fraction == 0.2
    assert settings.cv_instability_std == 0.05
    assert settings.cv_instability_cv == 0.10
    assert settings.cv_outlier_multiplier == 3.0
    assert settings.cv_outlier_min_gap == 0.02


def test_cv_settings_come_from_the_environment(monkeypatch, price_table_path):
    monkeypatch.setenv("ML_FACTORY_CV_DEFAULT_FOLDS", "4")
    monkeypatch.setenv("ML_FACTORY_CV_MAX_FOLDS", "8")
    monkeypatch.setenv("ML_FACTORY_CV_MAX_FITS", "40")
    monkeypatch.setenv("ML_FACTORY_CV_TRAINING_MAX_RUNTIME_SECONDS", "7200")
    monkeypatch.setenv("ML_FACTORY_DEFAULT_TEST_FRACTION", "0.25")
    loaded = Settings(artifact_root="s3://mlf-env-bucket/", price_table_path=price_table_path)
    assert loaded.cv_default_folds == 4
    assert loaded.cv_max_folds == 8
    assert loaded.cv_max_fits == 40
    assert loaded.cv_training_max_runtime_seconds == 7200
    assert loaded.default_test_fraction == 0.25


def test_default_folds_above_max_folds_is_refused(price_table_path):
    with pytest.raises(ValidationError, match="cv_default_folds"):
        Settings(cv_default_folds=12, cv_max_folds=10, price_table_path=price_table_path)


def test_default_folds_below_two_is_refused(price_table_path):
    with pytest.raises(ValidationError):
        Settings(cv_default_folds=1, price_table_path=price_table_path)


def test_max_fits_below_three_is_refused(price_table_path):
    with pytest.raises(ValidationError, match="cv_max_fits"):
        Settings(cv_max_fits=2, price_table_path=price_table_path)


def test_a_zero_cv_runtime_ceiling_is_refused(price_table_path):
    with pytest.raises(ValidationError):
        Settings(cv_training_max_runtime_seconds=0, price_table_path=price_table_path)


# --- price table --------------------------------------------------------------------------


def test_the_shipped_price_table_loads_and_says_it_is_unverified():
    """Until a person checks the rates against AWS, the table must say so (fail closed)."""
    table = PriceTable.load(DEFAULT_PRICE_TABLE)
    assert table.verified is False
    assert table.source.startswith("https://aws.amazon.com/")
    for instance in ("ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge"):
        assert table.rate("us-east-1", JobKind.TRAINING, instance) is not None


def test_an_unknown_instance_or_region_has_no_price(price_table_path):
    table = PriceTable.load(price_table_path)
    assert table.rate("us-east-1", JobKind.TRAINING, "ml.m5.24xlarge") is None
    assert table.rate("eu-west-1", JobKind.TRAINING, "ml.m5.large") is None


def test_every_rate_carries_its_provenance(price_table_path):
    rate = PriceTable.load(price_table_path).rate("us-east-1", JobKind.PROCESSING, "ml.m5.large")
    assert rate.source and rate.version == "test" and rate.retrieved_on


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.pop("regions"),
        lambda raw: raw["regions"]["us-east-1"].update({"inference": {"ml.m5.large": 1}}),
        lambda raw: raw["regions"]["us-east-1"]["training"].update({"ml.m5.large": 0}),
        lambda raw: raw["regions"]["us-east-1"]["training"].update({"ml.m5.large": "cheap"}),
    ],
)
def test_a_malformed_price_table_is_refused(tmp_path, mutate):
    raw = json.loads(DEFAULT_PRICE_TABLE.read_text(encoding="utf-8"))
    mutate(raw)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(PriceTableError):
        PriceTable.load(path)


def test_reserved_cost_is_rate_times_instances_times_hours(price_table_path):
    rate = PriceTable.load(price_table_path).rate("us-east-1", JobKind.TRAINING, "ml.m5.xlarge")
    assert reserved_cost_usd(rate, instance_count=2, max_billable_seconds=1800) == pytest.approx(
        rate.hourly_usd * 2 * 0.5
    )
