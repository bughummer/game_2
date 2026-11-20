"""The stage-image resolver (stock-image migration, design C1, C10).

Offline: the AWS SDK's ``image_uris.retrieve`` is stubbed via ``sys.modules`` so the lookup path
runs with no real AWS and no sagemaker SDK installed.
"""

import sys
import types

import pytest

from backend.aws.images import (
    STAGE_IMAGE_FRAMEWORK,
    ImageResolutionError,
    resolve_stage_image,
)
from backend.config import Settings
from tests.support.datasets import ARTIFACT_ROOT

ROLE = "arn:aws:iam::123456789012:role/mlf-sagemaker-job"
EXPLICIT = "123456789012.dkr.ecr.us-east-1.amazonaws.com/custom@sha256:" + "a" * 64


def _settings(**overrides) -> Settings:
    return Settings(
        artifact_root=ARTIFACT_ROOT,
        sagemaker_execution_role_arn=ROLE,
        **overrides,
    )


def _stub_sagemaker(monkeypatch, retrieve):
    """Install a fake ``sagemaker.image_uris`` module with the given ``retrieve`` callable."""
    module = types.ModuleType("sagemaker")
    image_uris = types.ModuleType("sagemaker.image_uris")
    image_uris.retrieve = retrieve
    module.image_uris = image_uris
    monkeypatch.setitem(sys.modules, "sagemaker", module)
    monkeypatch.setitem(sys.modules, "sagemaker.image_uris", image_uris)


def test_an_explicit_uri_passes_through_without_any_lookup(monkeypatch):
    def boom(**_kwargs):
        raise AssertionError("retrieve must not be called when an override is set")

    _stub_sagemaker(monkeypatch, boom)
    resolved = resolve_stage_image(_settings(stage_image_uri=EXPLICIT))
    assert resolved.uri == EXPLICIT
    assert resolved.source == "config"


def test_unset_calls_the_sdk_lookup_with_the_region_and_version(monkeypatch):
    calls = {}

    def retrieve(**kwargs):
        calls.update(kwargs)
        return "246618743249.dkr.ecr.eu-west-1.amazonaws.com/sagemaker-xgboost:1.7-1"

    _stub_sagemaker(monkeypatch, retrieve)
    resolved = resolve_stage_image(_settings(aws_region="eu-west-1", xgboost_image_version="1.7-1"))
    assert resolved.source == "aws_lookup"
    assert resolved.uri.endswith("sagemaker-xgboost:1.7-1")
    assert calls == {
        "framework": STAGE_IMAGE_FRAMEWORK,
        "region": "eu-west-1",
        "version": "1.7-1",
    }


def test_a_lookup_failure_raises_a_structured_error_naming_the_region(monkeypatch):
    def retrieve(**_kwargs):
        raise ValueError("unknown region")

    _stub_sagemaker(monkeypatch, retrieve)
    with pytest.raises(ImageResolutionError) as excinfo:
        resolve_stage_image(_settings(aws_region="ap-south-2"))
    assert excinfo.value.code == "IMAGE_LOOKUP_FAILED"
    assert excinfo.value.region == "ap-south-2"


def test_an_empty_lookup_result_is_rejected(monkeypatch):
    _stub_sagemaker(monkeypatch, lambda **_kwargs: "")
    with pytest.raises(ImageResolutionError) as excinfo:
        resolve_stage_image(_settings())
    assert excinfo.value.code == "IMAGE_LOOKUP_EMPTY"


def test_a_missing_sdk_raises_a_structured_error(monkeypatch):
    # Make `import sagemaker` fail even if it happens to be installed.
    monkeypatch.setitem(sys.modules, "sagemaker", None)
    with pytest.raises(ImageResolutionError) as excinfo:
        resolve_stage_image(_settings())
    assert excinfo.value.code == "SAGEMAKER_SDK_MISSING"


def test_problems_no_longer_requires_a_stage_image(monkeypatch):
    # A valid config with no stage image must not be flagged incomplete on that account.
    settings = _settings()
    assert settings.stage_image_uri == ""
    assert not any("STAGE_IMAGE_URI" in p for p in settings.problems())
