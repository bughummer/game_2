"""Agent settings: defaults, range checks in ``problems()`` and the model-id fallback.

The expected ranges below are copied from the design's settings table (C3), not imported from
``backend.config``, so a drift between the two is caught here.
"""

from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from backend.config import Settings
from tests.support.datasets import ARTIFACT_ROOT

ROLE = "arn:aws:iam::123456789012:role/mlf-sagemaker-job"
BASE = {"artifact_root": ARTIFACT_ROOT, "sagemaker_execution_role_arn": ROLE}

#: (field, default, low, high, is_int) from design C3. Inclusive ranges.
RANGED = (
    ("agent_max_output_tokens", 2048, 1, 8192, True),
    ("agent_converse_retries", 3, 0, 5, True),
    ("agent_converse_backoff_seconds", 1.0, 0.0, 30.0, False),
    ("agent_converse_read_timeout_seconds", 120, 10, 600, True),
    ("agent_approval_ttl_seconds", 900, 60, 86_400, True),
    ("agent_tool_result_max_bytes", 8192, 1_024, 1_048_576, True),
    ("agent_untrusted_string_max_chars", 256, 1, 100_000, True),
    ("agent_context_budget_tokens", 100_000, 4_000, 1_000_000, True),
    ("agent_cli_poll_seconds", 1.0, 0.5, 60, False),
    ("agent_web_poll_seconds", 1.0, 0.5, 60, False),
    ("agent_watch_interval_seconds", 15, 1, 60, True),
    ("agent_lease_seconds", 300, 30, 3600, True),
    ("agent_max_concurrent_sessions", 4, 1, 64, True),
)
POSITIVE_INTS = (
    ("agent_limit_max_converse_turns", 40),
    ("agent_limit_max_tool_calls", 60),
    ("agent_limit_max_costly_tool_calls", 10),
    ("agent_limit_max_active_seconds", 7200),
    ("agent_limit_max_repeated_identical_calls", 3),
)
SPEND = "agent_limit_max_estimated_spend_usd"

MODEL_IDS = (
    "",
    "anthropic.claude-3-5-sonnet-20240620-v1:0",
    "us.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "eu.anthropic.claude-3-5-sonnet-20240620-v1:0",
    "apac.anthropic.claude-3-5-sonnet-20240620-v1:0",
)
REGIONS = ("us-east-1", "us-west-2", "eu-west-1", "ap-southeast-2")
PROFILE_GROUPS = {"us": "us-", "eu": "eu-", "apac": "ap-"}


def _env(name: str) -> str:
    return f"ML_FACTORY_{name.upper()}"


def _num(value: float) -> str:
    """A bound as a reader writes it: 86400, 1048576, 0.5, 0."""
    return str(int(value)) if float(value).is_integer() else str(value)


def _agent_messages(found: list[str]) -> list[str]:
    return [p for p in found if "ML_FACTORY_AGENT_" in p]


def _non_agent_fields(s: Settings) -> dict:
    return {k: v for k, v in s.model_dump().items() if not k.startswith("agent_")}


BASELINE = Settings(**BASE)


def _ranged_value(low: float, high: float, is_int: bool) -> st.SearchStrategy:
    if is_int:
        span = int(high - low) + 10
        return st.one_of(
            st.integers(int(low), int(high)),
            st.integers(int(low) - span, int(high) + span),
            st.sampled_from([int(low), int(high), int(low) - 1, int(high) + 1]),
        )
    return st.one_of(
        st.floats(float(low), float(high)),
        st.floats(allow_nan=True, allow_infinity=True),
        st.sampled_from([float(low), float(high), low - 0.01, high + 0.01]),
    )


@st.composite
def _ranged_case(draw):
    name, _, low, high, is_int = draw(st.sampled_from(RANGED))
    return name, low, high, draw(_ranged_value(low, high, is_int))


def test_agent_defaults_match_the_design_and_add_no_problems():
    for name, default, *_ in RANGED:
        assert getattr(BASELINE, name) == default, name
    for name, default in POSITIVE_INTS:
        assert getattr(BASELINE, name) == default, name
    assert BASELINE.agent_limit_max_estimated_spend_usd == Decimal("10.0")
    assert BASELINE.agent_enabled is False
    assert BASELINE.agent_autopilot_allowed is True
    assert BASELINE.problems() == []


def test_a_wrong_type_is_rejected_at_load():
    with pytest.raises(ValidationError):
        Settings(**BASE, agent_approval_ttl_seconds="soon")


# Feature: ml-assistant, Property 47: Settings validation reports name and range
@settings(max_examples=100, deadline=None)
@given(case=_ranged_case())
def test_ranged_setting_reports_name_and_range_iff_out_of_range(case):
    """**Validates: Requirements 13.2, 13.3, 13.4, 13.6**"""
    name, low, high, value = case
    s = Settings(**BASE, **{name: value})
    messages = _agent_messages(s.problems())
    in_range = low <= value <= high  # False for NaN
    if in_range:
        assert messages == []
    else:
        assert len(messages) == 1, messages
        message = messages[0]
        assert message.startswith(_env(name) + " ")
        assert f"between {_num(low)} and {_num(high)}" in message
        assert f"(got {getattr(s, name)})" in message
    # Existing fields are untouched, and so is their part of problems().
    assert _non_agent_fields(s) == _non_agent_fields(BASELINE)
    assert [p for p in s.problems() if p not in messages] == BASELINE.problems()


# Feature: ml-assistant, Property 47: Settings validation reports name and range
@settings(max_examples=100, deadline=None)
@given(
    field=st.sampled_from([n for n, _ in POSITIVE_INTS]),
    value=st.one_of(st.integers(-1000, 1000), st.sampled_from([0, 1, -1])),
    spend=st.one_of(
        st.decimals(min_value=-100, max_value=100, allow_nan=False, allow_infinity=False),
        st.sampled_from([Decimal("0"), Decimal("0.01"), Decimal("-0.01")]),
    ),
)
def test_limit_settings_report_name_iff_not_positive(field, value, spend):
    """**Validates: Requirements 13.2, 13.3**"""
    s = Settings(**BASE, **{field: value, SPEND: spend})
    messages = _agent_messages(s.problems())
    expected = (value < 1) + (spend <= 0)
    assert len(messages) == expected, messages
    named = [m for m in messages if m.startswith(_env(field) + " ")]
    assert len(named) == (value < 1)
    if named:
        assert "positive integer" in named[0] and f"(got {value})" in named[0]
    spent = [m for m in messages if m.startswith(_env(SPEND) + " ")]
    assert len(spent) == (spend <= 0)
    if spent:
        assert "greater than 0" in spent[0]
    assert _non_agent_fields(s) == _non_agent_fields(BASELINE)


def _expected_mismatch(model_id: str, region: str) -> bool:
    prefix = model_id.split(".", 1)[0]
    group = PROFILE_GROUPS.get(prefix)
    return group is not None and not region.startswith(group)


# Feature: ml-assistant, Property 47: Settings validation reports name and range
@settings(max_examples=100, deadline=None)
@given(
    enabled=st.booleans(),
    agent_model=st.sampled_from(MODEL_IDS),
    bedrock_model=st.sampled_from(MODEL_IDS),
    region=st.sampled_from(REGIONS),
)
def test_model_id_fallback_and_enabled_checks(enabled, agent_model, bedrock_model, region):
    """**Validates: Requirements 13.4, 13.6**"""
    s = Settings(
        **BASE,
        agent_enabled=enabled,
        agent_model_id=agent_model,
        bedrock_model_id=bedrock_model,
        aws_region=region,
    )
    effective = agent_model if agent_model else bedrock_model
    assert s.effective_agent_model_id == effective

    baseline = Settings(**BASE, bedrock_model_id=bedrock_model, aws_region=region)
    added = [p for p in s.problems() if p not in baseline.problems()]
    if not enabled:
        assert added == []
        return
    missing = [p for p in added if _env("agent_model_id") in p]
    mismatch = [p for p in added if "inference profile" in p]
    assert len(missing) == (effective == "")
    if missing:
        assert _env("bedrock_model_id") in missing[0]
    assert len(mismatch) == _expected_mismatch(effective, region)
    if mismatch:
        assert effective in mismatch[0] and region in mismatch[0]
    assert len(added) == len(missing) + len(mismatch)
