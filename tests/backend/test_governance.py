"""Data governance for Bedrock and the region/profile check (requirements 15.5, 16.4)."""

import pytest

from backend.governance import GovernancePolicy, apply_governance, bedrock_region_mismatch


@pytest.mark.parametrize(
    ("model_id", "region", "ok"),
    [
        ("us.anthropic.claude-3-5-sonnet-20240620-v1:0", "us-east-1", True),
        ("us.anthropic.claude-3-5-sonnet-20240620-v1:0", "eu-west-1", False),
        ("eu.anthropic.claude-3-5-sonnet-20240620-v1:0", "eu-west-1", True),
        ("apac.anthropic.claude-3-5-sonnet-20240620-v1:0", "ap-southeast-2", True),
        ("apac.anthropic.claude-3-5-sonnet-20240620-v1:0", "us-east-1", False),
        # A plain model id (no region-group prefix) is never flagged.
        ("anthropic.claude-3-5-sonnet-20240620-v1:0", "eu-west-1", None),
    ],
)
def test_region_profile_mismatch(model_id, region, ok):
    result = bedrock_region_mismatch(model_id, region)
    if ok is None:
        assert result is None  # plain id, nothing to check
    elif ok:
        assert result is None
    else:
        assert result is not None and "requires an AWS region" in result


def test_bedrock_config_flags_a_region_mismatch(settings):
    settings.bedrock_enabled = True
    settings.bedrock_model_id = "us.anthropic.claude-3-5-sonnet-20240620-v1:0"
    settings.aws_region = "eu-west-1"
    assert any("inference profile" in p for p in settings.problems())


def test_bedrock_enabled_without_a_model_id_is_a_problem(settings):
    settings.bedrock_enabled = True
    settings.bedrock_model_id = ""
    assert any("BEDROCK_MODEL_ID is required" in p for p in settings.problems())


# --- the governance gate ------------------------------------------------------------------------


def _policy(send=True, per_column=2, max_chars=5) -> GovernancePolicy:
    return GovernancePolicy(
        send_sample_values=send,
        sample_values_per_column=per_column,
        sample_value_max_chars=max_chars,
    )


def test_sample_values_are_capped_in_count_and_length():
    context = {
        "columns": [{"name": "notes", "sample_values": ["abcdefgh", "ij", "kl", "mn"]}],
    }
    scrubbed = apply_governance(context, _policy(per_column=2, max_chars=5))
    samples = scrubbed["columns"][0]["sample_values"]
    assert len(samples) == 2  # capped to 2 per column
    assert samples[0] == "abcde…"  # truncated to 5 chars + ellipsis


def test_sample_values_can_be_stripped_entirely():
    context = {"columns": [{"name": "notes", "sample_values": ["secret-value"]}]}
    scrubbed = apply_governance(context, _policy(send=False))
    assert scrubbed["columns"][0]["sample_values"] == []


def test_top_value_entries_are_truncated_but_counts_kept():
    context = {
        "columns": [{"name": "c", "top_values": [{"value": "a-very-long-category", "count": 42}]}]
    }
    scrubbed = apply_governance(context, _policy(max_chars=4))
    entry = scrubbed["columns"][0]["top_values"][0]
    assert entry["value"] == "a-ve…" and entry["count"] == 42


def test_result_data_passes_through_unchanged():
    context = {
        "target_association": [{"feature": "charges", "method": "pearson", "score": 0.95}],
        "model_results": {"best_model": "xgboost", "primary_metric": "roc_auc"},
    }
    assert apply_governance(context, _policy()) == context  # no sample keys: untouched


def test_governance_only_ever_removes_information():
    context = {"columns": [{"name": "c", "sample_values": ["x", "y", "z"]}]}
    scrubbed = apply_governance(context, _policy(per_column=10, max_chars=100))
    assert len(scrubbed["columns"][0]["sample_values"]) <= 3  # never adds samples
