"""Data governance for what reaches Bedrock, and the Bedrock region/profile check.

Two concerns, both from requirement 15.5 / 16.4 and handover decision 9:

* ``bedrock_region_mismatch`` — a cross-region inference profile id is prefixed with a region
  group (``us.``, ``eu.``, ``apac.``); the configured region must be in that group, or Bedrock
  rejects the call with "model identifier is invalid". We catch it at config time instead.
* ``apply_governance`` — the reasoning context is already bounded and data-free except for capped
  category sample values. This gate enforces the configured caps on those samples and can strip
  them entirely. A full dataset is never in the context to begin with; this is defence in depth.
"""

from dataclasses import dataclass
from typing import Any

#: Inference-profile region-group prefixes and the AWS region prefixes they allow.
_PROFILE_REGION_GROUPS = {
    "us": ("us-",),
    "eu": ("eu-",),
    "apac": ("ap-",),
    "us-gov": ("us-gov-",),
}


def bedrock_region_mismatch(model_id: str, region: str) -> str | None:
    """Return an error message if a cross-region profile id does not match the region, else None.

    A plain model id (no region-group prefix) is left alone — only inference-profile ids carry
    the prefix.
    """
    prefix = model_id.split(".", 1)[0].lower() if "." in model_id else ""
    allowed = _PROFILE_REGION_GROUPS.get(prefix)
    if allowed is None:
        return None  # not a cross-region inference profile id
    if not any(region.startswith(group) for group in allowed):
        return (
            f"Bedrock model '{model_id}' is a '{prefix}.' inference profile, which requires an "
            f"AWS region starting with {' or '.join(allowed)}; configured region is '{region}'."
        )
    return None


@dataclass(frozen=True, slots=True)
class GovernancePolicy:
    """The configured limits on what reaches Bedrock."""

    send_sample_values: bool
    sample_values_per_column: int
    sample_value_max_chars: int

    @classmethod
    def from_settings(cls, settings: Any) -> "GovernancePolicy":
        return cls(
            send_sample_values=settings.bedrock_send_sample_values,
            sample_values_per_column=settings.bedrock_sample_values_per_column,
            sample_value_max_chars=settings.bedrock_sample_value_max_chars,
        )


#: Keys in a reasoning context that could carry sample values from the data.
_SAMPLE_KEYS = ("sample_values", "top_values")


def apply_governance(context: dict[str, Any], policy: GovernancePolicy) -> dict[str, Any]:
    """Return a copy of the context with sample values capped or removed per policy.

    Everything else — column names, association/importance scores, counts, model metrics — is
    already deterministic result data and passes through unchanged. This never adds data; it
    only ever removes or shortens it.
    """
    return _scrub(context, policy)


def _scrub(value: Any, policy: GovernancePolicy) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in _SAMPLE_KEYS:
                result[key] = _cap_samples(item, policy)
            else:
                result[key] = _scrub(item, policy)
        return result
    if isinstance(value, list):
        return [_scrub(item, policy) for item in value]
    return value


def _cap_samples(samples: Any, policy: GovernancePolicy) -> Any:
    if not policy.send_sample_values:
        return []
    if not isinstance(samples, list):
        return samples
    capped = samples[: policy.sample_values_per_column]
    return [_truncate(item, policy.sample_value_max_chars) for item in capped]


def _truncate(item: Any, max_chars: int) -> Any:
    if isinstance(item, str) and len(item) > max_chars:
        return item[:max_chars] + "…"
    if isinstance(item, dict):
        # A top-value entry: truncate its displayed value, keep its count.
        out = dict(item)
        for field in ("value", "name"):
            if isinstance(out.get(field), str) and len(out[field]) > max_chars:
                out[field] = out[field][:max_chars] + "…"
        return out
    return item
