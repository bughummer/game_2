"""The static SageMaker price table (design DD-4, handover decision 6).

Prices are read from ``infra/pricing/sagemaker_prices.json`` and never invented: an instance or
region missing from the table has no price, and callers must treat that as "unknown", not zero.
Every estimate carries the table's source, version and retrieval date.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from ml_engine.contracts.execution import JobKind


class PriceTableError(ValueError):
    """The price table is malformed."""


@dataclass(frozen=True, slots=True)
class Rate:
    hourly_usd: float
    source: str
    version: str
    retrieved_on: str
    verified: bool


@dataclass(frozen=True, slots=True)
class PriceTable:
    source: str
    version: str
    retrieved_on: str
    verified: bool
    _rates: dict[tuple[str, str, str], float]

    @classmethod
    def load(cls, path: Path) -> "PriceTable":
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            rates: dict[tuple[str, str, str], float] = {}
            for region, kinds in raw["regions"].items():
                for kind, instances in kinds.items():
                    JobKind(kind)  # reject unknown job kinds early
                    for instance, price in instances.items():
                        price = float(price)
                        if price <= 0:
                            raise PriceTableError(f"non-positive price for {instance}")
                        rates[(region, kind, instance)] = price
            return cls(
                source=str(raw["source"]),
                version=str(raw["version"]),
                retrieved_on=str(raw["retrieved_on"]),
                verified=bool(raw.get("verified", False)),
                _rates=rates,
            )
        except (KeyError, TypeError, ValueError) as error:
            if isinstance(error, PriceTableError):
                raise
            raise PriceTableError(f"invalid price table {path}: {error}") from error

    def rate(self, region: str, kind: JobKind, instance_type: str) -> Rate | None:
        price = self._rates.get((region, kind.value, instance_type))
        if price is None:
            return None
        return Rate(price, self.source, self.version, self.retrieved_on, self.verified)


def reserved_cost_usd(rate: Rate, *, instance_count: int, max_billable_seconds: int) -> float:
    """The most a job can cost: on-demand rate x instances x its maximum billable time.

    For a Spot job SageMaker bills at most the training time, which is capped by
    ``MaxRuntimeInSeconds``, and never above the on-demand rate — so this is a true upper bound
    for compute. It excludes S3, CloudWatch and data transfer (requirement 7.4).
    """
    return round(rate.hourly_usd * instance_count * max_billable_seconds / 3600.0, 6)
