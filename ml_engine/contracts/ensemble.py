"""Ensemble contracts (ensembles-and-learners design C3).

An ensemble combines several already-chosen base learners into one model. Two kinds: a
**voting** ensemble (a fixed combining rule, nothing learned on top) and a **stacking** ensemble
(a learned meta-learner trained on the base learners' cross-validated out-of-fold predictions).

An ensemble is modelled as an ordinary model: the service expands a frozen ``EnsembleSpec`` into a
single ``ModelSpec`` whose name is the spec's stable ``ensemble_id`` and whose params carry the
spec, so the existing training/evaluation/model-version machinery runs it unchanged. These are the
versioned, approved definitions; nothing is auto-selected.
"""

import hashlib
import json
from enum import StrEnum

from pydantic import Field, model_validator

from ml_engine.contracts.common import ProblemType, StrictModel

ENSEMBLE_SCHEMA_VERSION = "1.0"

#: Separates an ensemble's type from its spec hash in its model name: ``ensemble_voting__ab12cd``.
#: Shares the grid-point separator so the registry's base-name split treats it uniformly.
ENSEMBLE_NAME_PREFIX = "ensemble_"
MIN_BASE_LEARNERS = 2


class EnsembleType(StrEnum):
    VOTING = "voting"
    STACKING = "stacking"


class VotingRule(StrEnum):
    """How a voting ensemble combines classifier outputs. Regression always averages."""

    SOFT = "soft"  # mean of predicted probabilities
    HARD = "hard"  # majority predicted class


class BaseLearnerSpec(StrictModel):
    """One base learner in an ensemble: a registered learner name, its params, and a weight."""

    name: str
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)
    weight: float = Field(default=1.0, gt=0.0, description="Voting weight; equal by default.")


class MetaLearnerSpec(StrictModel):
    """The second-stage learner of a stacking ensemble (default resolved by problem type)."""

    name: str
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


def default_meta_learner(problem_type: ProblemType) -> MetaLearnerSpec:
    """Logistic regression for classification, ridge/elastic-net for regression (open q 2)."""
    if problem_type.is_classification:
        return MetaLearnerSpec(name="logistic_regression")
    return MetaLearnerSpec(name="elastic_net")


class EnsembleSpec(StrictModel):
    """A user-approved ensemble definition, frozen into the experiment configuration."""

    schema_version: str = ENSEMBLE_SCHEMA_VERSION
    type: EnsembleType
    base_learners: list[BaseLearnerSpec]
    voting_rule: VotingRule = VotingRule.SOFT
    meta_learner: MetaLearnerSpec | None = Field(
        default=None, description="Stacking only; None resolves to the problem-type default."
    )

    @model_validator(mode="after")
    def _at_least_two_base_learners(self) -> "EnsembleSpec":
        if len(self.base_learners) < MIN_BASE_LEARNERS:
            raise ValueError(
                f"an ensemble needs at least {MIN_BASE_LEARNERS} base learners; "
                f"got {len(self.base_learners)}"
            )
        return self

    @property
    def ensemble_id(self) -> str:
        """A stable, job-name-safe id: ``ensemble_<type>__<hash8>`` over the sorted spec.

        The hash covers the base learners (name + params + weight), the voting rule and the
        meta-learner, so the same definition always yields the same id and distinct definitions
        yield distinct ids (mirroring the grid-point id scheme).
        """
        payload = {
            "type": self.type.value,
            "voting_rule": self.voting_rule.value,
            "base_learners": sorted(
                (b.name, json.dumps(b.params, sort_keys=True, default=str), b.weight)
                for b in self.base_learners
            ),
            "meta_learner": None
            if self.meta_learner is None
            else (
                self.meta_learner.name,
                json.dumps(self.meta_learner.params, sort_keys=True, default=str),
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()[:8]
        return f"{ENSEMBLE_NAME_PREFIX}{self.type.value}__{digest}"


class EnsembleLineage(StrictModel):
    """The base-learner Model_Versions an ensemble combines, for traceable lineage (Req 6.4)."""

    schema_version: str = ENSEMBLE_SCHEMA_VERSION
    ensemble_type: EnsembleType
    base_learner_names: list[str] = Field(default_factory=list)
    meta_learner_name: str | None = None
    base_version_ids: list[str] = Field(
        default_factory=list, description="version_id of each combined base learner, when known."
    )


def is_empty_ensembles(value: object) -> bool:
    """``exclude_if`` predicate: an empty ensemble list is omitted from the serialised config."""
    return not value
