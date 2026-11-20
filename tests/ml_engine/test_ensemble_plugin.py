"""The EnsemblePlugin and installed-learner gating (ensembles-and-learners design C2, C4).

Property 2 (installed-learner gating) and the plugin-build behaviour, offline.
"""

from sklearn.ensemble import StackingClassifier, VotingClassifier

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.ensemble import (
    BaseLearnerSpec,
    EnsembleSpec,
    EnsembleType,
    MetaLearnerSpec,
    VotingRule,
)
from ml_engine.models import build_ensemble_plugin, get_plugin
from ml_engine.models.base import TrainingContext

BINARY = ProblemType.BINARY_CLASSIFICATION


def _context() -> TrainingContext:
    return TrainingContext(problem_type=BINARY, random_seed=42)


def _voting_spec() -> EnsembleSpec:
    return EnsembleSpec(
        type=EnsembleType.VOTING,
        base_learners=[
            BaseLearnerSpec(name="logistic_regression"),
            BaseLearnerSpec(name="random_forest", weight=2.0),
        ],
        voting_rule=VotingRule.SOFT,
    )


def _stacking_spec() -> EnsembleSpec:
    return EnsembleSpec(
        type=EnsembleType.STACKING,
        base_learners=[
            BaseLearnerSpec(name="logistic_regression"),
            BaseLearnerSpec(name="random_forest"),
        ],
        meta_learner=MetaLearnerSpec(name="logistic_regression"),
    )


def test_voting_plugin_builds_a_sklearn_voting_classifier():
    plugin = build_ensemble_plugin(_voting_spec(), BINARY)
    estimator = plugin.build({}, _context())
    assert isinstance(estimator, VotingClassifier)
    assert estimator.voting == "soft"
    assert estimator.weights == [1.0, 2.0]
    assert len(estimator.estimators) == 2


def test_stacking_plugin_builds_a_sklearn_stacking_classifier_with_cv():
    plugin = build_ensemble_plugin(_stacking_spec(), BINARY, stacking_cv=4)
    estimator = plugin.build({}, _context())
    assert isinstance(estimator, StackingClassifier)
    assert estimator.cv == 4
    assert len(estimator.estimators) == 2
    assert estimator.final_estimator is not None


def test_ensemble_id_is_stable_and_distinct():
    a = _voting_spec().ensemble_id
    b = _voting_spec().ensemble_id
    assert a == b and a.startswith("ensemble_voting__")
    assert a != _stacking_spec().ensemble_id


def test_ensemble_availability_is_the_conjunction_of_base_learners():
    # An ensemble naming an uninstalled learner (lightgbm) reports unavailable with a clear reason.
    spec = EnsembleSpec(
        type=EnsembleType.VOTING,
        base_learners=[
            BaseLearnerSpec(name="logistic_regression"),
            BaseLearnerSpec(name="lightgbm"),
        ],
    )
    plugin = build_ensemble_plugin(spec, BINARY)
    available, reason = plugin.is_available()
    assert available is False
    assert "lightgbm" in reason


def test_lightgbm_plugin_reports_unavailable_without_crashing():
    plugin = get_plugin("lightgbm")
    available, reason = plugin.is_available()
    # lightgbm is not installed in the test environment; it must degrade, not crash.
    assert available is False
    assert "requirements" in reason
    # Its descriptor is still produced (catalogue shows it as unavailable).
    descriptor = plugin.descriptor(BINARY)
    assert descriptor.available is False
    assert descriptor.name == "lightgbm"
