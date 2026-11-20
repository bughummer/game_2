"""The ensemble plugin (ensembles-and-learners design C4).

An ensemble is a ``ModelPlugin`` whose estimator embeds other plugins' estimators. ``build``
assembles scikit-learn's ``VotingClassifier``/``VotingRegressor`` (fixed rule) or
``StackingClassifier``/``StackingRegressor`` (learned meta-learner on cross-validated out-of-fold
predictions) over the base learners named in a frozen ``EnsembleSpec``. Because the whole plugin is
fit through the unchanged ``train_model`` path — per fold under CV — the base learners are fit on
exactly the training rows they are given, and ``Stacking*`` builds leak-free OOF meta-features and
refits the base learners on the full training set itself. So fold-safety, OOF and refit are the
sklearn estimator's own behaviour; the plugin just assembles it.
"""

from typing import Any

import sklearn
from sklearn.ensemble import (
    StackingClassifier,
    StackingRegressor,
    VotingClassifier,
    VotingRegressor,
)

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.ensemble import (
    EnsembleSpec,
    EnsembleType,
    VotingRule,
    default_meta_learner,
)
from ml_engine.contracts.model import FeatureImportance
from ml_engine.models.base import ModelPlugin, TrainingContext

#: Folds for the stacking meta-feature generation when the plugin builds its own OOF. The CV
#: feature governs the experiment's K; this bounds sklearn's internal stacking CV.
DEFAULT_STACKING_CV = 5


class EnsemblePlugin(ModelPlugin):
    """A voting or stacking ensemble assembled from a frozen ``EnsembleSpec``.

    Built per model at training time from the spec carried on the ``ModelSpec``; it never reads
    global state, so ``get_plugin`` stays pure for plain and grid names.
    """

    library = "scikit-learn"
    requires_dense_numeric = True
    supports_class_weighting = False  # weighting is a base-learner concern, set in its params

    def __init__(
        self, spec: EnsembleSpec, problem_type: ProblemType, *, stacking_cv: int | None = None
    ) -> None:
        self.spec = spec
        self.problem_type = problem_type
        self.stacking_cv = stacking_cv or DEFAULT_STACKING_CV
        self.name = spec.ensemble_id
        self.display_name = f"{spec.type.value.title()} ensemble"
        self.description = (
            f"{spec.type.value.title()} ensemble of "
            f"{', '.join(b.name for b in spec.base_learners)}."
        )
        self.supported_problem_types = (problem_type,)

    def is_available(self) -> tuple[bool, str | None]:  # type: ignore[override]
        """Available only if every base learner (and meta-learner) is available."""
        from ml_engine.models.registry import get_plugin

        for base in self.spec.base_learners:
            available, reason = get_plugin(base.name).is_available()
            if not available:
                return False, f"base learner '{base.name}' is unavailable: {reason}"
        meta = self._meta_spec()
        if meta is not None:
            available, reason = get_plugin(meta.name).is_available()
            if not available:
                return False, f"meta learner '{meta.name}' is unavailable: {reason}"
        return True, None

    def library_version(self) -> str:
        return sklearn.__version__

    def supports_probabilities(self, problem_type: ProblemType) -> bool:
        # Soft voting and stacking classifiers expose predict_proba; hard voting does not.
        if not problem_type.is_classification:
            return False
        is_hard_voting = (
            self.spec.type is EnsembleType.VOTING and self.spec.voting_rule is VotingRule.HARD
        )
        return not is_hard_voting

    def default_params(self, context: TrainingContext) -> dict[str, Any]:  # noqa: ARG002
        return {}

    def _meta_spec(self):
        if self.spec.type is not EnsembleType.STACKING:
            return None
        return self.spec.meta_learner or default_meta_learner(self.problem_type)

    def _base_estimators(self, context: TrainingContext) -> list[tuple[str, Any]]:
        from ml_engine.models.registry import get_plugin

        estimators: list[tuple[str, Any]] = []
        for index, base in enumerate(self.spec.base_learners):
            plugin = get_plugin(base.name)
            params = plugin.resolve_params(dict(base.params), context)
            # A unique label per member so repeated base learners do not collide.
            estimators.append((f"{base.name}_{index}", plugin.build(params, context)))
        return estimators

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        estimators = self._base_estimators(context)
        if self.spec.type is EnsembleType.VOTING:
            return self._build_voting(estimators)
        return self._build_stacking(estimators, context)

    def _build_voting(self, estimators: list[tuple[str, Any]]) -> Any:
        weights = [b.weight for b in self.spec.base_learners]
        if self.problem_type.is_classification:
            return VotingClassifier(
                estimators=estimators,
                voting=self.spec.voting_rule.value,
                weights=weights,
                n_jobs=-1,
            )
        return VotingRegressor(estimators=estimators, weights=weights, n_jobs=-1)

    def _build_stacking(self, estimators: list[tuple[str, Any]], context: TrainingContext) -> Any:
        from ml_engine.models.registry import get_plugin

        meta_spec = self._meta_spec()
        assert meta_spec is not None
        meta_plugin = get_plugin(meta_spec.name)
        meta_params = meta_plugin.resolve_params(dict(meta_spec.params), context)
        final_estimator = meta_plugin.build(meta_params, context)
        if self.problem_type.is_classification:
            return StackingClassifier(
                estimators=estimators,
                final_estimator=final_estimator,
                cv=self.stacking_cv,
                n_jobs=-1,
            )
        return StackingRegressor(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=self.stacking_cv,
            n_jobs=-1,
        )

    def feature_importance(
        self,
        estimator: Any,  # noqa: ARG002 - no single native importance for an ensemble
        feature_names: list[str],  # noqa: ARG002
    ) -> FeatureImportance | None:
        # A voting/stacking estimator has no single native importance; the training job's
        # permutation path provides importance (Req 1.6).
        return None
