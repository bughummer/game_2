"""Built-in bagging learners (ensembles-and-learners design C1).

Bagging and extra-trees, classifier + regressor, built only from the stock AWS image's
scikit-learn. Ordinary ``ModelPlugin``s that flow through every stage with no orchestration change.
"""

from typing import Any

import sklearn
from sklearn.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.grid import ParameterSpec
from ml_engine.models.base import ModelPlugin, TrainingContext


class _BaggingBase(ModelPlugin):
    library = "scikit-learn"
    requires_dense_numeric = True

    def library_version(self) -> str:
        return sklearn.__version__

    def parameter_space(self) -> dict[str, ParameterSpec]:
        return {
            "n_estimators": ParameterSpec(type="int", minimum=5, maximum=2000),
            "max_samples": ParameterSpec(type="float", minimum=0.1, maximum=1.0),
            "max_features": ParameterSpec(type="float", minimum=0.1, maximum=1.0),
        }

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        return {
            "n_estimators": 100,
            "max_samples": 1.0,
            "max_features": 1.0,
            "n_jobs": -1,
            "random_state": context.random_seed,
        }


class BaggingClassifierPlugin(_BaggingBase):
    name = "bagging"
    display_name = "Bagging"
    description = "Bootstrap-aggregated decision trees (scikit-learn)."
    supported_problem_types = (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )
    supports_class_weighting = False

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return BaggingClassifier(**params)


class BaggingRegressorPlugin(_BaggingBase):
    name = "bagging_regressor"
    display_name = "Bagging Regressor"
    description = "Bootstrap-aggregated regression trees (scikit-learn)."
    supported_problem_types = (ProblemType.REGRESSION,)
    supports_class_weighting = False

    def supports_probabilities(self, problem_type: ProblemType) -> bool:  # noqa: ARG002
        return False

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return BaggingRegressor(**params)


class _ExtraTreesBase(ModelPlugin):
    library = "scikit-learn"
    requires_dense_numeric = True

    def library_version(self) -> str:
        return sklearn.__version__

    def parameter_space(self) -> dict[str, ParameterSpec]:
        return {
            "n_estimators": ParameterSpec(type="int", minimum=10, maximum=5000),
            "max_depth": ParameterSpec(type="int", minimum=1, maximum=100),
            "min_samples_leaf": ParameterSpec(type="int", minimum=1, maximum=1000),
            "max_features": ParameterSpec(type="str", choices=["sqrt", "log2"]),
        }

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        min_samples_leaf = 1 if context.row_count > 10_000 else 2
        return {
            "n_estimators": 300,
            "max_depth": None,
            "min_samples_leaf": min_samples_leaf,
            "max_features": "sqrt",
            "n_jobs": -1,
            "random_state": context.random_seed,
        }


class ExtraTreesClassifierPlugin(_ExtraTreesBase):
    name = "extra_trees"
    display_name = "Extremely Randomized Trees"
    description = "Extra-trees classifier: more randomized, lower-variance than a random forest."
    supported_problem_types = (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )
    supports_class_weighting = True

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        params = super().default_params(context)
        if context.wants_class_weighting:
            # 'balanced_subsample' recomputes weights per bootstrap, so enable bootstrapping
            # (ExtraTrees defaults to bootstrap=False) to match the random-forest weighting path.
            params["class_weight"] = "balanced_subsample"
            params["bootstrap"] = True
        return params

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return ExtraTreesClassifier(**params)


class ExtraTreesRegressorPlugin(_ExtraTreesBase):
    name = "extra_trees_regressor"
    display_name = "Extremely Randomized Trees Regressor"
    description = "Extra-trees regressor; a robust, low-variance non-linear regression baseline."
    supported_problem_types = (ProblemType.REGRESSION,)
    supports_class_weighting = False

    def supports_probabilities(self, problem_type: ProblemType) -> bool:  # noqa: ARG002
        return False

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        params = super().default_params(context)
        params["max_features"] = 1.0
        return params

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return ExtraTreesRegressor(**params)
