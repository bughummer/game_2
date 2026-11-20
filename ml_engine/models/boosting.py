"""Built-in boosting learners (ensembles-and-learners design C1).

Histogram gradient boosting and AdaBoost, classifier + regressor, built only from the stock AWS
image's scikit-learn — no install needed. Each is an ordinary ``ModelPlugin`` and flows through
training, CV, evaluation, importance, grid search and validation with no orchestration change.
"""

from typing import Any

import sklearn
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.grid import ParameterSpec
from ml_engine.models.base import ModelPlugin, TrainingContext


class _HistGradientBoostingBase(ModelPlugin):
    library = "scikit-learn"
    requires_dense_numeric = True

    def library_version(self) -> str:
        return sklearn.__version__

    def parameter_space(self) -> dict[str, ParameterSpec]:
        return {
            "learning_rate": ParameterSpec(type="float", minimum=0.001, maximum=1.0),
            "max_iter": ParameterSpec(type="int", minimum=10, maximum=5000),
            "max_depth": ParameterSpec(type="int", minimum=1, maximum=100),
            "max_leaf_nodes": ParameterSpec(type="int", minimum=2, maximum=1000),
            "l2_regularization": ParameterSpec(type="float", minimum=0.0, maximum=1000.0),
        }

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        return {
            "learning_rate": 0.1,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "l2_regularization": 0.0,
            "random_state": context.random_seed,
        }


class HistGradientBoostingClassifierPlugin(_HistGradientBoostingBase):
    name = "hist_gradient_boosting"
    display_name = "Histogram Gradient Boosting"
    description = (
        "Fast histogram-based gradient boosting (scikit-learn). A strong tree-boosting baseline "
        "that needs no install beyond the stock image."
    )
    supported_problem_types = (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )
    supports_class_weighting = True

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        params = super().default_params(context)
        if context.wants_class_weighting:
            params["class_weight"] = "balanced"
        return params

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return HistGradientBoostingClassifier(**params)


class HistGradientBoostingRegressorPlugin(_HistGradientBoostingBase):
    name = "hist_gradient_boosting_regressor"
    display_name = "Histogram Gradient Boosting Regressor"
    description = "Fast histogram-based gradient boosting regression (scikit-learn)."
    supported_problem_types = (ProblemType.REGRESSION,)
    supports_class_weighting = False

    def supports_probabilities(self, problem_type: ProblemType) -> bool:  # noqa: ARG002
        return False

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return HistGradientBoostingRegressor(**params)


class _AdaBoostBase(ModelPlugin):
    library = "scikit-learn"
    requires_dense_numeric = True

    def library_version(self) -> str:
        return sklearn.__version__

    def parameter_space(self) -> dict[str, ParameterSpec]:
        return {
            "n_estimators": ParameterSpec(type="int", minimum=10, maximum=5000),
            "learning_rate": ParameterSpec(type="float", minimum=0.001, maximum=10.0),
        }

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        return {
            "n_estimators": 200,
            "learning_rate": 1.0,
            "random_state": context.random_seed,
        }


class AdaBoostClassifierPlugin(_AdaBoostBase):
    name = "adaboost"
    display_name = "AdaBoost"
    description = "Adaptive boosting of shallow trees (scikit-learn)."
    supported_problem_types = (
        ProblemType.BINARY_CLASSIFICATION,
        ProblemType.MULTICLASS_CLASSIFICATION,
    )
    supports_class_weighting = False

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return AdaBoostClassifier(**params)


class AdaBoostRegressorPlugin(_AdaBoostBase):
    name = "adaboost_regressor"
    display_name = "AdaBoost Regressor"
    description = "Adaptive boosting for regression (scikit-learn)."
    supported_problem_types = (ProblemType.REGRESSION,)
    supports_class_weighting = False

    def supports_probabilities(self, problem_type: ProblemType) -> bool:  # noqa: ARG002
        return False

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        return AdaBoostRegressor(**params)
