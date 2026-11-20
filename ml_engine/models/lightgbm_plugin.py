"""LightGBM plugins (ensembles-and-learners design C2, open catalogue).

LightGBM is not in the stock AWS image, so it is an *installed* learner: reachable only when the
experiment's ``requirements`` (stock-image B10) installed ``lightgbm`` at job start. The import is
lazy and gated by ``is_available()`` so the offline suite and a no-requirements run never need the
package — an absent library degrades to a clear "add it to this experiment's requirements" reason
rather than an import crash.
"""

from typing import Any

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.grid import ParameterSpec
from ml_engine.models.base import ModelNotAvailableError, ModelPlugin, TrainingContext

#: The message an unavailable LightGBM plugin reports, naming the fix (Req 2.1, 2.2).
_UNAVAILABLE_REASON = "lightgbm is not installed; add 'lightgbm' to this experiment's requirements"


def _lightgbm_module():
    """Import lightgbm lazily; returns the module or None if it is not installed."""
    try:
        import lightgbm

        return lightgbm
    except ImportError:
        return None


class _LightGBMBase(ModelPlugin):
    library = "lightgbm"
    requires_dense_numeric = True

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        return (_lightgbm_module() is not None), (
            None if _lightgbm_module() is not None else _UNAVAILABLE_REASON
        )

    def library_version(self) -> str:
        module = _lightgbm_module()
        if module is None:
            raise ModelNotAvailableError(_UNAVAILABLE_REASON)
        return module.__version__

    def parameter_space(self) -> dict[str, ParameterSpec]:
        return {
            "n_estimators": ParameterSpec(type="int", minimum=10, maximum=5000),
            "learning_rate": ParameterSpec(type="float", minimum=0.001, maximum=1.0),
            "num_leaves": ParameterSpec(type="int", minimum=2, maximum=1024),
            "max_depth": ParameterSpec(type="int", minimum=-1, maximum=100),
            "reg_lambda": ParameterSpec(type="float", minimum=0.0, maximum=1000.0),
        }

    def default_params(self, context: TrainingContext) -> dict[str, Any]:
        return {
            "n_estimators": 300,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "random_state": context.random_seed,
            "n_jobs": -1,
            "verbosity": -1,
        }


class LightGBMClassifierPlugin(_LightGBMBase):
    name = "lightgbm"
    display_name = "LightGBM"
    description = "Gradient-boosted trees (LightGBM). Installed via the experiment's requirements."
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
        module = _lightgbm_module()
        if module is None:
            raise ModelNotAvailableError(_UNAVAILABLE_REASON)
        return module.LGBMClassifier(**params)


class LightGBMRegressorPlugin(_LightGBMBase):
    name = "lightgbm_regressor"
    display_name = "LightGBM Regressor"
    description = "Gradient-boosted regression trees (LightGBM), installed via requirements."
    supported_problem_types = (ProblemType.REGRESSION,)
    supports_class_weighting = False

    def supports_probabilities(self, problem_type: ProblemType) -> bool:  # noqa: ARG002
        return False

    def build(self, params: dict[str, Any], context: TrainingContext) -> Any:  # noqa: ARG002
        module = _lightgbm_module()
        if module is None:
            raise ModelNotAvailableError(_UNAVAILABLE_REASON)
        return module.LGBMRegressor(**params)
