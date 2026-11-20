"""The model catalogue.

Plugins register here and nowhere else. Adding a model requires no change to the jobs, the
state machine, the API or the UI.
"""

from ml_engine.contracts.common import ProblemType
from ml_engine.contracts.model import ModelDescriptor
from ml_engine.models.bagging_trees import (
    BaggingClassifierPlugin,
    BaggingRegressorPlugin,
    ExtraTreesClassifierPlugin,
    ExtraTreesRegressorPlugin,
)
from ml_engine.models.base import ModelPlugin
from ml_engine.models.boosting import (
    AdaBoostClassifierPlugin,
    AdaBoostRegressorPlugin,
    HistGradientBoostingClassifierPlugin,
    HistGradientBoostingRegressorPlugin,
)
from ml_engine.models.lightgbm_plugin import LightGBMClassifierPlugin, LightGBMRegressorPlugin
from ml_engine.models.linear import ElasticNetPlugin, LogisticRegressionPlugin
from ml_engine.models.random_forest import (
    RandomForestClassifierPlugin,
    RandomForestRegressorPlugin,
)
from ml_engine.models.xgboost_plugin import XGBoostClassifierPlugin, XGBoostRegressorPlugin


class UnknownModelError(KeyError):
    """Raised when a configuration references a model that is not registered."""


_REGISTRY: dict[str, ModelPlugin] = {}


def register(plugin: ModelPlugin) -> ModelPlugin:
    if not plugin.name:
        raise ValueError("A model plugin must declare a name")
    if plugin.name in _REGISTRY and type(_REGISTRY[plugin.name]) is not type(plugin):
        raise ValueError(f"Duplicate model plugin name: {plugin.name}")
    _REGISTRY[plugin.name] = plugin
    return plugin


# The built-in set matches what the stock AWS XGBoost image runs with no install: the
# scikit-learn plugins and XGBoost (stock-image migration, decision B2/B10). CatBoost is no
# longer registered by default; it is reachable by adding ``catboost`` to an experiment's
# requirements (the open catalogue). ``ml_engine/models/catboost_plugin.py`` is kept on disk,
# unimported, so it can be re-registered on demand.
for _plugin in (
    LogisticRegressionPlugin(),
    ElasticNetPlugin(),
    RandomForestClassifierPlugin(),
    RandomForestRegressorPlugin(),
    XGBoostClassifierPlugin(),
    XGBoostRegressorPlugin(),
    # ensembles-and-learners: built-in bagging/boosting learners (scikit-learn, no install).
    HistGradientBoostingClassifierPlugin(),
    HistGradientBoostingRegressorPlugin(),
    AdaBoostClassifierPlugin(),
    AdaBoostRegressorPlugin(),
    BaggingClassifierPlugin(),
    BaggingRegressorPlugin(),
    ExtraTreesClassifierPlugin(),
    ExtraTreesRegressorPlugin(),
    # ensembles-and-learners (open catalogue): LightGBM ships as an installed learner — it
    # appears in the catalogue as unavailable until the experiment installs it via requirements
    # (stock-image B10), and its import is lazy so the offline suite never needs the package.
    LightGBMClassifierPlugin(),
    LightGBMRegressorPlugin(),
):
    register(_plugin)


#: Separates a grid point's base model from its parameter hash: ``xgboost__ab12cd``. Chosen over
#: ``#`` because it is URL/S3-safe and survives SageMaker job-name sanitisation unchanged
#: (grid-search design C1).
GRID_POINT_SEPARATOR = "__"


def base_model_name(name: str) -> str:
    """The registered plugin name behind a possibly grid-point name.

    ``xgboost__ab12cd`` -> ``xgboost``; a plain model name is returned unchanged. Grid points
    (grid-search spec) are ``ModelSpec``s whose name carries a parameter hash so each point has its
    own artifacts; they still train with their base plugin.
    """
    return name.split(GRID_POINT_SEPARATOR, 1)[0]


def grid_point_id(base: str, params: dict[str, object]) -> str:
    """A stable, job-name-safe id for one grid point: ``<base>__<hash8>``.

    The hash is over the sorted params, so the same combination always yields the same id and
    distinct combinations yield distinct ids. The result is alphanumeric + ``__`` only, so it
    sanitises to a unique SageMaker job name and is a safe S3 path segment.
    """
    import hashlib
    import json

    payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:8]
    return f"{base}{GRID_POINT_SEPARATOR}{digest}"


def get_plugin(name: str) -> ModelPlugin:
    # A grid point's name (``<base>__<hash>``) resolves to its base plugin; a plain name is a
    # direct lookup. Ensemble names (``ensemble_<type>__<hash>``) are not registered here; they are
    # built from their frozen EnsembleSpec via build_ensemble_plugin / resolve_plugin.
    key = base_model_name(name)
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        raise UnknownModelError(
            f"Unknown model {name!r}. Registered models: {', '.join(sorted(_REGISTRY))}."
        ) from exc


def is_ensemble_name(name: str) -> bool:
    """True for an ensemble Model_Version name (``ensemble_<type>__<hash>``)."""
    from ml_engine.contracts.ensemble import ENSEMBLE_NAME_PREFIX

    return name.startswith(ENSEMBLE_NAME_PREFIX)


def build_ensemble_plugin(spec, problem_type, *, stacking_cv: int | None = None) -> ModelPlugin:
    """Construct an ``EnsemblePlugin`` from a frozen ``EnsembleSpec`` (ensembles-and-learners)."""
    from ml_engine.models.ensemble import EnsemblePlugin

    return EnsemblePlugin(spec, problem_type, stacking_cv=stacking_cv)


def resolve_plugin(config, model_name: str) -> ModelPlugin:
    """Resolve a configured model name to its plugin.

    An ensemble name resolves to an ``EnsemblePlugin`` built from the matching ``EnsembleSpec`` in
    the frozen config; every other name is the ordinary registry lookup. The training job uses this
    so ``get_plugin`` stays pure for plain and grid names.
    """
    if is_ensemble_name(model_name):
        spec = next((e for e in config.ensembles if e.ensemble_id == model_name), None)
        if spec is None:
            raise UnknownModelError(
                f"Ensemble {model_name!r} is not defined in the experiment configuration."
            )
        stacking_cv = config.split.cv.folds if config.split.cv is not None else None
        return build_ensemble_plugin(spec, config.problem_type, stacking_cv=stacking_cv)
    return get_plugin(model_name)


def all_plugins() -> list[ModelPlugin]:
    return [_REGISTRY[name] for name in sorted(_REGISTRY)]


def plugins_for(problem_type: ProblemType, *, only_available: bool = True) -> list[ModelPlugin]:
    """Plugins that support this problem type, optionally filtered to installed libraries."""
    plugins = [p for p in all_plugins() if p.supports(problem_type)]
    if only_available:
        plugins = [p for p in plugins if p.is_available()[0]]
    return plugins


def default_model_names(problem_type: ProblemType) -> list[str]:
    """The default model set for a problem type: every supported, installed plugin."""
    return [p.name for p in plugins_for(problem_type)]


def catalogue(problem_type: ProblemType | None = None) -> list[ModelDescriptor]:
    """Descriptors for the API, including unavailable plugins and why they are unavailable."""
    plugins = all_plugins()
    if problem_type is not None:
        plugins = [p for p in plugins if p.supports(problem_type)]
    return [p.descriptor(problem_type) for p in plugins]
