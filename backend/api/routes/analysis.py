"""Post-training analysis, holdout evaluation and interpretation routes (control-ui-api C3).

Thin wrappers over ``ExperimentService``, ``AnalysisService``, ``LeakageService`` and
``ReasoningService`` mirroring the matching MCP tools. Every number comes from the deterministic
engine; the interpretation route stores an LLM narrative and needs Bedrock (``FeatureDisabledError``
-> 403 via the ``safe`` wrapper).
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import audit_mutation, principal, require_owner
from backend.api.errors import json_ok
from backend.api.request import json_body
from backend.api.views import REQUIREMENTS_LINE_CAP, holdout_summary, requirements_uri
from backend.container import AppContainer
from ml_engine.contracts.environment import RESULT_CRITICAL_PACKAGES

#: Permutation entries returned per model, matching the MCP's cap.
PERMUTATION_ENTRY_CAP = 25


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    return [
        ("GET", "/experiments/{experiment_id}/comparison", _comparison),
        ("GET", "/experiments/{experiment_id}/permutation-importance", _permutation),
        ("POST", "/experiments/{experiment_id}/ablation", _start_ablation),
        ("GET", "/experiments/{experiment_id}/ablation", _get_ablation),
        (
            "POST",
            "/experiments/{experiment_id}/leakage-investigation/post-training",
            _post_training_leakage,
        ),
        ("POST", "/experiments/{experiment_id}/interpret", _interpret),
        ("GET", "/experiments/{experiment_id}/environment", _environment),
        ("POST", "/experiments/{experiment_id}/holdout-evaluation", _start_holdout),
        ("GET", "/experiments/{experiment_id}/holdout-evaluation", _get_holdout),
    ]


def _comparison(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        return json_ok(container.experiments.comparison(experiment_id).model_dump(mode="json"))

    return handler


def _permutation(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        model_name = request.query_params.get("model_name")
        results = container.analysis.permutation_importance(experiment_id, model_name)
        return json_ok(
            {
                "experiment_id": experiment_id,
                "models": [
                    {
                        **r.model_dump(mode="json", exclude={"entries"}),
                        "entries": [
                            e.model_dump(mode="json") for e in r.entries[:PERMUTATION_ENTRY_CAP]
                        ],
                    }
                    for r in results
                ],
            }
        )

    return handler


def _start_ablation(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        started = container.analysis.start_ablation(
            experiment_id, features=body.get("features"), model_name=body.get("model_name")
        )
        audit_mutation(
            container,
            experiment_id,
            "start_ablation",
            who.user,
            features=",".join(started["features"]),
        )
        return json_ok(started)

    return handler


def _get_ablation(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        return json_ok(container.analysis.ablation(experiment_id))

    return handler


def _post_training_leakage(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        report = container.leakage.build_post_training_phase(experiment_id)
        audit_mutation(
            container,
            experiment_id,
            "investigate_leakage_post_training",
            who.user,
            warnings=len(report.warnings),
        )
        return json_ok(report.model_dump(mode="json"))

    return handler


def _interpret(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        report = container.reasoning.interpret(experiment_id, question=body.get("question"))
        audit_mutation(container, experiment_id, "interpret_results", who.user)
        return json_ok(report.model_dump(mode="json"))

    return handler


def _environment(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        environment, requirements = container.experiments.environment(experiment_id)
        lines = requirements.splitlines()
        return json_ok(
            {
                "experiment_id": experiment_id,
                "image_uri": environment.image_uri,
                "python_version": environment.python_version,
                "platform": environment.platform,
                "captured_at": environment.captured_at.isoformat(),
                "key_packages": {
                    name: environment.packages[name]
                    for name in RESULT_CRITICAL_PACKAGES
                    if name in environment.packages
                },
                "package_count": len(environment.packages),
                "requirements_txt": "\n".join(lines[:REQUIREMENTS_LINE_CAP]),
                "requirements_truncated": len(lines) > REQUIREMENTS_LINE_CAP,
                "requirements_uri": requirements_uri(container.experiments, experiment_id),
            }
        )

    return handler


def _start_holdout(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        status = container.experiments.start_holdout_evaluation(
            experiment_id,
            dataset_uri=body["dataset_uri"],
            file_format=body.get("file_format"),
            user=who.user,
        )
        audit_mutation(
            container,
            experiment_id,
            "start_holdout_evaluation",
            who.user,
            evaluation_id=status.evaluation_id,
            dataset_uri=body["dataset_uri"],
        )
        return json_ok(holdout_summary(status))

    return handler


def _get_holdout(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        evaluation_id = request.query_params.get("evaluation_id")
        return json_ok(
            holdout_summary(container.experiments.holdout_evaluation(experiment_id, evaluation_id))
        )

    return handler
