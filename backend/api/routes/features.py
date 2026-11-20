"""EDA, leakage, feature-proposal and feature-decision routes (control-ui-api design C3).

Thin wrappers over ``ExperimentService``, ``FeatureProposalService`` and ``LeakageService``
mirroring the matching MCP tools exactly. Reads return the service artifact; every mutation
resolves the caller, enforces ownership and audits. Bedrock-off is not special-cased:
``proposals.generate`` raises ``FeatureDisabledError`` and the ``safe`` wrapper maps it to 403.
"""

from collections.abc import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse

from backend.api.deps import audit_mutation, principal, require_owner
from backend.api.errors import json_ok
from backend.api.request import json_body
from backend.container import AppContainer
from backend.schemas import ApproveProposalsRequest, ProposeFeaturesRequest
from ml_engine.contracts.leakage_investigation import InvestigationPhase, UserDecision
from ml_engine.contracts.proposals import FeatureProposal


def routes() -> list[tuple[str, str, Callable[[AppContainer], Callable]]]:
    return [
        ("GET", "/experiments/{experiment_id}/eda", _eda),
        ("GET", "/experiments/{experiment_id}/leakage", _leakage),
        ("GET", "/experiments/{experiment_id}/feature-review", _feature_review),
        ("POST", "/experiments/{experiment_id}/feature-proposals", _generate_proposals),
        ("GET", "/experiments/{experiment_id}/feature-proposals", _stored_proposals),
        ("POST", "/experiments/{experiment_id}/feature-proposals/define", _define_features),
        ("POST", "/experiments/{experiment_id}/feature-proposals/approve", _approve_features),
        ("POST", "/experiments/{experiment_id}/features", _decide_features),
        ("GET", "/experiments/{experiment_id}/leakage-investigation", _get_phase),
        ("POST", "/experiments/{experiment_id}/leakage-investigation/decide", _decide_warning),
        ("POST", "/experiments/{experiment_id}/leakage-exclusions/apply", _apply_exclusions),
    ]


def _eda(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        return json_ok(container.experiments.eda(experiment_id).model_dump(mode="json"))

    return handler


def _leakage(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        report = container.experiments.leakage(experiment_id)
        return json_ok(
            {
                "experiment_id": experiment_id,
                "target_column": report.target_column,
                "findings": [f.model_dump(mode="json") for f in report.findings],
                "feature_risks": [r.model_dump(mode="json") for r in report.feature_risks],
            }
        )

    return handler


def _feature_review(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        return json_ok(container.experiments.feature_review(experiment_id).model_dump(mode="json"))

    return handler


def _generate_proposals(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        report = container.proposals.generate(
            experiment_id,
            ProposeFeaturesRequest(
                target_definition=body.get("target_definition"),
                prediction_timing=body.get("prediction_timing"),
            ),
        )
        audit_mutation(
            container,
            experiment_id,
            "propose_features",
            who.user,
            candidates=len(report.candidates),
        )
        return json_ok(report.model_dump(mode="json"))

    return handler


def _stored_proposals(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        report = container.proposals.stored(experiment_id)
        return json_ok(report.model_dump(mode="json") if report else None)

    return handler


def _define_features(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        features = [FeatureProposal.model_validate(f) for f in body.get("features", [])]
        result = container.proposals.define(
            experiment_id, features, approve=body.get("approve", True), user=who.user
        )
        audit_mutation(
            container,
            experiment_id,
            "define_derived_features",
            who.user,
            added=",".join(result.added),
            refused=len(result.rejected),
        )
        return json_ok(
            {
                "experiment_id": experiment_id,
                "added": result.added,
                "approved_names": result.report.approved_names,
                "rejected": [r.model_dump(mode="json") for r in result.rejected],
                "profiling_execution_id": result.profiling_execution_id,
            }
        )

    return handler


def _approve_features(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        result = container.proposals.approve(
            experiment_id,
            ApproveProposalsRequest(approved_names=body["approved_names"]),
            user=who.user,
        )
        report = result.report
        audit_mutation(
            container,
            experiment_id,
            "approve_features",
            who.user,
            approved=len(report.approved_names),
        )
        return json_ok(
            {
                "experiment_id": experiment_id,
                "approved_names": report.approved_names,
                "candidate_count": len(report.candidates),
                "profiling_execution_id": result.profiling_execution_id,
            }
        )

    return handler


def _decide_features(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        selection = container.experiments.save_features(
            experiment_id,
            selected_features=body["selected_features"],
            exclusion_reasons=body.get("exclusion_reasons"),
            user=who.user,
        )
        audit_mutation(
            container,
            experiment_id,
            "decide_features",
            who.user,
            selected=len(selection.selected_features),
        )
        return json_ok(
            {
                "experiment_id": experiment_id,
                "selected_count": len(selection.selected_features),
                "excluded_count": len(selection.excluded_features),
            }
        )

    return handler


def _get_phase(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        phase = request.query_params.get("phase", "pre_training")
        report = container.leakage.get_phase(experiment_id, InvestigationPhase(phase))
        return json_ok(report.model_dump(mode="json") if report else None)

    return handler


def _decide_warning(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        body = await json_body(request)
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        phase = InvestigationPhase(body.get("phase", "pre_training"))
        warning_id = body["warning_id"]
        report = container.leakage.decide_warning(
            experiment_id,
            phase=phase,
            warning_id=warning_id,
            decision=UserDecision(body["decision"]),
            user=who.user,
        )
        warning = report.warning_for(warning_id)
        audit_mutation(
            container,
            experiment_id,
            "decide_leakage_warning",
            who.user,
            warning_id=warning_id,
            decision=body["decision"],
        )
        return json_ok(
            {
                "experiment_id": experiment_id,
                "warning_id": warning_id,
                "user_decision": warning.user_decision.value,
                "review_status": warning.review_status.value,
            }
        )

    return handler


def _apply_exclusions(container: AppContainer) -> Callable:
    async def handler(request: Request) -> JSONResponse:
        experiment_id = request.path_params["experiment_id"]
        who = principal(container, request)
        require_owner(container, who, experiment_id)
        selection = container.leakage.apply_exclusions_to_selection(experiment_id, user=who.user)
        audit_mutation(container, experiment_id, "apply_leakage_exclusions", who.user)
        return json_ok(
            {
                "experiment_id": experiment_id,
                "selected_count": len(selection.selected_features),
                "excluded_count": len(selection.excluded_features),
            }
        )

    return handler
