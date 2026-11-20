"""Feature tools (ml-assistant task 10.2; Req 3.1, 5.7, 5.8) over the offline inline container."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from backend.agent import tools_features
from backend.agent.tools import AwaitedJobSpec, ToolContext, ToolRegistry
from backend.auth import Principal
from ml_engine.contracts.agent import ToolClass
from ml_engine.io import ExperimentLayout
from tests.mcp import conftest as mcp_fixtures
from tests.mcp.conftest import DEV_USER
from tests.support.datasets import APPROVED_PREFIX

# The offline MCP fixtures: in-memory (moto) S3, fake SageMaker, stages run in-process.
settings = mcp_fixtures.settings
s3_client = mcp_fixtures.s3_client
sagemaker = mcp_fixtures.sagemaker
inline_container = mcp_fixtures.inline_container

RATIO = {
    "op": "ratio",
    "name": "charges_per_tenure",
    "numerator": "monthly_charges",
    "denominator": "tenure_months",
    "rationale": "spend per month of tenure",
    "available_at_prediction_time": True,
}


@pytest.fixture
def registry() -> ToolRegistry:
    registry = ToolRegistry()
    tools_features.register(registry, SimpleNamespace())
    return registry


@pytest.fixture
def experiment_id(inline_container, s3_client, classification_frame: pd.DataFrame) -> str:
    s3_client.put_object(
        Bucket=APPROVED_PREFIX.split("/")[2],
        Key="curated/churn.csv",
        Body=classification_frame.to_csv(index=False),
    )
    record = inline_container.experiments.create(
        name="churn",
        dataset_uri=f"{APPROVED_PREFIX}/churn.csv",
        target_column="churned",
        user=DEV_USER,
    )
    return record.experiment_id


def _ctx(container) -> ToolContext:
    return ToolContext(session=None, principal=Principal(user=DEV_USER), services=container)


def _call(registry, container, name, **arguments):
    spec = registry.get(name)
    return spec.handler(_ctx(container), spec.input_model.model_validate(arguments))


def _candidates(container, experiment_id) -> list[str]:
    return [f.feature for f in container.experiments.feature_review(experiment_id).features]


def test_registration_classes_and_flags(registry):
    expected = {
        "propose_features": (ToolClass.MUTATING, False, "propose_features"),
        "get_feature_proposals": (ToolClass.READ, False, None),
        "define_derived_features": (ToolClass.COSTLY, True, "define_derived_features"),
        "approve_features": (ToolClass.COSTLY, True, "approve_features"),
        "decide_features": (ToolClass.MUTATING, True, "decide_features"),
    }
    assert set(registry.names()) == set(expected)
    for name, (tool_class, changes_config, audit_action) in expected.items():
        spec = registry.get(name)
        assert spec.tool_class is tool_class, name
        assert spec.flags.changes_config is changes_config, name
        assert spec.flags.audit_action == audit_action, name
        assert spec.flags.experiment_arg == "experiment_id", name
    for name in ("define_derived_features", "approve_features"):
        assert registry.get(name).flags.estimator == "processing"
        assert registry.get(name).job_extractor is not None
    assert registry.get("decide_features").protected is not None
    assert all(registry.get(n).protected is None for n in expected if n != "decide_features")


def test_job_extractor_reports_the_profiling_job():
    extract = tools_features.profiling_job_extractor
    assert extract({"experiment_id": "e1", "profiling_execution_id": "x1"}) == [
        AwaitedJobSpec(execution_id="x1", experiment_id="e1", kind="derivedprofiling")
    ]
    assert extract({"experiment_id": "e1", "profiling_execution_id": None}) == []


def test_decide_features_is_protected_when_a_column_is_dropped(
    registry, inline_container, experiment_id
):
    spec = registry.get("decide_features")
    candidates = _candidates(inline_container, experiment_id)
    keep_all = spec.input_model(experiment_id=experiment_id, selected_features=candidates)
    drop_one = spec.input_model(experiment_id=experiment_id, selected_features=candidates[1:])
    assert spec.protected(_ctx(inline_container), keep_all) is False
    assert spec.protected(_ctx(inline_container), drop_one) is True


def test_decide_features_is_protected_when_evaluation_raises(registry, inline_container):
    spec = registry.get("decide_features")
    args = spec.input_model(experiment_id="exp-missing", selected_features=["a"])
    assert spec.protected(_ctx(inline_container), args) is True

    class _Broken:
        def feature_review(self, experiment_id):
            raise RuntimeError("store down")

    broken = ToolContext(
        session=None,
        principal=Principal(user=DEV_USER),
        services=SimpleNamespace(experiments=_Broken()),
    )
    assert spec.protected(broken, args) is True


def test_decide_features_saves_the_selection_with_route_audit_keys(
    registry, inline_container, experiment_id
):
    candidates = _candidates(inline_container, experiment_id)
    out = _call(
        registry,
        inline_container,
        "decide_features",
        experiment_id=experiment_id,
        selected_features=candidates[:2],
    )
    assert out.data == {
        "experiment_id": experiment_id,
        "selected_count": 2,
        "excluded_count": len(candidates) - 2,
    }
    assert out.audit_detail == {"selected": "2"}


def test_define_then_approve_and_reads_leave_the_selection_alone(
    registry, inline_container, experiment_id
):
    candidates = _candidates(inline_container, experiment_id)
    _call(
        registry,
        inline_container,
        "decide_features",
        experiment_id=experiment_id,
        selected_features=candidates,
    )
    record = inline_container.experiments.get(experiment_id)
    selection_uri = ExperimentLayout(base=record.artifact_prefix).selected_features

    def stored_selection() -> bytes:
        return inline_container.store.read_bytes(selection_uri)

    before = stored_selection()

    defined = _call(
        registry,
        inline_container,
        "define_derived_features",
        experiment_id=experiment_id,
        features=[RATIO],
        approve=False,
    )
    assert defined.data["added"] == ["charges_per_tenure"]
    assert defined.data["profiling_execution_id"] is None and defined.awaited == ()
    assert defined.audit_detail == {"added": "charges_per_tenure", "refused": "0"}

    proposals = _call(registry, inline_container, "get_feature_proposals", experiment_id=experiment_id)
    assert [c["name"] for c in proposals.data["candidates"]] == ["charges_per_tenure"]
    assert stored_selection() == before

    approved = _call(
        registry,
        inline_container,
        "approve_features",
        experiment_id=experiment_id,
        approved_names=["charges_per_tenure"],
    )
    assert approved.data["approved_names"] == ["charges_per_tenure"]
    assert approved.audit_detail == {"approved": "1"}
    execution_id = approved.data["profiling_execution_id"]
    if execution_id:
        assert approved.awaited == (
            AwaitedJobSpec(execution_id, experiment_id, "derivedprofiling"),
        )
    # Approval is a config change but never rewrites the saved selection.
    assert stored_selection() == before
