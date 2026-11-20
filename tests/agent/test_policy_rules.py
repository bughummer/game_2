"""Table-driven unit tests for the approval gate rules (design C6)."""

from dataclasses import replace

import pytest
from pydantic import BaseModel

from backend.agent.policy import GateAction, GateInput, gate
from backend.agent.tools import ToolFlags, ToolOutput, ToolSpec
from ml_engine.contracts.agent import Mode, ToolClass

S, A = Mode.SUPERVISED, Mode.AUTOPILOT
EX, APP, REF = GateAction.EXECUTE, GateAction.REQUIRE_APPROVAL, GateAction.REFUSE


class _In(BaseModel):
    pass


def _spec(tool_class: ToolClass, **flags) -> ToolSpec:
    return ToolSpec(
        name="t",
        description="",
        input_model=_In,
        tool_class=tool_class,
        handler=lambda ctx, args: ToolOutput(data={}),
        flags=ToolFlags(**flags),
    )


def _inp(spec: ToolSpec, mode: Mode, **kw) -> GateInput:
    base = GateInput(
        spec=spec,
        mode=mode,
        plan_accepted=True,
        protected=False,
        estimate_usd=1.0,
        session_spend_usd=0.0,
        autopilot_max_spend_usd=10.0,
        guard_refused_before=False,
        upload_approved_before=False,
    )
    return replace(base, **kw)


READ, MUT, COST = ToolClass.READ, ToolClass.MUTATING, ToolClass.COSTLY

CASES = [
    # (id, spec, mode, overrides, action, reason)
    ("no_plan", _spec(MUT), S, {"plan_accepted": False}, REF, "plan_required"),
    ("control_no_plan", _spec(MUT, control=True), S, {"plan_accepted": False}, EX, "mutating"),
    ("read_supervised", _spec(READ), S, {}, EX, "read"),
    ("read_protected_autopilot", _spec(READ), A, {"protected": True}, EX, "read"),
    ("script_run_autopilot", _spec(COST, always_approve=True), A, {}, APP, "script_run"),
    ("script_run_supervised", _spec(COST, always_approve=True), S, {}, APP, "script_run"),
    ("first_upload_autopilot", _spec(MUT, upload=True), A, {}, APP, "first_upload"),
    (
        "later_upload",
        _spec(MUT, upload=True),
        A,
        {"upload_approved_before": True},
        EX,
        "mutating",
    ),
    ("protected_mutating", _spec(MUT), A, {"protected": True}, APP, "protected_action"),
    ("protected_costly", _spec(COST), A, {"protected": True}, APP, "protected_action"),
    ("config_supervised", _spec(MUT, changes_config=True), S, {}, APP, "config_change"),
    ("config_autopilot", _spec(MUT, changes_config=True), A, {}, EX, "mutating"),
    ("plain_mutating_supervised", _spec(MUT), S, {}, EX, "mutating"),
    ("costly_supervised", _spec(COST), S, {}, APP, "costly"),
    ("no_grant", _spec(COST), A, {"autopilot_max_spend_usd": None}, APP, "no_grant"),
    ("no_estimate", _spec(COST), A, {"estimate_usd": None}, APP, "estimate_unavailable"),
    ("over_cap", _spec(COST), A, {"session_spend_usd": 9.5}, APP, "spend_cap"),
    ("at_cap", _spec(COST), A, {"session_spend_usd": 9.0}, EX, "autopilot"),
    ("guard_refused", _spec(COST), A, {"guard_refused_before": True}, APP, "guard_refused_before"),
    ("costly_autopilot", _spec(COST), A, {}, EX, "autopilot"),
]


@pytest.mark.parametrize(
    ("spec", "mode", "overrides", "action", "reason"),
    [c[1:] for c in CASES],
    ids=[c[0] for c in CASES],
)
def test_gate_rule(spec, mode, overrides, action, reason):
    decision = gate(_inp(spec, mode, **overrides))
    assert (decision.action, decision.reason) == (action, reason)
