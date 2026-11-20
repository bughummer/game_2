"""Approval gate for the ML Assistant (design C6). Pure and total.

Rules 1 to 8 from the design, first match wins. Anything not matched fails closed.
"""

from dataclasses import dataclass
from enum import StrEnum

from backend.agent.tools import ToolSpec
from ml_engine.contracts.agent import Mode, ToolClass


class GateAction(StrEnum):
    EXECUTE = "execute"
    REQUIRE_APPROVAL = "require_approval"
    REFUSE = "refuse"


@dataclass(frozen=True, slots=True)
class GateInput:
    spec: ToolSpec
    mode: Mode
    plan_accepted: bool
    protected: bool
    estimate_usd: float | None
    session_spend_usd: float
    autopilot_max_spend_usd: float | None
    guard_refused_before: bool
    upload_approved_before: bool


@dataclass(frozen=True, slots=True)
class GateDecision:
    action: GateAction
    reason: str


def _approval(reason: str) -> GateDecision:
    return GateDecision(GateAction.REQUIRE_APPROVAL, reason)


def gate(inp: GateInput) -> GateDecision:
    """Decide whether a tool call executes, needs approval, or is refused.

    Rules are evaluated in the design's order; the first match wins. The decision depends
    only on ``inp`` and never on model text.
    """
    flags = inp.spec.flags
    tool_class = inp.spec.tool_class
    # 1. Plan required for every non-control tool (Req 1.3).
    if not flags.control and not inp.plan_accepted:
        return GateDecision(GateAction.REFUSE, "plan_required")
    # 2. Read tools execute in both modes (Req 4.4).
    if tool_class == ToolClass.READ:
        return GateDecision(GateAction.EXECUTE, "read")
    # 3. Free-form runs need approval in every mode.
    if flags.always_approve:
        return _approval("script_run")
    # 4. First upload in the session needs approval in both modes (Req 7.4).
    if flags.upload and not inp.upload_approved_before:
        return _approval("first_upload")
    # 5. Protected actions need approval in both modes (Req 5.7, 5.8).
    if inp.protected:
        return _approval("protected_action")
    # 6. Mutating: config changes need approval in Supervised (Req 4.3).
    if tool_class == ToolClass.MUTATING:
        if flags.changes_config and inp.mode != Mode.AUTOPILOT:
            return _approval("config_change")
        return GateDecision(GateAction.EXECUTE, "mutating")
    if tool_class == ToolClass.COSTLY:
        # 7. Costly in Supervised (Req 4.2).
        if inp.mode != Mode.AUTOPILOT:
            return _approval("costly")
        # 8. Costly in Autopilot (Req 5.5, 5.6, D12).
        if inp.autopilot_max_spend_usd is None:
            return _approval("no_grant")
        if inp.estimate_usd is None:
            return _approval("estimate_unavailable")
        if inp.session_spend_usd + inp.estimate_usd > inp.autopilot_max_spend_usd:
            return _approval("spend_cap")
        if inp.guard_refused_before:
            return _approval("guard_refused_before")
        return GateDecision(GateAction.EXECUTE, "autopilot")
    # Unknown class: fail closed.
    return _approval("default_require_approval")
