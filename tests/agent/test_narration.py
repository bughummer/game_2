"""Narration verifier and final answer (task 11.1, design C9, Req 10)."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import BaseModel, ConfigDict, Field

from backend.agent import tools_free_form
from backend.agent.execute import ToolExecutor
from backend.agent.inputs import ToolInputs
from backend.agent.loop import AgentLoop
from backend.agent.narration import (
    ADVISORY_LABEL,
    OVERCLAIM_MARKER,
    UNVERIFIED_MARKER,
    NumberLedger,
    ToolResult,
    build_final_answer,
    verify_narration,
)
from backend.agent.results import _ledger
from backend.agent.store import SessionStore
from backend.agent.tools import ToolFlags, ToolOutput, ToolRegistry
from backend.agent.view import build_view
from backend.governance import GovernancePolicy
from ml_engine.contracts.agent import (
    ApprovalStatus,
    CallState,
    Session,
    SessionLimits,
    SessionStatus,
    ToolClass,
)
from ml_engine.contracts.free_form import content_hash
from ml_engine.reasoning.numbers import drop_unverified_numbers
from tests.agent.strategies import SETTINGS
from tests.support.agent_fakes import FakeClock
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.scripted_converse import (
    ScriptedFakeConverseClient,
    ScriptStep,
    text_turn,
    tool_turn,
)

ROOT = "s3://bucket-test/root"
SID = "sess-n"
T0 = datetime(2025, 1, 1, tzinfo=UTC)


def _ledger_of(full: dict, call_id: str = "tu-1") -> NumberLedger:
    return NumberLedger(_ledger(full, call_id, "exp-1"))


# ----------------------------------------------------------------------------- verification


def test_verified_and_unverified_sentences_mark_policy():
    ledger = _ledger_of({"auc": 0.8123})
    out = verify_narration("AUC is 0.81. Accuracy is 0.99. No numbers here.", ledger)
    assert [(s.text, s.status) for s in out.sentences] == [
        ("AUC is 0.81.", "verified"),
        ("Accuracy is 0.99.", "unverified"),
        ("No numbers here.", "verified"),
    ]
    assert [s.unmatched for s in out.unverified] == [("0.99",)]
    answer = build_final_answer(out)
    assert answer.sentences[0].text == "AUC is 0.81."
    assert answer.sentences[1].text == UNVERIFIED_MARKER + "Accuracy is 0.99."


def test_withhold_policy_removes_only_unverified_sentences():
    ledger = _ledger_of({"auc": 0.8123})
    out = verify_narration("AUC is 0.81. Accuracy is 0.99.", ledger, policy="withhold")
    assert [s.text for s in out.sentences] == ["AUC is 0.81."]
    assert [s.text for s in out.withheld] == ["Accuracy is 0.99."]
    assert len(out.unverified) == 1
    answer = build_final_answer(out)
    assert [s.text for s in answer.sentences] == ["AUC is 0.81."]
    assert any("withheld" in n for n in answer.notes)


@pytest.mark.parametrize(
    ("text", "value", "verified"),
    [
        ("Recall is 12%.", 0.12, True),  # percent matches its fraction
        ("Recall is 12.5%.", 0.125, True),
        ("Recall is 12%.", 12.0, True),  # or its written value
        ("Recall is 13%.", 0.12, False),
        ("Score 0.80.", 0.7987, True),  # token precision
        ("Score 0.81.", 0.7987, False),
        ("Score 1.", 0.6, True),
    ],
)
def test_matching_agrees_with_numbers_py(text, value, verified):
    out = verify_narration(text, _ledger_of({"v": value}))
    assert (out.sentences[0].status == "verified") is verified
    assert (drop_unverified_numbers(text, {"v": value}) == text) is verified


# ----------------------------------------------------------------------------- final answer


def test_attributions_carry_call_id_and_path_and_free_form_label_beside_number():
    full = {"metrics": [{"name": "auc", "value": 0.734, "label": "self_reported"}], "auc": 0.912}
    ledger = _ledger_of(full, "tu-ff")
    out = verify_narration("Free-form AUC 0.73 versus managed 0.91.", ledger)
    answer = build_final_answer(out, [ToolResult("tu-ff", "get_free_form_run", 2, full)])
    sentence = answer.sentences[0]
    assert sentence.status == "verified"
    assert sentence.text == "Free-form AUC 0.73 (self_reported) versus managed 0.91."
    by_value = {a.value_text: a for a in sentence.attributions}
    assert by_value["0.73"].tool_call_id == "tu-ff"
    assert by_value["0.73"].path == "metrics[0].value"
    assert by_value["0.73"].metric_label == "self_reported"
    assert by_value["0.91"].path == "auc" and by_value["0.91"].metric_label is None


# Feature: ml-assistant, Property 41: A final answer carries attributions, labels and verbatim blocks
@settings(**SETTINGS)
@given(values=st.lists(st.integers(1, 1000), min_size=1, max_size=20))
def test_verbatim_block_deep_equals_stored_full_result(values):
    full = {"rows_omitted": 0, "table": [{"model": "xgb", "auc": 0.9, "nested": {"a": values}}]}
    result = ToolResult("tu-c", "get_report", 3, full, verbatim=True)
    answer = build_final_answer(
        verify_narration(f"Value {values[0]}.", _ledger_of(full, "tu-c")), [result]
    )
    assert any(
        a.tool_call_id == "tu-c" and a.path == "table[0].nested.a[0]"
        for a in answer.sentences[0].attributions
    )
    assert [b.content for b in answer.verbatim] == [full]
    assert answer.verbatim[0].tool_call_id == "tu-c"
    assert answer.verbatim[0].content is not full  # a copy, not the caller's object


# Feature: ml-assistant, Property 42: Comparisons and advisory results are described honestly
@settings(**SETTINGS)
@given(
    verb=st.sampled_from(["is better than", "beats", "outperforms", "ranks higher than"]),
    run=st.from_regex(r"trial-[a-z]{2,8}", fullmatch=True),
)
def test_not_comparable_note_and_ranking_sentence_dropped(verb, run):
    full = {
        "entries": [
            {"name": "xgb", "primary_score": 0.9, "metric_label": "engine", "comparable": True},
            {
                "name": run + ":auc",
                "run_id": run,
                "value": 0.95,
                "metric_label": "self_reported",
                "comparable": False,
            },
        ]
    }
    ledger = _ledger_of(full, "tu-cmp")
    text = f"{run} {verb} xgb. Both runs finished."
    result = ToolResult("tu-cmp", "compare_free_form", 4, full, verbatim=True, advisory=True)
    answer = build_final_answer(verify_narration(text, ledger), [result])
    assert [s.text for s in answer.sentences] == ["Both runs finished."]
    assert any(run + ":auc" in n and "not comparable" in n for n in answer.notes)
    assert answer.labels == [ADVISORY_LABEL]


# Feature: ml-assistant, Property 42: Comparisons and advisory results are described honestly
@settings(**SETTINGS)
@given(
    certainty=st.sampled_from(["guaranteed", "certain", "proven", "definitely"]),
    column=st.from_regex(r"[a-z]{1,8}", fullmatch=True),
)
def test_advisory_label_and_overclaim(certainty, column):
    full = {"findings": [{"column": "refund_flag", "risk": 0.9}]}
    result = ToolResult("tu-l", "get_leakage_findings", 2, full, advisory=True)
    text = f"The leakage in {column} is {certainty}. Leakage risk is high."
    answer = build_final_answer(verify_narration(text, _ledger_of(full, "tu-l")), [result])
    assert answer.labels == [ADVISORY_LABEL]
    assert [s.status for s in answer.sentences] == ["overclaim", "verified"]
    assert answer.sentences[0].text.startswith(OVERCLAIM_MARKER)


def test_no_advisory_label_without_advisory_results():
    answer = build_final_answer(
        verify_narration("Done.", NumberLedger()), [ToolResult("tu-x", "get_status", 1, {})]
    )
    assert answer.labels == []


# ----------------------------------------------------------------------------- loop wiring


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PlanArgs(_Strict):
    steps: list[str] = Field(min_length=1, max_length=10)


class NoArgs(_Strict):
    pass


class _Audit:
    def __init__(self) -> None:
        self.records: list = []

    def write(self, session, rec):
        self.records.append(rec)
        session.audit_seq += 1
        return f"a-{session.audit_seq}"


class _Authorizer:
    def require_owner(self, principal, experiment_id):
        return None


SCRIPT = "import pandas as pd\nprint('hello')\n"


class _FreeForm:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.started = False

    def draft_revision(self, experiment_id, draft_id, revision):
        if self.fail:
            raise RuntimeError("store down")
        rev = SimpleNamespace(
            revision=revision, content=SCRIPT, content_hash=content_hash(SCRIPT), assumptions=["a"]
        )
        return SimpleNamespace(dataset_uri="s3://b/d.csv", data_context=None), rev

    def start_run(self, *a, **k):  # pragma: no cover - must never be reached here
        self.started = True
        raise AssertionError("start_run must not run without approval")


def _harness(steps, *, free_form=None):
    registry = ToolRegistry()
    registry.register(
        "submit_plan",
        classes=[ToolClass.READ],
        description="plan",
        input_model=PlanArgs,
        handler=lambda ctx, a: ToolOutput(data={"steps": a.steps}),
        flags=ToolFlags(control=True, ledger_source=False),
    )
    registry.register(
        "get_metrics",
        classes=[ToolClass.READ],
        description="metrics",
        input_model=NoArgs,
        handler=lambda ctx, a: ToolOutput(data={"auc": 0.8123, "table": [{"f1": 0.66}]}),
        flags=ToolFlags(verbatim=True),
    )
    registry.register(
        "get_hidden",
        classes=[ToolClass.READ],
        description="not a ledger source",
        input_model=NoArgs,
        handler=lambda ctx, a: ToolOutput(data={"secret_score": 0.4242}),
        flags=ToolFlags(ledger_source=False),
    )
    tools_free_form.register(registry, None)
    objects = InMemoryObjectStore()
    clock = FakeClock(T0)
    store = SessionStore(objects, ROOT, clock=clock)
    store.create(
        Session(
            session_id=SID,
            owner="alice",
            created_at=T0,
            updated_at=T0,
            last_activity_at=T0,
            status=SessionStatus.RUNNING,
            limits=SessionLimits(
                max_converse_turns=10,
                max_tool_calls=20,
                max_costly_tool_calls=2,
                max_estimated_spend_usd=5.0,
                max_active_seconds=600,
                max_repeated_identical_calls=3,
            ),
            goal="g",
            experiment_ids=["exp-1"],
        )
    )
    audit = _Audit()
    services = SimpleNamespace(free_form=free_form or _FreeForm())
    inputs = ToolInputs(audit=audit, authorizer=_Authorizer(), allowed_instance_types=[])
    loop = AgentLoop(
        store=store,
        converse=ScriptedFakeConverseClient(steps=[ScriptStep(s) for s in steps]),
        registry=registry,
        executor=ToolExecutor(inputs=inputs, services=services, clock=clock),
        inputs=inputs,
        audit=audit,
        governance=GovernancePolicy(
            send_sample_values=False, sample_values_per_column=0, sample_value_max_chars=0
        ),
        services=services,
        clock=clock,
    )
    loop.run(SID, "holder")
    return store.load(SID).session, audit, store


PLAN = ("tu-plan", "submit_plan", {"steps": ["look"]})


def test_loop_builds_ledger_from_stored_results_of_ledger_sources():
    s, audit, store = _harness(
        [
            tool_turn(PLAN),
            tool_turn(("tu-m", "get_metrics", {}), ("tu-h", "get_hidden", {})),
            text_turn("AUC is 0.81. F1 is 0.66. Hidden is 0.42."),
        ]
    )
    assert s.status is SessionStatus.COMPLETED
    answer = s.final_answer
    assert [x.status for x in answer.sentences] == ["verified", "verified", "unverified"]
    assert answer.sentences[0].attributions[0].tool_call_id == "tu-m"
    assert answer.sentences[0].attributions[0].path == "auc"
    assert answer.sentences[1].attributions[0].path == "table[0].f1"
    record = next(r for r in s.tool_calls if r.tool_call_id == "tu-m")
    assert [b.content for b in answer.verbatim] == [store.get_result(SID, record.result_ref)]
    unverified = [r for r in audit.records if r.event_type == "narration_unverified"]
    assert [r.detail["unmatched"] for r in unverified] == ["0.42"]


START = {"experiment_id": "exp-1", "draft_id": "draft-abc", "revision": 2}


def test_free_form_approval_shows_code_and_hash():
    s, _audit, _store = _harness([tool_turn(PLAN), tool_turn(("tu-ff", "start_free_form_run", START))])
    assert s.status is SessionStatus.WAITING_FOR_APPROVAL
    (approval,) = [a for a in s.approvals if a.status is ApprovalStatus.PENDING]
    assert approval.review["content"] == SCRIPT
    assert approval.review["content_hash"] == content_hash(SCRIPT)
    assert approval.review["revision"] == 2
    view = build_view(s)
    assert view.pending_approvals[0].review["content_hash"] == content_hash(SCRIPT)
    assert "content" in view.pending_approvals[0].review


def test_free_form_approval_fails_closed_when_code_cannot_be_loaded():
    free_form = _FreeForm(fail=True)
    s, _audit, _store = _harness(
        [
            tool_turn(PLAN),
            tool_turn(("tu-ff", "start_free_form_run", START)),
            text_turn("It could not start."),
        ],
        free_form=free_form,
    )
    record = next(r for r in s.tool_calls if r.tool_call_id == "tu-ff")
    assert record.state is CallState.FAILED
    assert record.error_code == "APPROVAL_VIEW_FAILED"
    assert s.approvals == []
    assert not free_form.started


# Feature: ml-assistant, Property 40: Narration is verified against the ledger
@settings(**SETTINGS)
@given(value=st.integers(1, 10000), delta=st.integers(-1, 1), percent=st.booleans())
def test_numbers_match_at_displayed_precision(value, delta, percent):
    displayed = value + delta
    text = f"Measured {displayed}{'%' if percent else ''}."
    raw = {"metric": value / 100 if percent else value}
    verified = drop_unverified_numbers(text, raw) == text
    answer = build_final_answer(verify_narration(text, _ledger_of(raw)))
    sentence = answer.sentences[0]
    assert (sentence.status == "verified") == verified
    assert sentence.text == (text if verified else UNVERIFIED_MARKER + text)
