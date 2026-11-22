"""Offline checks for the opt-in pytest gate, smoke budget and report."""
from decimal import Decimal
from pathlib import Path
from runpy import run_path

import pytest

from backend.config import Settings
from scripts import agent_smoke as smoke


def test_smoke_test_is_skipped_without_its_flag(monkeypatch):
    monkeypatch.delenv("ML_FACTORY_AGENT_SMOKE", raising=False)
    namespace = run_path(str(Path(__file__).parents[1] / "smoke" / "test_agent_smoke.py"))
    test = namespace["test_the_agent_smoke_harness_passes_against_real_aws"]
    assert any(mark.name == "skipif" and mark.args == (True,) for mark in test.pytestmark)
    assert any(mark.name == "aws" for mark in test.pytestmark)


@pytest.mark.parametrize("spend,expected", [(3, Decimal("1.5")), (12, Decimal("5"))])
def test_smoke_limits_are_halved_and_spend_capped(spend, expected):
    base = Settings().model_copy(
        update={**dict.fromkeys(smoke.SMOKE_INT_LIMITS, 7), smoke.SMOKE_SPEND_LIMIT: spend}
    )
    result = smoke.smoke_settings(base)
    assert all(getattr(result, name) == 3 for name in smoke.SMOKE_INT_LIMITS)
    assert getattr(result, smoke.SMOKE_SPEND_LIMIT) == expected


def test_report_identifies_failure_session_jobs_and_spend():
    report = smoke.SmokeReport(
        [smoke.CheckResult("eda", False, "timed out")], "sid", ["eda:job-1 (failed)"], 0.75
    )
    lines = smoke.format_report(report).splitlines()
    assert {
        "  [FAIL] eda: timed out",
        "Result: FAILED at eda",
        "Session id: sid",
        "  eda:job-1 (failed)",
        "Estimated spend: 0.75 USD",
    } <= set(lines)
