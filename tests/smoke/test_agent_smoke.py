"""Opt-in real-AWS smoke test for the ML Assistant (ml-assistant Req 17, design Smoke_Test).

Runs only with both flags set, because it makes real Bedrock, SageMaker and S3 calls:

    ML_FACTORY_AWS_TESTS=1 ML_FACTORY_AGENT_SMOKE=1 pytest tests/smoke -rs

``ML_FACTORY_AWS_TESTS=1`` is the repository-wide gate for every ``aws`` test (tests/conftest.py);
``ML_FACTORY_AGENT_SMOKE=1`` is the extra, explicit opt-in for this test. Without either flag the
test is skipped before its body runs, so no settings are loaded and no client is built.

It goes through the same code path as ``python -m scripts.agent_smoke``: ``run_smoke(Settings())``.
"""

import os

import pytest

SMOKE_FLAG = "ML_FACTORY_AGENT_SMOKE"


@pytest.mark.aws
@pytest.mark.skipif(
    os.environ.get(SMOKE_FLAG) != "1",
    reason=f"agent smoke test; set {SMOKE_FLAG}=1 (and ML_FACTORY_AWS_TESTS=1) to run",
)
def test_the_agent_smoke_harness_passes_against_real_aws():  # pragma: no cover - opt-in
    from backend.config import Settings
    from scripts.agent_smoke import format_report, run_smoke

    report = run_smoke(Settings())
    print(format_report(report))  # session id, job references and spend, pass or fail

    assert report.passed, format_report(report)
