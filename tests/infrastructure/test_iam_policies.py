"""IAM policy invariants (requirement 15.3): least privilege, no wildcards, no endpoints.

These are account-independent checks on the policy documents. Deploy-time validation (cfn-lint,
the IAM policy simulator) is separate and not part of the offline suite.
"""

import json
from pathlib import Path

import pytest

IAM_DIR = Path(__file__).resolve().parents[2] / "infra" / "iam"
POLICIES = sorted(IAM_DIR.glob("*_policy.json"))
#: Actions that would let the platform create inference endpoints or models (requirement 4.6).
FORBIDDEN_ACTIONS = {
    "sagemaker:CreateEndpoint",
    "sagemaker:CreateEndpointConfig",
    "sagemaker:CreateModel",
    "sagemaker:CreateTransformJob",
    "sagemaker:*",
    "s3:*",
    "iam:*",
    "*",
}


def _statements(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["Statement"]


@pytest.mark.parametrize("path", POLICIES, ids=lambda p: p.name)
def test_policies_are_valid_json_with_statements(path):
    assert _statements(path), f"{path.name} has no statements"


@pytest.mark.parametrize("path", POLICIES, ids=lambda p: p.name)
def test_no_statement_grants_a_forbidden_or_wildcard_action(path):
    for statement in _statements(path):
        actions = statement["Action"]
        actions = [actions] if isinstance(actions, str) else actions
        offenders = set(actions) & FORBIDDEN_ACTIONS
        assert not offenders, f"{path.name} grants {offenders}"


#: Actions AWS does not support with resource-level permissions: they only work with
#: ``Resource: "*"``. A bare wildcard is allowed for these and nothing else.
NO_RESOURCE_LEVEL_ACTIONS = {
    "ecr:GetAuthorizationToken",
    "cloudwatch:PutMetricData",
    "ec2:DescribeNetworkInterfaces",
    "ec2:DescribeVpcs",
    "ec2:DescribeSubnets",
    "ec2:DescribeSecurityGroups",
    "ec2:DescribeDhcpOptions",
    "sagemaker:ListTrainingJobs",  # read-only; the platform-wide training-job cap
}


@pytest.mark.parametrize("path", POLICIES, ids=lambda p: p.name)
def test_no_resource_is_a_bare_wildcard(path):
    """A bare '*' resource only for actions that cannot be scoped any other way."""
    for statement in _statements(path):
        resources = statement["Resource"]
        resources = [resources] if isinstance(resources, str) else resources
        if "*" not in resources:
            continue
        actions = statement["Action"]
        actions = {actions} if isinstance(actions, str) else set(actions)
        unscopable = actions - NO_RESOURCE_LEVEL_ACTIONS
        assert not unscopable, f"{path.name} grants {unscopable} on a bare '*' resource"


def test_the_job_role_can_pull_its_image():
    """Without these, every SageMaker job fails before the container starts."""
    actions = {
        a
        for s in _statements(IAM_DIR / "sagemaker_job_policy.json")
        for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])
    }
    assert {
        "ecr:GetAuthorizationToken",
        "ecr:BatchGetImage",
        "ecr:GetDownloadUrlForLayer",
    } <= actions


def test_passrole_is_scoped_to_sagemaker_only():
    for path in POLICIES:
        for statement in _statements(path):
            actions = statement["Action"]
            actions = [actions] if isinstance(actions, str) else actions
            if "iam:PassRole" in actions:
                condition = statement.get("Condition", {})
                passed_to = condition.get("StringEquals", {}).get("iam:PassedToService")
                assert passed_to == "sagemaker.amazonaws.com", f"{path.name}: PassRole not scoped"


def test_the_job_role_cannot_pass_a_role_or_submit_jobs():
    """The job role runs inside a stage; it must not be able to start more jobs or pass roles."""
    statements = _statements(IAM_DIR / "sagemaker_job_policy.json")
    all_actions = {
        a
        for s in statements
        for a in ([s["Action"]] if isinstance(s["Action"], str) else s["Action"])
    }
    assert not any(a.startswith("sagemaker:Create") for a in all_actions)
    assert "iam:PassRole" not in all_actions
