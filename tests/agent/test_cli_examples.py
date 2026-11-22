"""Offline examples for CLI exit codes, bounds, and direct-service wiring."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_factory_cli import __main__ as cli
from ml_factory_cli.transport import InProcessTransport, TransportError


@pytest.mark.parametrize("status, expected", [(None, 3), (404, 4), (403, 4)])
def test_control_plane_and_unknown_session_exit_codes(monkeypatch, capsys, status, expected):
    def fail(_args):
        raise TransportError("unavailable", status=status)

    monkeypatch.setattr(cli, "_run", fail)
    assert cli.main(["list"]) == expected
    assert "unavailable" in capsys.readouterr().err


@pytest.mark.parametrize("interval", [4, 3601])
def test_watch_rejects_interval_outside_bounds(interval):
    with pytest.raises(SystemExit) as caught:
        cli.main(["watch", "ses_" + "a" * 32, "--interval", str(interval)])
    assert caught.value.code == 2


def test_in_process_uses_configured_aws_session(monkeypatch):
    import backend.aws
    import backend.container
    import ml_engine.io

    calls = []
    settings = SimpleNamespace(
        aws_profile="test-profile",
        aws_role_arn="arn:aws:iam::123456789012:role/test",
        s3_sse="AES256",
        s3_kms_key_id=None,
        dev_identity="owner",
    )

    def build_session(given):
        calls.append((given.aws_profile, given.aws_role_arn))
        return SimpleNamespace(client=lambda name: f"fake-{name}")

    authorizer = SimpleNamespace(principal=lambda _name: "owner-principal")
    monkeypatch.setattr(backend.aws, "build_session", build_session)
    monkeypatch.setattr(
        backend.container,
        "build_container",
        lambda *_a, **_kw: SimpleNamespace(agent="agent", authorizer=authorizer),
    )
    monkeypatch.setattr(ml_engine.io, "S3ObjectStore", lambda *_a, **_kw: "fake-store")

    transport = InProcessTransport(settings)
    assert calls == [("test-profile", "arn:aws:iam::123456789012:role/test")]
    assert transport._agent == "agent"
    assert transport._who == "owner-principal"


def test_cli_created_session_can_resume_through_api(price_table_path):
    from backend.auth import Principal
    from ml_engine.contracts.agent import SessionStatus
    from tests.agent.test_agent_routes import OWNER, PLAN_ASK, SID, _call, _env

    env = _env(price_table_path, PLAN_ASK)
    transport = object.__new__(InProcessTransport)
    transport._agent = env.service
    transport._who = Principal(user=OWNER)
    transport._settings = env.container.settings
    view = transport.request("POST", "/agent/sessions", {"goal": "predict churn"})
    sid = view["session_id"]
    env.runner.bind(None)
    with env.service.store.transaction(sid) as session:
        session.status = SessionStatus.FAILED
        session.retriable = True
    status, resumed = _call(env, "POST", SID + "/resume", session_id=sid)
    assert status == 200
    assert resumed["session_id"] == sid
    assert resumed["status"] == "running"


def test_cli_import_boundary():
    root = Path(__file__).resolve().parents[2] / "ml_factory_cli"
    forbidden = {
        "backend.agent.loop",
        "backend.agent.policy",
        "backend.agent.limits",
        "backend.agent.store",
    }
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not forbidden.intersection(imports), path.name
        assert not any(
            alias.name == "boto3"
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ), path.name
        if path.name == "transport.py":
            direct = next(
                node
                for node in tree.body
                if isinstance(node, ast.ClassDef) and node.name == "InProcessTransport"
            )
            assert any(
                isinstance(node, ast.ImportFrom) and node.module == "backend.container"
                for node in ast.walk(direct)
            )
            assert all(
                not (isinstance(node, ast.ImportFrom) and node.module == "backend.container")
                for node in tree.body
            )
