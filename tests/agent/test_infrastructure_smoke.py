"""Smoke checks that the agent test infrastructure imports and behaves."""

import json
import socket
from pathlib import Path

import pytest
from hypothesis import given, settings

from backend.agent.converse import ConverseMessage, TextBlock
from tests.agent import strategies as st_
from tests.support.agent_fakes import CountingStore, FakeClock, FlakyStore, InlineAgentRunner
from tests.support.in_memory_store import InMemoryObjectStore
from tests.support.scripted_converse import ScriptedFakeConverseClient, ScriptStep, text_turn

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "converse"


def _call(client):
    return client.converse(
        system="s", messages=[ConverseMessage("user", (TextBlock("hi"),))], tools=[], max_tokens=10
    )


def test_scripted_client_records_and_raises_after_exhaustion():
    client = ScriptedFakeConverseClient([ScriptStep(text_turn("ok"))])
    assert _call(client).text == "ok"
    assert client.calls == 1
    with pytest.raises(AssertionError, match="exhausted"):
        _call(client)


def test_scripted_client_names_failed_expectation():
    client = ScriptedFakeConverseClient([ScriptStep(text_turn("x"), expect=lambda r: False)])
    with pytest.raises(AssertionError, match="expectation"):
        _call(client)


def test_flaky_and_counting_store_and_clock():
    flaky = FlakyStore(InMemoryObjectStore(), fail_writes=1)
    with pytest.raises(OSError):
        flaky.write_bytes("s3://bucket-test/k", b"x")
    flaky.write_bytes("s3://bucket-test/k", b"x")
    assert flaky.read_bytes("s3://bucket-test/k") == b"x"
    assert flaky.writes == 2
    assert CountingStore(InMemoryObjectStore()).writes == 0
    clock = FakeClock()
    start = clock.now()
    clock.sleep(2.0)
    assert clock.sleeps == [2.0] and (clock.now() - start).total_seconds() == 2.0
    assert InlineAgentRunner is not None


def test_sockets_are_blocked():
    with pytest.raises(RuntimeError, match="blocked"):
        socket.create_connection(("example.com", 80))


@pytest.mark.parametrize(
    "name", ["text_turn", "tool_use_turn", "throttling", "unsupported_model", "expired_credentials"]
)
def test_recorded_fixtures_are_valid_json(name):
    assert isinstance(json.loads((FIXTURES / f"{name}.json").read_text("utf-8")), dict)


@settings(max_examples=100, deadline=None)
@given(st_.run_scripts, st_.trees_with_injected_keys(), st_.credentials, st_.s3_uris)
def test_strategies_generate(script, tree, credential, uri):
    assert script and tree and credential and uri.startswith("s3://")


def test_agent_service_factory_collects(agent_service_factory):
    assert callable(agent_service_factory)
