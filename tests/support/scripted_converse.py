"""A scripted, offline ``ConverseClient`` for the agent suite.

Each request consumes one ``ScriptStep``. An unexpected or post-exhaustion request raises an
``AssertionError`` that names it, and every request is recorded for later assertions.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from backend.agent.converse import (
    ConverseMessage,
    ConverseTurn,
    TokenUsage,
    ToolDefinition,
    ToolUseRequest,
)


@dataclass(frozen=True, slots=True)
class RecordedRequest:
    system: str
    messages: tuple[ConverseMessage, ...]
    tools: tuple[ToolDefinition, ...]
    max_tokens: int


@dataclass(frozen=True, slots=True)
class ScriptStep:
    """One scripted reply: a turn to return, or an exception to raise.

    ``expect`` is an optional predicate over the ``RecordedRequest``; a false result fails the
    test with an ``AssertionError``.
    """

    turn: ConverseTurn | BaseException
    expect: Callable[[RecordedRequest], bool] | None = None


def text_turn(text: str, *, model_id: str = "scripted-model") -> ConverseTurn:
    return ConverseTurn(
        text=text,
        tool_uses=(),
        stop_reason="end_turn",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        model_id=model_id,
    )


def tool_turn(
    *calls: tuple[str, str, dict[str, Any]], text: str = "", model_id: str = "scripted-model"
) -> ConverseTurn:
    """``calls`` are ``(request_id, tool_name, arguments)`` triples."""
    return ConverseTurn(
        text=text,
        tool_uses=tuple(ToolUseRequest(request_id=i, name=n, arguments=a) for i, n, a in calls),
        stop_reason="tool_use",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        model_id=model_id,
    )


@dataclass
class ScriptedFakeConverseClient:
    steps: Sequence[ScriptStep]
    model_id: str = "scripted-model"
    requests: list[RecordedRequest] = field(default_factory=list)
    _cursor: int = 0

    def converse(
        self,
        *,
        system: str,
        messages: Sequence[ConverseMessage],
        tools: Sequence[ToolDefinition],
        max_tokens: int,
    ) -> ConverseTurn:
        request = RecordedRequest(system, tuple(messages), tuple(tools), max_tokens)
        self.requests.append(request)
        if self._cursor >= len(self.steps):
            raise AssertionError(
                f"unexpected Converse request #{len(self.requests)} after the script of "
                f"{len(self.steps)} step(s) was exhausted"
            )
        step = self.steps[self._cursor]
        self._cursor += 1
        if step.expect is not None and not step.expect(request):
            raise AssertionError(
                f"Converse request #{len(self.requests)} did not satisfy the expectation of "
                f"script step {self._cursor - 1}"
            )
        if isinstance(step.turn, BaseException):
            raise step.turn
        return step.turn

    @property
    def calls(self) -> int:
        return len(self.requests)

    @property
    def remaining(self) -> int:
        return len(self.steps) - self._cursor
