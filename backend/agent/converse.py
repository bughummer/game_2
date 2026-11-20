"""Bedrock Converse tool-use adapter.

A new client beside ``BedrockReasoningClient`` (which is untouched). The conversation types are
provider-neutral; this module maps them to and from the Bedrock Converse wire format. boto3 is
never imported here: the client is built lazily through ``backend.aws.build_client``.

Nothing is logged beyond the error class, the model id and the Bedrock error code. Exception
text, requests and credentials are never logged.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

LOGGER = logging.getLogger("ml_factory.agent.converse")

StopReason = Literal["end_turn", "tool_use", "max_tokens", "other"]

TRANSIENT_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
        "ModelNotReadyException",
    }
)
CREDENTIAL_CODES = frozenset({"ExpiredTokenException", "ExpiredToken", "UnrecognizedClientException"})
# botocore exception class names for timeouts and connection failures (matched by name so that
# botocore is not imported at module level).
TRANSIENT_CLASS_NAMES = frozenset(
    {
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "ConnectionClosedError",
        "EndpointConnectionError",
        "ConnectionError",
        "SSLError",
        "ProxyConnectionError",
        "TimeoutError",
    }
)


# ----- neutral turn types -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolUseRequest:
    request_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class ConverseTurn:
    text: str
    tool_uses: tuple[ToolUseRequest, ...]
    stop_reason: StopReason
    usage: TokenUsage
    model_id: str


# ----- neutral conversation types -----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TextBlock:
    text: str


@dataclass(frozen=True, slots=True)
class ToolUseBlock:
    request_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolResultBlock:
    request_id: str
    content: dict[str, Any]  # already enveloped by the results pipeline


Block = TextBlock | ToolUseBlock | ToolResultBlock


@dataclass(frozen=True, slots=True)
class ConverseMessage:
    role: Literal["user", "assistant"]
    blocks: tuple[Block, ...]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]


# ----- errors -------------------------------------------------------------------------------


class ConverseError(RuntimeError):
    retriable: bool = False

    def __init__(self, message: str = "", *, code: str = "") -> None:
        super().__init__(message)
        self.code = code


class ConverseTransientError(ConverseError):
    """Throttling, 5xx, timeout or connection failure."""

    retriable = True


class ConverseModelUnsupportedError(ConverseError):
    """The configured model does not support Converse tool use."""

    retriable = False

    def __init__(self, model_id: str, reason: str, *, code: str = "") -> None:
        super().__init__(f"model {model_id} does not support tool use: {reason}", code=code)
        self.model_id = model_id
        self.reason = reason


class ConverseRequestError(ConverseError):
    """Invalid request or access denied."""

    retriable = False


class ConverseCredentialsError(ConverseError):
    """Expired or unrecognized credentials."""

    retriable = True  # the session can be resumed once credentials are refreshed


class ConverseClient(Protocol):
    model_id: str

    def converse(
        self,
        *,
        system: str,
        messages: Sequence[ConverseMessage],
        tools: Sequence[ToolDefinition],
        max_tokens: int,
    ) -> ConverseTurn: ...


# ----- wire mapping -------------------------------------------------------------------------


def _block_to_wire(block: Block) -> dict[str, Any]:
    if isinstance(block, TextBlock):
        return {"text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "toolUse": {
                "toolUseId": block.request_id,
                "name": block.name,
                "input": block.arguments,
            }
        }
    return {
        "toolResult": {
            "toolUseId": block.request_id,
            "content": [{"json": block.content}],
        }
    }


def build_request(
    model_id: str,
    system: str,
    messages: Sequence[ConverseMessage],
    tools: Sequence[ToolDefinition],
    max_tokens: int,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "modelId": model_id,
        "system": [{"text": system}],
        "messages": [
            {"role": m.role, "content": [_block_to_wire(b) for b in m.blocks]} for m in messages
        ],
        "inferenceConfig": {"temperature": 0, "maxTokens": max_tokens},
    }
    if tools:
        request["toolConfig"] = {
            "tools": [
                {
                    "toolSpec": {
                        "name": t.name,
                        "description": t.description,
                        "inputSchema": {"json": t.input_schema},
                    }
                }
                for t in tools
            ]
        }
    return request


def parse_response(response: dict[str, Any], model_id: str) -> ConverseTurn:
    message = (response.get("output") or {}).get("message") or {}
    texts: list[str] = []
    uses: list[ToolUseRequest] = []
    for block in message.get("content") or []:
        if "text" in block:
            texts.append(str(block["text"]))
        elif "toolUse" in block:
            use = block["toolUse"]
            arguments = use.get("input")
            uses.append(
                ToolUseRequest(
                    request_id=str(use.get("toolUseId", "")),
                    name=str(use.get("name", "")),
                    arguments=arguments if isinstance(arguments, dict) else {},
                )
            )
    raw_stop = response.get("stopReason")
    stop: StopReason = raw_stop if raw_stop in ("end_turn", "tool_use", "max_tokens") else "other"
    usage = response.get("usage") or {}
    return ConverseTurn(
        text="".join(texts),
        tool_uses=tuple(uses),
        stop_reason=stop,
        usage=TokenUsage(
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
        ),
        model_id=model_id,
    )


# ----- error classification -----------------------------------------------------------------


def _error_code(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str((response.get("Error") or {}).get("Code", ""))
    return ""


def _error_message(exc: BaseException) -> str:
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        return str((response.get("Error") or {}).get("Message", ""))
    return ""


def _looks_unsupported(message: str) -> bool:
    text = message.lower()
    if "toolconfig" in text:
        return True
    return "tool" in text and ("not support" in text or "doesn't support" in text or "unsupported" in text)


def classify_error(exc: BaseException, model_id: str) -> ConverseError:
    """Map a boto/botocore exception to a Converse error. Never carries the exception text."""
    code = _error_code(exc)
    if code in CREDENTIAL_CODES:
        return ConverseCredentialsError("credentials expired or not recognized", code=code)
    if code in TRANSIENT_CODES:
        return ConverseTransientError("transient Bedrock error", code=code)
    if code == "ValidationException" and _looks_unsupported(_error_message(exc)):
        return ConverseModelUnsupportedError(model_id, "tool use is not supported", code=code)
    if code:
        return ConverseRequestError("Bedrock rejected the request", code=code)
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & TRANSIENT_CLASS_NAMES:
        return ConverseTransientError("transient connection error", code=type(exc).__name__)
    return ConverseRequestError("Bedrock call failed", code=type(exc).__name__)


# ----- client -------------------------------------------------------------------------------


class BedrockConverseClient:
    def __init__(
        self,
        client: Any = None,
        *,
        model_id: str,
        retries: int,
        backoff_seconds: float,
        sleep: Callable[[float], None] = time.sleep,
        client_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._client = client
        self._factory = client_factory
        self.model_id = model_id
        self._retries = max(0, retries)
        self._backoff = backoff_seconds
        self._sleep = sleep

    @classmethod
    def from_settings(
        cls, settings: Any, *, sleep: Callable[[float], None] = time.sleep
    ) -> "BedrockConverseClient":
        """Build a client whose boto3 client is created on first use."""

        def factory() -> Any:
            from backend.aws import build_client

            return build_client(
                "bedrock-runtime",
                settings,
                read_timeout=settings.agent_converse_read_timeout_seconds,
            )

        return cls(
            model_id=settings.effective_agent_model_id,
            retries=settings.agent_converse_retries,
            backoff_seconds=settings.agent_converse_backoff_seconds,
            sleep=sleep,
            client_factory=factory,
        )

    def _boto(self) -> Any:
        if self._client is None:
            if self._factory is None:
                raise ConverseRequestError("no Bedrock client configured")
            self._client = self._factory()
        return self._client

    def converse(
        self,
        *,
        system: str,
        messages: Sequence[ConverseMessage],
        tools: Sequence[ToolDefinition],
        max_tokens: int,
    ) -> ConverseTurn:
        request = build_request(self.model_id, system, messages, tools, max_tokens)
        attempt = 0
        while True:
            try:
                response = self._boto().converse(**request)
                return parse_response(response, self.model_id)
            except ConverseError:
                raise
            except Exception as exc:  # noqa: BLE001 - classified, then re-raised sanitised
                error = classify_error(exc, self.model_id)
                LOGGER.warning(
                    "Converse call failed: %s model=%s code=%s",
                    type(error).__name__,
                    self.model_id,
                    error.code,
                )
                if not isinstance(error, ConverseTransientError) or attempt >= self._retries:
                    raise error from None
                self._sleep(self._backoff * 2**attempt)
                attempt += 1
