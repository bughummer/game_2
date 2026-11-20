"""Tool registry for the ML Assistant (design C5).

Defines the tool metadata (``ToolFlags``, ``ToolSpec``), the per-call context and output types,
and the ``ToolRegistry``. Registration is atomic and classifies every tool exactly once.
"""

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import BaseModel

from backend.agent.converse import ToolDefinition
from ml_engine.contracts.agent import ToolClass

# Name prefixes the model must never be able to call (Req 3.7, 4.11).
RESERVED_PRIVILEGED_PREFIXES: tuple[str, ...] = (
    "approve_request",
    "reject_request",
    "edit_request",
    "set_mode",
    "set_limits",
    "enable_autopilot",
    "promote",
    "delete",
    "approve_script",
)


class ToolRegistrationError(ValueError):
    """A tool could not be registered. ``tool`` names the offending tool."""

    def __init__(self, tool: str, message: str) -> None:
        super().__init__(f"cannot register tool {tool!r}: {message}")
        self.tool = tool


@dataclass(frozen=True, slots=True)
class ToolFlags:
    changes_config: bool = False
    always_approve: bool = False
    upload: bool = False
    control: bool = False
    experiment_arg: str | None = None
    dataset_args: tuple[str, ...] = ()
    instance_args: tuple[str, ...] = ()
    verbatim: bool = False
    ledger_source: bool = True
    estimator: str | None = None
    audit_action: str | None = None
    #: The result is a leakage finding, a feature proposal or a model ranking (Req 10.8).
    advisory: bool = False


@dataclass(frozen=True, slots=True)
class AwaitedJobSpec:
    """A job a handler started, as reported by its ``job_extractor``."""

    execution_id: str
    experiment_id: str
    kind: str


@dataclass(slots=True)
class ToolContext:
    """What a handler may use. Identity comes from the session owner only."""

    session: Any
    principal: Any
    services: Any
    clock: Callable[[], datetime] | None = None
    inputs: Any = None


@dataclass(slots=True)
class ToolOutput:
    data: dict[str, Any]
    experiment_ids: tuple[str, ...] = ()
    datasets: tuple[Any, ...] = ()
    awaited: tuple[AwaitedJobSpec, ...] = field(default_factory=tuple)
    #: Detail keys for the experiment audit entry, same keys as the matching route writes.
    audit_detail: dict[str, str] = field(default_factory=dict)
    #: A local file id the client must now upload. The loop keeps the call pending and creates
    #: the session's ``ClientAction``; ``AgentService.complete_upload`` gives the call its result.
    client_upload: str | None = None


ToolHandler = Callable[[ToolContext, Any], ToolOutput]
ProtectedFn = Callable[[ToolContext, Any], bool]
JobExtractor = Callable[[dict], list[AwaitedJobSpec]]
#: Extra content an approval request must show besides the arguments (e.g. script code).
ApprovalViewFn = Callable[[ToolContext, Any], dict[str, Any]]
#: Runs before the gate. ``None``: gate as usual. A ``ToolOutput``: the call succeeds with it and
#: no approval is created. Any other value is a ``ToolError`` the call is refused with.
PreGateFn = Callable[[ToolContext, Any], Any]


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    description: str
    input_model: type[BaseModel]
    tool_class: ToolClass
    handler: ToolHandler
    flags: ToolFlags
    protected: ProtectedFn | None = None
    job_extractor: JobExtractor | None = None
    approval_view: ApprovalViewFn | None = None
    pre_gate: PreGateFn | None = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(
        self,
        name: str,
        *,
        classes: Iterable[ToolClass],
        description: str,
        input_model: type[BaseModel],
        handler: ToolHandler,
        flags: ToolFlags | None = None,
        protected: ProtectedFn | None = None,
        job_extractor: JobExtractor | None = None,
        approval_view: ApprovalViewFn | None = None,
        pre_gate: PreGateFn | None = None,
    ) -> None:
        distinct = set(classes)
        if len(distinct) != 1:
            raise ToolRegistrationError(
                name, f"exactly one tool class is required (got {len(distinct)})"
            )
        (tool_class,) = distinct
        if not isinstance(tool_class, ToolClass):
            try:
                tool_class = ToolClass(tool_class)
            except ValueError:
                raise ToolRegistrationError(name, "unknown tool class") from None
        if not isinstance(name, str) or not name:
            raise ToolRegistrationError(str(name), "name must be a non-empty string")
        if name.startswith(RESERVED_PRIVILEGED_PREFIXES):
            raise ToolRegistrationError(name, "name is reserved for privileged actions")
        if name in self._tools:
            raise ToolRegistrationError(name, "name is already registered")
        self._tools[name] = ToolSpec(
            name=name,
            description=description,
            input_model=input_model,
            tool_class=tool_class,
            handler=handler,
            flags=flags or ToolFlags(),
            protected=protected,
            job_extractor=job_extractor,
            approval_view=approval_view,
            pre_gate=pre_gate,
        )

    def get(self, name: str) -> ToolSpec | None:
        return self._tools.get(name)

    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name=spec.name,
                description=spec.description,
                input_schema=spec.input_model.model_json_schema(),
            )
            for spec in self._tools.values()
        ]
