"""Execution shell for agent tool calls (design C5 shell rules; Req 3.4, 3.9, 3.10, 7.3, 7.12,
8.10, 14.4).

``ToolExecutor.execute`` is the one place a handler runs. It:

1. takes identity only from ``Principal(user=session.owner)``; no argument can set it;
2. validates the arguments against the tool's input model, then runs ``ToolInputs.check``;
3. replaces dataset ids with the session's URIs, so the model never supplies a URI;
4. runs the handler and maps ``AppError``, ``GuardViolation`` and ``OrchestrationError`` to
   structured errors. Any other exception becomes a generic ``TOOL_FAILED`` naming only the tool;
5. writes the experiment audit entry named by ``flags.audit_action`` through ``AuditLog.write`` and
   returns its URI as ``service_audit_ref``.

A failed experiment audit write does not discard the handler's result. It is reported as
``audit_failed`` so the loop can stop further Mutating and Costly tools (Req 11.9).
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from backend.agent.inputs import InputRejection, ToolInputs
from backend.agent.tools import ToolContext, ToolOutput, ToolSpec
from backend.audit import AuditWriteError
from backend.auth import Principal
from backend.errors import AppError
from backend.orchestration.guards import GuardViolation
from backend.orchestration.sagemaker import OrchestrationError
from ml_engine.contracts.agent import Session

LOGGER = logging.getLogger("ml_factory.agent.execute")

CODE_TOOL_FAILED = "TOOL_FAILED"
CODE_INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
CODE_INPUT_REJECTED = "INPUT_REJECTED"


@dataclass(frozen=True, slots=True)
class ToolError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": dict(self.details)}


def invalid_arguments_error(exc: PydanticValidationError) -> ToolError:
    """The ``INVALID_ARGUMENTS`` error naming each argument that failed the input model."""
    fields = sorted({".".join(str(p) for p in e["loc"]) for e in exc.errors()})
    return ToolError(
        CODE_INVALID_ARGUMENTS, "The arguments are invalid for this tool.", {"fields": fields}
    )


def input_rejected_error(rejections: Any) -> ToolError:
    """The ``INPUT_REJECTED`` error listing each argument ``ToolInputs.check`` refused."""
    return ToolError(
        CODE_INPUT_REJECTED,
        "One or more arguments were rejected.",
        {
            "arguments": [
                {"argument": r.argument, "reason": r.reason, "message": r.message}
                for r in rejections
            ]
        },
    )


@dataclass(slots=True)
class ExecutionResult:
    """Outcome of one call. Exactly one of ``output`` and ``error`` is set."""

    output: ToolOutput | None = None
    error: ToolError | None = None
    #: ``guard`` for GuardViolation and OrchestrationError, so Autopilot avoids retrying.
    error_kind: str | None = None
    rejections: tuple[InputRejection, ...] = ()
    service_audit_ref: str | None = None
    #: True when the experiment audit write failed after the handler ran.
    audit_failed: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


class ToolExecutor:
    def __init__(
        self,
        *,
        inputs: ToolInputs,
        services: Any,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._inputs = inputs
        self._services = services
        self._clock = clock or (lambda: datetime.now(UTC))

    def execute(
        self,
        session: Session,
        spec: ToolSpec,
        arguments: dict[str, Any],
        *,
        args_hash: str | None = None,
    ) -> ExecutionResult:
        """Run one call. Raises only ``AuditWriteError`` from a rejection audit write."""
        try:
            model = spec.input_model.model_validate(arguments)
        except PydanticValidationError as exc:
            return ExecutionResult(error=invalid_arguments_error(exc))

        checked = self._inputs.check(session, spec, arguments, args_hash=args_hash)
        if not checked.ok:
            return ExecutionResult(
                error=input_rejected_error(checked.rejections),
                rejections=tuple(checked.rejections),
            )

        if checked.resolved_datasets:
            model = model.model_copy(update=dict(checked.resolved_datasets))

        ctx = ToolContext(
            session=session,
            principal=Principal(user=session.owner),
            services=self._services,
            clock=self._clock,
            inputs=self._inputs,
        )
        try:
            output = spec.handler(ctx, model)
        except AppError as exc:
            return ExecutionResult(error=ToolError(exc.code, exc.message, dict(exc.details)))
        except (GuardViolation, OrchestrationError) as exc:
            return ExecutionResult(error=ToolError(exc.code, exc.message), error_kind="guard")
        except Exception as exc:
            # Handler exceptions can embed credentials or raw data in their messages.
            LOGGER.error("Tool %s failed unexpectedly (%s)", spec.name, type(exc).__name__)
            return ExecutionResult(
                error=ToolError(
                    CODE_TOOL_FAILED, f"The tool {spec.name} failed.", {"tool": spec.name}
                )
            )

        result = ExecutionResult(output=output)
        self._write_experiment_audit(session, spec, model, output, result)
        return result

    def _write_experiment_audit(
        self,
        session: Session,
        spec: ToolSpec,
        model: Any,
        output: ToolOutput,
        result: ExecutionResult,
    ) -> None:
        action = spec.flags.audit_action
        if not action:
            return
        experiment_id = self._audit_experiment(spec, model, output)
        if experiment_id is None:
            return
        try:
            prefix = self._services.experiments.get(experiment_id).artifact_prefix
            result.service_audit_ref = self._services.audit.write(
                artifact_prefix=prefix,
                experiment_id=experiment_id,
                action=action,
                actor=session.owner,
                detail={str(k): str(v) for k, v in output.audit_detail.items()},
                at=self._clock(),
            )
        except (AuditWriteError, AppError):
            LOGGER.error("Experiment audit entry for tool %s could not be written", spec.name)
            result.audit_failed = True
        except Exception:
            LOGGER.exception("Experiment audit entry for tool %s failed", spec.name)
            result.audit_failed = True

    @staticmethod
    def _audit_experiment(spec: ToolSpec, model: Any, output: ToolOutput) -> str | None:
        if output.experiment_ids:
            return output.experiment_ids[0]
        arg = spec.flags.experiment_arg
        value = getattr(model, arg, None) if arg else None
        return value if isinstance(value, str) else None
