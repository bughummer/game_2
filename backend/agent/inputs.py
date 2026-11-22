"""Input checks for agent tool calls (design C5 shell rules; Req 3.4, 7.3, 7.12, 8.10).

``ToolInputs.check`` runs after the arguments validate against the tool's input model and before
any handler runs. It rejects:

* local file paths, ``s3://`` and other URIs, and signed-URL material in any string argument;
* instance types outside the allow-list;
* dataset ids that are not in the session's dataset list;
* experiment ids that are not linked to the session, or not owned by the session owner. The
  owner confirmed this stricter-than-routes rule for reads as well as mutations.

Each rejected argument writes one ``input_rejected`` audit record. The record names the argument
and a stable reason, never the rejected value.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from backend.agent.audit import AgentAudit
from backend.agent.results import ScrubContext, scrub_sensitive
from backend.agent.tools import ToolSpec
from backend.auth import Authorizer, Principal
from backend.errors import AppError
from ml_engine.contracts.agent import AgentAuditRecord, Session

REASON_LOCAL_PATH = "local_path"
REASON_URI = "uri_not_allowed"
REASON_SIGNED_URL = "signed_url"
REASON_INSTANCE = "instance_not_allowed"
REASON_DATASET = "unknown_dataset"
REASON_EXPERIMENT_NOT_LINKED = "experiment_not_linked"
REASON_EXPERIMENT_NOT_OWNED = "experiment_not_owned"

_EMBEDDED_URI = re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://")
_SIGNED = re.compile(r"(?i)x-amz-(?:signature|credential|security-token|algorithm)|[?&]signature=")
_LOCAL_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/|~(?:[\\/]|$)|\.{1,2}[\\/]|file:)")


@dataclass(frozen=True, slots=True)
class InputRejection:
    argument: str
    reason: str
    message: str


@dataclass(slots=True)
class InputCheck:
    rejections: list[InputRejection] = field(default_factory=list)
    #: argument name -> server-side URI (or list of URIs) for each dataset argument.
    resolved_datasets: dict[str, str | list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.rejections


def _walk_strings(value: Any, path: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk_strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")


def _string_problem(value: str) -> str | None:
    if _SIGNED.search(value):
        return REASON_SIGNED_URL
    if _EMBEDDED_URI.search(value):
        return REASON_URI
    if _LOCAL_PATH.match(value.strip()):
        return REASON_LOCAL_PATH
    return None


_MESSAGES = {
    "sensitive_input": "Credentials and sensitive arguments are not accepted.",
    REASON_LOCAL_PATH: "Local file paths are not accepted. Use a dataset id.",
    REASON_URI: "URIs are not accepted. Use a dataset id or an artifact reference.",
    REASON_SIGNED_URL: "Signed URLs are not accepted.",
    REASON_INSTANCE: "That instance type is not allowed.",
    REASON_DATASET: "That dataset id is not known to this session.",
    REASON_EXPERIMENT_NOT_LINKED: "That experiment is not linked to this session.",
    REASON_EXPERIMENT_NOT_OWNED: "That experiment is not owned by the session owner.",
}


class ToolInputs:
    def __init__(
        self,
        *,
        audit: AgentAudit,
        authorizer: Authorizer,
        allowed_instance_types: tuple[str, ...] | list[str],
    ) -> None:
        self._audit = audit
        self._authorizer = authorizer
        self._allowed_instances = frozenset(allowed_instance_types)

    def check(
        self,
        session: Session,
        spec: ToolSpec,
        arguments: dict[str, Any],
        *,
        args_hash: str | None = None,
    ) -> InputCheck:
        """Check one call's arguments. Mutates ``session.audit_seq`` when it audits.

        Raises ``AuditWriteError`` when a rejection cannot be audited; the caller owns that case.
        """
        result = InputCheck()
        rejected: set[str] = set()

        def reject(argument: str, reason: str) -> None:
            if argument in rejected:
                return
            rejected.add(argument)
            result.rejections.append(InputRejection(argument, reason, _MESSAGES[reason]))

        for path, text in _walk_strings(arguments, ""):
            reason = _string_problem(text)
            if reason:
                reject(path, reason)

        for key, value in arguments.items():
            if scrub_sensitive({key: value}, ScrubContext()) != {key: value} and not any(
                path == key or path.startswith((key + ".", key + "[")) for path in rejected
            ):
                reject(scrub_sensitive(key, ScrubContext()), "sensitive_input")

        flags = spec.flags
        for name in flags.instance_args:
            value = arguments.get(name)
            if value is not None and value not in self._allowed_instances:
                reject(name, REASON_INSTANCE)

        known = {d.dataset_id: d.uri for d in session.datasets}
        for name in flags.dataset_args:
            value = arguments.get(name)
            if value is None or name in rejected:
                continue
            ids = value if isinstance(value, (list, tuple)) else [value]
            if not all(isinstance(i, str) and i in known for i in ids):
                reject(name, REASON_DATASET)
            elif isinstance(value, (list, tuple)):
                result.resolved_datasets[name] = [known[i] for i in ids]
            else:
                result.resolved_datasets[name] = known[value]

        if flags.experiment_arg:
            value = arguments.get(flags.experiment_arg)
            if value is not None and flags.experiment_arg not in rejected:
                reason = self._experiment_problem(session, value)
                if reason:
                    reject(flags.experiment_arg, reason)

        for item in result.rejections:
            self._audit.write(
                session,
                AgentAuditRecord(
                    session_id=session.session_id,
                    owner=session.owner,
                    actor=session.owner,
                    seq=0,
                    event_type="input_rejected",
                    at=datetime.now(UTC),
                    result_status="rejected",
                    tool=spec.name,
                    args_hash=args_hash,
                    detail={
                        "argument": scrub_sensitive(item.argument, ScrubContext()),
                        "reason": item.reason,
                    },
                ),
            )
        return result

    def _experiment_problem(self, session: Session, experiment_id: Any) -> str | None:
        if not isinstance(experiment_id, str) or experiment_id not in session.experiment_ids:
            return REASON_EXPERIMENT_NOT_LINKED
        try:
            self._authorizer.require_owner(Principal(user=session.owner), experiment_id)
        except AppError:
            return REASON_EXPERIMENT_NOT_OWNED
        return None
