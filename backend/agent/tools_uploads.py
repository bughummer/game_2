"""Local-folder upload tool (design C16; Req 7.3, 7.4, 7.6, 7.7, 7.8, 7.11).

``request_local_upload`` (Mutating, ``flags.upload``) names a file from the manifest the client
registered (``session.local_files``). It never reads a path: the server has no access to the
user's files (Req 7.3). The flow:

1. Before the gate (``pre_gate``): an unknown file, a bad extension or an oversized file is
   refused, a file whose SHA-256 matches a session dataset reuses it, and when several
   candidates exist and the user has not named one the user is asked. None of these creates an
   approval request (Req 7.6, 7.7, 7.8, 7.11).
2. The gate's upload rule asks for approval in both modes until one upload was approved in the
   session (Req 7.4). The request shows the file name, size and destination prefix.
3. The handler transfers nothing. It hands the file id to the loop (``ToolOutput.client_upload``),
   which creates the session's ``ClientAction`` and waits for the user's client to upload.
   ``AgentService.complete_upload`` then gives the call its result.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.agent.execute import ToolError
from backend.agent.tools import ToolContext, ToolFlags, ToolOutput, ToolRegistry
from backend.agent.uploads import UploadRefusal, UploadReuse, reuse_or_plan
from backend.errors import ValidationError
from ml_engine.contracts.agent import LocalFileRef, Session, ToolClass

TOOL_NAME = "request_local_upload"
CODE_CHOICE_REQUIRED = "USER_CHOICE_REQUIRED"
CODE_UPLOAD_REFUSED = "UPLOAD_REFUSED"
MAX_CANDIDATES_SHOWN = 20


class RequestLocalUploadArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="The file id from the registered local files. Omit when there is one file.",
    )


def _settings(ctx: ToolContext) -> Any:
    return ctx.services.settings


def _user_named(session: Session, file: LocalFileRef) -> bool:
    """The user's own text (goal or messages) names the file."""
    name = file.name.lower()
    texts = [session.goal, *(e.text for e in session.transcript if e.kind == "user_message")]
    return any(text and name in text.lower() for text in texts)


def _chosen(session: Session, args: RequestLocalUploadArgs) -> LocalFileRef | ToolError:
    """The file to upload, or the question to ask. The agent never picks among several."""
    files = list(session.local_files)
    if not files:
        return ToolError(
            CODE_UPLOAD_REFUSED,
            "No local files are registered. Ask the user to name a folder or file to list.",
            {"reason": "no_local_files"},
        )
    if args.file_id is not None:
        file = next((f for f in files if f.file_id == args.file_id), None)
        if file is None:
            return ToolError(
                CODE_UPLOAD_REFUSED,
                "That file is not in the list the client registered. List the folder again.",
                {"reason": "unknown_file", "file_id": args.file_id},
            )
        if len(files) == 1 or _user_named(session, file):
            return file
    elif len(files) == 1:
        return files[0]
    shown = files[:MAX_CANDIDATES_SHOWN]
    return ToolError(
        CODE_CHOICE_REQUIRED,
        "Several files could be uploaded. Which file should I use?",
        {
            "reason": "user_choice_required",
            "candidate_count": len(files),
            "candidates": [
                {"file_id": f.file_id, "name": f.name, "size_bytes": f.size_bytes} for f in shown
            ],
        },
    )


def _refusal_error(refusal: UploadRefusal) -> ToolError:
    return ToolError(refusal.code, refusal.message, {"reason": refusal.reason, **refusal.details})


def pre_gate(ctx: ToolContext, args: RequestLocalUploadArgs) -> ToolOutput | ToolError | None:
    """Refuse, reuse or ask before any approval exists; ``None`` lets the gate decide."""
    file = _chosen(ctx.session, args)
    if isinstance(file, ToolError):
        return file
    outcome = reuse_or_plan(ctx.session, file, _settings(ctx))
    if isinstance(outcome, UploadRefusal):
        return _refusal_error(outcome)
    if isinstance(outcome, UploadReuse):
        return _reused(outcome)
    return None


def _reused(reuse: UploadReuse) -> ToolOutput:
    return ToolOutput(
        data={
            **reuse.to_result(),
            "message": "This file is already registered in the session; the existing dataset "
            "is reused and nothing was uploaded.",
        }
    )


def approval_view(ctx: ToolContext, args: RequestLocalUploadArgs) -> dict[str, Any]:
    """What the user must see: file name, size in bytes and destination prefix (Req 7.4)."""
    file = _chosen(ctx.session, args)
    if isinstance(file, ToolError):
        raise ValidationError(file.message, file.details)
    outcome = reuse_or_plan(ctx.session, file, _settings(ctx))
    if not hasattr(outcome, "destination_prefix"):
        raise ValidationError("The file can no longer be uploaded.", {"file_id": file.file_id})
    return {
        "file_id": file.file_id,
        "file_name": file.name,
        "size_bytes": file.size_bytes,
        "destination_prefix": outcome.destination_prefix,  # type: ignore[union-attr]
    }


def request_local_upload(ctx: ToolContext, args: RequestLocalUploadArgs) -> ToolOutput:
    """Hand the approved file to the client. Transfers nothing (design C16 step 5)."""
    file = _chosen(ctx.session, args)
    if isinstance(file, ToolError):
        raise ValidationError(file.message, file.details)
    outcome = reuse_or_plan(ctx.session, file, _settings(ctx))
    if isinstance(outcome, UploadRefusal):
        raise ValidationError(outcome.message, {"reason": outcome.reason, **outcome.details})
    if isinstance(outcome, UploadReuse):
        return _reused(outcome)
    return ToolOutput(
        data={
            "status": "waiting_for_client_upload",
            "file_id": file.file_id,
            "file_name": file.name,
            "size_bytes": file.size_bytes,
        },
        client_upload=file.file_id,
    )


def register(registry: ToolRegistry, services: Any) -> None:
    del services  # handlers use ctx.services at call time
    registry.register(
        TOOL_NAME,
        classes=[ToolClass.MUTATING],
        description=(
            "Upload one of the user's registered local files (CSV or Parquet) as a dataset. "
            "Give its file_id. Returns the dataset id, columns, types and row count; never rows."
        ),
        input_model=RequestLocalUploadArgs,
        handler=request_local_upload,
        flags=ToolFlags(upload=True),
        approval_view=approval_view,
        pre_gate=pre_gate,
    )
