"""Opt-in real-AWS smoke harness for the ML Assistant (ml-assistant Req 17, design Smoke_Test).

    python -m scripts.agent_smoke        # from the repository root; makes real AWS calls

Importing this module makes no AWS call. Running it does, in this order, stopping at the first
precondition that does not hold (Req 17.7):

  (a) converse     - one Converse call with a trivial tool on the configured model and region;
                     the model must answer with that tool.
  (b) aws_access   - the configuration is complete, SageMaker job listing works, and an object
                     can be written, read back and removed under the agent's S3 namespace.
  (c) session      - a smoke session is created through ``AgentService.create_session``, saved
                     to S3, and loads with identical state in a freshly built service.

The smoke session uses half the configured production limits, a spend maximum of at most 5 USD,
and the smallest allowed training instance on Managed Spot (Req 17.5). Check (c) creates it with
a runner that never starts the loop.

Then the step path (Req 17.3, 17.4, 17.6) resumes that same session in a new service whose
runner really runs the loop, in Supervised mode, on a synthetic labelled dataset plus a labelled
validation file written to a temporary folder. The harness plays the user: it grants every
approval with the hash it was shown, performs each upload through the client flow
(``client_action``, PUT to the pre-signed URL, ``complete_upload``), advances the platform's
job poller and resumes the session when its jobs end. Each step - upload, experiment, EDA,
training, validation, report - is recorded as passed or failed, in that order. When the
estimated spend reaches the smoke maximum it approves nothing more, stops the running jobs
through the same code as ``cancel(stop_jobs=True)``, fails naming the limit and leaves the
session resumable (it is not cancelled).

Pass or fail, the harness prints the session id, every job reference and the estimated spend
(Req 17.9), exits non-zero on any failure, and keeps the session for diagnosis (Req 17.10).
Settings are read only through ``backend.config.Settings``.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import random
import re
import sys
import tempfile
import time
import urllib.request
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from backend.agent.converse import (
    BedrockConverseClient,
    ConverseClient,
    ConverseError,
    ConverseMessage,
    TextBlock,
    ToolDefinition,
)
from backend.agent.limits import CODE_LIMIT_REACHED, LimitName
from backend.agent.runner import InlineAgentRunner
from backend.agent.service import AgentService, ClientActionView, limits_from_settings
from backend.agent.store import SessionInUse, SessionLoadError
from backend.agent.wiring import build_agent_service
from backend.auth import Principal
from backend.aws import build_client
from backend.config import GPU_FAMILY_PREFIXES, Settings
from backend.errors import AppError, ConflictError
from ml_engine.contracts.agent import CallState, Session, SessionLimits, SessionStatus
from ml_engine.io import S3ObjectStore
from ml_engine.io.layout import AgentLayout

#: Req 17.5: the smoke spend maximum never exceeds this, whatever production allows.
SMOKE_SPEND_CAP_USD = Decimal("5")

#: The integer ``agent_limit_*`` settings halved for the smoke session.
SMOKE_INT_LIMITS: tuple[str, ...] = (
    "agent_limit_max_converse_turns",
    "agent_limit_max_tool_calls",
    "agent_limit_max_costly_tool_calls",
    "agent_limit_max_active_seconds",
    "agent_limit_max_repeated_identical_calls",
)
SMOKE_SPEND_LIMIT = "agent_limit_max_estimated_spend_usd"

SMOKE_OWNER = "agent-smoke"

#: Req 17.1: the synthetic dataset stays within these bounds.
SMOKE_MAX_ROWS = 10_000
SMOKE_MAX_BYTES = 5 * 1024 * 1024
SMOKE_TRAIN_FILE = "smoke_train.csv"
SMOKE_VALIDATION_FILE = "smoke_validation.csv"
SMOKE_TARGET = "churned"
SMOKE_FEATURES = ("tenure_months", "monthly_charges", "contract", "support_tickets")

SMOKE_GOAL = (
    "ML Factory agent smoke test. Do these steps in order and keep everything small: "
    f"(1) upload the local file {SMOKE_TRAIN_FILE}; "
    f"(2) create an experiment from it with target column '{SMOKE_TARGET}'; "
    "(3) wait for EDA, read its summary and confirm the feature selection with "
    f"decide_features keeping {', '.join(SMOKE_FEATURES)}; "
    "(4) start one training run with only the logistic_regression model and default settings; "
    f"(5) upload the local file {SMOKE_VALIDATION_FILE} and start a validation of the trained "
    "model on it; (6) when the validation has finished, give a short final report of the "
    "results. Use defaults and do not ask questions unless you cannot continue."
)
#: The harness's answer whenever the agent asks the user something.
SMOKE_REPLY = "Proceed with your plan using the recommended defaults. Do not ask again."
SMOKE_MAX_REPLIES = 3
SMOKE_POLL_SECONDS = 20.0
SMOKE_TIMEOUT_SECONDS = 3 * 3600.0
SMOKE_MAX_ITERATIONS = 2_000

#: The steps of Req 17.4, in order: (name, awaited job kind or None).
SMOKE_STEPS: tuple[tuple[str, str | None], ...] = (
    ("upload", None),
    ("experiment", None),
    ("eda", "eda"),
    ("training", "training"),
    ("validation", "validation"),
    ("report", None),
)
PING_TOOL = ToolDefinition(
    name="smoke_ping",
    description="Connectivity check. Call it exactly once with echo set to 'ok'.",
    input_schema={
        "type": "object",
        "properties": {"echo": {"type": "string"}},
        "required": ["echo"],
    },
)


# ------------------------------------------------------------------ pure helpers (unit-tested)


def smoke_settings(settings: Settings) -> Settings:
    """A copy of ``settings`` for the smoke run: half of each production limit, spend <= 5 USD.

    Integer limits are floored and kept at least 1 (a limit must stay positive). The agent is
    switched on in the copy because running the harness is the explicit opt-in. The agent's
    tools offer no instance choice, so the copy also sets the training instance to the smallest
    allowed type and keeps Managed Spot on (Req 17.5); the Guards still check both.
    """
    update: dict[str, Any] = {
        name: max(1, int(getattr(settings, name)) // 2) for name in SMOKE_INT_LIMITS
    }
    spend = Decimal(str(getattr(settings, SMOKE_SPEND_LIMIT))) / 2
    update[SMOKE_SPEND_LIMIT] = min(spend, SMOKE_SPEND_CAP_USD)
    update["agent_enabled"] = True
    update["spot_enabled"] = True
    allowed = list(getattr(settings, "allowed_instance_types", None) or [])
    if allowed:
        update["training_instance_type"] = smallest_instance_type(allowed)
    return settings.model_copy(update=update)


_SIZE_RANK = {"nano": 0, "micro": 1, "small": 2, "medium": 3, "large": 4, "xlarge": 5}
_MULTI_XLARGE = re.compile(r"^(\d+)xlarge$")


def _size_rank(instance_type: str) -> float:
    """Rank of ``ml.<family>.<size>`` by size; an unknown size sorts last."""
    size = instance_type.rsplit(".", 1)[-1]
    if size in _SIZE_RANK:
        return _SIZE_RANK[size]
    match = _MULTI_XLARGE.match(size)
    return 4 + int(match.group(1)) if match else float("inf")


def smallest_instance_type(allowed: Sequence[str]) -> str:
    """The smallest CPU instance type in ``allowed`` (ties broken by name, so it is stable)."""
    cpu = [t for t in allowed if not t.startswith(GPU_FAMILY_PREFIXES)]
    if not cpu:
        raise ValueError("allowed_instance_types holds no CPU instance type")
    return min(cpu, key=lambda t: (_size_rank(t), t))


def smoke_limits(settings: Settings) -> SessionLimits:
    """The ``SessionLimits`` a smoke session receives from ``smoke_settings(settings)``."""
    return limits_from_settings(smoke_settings(settings))


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass(slots=True)
class SmokeReport:
    checks: list[CheckResult] = field(default_factory=list)
    session_id: str | None = None
    job_refs: list[str] = field(default_factory=list)
    estimated_spend_usd: float = 0.0

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(check.passed for check in self.checks)


def job_refs(session: Session) -> list[str]:
    """One ``<kind>:<execution_id> (<state>)`` reference per job the session started."""
    return [f"{job.kind}:{job.execution_id} ({job.state})" for job in session.awaited_jobs]


def format_report(report: SmokeReport) -> str:
    """The final report (Req 17.9, 17.10). Always names the session, jobs and spend."""
    lines = ["ML Assistant smoke test"]
    for check in report.checks:
        lines.append(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    failed = next((check for check in report.checks if not check.passed), None)
    if failed is not None:
        lines.append(f"Result: FAILED at {failed.name}")
    elif report.passed:
        lines.append("Result: PASSED")
    else:
        lines.append("Result: FAILED (no check ran)")
    lines.append(f"Session id: {report.session_id or '(none created)'}")
    if report.job_refs:
        lines.append("Jobs:")
        lines.extend(f"  {ref}" for ref in report.job_refs)
    else:
        lines.append("Jobs: (none started)")
    lines.append(f"Estimated spend: {report.estimated_spend_usd:.2f} USD")
    if report.session_id:
        lines.append("The session is kept for diagnosis; it is not deleted.")
    return "\n".join(lines)


def describe_error(exc: BaseException) -> str:
    """A cause safe to print: our own error messages, an AWS error code, or the type name."""
    if isinstance(exc, (AppError, ConverseError, SessionLoadError)):
        code = getattr(exc, "code", "") or ""
        return f"{type(exc).__name__}{f' [{code}]' if code else ''}: {exc}"
    if isinstance(exc, PydanticValidationError):  # field names and messages, never the values
        parts = [
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}"
            for e in exc.errors(include_input=False, include_url=False)
        ]
        return f"{type(exc).__name__}: " + "; ".join(parts)
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        code = (response.get("Error") or {}).get("Code", "")
        return f"{type(exc).__name__} [{code}]"
    return type(exc).__name__


# ------------------------------------------------------------------ preconditions (real AWS)


class HeldRunner:
    """A runner that records kicks and never runs the loop: no model turn starts here."""

    def __init__(self) -> None:
        self.kicked: list[str] = []

    def kick(self, session_id: str) -> None:
        self.kicked.append(session_id)


def check_converse(settings: Settings, client: ConverseClient | None = None) -> CheckResult:
    """(a) The configured model accepts a tool-use request in the configured region."""
    name = "converse"
    model_id = settings.effective_agent_model_id
    if not model_id:
        return CheckResult(
            name, False, "ML_FACTORY_AGENT_MODEL_ID or ML_FACTORY_BEDROCK_MODEL_ID is required"
        )
    client = client if client is not None else BedrockConverseClient.from_settings(settings)
    turn = client.converse(
        system="You are a connectivity check. Call the smoke_ping tool exactly once.",
        messages=[ConverseMessage(role="user", blocks=(TextBlock("Call smoke_ping now."),))],
        tools=[PING_TOOL],
        max_tokens=256,
    )
    if not any(use.name == PING_TOOL.name for use in turn.tool_uses):
        return CheckResult(
            name,
            False,
            f"model {model_id} answered without using the tool (stop_reason={turn.stop_reason})",
        )
    return CheckResult(name, True, f"model {model_id} in {settings.aws_region} used the tool")


def check_aws_access(settings: Settings, s3_client: Any, sagemaker_client: Any) -> CheckResult:
    """(b) The configuration is complete and the role can reach SageMaker and the S3 locations.

    Listing jobs proves SageMaker access without spending; starting a job is proven by the full
    step path (task 12.7). The S3 probe is written under the agent namespace, read back and
    removed again.
    """
    name = "aws_access"
    problems = settings.problems()
    if problems:
        return CheckResult(name, False, "configuration incomplete: " + "; ".join(problems))
    sagemaker_client.list_training_jobs(MaxResults=1)
    sagemaker_client.list_processing_jobs(MaxResults=1)
    store = _s3_store(settings, s3_client)
    probe = AgentLayout.for_session(settings.artifact_root, f"smoke_{uuid.uuid4().hex}")
    uri = probe.tool_result("probe")
    payload = b'{"smoke": "probe"}'
    store.write_bytes(uri, payload)
    try:
        if store.read_bytes(uri) != payload:
            return CheckResult(name, False, "the S3 probe read back different bytes")
    finally:
        store.delete(uri)
    return CheckResult(name, True, "SageMaker listing and S3 write/read/delete succeeded")


def check_session_round_trip(
    settings: Settings,
    store: Any,
    fresh_store: Callable[[], Any],
    converse_client: ConverseClient | None = None,
) -> tuple[CheckResult, str | None]:
    """(c) A session saved to S3 loads in a freshly built service with identical state.

    Returns the check and the session id (kept, whatever the outcome).
    """
    name = "session"
    service = build_agent_service(
        settings, store=store, converse_client=converse_client, runner=HeldRunner()
    )
    owner = settings.dev_identity or SMOKE_OWNER
    session_id = service.create_session(Principal(user=owner), SMOKE_GOAL).session_id
    saved = service.store.load(session_id).session
    fresh = build_agent_service(
        settings, store=fresh_store(), converse_client=converse_client, runner=HeldRunner()
    )
    loaded = fresh.store.load(session_id).session
    if loaded.model_dump(mode="json") != saved.model_dump(mode="json"):
        result = CheckResult(name, False, "the reloaded session differs from the saved one")
    elif loaded.limits != limits_from_settings(settings):
        result = CheckResult(name, False, "the reloaded session does not carry the smoke limits")
    else:
        result = CheckResult(name, True, "saved to S3 and reloaded with identical state")
    return result, session_id


# ------------------------------------------------------------------ step path: data


def write_smoke_datasets(
    folder: Path, *, train_rows: int = 2_000, validation_rows: int = 500, seed: int = 17
) -> dict[str, Path]:
    """Write the labelled training and validation CSVs; deterministic for a given seed.

    Raises ``ValueError`` if a file would break the Req 17.1 bounds.
    """
    rng = random.Random(seed)
    paths: dict[str, Path] = {}
    for name, rows in ((SMOKE_TRAIN_FILE, train_rows), (SMOKE_VALIDATION_FILE, validation_rows)):
        if not 0 < rows <= SMOKE_MAX_ROWS:
            raise ValueError(f"{name}: {rows} rows is outside 1..{SMOKE_MAX_ROWS}")
        path = Path(folder) / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([*SMOKE_FEATURES, SMOKE_TARGET])
            for _ in range(rows):
                tenure = rng.randint(1, 72)
                charges = round(rng.uniform(20, 120), 2)
                contract = rng.choice(("monthly", "annual", "two_year"))
                tickets = rng.randint(0, 6)
                risk = 0.03 * charges - 0.05 * tenure + 0.4 * tickets
                risk += 1.2 if contract == "monthly" else -0.8
                label = int(risk + rng.gauss(0, 1) > 0.5)
                writer.writerow([tenure, charges, contract, tickets, label])
        if path.stat().st_size > SMOKE_MAX_BYTES:
            raise ValueError(f"{name} is over {SMOKE_MAX_BYTES} bytes")
        paths[name] = path
    return paths


def local_file_entry(path: Path) -> dict[str, Any]:
    """What the client registers for one local file: name, size, time and SHA-256."""
    stat = path.stat()
    return {
        "name": path.name,
        "size_bytes": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


# ------------------------------------------------------------------ step path: pure checks


def spend_limit_reached(session: Session) -> bool:
    """The estimated spend reached the smoke maximum (Req 17.6).

    True when the counter is at the limit, the loop stopped on the spend limit, or a Costly
    call was refused because it would have passed the spend limit.
    """
    if session.counters.estimated_spend_usd >= session.limits.max_estimated_spend_usd:
        return True
    if any(
        e.kind == "limit_stop" and e.data.get("limit") == LimitName.SPEND.value
        for e in session.transcript
    ):
        return True
    return any(
        r.error_code == CODE_LIMIT_REACHED and LimitName.SPEND.value in (r.result_summary or "")
        for r in session.tool_calls
    )


def spend_stop_detail(session: Session) -> str:
    return (
        f"estimated spend {session.counters.estimated_spend_usd:.2f} USD reached the smoke "
        f"maximum {LimitName.SPEND.value} = {session.limits.max_estimated_spend_usd:.2f} USD; "
        "no further job was started, running jobs were stopped and the session was left "
        "resumable"
    )


def reached_limit(session: Session) -> str | None:
    """The limit named by the last limit stop, if the session waits on one."""
    if session.status is not SessionStatus.WAITING_FOR_USER:
        return None
    for entry in reversed(session.transcript):
        if entry.kind == "limit_stop":
            limit = entry.data.get("limit")
            return limit if isinstance(limit, str) else None
        if entry.kind == "user_message":
            return None
    return None


def _job_step(session: Session, kind: str) -> tuple[bool, str]:
    jobs = [j for j in session.awaited_jobs if j.kind == kind]
    done = next((j for j in jobs if j.state == "succeeded"), None)
    if done is not None:
        return True, f"{kind} job {done.execution_id} succeeded"
    if not jobs:
        return False, f"no {kind} job was started"
    last = jobs[-1]
    reason = f": {last.failure_reason}" if last.failure_reason else ""
    return False, f"{kind} job {last.execution_id} is {last.state}{reason}"


def _step(session: Session, name: str, kind: str | None) -> tuple[bool, str]:
    if kind is not None:
        return _job_step(session, kind)
    if name == "upload":
        ref = next((d for d in session.datasets if d.filename == SMOKE_TRAIN_FILE), None)
        if ref is None:
            return False, f"{SMOKE_TRAIN_FILE} was not uploaded"
        return True, f"{SMOKE_TRAIN_FILE} uploaded as {ref.dataset_id} ({ref.row_count} rows)"
    if name == "experiment":
        created = any(
            r.tool == "create_experiment" and r.state is CallState.SUCCEEDED
            for r in session.tool_calls
        )
        if not created or not session.experiment_ids:
            return False, "no experiment was created"
        return True, f"experiment {session.experiment_ids[0]} created"
    # report
    if session.status is SessionStatus.COMPLETED and session.final_answer is not None:
        return True, "the session completed with a final report"
    failure = f" ({session.failure.message})" if session.failure else ""
    return False, f"no final report; the session is {session.status.value}{failure}"


def step_results(session: Session, stop_reason: str | None = None) -> list[CheckResult]:
    """One result per Req 17.4 step, in order. The first failed step carries the stop reason."""
    results: list[CheckResult] = []
    for name, kind in SMOKE_STEPS:
        passed, detail = _step(session, name, kind)
        first_failure = not passed and all(r.passed for r in results)
        if first_failure and stop_reason:
            detail = f"{detail}; stopped: {stop_reason}"
        results.append(CheckResult(name, passed, detail))
    return results


def displayed_hash(approval: Any) -> str:
    """The hash of the arguments the user is shown for a pending approval."""
    if approval.status == "edit_proposed" and approval.edited_hash:
        return str(approval.edited_hash)
    return str(approval.args_hash)


# ------------------------------------------------------------------ step path: driving

PutFn = Callable[[ClientActionView, bytes], None]


def http_put(ticket: ClientActionView, body: bytes) -> None:
    """PUT the file to the pre-signed URL, as the user's client does. Never logs the URL."""
    if not ticket.upload_url.startswith("https://"):
        raise ValueError("the upload URL is not HTTPS")
    request = urllib.request.Request(
        ticket.upload_url, data=body, method=ticket.method, headers=dict(ticket.headers)
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status >= 300:
            raise RuntimeError(f"upload PUT returned HTTP {response.status}")


def upload_client_action(
    service: AgentService, who: Principal, session_id: str, files: Mapping[str, Path], put: PutFn
) -> None:
    """Perform the session's pending upload through the client flow (Req 7.9)."""
    ticket = service.client_action(who, session_id)
    if ticket is None:
        return
    path = files.get(ticket.file_name)
    if path is None:
        raise ValueError(f"the agent asked for an unknown local file {ticket.file_name!r}")
    put(ticket, path.read_bytes())
    service.complete_upload(who, session_id, ticket.action_id)


def stop_session_jobs(service: AgentService, who: Principal, session_id: str) -> None:
    """Stop every running job the session started, through the code ``cancel(stop_jobs=True)``
    uses, without cancelling the session, so it stays resumable (Req 17.6)."""
    with service.store.transaction(session_id) as s:
        service._stop_jobs(s, who.user)  # the cancel path's job stop, minus the cancel


@dataclass(frozen=True, slots=True)
class DriveOutcome:
    """Why driving stopped: ``None`` once the session finished on its own."""

    stop_reason: str | None = None
    spend_stopped: bool = False


def drive_session(
    service: AgentService,
    who: Principal,
    session_id: str,
    files: Mapping[str, Path],
    *,
    put: PutFn = http_put,
    tick: Callable[[], None] = lambda: None,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    poll_seconds: float = SMOKE_POLL_SECONDS,
    timeout_seconds: float = SMOKE_TIMEOUT_SECONDS,
    max_iterations: int = SMOKE_MAX_ITERATIONS,
) -> DriveOutcome:
    """Play the user until the session finishes, a limit stops it, or time runs out.

    Each pass reads the session view (which records ended jobs) and acts on its status:
    approve what is shown, upload what is asked for, answer a question, advance the job
    poller (``tick``) and resume once the awaited jobs ended. The spend check comes first, so
    once the spend maximum is reached nothing more is approved or resumed.
    """
    deadline = clock() + timeout_seconds
    replies = 0
    for _ in range(max_iterations):
        view = service.get_session(who, session_id)
        session = service.store.load(session_id).session
        if spend_limit_reached(session):
            stop_session_jobs(service, who, session_id)
            return DriveOutcome(spend_stop_detail(session), spend_stopped=True)
        status = view.status
        if status in (SessionStatus.COMPLETED, SessionStatus.FAILED, SessionStatus.CANCELLED):
            return DriveOutcome()
        if clock() > deadline:
            return DriveOutcome(f"timed out after {timeout_seconds:g} s ({status.value})")
        if status is SessionStatus.WAITING_FOR_APPROVAL:
            for approval in view.pending_approvals:
                service.decide_approval(
                    who, session_id, approval.approval_id, "approve",
                    args_hash=displayed_hash(approval),
                )
        elif status is SessionStatus.WAITING_FOR_USER:
            action = view.client_action
            limit = reached_limit(session)
            if action is not None and action.get("status") == "ready":
                upload_client_action(service, who, session_id, files, put)
            elif limit is not None:
                return DriveOutcome(f"session limit {limit} reached")
            elif replies >= SMOKE_MAX_REPLIES:
                return DriveOutcome(f"the agent asked the user more than {replies} times")
            else:
                replies += 1
                service.send_message(who, session_id, SMOKE_REPLY)
        elif status is SessionStatus.WAITING_FOR_JOB and view.jobs_ready:
            service.resume(who, session_id)
        else:
            # Jobs still running, or ``running`` with no loop attached (a fresh process, or a
            # loop that stopped early): advance the platform, wait, then try to resume.
            tick()
            if status is SessionStatus.RUNNING:
                with contextlib.suppress(SessionInUse, ConflictError):
                    service.resume(who, session_id)
            sleep(poll_seconds)
    return DriveOutcome(f"gave up after {max_iterations} polls")


def build_step_service(
    settings: Settings, container: Any, converse_client: ConverseClient | None = None
) -> AgentService:
    """The agent service on the platform ``container``, with a runner that runs the loop."""
    return build_agent_service(
        settings,
        store=container.store,
        services=container,
        converse_client=converse_client,
        audit_log=container.audit,
        authorizer=container.authorizer,
        governance=container.governance,
        runner=InlineAgentRunner(),
    )


def run_step_path(
    service: AgentService,
    who: Principal,
    session_id: str,
    folder: Path,
    **drive: Any,
) -> list[CheckResult]:
    """Register the synthetic files, resume ``session_id`` and drive it. Never raises.

    ``drive`` is passed to ``drive_session`` (``put``, ``tick``, ``sleep``, ``clock``...).
    Returns the step results in order, then a failed ``spend_limit`` or ``step_path`` result
    when driving stopped early.
    """
    try:
        files = write_smoke_datasets(folder)
        service.register_local_files(
            who, session_id, [local_file_entry(p) for p in files.values()]
        )
        outcome = drive_session(service, who, session_id, files, **drive)
    except Exception as exc:  # every failure is reported, none escapes
        outcome = DriveOutcome(describe_error(exc))
    try:
        session = service.store.load(session_id).session
    except Exception as exc:
        return [CheckResult("step_path", False, describe_error(exc))]
    results = step_results(session, outcome.stop_reason)
    if outcome.spend_stopped:
        results.append(CheckResult("spend_limit", False, outcome.stop_reason or ""))
    elif outcome.stop_reason:
        results.append(CheckResult("step_path", False, outcome.stop_reason))
    return results


# ------------------------------------------------------------------ the run


def _s3_store(settings: Settings, s3_client: Any) -> S3ObjectStore:
    return S3ObjectStore(
        s3_client, server_side_encryption=settings.s3_sse, kms_key_id=settings.s3_kms_key_id
    )


def run_smoke(
    settings: Settings,
    *,
    converse_client: ConverseClient | None = None,
    s3_client_factory: Callable[[], Any] | None = None,
    sagemaker_client: Any = None,
) -> SmokeReport:
    """Run the preconditions in order, stopping at the first failure. Never raises.

    Clients are built from ``settings`` only when their check runs; tests may inject them.
    ``s3_client_factory`` is called once per service so check (c) reloads through a new client.
    """
    config = smoke_settings(settings)
    report = SmokeReport()
    make_s3 = s3_client_factory or (lambda: build_client("s3", config))

    def attempt(name: str, check: Callable[[], CheckResult]) -> bool:
        try:
            result = check()
        except Exception as exc:  # every failure is reported, none escapes
            result = CheckResult(name, False, describe_error(exc))
        report.checks.append(result)
        return result.passed

    if not attempt("converse", lambda: check_converse(config, converse_client)):
        return report
    s3_client = None

    def aws_access() -> CheckResult:
        nonlocal s3_client
        s3_client = make_s3()
        sagemaker = sagemaker_client or build_client("sagemaker", config)
        return check_aws_access(config, s3_client, sagemaker)

    if not attempt("aws_access", aws_access):
        return report

    def session() -> CheckResult:
        result, report.session_id = check_session_round_trip(
            config,
            _s3_store(config, s3_client),
            lambda: _s3_store(config, make_s3()),
            converse_client,
        )
        return result

    if attempt("session", session) and report.session_id:
        _run_steps(report, config, make_s3, sagemaker_client, converse_client)
    if report.session_id:
        _collect(report, config, make_s3)
    return report


def _run_steps(
    report: SmokeReport,
    settings: Settings,
    make_s3: Callable[[], Any],
    sagemaker_client: Any,
    converse_client: ConverseClient | None,
) -> None:
    """Build the platform on real AWS and run the step path on the precondition session."""
    from backend.container import build_container  # the container wires every service

    try:
        s3_client = make_s3()
        container = build_container(
            settings,
            store=_s3_store(settings, s3_client),
            s3_client=s3_client,
            sagemaker_client=sagemaker_client or build_client("sagemaker", settings),
            logs_client=build_client("logs", settings),
            converse_client=converse_client,
        )
        service = build_step_service(settings, container, converse_client)
    except Exception as exc:  # a platform that cannot be built is a failed step
        report.checks.append(CheckResult("step_path", False, describe_error(exc)))
        return
    who = Principal(user=settings.dev_identity or SMOKE_OWNER)
    with tempfile.TemporaryDirectory(prefix="mlf-agent-smoke-") as folder:
        report.checks.extend(
            run_step_path(
                service, who, report.session_id or "", Path(folder), tick=container.poller.tick
            )
        )


def _collect(report: SmokeReport, settings: Settings, make_s3: Callable[[], Any]) -> None:
    """Fill job references and spend from the stored session; a read failure leaves defaults."""
    try:
        service = build_agent_service(
            settings, store=_s3_store(settings, make_s3()), runner=HeldRunner()
        )
        session = service.store.load(report.session_id or "").session
    except Exception as exc:  # a failed read is reported as its own failed step
        report.checks.append(CheckResult("report", False, describe_error(exc)))
        return
    report.job_refs = job_refs(session)
    report.estimated_spend_usd = session.counters.estimated_spend_usd


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.parse_args(argv)
    try:
        settings = Settings()
    except Exception as exc:  # invalid configuration is a failed precondition
        report = SmokeReport(checks=[CheckResult("settings", False, describe_error(exc))])
    else:
        report = run_smoke(settings)
    print(format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
