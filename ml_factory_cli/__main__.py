"""Command line entry point for ML Assistant sessions."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

from ml_factory_cli.local_files import list_candidates, sha256_file
from ml_factory_cli.render import render_approval, render_entry, render_mode
from ml_factory_cli.transport import HttpTransport, InProcessTransport, TransportError

# The contract documents 16 hex characters; current service ids use uuid4's 32.
SESSION_ID = re.compile(r"^ses_(?:[0-9a-f]{16}|[0-9a-f]{32})$")
TERMINAL = {"completed", "failed", "cancelled"}


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mlf", description="ML Factory Assistant")
    p.add_argument("--in-process", action="store_true", help="connect to the service directly")
    p.add_argument("--api-url", help="control plane URL")
    p.add_argument("--user", help="claimed user; the server still resolves identity")
    p.add_argument("--watch", action="store_true", help="keep polling while jobs run")
    p.add_argument("--interval", type=int, default=30, help="watch interval in seconds (5-3600)")
    sub = p.add_subparsers(dest="command")
    sub.add_parser("list")
    for name in ("approve", "reject"):
        c = sub.add_parser(name)
        c.add_argument("session_id")
        c.add_argument("approval_id")
    c = sub.add_parser("edit")
    c.add_argument("session_id")
    c.add_argument("approval_id")
    c.add_argument("arguments", help="JSON object or @path to JSON file")
    c = sub.add_parser("mode")
    c.add_argument("session_id")
    c.add_argument("mode", choices=("supervised", "autopilot"))
    c.add_argument("--max-spend", type=float)
    c = sub.add_parser("cancel")
    c.add_argument("session_id")
    c.add_argument("--stop-jobs", action="store_true")
    c = sub.add_parser("audit")
    c.add_argument("session_id")
    c = sub.add_parser("watch")
    c.add_argument("session_id")
    c.add_argument("--interval", type=int, default=30)
    c = sub.add_parser("data")
    c.add_argument("session_id")
    c.add_argument("path", type=Path)
    c.add_argument(
        "--upload", action="store_true", help="register files and perform requested upload"
    )
    c = sub.add_parser("run")
    c.add_argument("goal_or_session")
    c.add_argument("--watch", action="store_true", default=argparse.SUPPRESS)
    c.add_argument("--interval", type=int, default=argparse.SUPPRESS)
    c.add_argument("--in-process", action="store_true", default=argparse.SUPPRESS)
    c.add_argument("--api-url", default=argparse.SUPPRESS)
    c.add_argument("--user", default=argparse.SUPPRESS)
    return p


def _approval_hash(approval: dict[str, Any]) -> str:
    return approval.get("edited_hash") or approval["args_hash"]


def _decide(t: Any, sid: str, approval: dict[str, Any], decision: str,
            edited: dict[str, Any] | None = None) -> dict[str, Any]:
    return t.request("POST", f"/agent/sessions/{sid}/approvals/{approval['approval_id']}", {
        "decision": decision, "args_hash": _approval_hash(approval), "edited_args": edited,
    })


def _interactive_decision(t: Any, sid: str, approval: dict[str, Any]) -> dict[str, Any] | None:
    answer = input("Approve, reject, edit, or leave pending? [a/r/e/Enter] ").strip().lower()
    if answer == "a":
        return _decide(t, sid, approval, "approve")
    if answer == "r":
        return _decide(t, sid, approval, "reject")
    if answer == "e":
        raw = input("Edited arguments (JSON object): ")
        edited = json.loads(raw)
        if not isinstance(edited, dict):
            raise ValueError("Edited arguments must be a JSON object")
        return _decide(t, sid, approval, "edit", edited)
    return None


def _stream(t: Any, sid: str, *, poll: float, watch: bool, interval: int,
            initial: dict[str, Any] | None = None) -> int:
    seq = 0
    shown: set[str] = set()
    view = initial
    while True:
        if view is None:
            view = t.request("GET", f"/agent/sessions/{sid}?after_seq={seq}")
        print(f"Session {sid}: {view['status']}" if seq == 0 else f"Status: {view['status']}")
        if seq == 0:
            print(render_mode(view))
        for entry in sorted(view.get("transcript") or [], key=lambda e: e["seq"]):
            if entry["seq"] > seq:
                print(render_entry(entry))
                seq = entry["seq"]
        for approval in view.get("pending_approvals") or []:
            key = f"{approval['approval_id']}:{approval['status']}:{_approval_hash(approval)}"
            if key not in shown:
                print(render_approval(approval))
                shown.add(key)
            if not sys.stdout.isatty():
                return 75
            decided = _interactive_decision(t, sid, approval)
            if decided is not None:
                view = decided
                break
            return 75
        else:
            if view["status"] in TERMINAL:
                return 5 if view["status"] == "failed" else 0
            if view.get("jobs_ready"):
                view = t.request("POST", f"/agent/sessions/{sid}/resume", {})
                continue
            if view["status"] == "waiting_for_job" and not watch:
                return 0
            if view["status"] == "waiting_for_user":
                return 0
            time.sleep(interval if view["status"] == "waiting_for_job" else poll)
            view = None


def _get_approval(t: Any, sid: str, approval_id: str) -> dict[str, Any]:
    view = t.request("GET", f"/agent/sessions/{sid}")
    for approval in view.get("pending_approvals") or []:
        if approval["approval_id"] == approval_id:
            print(render_approval(approval))
            return approval
    raise ValueError(f"No pending approval {approval_id}")


def _upload(t: Any, sid: str, path: Path) -> None:
    import httpx

    action = t.request("GET", f"/agent/sessions/{sid}/client-action")
    if not action:
        print("No upload is requested.")
        return
    file = path / action["file_name"] if path.is_dir() else path
    if file.name != action["file_name"]:
        raise ValueError("The requested file is not the selected file")
    if file.stat().st_size != action["size_bytes"] or sha256_file(file) != action["sha256"]:
        raise ValueError("The selected file changed since it was registered")
    headers = {**(action.get("headers") or {}), "Content-Length": str(file.stat().st_size)}
    with file.open("rb") as fh:
        response = httpx.put(action["upload_url"], content=fh, headers=headers, timeout=120)
    response.raise_for_status()
    t.request("POST", f"/agent/sessions/{sid}/client-action/{action['action_id']}/complete", {})
    print(f"Uploaded {file.name}.")


def _run(args: argparse.Namespace) -> int:
    from backend.config import Settings

    settings = Settings()
    t = InProcessTransport(settings) if args.in_process else HttpTransport(
        args.api_url or settings.agent_api_url, args.user or settings.dev_identity
    )
    try:
        if args.command == "list":
            page = t.request("GET", "/agent/sessions?limit=100")
            for item in page.get("items") or []:
                print(f"{item['session_id']}  {item['status']}  {item['mode']}")
            return 0
        if args.command in ("approve", "reject", "edit"):
            approval = _get_approval(t, args.session_id, args.approval_id)
            edited = None
            if args.command == "edit":
                raw = (Path(args.arguments[1:]).read_text(encoding="utf-8")
                       if args.arguments.startswith("@") else args.arguments)
                edited = json.loads(raw)
                if not isinstance(edited, dict):
                    raise ValueError("Edited arguments must be a JSON object")
            view = _decide(t, args.session_id, approval, args.command, edited)
            return _stream(t, args.session_id, poll=settings.agent_cli_poll_seconds,
                           watch=False, interval=args.interval, initial=view)
        if args.command == "mode":
            view = t.request("POST", f"/agent/sessions/{args.session_id}/mode", {
                "mode": args.mode, "max_estimated_spend_usd": args.max_spend,
            })
            print(render_mode(view))
            return 0
        if args.command == "cancel":
            stop = args.stop_jobs
            if not stop and sys.stdin.isatty():
                stop = input("Stop running jobs too? [y/N] ").strip().lower() == "y"
            view = t.request(
                "POST", f"/agent/sessions/{args.session_id}/cancel", {"stop_jobs": stop}
            )
            print(f"Session {args.session_id}: {view['status']}")
            return 0
        if args.command == "audit":
            result = t.request("GET", f"/agent/sessions/{args.session_id}/audit")
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "data":
            candidates = list_candidates(args.path)
            print(f"{candidates.total} candidate file(s)")
            for item in candidates.files:
                print(f"{item['name']}  {item['size_bytes']} bytes  {item['sha256']}")
            if args.upload:
                _upload(t, args.session_id, args.path)
            else:
                t.request(
                    "POST", f"/agent/sessions/{args.session_id}/local-files",
                    {"files": candidates.files},
                )
            return 0
        if args.command == "watch":
            sid, watch = args.session_id, True
        else:
            value = args.goal_or_session
            if SESSION_ID.fullmatch(value):
                sid = value
                t.request("POST", f"/agent/sessions/{sid}/resume", {})
            else:
                view = t.request("POST", "/agent/sessions", {"goal": value})
                sid = view["session_id"]
                print(sid, flush=True)
            watch = args.watch
        return _stream(t, sid, poll=settings.agent_cli_poll_seconds,
                       watch=watch, interval=args.interval)
    finally:
        if hasattr(t, "close"):
            t.close()


def main(argv: list[str] | None = None) -> int:
    p = _parser()
    raw = list(sys.argv[1:] if argv is None else argv)
    commands = {"list", "approve", "reject", "edit", "mode", "cancel", "audit", "watch", "data"}
    # A bare goal or session id is the common form; argparse uses an internal `run` command.
    if not any(arg in commands for arg in raw):
        for i, arg in enumerate(raw):
            if not arg.startswith("-") and (i == 0 or raw[i - 1] not in
                                            ("--api-url", "--user", "--interval")):
                raw.insert(i, "run")
                break
    args = p.parse_args(raw)
    if args.command is None:
        p.error("provide a goal, session id or subcommand")
    if not 5 <= args.interval <= 3600:
        p.error("--interval must be between 5 and 3600")
    try:
        return _run(args)
    except TransportError as exc:
        print(str(exc), file=sys.stderr)
        return 4 if exc.status in (403, 404) else 3 if exc.status is None else 1
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    except Exception:
        print("Unexpected CLI error", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
