"""Thin HTTP and direct-service transports for the ML Assistant CLI."""

from __future__ import annotations

from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit


class TransportError(Exception):
    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


class AgentTransport(Protocol):
    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any: ...


class HttpTransport:
    def __init__(self, base_url: str, user: str | None = None) -> None:
        import httpx

        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=30)
        self._headers = {"X-ML-Factory-User": user} if user else {}

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        import httpx

        try:
            response = self._client.request(
                method, "/api" + path, json=body if method != "GET" else None,
                headers=self._headers,
            )
        except httpx.RequestError as exc:
            raise TransportError("Control plane is unreachable") from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise TransportError(
                "Control plane returned an invalid response", status=response.status_code
            ) from exc
        if response.is_error:
            raise TransportError(
                payload.get("message", "Request failed"), status=response.status_code
            )
        return payload

    def close(self) -> None:
        self._client.close()


class InProcessTransport:
    """Use the same AgentService as the API, with AWS construction delayed until selected."""

    def __init__(self, settings: Any) -> None:
        from backend.aws import build_session
        from backend.container import build_container
        from ml_engine.io import S3ObjectStore

        session = build_session(settings)
        s3 = session.client("s3")
        container = build_container(
            settings,
            store=S3ObjectStore(
                s3, server_side_encryption=settings.s3_sse, kms_key_id=settings.s3_kms_key_id
            ),
            s3_client=s3,
            sagemaker_client=session.client("sagemaker"),
            logs_client=session.client("logs"),
        )
        self._agent = container.agent
        self._who = container.authorizer.principal(settings.dev_identity)
        self._settings = settings

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        from backend.errors import AppError

        body = body or {}
        parsed = urlsplit(path)
        parts = parsed.path.strip("/").split("/")
        if parts[:1] != ["agent"]:
            raise TransportError("Unknown endpoint", status=404)
        agent, who = self._agent, self._who
        try:
            if parts == ["agent", "config"] and method == "GET":
                return {"enabled": self._settings.agent_enabled,
                        "cli_poll_seconds": self._settings.agent_cli_poll_seconds}
            if parts == ["agent", "sessions"]:
                if method == "POST":
                    out = agent.create_session(
                        who, body.get("goal"), mode=body.get("mode", "supervised"),
                        autopilot_max_spend_usd=body.get("autopilot_max_spend_usd"),
                    )
                else:
                    q = parse_qs(parsed.query)
                    out = agent.list_sessions(
                        who, limit=int(q.get("limit", [100])[0]),
                        cursor=q.get("cursor", [None])[0],
                    )
            elif len(parts) >= 3 and parts[:2] == ["agent", "sessions"]:
                sid = parts[2]
                tail = parts[3:]
                if not tail and method == "GET":
                    after = int(parse_qs(parsed.query).get("after_seq", [0])[0])
                    out = agent.get_session(who, sid, after_seq=after)
                elif tail == ["resume"]:
                    out = agent.resume(who, sid)
                elif tail == ["messages"]:
                    out = agent.send_message(who, sid, body.get("text"))
                elif len(tail) == 2 and tail[0] == "approvals":
                    out = agent.decide_approval(
                        who, sid, tail[1], body.get("decision"), args_hash=body.get("args_hash"),
                        edited_args=body.get("edited_args"), reason=body.get("reason"),
                    )
                elif tail == ["mode"]:
                    out = agent.set_mode(
                        who, sid, body.get("mode"),
                        max_estimated_spend_usd=body.get("max_estimated_spend_usd"),
                    )
                elif tail == ["cancel"]:
                    out = agent.cancel(who, sid, stop_jobs=body.get("stop_jobs", False))
                elif tail == ["audit"]:
                    records = [r.model_dump(mode="json") for r in agent.audit_trail(who, sid)]
                    out = {"session_id": sid, "records": records}
                elif tail == ["local-files"]:
                    out = agent.register_local_files(who, sid, body.get("files"))
                elif tail == ["client-action"]:
                    out = agent.client_action(who, sid)
                elif len(tail) == 3 and tail[0] == "client-action" and tail[2] == "complete":
                    out = agent.complete_upload(who, sid, tail[1])
                else:
                    raise TransportError("Unknown endpoint", status=404)
            else:
                raise TransportError("Unknown endpoint", status=404)
        except AppError as exc:
            status = 404 if exc.code in ("NOT_FOUND", "NOT_AUTHORIZED") else 400
            raise TransportError(exc.message, status=status) from exc
        return out.model_dump(mode="json") if hasattr(out, "model_dump") else out
