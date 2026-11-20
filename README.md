# ML Factory 2.0

Deterministic EDA, leakage screening and classical-ML baselines, running on AWS SageMaker with S3
as the only artifact store, exposed to LibreChat through a custom MCP server.

The ML engine (`ml_engine/`) and stage entrypoints (`jobs/`) are inherited from `../old_repo`
and are tracked by `tests/inherited_manifest.json`. Spec: `.kiro/specs/ml-factory-2/`.

Current work (the ML Assistant), decisions and the to-do list: see `AGENTS.md`.

## Setup (Windows, PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -e ".[jobs,models,dictionary,backend,mcp,dev]"
.venv\Scripts\python -m pytest          # offline, no AWS needed
.venv\Scripts\python -m ruff check .
```

Real-AWS tests are opt-in: set `ML_FACTORY_AWS_TESTS=1` and run `pytest -m aws`.

## Serve the MCP (for LibreChat)

```powershell
.venv\Scripts\python -m mcp_server.app      # reads ML_FACTORY_* from the environment
```

Serves the MCP tools over **SSE at `/sse`** (port 8000) and a health check at `/health`.
Transport and path can be changed with `ML_FACTORY_MCP_TRANSPORT` / `ML_FACTORY_MCP_PATH`.

The same process runs the **status poller** that moves experiments from stage to stage
(EDA → feature review, preparation → training → evaluation). Keep exactly one poller per artifact
root: on any additional replica set `ML_FACTORY_STATUS_POLL_INTERVAL_SECONDS=0`.

Deploying on an internal server next to LibreChat: see `infra/server/`. IAM roles: `infra/iam/`.
LibreChat config: `infra/librechat/`.
