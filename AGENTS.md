# Notes for AI coding agents (Codex, Kiro, Claude, ...)

Read this first. It covers the project aim, the decisions already made, where the work stands
and what is left. The task list itself lives in `.kiro/specs/ml-assistant/tasks.md`; this file
summarises it and records what that file does not.

## Project aim

ML Factory 2 is an ML platform on AWS: SageMaker for all compute (Managed Spot for training),
S3 for all artifacts. It already does deterministic EDA, leakage screening, feature proposals,
training, comparison, validation and reports, exposed as REST routes and MCP tools.

The current feature is the **ML Assistant**: a chat agent for classical ML. The user states a
goal in plain language; the agent plans, does EDA, suggests features, flags possible leakage,
trains and compares models, validates, and reports results. New algorithms, grid searches,
time series and segmentation go through **free-form scripts** (model-written Python with a pip
package list, run on SageMaker). The agent does everything, but with the user's approval.

How it works: a Bedrock Converse tool-use loop (`backend/agent/`) whose tools wrap the existing
services. Two surfaces share one service: the `mlf` CLI and a chat panel in the static web UI.
Sessions are persisted in S3, so they survive long jobs and restarts.

Full spec: `.kiro/specs/ml-assistant/requirements.md`, `design.md`, `tasks.md`.

## Decisions made by the owner (do not change without asking)

- The agent model reuses `ML_FACTORY_BEDROCK_MODEL_ID` (fallback when `ML_FACTORY_AGENT_MODEL_ID`
  is empty).
- Default session limits: 40 model turns, 60 tool calls, 10 costly calls, 2 h active time,
  10 USD estimated spend, 3 identical calls in a row.
- Autopilot is allowed by default; an operator can switch it off with
  `ML_FACTORY_AGENT_AUTOPILOT_ALLOWED=false`.
- Approval requests expire after **15 minutes** (900 s).
- The web chat uploads by browser file picker only; the server never reads a local path.
- `create_experiment` is a costly tool (it starts EDA).
- Active time counts only time in `running`.
- A call the guards cannot price is never run unattended; Autopilot asks for approval.
- Unverified numbers in the final answer are marked, not removed.
- The agent can touch only experiments that are linked to its session and owned by the user.
- The server never resumes a session on its own after a job ends; the CLI or web panel does.
- The idempotency key of a costly call includes the approval id.
- The spend limit is decimal USD; other limits are integers.
- **Free-form runs always need the user's approval, in every mode including Autopilot**, and the
  approval shows the script code and its content hash.
- Time series and segmentation stay in free-form scripts; their metrics are labelled
  `self_reported`.

## Platform rules (enforced by tests)

- SageMaker-only compute, S3-only artifacts. No local database, job runner or store.
- Every number shown to the user comes from the engine, never from the model.
- The model never sees raw data rows, storage URIs, signed URLs or credentials.
- New artifact paths are defined only in `ml_engine/io/layout.py`.
- Only `backend/config.py` reads environment variables (`ML_FACTORY_*` prefix).
- `ml_engine/` imports nothing from `backend/`; `backend/` imports `boto3` only lazily.
- Inherited engine files change only additively and are recorded in
  `tests/inherited_manifest.json` (`scripts/inherited_manifest.py`).
- The default test suite runs offline with fakes and blocks sockets in `tests/agent/`.

## Where things are

| Path | What |
|---|---|
| `backend/agent/` | Agent: loop, service, store, policy, limits, estimate, tools, narration, audit, uploads, watcher |
| `backend/agent/tools_*.py` | Tool catalogue, one module per area, each with `register(registry, services)` |
| `ml_engine/contracts/agent.py` | Session, approval, audit contracts |
| `backend/api/routes/agent.py` | `/api/agent/...` routes |
| `ml_factory_cli/` | `mlf` CLI (partly built) |
| `scripts/agent_smoke.py` | Opt-in real-AWS smoke test harness |
| `tests/agent/` | Agent tests |
| `infra/server/` | systemd unit and `ml-factory.env.example` (the deployment settings) |

## Status (as of 2026-11-21)

The required ML Assistant implementation is complete: agent service, API, `mlf` CLI and web
chat. The session-store corrupt-record fix is in place. The inherited manifest is current.
The optional ML Assistant coverage is implemented in the offline suite. The task list links
each item to its tests. The full offline suite and `ruff check .` pass. The local test environment uses scikit-learn
1.9.0 because the byte-for-byte golden fixture records that version.

**Never run against real AWS.** Whether the Bedrock model supports tool use, whether IAM allows
everything, and whether a real model follows the system prompt are all unverified.

## To-do

- **Do not run the real-AWS smoke test.** Model tool-use support, IAM permissions and behavior
  against a real model remain unverified. The owner's no-real-AWS instruction above applies.

Known gaps to keep in mind:
- `scripts/agent_smoke.py` calls the private `AgentService._stop_jobs` to stop jobs without
  cancelling; a public method would be cleaner.
- `RowCounter` reads the private `DatasetService._s3`.

## Running things

```powershell
.venv\Scripts\python -m pytest -q            # offline suite
.venv\Scripts\python -m pytest tests/agent -q
.venv\Scripts\python -m ruff check .
.venv\Scripts\python -m mcp_server.app       # serves API, MCP and web UI on port 8000
```

Use `.venv\Scripts\python`; the system Python lacks test dependencies such as `moto`.

To turn the agent on, set `ML_FACTORY_AGENT_ENABLED=true` in the server's environment (for the
systemd setup, the env file next to `infra/server/ml-factory.env.example`) and restart.
Keep real secrets in `.env` files, which `.gitignore` excludes.
