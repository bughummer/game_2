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

## Status (as of 2026-11-23)

All 125 tasks in `.kiro/specs/ml-assistant/tasks.md` are marked done, optional tests included.
The offline suite passes (`pytest -q`: no failures, 2 skipped real-AWS tests) and `ruff check .`
is clean. Built: contracts, settings, Converse client, session store, tool catalogue, loop with
approvals, Autopilot, limits, cost estimates, job waiting and crash recovery, cancel, narration
grounding, audit trail, local uploads, API routes, smoke harness, the `mlf` CLI (`mlf --help`
works) and the web chat files (`assistant.js`, `assistant.css`, registered in `static.py` and
`index.html`).

**Never run against real AWS.** Whether the Bedrock model supports tool use, whether IAM allows
everything, and whether a real model follows the system prompt are all unverified. Nothing has
been tried in a browser or with a real terminal session either.

## To-do, in order

1. **Real-AWS smoke test** (costs at most 5 USD). In cmd, from `ml_factory_2`, with AWS
   credentials set and `ML_FACTORY_BEDROCK_MODEL_ID` set to a tool-use model:
   ```
   set ML_FACTORY_AWS_TESTS=1
   set ML_FACTORY_AGENT_SMOKE=1
   .venv\Scripts\python -m scripts.agent_smoke
   ```
   Fix whatever it finds (model tool support, IAM, prompt quality).
2. **Try it by hand**: start the server with `ML_FACTORY_AGENT_ENABLED=true`, run
   `mlf "your goal"`, then open the web UI and use the Assistant panel.
3. **Review the system prompt** (`backend/agent/prompts.py`) after seeing a real session.
4. **Deployment**: there is no Dockerfile or compose file; the project deploys as the systemd
   service in `infra/server/`. Add container files only if needed. Add the agent model ARN to
   `infra/iam/control_plane_policy.json` if the agent model differs from the reasoning model, and
   a CORS rule on the artifact bucket allowing PUT from the console origin (browser uploads).
5. **Commit**: the repo has no commits and no remote yet.

Known gaps to keep in mind:
- `scripts/agent_smoke.py` calls the private `AgentService._stop_jobs`; a public method would be
  cleaner.
- `RowCounter` reads the private `DatasetService._s3`.
- Not checked: whether the engine's own hyperparameter search is reachable through the tools.
- Sessions assume one backend process; several replicas would weaken the session lease.
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
