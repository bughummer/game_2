# Decision: run on a stock AWS SageMaker image (Option B)

Status: accepted by the owner, not yet implemented. This records the decision and the scope so
the change can be done as one reviewed spec rather than piecemeal. It supersedes the custom-image
decisions in `.kiro/specs/ml-factory-2/design.md` (Q5, DD-7) where they conflict.

## What changes

The platform stops building and shipping a custom stage image. Every SageMaker Processing and
Training job runs **one AWS-prebuilt image** — the SageMaker XGBoost image, which also ships
scikit-learn, pandas, numpy and scipy — named by its ECR URI. Nothing is built or pushed by us.

On top of that image, a job may **install additional user-approved packages at job start** (a
controlled emulation of the notebook `pip install` workflow), so the model catalogue is no longer
closed to sklearn + XGBoost. See the "Controlled package install" section and decisions B10–B13.

"Calling SageMaker" is unchanged: the orchestrator still submits `create_training_job` /
`create_processing_job` with an image URI, a role, an instance type and
`ContainerEntrypoint = python -m jobs.<stage>.main`. Only the image URI changes (from our ECR
image to AWS's), and the versions inside it are AWS's, not ours.

## Owner decisions

| # | Decision |
|---|---|
| B1 | One AWS-prebuilt image for all stages (the SageMaker XGBoost image, which includes scikit-learn). No custom image, no `docker build`, no ECR push by us. |
| B2 | **Superseded by B10.** (Was: drop CatBoost, closed catalogue of sklearn + XGBoost.) The catalogue is no longer closed — any user-approved package may be installed per experiment. CatBoost/LightGBM/etc. are reachable through B10 without a custom image. Built-in plugins still ship for the core models; others arrive as installed packages. |
| B3 | Keep XGBoost, at whatever version the AWS image ships unless an experiment pins a different version via its approved requirements file (B11). |
| B4 | Drop Spot checkpoint/resume entirely. Every model is restart-on-interruption. No `CheckpointConfig` is set on any job. Managed Spot (the discount) is kept — and it works on the training-job route (notebooks cannot use Spot; training jobs can). |
| B5 | **Amended by B11.** Reproducibility = the recorded **image digest** PLUS the **resolved `requirements.txt` saved on every experiment** (the exact installed versions, captured after install). An experiment is reproducible from (digest + saved requirements), not from the image digest alone. |
| B6 | **No exact version pins.** The owner does not want library versions fixed to any current set. `pyproject.toml` uses loose lower-bound ranges only where the code needs a specific API (e.g. `pandas>=2`, `scikit-learn>=1.3`), or no bound at all; `requires-python` becomes a loose floor. `pip install` inside the AWS image keeps the versions the image already ships; locally, pip resolves current ones. No `==` pins for the ML libraries. |
| B7 | The LLM reasoning layer (feature proposals, four-signal leak reasoning, governed interpretation) is unchanged. It does not depend on the image or the model set. |
| B8 | **The control plane resolves the image URI itself; no one manages it by hand.** `stage_image_uri` becomes optional. When unset, the orchestrator resolves the AWS XGBoost image at submission with `sagemaker.image_uris.retrieve(framework="xgboost", region=aws_region)` and records the resolved digest on the run. Operators set only the region. The URI is a deterministic lookup; per B12 the LLM may *propose/surface* it but the resolved value and any override are confirmed by the platform/user, never authored unilaterally by the LLM. An explicit `stage_image_uri` still overrides, for pinning a specific digest. |
| B9 | **Read-only `resolve_stage_image` MCP tool.** The assistant (Kiro / LibreChat) can ask the platform, on demand, which image a run would use; the tool calls the same resolver as B8 and returns the URI. It is read-only (`readOnlyHint`), performs no AWS mutation, and starts nothing. Training still goes only through `start_training`, which resolves the image itself and runs the guards. |
| B10 | **Controlled per-experiment package install (open catalogue).** An experiment may carry a user-authored **`requirements`** list — its own `requirements.txt`, exactly like the one you'd use in a notebook. At job start, on top of the stock image, the job installs those packages before the `python -m jobs.<stage>.main` entrypoint runs. This emulates the notebook `pip install` workflow and reopens arbitrary models (CatBoost, LightGBM, anything on PyPI) with **no custom image and no ECR push by us**. The list is a user input; when empty, nothing is installed and the run behaves as B-one. (No separately-named `extra_packages` field — the concept is just the experiment's requirements file.) |
| B11 | **Reproducibility via a saved per-experiment requirements file.** After the install resolves, the job captures the exact installed versions (`pip freeze`) and saves that `requirements.txt` as an experiment artifact (path defined only in `ml_engine/io/layout.py`), recorded on the model-version record alongside the image digest. A later run reproduces an experiment from (image digest + saved requirements.txt). The owner does not pre-pin versions (B6 stands for authoring); the pin is *captured after the fact* and frozen per experiment. |
| B12 | **LLM boundary reworded, platform-wide: "the LLM always consults and waits for user approval."** The former rule "the LLM never decides infrastructure" is removed. The LLM MAY now propose infrastructure and treatment choices — the package list, the image, the instance type, outlier/feature/grid plans — but SHALL present each as a proposal and SHALL NOT act until the user approves. The guarantee shifts from "LLM is forbidden to touch X" to "nothing the LLM proposes is applied without explicit user approval." The platform still executes only through guarded, audited service calls; approval is recorded in the audit trail. This supersedes the "LLM never decides infrastructure" wording wherever it appears (B8, Req 9.6 narration, and every spec's LLM-boundary clause). |
| B13 | **Install is open by default, validated and audited — not restricted.** The platform is used by the owner and trusted people, so there is **no allowlist/denylist and no package-count cap** by default: any user-approved package installs. The only checks are non-restricting: package names/versions must be well-formed (rejects typos before submission, not choices), the existing time/budget guards bound cost, and the experiment's `requirements` list plus who approved it is recorded in the audit trail. A failed install fails the job before training as a structured error — it never silently falls back to a different package set. An allowlist exists in the config but is **off unless an operator explicitly turns it on** (for a future less-trusted deployment); it is not enforced in the trusted default. |

## Controlled package install (B10–B13), in full

This is the "emulate my Jupyter workflow" route, made safe for a multi-run platform.

- **Mechanism.** The stock framework image supports a `requirements.txt` / dependencies list that
  SageMaker `pip install`s before the entrypoint. We pass the user's approved `extra_packages`
  that way. No custom image, no build, no ECR push by us. Our engine/plugin contract is unchanged:
  an installed model is still trained, saved as a Model_Version, and independently validatable.
- **Spot.** Because this runs as a **training job** (not an interactive notebook), Managed Spot
  applies and gives the discount. The literal notebook workflow you described cannot use Spot;
  this route can, so it is strictly better on cost for the same ergonomics.
- **Reproducibility (B11).** Authoring stays loose (B6: the owner does not pin versions). The job
  captures `pip freeze` *after* resolving the install and saves it per experiment. Reproduce =
  replay (image digest + that saved file). This is how you get "install anything" *and*
  reproducibility — the pin is recorded, not pre-declared.
- **Cost of the route (accepted):** every job with `extra_packages` pays install time (minutes for
  native-wheel packages, multiplied by K folds / grid points), needs working PyPI access through
  the corporate proxy at job start, and widens the supply-chain surface (arbitrary PyPI code runs
  with the job role). For a trusted-user deployment this is accepted: there is no allowlist by
  default (B13); the existing time/budget guards bound cost, and well-formedness validation catches
  typos. A run with no `extra_packages` pays none of this and behaves exactly as B-one.
- **Open items to verify at first real-AWS run:** the stock image honours the install step before
  our entrypoint under the proxy; `pip freeze` capture round-trips to S3; a native-wheel package
  (e.g. lightgbm) installs cleanly on the image's stack.

## What does NOT change

- SageMaker-only execution; S3-only artifacts; the orchestrator, guards, tracking, status
  monitor, MCP server, leakage investigation, reporting, audit.
- Direct S3 I/O from the stages and the `python -m jobs.<stage>.main` entrypoint (confirmed it
  works on a stock image; we do not adopt the SageMaker channel model).
- The deterministic engine and its contracts, except the two edits below.
- The corporate HTTPS proxy support already in `Settings.https_proxy` / `backend/aws/clients.py`.

## Spot-resume, defined (for the record)

Spot instances are discounted spare capacity AWS can reclaim mid-job. "Spot-resume" meant a model
periodically checkpointed its progress to S3 so an interrupted job continued from the last
checkpoint instead of from zero. B4 drops this: an interrupted Spot job restarts from scratch.
For classical models that train in minutes the only cost is occasionally wasted compute; there is
no correctness impact. Managed Spot itself (the discount) is kept.

## Scope of the implementation (a later task)

Spec / docs:
- `.kiro/specs/ml-factory-2/design.md`: revise Q5 (ECR required) and DD-7 (one custom image);
  revise the reproducibility property (Req 3.6 / Property 5) to image-digest; drop CatBoost from
  the model list; mark Req 4.3 (checkpoint resume) not applicable.
- `.kiro/specs/cross-validation/design.md` + requirements.md: the CV plugin set loses CatBoost;
  the resumability note (Req 9.4 "resume an interrupted CV Run without refitting completed folds")
  still works at the *fold* level via the per-fold artifacts we persist, but native checkpoint
  resume inside a fold is gone — reword to fold-level only.

Engine (inherited, manifest-protected — additive/reviewed):
- `ml_engine/models/registry.py`: remove the two CatBoost plugins from the registration loop.
  Delete or keep-but-unregister `ml_engine/models/catboost_plugin.py` (prefer unregister + leave
  the file, or delete it and drop it from the manifest). Manifest update + justification.
- `ml_engine/contracts/execution.py`: `RESUMABLE_MODELS` becomes empty (B4). Keep the constant so
  the orchestrator/checkpointing imports still resolve.

Jobs / orchestration:
- `jobs/training/checkpointing.py`: with `RESUMABLE_MODELS` empty it becomes a no-op; keep the
  module (the training job imports `checkpointing_factory`) or make the training `--checkpoint-dir`
  path inert. The CV fold loop's own per-fold persistence/resume (`jobs/training/cv.py`) is
  unaffected — that is our artifact-level resume, not native checkpointing.
- `backend/orchestration/sagemaker.py`: stop setting `CheckpointConfig` (no resumable models);
  `_training_plan` no longer scopes a checkpoint path. Managed Spot stays.

Config / infra:
- `backend/config.py`: `stage_image_uri` becomes **optional** (B8). `problems()` no longer
  requires it. Add a small resolver (e.g. `backend/aws/images.py::resolve_stage_image(settings)`)
  that returns `settings.stage_image_uri` if set, else calls
  `sagemaker.image_uris.retrieve(framework="xgboost", region=settings.aws_region, version=...)`.
  The sagemaker SDK import is lazy (it is an AWS dependency), kept out of `ml_engine` per the
  architecture boundary. The orchestrator calls the resolver at submission and records the
  resolved URI/digest on the execution (it already stores `image_uri` on `ExecutionRecord`).
  `pyproject.toml` `backend` extra gains `sagemaker` for `image_uris` (SDK, not a stage dep).
- `pyproject.toml`: drop `catboost`; replace the `==` pins with loose lower bounds (or none) so
  the AWS image's own versions are used inside the image and local installs resolve current
  versions (B6). `requires-python` becomes a loose floor. No version is fixed to the owner's
  current set.
- `container/Dockerfile` + `.dockerignore`: delete, or keep only as a developer convenience with
  a loud note that production uses the AWS image. `infra/iam`: the job role still needs ECR pull
  for the AWS image (the pull is from AWS's registry, permissions are the same shape).

MCP (B9):
- `mcp_server/server.py`: add a read-only `resolve_stage_image` tool that returns the resolved
  stage image URI (and whether it came from config or the AWS lookup) via the B8 resolver.
  Annotated `readOnlyHint`; no ownership mutation; covered by the in-process FastMCP client test
  (the resolver is stubbed so the test needs no AWS). Update the tool-discovery test's expected
  set.

Tests:
- Remove/limit CatBoost assertions; the "only resumable models get a checkpoint" test inverts to
  "no job sets a checkpoint"; the reproducibility test relaxes from version-match to digest;
  `tests/infrastructure/test_stage_image.py` (static Dockerfile checks) is dropped or repurposed.
- New: the image resolver (explicit URI passes through; unset → `image_uris.retrieve` called with
  the configured region, stubbed in tests) and the `resolve_stage_image` MCP tool.

## Open items (resolved at first real-AWS run, not by pinning versions)

No exact versions are needed up front (B6: the pins are loose, the image's versions win). The
remaining unknowns are verified by the first real-AWS run rather than by the owner pasting
versions:

1. **The AWS XGBoost image must include scikit-learn** at a version the sklearn plugins
   (logistic_regression, elastic_net, random_forest) run on. If it does not, Option B needs two
   images (sklearn image + XGBoost image), which reopens DD-7. Verified by running one sklearn
   model and one XGBoost model on the image once.
2. **Entrypoint override:** confirm the AWS image accepts a plain `ContainerEntrypoint`
   (`python -m jobs.<stage>.main`) rather than forcing the SageMaker training toolkit. Expected to
   work for Processing jobs; verify for Training jobs.
3. **Our code installs cleanly on top of the image's stack** with loose pins (no resolver
   conflict against the image's preinstalled versions). Verified by the install step of the first
   job, or a one-off `pip install` against the image.

None of these block writing the spec or the code; they are confirmations for the first run.

## Considered and rejected: per-framework AWS images for CatBoost / LightGBM

AWS does ship built-in CatBoost and LightGBM support, but **not inside the XGBoost image** — each
is a separate AWS prebuilt image resolved with its own framework name
(`image_uris.retrieve(framework="catboost"|"lightgbm", ...)`), and the built-ins run AWS's own
script-mode entrypoint rather than our `python -m jobs.<stage>.main`. Allowing them would therefore
mean multiple images and digests per experiment (one per framework), a per-model image resolver,
and a first-run check that each image can import its package under our own entrypoint. The owner
considered this ("B-multi"), rejected it, briefly chose "B-one" (closed sklearn + XGBoost
catalogue), then **superseded both with B10** (controlled per-experiment install). B10 reopens
CatBoost/LightGBM/anything on PyPI without per-framework images and without a custom image, keeping
one base image (XGBoost) plus an approved install layer. Per-framework AWS images are still not
used: they force AWS's script-mode entrypoint and multiple base digests for no gain once B10
exists. B1 stands as the *base* image; B10 adds the install layer on top.

## Recommended sequencing

Do the cross-validation feature's remaining tasks (8–13) **first** or decide to pause them: this
migration changes the model set and the reproducibility rules that CV tests assert, so landing it
mid-CV would churn the same files twice. Then implement this spec as one change with the manifest
and the full suite green.
