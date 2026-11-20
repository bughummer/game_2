---
inclusion: always
---

# Architecture steering — ML Factory 2.0

A new cloud-native project. The deterministic engine is inherited from `../old_repo` and is the
specification, not code to improve. SageMaker is the only execution backend; S3 is the only
artifact store. There is no local execution path.

## Dependency direction (enforced by tests/test_architecture_boundaries.py)

```
mcp_server → backend → { ml_engine.contracts, ml_engine.io, AWS adapters (backend/aws) }
jobs       → ml_engine
ml_engine  → pandas/numpy/sklearn/model libs only
```

- `ml_engine` imports no `fastapi`, `backend`, `jobs`, `mcp_server`, SageMaker SDK, and no
  `boto3` at module import time. `ml_engine/io/s3.py` receives an injected boto3 client.
- The backend never imports a stage (`jobs/*`) and never runs heavy compute in a request handler.
  Bounded exception: reading an upload's header rows for the column list.
- The MCP wraps backend services; it holds no business logic or ML algorithm.

## One store, one orchestrator

- **Storage**: always `S3ObjectStore`. No scheme-based selection, no filesystem backend, no
  `file://`. It is constructed in exactly two places: `jobs/_common/runtime.py::build_store` and
  `backend/container.py`. The `ObjectStore` Protocol stays so tests can use
  `tests/support/InMemoryObjectStore`.
- **Orchestration**: `SageMakerOrchestrator` (boto3), graduating to the Step Functions
  implementation, both behind `ExperimentOrchestrator`. No queue, no worker. `start_*` returns an
  `ExecutionHandle` immediately. Never SageMaker Pipelines.

## Stage entrypoints

`jobs/{profiling,preprocessing,training,evaluation}/main.py` take `--experiment-id` and
`--artifact-base s3://...` and run inside the stage container as SageMaker Processing or Training
jobs. One training invocation = one model = one Training Job.

## Contracts and artifacts

- Every artifact is a versioned Pydantic contract in `ml_engine/contracts`; the API and MCP reuse
  them verbatim.
- Artifact paths are defined only in `ml_engine/io/layout.py`.
- An experiment is a self-contained S3 prefix. No relational database without justification.
- One writer per document. Parallel runs write per-run records; one aggregation step composes
  shared views.

## Inherited code

Files copied from `old_repo` are tracked in `tests/inherited_manifest.json`. Do not edit them to
accommodate AWS — add an adapter outside `ml_engine`. If an edit is unavoidable, stop, explain why,
and update the manifest in the same reviewed change.
