---
inclusion: always
---

# Testing steering — ML Factory 2.0

No local execution does not mean tests need the cloud. The default suite runs offline with no AWS
account; never weaken a test to make a change pass.

## Ground rules

- Run the suite before and after every task. New behavior ships with its test in the same change.
- Tests live at `tests/<package>/test_<module>.py`, mirroring the source layout.
- The inherited engine tests from `old_repo` must stay green; only their fixtures changed (the
  local store was replaced by `tests/support/InMemoryObjectStore`).

## AWS is mocked by default

- `moto` for S3 and CloudWatch Logs; botocore `Stubber` for SageMaker and Step Functions request
  shapes; `tests/support/InlineOrchestrator` runs stage entrypoints in-process against the
  in-memory store.
- Real-AWS tests are opt-in behind `ML_FACTORY_AWS_TESTS=1` and are never required for the
  default suite.
- The store contract suite runs against both `InMemoryObjectStore` and `moto`-backed
  `S3ObjectStore`; the two must behave identically.

## Reproducibility

- Default suite: the same frozen config run twice in-process gives identical primary metrics.
- Opt-in: two SageMaker runs agree within 1e-6 absolute or 0.1% relative.

## What each area must test

- **Storage**: S3 store semantics, URI validation, byte-identical round-trip, listing, missing
  objects, SSE parameters.
- **SageMaker**: exact request params, Spot defaults, checkpoint config only for resumable models,
  guard rejections (instance, concurrency, budget, On-Demand without opt-in), no endpoint calls,
  per-model isolation, non-blocking start, stop/cancel.
- **Tracking**: run relationships, status normalization incl. Unknown, savings formula edge cases,
  missing billing data, labelled estimates with price source, monthly rollups, concurrent writers.
- **MCP**: discovery, validation, authorization, async behavior, bounded outputs, structured errors.
- **ML correctness**: fit on train only, target exclusion, inherited leakage rules unchanged, spec
  validation, approval vs. selection, CV integrity.
- **Reasoning**: hallucinated columns dropped, invalid specs rejected, unsupported claims refused,
  schema validation, governance gate limits what is sent.

## Regression gates

- Correctness Properties 1–11 in `.kiro/specs/ml-factory-2/design.md` each have an asserting test.
- The architecture-boundary test and the inherited-file manifest test are gates.

## Tooling

- `make test` (pytest), `make lint` (ruff check + format check). Seed all randomized tests.
