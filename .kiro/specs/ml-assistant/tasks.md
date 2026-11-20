# Implementation Plan: ML Assistant

## Overview

The implementation language is Python 3.x, as used in `design.md`. The plan follows the delivery order in the design (Req 17.8):

1. Contracts, layout and settings.
2. Converse client, tool registry with a few Read tools, and a minimal loop.
3. The opt-in Smoke_Test.
4. The full catalogue, approvals, limits and uploads.
5. The API.
6. The CLI.
7. The web chat, last.

All new backend code goes under `backend/agent/`. Contracts go in `ml_engine/contracts/agent.py`, and artifact paths go only in `ml_engine/io/layout.py`. Tests go under `tests/agent/` and use `hypothesis` with `max_examples=100, deadline=None`. Each property test carries the comment `# Feature: ml-assistant, Property N: <title>`. The default suite runs offline with `ScriptedFakeConverseClient`, the in-memory store and the fake SageMaker.

Tool modules are split into `backend/agent/tools_*.py`. Each exposes `register(registry, services)`, and `backend/agent/catalogue.py` aggregates them. This lets catalogue tasks run in parallel without touching the same file.

## Tasks

- [x] 1. Contracts, layout and settings
  - [x] 1.1 Create versioned agent contracts in `ml_engine/contracts/agent.py`
    - Define all models and enums from the design's Data Models section: `Mode`, `SessionStatus`, `ToolClass`, `CallState`, `ApprovalStatus`, `SessionLimits`, `SessionCounters`, `AutopilotGrant`, `Plan`, `ToolCallRecord`, `ResultRef`, `ApprovalRequest`, `ApprovalDecision`, `AwaitedJob`, `DatasetRef`, `LocalFileRef`, `ClientAction`, `TranscriptEntry`, `Attribution`, `AnswerSentence`, `VerbatimBlock`, `FinalAnswer`, `Lease`, `Session`, `SessionSummary` and `AgentAuditRecord`.
    - Also define `SessionFailure`, `GuardRefusal`, `CompactionInfo` and `LedgerEntry`. `LedgerEntry` lives here so that `results.py` and `narration.py` do not import each other.
    - Every contract carries `schema_version = "1.0"`. Add `SUPPORTED_VERSIONS`.
    - Add the canonical `args_hash(tool, arguments)` and `idempotency_key(session_id, approval_id, args_hash)` helpers (standard library only).
    - Import only `pydantic` and the standard library.
    - _Requirements: 6.1, 4.9, 14.7, 15.1_

  - [x] 1.2 Add agent paths to `ml_engine/io/layout.py` and record them in the inherited manifest
    - Add `AGENT_NAMESPACE`, `owner_key`, and `AgentLayout` with `for_session`, `owner_index_prefix`, `owner_index_object`, `session`, `tool_result`, `inflight` and `audit_prefix`.
    - Change inherited paths in no way.
    - Update `scripts/inherited_manifest.py`: add `ml_engine/contracts/agent.py` to `NEW`, and extend the `ADAPTED` text for `layout.py` as the design states.
    - Re-record `tests/inherited_manifest.json`.
    - _Requirements: 6.1, 15.1, 15.3, 15.5_

  - [x] 1.3 Add agent settings to `backend/config.py`
    - Add every field in the design's settings table, with the stated defaults and no `ge`/`le` constraints.
    - Add `_agent_problems()`, called from `problems()`. It reports `ML_FACTORY_<NAME> must be <range> (got <value>)`, a missing model id when `agent_enabled` is true, and a region mismatch.
    - Add an effective-model-id property: `agent_model_id`, falling back to `bedrock_model_id`.
    - Existing field values and `problems()` output stay unchanged with defaults.
    - _Requirements: 13.2, 13.3, 13.4, 13.5, 13.6, 15.4_

  - [x] 1.4 Create the test infrastructure under `tests/agent/` and `tests/support/`
    - `tests/support/scripted_converse.py`: `ScriptedFakeConverseClient` with `ScriptStep(turn, expect=None)`. It raises an `AssertionError` that names an unexpected or post-exhaustion request, and records every request.
    - Add `FlakyStore`, `CountingStore`, `FakeClock` (with injectable `sleep`) and `InlineAgentRunner` stubs where the real class does not exist yet.
    - `tests/agent/strategies.py`: bounded generators for run scripts, user actions, JSON trees with injected keys, credentials, URIs and injection templates.
    - `tests/agent/conftest.py`: socket-blocking fixture and an `agent_service_factory` fixture. It uses `backend.agent.wiring.build_agent_service` once that exists.
    - `tests/fixtures/converse/*.json`: recorded response shapes for a text turn, a tool-use turn, throttling, an unsupported model and expired credentials.
    - _Requirements: 16.1, 16.2, 16.3, 16.16, 16.17_

  - [x]* 1.5 Write property test for settings validation
    - File: `tests/agent/test_settings.py`
    - **Property 47: Settings validation reports name and range**
    - **Validates: Requirements 13.2, 13.3, 13.4, 13.6**

  - [x]* 1.6 Write unit tests for contracts and layout
    - File: `tests/agent/test_agent_boundaries.py`
    - Check that `ml_engine/contracts/agent.py` imports only `pydantic` and the standard library.
    - Check that `AgentLayout` paths are built only in `layout.py`.
    - Check that `test_inherited_manifest.py` still passes.
    - _Requirements: 15.1, 15.3, 15.5, 16.6_

- [x] 2. Converse client
  - [x] 2.1 Implement `backend/agent/converse.py`
    - Define the neutral types: `ToolUseRequest`, `TokenUsage`, `ConverseTurn`, `TextBlock`, `ToolUseBlock`, `ToolResultBlock`, `ConverseMessage` and `ToolDefinition`.
    - Define the error classes: `ConverseError`, `ConverseTransientError`, `ConverseModelUnsupportedError`, `ConverseRequestError` and `ConverseCredentialsError`.
    - Define the `ConverseClient` Protocol.
    - Implement `BedrockConverseClient` with `temperature=0`, `maxTokens`, and exponential backoff (`backoff * 2**attempt`) that retries only transient errors.
    - The boto client is built lazily through `build_client("bedrock-runtime", settings, read_timeout=...)`.
    - Log no exception text, request or credential.
    - Leave `BedrockReasoningClient` and `ReasoningClient` unchanged.
    - _Requirements: 2.1, 2.2, 2.7, 2.8, 2.10, 13.7, 13.8, 13.9, 14.9, 15.8_

  - [x]* 2.2 Write property test for request and response mapping
    - File: `tests/agent/test_converse_client.py`
    - **Property 45: Converse requests and responses map faithfully**
    - **Validates: Requirements 2.1, 2.7, 2.10**

  - [x]* 2.3 Write property test for retry backoff
    - File: `tests/agent/test_converse_client.py`
    - **Property 46: Retry follows the configured backoff**
    - **Validates: Requirements 2.8**

  - [x]* 2.4 Write example tests for recorded Converse fixtures
    - File: `tests/agent/test_converse_fixtures.py`
    - Cover an unsupported model (error carries the model id, no retry), expired credentials, lazy client construction, and an unchanged `BedrockReasoningClient`.
    - _Requirements: 2.2, 13.7, 13.8, 14.9, 16.16_

- [x] 3. Checkpoint - Contracts, settings and Converse client
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 4. Session store, tool registry and minimal loop
  - [x] 4.1 Add audit writing in `backend/audit.py` and `backend/agent/audit.py`
    - Add `AuditWriteError` and `AuditLog.write(...)` to `backend/audit.py`. It returns the object URI and raises on failure, and returns `None` when auditing is disabled. `record` stays unchanged.
    - Implement `AgentAudit.write(session, rec)`: convert an `AgentAuditRecord` to an `AuditEvent` with action `agent.<event_type>`, plus the reverse conversion.
    - Keep a per-session `audit_seq`, and scrub credential-shaped strings from `detail`.
    - _Requirements: 11.1, 11.2, 11.4, 11.5, 11.8_

  - [x] 4.2 Implement `backend/agent/store.py`
    - Implement `SessionStore` with `create`, `load`, `save`, `transaction`, `put_result`, `get_result`, `list_for_owner`, `acquire_lease` and `release_lease`.
    - Define `SessionLoadError`, `SessionConflict`, `SessionPersistError` and `SessionInUse`.
    - `load` checks `schema_version` before validating, and never writes.
    - `save` retries 3 times, compares ETags, and writes the owner index.
    - The lease expires after `agent_lease_seconds`.
    - _Requirements: 6.1, 6.2, 6.8, 6.10, 6.14, 14.8, 15.2_

  - [x] 4.3 Implement `backend/agent/view.py`
    - Build `SessionView` from a session. It omits dataset URIs and signed URLs.
    - Include status, mode, counters, pending approvals, `running_jobs`, `jobs_ready` and transcript entries after `after_seq`.
    - _Requirements: 6.8, 6.9, 5.10_

  - [x]* 4.4 Write property test for session round trip and view determinism
    - File: `tests/agent/test_contracts_store.py`
    - **Property 1: Session round trip and view determinism**
    - **Validates: Requirements 6.1, 6.9, 15.1**

  - [x]* 4.5 Write property test for unsupported or corrupt records
    - File: `tests/agent/test_contracts_store.py`
    - **Property 2: Unsupported or corrupt records are rejected untouched**
    - **Validates: Requirements 14.8, 15.2**

  - [ ]* 4.6 Write property test for session listing
    - File: `tests/agent/test_contracts_store.py`
    - **Property 28: Session listing**
    - **Validates: Requirements 6.8**

  - [x] 4.7 Implement `backend/agent/tools.py`
    - Define `ToolFlags`, `ToolSpec`, `ToolContext`, `ToolOutput`, `ToolRegistrationError` and `ToolRegistry`.
    - `register` is atomic. It requires exactly one distinct class and a new name. It refuses the reserved privileged names (`approve_request*`, `reject_request*`, `edit_request*`, `set_mode*`, `set_limits*`, `enable_autopilot*`, `promote*`, `delete*`, `approve_script*`).
    - `definitions()` derives the input schema from the Pydantic model.
    - _Requirements: 3.2, 3.3, 3.7, 4.11_

  - [x] 4.8 Implement `backend/agent/inputs.py` and `backend/agent/execute.py`
    - `inputs.check` rejects local paths, `s3://` URIs, signed URLs, disallowed instance types, unknown dataset ids, and experiment ids that are not linked or not owned by the session owner. It writes one `input_rejected` audit record per rejection.
    - The execution shell takes identity only from `Principal(user=session.owner)`.
    - It resolves dataset ids to URIs server-side and writes the experiment audit entry through `AuditLog.write`. It records `service_audit_ref`.
    - It maps `AppError`, `GuardViolation` and `OrchestrationError` to structured errors. An unexpected exception becomes a generic `TOOL_FAILED` with the tool name only.
    - _Requirements: 3.4, 3.9, 3.10, 7.3, 7.12, 8.10, 14.4_

  - [x] 4.9 Implement `backend/agent/results.py`
    - Implement `shape_result`, `envelope` and `detect_injection` with these steps in order: JSON-safe conversion, scrub, govern, injection detection, string truncation, size cap, ledger.
    - Scrub rewrites URIs to `artifact://` or `dataset:<id>` references, and redacts credentials and signed-URL parameters.
    - Govern removes row-shaped keys and then applies `apply_governance`.
    - The final envelope never exceeds `max_bytes`.
    - _Requirements: 3.6, 3.8, 8.1, 8.2, 8.3, 8.4, 8.7, 8.8, 8.9_

  - [x] 4.10 Implement a minimal `backend/agent/policy.py`
    - Define `GateAction`, `GateInput`, `GateDecision` and `gate`.
    - Implement rule 1 (plan required) and rule 2 (Read tools execute).
    - Every other tool class defaults to `REQUIRE_APPROVAL`, so the gate fails closed until task 8.1 completes the rules.
    - _Requirements: 1.3, 4.4_

  - [x] 4.11 Implement `backend/agent/catalogue.py` and `backend/agent/tools_core.py`
    - `catalogue.py` discovers the `tools_*.py` modules and builds the registry.
    - `tools_core.py` registers `submit_plan` and `ask_user` (control tools: 1 to 10 steps of at most 200 characters, one question).
    - It also registers the Read tools `list_datasets`, `list_experiments` (always owner-filtered, links the returned ids), `get_experiment` and `get_eda_summary`.
    - _Requirements: 1.3, 1.5, 3.1, 3.2, 3.5_

  - [x] 4.12 Implement `backend/agent/prompts.py` and `backend/agent/context.py`
    - `SYSTEM_PROMPT_VERSION` and `system_prompt(session, registry)` as the design describes. The prompt contains no user data.
    - `build_messages(session, budget_tokens)` rebuilds the messages from the transcript, with enveloped results and job updates.
    - Compaction is added in task 11.2.
    - _Requirements: 8.4, 8.5, 2.1_

  - [x] 4.13 Implement a minimal `backend/agent/narration.py`
    - Define `NumberLedger`, `verify_narration` and `build_final_answer`.
    - The minimal version marks every sentence that contains a digit as `unverified`, which fails safe until task 11.1.
    - Keep the public signatures stable so the loop does not change.
    - _Requirements: 1.7, 10.1, 10.2_

  - [x] 4.14 Implement `backend/agent/loop.py` and `backend/agent/runner.py`
    - Implement `AgentLoop.run` and `AgentLoop.step` as short transactions: load, one step, save. Model calls and handlers run outside a transaction.
    - Planning rules: only `submit_plan` and `ask_user` are available until a plan is accepted. A first turn with neither sets `failed` with "planning failed".
    - In a turn that contains `ask_user`, every Mutating or Costly use gets a `deferred_waiting_for_user` result.
    - Each model turn with N tool uses yields exactly N results in request order.
    - Unknown tools, invalid arguments and unapproved inputs yield structured errors without calling the handler.
    - Record the model id and token counts. A cancel or epoch change discards an in-flight model result.
    - A text-only turn sets `completed` with a verified final answer.
    - Handle `stop_reason = max_tokens` as described.
    - Persist after every produced item and stop on `SessionPersistError`.
    - Map Bedrock outage, `ConverseModelUnsupportedError` and `ConverseCredentialsError` to `failed` with the right `retriable` flag.
    - `runner.py` provides `InlineAgentRunner` and `ThreadAgentRunner` (de-duplicated per session).
    - _Requirements: 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 2.3, 2.4, 2.5, 2.6, 2.9, 2.10, 6.2, 6.14, 13.8, 14.9_

  - [x] 4.15 Implement a minimal `backend/agent/service.py` and `backend/agent/wiring.py`
    - `AgentService` has `create_session`, `send_message`, `resume`, `get_session`, `list_sessions` and a minimal `cancel`.
    - Every method checks `session.owner == who.user` before reading or writing.
    - `create_session` checks, in order: the feature is enabled (the error names `ML_FACTORY_AGENT_ENABLED`), the goal (1 to 4,000 non-blank characters), the limits and the Autopilot arguments. It then creates the session, writes the `session_created` audit record, and kicks the runner.
    - `build_agent_service(...)` in `wiring.py` assembles the service from settings, store and services without opening a connection.
    - _Requirements: 1.1, 1.2, 1.6, 6.8, 6.10, 13.5, 15.8_

  - [x] 4.16 Wire the agent into `backend/container.py`
    - Add `agent: Any = None` as the last field of `AppContainer`.
    - Add the `converse_client` keyword to `build_container`, calling `build_agent_service`.
    - Build the Bedrock Converse client lazily, and only when `agent_enabled` is true and no client was injected.
    - No existing behaviour changes.
    - Remove the now-redundant `client.model_id = settings.effective_agent_model_id` line in `backend/agent/wiring.py`, because `BedrockConverseClient.from_settings` now uses `effective_agent_model_id`.
    - _Requirements: 15.7, 15.8, 13.7_

  - [ ]* 4.17 Write property test for goal validation
    - File: `tests/agent/test_intake.py`
    - **Property 5: Goal validation**
    - **Validates: Requirements 1.1, 1.2**

  - [ ]* 4.18 Write property test for plan validity and plan-first
    - File: `tests/agent/test_intake.py`
    - **Property 6: Plan validity and plan first**
    - **Validates: Requirements 1.3**

  - [ ]* 4.19 Write property test for the clarifying-question deferral
    - File: `tests/agent/test_intake.py`
    - **Property 7: A clarifying question defers every side effect in its turn**
    - **Validates: Requirements 1.5**

  - [x]* 4.20 Write property test for registration classification
    - File: `tests/agent/test_registry_inputs.py`
    - **Property 8: Registration classifies every tool exactly once**
    - **Validates: Requirements 3.2, 3.3, 3.7**

  - [x]* 4.21 Write property test for calls that cannot validate
    - File: `tests/agent/test_registry_inputs.py`
    - **Property 9: Calls that cannot validate never execute**
    - **Validates: Requirements 2.3, 2.4, 2.5, 3.10**

  - [ ]* 4.22 Write property test for unapproved inputs
    - File: `tests/agent/test_registry_inputs.py`
    - **Property 10: Unapproved inputs never execute**
    - **Validates: Requirements 3.4, 7.3, 8.10**

  - [ ]* 4.23 Write property test for service failures
    - File: `tests/agent/test_registry_inputs.py`
    - **Property 12: Service failures become structured errors**
    - **Validates: Requirements 3.9, 14.4**

  - [x]* 4.24 Write property test for governance
    - File: `tests/agent/test_results_pipeline.py`
    - **Property 35: Governance holds for every tool**
    - **Validates: Requirements 3.6, 8.1, 8.2, 8.3**

  - [x]* 4.25 Write property test for untrusted string truncation
    - File: `tests/agent/test_results_pipeline.py`
    - **Property 36: Untrusted strings are truncated and marked**
    - **Validates: Requirements 8.8**

  - [ ]* 4.26 Write property test for the envelope and size cap
    - File: `tests/agent/test_results_pipeline.py`
    - **Property 37: Every result sent to the model is enveloped and within the size cap**
    - **Validates: Requirements 3.8, 8.4**

  - [ ]* 4.27 Write property test for credential and URI scrubbing
    - File: `tests/agent/test_results_pipeline.py`
    - **Property 38: Credentials, signed URLs and foreign S3 keys never reach the model, the session or an audit record**
    - **Validates: Requirements 8.9, 11.5, 13.9**

  - [ ]* 4.28 Write property test for injection flagging
    - File: `tests/agent/test_results_pipeline.py`
    - **Property 39: Instruction-like results are flagged and still enveloped**
    - **Validates: Requirements 8.7**

  - [ ]* 4.29 Write property test for persistence retries
    - File: `tests/agent/test_contracts_store.py`
    - **Property 3: Persistence is retried exactly as specified**
    - **Validates: Requirements 6.14**

  - [ ]* 4.30 Write example tests for the minimal loop
    - File: `tests/agent/test_loop_examples.py`
    - Planning failure, Bedrock errors, credentials, unsupported model, question answer, completion, cancel, ownership and the disabled agent are already covered in `tests/agent/test_loop.py` and `tests/agent/test_service.py`.
    - Test only this: `build_container(..., converse_client=fake)` exposes `container.agent` offline, with no network access and no S3 write.
    - _Requirements: 15.7, 15.8_

- [x] 5. Checkpoint - Minimal loop works offline
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Opt-in Smoke_Test
  - [x] 6.1 Create the runnable harness `scripts/agent_smoke.py`
    - Check the preconditions, failing at the first that does not hold: (a) a Converse call with a trivial tool succeeds on the configured model and region, (b) the role can use SageMaker and S3, (c) a session saved to S3 loads in a fresh service with identical state.
    - Build the smoke configuration: half the production limits and a spend maximum of at most 5 USD.
    - Always print the final report: the session id, every job reference and the estimated spend. Exit non-zero on any failure and keep the session.
    - Build only the three precondition checks, the smoke configuration and the final report printing here. The full step path (upload, experiment creation, EDA, training, validation, report) and the spend-limit stop are built only in task 12.7.
    - _Requirements: 17.1, 17.2, 17.5, 17.9_

  - [x] 6.2 Create `tests/smoke/test_agent_smoke.py`
    - Use the `aws` marker plus the explicit flag `ML_FACTORY_AGENT_SMOKE=1`.
    - Without the flag the test is skipped and no client is built.
    - It uses the same code path as the harness.
    - _Requirements: 17.1, 17.2, 17.8_

  - [ ]* 6.3 Write unit tests for the smoke wiring
    - File: `tests/agent/test_smoke_wiring.py`
    - Cover the skip without the flag, the halved limits with spend capped at 5 USD, and the final report formatter output.
    - _Requirements: 17.2, 17.5, 17.9_

- [x] 7. Checkpoint - Smoke_Test harness ready
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Approvals, autopilot, limits and long-running jobs
  - [x] 8.1 Complete `backend/agent/policy.py`
    - Implement gate rules 3 to 8 from the design: `always_approve`, first upload, `protected`, config-changing Mutating in Supervised, Costly in Supervised, and Costly in Autopilot with the reasons `no_grant`, `estimate_unavailable`, `spend_cap` and `guard_refused_before`.
    - The gate depends only on its input and is total.
    - _Requirements: 4.2, 4.3, 4.4, 5.1, 5.5, 5.6, 5.7, 5.8, 7.4_

  - [x] 8.2 Implement `backend/agent/limits.py`
    - Implement `LimitName`, `validate_limits`, `check_loop_top`, `check_before_model_call`, `check_before_tool_call`, `would_exceed_costly`, `register_call`, `raise_limit` and `accrue_active`.
    - A `LimitStop` names the limit, its configured value and its current value.
    - Active time accrues only while the status is `running`.
    - A Costly call over a cap is never executed and returns a `LIMIT_REACHED` error in every mode.
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.8, 9.9, 14.3_

  - [x] 8.3 Implement `backend/agent/estimate.py`
    - Implement `CostEstimator`. It builds the same `PlannedJob` objects the orchestrator would and calls `guards.reserve`.
    - It loads the price table from `price_table_path` regardless of `enforce_budget`.
    - It returns `usd=None` when the estimate fails or no estimator exists, and labels the price source and version.
    - _Requirements: 4.5, 5.5, 9.6_

  - [x] 8.4 Add approval handling to `AgentService`
    - Implement `decide_approval` (approve, reject, edit) and `decide_batch` (at most 10 items).
    - Reject non-owners. Mark expired requests `expired` and refuse later decisions with `approval_expired`.
    - A stale `args_hash` is refused with `approval_mismatch`.
    - A rejection stores at most 500 characters of the reason.
    - An edit is validated against the tool's input model and `inputs.check`. A failed edit leaves the request unchanged and names the failing arguments. A valid edit moves to `edit_proposed` with the new hash.
    - Add the approval expiry sweep.
    - Shared hand-off rule (same as task 8.6): the loop executes a call in state `AWAITING_APPROVAL` only when its `ApprovalRequest` has status `approved` and `approved_hash` equals the call's `args_hash`; after execution it marks the approval `executed`. `decide_approval` only records the decision (`status`, `approved_hash`, `decided_by`, `decided_at`), sets the session `running` and kicks the runner; it never runs a handler. A rejected approval resolves the call with a rejection result.
    - _Requirements: 4.6, 4.7, 4.8, 4.9, 4.10, 4.11, 4.12, 4.13_

  - [x] 8.5 Add mode and limit actions to `AgentService`
    - Implement `set_mode`. An Autopilot request requires `agent_autopilot_allowed` and a maximum spend in `(0, limits.max_estimated_spend_usd]`. It records `AutopilotGrant` (enabling user, UTC timestamp with second precision) before the next tool call.
    - `set_mode(SUPERVISED)` clears the grant.
    - Implement `continue_after_limit`, which changes exactly the named limit by the confirmed positive amount.
    - Write `mode_changed` and `limit_raised` audit records.
    - _Requirements: 5.2, 5.3, 5.4, 5.9, 5.10, 9.3, 9.7_

  - [x] 8.6 Extend `backend/agent/loop.py` with gating, approvals and the Costly path
    - Apply the repeated-call rule and the limit checks. Re-read the mode inside the gating transaction.
    - Evaluate `protected` (fail closed) and the estimate, then call `policy.gate`.
    - On `REQUIRE_APPROVAL`, create the `ApprovalRequest` with the exact arguments, rationale and estimate (or "no estimate available"). Pre-create requests for the remaining Costly uses of the turn (at most 10) and set `waiting_for_approval`.
    - Execute only when the call hash equals the approved hash.
    - For Costly calls, derive the idempotency key, write the in-flight marker, persist state `executing`, run the handler, then record the result in a second transaction. Update counters and spend.
    - Record guard refusals and never retry them with changed arguments.
    - Refuse Mutating or Costly calls for an experiment with an unresolved awaited job.
    - Handle approval expiry at the next step.
    - Shared hand-off rule (same as task 8.4): the loop executes a call in state `AWAITING_APPROVAL` only when its `ApprovalRequest` has status `approved` and `approved_hash` equals the call's `args_hash`; after execution it marks the approval `executed`. `decide_approval` only records the decision (`status`, `approved_hash`, `decided_by`, `decided_at`), sets the session `running` and kicks the runner; it never runs a handler. A rejected approval resolves the call with a rejection result.
    - _Requirements: 1.8, 2.6, 4.2, 4.3, 4.5, 4.6, 4.9, 4.12, 5.1, 5.5, 5.6, 5.7, 9.2, 9.4, 9.6, 9.8, 9.9, 14.2, 14.3, 14.7_

  - [x] 8.7 Add job waiting, recovery and the watcher
    - In `loop.py`: record awaited jobs from each handler's `job_extractor`, set `waiting_for_job`, and return without a model call.
    - Add recovery to `loop.py`: reconcile every `executing` Costly call against `ExecutionRecord`s started at or after the marker. Adopt exactly one match as the result, otherwise report `not_submitted` or `unknown_outcome` (30 second query bound). Never re-execute.
    - Create `backend/agent/jobs.py` with `refresh_jobs(session)`. It maps `AnalysisService.execution_status` to `succeeded`, `failed` (with the reason), `stopped` or `not_found`, and sets `jobs_ready` when all awaited jobs are terminal.
    - Create `backend/agent/watcher.py` with `AgentJobWatcher.discover`, `sweep_jobs` and `start`. It never calls the model and never raises into its thread loop.
    - Start the watcher from `mcp_server/app.py::serve` when `agent_enabled` is true.
    - On resume, deliver one enveloped job update per awaited job.
    - _Requirements: 6.3, 6.4, 6.5, 6.6, 14.1, 14.5, 14.6, 14.11_

  - [x] 8.8 Complete cancel and job refresh in `AgentService`
    - `cancel` sets `cancelled`, rejects every pending approval, resolves pending tool uses with `cancelled` results, and persists.
    - With `stop_jobs=true` it stops each non-terminal awaited job through the existing stop paths and audits it. The default leaves jobs running.
    - Call `refresh_jobs` from `get_session` and `resume`.
    - Reject `resume` while a lease is held with `session_in_use`.
    - _Requirements: 6.5, 6.6, 6.10, 6.12, 6.13, 14.1_

  - [ ]* 8.9 Write property test for the approval gate
    - File: `tests/agent/test_policy.py`
    - **Property 13: Approval gate**
    - **Validates: Requirements 4.2, 4.3, 4.4, 5.7, 7.4**

  - [ ]* 8.10 Write property test for the approved call
    - File: `tests/agent/test_policy.py`
    - **Property 14: The approved call is exactly the shown call, run at most once**
    - **Validates: Requirements 4.5, 4.6, 4.9, 14.2, 14.7**

  - [ ]* 8.11 Write property test for edits
    - File: `tests/agent/test_policy.py`
    - **Property 15: Edits are validated and confirmed before they run**
    - **Validates: Requirements 4.8**

  - [ ]* 8.12 Write property test for rejection and expiry
    - File: `tests/agent/test_policy.py`
    - **Property 16: Rejection and expiry never execute**
    - **Validates: Requirements 4.7, 4.12**

  - [ ]* 8.13 Write property test for batch approval
    - File: `tests/agent/test_policy.py`
    - **Property 17: Batch approval only covers what was displayed**
    - **Validates: Requirements 4.13**

  - [ ]* 8.14 Write property test for Autopilot enablement and switching
    - File: `tests/agent/test_policy.py`
    - **Property 20: Autopilot enablement and switching**
    - **Validates: Requirements 5.2, 5.4, 5.9**

  - [ ]* 8.15 Write property test for Autopilot limits
    - File: `tests/agent/test_policy.py`
    - **Property 21: Autopilot executes only within every limit**
    - **Validates: Requirements 5.1, 5.5, 5.6, 9.8**

  - [ ]* 8.16 Write property test for the one-result-per-request invariant
    - File: `tests/agent/test_registry_inputs.py`
    - **Property 11: One result per request, in request order**
    - **Validates: Requirements 1.8, 2.6**

  - [ ]* 8.17 Write property test for exact counters
    - File: `tests/agent/test_limits.py`
    - **Property 23: Counters are bounded and exact**
    - **Validates: Requirements 9.2, 9.6, 9.9, 14.3**

  - [ ]* 8.18 Write property test for limit validation and raising
    - File: `tests/agent/test_limits.py`
    - **Property 24: Limits change only by validated user action**
    - **Validates: Requirements 9.1, 9.3**

  - [ ]* 8.19 Write property test for repeated calls
    - File: `tests/agent/test_limits.py`
    - **Property 25: Repeated identical calls**
    - **Validates: Requirements 9.4**

  - [ ]* 8.20 Write property test for active time
    - File: `tests/agent/test_limits.py`
    - **Property 26: Active time counts only running time**
    - **Validates: Requirements 9.5**

  - [ ]* 8.21 Write property test for restart equivalence
    - File: `tests/agent/test_loop_recovery.py`
    - **Property 4: Restart equivalence and exactly-once submission**
    - **Validates: Requirements 6.2, 14.5**

  - [ ]* 8.22 Write property test for waiting and job delivery
    - File: `tests/agent/test_loop_recovery.py`
    - **Property 27: Waiting holds nothing open, and job results are delivered**
    - **Validates: Requirements 6.4, 6.5, 6.6, 14.1, 14.11**

  - [ ]* 8.23 Write property test for cancellation
    - File: `tests/agent/test_loop_recovery.py`
    - **Property 30: Cancellation is final**
    - **Validates: Requirements 6.12, 6.13**

  - [ ]* 8.24 Write property test for control-state immunity
    - File: `tests/agent/test_ownership_control.py`
    - **Property 19: Control state is immune to model output and untrusted content**
    - **Validates: Requirements 4.11, 5.3, 8.5, 8.6, 9.7**

- [x] 9. Checkpoint - Approvals, limits and job waiting
  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Full tool catalogue
  - [x] 10.1 Create `backend/agent/tools_experiments.py`
    - Register `create_experiment` as Costly (it submits EDA inside `create`), and the Read tools `get_leakage_findings`, `get_feature_review` and `get_business_context`.
    - Register the Mutating tools `set_business_context` and `investigate_leakage`, and the Read tool `get_leakage_investigation`.
    - Each wraps the same service call the route or MCP tool makes, with the experiment audit action set.
    - _Requirements: 3.1, 3.2, 3.4, 3.5, 5.8_

  - [x] 10.2 Create `backend/agent/tools_features.py`
    - Register `propose_features` (Mutating, advisory) and `get_feature_proposals` (Read).
    - Register `define_derived_features` and `approve_features` (Costly, `changes_config`).
    - Register `decide_features` (Mutating, `changes_config`, `protected` unless the selection keeps every candidate column; any error evaluating `protected` means protected).
    - Leakage and proposal tools never change the feature set.
    - _Requirements: 3.1, 5.7, 5.8_

  - [x] 10.3 Create `backend/agent/tools_training.py`
    - Register `preflight_check`, `start_training`, `rerun_training`, `get_training_status`, `get_execution_status`, `get_comparison` (verbatim), `get_threshold_analysis`, `list_model_versions`, `start_validation`, `get_validation` (verbatim), `start_holdout_evaluation`, `get_holdout_evaluation`, `start_ablation` and `get_ablation`.
    - Compose `start_training` as `routes/training.py` does.
    - Add a `job_extractor` for every tool that starts a job.
    - Add no promote, delete or download tool.
    - _Requirements: 3.1, 3.5, 3.7, 6.3_

  - [x] 10.4 Create `backend/agent/tools_reports.py` and the free-form tools
    - Register `generate_report` (returns artifact refs, not URIs), `get_report` (verbatim) and `interpret_results` (`ledger_source=False`).
    - Register `draft_free_form_script`, `start_free_form_run` (Costly, `always_approve`, `protected`; takes `draft_id` and `revision`, and loads the script bytes server-side so the approval shows the code and its content hash), `get_free_form_run`, `list_free_form_runs` and `compare_free_form` (verbatim).
    - If `FreeFormService` has no accessor for a stored draft revision, add the additive method `draft_revision(experiment_id, draft_id, revision)`.
    - _Requirements: 3.1, 3.7, 5.7, 10.5, 10.6, 10.7_

  - [ ]* 10.5 Write example tests for the catalogue and audit parity
    - File: `tests/agent/test_catalogue_examples.py`
    - Every capability named in Requirement 3.1 has a tool. No tool is named like delete, promote or approve-script.
    - After a Mutating tool, the experiment audit entry has the same action and detail keys as the matching route, and the agent record's `service_audit_ref` resolves to it.
    - With `audit_log_enabled=false` nothing is written.
    - Cover the Autopilot scenario: a Guard-blocked action is refused and not retried, the loop stops before the spend maximum, and the feature selection is untouched.
    - _Requirements: 3.1, 3.7, 5.6, 5.8, 11.3, 11.8, 16.11_

  - [ ]* 10.6 Write property test for feature exclusion
    - File: `tests/agent/test_policy.py`
    - **Property 22: Features are never excluded without an approved decision**
    - **Validates: Requirements 5.8**

- [x] 11. Narration, compaction and audit trail
  - [x] 11.1 Complete `backend/agent/narration.py`
    - Implement the ledger with the same matching rule as `ml_engine/reasoning/numbers.py`: token precision, and a percent token also matches its fraction.
    - Implement `verify_narration` with the `mark` (default) and `withhold` policies, and one `narration_unverified` audit record per affected sentence.
    - Implement `build_final_answer`: attributions with tool-call id and path, the `self_reported` or `engine` label for free-form numbers, verbatim blocks from the stored full result, "not comparable" notes, the *Advisory* label, and `overclaim` marking.
    - Also change `backend/agent/loop.py`. Today the loop passes an empty `NumberLedger()` to `verify_narration`, and `_resolve` discards `ShapedResult.ledger`. At final-answer time, build the `NumberLedger` from the session's stored full tool results: call `SessionStore.get_result` for each `ToolCallRecord.result_ref`, run each result through the same ledger walk as `results.py`, include only tools with `flags.ledger_source` true, and pass the ledger to both `verify_narration` and `build_final_answer`. Do not add a new `Session` field.
    - _Requirements: 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7, 10.8_

  - [x] 11.2 Add compaction to `backend/agent/context.py`
    - When the estimated size exceeds `agent_context_budget_tokens`, replace turns older than the most recent 6 with one deterministic summary.
    - Each summarized call keeps `[<call_id>] <tool> <status>` and every `path=value` ledger pair verbatim. Keep the accepted plan.
    - The persisted history is unchanged.
    - _Requirements: 6.11_

  - [x] 11.3 Complete audit trail and audit-failure handling
    - Add `AgentAudit.trail(session)`, ordered by `(at, seq)`, which resolves each `service_audit_ref`.
    - Emit all event types: `tool_executed`, `approval_decision`, `mode_changed`, `limit_stop`, `limit_raised`, `cancelled`, `input_rejected`, `narration_unverified`, `session_failed` and `approval_expired`.
    - On a write failure for a Mutating or Costly tool, set `session.audit_failed`. The loop then runs no further Mutating or Costly tool, Read tools continue, and the owner gets an `audit_failed` error.
    - Add `AgentService.audit_trail`.
    - _Requirements: 11.1, 11.2, 11.3, 11.4, 11.6, 11.7, 11.8, 11.9_

  - [ ]* 11.4 Write property test for ledger verification
    - File: `tests/agent/test_narration.py`
    - **Property 40: Narration is verified against the ledger**
    - **Validates: Requirements 10.1, 10.2, 10.3**

  - [ ]* 11.5 Write property test for final-answer attributions and verbatim blocks
    - File: `tests/agent/test_narration.py`
    - **Property 41: A final answer carries attributions, labels and verbatim blocks**
    - **Validates: Requirements 10.4, 10.5, 10.7**

  - [ ]* 11.6 Write property test for comparisons and advisory results
    - File: `tests/agent/test_narration.py`
    - **Property 42: Comparisons and advisory results are described honestly**
    - **Validates: Requirements 10.6, 10.8**

  - [ ]* 11.7 Write property test for compaction
    - File: `tests/agent/test_compaction.py`
    - **Property 29: Compaction preserves ids and engine numbers**
    - **Validates: Requirements 6.11**

  - [ ]* 11.8 Write property test for audit completeness and order
    - File: `tests/agent/test_audit.py`
    - **Property 43: Audit records are complete, ordered and lossless**
    - **Validates: Requirements 11.1, 11.2, 11.3, 11.4, 11.6, 11.8**

  - [ ]* 11.9 Write property test for audit failure
    - File: `tests/agent/test_audit.py`
    - **Property 44: A failed audit write stops further side effects**
    - **Validates: Requirements 11.9**

- [x] 12. Local-folder uploads
  - [x] 12.1 Implement `backend/agent/uploads.py`
    - Implement `reuse_or_plan` (size and extension checks from the manifest before any transfer, SHA-256 reuse) and `RowCounter` (Parquet footer, chunked CSV read).
    - Reuse the existing `dataset_id_for` from `backend/agent/tools_core.py`; do not define a second copy.
    - Supported formats are CSV and Parquet only, and errors name both.
    - _Requirements: 7.6, 7.7, 7.11, 7.12_

  - [x] 12.2 Add the upload flow to `AgentService` and `backend/agent/tools_uploads.py`
    - Add `register_local_files`, `client_action` (mints a pre-signed PUT through `DatasetService.create_upload_ticket`, persisting nothing signed) and `complete_upload`.
    - `complete_upload` calls `DatasetService.register`. On any failure it deletes the object, keeps the action `ready` so the same file can be retried, and returns the reason. It re-checks the size and deletes an oversized object.
    - On success it adds a `DatasetRef` with the profile, marks the upload as approved once, resolves the pending tool use with the governed profile (no raw rows), and resumes the loop.
    - Register `request_local_upload` (Mutating, `upload`). It asks the user which file to use when several candidates exist and the user has not chosen, and creates no approval until the user chooses.
    - The approval shows the file name, the size and the destination prefix.
    - _Requirements: 7.2, 7.3, 7.4, 7.5, 7.8, 7.9, 7.10, 7.11, 7.12_

  - [ ]* 12.3 Write property test for upload refusal before transfer
    - File: `tests/agent/test_uploads.py`
    - **Property 32: Upload refusal happens before transfer**
    - **Validates: Requirements 7.6, 7.7**

  - [ ]* 12.4 Write property test for duplicate reuse and id stability
    - File: `tests/agent/test_uploads.py`
    - **Property 33: Duplicate uploads reuse the dataset, and ids are stable**
    - **Validates: Requirements 7.11, 7.12**

  - [ ]* 12.5 Write property test for failed uploads
    - File: `tests/agent/test_uploads.py`
    - **Property 34: A failed upload leaves nothing behind**
    - **Validates: Requirements 7.10**

  - [ ]* 12.6 Write example tests for uploads
    - File: `tests/agent/test_uploads_examples.py`
    - Use a temporary folder with an oversized file, a bad extension, corrupt content and a duplicate upload.
    - Cover the multi-candidate question and the first-upload approval in both modes.
    - _Requirements: 7.4, 7.6, 7.7, 7.8, 7.11, 16.13_

  - [x] 12.7 Enable the full path in the smoke harness
    - Edit `scripts/agent_smoke.py` to add the full step path, which task 6.1 does not build: upload, experiment creation, EDA, one training run on the smallest allowed instance, validation and the report. Use the delivered catalogue, approvals, limits and uploads.
    - Generate a synthetic dataset of at most 10,000 rows and 5 MB, plus a labelled validation file. Run a Supervised session that grants each approval automatically, and record each step as passed or failed in order.
    - Add the spend-limit stop: on reaching the spend maximum, stop starting jobs, stop running jobs through the same code as `cancel(stop_jobs=True)`, fail naming the limit, and leave the session resumable.
    - _Requirements: 17.3, 17.4, 17.6, 17.7_

  - [ ]* 12.8 Write property test for ownership isolation
    - File: `tests/agent/test_ownership_control.py`
    - **Property 18: Ownership isolation**
    - **Validates: Requirements 4.10, 11.7, 12.9**

- [ ] 13. Checkpoint - Backend agent complete
  - Ensure all tests pass, ask the user if questions arise.

- [x] 14. API routes
  - [x] 14.1 Create `backend/api/routes/agent.py` and register it in `backend/api/app.py`
    - Add every route in the design's route table, plus `GET /agent/config`. Follow the `principal(...)`, service call and `json_ok` pattern.
    - Add `agent` to `_collect_routes` additively.
    - When the agent is disabled, every `/agent/...` route returns `FEATURE_DISABLED` before any S3 access.
    - No pre-existing route changes. `SessionView` omits URIs and signed URLs.
    - _Requirements: 12.1, 12.9, 12.13, 13.5, 15.8_

  - [ ]* 14.2 Write API example tests
    - File: `tests/agent/test_api_examples.py`
    - The owner reads and decides the same session and gets equal views.
    - A non-owner is refused for read, decide and cancel.
    - The disabled agent returns `FEATURE_DISABLED` and the existing routes match a recorded baseline.
    - The MCP tool names, parameters and result keys equal the recorded baseline.
    - _Requirements: 6.9, 12.9, 12.15, 13.5, 15.8, 16.12_

  - [ ]* 14.3 Extend the boundary tests for `backend/agent`
    - File: `tests/agent/test_agent_boundaries.py`
    - `backend/agent/*` imports `boto3` only lazily, and imports no `jobs`, `mcp_server` or `ml_factory_cli`. (`backend/agent/jobs.py` is excluded from the `jobs` check.)
    - `backend/agent/*` builds no agent path from a string literal.
    - `catalogue.py` and `tools_*.py` import no store implementation and no boto client.
    - _Requirements: 15.3, 15.6, 15.8, 16.6_

- [ ] 15. Checkpoint - API complete
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 16. CLI (`mlf`)
  - [x] 16.1 Implement `ml_factory_cli/local_files.py`
    - `list_candidates(path)`: `.csv` and `.parquet` (case-insensitive), non-recursive, sorted by file name ascending, at most 100 with the total count, and name, size in bytes and UTC ISO 8601 modification time.
    - `sha256_file(path)`.
    - _Requirements: 7.1, 7.2_

  - [ ] 16.2 Implement `ml_factory_cli/transport.py`
    - Define the `AgentTransport` Protocol, `HttpTransport` (httpx, same identity header as the web UI) and `InProcessTransport(AgentService)`.
    - `InProcessTransport` imports `boto3` and the container lazily, builds clients through `build_session`, and honours `aws_profile` and `aws_role_arn`.
    - _Requirements: 12.1, 12.13, 12.14_

  - [x] 16.3 Implement `ml_factory_cli/render.py`
    - Format transcript entries. Tool-call argument summaries are at most 200 characters, and result summaries are at most 500 characters.
    - Print every approval request and the current mode.
    - _Requirements: 12.5, 5.10_

  - [ ] 16.4 Implement `ml_factory_cli/__main__.py`
    - `main(argv)` supports `mlf "<goal>"`, `mlf <session_id>` (matches `ses_[0-9a-f]{16}`), and the subcommands `list`, `approve`, `reject`, `edit`, `mode`, `cancel`, `audit`, `watch` and `data`.
    - Stream by polling `after_seq` every `agent_cli_poll_seconds`, in `seq` order.
    - On an interactive terminal prompt approve, reject or edit. When stdout is not a TTY and an approval is pending, print the request and exit 75 without approving. There is no `--yes` flag.
    - `--watch` polls at `--interval` (default 30, range 5 to 3600) and resumes when awaited jobs end.
    - Exit codes: `0` completed or waiting for a job, `1` unexpected error, `3` control plane unreachable, `4` unknown session, `75` approval pending on a non-TTY.
    - Offer a cancel prompt that asks whether to stop running jobs.
    - _Requirements: 4.13, 6.7, 6.13, 7.1, 12.2, 12.3, 12.4, 12.5, 12.6, 12.7, 12.8_

  - [ ] 16.5 Update `pyproject.toml` for the CLI
    - Add `[project.scripts] mlf = "ml_factory_cli.__main__:main"`, add `ml_factory_cli*` to the packages, and add the optional extra `cli = ["httpx>=0.27"]`.
    - _Requirements: 12.1, 15.4_

  - [ ]* 16.6 Write property test for candidate listing
    - File: `tests/agent/test_uploads.py`
    - **Property 31: Candidate listing**
    - **Validates: Requirements 7.1**

  - [ ]* 16.7 Write property test for CLI stream output
    - File: `tests/agent/test_cli.py`
    - **Property 49: CLI stream output is bounded and ordered**
    - **Validates: Requirements 12.5**

  - [ ]* 16.8 Write property test for the non-interactive CLI
    - File: `tests/agent/test_cli.py`
    - **Property 48: A non-interactive CLI never approves**
    - **Validates: Requirements 12.6**

  - [ ]* 16.9 Write CLI example and boundary tests
    - File: `tests/agent/test_cli_examples.py`
    - Cover exit codes 3 and 4, `--in-process` honouring `aws_profile` and `aws_role_arn` (stubbed `build_session`), `--watch` interval bounds, and a session created in the CLI and resumed through the API.
    - Check that `ml_factory_cli/*` imports no `backend.agent.loop`, `policy`, `limits` or `store`, and imports `boto3` and the container only inside `InProcessTransport`.
    - _Requirements: 6.7, 12.3, 12.4, 12.7, 12.8, 12.14, 16.12_

- [ ] 17. Web chat panel (last)
  - [ ] 17.1 Create `backend/api/ui/assistant.js` and `backend/api/ui/assistant.css`
    - Add an "Assistant" entry to the rail, reusing the globals from `app.js` (`api`, `el`, `toast`, `title`, `view`). Do not edit `app.js`.
    - Show the transcript, tool calls, pending approvals with approve, reject and edit controls, the mode switch, the counters and the status.
    - Poll `GET /agent/sessions/{id}?after_seq=N` every `agent_web_poll_seconds`, also while `waiting_for_job`. Call `resume` once when `jobs_ready` is true.
    - Show attributions next to numbers, free-form metric labels, and verbatim tables rendered from the stored result.
    - Use a browser file picker with `crypto.subtle.digest` for uploads.
    - Render all model, tool, file-name and column-name text with `textContent` or text nodes, never `innerHTML`.
    - _Requirements: 4.13, 5.10, 7.2, 8.4, 10.4, 10.5, 12.10, 12.11, 12.12_

  - [ ] 17.2 Register the assets in `backend/api/static.py` and `backend/api/ui/index.html`
    - Add the two files to `_ASSETS` additively.
    - Add one `<script>` and one `<link>` to `index.html`.
    - _Requirements: 12.10, 15.8_

  - [ ]* 17.3 Write web UI tests
    - File: `tests/agent/test_web_ui.py`
    - The new assets are served from the allow-list.
    - `index.html` references them.
    - A text scan finds no `innerHTML` assignment from session data in `assistant.js`.
    - _Requirements: 12.10, 16.14_

- [ ] 18. Final manifest and checkpoint
  - [ ] 18.1 Re-record the inherited manifest
    - Run the manifest recorder after all additive edits (`config.py`, `container.py`, `audit.py`, `layout.py`, `static.py`, `index.html`, `app.py`, `pyproject.toml`).
    - Confirm `tests/test_inherited_manifest.py` and `tests/test_architecture_boundaries.py` pass.
    - _Requirements: 15.3, 15.5_

  - [ ] 18.2 Final checkpoint - Ensure all tests pass
    - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP.
- Each task references specific requirements for traceability.
- Checkpoints ensure incremental validation.
- Property tests validate the 49 universal correctness properties. Each is its own sub-task, placed close to the code it checks.
- Unit and example tests validate specific scenarios and the criteria that have no property.
- Tasks that write the same file sit in different waves of the dependency graph.
- In the dependency graph, waves 0 to 16 hold only required tasks, and the optional `*` test tasks sit in waves 17 to 24. No required task waits on an optional one, and every optional test runs after the code it tests.
- The Smoke_Test (task 6) is opt-in and excluded from the default suite.
- Free-form runs always require user approval, in every mode, including Autopilot. Time series and segmentation are done through free-form scripts; their metrics are labelled self-reported.
- The session lease and the transaction lock assume a single backend process. Running several replicas weakens them.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1.1", "1.2", "1.3", "2.1"] },
    { "id": 1, "tasks": ["1.4", "4.1", "4.2", "4.7", "4.9", "4.13"] },
    { "id": 2, "tasks": ["4.3", "4.8", "4.10", "4.12"] },
    { "id": 3, "tasks": ["4.11", "4.14"] },
    { "id": 4, "tasks": ["4.15"] },
    { "id": 5, "tasks": ["4.16"] },
    { "id": 6, "tasks": ["6.1", "8.1", "8.2", "8.3", "8.4", "11.2", "12.1"] },
    { "id": 7, "tasks": ["6.2", "8.5", "8.6"] },
    { "id": 8, "tasks": ["8.7", "10.1", "10.2", "10.3", "10.4"] },
    { "id": 9, "tasks": ["8.8", "11.1"] },
    { "id": 10, "tasks": ["11.3"] },
    { "id": 11, "tasks": ["12.2"] },
    { "id": 12, "tasks": ["12.7", "14.1", "16.1", "16.3"] },
    { "id": 13, "tasks": ["16.2", "17.1"] },
    { "id": 14, "tasks": ["16.4", "17.2"] },
    { "id": 15, "tasks": ["16.5"] },
    { "id": 16, "tasks": ["18.1"] },
    { "id": 17, "tasks": ["1.5", "1.6", "2.2", "2.4", "4.4", "4.17", "4.20", "4.24", "4.30", "6.3", "8.9", "8.17", "8.21", "8.24", "10.5", "11.4", "11.7", "11.8", "12.3", "12.6", "14.2", "16.7", "16.9", "17.3"] },
    { "id": 18, "tasks": ["2.3", "4.5", "4.18", "4.21", "4.25", "8.10", "8.18", "8.22", "11.5", "11.9", "12.4", "12.8", "14.3", "16.8"] },
    { "id": 19, "tasks": ["4.6", "4.19", "4.22", "4.26", "8.11", "8.19", "8.23", "11.6", "12.5"] },
    { "id": 20, "tasks": ["4.23", "4.27", "4.29", "8.12", "8.20", "16.6"] },
    { "id": 21, "tasks": ["4.28", "8.13", "8.16"] },
    { "id": 22, "tasks": ["8.14"] },
    { "id": 23, "tasks": ["8.15"] },
    { "id": 24, "tasks": ["10.6"] }
  ]
}
```
