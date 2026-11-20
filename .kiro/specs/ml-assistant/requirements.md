# Requirements Document

## Introduction

The ML Assistant is a Kiro-like agent built into ML Factory 2. The user states a goal in plain language, for example `mlf "build a churn model from ./data/customers.csv and validate it"`. The agent plans the work, calls the platform's existing services as tools, waits for SageMaker jobs, and reports engine-computed metrics with attribution.

The agent runs a Bedrock Converse tool-use loop. The loop's tools are thin wrappers over the existing services (ExperimentService, AnalysisService, ValidationService, FreeFormService, ModelVersionService, ReportingService, LeakageService, FeatureProposalService, ReasoningService, DatasetService). The agent adds no new compute path, no new number source and no new storage backend.

Two execution modes exist. **Supervised** is the default: the agent pauses for user approval before every cost-incurring submit. **Autopilot** is opt-in: the agent proceeds without per-step approval, but only inside the existing budget and concurrency guards and the limits in this document. Autopilot never auto-excludes a feature, never approves a free-form script, and never promotes a model.

Two surfaces share one agent service: a CLI (`mlf`) first, then a chat panel in the existing static web UI (`backend/api/ui`, no build step). Sessions are persisted in S3 so a conversation survives long SageMaker jobs, process restarts and surface changes. The agent never holds a request open while a job runs.

Platform invariants that this feature MUST preserve:

- SageMaker-only compute; S3-only artifacts.
- Every number comes from the deterministic engine, never the LLM.
- LLM boundary B12: the LLM proposes and explains, applies nothing without explicit user approval, and never sees raw data rows.
- Nothing is auto-excluded except the target.
- Ownership and audit on every mutation.
- Budget and concurrency guards in `backend/orchestration/guards.py` apply to every agent action.
- New contracts are versioned Pydantic contracts, and their paths live only in `ml_engine/io/layout.py`.
- The default test suite runs offline with fakes.
- Inherited engine files change only additively, with an entry in `tests/inherited_manifest.json`.

## Glossary

- **Platform**: ML Factory 2 as a whole.
- **Agent_Service**: the single backend component that runs the agent loop, owns sessions, and is used by every surface.
- **Agent_Loop**: the cycle in which the Agent_Service sends the conversation and tool definitions to Bedrock via the Converse API, executes the requested tools, and returns the results to the model until the model gives a final answer or the loop pauses.
- **Converse_Client**: the new Bedrock adapter that implements the Converse tool-use protocol. It sits beside, and does not replace, `BedrockReasoningClient`.
- **Tool**: a named, schema-described function exposed to the model. Each Tool wraps one existing service operation.
- **Read_Tool**: a Tool that changes nothing and incurs no cost (for example list experiments, get EDA summary, get metrics).
- **Mutating_Tool**: a Tool that creates or changes platform state without incurring compute cost (for example create experiment, upload dataset, save a configuration).
- **Costly_Tool**: a Tool that submits SageMaker work or otherwise incurs spend (for example start EDA, start training, start validation, run an approved free-form script).
- **Session**: the persisted record of one conversation: messages, tool calls and results, pending approvals, linked experiments and jobs, mode, limits and counters.
- **Session_Status**: one of `running`, `waiting_for_approval`, `waiting_for_job`, `waiting_for_user`, `completed`, `failed`, `cancelled`.
- **Approval_Request**: a persisted record of a proposed Costly_Tool or Mutating_Tool call that needs a user decision. It holds the tool name, the exact arguments, an estimated cost where the guards can produce one, and a plain-language rationale.
- **Approval_Decision**: the user's response to an Approval_Request: approve, reject, or edit.
- **Supervised_Mode**: the default mode; every Costly_Tool call needs an Approval_Decision.
- **Autopilot_Mode**: the opt-in mode in which Costly_Tool calls run without per-call approval, within the Session_Limits.
- **Session_Limits**: the caps on a Session: maximum tool calls, maximum Converse turns, maximum Costly_Tool calls, maximum estimated spend, maximum wall-clock duration, and maximum repeated identical tool calls.
- **Local_Dataset_Folder**: a directory on the user's machine that holds the data to model.
- **Dataset_Upload**: the step that copies a file from a Local_Dataset_Folder to the platform's S3 dataset location and registers it.
- **Engine_Number**: any metric, statistic, score, count or estimate produced by the deterministic engine or the control plane (including guard cost estimates).
- **Narration**: the agent's plain-language text that explains Engine_Numbers.
- **Attribution**: a reference from a number in Narration to the Tool result that produced it (tool call id, experiment id, artifact reference).
- **Untrusted_Content**: any text that did not come from the user's own typed message or the system prompt, including tool results, dataset column names and values, file names, job logs and error messages.
- **Agent_Audit_Record**: an audit entry for an agent action, recording the session id, the acting user, the tool, the arguments, and who approved.
- **Scripted_Fake_Converse_Client**: a test double that implements the Converse_Client protocol and replays a scripted sequence of model turns.
- **Smoke_Test**: an opt-in real-AWS test of the assistant's end-to-end path, excluded from the default suite.
- **Guards**: the budget, concurrency, instance-type and runtime checks in `backend/orchestration/guards.py` and `backend/config.py`.
- **Free_Form_Run**: a run of a user-approved free-form script through FreeFormService (see the `free-form-experiments` spec).
- **Self_Reported_Metric**: a metric that a Free_Form_Run's script reports about itself and that the engine has not recomputed.
- **Engine_Metric**: a metric that the deterministic engine computed from the run's predictions.

## Requirements

### Requirement 1: Natural-language goal to a plan

**User Story:** As a data scientist, I want to describe what I want in plain language, so that the agent works out the steps without my knowing the API.

#### Acceptance Criteria

1. WHEN the user submits a goal of 1 to 4,000 characters, THE Agent_Service SHALL create a Session owned by that user, set the Session_Status to `running`, and start the Agent_Loop.
2. IF the submitted goal is empty, contains only whitespace, or exceeds 4,000 characters, THEN THE Agent_Service SHALL reject the submission with an error indicating the length limit, SHALL NOT create a Session, and SHALL NOT start the Agent_Loop.
3. WHEN the Agent_Loop starts a Session, THE Agent_Service SHALL ask the model for a step plan of 1 to 10 steps, each step described in at most 200 characters, and SHALL show the plan to the user before any Mutating_Tool or Costly_Tool call is made.
4. IF the model call to produce the plan fails or returns no plan, THEN THE Agent_Service SHALL set the Session_Status to `failed`, SHALL show the user an error indicating that planning failed, and SHALL make no Mutating_Tool or Costly_Tool call.
5. IF the goal lacks information the tools require (for example the target column or the problem type), THEN THE Agent_Service SHALL ask the user one question naming the missing information, SHALL set the Session_Status to `waiting_for_user`, and SHALL make no Mutating_Tool or Costly_Tool call until the user answers.
6. WHEN the user answers a question while the Session_Status is `waiting_for_user`, THE Agent_Service SHALL resume the Agent_Loop with the answer as a user message and SHALL set the Session_Status to `running`.
7. WHEN the model produces a final answer, THE Agent_Service SHALL set the Session_Status to `completed` and SHALL present the answer with its Attributions.
8. THE Agent_Service SHALL allow the model to select, as Tool calls, each of the steps upload data, create experiment, run EDA, propose features, train, validate, and report, and SHALL execute the steps the model selects in the order the model requests them, with no step executed before the step it depends on has finished successfully.

### Requirement 2: Bedrock Converse tool-use loop

**User Story:** As a developer, I want a tool-use loop on Bedrock, so that the model can call platform services and see their results.

#### Acceptance Criteria

1. WHEN the Agent_Service requests a model turn, THE Converse_Client SHALL call the Bedrock Converse API with the system prompt, the conversation messages and the Tool definitions, and SHALL return the model's text and tool-use requests in a provider-neutral structure. Each tool-use request SHALL carry a request identifier, a Tool name and an arguments object.
2. THE Converse_Client SHALL be a new class that implements a new Protocol, and THE Platform SHALL leave `BedrockReasoningClient` and the `ReasoningClient` Protocol unchanged.
3. WHEN the model requests a Tool, THE Agent_Service SHALL validate the requested arguments against the Tool's input schema before executing the Tool.
4. IF the requested arguments fail schema validation, THEN THE Agent_Service SHALL return a structured validation error to the model as the Tool result, identifying each invalid or missing argument by name, SHALL NOT execute the Tool, and SHALL continue the loop with the Session unchanged apart from the recorded error result.
5. IF the model requests a Tool name that is not registered, THEN THE Agent_Service SHALL return a structured unknown-tool error to the model as the Tool result, naming the requested Tool and listing the registered Tool names, and SHALL NOT execute any Tool for that request.
6. WHEN the model requests several Tools in one turn, THE Agent_Service SHALL execute them one at a time in the order requested, SHALL apply the approval rules of Requirement 4 to each before it executes, and SHALL return exactly one Tool result per request identifier in the same order.
7. THE Converse_Client SHALL send every Converse call with a temperature of 0 and the configured maximum output tokens.
8. IF Bedrock throttles a Converse call or returns a transient error, THEN THE Converse_Client SHALL retry the call with exponential backoff, doubling the delay after each attempt, up to the configured retry count, and SHALL raise an error to the caller after the last retry fails. A non-transient error (for example, an invalid request or access denied) SHALL be raised immediately without retry.
9. IF Bedrock remains unavailable after the retries, THEN THE Agent_Service SHALL persist the Session with all messages and Tool results recorded so far, SHALL set the Session_Status to `failed` with the retriable flag set to true, and SHALL tell the user that the session can be resumed.
10. WHEN a Converse call returns successfully, THE Agent_Service SHALL record in the Session the model id, the input token count and the output token count of that call.

### Requirement 3: Tool catalogue over existing services

**User Story:** As a maintainer, I want the tools to be thin wrappers over the existing services, so that no business logic is duplicated and every guard still applies.

#### Acceptance Criteria

1. THE Agent_Service SHALL expose Tools that call the existing service methods for dataset registration, experiment creation, EDA, feature selection and proposals, leakage screening, training start, job status, metric retrieval, validation, free-form script drafting, model version listing and report retrieval.
2. THE Agent_Service SHALL classify every Tool as exactly one of Read_Tool, Mutating_Tool or Costly_Tool in a single registry.
3. IF a Tool is registered without a classification or with more than one classification, THEN THE Agent_Service SHALL refuse to register that Tool, SHALL raise a registration error that names the Tool, and SHALL leave the registry unchanged.
4. WHEN a Tool executes, THE Agent_Service SHALL call the service with the Session owner's identity, taken from the Session and never from Tool arguments supplied by the model, so that the service's ownership checks and Guards apply exactly as they do for a direct API call.
5. THE Agent_Service SHALL NOT expose a Tool that reads or writes S3 objects directly, submits a SageMaker job directly, or bypasses a service method.
6. THE Agent_Service SHALL NOT expose a Tool that returns raw dataset rows. A Tool result MAY contain aggregate statistics, column names, column types and counts.
7. THE Agent_Service SHALL NOT expose a Tool that approves a free-form script, promotes a model version, or deletes an experiment, dataset or artifact; those actions SHALL remain explicit user actions through the existing API or UI.
8. WHEN a Tool returns a result, THE Agent_Service SHALL return it to the model as a structured summary no larger than the configured maximum result size (default 8,192 bytes when not configured), and IF the result exceeds that size, THEN THE Agent_Service SHALL truncate it to the limit and set a truncated indicator to true on the result.
9. IF a Tool call fails with a service error (including an ownership or Guard rejection), THEN THE Agent_Service SHALL return the structured error code and message to the model, SHALL omit any stack trace, and SHALL leave the state that the service preserved unchanged.
10. IF the model requests a Tool name that is not in the registry or supplies arguments that fail the Tool's input validation, THEN THE Agent_Service SHALL NOT call any service method and SHALL return a structured error to the model that identifies the unknown Tool or the invalid argument.

### Requirement 4: Supervised mode and approval gates

**User Story:** As a user, I want the agent to ask before it spends money, so that I stay in control of cost and of every decision that changes my experiment.

#### Acceptance Criteria

1. WHEN a Session is created, THE Agent_Service SHALL set the Session's mode to Supervised_Mode, unless the creating request explicitly selects Autopilot_Mode.
2. WHILE a Session is in Supervised_Mode, WHEN the model requests a Costly_Tool, THE Agent_Service SHALL create an Approval_Request, set the Session_Status to `waiting_for_approval`, and SHALL NOT execute the Tool until an Approval_Decision of approve or edit is recorded.
3. WHILE a Session is in Supervised_Mode, WHEN the model requests a Mutating_Tool that changes an experiment's configuration (feature selection, target, treatment, split or training settings), THE Agent_Service SHALL create an Approval_Request, set the Session_Status to `waiting_for_approval`, and SHALL NOT execute the Tool until an Approval_Decision of approve or edit is recorded.
4. WHEN the model requests a Read_Tool, THE Agent_Service SHALL execute it without creating an Approval_Request, in both Supervised_Mode and Autopilot_Mode.
5. WHEN an Approval_Request is created, THE Agent_Service SHALL present to the user the Tool name, the exact Tool arguments, the model's stated rationale, and the Guards' cost estimate. IF the Guards cannot produce a cost estimate, THEN THE Agent_Service SHALL present the request with an indication that no estimate is available.
6. WHEN the user approves an Approval_Request, THE Agent_Service SHALL execute the Tool exactly once with exactly the arguments shown, return the result to the model, and set the Session_Status back to `running`.
7. WHEN the user rejects an Approval_Request, THE Agent_Service SHALL NOT execute the Tool, SHALL return a rejection result containing the user's optional reason (maximum 500 characters) to the model, and SHALL resume the Agent_Loop with the Session_Status `running`.
8. WHEN the user edits an Approval_Request, THE Agent_Service SHALL validate the edited arguments against the Tool's input schema. IF validation passes, THEN THE Agent_Service SHALL show the edited request back to the user and execute only the arguments the user then confirms. IF validation fails, THEN THE Agent_Service SHALL NOT execute the Tool, SHALL indicate which arguments failed validation, and SHALL keep the Approval_Request pending with its original arguments unchanged.
9. THE Approval_Request SHALL bind to a hash of the Tool name and arguments, and THE Agent_Service SHALL execute only a Tool call whose hash equals the hash of the recorded approval. IF the hash does not match, THEN THE Agent_Service SHALL NOT execute the Tool and SHALL return an error result indicating an approval mismatch to the model.
10. IF an Approval_Decision is submitted by a user other than the Session owner, THEN THE Agent_Service SHALL reject the decision, leave the Approval_Request pending and unchanged, and return an authorization error to the caller.
11. THE Agent_Service SHALL NOT let the model create, approve, edit or reject an Approval_Request on the user's behalf. IF the model's output contains an approval decision, THEN THE Agent_Service SHALL ignore it and leave the Approval_Request state unchanged.
12. IF an Approval_Request has no decision after the configured approval time-to-live (default 900 seconds, allowed range 60 to 86400 seconds), THEN THE Agent_Service SHALL mark the request `expired`, SHALL NOT execute the Tool, and SHALL set the Session_Status to `waiting_for_user` so that the user can ask the agent to propose the step again. IF an Approval_Decision arrives for an `expired` request, THEN THE Agent_Service SHALL reject it with an error indicating expiry.
13. THE CLI and the web chat SHALL let the user approve, reject or edit a pending Approval_Request, and SHALL offer a batch approval of a plan's consecutive Costly_Tool calls (maximum 10 calls per batch) only when the user chooses it explicitly and only for the exact arguments displayed. IF any call in a batch differs in Tool name or arguments from what was displayed, THEN THE Agent_Service SHALL NOT execute that call and SHALL create a separate Approval_Request for it.

### Requirement 5: Autopilot mode

**User Story:** As an experienced user, I want an opt-in autopilot, so that routine runs complete without my clicking approve at every step.

#### Acceptance Criteria

1. WHERE Autopilot_Mode is enabled by the user for a Session, THE Agent_Service SHALL execute each Costly_Tool call without creating an Approval_Request, provided the call is within the Session_Limits and passes every Guard and the call is not one of the categories listed in criterion 7.
2. WHEN a user enables Autopilot_Mode through an explicit user action (a CLI flag or a web control), THE Agent_Service SHALL record the enabling user's identity and the enabling timestamp (UTC, second precision) against the Session before the next Tool call is executed.
3. IF Autopilot_Mode is requested for a Session and no other path (for example, a default setting, an environment variable, or a model-issued call) is the source of the request, THEN THE Agent_Service SHALL keep the Session in Supervised_Mode and SHALL NOT enable Autopilot_Mode.
4. IF Autopilot_Mode is requested without a maximum estimated spend greater than 0 and at most the Session_Limits budget, THEN THE Agent_Service SHALL reject the request with an error indicating the missing or out-of-range maximum, SHALL keep the Session in Supervised_Mode, and SHALL NOT execute any Costly_Tool call under Autopilot_Mode.
5. WHILE a Session is in Autopilot_Mode, IF a Costly_Tool call would take the Session's cumulative estimated spend above the stated maximum, THEN THE Agent_Service SHALL NOT execute the call, SHALL create an Approval_Request for that call, and SHALL set the Session_Status to `waiting_for_approval`.
6. WHILE a Session is in Autopilot_Mode, IF a Guard refuses a call (budget, concurrency, instance type or runtime), THEN THE Agent_Service SHALL return the refusal to the model, SHALL NOT execute the call, and SHALL NOT retry the call with changed arguments that evade the Guard.
7. WHILE a Session is in Autopilot_Mode, IF a Mutating_Tool call excludes, removes or drops a feature, approves a free-form script, promotes a model, or deletes any resource, THEN THE Agent_Service SHALL create an Approval_Request for that call and SHALL NOT execute it until the user approves it.
8. THE Agent_Service SHALL NOT exclude any feature other than the target in any mode, and SHALL pass leakage findings and feature proposals to the user as advisory output that does not itself change the feature set.
9. WHEN the user selects Supervised_Mode during a running Autopilot_Mode Session, THE Agent_Service SHALL apply Supervised_Mode to every Tool call that starts after the selection, SHALL NOT start a Tool call under Autopilot_Mode after the selection, and SHALL let any Tool call already executing complete.
10. THE Agent_Service SHALL show the current mode (Autopilot_Mode or Supervised_Mode) in every Session view on both the CLI and the web surface, and SHALL update the displayed mode within 2 seconds of a mode change.

### Requirement 6: Session state and resumability

**User Story:** As a user, I want to close my terminal while a job runs and pick the session up later, so that long SageMaker jobs do not tie up my machine or a request.

#### Acceptance Criteria

1. THE Agent_Service SHALL persist each Session as a versioned Pydantic contract in S3 at a path defined only in `ml_engine/io/layout.py`.
2. WHEN a model turn, Tool result, Approval_Request or Approval_Decision is produced, THE Agent_Service SHALL persist the Session within 5 seconds and before starting the next model call or Tool call, so that a process restart loses at most the in-flight step.
3. WHEN a Tool call starts a SageMaker job, THE Agent_Service SHALL record the job reference in the Session, set the Session_Status to `waiting_for_job`, and release the request or process that was serving the user within 5 seconds.
4. WHILE a Session is `waiting_for_job`, THE Agent_Service SHALL NOT hold an HTTP request, a model call or a CLI process open for the job's duration.
5. WHEN the existing status monitor or poller records a terminal state (succeeded, failed or stopped) for a job that a Session waits on, THE Agent_Service SHALL make the Session eligible to continue within 60 seconds of that record.
6. WHEN the user opens or resumes a Session whose awaited job has reached a terminal state, THE Agent_Service SHALL continue the Agent_Loop with the job's result, including a failed or stopped result.
7. WHERE the user passes a `--watch` option to the CLI, THE CLI SHALL poll the Session status at the configured interval (default 30 seconds, allowed range 5 to 3600 seconds) and continue the Agent_Loop when the awaited job reaches a terminal state, until the user interrupts it.
8. WHEN the user requests a list of Sessions, THE Agent_Service SHALL return the user's Sessions, up to 100 per page, each with its status, mode, linked experiment ids and last activity time, ordered by last activity time with the most recent first.
9. WHEN the user resumes a Session from a different surface than the one that created it, THE Agent_Service SHALL present the same history, pending Approval_Requests and Session_Status as the persisted Session.
10. IF a resume is attempted on a Session whose Agent_Loop is already running, THEN THE Agent_Service SHALL reject the attempt with an error indicating that the Session is in use, and SHALL leave the running Agent_Loop and the persisted Session unchanged.
11. IF the model context would exceed the model's limit, THEN THE Agent_Service SHALL compact earlier turns into a deterministic summary that keeps Tool result ids and Engine_Numbers verbatim, and SHALL keep the full history in the persisted Session.
12. WHEN the user cancels a Session, THE Agent_Service SHALL set the Session_Status to `cancelled`, SHALL reject every pending Approval_Request, SHALL NOT start further Tool calls, and SHALL persist the Session.
13. WHEN the user cancels a Session that waits on a running job, THE Agent_Service SHALL ask the user whether to stop the job, and SHALL stop the job only if the user answers yes. IF the user gives no answer or answers no, THEN THE Agent_Service SHALL leave the job running and still set the Session_Status to `cancelled`.
14. IF persisting the Session to S3 fails, THEN THE Agent_Service SHALL retry up to 3 times, and if all retries fail SHALL stop the Agent_Loop, SHALL NOT start further Tool calls, and SHALL return an error to the user indicating that the Session could not be saved.

### Requirement 7: Local folder data upload and dataset discovery

**User Story:** As a user whose data sits in a local folder, I want the agent to find the data and upload it to S3, so that I can start without leaving my terminal.

#### Acceptance Criteria

1. WHEN the user names a Local_Dataset_Folder or file, THE CLI SHALL list the candidate dataset files in it (files with the extensions .csv and .parquet, non-recursive) with file name, size in bytes and modification time (UTC, ISO 8601), sorted by file name ascending, up to a maximum of 100 files. IF more than 100 candidates exist, THEN THE CLI SHALL show the first 100 and state the total count.
2. THE CLI SHALL perform local file listing and reading on the user's machine, and THE web chat SHALL accept a file or folder selection from the browser instead of reading the server's file system.
3. THE Agent_Service SHALL NOT read the server's local file system on behalf of a model-supplied path.
4. WHEN the user asks to upload a dataset, THE Agent_Service SHALL create an Approval_Request that shows the file name, the file size in bytes and the destination S3 location, in Supervised_Mode and in Autopilot_Mode, until the user has approved an upload for that Session's dataset path once.
5. WHEN an upload is approved, THE Platform SHALL upload the file through DatasetService to the S3 dataset location with the configured server-side encryption, and SHALL register the dataset for the Session owner.
6. IF the file is larger than `max_dataset_upload_bytes`, THEN THE Platform SHALL refuse the upload before transferring any bytes, report the file size in bytes and the limit in bytes, and register no dataset.
7. IF the file extension is not .csv or .parquet, or the content cannot be parsed as the format its extension names, THEN THE Platform SHALL refuse the upload with a structured error that names the supported formats (CSV, Parquet) and SHALL register no dataset.
8. IF several candidate files exist and the user did not choose one, THEN THE Agent_Service SHALL ask the user which file to use, SHALL NOT pick one, and SHALL NOT create an Approval_Request until the user has chosen.
9. WHEN an upload completes, THE Agent_Service SHALL give the model the dataset identifier, column names, inferred types, row count and the capped, governed profile that the existing governance policy allows, and SHALL NOT give the model raw rows.
10. IF an upload is interrupted or fails, THEN THE Platform SHALL report the failure to the user with the failure reason, SHALL leave no registered partial dataset and no partial object at the S3 dataset location, and SHALL allow the user to retry the upload of the same file.
11. WHEN a file whose SHA-256 content hash equals that of a dataset already registered in the same Session is uploaded, THE Platform SHALL reuse the existing registered dataset without creating a new S3 object or registration, and SHALL tell the user that it reused the existing dataset and give its dataset identifier.
12. WHEN an upload completes or is reused under criterion 11, THE Platform SHALL return a stable dataset identifier that stays the same for the life of the registered dataset, and THE Agent_Service SHALL reference the dataset in later Tool calls by that identifier and SHALL NOT pass a local path.

### Requirement 8: LLM boundary and data safety

**User Story:** As a data owner, I want a hard guarantee that raw rows never reach the model and that data cannot steer the agent.

#### Acceptance Criteria

1. THE Agent_Service SHALL send to Bedrock only the user's typed messages, the system prompt, Tool definitions, and Tool results that conform to the existing governance policy (column names, aggregate statistics and capped sample values).
2. WHERE `bedrock_send_sample_values` is false, THE Agent_Service SHALL send zero sample values to Bedrock in any request, including the Tool results of every Tool.
3. WHEN a Tool returns a result, THE Agent_Service SHALL apply the governance policy to that result in the Agent_Service before the result is added to any Bedrock request, independently of any filtering done inside the Tool.
4. WHEN a Tool result is added to a Bedrock request, THE Agent_Service SHALL wrap it in a delimited data envelope, and THE Agent_Service SHALL include in the system prompt a statement that content inside the envelope is data and carries no instructions.
5. IF Untrusted_Content contains text that attempts to change the Session's mode, Session_Limits, approvals or Tool permissions (in a column name, a cell value, a file name, a log line or an error message), THEN THE Agent_Service SHALL leave the mode, Session_Limits, approvals and Tool permissions unchanged.
6. THE Agent_Service SHALL set the mode, the Session_Limits, the Tool permissions and the Approval_Request state only from user actions and configuration, and SHALL NOT set them from model output.
7. IF a Tool result contains text that instructs the agent (for example a request to ignore earlier rules, to reveal the system prompt, or to call a Tool), THEN THE Agent_Service SHALL deliver the result to the model inside the data envelope and SHALL record a prompt-injection-suspected flag in the Agent_Audit_Record for that Tool call.
8. WHEN a string from Untrusted_Content is prepared for a Bedrock request, THE Agent_Service SHALL truncate it to the configured maximum length (default 256 characters) and SHALL mark the string as truncated.
9. THE Agent_Service SHALL exclude credentials, signed URLs and full S3 object keys outside the Session owner's data prefix from every message sent to the model.
10. IF a Tool input supplied by the model contains a file system path, S3 URI or instance type that is not present in the dataset registry or the allowed-instance configuration, THEN THE Agent_Service SHALL reject the Tool call without executing it, SHALL return to the model an error result indicating the input was not approved, and SHALL record the rejection in the Agent_Audit_Record.

### Requirement 9: Session limits and iteration caps

**User Story:** As a budget owner, I want hard caps on what one agent session can do, so that a confused or manipulated model cannot loop or overspend.

#### Acceptance Criteria

1. THE Agent_Service SHALL enforce the Session_Limits on every Session in every mode, using the default value of each limit from configuration. Each limit SHALL be a positive integer, and IF a configured value is missing, zero, negative, or non-integer, THEN THE Agent_Service SHALL refuse to start the Session and SHALL return an error indicating which limit is invalid.
2. WHEN a Session reaches the maximum number of Converse turns or the maximum number of tool calls, THE Agent_Service SHALL stop the Agent_Loop before issuing the next Converse request or tool call, set the Session_Status to `waiting_for_user`, retain the Session history, and show the user a message naming the limit reached, its configured value, and its current value.
3. WHEN the user confirms continuing after a limit stop, THE Agent_Service SHALL raise only the reached limit, and only by the positive integer amount the user confirmed, and SHALL resume the Agent_Loop. IF the user declines or gives no amount, THEN THE Agent_Service SHALL keep the Session_Status as `waiting_for_user` and leave all limits unchanged.
4. IF the model requests the same Tool with identical arguments more than the configured number of consecutive times, THEN THE Agent_Service SHALL NOT execute the call, SHALL return a repeated-call error to the model as the tool result, and SHALL count that request as a tool call. A request for a different Tool or different arguments resets the consecutive count to zero.
5. WHEN the accumulated active time of a Session reaches the configured maximum wall-clock duration in seconds, THE Agent_Service SHALL stop the Agent_Loop, set the Session_Status to `waiting_for_user`, and show the user a message naming the duration limit. Active time SHALL exclude time spent with Session_Status `waiting_for_user`.
6. WHEN a Costly_Tool call is made in a Session, THE Agent_Service SHALL add 1 to the Session's Costly_Tool call count and add the Guards' estimated spend for that call, in the Guards' currency unit, to the Session's estimated spend, and SHALL show both counters in the Session view within 2 seconds.
7. THE Agent_Service SHALL NOT change any Session_Limit in response to model output, including a model request, tool arguments, or text in tool results.
8. THE Agent_Service SHALL apply the Session_Limits in addition to the per-experiment budget Guards, the monthly budget Guards, and the concurrency Guards, so that a call proceeds only if every one of them allows it.
9. WHERE the Guards' budget enforcement is off, THE Agent_Service SHALL still stop the Agent_Loop and set the Session_Status to `waiting_for_user` when the Session's Costly_Tool call count reaches the Costly_Tool call cap.

### Requirement 10: Narration with attribution

**User Story:** As a decision maker, I want every number the agent quotes to be traceable to the engine, so that I can trust the summary.

#### Acceptance Criteria

1. THE Agent_Service SHALL obtain every Engine_Number shown to the user from a Tool result and SHALL NOT let the model compute, estimate, round, or alter an Engine_Number.
2. WHEN the model produces a Narration, THE Agent_Service SHALL check every numeric value in the Narration against the Tool results of the same Session before the Narration is shown, and SHALL treat a value as matching only if it equals a Tool result value after applying the display rounding rule.
3. IF a Narration contains a numeric value that matches no Tool result of the same Session, THEN THE Agent_Service SHALL either withhold the sentence containing it or mark that sentence as unverified in the displayed answer, SHALL keep the remaining verified sentences unchanged, and SHALL record one Agent_Audit_Record entry per affected sentence that identifies the Session and the unmatched value.
4. WHEN the Agent_Service returns a final answer, THE Agent_Service SHALL attach an Attribution to 100% of the Engine_Numbers in that answer, and THE surfaces SHALL display the Attribution next to each Engine_Number.
5. WHEN the answer includes a Free_Form_Run metric, THE Agent_Service SHALL show its label (Self_Reported_Metric or Engine_Metric) immediately beside the number.
6. WHEN the answer compares models, THE Agent_Service SHALL take the comparison from the platform's comparison output, and IF the comparison output marks two entries as not comparable, THEN THE Agent_Service SHALL state in the answer that those entries are not comparable and SHALL NOT rank them against each other.
7. WHEN a Tool result contains a report or a metrics table, THE Agent_Service SHALL include that report or table in the answer verbatim, with no value changed, omitted, or reformatted, in addition to the model's prose.
8. WHEN the answer includes a leakage finding, a feature proposal, or a model ranking, THE Agent_Service SHALL label it as advisory in the answer and SHALL NOT describe it as guaranteed, certain, or proven.

### Requirement 11: Audit of agent actions

**User Story:** As an auditor, I want to see what the agent did, for whom, and who approved it.

#### Acceptance Criteria

1. WHEN a Mutating_Tool execution, a Costly_Tool execution, an Approval_Decision, a mode change, a limit stop, a cancellation or a Session creation occurs, THE Agent_Service SHALL write one Agent_Audit_Record for that occurrence.
2. THE Agent_Audit_Record SHALL contain the Session id, the Session owner, the acting identity, the event type, the time in UTC, the result status (one of success, failure, rejected or cancelled), the Tool name and a hash of the Tool arguments (both empty for events that are not Tool executions), and, for an Approval_Decision or a Tool execution that required approval, the Approval_Decision and who made it (the approving user, or `autopilot` together with the user who enabled it).
3. WHEN a Tool executes inside a service that writes its own audit entry, THE Agent_Service SHALL leave that entry unmodified and SHALL include in the Agent_Audit_Record a reference to that entry, so that a query by Session id returns both the Agent_Audit_Record and the service's entry.
4. THE Agent_Service SHALL store each Agent_Audit_Record through the existing audit mechanism and SHALL NOT replace that mechanism.
5. THE Agent_Service SHALL NOT include credentials, access tokens or raw data rows in any Agent_Audit_Record.
6. WHEN a Session owner requests the audit trail of that Session, THE Agent_Service SHALL return all Agent_Audit_Records of that Session in chronological order, or an empty list if there are none.
7. IF a user who is not the Session owner requests the audit trail of a Session, THEN THE Agent_Service SHALL reject the request with an error indicating that access is denied and SHALL return no Agent_Audit_Records.
8. WHERE `audit_log_enabled` is false, THE Agent_Service SHALL write no Agent_Audit_Record, as the existing audit mechanism writes no entry when audit is disabled, and SHALL continue the Session normally.
9. IF writing an Agent_Audit_Record for a Mutating_Tool or Costly_Tool execution fails, THEN THE Agent_Service SHALL NOT execute further Mutating_Tools or Costly_Tools in that Session, SHALL return an error to the Session owner indicating that auditing failed, and SHALL keep the Session and its existing records.

### Requirement 12: Shared agent service, CLI and web chat

**User Story:** As a user, I want the same agent in my terminal and in the browser, so that I can start in one and continue in the other.

#### Acceptance Criteria

1. THE Agent_Service SHALL implement the Agent_Loop, the Session store, the approval logic and the limits in one place, and THE CLI and the web chat SHALL call the Agent_Service without re-implementing any of them.
2. WHEN a user runs the CLI command `mlf` with a goal as an argument and no Session identifier, THE CLI SHALL create a new Session and print its Session identifier within 5 seconds.
3. WHEN a user runs `mlf` with a Session identifier of an existing Session owned by that user, THE CLI SHALL resume that Session and continue from its stored state.
4. THE CLI SHALL provide subcommands or options to list Sessions, resume a Session, approve an Approval_Request, reject an Approval_Request, edit an Approval_Request, select the mode, cancel a Session, show a Session's audit trail and watch a Session.
5. WHEN the CLI streams a Session, THE CLI SHALL print, in the order the events occurred, the model's text, each Tool call with an argument summary of at most 200 characters, each result summary of at most 500 characters, and every Approval_Request.
6. WHEN stdout is not an interactive terminal and an Approval_Request is pending, THE CLI SHALL print the pending Approval_Request, SHALL exit with the exit code 75 that means "waiting for approval", and SHALL NOT approve the Approval_Request.
7. IF the CLI cannot reach the control plane, the Session identifier does not exist, or the Session belongs to another user, THEN THE CLI SHALL print an error message indicating the cause, SHALL exit with a non-zero code other than 75, and SHALL leave the stored Session state unchanged.
8. THE Platform SHALL expose the Agent_Service through `/api` routes for creating a Session, sending a message, reading a Session, listing Sessions, deciding an Approval_Request, setting the mode and cancelling a Session, using the existing authentication and ownership handling.
9. IF a request to the Agent_Service routes is unauthenticated, or targets a Session owned by another user, THEN THE Platform SHALL reject the request with an error indicating authentication failure or access denial, and SHALL NOT disclose or change the Session.
10. THE Platform SHALL add a chat panel to the static web UI under `backend/api/ui` that works without a build step.
11. THE web chat SHALL show the transcript, the Tool calls, the pending Approval_Requests with approve, reject and edit controls, the mode, the counters and the Session_Status, and SHALL update them within 2 seconds of the Agent_Service reporting a change.
12. WHILE a Session is `waiting_for_job`, THE web chat SHALL refresh the Session status by polling at the configured interval and SHALL NOT keep a model call open.
13. THE CLI SHALL authenticate to the control plane the same way the existing API clients do, and SHALL record the same user identity that the web UI records for that person.
14. THE CLI SHALL read AWS credentials through the standard AWS credential chain, and WHERE `aws_profile` or `aws_role_arn` is configured, THE CLI SHALL use the configured value.
15. THE Platform SHALL provide the MCP server's existing Tools with unchanged names, input parameters and result formats, and THE Agent_Service SHALL run with LibreChat not deployed.

### Requirement 13: Bedrock configuration, credentials and model choice

**User Story:** As an operator, I want to configure which model the agent uses and how it authenticates, so that I can control cost, region and access.

#### Acceptance Criteria

1. THE Platform SHALL read the agent's settings from `Settings` with the `ML_FACTORY_` prefix, and SHALL add them as new optional fields with defaults, leaving the existing fields' behaviour unchanged.
2. THE Platform SHALL provide settings for: the agent model id, the maximum output tokens per turn (integer, 1 to 8,192), the Converse retry count (integer, 0 to 5), the Session_Limits defaults, the approval time-to-live (integer seconds, 60 to 86,400), the Tool result size cap (integer bytes, 1,024 to 1,048,576), the Untrusted_Content string cap (integer characters, 1 to 100,000), the CLI and web poll intervals (each in seconds, 0.5 to 60), and an agent-enabled flag (boolean).
3. IF a setting from criterion 2 is set to a value outside its stated range or of the wrong type, THEN THE `Settings.problems()` check SHALL report the setting name and its allowed range, and THE Platform SHALL keep the existing fields' values unchanged.
4. WHERE the agent model id is not set or is an empty string, THE Agent_Service SHALL fall back to `bedrock_model_id`.
5. WHILE the agent-enabled flag is false (the default), THE Platform SHALL refuse to create Sessions, SHALL create no Session record, and SHALL return an error message that names the setting `ML_FACTORY_AGENT_ENABLED` as the missing setting.
6. WHEN the agent is enabled, THE `Settings.problems()` check SHALL report a missing model id (both the agent model id and `bedrock_model_id` empty), and SHALL report a region mismatch between the model id and `aws_region`, using the existing `bedrock_region_mismatch` rule.
7. THE Converse_Client SHALL use the existing `backend.aws.build_client` for the `bedrock-runtime` client so that proxy, profile and role settings apply.
8. IF the configured model does not support Converse tool use, THEN THE Platform SHALL report, on the first Converse call with that model, an error that contains the model id and the reason, SHALL end the Session turn with a failed status, and SHALL NOT fall back to another model.
9. THE Platform SHALL NOT write AWS credentials (access key id, secret access key, session token) to any log entry, SHALL NOT persist them, and SHALL NOT include them in the Session or in any Tool result or error message returned to the user.

### Requirement 14: Failure handling and recovery

**User Story:** As a user, I want clear failures and a way to continue, so that a failed job or a lost connection does not lose my session.

#### Acceptance Criteria

1. IF a SageMaker job started by a Tool reaches a terminal failed state, THEN THE Agent_Service SHALL, within 5 seconds of detecting the state, return the job's failure reason to the model as a structured tool result containing the job identifier, the terminal state, and the failure reason text, and SHALL let the model propose a next step under the normal approval rules.
2. IF the model requests a resubmission of a failed Costly_Tool call in Supervised_Mode, THEN THE Agent_Service SHALL NOT execute it until a new Approval_Request for that call has been approved.
3. WHEN the model resubmits a failed Costly_Tool call in Autopilot_Mode, THE Agent_Service SHALL count the resubmission as one additional Costly_Tool call against the Session_Limits, and IF the Session_Limits would be exceeded, THEN THE Agent_Service SHALL NOT execute the call and SHALL return a structured error to the model stating that the limit was reached.
4. IF a Tool call raises an unexpected exception, THEN THE Agent_Service SHALL record the failure in the Session, return a structured error to the model that contains the Tool name and a generic failure indication and excludes the exception message and stack trace, and keep the Session_Status resumable.
5. WHEN a Session is resumed and a Costly_Tool call was in flight when the process stopped, THE Agent_Service SHALL query the platform's records for that call's idempotency key before taking any other action on the call, and SHALL NOT execute the call again without a new decision under the normal approval rules.
6. IF the platform's records cannot be queried within 30 seconds on resume of an in-flight call, THEN THE Agent_Service SHALL mark the call's outcome as unknown, report it to the user and the model as a structured error, and SHALL NOT execute the call again.
7. THE Agent_Service SHALL give every Costly_Tool call an idempotency key derived from the Session id and the hash of the approved arguments, so that executing the same approved call more than once submits at most one job.
8. IF a persisted Session fails to load because its contract version is unknown or its content is corrupt, THEN THE Agent_Service SHALL return an error that contains the Session id and the failure category (unknown contract version or corrupt content), and SHALL NOT modify or delete the stored object.
9. IF the user's credentials expire during a Session, THEN THE Agent_Service SHALL set the Session_Status to `failed` with the retriable flag set to true and a message instructing the user to refresh credentials, and SHALL retain all Session data recorded up to that point.
10. WHEN a Session enters the `failed` status, THE Agent_Service SHALL show the failure cause and the last successful step, and SHALL allow the user to resume the Session.
11. IF an awaited job is deleted or cannot be found, THEN THE Agent_Service SHALL return a structured error to the model that contains the job identifier and a not-found indication, SHALL NOT report the job as succeeded, and SHALL keep the Session resumable.

### Requirement 15: Contracts, paths and compatibility

**User Story:** As a maintainer, I want the agent added through versioned, additive contracts, so that existing behaviour and data stay intact.

#### Acceptance Criteria

1. THE Platform SHALL define the Session, Approval_Request, Approval_Decision, Tool call record, Session_Limits and Agent_Audit_Record as Pydantic contracts, each carrying a schema version field with a default value.
2. IF a stored record of one of the contracts in criterion 1 has a schema version that the Platform does not support, THEN THE Platform SHALL reject the record with an error identifying the contract name and the unsupported version, and SHALL leave the stored record unchanged.
3. THE Platform SHALL define every new artifact path only in `ml_engine/io/layout.py`, and no other module SHALL build an agent artifact path from string literals.
4. THE Platform SHALL place the agent code in new modules and SHALL add to existing modules only additively (for example new `Settings` fields, new container wiring and new API routers), without removing or changing the signature, default value or behaviour of any existing public function, class, `Settings` field, route or stored artifact.
5. WHERE an inherited engine file changes, THE change SHALL be additive and SHALL be recorded in `tests/inherited_manifest.json` with a non-empty written justification for each changed file.
6. THE Agent_Service SHALL use SageMaker for every compute action and S3 for every stored artifact, and SHALL NOT add a local database, a local job runner or a local artifact store.
7. THE `AppContainer` SHALL expose the Agent_Service, built from `Settings` and injectable clients, so that tests can substitute the Converse_Client and the store without network access or AWS credentials.
8. WHEN the agent is disabled, THE Platform SHALL return the same responses for every pre-existing API route as it does without this feature, SHALL NOT call the Converse_Client, and SHALL NOT write any agent artifact to S3.

### Requirement 16: Offline, deterministic testing

**User Story:** As a developer, I want the agent covered by fast offline tests, so that safety properties are verified on every run.

#### Acceptance Criteria

1. WHEN the default test suite runs, THE test suite SHALL run the Agent_Service with a Scripted_Fake_Converse_Client, `InMemoryObjectStore` and the fake SageMaker, and SHALL make zero outbound network connections.
2. THE Scripted_Fake_Converse_Client SHALL replay the scripted sequence of text and tool-use turns in order. IF the Agent_Service sends a request that does not match the next scripted step, THEN THE Scripted_Fake_Converse_Client SHALL fail the test with an error identifying the unexpected request. WHEN the script is exhausted before the Agent_Loop ends, THE Scripted_Fake_Converse_Client SHALL fail the test.
3. THE test suite SHALL include a test that, in Supervised_Mode, no Costly_Tool executes without a recorded approval, and that changing any argument after approval invalidates the approval and prevents execution.
4. THE test suite SHALL include a test that a model-issued tool call to create, approve or edit an Approval_Request, or to change the mode or Session_Limits, is rejected, and that the Approval_Request, mode and Session_Limits are unchanged afterwards.
5. THE test suite SHALL include a property-based test, run with at least 100 generated sequences of scripted model turns, asserting that the Session's Costly_Tool call count and turn count never exceed the Session_Limits.
6. WHILE sample values are disabled, THE test suite SHALL include a test that no Tool result sent to the Scripted_Fake_Converse_Client contains raw rows or sample values.
7. THE test suite SHALL include a test that a Tool result containing injected instructions leaves the mode, Session_Limits and approvals unchanged and sets the prompt-injection-suspected flag.
8. THE test suite SHALL include a property-based test, run with at least 100 generated cases, asserting that every number in an accepted Narration appears in a Tool result of the Session, and a test that a Narration containing a number absent from all Tool results is withheld or flagged.
9. THE test suite SHALL include a round-trip property test, run with at least 100 generated Sessions, asserting that serializing a Session, loading it, and serializing it again yields a Session equal to the original in every field.
10. THE test suite SHALL include a test that a Session resumes after a simulated process restart while a job is running, that the job completes after the restart, that the Agent_Loop continues, and that exactly one job submission is recorded for that job.
11. THE test suite SHALL include a test that Autopilot_Mode refuses any action that a Guard blocks, stops before the spend maximum is exceeded, and never excludes a feature.
12. THE test suite SHALL include a test that the owner can read and decide the same Session through both the CLI and the API with identical results, and that a non-owner is refused when reading, deciding or cancelling it, with the Session left unchanged.
13. THE test suite SHALL include a test of the local upload step using a temporary folder, covering candidate listing, rejection of a file above the size limit, rejection of a file that fails the format check, and reuse of an existing upload when the content is a duplicate.
14. THE test suite SHALL include a test that, in non-interactive mode with an Approval_Request pending, the CLI exits with the "waiting for approval" exit code, records no approval, and executes no Costly_Tool.
15. THE test suite SHALL include a test that every audit record carries the Session id, the approver and the arguments hash, and that no audit record contains credentials or data rows.
16. THE test suite SHALL include a contract test of the Converse_Client against recorded Bedrock Converse response shapes, covering a text turn, a tool-use turn, a throttling error and an unsupported-model error, and asserting that each is mapped to the Converse_Client's defined result or error type.
17. WHEN the default test suite runs, THE test suite SHALL complete the agent tests within 120 seconds on the development machine, excluding tests explicitly marked as requiring network access.

### Requirement 17: Early real-AWS smoke test

**User Story:** As the platform owner, I want the first real-AWS run to happen early and in a controlled way, so that fakes do not hide integration problems.

#### Acceptance Criteria

1. THE Platform SHALL provide an opt-in Smoke_Test that runs the assistant's path against real AWS (Bedrock Converse, S3, SageMaker) on a public or synthetic dataset of at most 10,000 rows and 5 MB.
2. WHERE the explicit Smoke_Test environment flag is set to an enabled value, THE test suite SHALL run the Smoke_Test, and IF the flag is unset or not enabled, THEN THE default test suite SHALL skip the Smoke_Test and report it as skipped without making any AWS call.
3. WHILE the Smoke_Test runs, THE Smoke_Test SHALL operate in Supervised_Mode, and THE test harness SHALL grant each approval request automatically.
4. WHEN the Smoke_Test runs, THE Smoke_Test SHALL execute these steps in order: upload, experiment creation, EDA, one training run, validation, final report. THE Smoke_Test SHALL mark each step as passed or failed.
5. THE Smoke_Test SHALL use the smallest instance type permitted for the training job, Session_Limits set to at most 50% of each production default, and a configured maximum estimated spend of no more than 5 USD.
6. IF the running estimated spend reaches the configured maximum, THEN THE Smoke_Test SHALL stop starting new jobs, SHALL stop any running job it started, SHALL report a failure that names the spend limit, and SHALL leave the Session resumable.
7. WHEN the Smoke_Test starts, THE Smoke_Test SHALL verify the following three conditions, and SHALL fail the run at the first one that does not hold. (a) The configured Converse model accepts a tool-use request in the configured region. (b) The execution role can start the required SageMaker jobs and read and write the S3 locations used. (c) The Session is saved to S3, and a new run loads it and resumes with identical state.
8. THE Platform SHALL schedule the Smoke_Test after the Converse_Client and a minimal Read_Tool loop exist and before the web chat build begins.
9. WHEN the Smoke_Test finishes, whether it passed or failed, THE Smoke_Test SHALL print the Session id, the job references for every job it started, and the estimated spend in USD, so that the user can inspect the run and clean up.
10. IF any Smoke_Test step or verification fails, THEN THE Smoke_Test SHALL report the failing step and an error indicating the cause, SHALL exit with a non-zero result, and SHALL keep the Session and its stored data so that it can be resumed for diagnosis.

## Out of Scope

- Autonomous deletion, promotion or deployment of models, and approval of free-form scripts by the agent.
- Any compute outside SageMaker, any artifact store outside S3, and any local model training.
- Voice input, multi-user shared Sessions, and multi-agent orchestration.
- A single-page application build pipeline for the web chat.
- Fine-tuning or hosting an LLM; the agent uses Bedrock models as they are.
- Streaming token-by-token output over the API (the first version returns whole turns; the CLI may render them incrementally).

## Open Questions for the Owner

| # | Question | Working assumption |
|---|---|---|
| 1 | Which Bedrock model id and region will the agent use for Converse tool use? | Use `bedrock_model_id`, and let the Smoke_Test confirm tool-use support. |
| 2 | What are the default Session_Limits (turns, tool calls, Costly_Tool calls, spend, duration)? | Conservative defaults set in configuration, for example 40 turns, 60 tool calls, 10 Costly_Tool calls, 2 hours of active time. The owner will confirm. |
| 3 | Should Autopilot_Mode be allowed in every deployment, or switched off by an operator setting? | Allowed, but an operator can disable it with a setting. |
| 4 | Should the web chat accept files by browser upload only, or also a server-side folder? | Browser upload only (Requirement 7.2), because the server must not read arbitrary local paths. |
| 5 | How long should an Approval_Request stay valid? | 15 minutes (900 seconds) by default, configurable up to 24 hours (Requirement 4.12). Decided by the owner. |
