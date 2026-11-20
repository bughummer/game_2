---
inclusion: always
---

# Security and governance steering — ML Factory 2.0

A corporate internal platform on AWS. Design for authentication, authorization, least privilege,
cost control and auditability from the first phase.

## Identity and authorization

- Do not trust raw identity headers without a verified auth boundary. Until the LibreChat auth
  mechanism is chosen, a single configured dev identity is used and logged as such.
- LibreChat authentication does not establish ML Factory authorization; the backend checks it for
  every state-changing operation.
- Enforce experiment ownership and user-scoped artifact access. Read-only MCP tools are marked;
  mutations require authorization.

## AWS credentials, IAM and spend

- No hardcoded credentials. Use the default chain, a profile, or an assumed role. Secrets are
  `SecretStr` settings loaded from a gitignored file or the environment, never committed.
- Separate least-privilege roles: control plane, SageMaker job, Step Functions. Job role access is
  scoped to the artifact bucket/prefix. The deployer identity is separate.
- S3 server-side encryption on every write; TLS in transit.
- Every job submission passes the guards: CPU instance allow-list and ceiling, concurrency limit,
  per-experiment and per-month budget. Managed Spot by default; On-Demand only with an explicit
  per-experiment opt-in. No inference endpoints.

## Data governance

- Ingestion is CSV/TSV/Parquet upload only. No database discovery, no SQL.
- Never move a full dataset through the LLM context. The MCP returns metadata, summaries, bounded
  previews and artifact references.
- What reaches Bedrock is governed by configuration; default is column names, association and
  importance stats, and capped sample values. Logs are bounded and PII-redacted before an LLM
  sees them.
- Enforce file-size limits, retention, audit history and quotas.

## Safe LLM tool execution

- The LLM never executes Python, shell or SQL. Tool arguments are validated against contracts; a
  tool never acts outside its declared scope. No arbitrary AWS CLI/SDK tool.

## Development boundaries

- Do not access corporate LibreChat, MCP servers, databases or networks from development.
- Use real AWS credentials or perform AWS mutations only when explicitly authorized. Tests mock
  AWS by default.
