# IAM policies — least-privilege, separated roles (requirement 15.3)

The MCP server runs on an internal server next to LibreChat. That server has a **base identity**
(an IAM user or role on the box) whose only permission is to assume the **control-plane role**.
The control plane passes the **SageMaker job role** to every job it starts. The deployer identity
that creates these roles is separate from all of them.

| Role / identity | Permissions | Trust policy | Used by |
|---|---|---|---|
| Server base identity | `server_base_identity_policy.json` | — | the internal server; set as the standard AWS credentials of the MCP process |
| Control plane | `control_plane_policy.json` | `control_plane_trust.json` | the MCP server, via `ML_FACTORY_AWS_ROLE_ARN` (credentials refresh automatically) |
| SageMaker job | `sagemaker_job_policy.json` (+ `sagemaker_job_kms_policy.json` if SSE-KMS, + `sagemaker_job_vpc_policy.json` if jobs run in a VPC) | `sagemaker_job_trust.json` | every Processing/Training job (the stage image) |
| Step Functions | `step_functions_policy.json` | — | **not used yet** (Phase 5); orchestration is direct boto3 today |

## Placeholders to replace

`${ARTIFACT_BUCKET}`, `${REGION}`, `${ACCOUNT_ID}`, `${SAGEMAKER_JOB_ROLE_ARN}`,
`${CONTROL_PLANE_ROLE_ARN}`, `${SERVER_BASE_IDENTITY_ARN}`, `${ECR_REPOSITORY_ARN}`
(e.g. `arn:aws:ecr:us-east-1:123456789012:repository/ml-factory-stage`), `${BEDROCK_MODEL_ARN}`
(only if Bedrock is enabled), `${KMS_KEY_ARN}` (only if SSE-KMS is used).

## Notes

- The job role can pull **only** the stage image repository. `ecr:GetAuthorizationToken`,
  `cloudwatch:PutMetricData` and the `ec2:Describe*` calls have no resource-level scope in AWS,
  so they are the only statements allowed `Resource: "*"` (the policy test enforces this list).
- The control plane may `iam:PassRole` **only** the SageMaker job role, and only to
  `sagemaker.amazonaws.com`.
- No role has `s3:*` or `sagemaker:*`; actions are enumerated.
- No role may create inference endpoints or models (requirement 4.6).
- **Datasets in another bucket** (training data or holdout-evaluation files): add the location to
  `ML_FACTORY_ALLOWED_DATASET_PREFIXES`, and grant `s3:GetObject` on it (plus `s3:ListBucket`
  with a prefix condition if you point at a folder of Parquet parts) to **both** the control-plane
  role (it checks the file exists) and the SageMaker job role (the job reads it). Files uploaded
  through the platform's upload flow land in the artifact bucket and need nothing extra.
- If the internal server reaches AWS through a proxy, set `ML_FACTORY_HTTPS_PROXY`. The server
  needs outbound HTTPS to STS, S3, SageMaker, CloudWatch Logs and (if enabled) Bedrock.
- `cfn-lint`/IAM policy simulator validation is a deployment-time step.
  `tests/infrastructure/test_iam_policies.py` checks the account-independent invariants.
