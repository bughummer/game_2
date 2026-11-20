# Running the MCP server on an internal server

The MCP server runs as one long-lived process on the same server as LibreChat. It does no ML
work itself: it receives tool calls from LibreChat, starts SageMaker jobs, and runs the status
poller that moves each experiment to its next stage. All heavy compute runs on SageMaker.

## What the platform engineer provides

| Item | Goes into |
|---|---|
| S3 artifact bucket (+ prefix) | `ML_FACTORY_ARTIFACT_ROOT` |
| SageMaker job role ARN (`infra/iam/sagemaker_job_*`) | `ML_FACTORY_SAGEMAKER_EXECUTION_ROLE_ARN` |
| Stage image pushed to ECR, **by digest** | `ML_FACTORY_STAGE_IMAGE_URI` |
| Control-plane role ARN (`infra/iam/control_plane_*`) | `ML_FACTORY_AWS_ROLE_ARN` |
| Base credentials on the server, allowed only `sts:AssumeRole` on that role | AWS profile or credentials file |
| HTTPS proxy, if outbound traffic needs one | `ML_FACTORY_HTTPS_PROXY` |
| Outbound HTTPS from the server to STS, S3, SageMaker, CloudWatch Logs (+ Bedrock) | firewall |
| Region, S3 encryption choice, VPC subnets/security groups for jobs (if any) | see the env example |

## Install (Linux)

```bash
sudo useradd --system --home /opt/ml-factory mlfactory
sudo git clone <repo> /opt/ml-factory && cd /opt/ml-factory
sudo python3 -m venv .venv
sudo .venv/bin/python -m pip install ".[backend,mcp]"
sudo mkdir -p /etc/ml-factory
sudo cp infra/server/ml-factory.env.example /etc/ml-factory/ml-factory.env   # then edit it
sudo chown -R mlfactory:mlfactory /opt/ml-factory /etc/ml-factory
sudo chmod 600 /etc/ml-factory/ml-factory.env
sudo cp infra/server/ml-factory-mcp.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now ml-factory-mcp
```

The server needs Python 3.14 (the version the stage image and `pyproject.toml` target).

## Check it

```bash
curl http://localhost:8000/health
```

`"status": "ok"` with an empty `configuration_problems` list means it is ready. If a required
setting is missing or invalid, the service does not start: `journalctl -u ml-factory-mcp` shows
"Configuration is incomplete: ..." listing exactly what to fix. Then add the `mcpServers` block
from `infra/librechat/librechat.mcp.example.yaml` to `librechat.yaml` with
`url: "http://localhost:8000/sse"` and reload LibreChat.

## Operating notes

- Run **one** instance per artifact root. The poller is not safe to run twice; any second
  instance must set `ML_FACTORY_STATUS_POLL_INTERVAL_SECONDS=0`.
- A restart is safe: on startup the poller rescans S3 for running executions and resumes them.
- Upgrading the engine means a new stage image: update `ML_FACTORY_STAGE_IMAGE_URI` to the new
  digest and `systemctl restart ml-factory-mcp`. Upgrading the server code is `git pull`,
  reinstall, restart.
- On Windows, run the same command (`.venv\Scripts\python -m mcp_server.app`) as a service with
  a service wrapper such as NSSM, with the same environment variables.
