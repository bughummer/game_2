# LibreChat integration

ML Factory exposes its tools over an SSE MCP endpoint — the same transport as the other MCP
servers in this deployment (e.g. successfactors). LibreChat connects to that endpoint and the
chat agent drives the whole workflow: upload a dataset, review leakage, start training, read
results and reports.

## Serve the MCP

```
python -m mcp_server.app        # reads ML_FACTORY_* from the environment
```

Serves the MCP over SSE at `/sse` and a load-balancer health check at `/health`. In production
this runs as the Fargate container (one task), private, behind an internal load balancer.
Transport/path are overridable with `ML_FACTORY_MCP_TRANSPORT` and `ML_FACTORY_MCP_PATH`.

## Point LibreChat at it

Copy the `mcpServers` block from `librechat.mcp.example.yaml` into your `librechat.yaml` and set
the `url` to the service's address. Reload LibreChat; the ML Factory tools appear to the agent.

## Still open (by design)

- **Auth** (task 6.5 / Q8). The server uses a flagged development identity and trusts no raw
  header. When the real mechanism is chosen, forward the verified identity in the `headers` block
  and resolve it in `backend/auth.py::Authorizer.principal`.
- **Network path.** If LibreChat is not in the same VPC, it needs a private connection (VPN or
  PrivateLink) to reach the internal endpoint. Confirm with the infrastructure team.
- **Deployment shape** (Q8). The one-app layout (service package + MCP + health) is what
  `mcp_server.app` serves; a separate FastAPI layer can be added over the same services later.
