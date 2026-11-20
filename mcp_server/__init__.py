"""ML Factory MCP server: typed tools over the control-plane services (requirement 9).

The tools hold no ML algorithm and no business logic — each is a thin wrapper over a
``backend.services`` call (design DD-5). Read-only tools are annotated as such; mutations require
authorization; long-running work returns an execution id immediately and never blocks; results
are metadata, summaries, bounded previews and artifact references — never a full dataset.
"""

from mcp_server.server import build_mcp_server

__all__ = ["build_mcp_server"]
