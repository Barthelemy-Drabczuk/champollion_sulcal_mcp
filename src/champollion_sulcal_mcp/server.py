from __future__ import annotations

import json

from fastmcp import Context, FastMCP

from .roots import get_roots
from .tools import jobs, pipeline, stages, utils

mcp = FastMCP("champollion-sulcal", mask_error_details=True)


@mcp.resource("champollion://data", mime_type="application/json")
async def list_data_root(ctx: Context) -> str:
    """List top-level entries within all client-declared data roots."""
    roots = await get_roots(ctx)
    if not roots:
        return json.dumps({"roots": [], "note": "No roots declared by client."})
    listing: dict[str, list[str]] = {}
    for root in roots:
        if root.is_dir():
            listing[str(root)] = sorted(
                p.name for p in root.iterdir() if not p.name.startswith(".")
            )
    return json.dumps({"roots": listing})

# Stage launchers
mcp.tool(stages.start_morphologist)
mcp.tool(stages.start_cortical_tiles)
mcp.tool(stages.start_config)
mcp.tool(stages.start_training)
mcp.tool(stages.start_embeddings)
mcp.tool(stages.start_combine)
mcp.tool(stages.start_snapshots)

# Composite pipeline
mcp.tool(pipeline.start_pipeline)

# Streaming (scan-centric parallel) pipeline
mcp.tool(stages.start_streaming)

# Maintenance / cleanup
mcp.tool(stages.purge_subject)
mcp.tool(stages.prune_failed_subjects)

# Job lifecycle
mcp.tool(jobs.get_job_status)
mcp.tool(jobs.list_jobs)
mcp.tool(jobs.cancel_job)
mcp.tool(jobs.get_job_log)

# Utilities
mcp.tool(utils.get_pipeline_info)
mcp.tool(utils.preflight_check)


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
