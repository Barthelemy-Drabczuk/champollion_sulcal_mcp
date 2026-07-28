from __future__ import annotations

from fastmcp import FastMCP

from .tools import jobs, pipeline, stages, utils

mcp = FastMCP("champollion-sulcal", mask_error_details=True)

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
