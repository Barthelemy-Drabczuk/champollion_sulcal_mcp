from __future__ import annotations

from fastmcp import Context

from .. import preflight

MCP_VERSION = "0.1.0"

STAGE_DESCRIPTIONS = {
    "morphologist": "Generate sulcal graphs from T1 MRI using Morphologist",
    "cortical_tiles": "Extract 28 standardized sulcal region crops",
    "config": "Generate Champollion dataset YAML configuration",
    "embeddings": "Compute 56-fold sulcal embeddings (28 regions × 2 hemispheres)",
    "combine": "Collect per-region embedding CSVs into single output directory",
    "snapshots": "Render sulcal graph meshes, cortical tile masks, and UMAP plots",
}

TOOL_NAMES = [
    "start_morphologist", "start_cortical_tiles", "start_config",
    "start_embeddings", "start_combine", "start_snapshots",
    "start_pipeline",
    "get_job_status", "list_jobs", "cancel_job", "get_job_log",
    "get_pipeline_info", "preflight_check",
]


async def get_pipeline_info(ctx: Context | None = None) -> dict:
    """Get metadata about the Champollion pipeline and available MCP tools."""
    if ctx:
        await ctx.info("Returning pipeline info")
    return {
        "version": MCP_VERSION,
        "stages": STAGE_DESCRIPTIONS,
        "tools": TOOL_NAMES,
        "transport": "stdio",
    }


async def preflight_check(ctx: Context | None = None) -> dict:
    """Check whether the Champollion pipeline is correctly configured and accessible."""
    try:
        result = preflight.summarize()
    except Exception as exc:
        result = {"ok": False, "error": str(exc)}
    if ctx:
        status = "ok" if result.get("ok") else "issues found"
        await ctx.info(f"Preflight check: {status}")
    return result
