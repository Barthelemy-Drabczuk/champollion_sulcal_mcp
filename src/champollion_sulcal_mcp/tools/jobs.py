from __future__ import annotations

from fastmcp import Context
from fastmcp.exceptions import ToolError

from .. import job_store, runner


async def get_job_status(output_dir: str, job_id: str, ctx: Context | None = None) -> dict:
    """Get the current status and progress of a running or completed job."""
    try:
        state = job_store.read_job(output_dir, job_id)
    except FileNotFoundError:
        raise ToolError(f"Unknown job_id {job_id!r} for output_dir {output_dir!r}") from None
    if ctx:
        await ctx.info(f"Job {job_id}: {state.status}")
    return state.model_dump()


async def list_jobs(
    output_dir: str,
    status: str | None = None,
    limit: int = 50,
    ctx: Context | None = None,
) -> list[dict]:
    """List all jobs for a given output directory, optionally filtered by status."""
    jobs = job_store.list_jobs(output_dir)
    if status:
        jobs = [j for j in jobs if j.status == status]
    if ctx:
        await ctx.info(f"Found {len(jobs)} jobs in {output_dir}")
    return [j.model_dump() for j in jobs[:limit]]


async def cancel_job(output_dir: str, job_id: str, ctx: Context | None = None) -> dict:
    """Cancel a running job by sending SIGTERM to its process."""
    try:
        state = await runner.cancel(output_dir, job_id)
    except FileNotFoundError:
        raise ToolError(f"Unknown job_id {job_id!r} for output_dir {output_dir!r}") from None
    if ctx:
        await ctx.info(f"Job {job_id} cancel requested, status: {state.status}")
    return state.model_dump()


async def get_job_log(
    output_dir: str,
    job_id: str,
    tail_lines: int = 200,
    ctx: Context | None = None,
) -> dict:
    """Retrieve the last N lines of a job's log output."""
    tail_lines = max(1, min(tail_lines, 5000))
    try:
        state = job_store.read_job(output_dir, job_id)
    except FileNotFoundError:
        raise ToolError(f"Unknown job_id {job_id!r} for output_dir {output_dir!r}") from None

    from pathlib import Path

    log = Path(state.log_path)
    if not log.exists():
        raise ToolError(f"Log file not available for job {job_id}")

    lines = log.read_text(errors="replace").splitlines()
    tail = lines[-tail_lines:]
    if ctx:
        await ctx.info(f"Returning {len(tail)} log lines for job {job_id}")
    return {"job_id": job_id, "log_path": state.log_path, "lines": tail, "total_lines": len(lines)}
