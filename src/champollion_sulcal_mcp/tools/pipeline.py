from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from fastmcp import Context
from fastmcp.exceptions import ToolError

from .. import job_store, runner
from ..job_store import JobProgress, JobState, write_job
from . import stages

logger = logging.getLogger(__name__)

DERIVATIVES_SUBDIR = "derivatives"
CHAMPOLLION_SUBDIR = "champollion_V1"
EMBEDDINGS_SUBDIR = "embeddings"

STAGE_ORDER = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]


def _compute_combined_embeddings_dir(dataset_root: str) -> str:
    """Return the directory start_pipeline's combine stage writes to and snapshots reads from.

    Args:
        dataset_root: start_pipeline's output_dir (dataset root R).

    Returns:
        str(Path(R) / "derivatives" / "champollion_V1" / "embeddings").

    Complexity: O(1).
    """
    return str(Path(dataset_root) / DERIVATIVES_SUBDIR / CHAMPOLLION_SUBDIR / EMBEDDINGS_SUBDIR)


async def start_pipeline(
    input_dir: str,
    output_dir: str,
    path_to_graph: str,
    path_sk_with_hull: str,
    crop_path: str,
    dataset: str,
    models_path: str,
    datasets_root: str,
    sk_qc_path: str | None = None,
    njobs: int | None = None,
    parallel: bool = False,
    cpu: bool = False,
    skip_stages: list[str] | None = None,
    labelling_session: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Launch the full Champollion pipeline (all 6 stages sequentially). Returns a pipeline job_id immediately."""
    skip = set(skip_stages or [])
    active_stages = [s for s in STAGE_ORDER if s not in skip]

    # Create umbrella job
    umbrella = JobState(
        stage="pipeline",
        status="running",
        output_dir=output_dir,
        log_path=str(job_store.log_path_for(output_dir, "PLACEHOLDER")),
        args={"input_dir": input_dir, "output_dir": output_dir, "skip_stages": list(skip)},
        progress=JobProgress(
            stages_done=0,
            stages_total=len(active_stages),
            current_stage=active_stages[0] if active_stages else None,
        ),
    )
    umbrella = umbrella.model_copy(update={"log_path": str(job_store.log_path_for(output_dir, umbrella.job_id))})
    job_store.jobs_dir(output_dir)
    write_job(umbrella)

    if ctx:
        await ctx.info(f"Launching pipeline: {len(active_stages)} stages, job_id={umbrella.job_id}")

    task = asyncio.create_task(
        _run_pipeline(
            umbrella_job_id=umbrella.job_id,
            output_dir=output_dir,
            active_stages=active_stages,
            input_dir=input_dir,
            path_to_graph=path_to_graph,
            path_sk_with_hull=path_sk_with_hull,
            crop_path=crop_path,
            dataset=dataset,
            models_path=models_path,
            datasets_root=datasets_root,
            sk_qc_path=sk_qc_path,
            njobs=njobs,
            parallel=parallel,
            cpu=cpu,
            labelling_session=labelling_session,
        )
    )
    task.add_done_callback(_handle_task_exception)

    return {"job_id": umbrella.job_id, "status": "running", "stage": "pipeline", "active_stages": active_stages}


def _handle_task_exception(task: asyncio.Task) -> None:
    """Log and suppress exceptions from the background pipeline task."""
    if not task.cancelled() and task.exception() is not None:
        logger.error("Unhandled exception in pipeline task: %s", task.exception(), exc_info=task.exception())


async def _run_pipeline(
    umbrella_job_id: str,
    output_dir: str,
    active_stages: list[str],
    **kwargs,
) -> None:
    try:
        await _execute_pipeline(umbrella_job_id, output_dir, active_stages, **kwargs)
    except Exception as exc:
        logger.exception("Pipeline task crashed unexpectedly")
        import contextlib

        with contextlib.suppress(Exception):
            await job_store.update_job(
                output_dir,
                umbrella_job_id,
                status="failed",
                error=f"Pipeline crashed: {exc}",
                ended_at=datetime.now(UTC).isoformat(),
            )


async def _execute_pipeline(
    umbrella_job_id: str,
    output_dir: str,
    active_stages: list[str],
    **kwargs,
) -> None:
    for i, stage_name in enumerate(active_stages):
        # Check if umbrella was cancelled
        try:
            umbrella = job_store.read_job(output_dir, umbrella_job_id)
        except FileNotFoundError:
            return
        if umbrella.status == "cancelled":
            return

        await job_store.update_job(
            output_dir,
            umbrella_job_id,
            progress={"current_stage": stage_name, "stages_done": i, "stages_total": len(active_stages)},
        )

        try:
            result = await _launch_stage(stage_name, umbrella_output_dir=output_dir, **kwargs)
        except ToolError as e:
            await job_store.update_job(
                output_dir,
                umbrella_job_id,
                status="failed",
                error=f"Stage {stage_name!r} failed to launch: {e}",
                ended_at=datetime.now(UTC).isoformat(),
            )
            return

        child_job_id = result["job_id"]
        child_output_dir = result["output_dir"]  # use child's actual output_dir for polling

        await job_store.update_job(
            output_dir,
            umbrella_job_id,
            progress={
                "current_stage": stage_name,
                "stages_done": i,
                "stages_total": len(active_stages),
                "current_child_job_id": child_job_id,
            },
        )

        # Wait for child job to complete
        while True:
            await asyncio.sleep(2.0)
            try:
                child = job_store.read_job(child_output_dir, child_job_id)
            except FileNotFoundError:
                # Job file missing — treat as failed
                await job_store.update_job(
                    output_dir,
                    umbrella_job_id,
                    status="failed",
                    error=f"Stage {stage_name!r} job file disappeared unexpectedly.",
                    ended_at=datetime.now(UTC).isoformat(),
                )
                return
            if child.status in ("succeeded", "failed", "cancelled"):
                break
            # Propagate cancellation
            try:
                umbrella = job_store.read_job(output_dir, umbrella_job_id)
            except FileNotFoundError:
                return
            if umbrella.status == "cancelled":
                await runner.cancel(child_output_dir, child_job_id)
                return

        if child.status != "succeeded":
            await job_store.update_job(
                output_dir,
                umbrella_job_id,
                status="failed",
                error=f"Stage {stage_name!r} failed (child job {child_job_id}). Status: {child.status}.",
                ended_at=datetime.now(UTC).isoformat(),
            )
            return

    await job_store.update_job(
        output_dir,
        umbrella_job_id,
        status="succeeded",
        ended_at=datetime.now(UTC).isoformat(),
        progress={"stages_done": len(active_stages), "stages_total": len(active_stages)},
    )


async def _launch_stage(stage_name: str, umbrella_output_dir: str, **kwargs) -> dict:
    if stage_name == "morphologist":
        return await stages.start_morphologist(
            input_dir=kwargs["input_dir"],
            output_dir=umbrella_output_dir,
            parallel=kwargs.get("parallel", False),
        )
    elif stage_name == "cortical_tiles":
        return await stages.start_cortical_tiles(
            input_dir=kwargs["input_dir"],
            output_dir=str(Path(umbrella_output_dir) / DERIVATIVES_SUBDIR),
            path_to_graph=kwargs["path_to_graph"],
            path_sk_with_hull=kwargs["path_sk_with_hull"],
            sk_qc_path=kwargs.get("sk_qc_path"),
            njobs=kwargs.get("njobs"),
            labelling_session=kwargs.get("labelling_session"),
        )
    elif stage_name == "config":
        return await stages.start_config(
            crop_path=kwargs["crop_path"],
            dataset=kwargs["dataset"],
        )
    elif stage_name == "embeddings":
        return await stages.start_embeddings(
            models_path=kwargs["models_path"],
            datasets_root=kwargs["datasets_root"],
            cpu=kwargs.get("cpu", False),
        )
    elif stage_name == "combine":
        datasets_root = kwargs["datasets_root"]
        embeddings_source = str(
            Path(datasets_root).parent / (Path(datasets_root).name + "embeddings")
        )
        return await stages.start_combine(
            embeddings_source=embeddings_source,
            output_path=_compute_combined_embeddings_dir(umbrella_output_dir),
        )
    elif stage_name == "snapshots":
        return await stages.start_snapshots(
            output_dir=umbrella_output_dir,
            morphologist_dir=kwargs.get("input_dir"),
            embeddings_dir=_compute_combined_embeddings_dir(umbrella_output_dir),
        )
    else:
        raise ToolError(f"Unknown stage: {stage_name!r}")
