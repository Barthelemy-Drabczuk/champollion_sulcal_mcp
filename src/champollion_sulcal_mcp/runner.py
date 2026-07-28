from __future__ import annotations

import asyncio
import contextlib
import os
import re
import signal
from asyncio.subprocess import PIPE, STDOUT
from datetime import UTC, datetime
from pathlib import Path

from .job_store import JobState, jobs_dir, log_path_for, release_lock, update_job, write_job

FOLD_RE = re.compile(r"[Ff]old\s+(\d+)\s*(?:/|of)\s*(\d+)")

# Track background streaming tasks and live processes
_TASKS: dict[str, asyncio.Task] = {}
_PROCS: dict[str, asyncio.subprocess.Process] = {}


async def launch(
    stage: str,
    argv: list[str],
    output_dir: str,
    cwd: str,
    env: dict,
    args_snapshot: dict,
) -> JobState:
    jobs_dir(output_dir)  # ensure directory exists

    state = JobState(
        stage=stage,
        status="running",
        output_dir=output_dir,
        log_path=str(log_path_for(output_dir, "PLACEHOLDER")),
        args=args_snapshot,
    )
    # Fix log path with actual job_id
    state = state.model_copy(update={"log_path": str(log_path_for(output_dir, state.job_id))})

    log_file = Path(state.log_path).open("ab")  # noqa: SIM115 — must stay open across async boundary

    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            env=env,
            stdout=PIPE,
            stderr=STDOUT,
        )
        state = state.model_copy(update={"pid": proc.pid})
        write_job(state)
    except Exception:
        log_file.close()
        raise

    _PROCS[state.job_id] = proc
    task = asyncio.create_task(_stream(proc, state.job_id, state.output_dir, log_file))
    _TASKS[state.job_id] = task

    return state


async def _stream(
    proc: asyncio.subprocess.Process,
    job_id: str,
    output_dir: str,
    log_file,
) -> None:
    last_fold: int | None = None
    last_progress_time = 0.0
    rc: int | None = None

    try:
        if proc.stdout is None:
            raise RuntimeError("stdout pipe unavailable")
        async for raw_line in proc.stdout:
            log_file.write(raw_line)
            log_file.flush()
            line = raw_line.decode("utf-8", errors="replace").rstrip()

            m = FOLD_RE.search(line)
            if m:
                cur, total = int(m.group(1)), int(m.group(2))
                now = asyncio.get_running_loop().time()
                if cur != last_fold or (now - last_progress_time) > 5.0:
                    last_fold = cur
                    last_progress_time = now
                    await update_job(
                        output_dir,
                        job_id,
                        progress={"fold_current": cur, "fold_total": total, "percent": round(100 * cur / total, 1)},
                    )

        await proc.wait()
        rc = proc.returncode
    finally:
        log_file.close()
        release_lock(job_id)

    # Read current status — may have been set to "cancelled"
    from .job_store import read_job

    try:
        current = read_job(output_dir, job_id)
        final_status = current.status
    except FileNotFoundError:
        final_status = "running"

    if final_status not in ("cancelled",):
        final_status = "succeeded" if rc == 0 else "failed"

    error_msg = None if rc == 0 else f"Process exited with code {rc}. See log for details."

    await update_job(
        output_dir,
        job_id,
        status=final_status,
        returncode=rc,
        ended_at=datetime.now(UTC).isoformat(),
        error=error_msg,
    )
    _TASKS.pop(job_id, None)
    _PROCS.pop(job_id, None)


async def cancel(output_dir: str, job_id: str) -> JobState:
    from .job_store import read_job

    state = read_job(output_dir, job_id)
    if state.status not in ("running", "pending"):
        return state

    # Mark cancelled first so _stream observes it
    state = await update_job(output_dir, job_id, status="cancelled")

    # Kill process
    proc = _PROCS.get(job_id)
    if proc is not None and proc.returncode is None:
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
    elif state.pid is not None:
        # Fallback: kill by PID
        with contextlib.suppress(ProcessLookupError):
            os.kill(state.pid, signal.SIGTERM)

    return read_job(output_dir, job_id)
