from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JobProgress(BaseModel):
    fold_current: int | None = None
    fold_total: int | None = None
    percent: float | None = None
    current_stage: str | None = None
    stages_done: int | None = None
    stages_total: int | None = None
    current_child_job_id: str | None = None


class JobState(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage: str
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"] = "pending"
    pid: int | None = None
    output_dir: str
    log_path: str
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    ended_at: str | None = None
    returncode: int | None = None
    progress: JobProgress = Field(default_factory=JobProgress)
    args: dict = Field(default_factory=dict)
    error: str | None = None


# Per-job asyncio locks to prevent concurrent write races
_JOB_LOCKS: dict[str, asyncio.Lock] = {}
_LOCKS_LOCK = asyncio.Lock()


async def _get_lock(job_id: str) -> asyncio.Lock:
    async with _LOCKS_LOCK:
        if job_id not in _JOB_LOCKS:
            _JOB_LOCKS[job_id] = asyncio.Lock()
        return _JOB_LOCKS[job_id]


def release_lock(job_id: str) -> None:
    """Remove the per-job lock once the job is terminal. Called from runner._stream finally block."""
    _JOB_LOCKS.pop(job_id, None)


def jobs_dir(output_dir: str) -> Path:
    d = Path(output_dir) / ".mcp_jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_path(output_dir: str, job_id: str) -> Path:
    return jobs_dir(output_dir) / f"{job_id}.json"


def log_path_for(output_dir: str, job_id: str) -> Path:
    return jobs_dir(output_dir) / f"{job_id}.log"


def write_job(state: JobState) -> None:
    path = job_path(state.output_dir, state.job_id)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(state.model_dump_json(indent=2))
    os.replace(tmp, path)


def read_job(output_dir: str, job_id: str) -> JobState:
    path = job_path(output_dir, job_id)
    if not path.exists():
        raise FileNotFoundError(f"Job {job_id} not found in {output_dir}")
    return JobState.model_validate_json(path.read_text())


def list_jobs(output_dir: str) -> list[JobState]:
    d = jobs_dir(output_dir)
    jobs = []
    for p in d.glob("*.json"):
        try:
            jobs.append(JobState.model_validate_json(p.read_text()))
        except Exception as exc:
            logger.warning("Skipping corrupt job file %s: %s", p, exc)
            continue
    return sorted(jobs, key=lambda j: j.started_at, reverse=True)


async def update_job(output_dir: str, job_id: str, **fields) -> JobState:
    lock = await _get_lock(job_id)
    async with lock:
        state = read_job(output_dir, job_id)
        # Handle nested progress updates
        if "progress" in fields and isinstance(fields["progress"], dict):
            current = state.progress.model_dump()
            current.update(fields.pop("progress"))
            fields["progress"] = JobProgress(**current)
        updated = state.model_copy(update=fields)
        write_job(updated)
        return updated
