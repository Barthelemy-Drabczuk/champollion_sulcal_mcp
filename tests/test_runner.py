from __future__ import annotations

import asyncio

import pytest

from champollion_sulcal_mcp import runner
from champollion_sulcal_mcp.runner import FOLD_RE


@pytest.mark.unit
@pytest.mark.parametrize("line", [
    "Fold 3/56",
    "fold 12 of 56",
    "Processing fold 1/56",
    "fold 28 of 56 done",
])
def test_fold_regex_matches(line):
    m = FOLD_RE.search(line)
    assert m is not None
    assert int(m.group(1)) >= 1
    assert int(m.group(2)) >= 1


@pytest.mark.integration
async def test_launch_echo_success(tmp_output_dir):
    state = await runner.launch(
        stage="test",
        argv=["/bin/sh", "-c", "echo hello world; exit 0"],
        output_dir=str(tmp_output_dir),
        cwd="/tmp",
        env={"PATH": "/bin:/usr/bin"},
        args_snapshot={},
    )
    assert state.status == "running"
    # Wait for completion
    for _ in range(30):
        await asyncio.sleep(0.2)
        from champollion_sulcal_mcp.job_store import read_job
        s = read_job(str(tmp_output_dir), state.job_id)
        if s.status in ("succeeded", "failed"):
            break
    assert s.status == "succeeded"
    assert s.returncode == 0
    from pathlib import Path
    log = Path(s.log_path).read_text()
    assert "hello world" in log


@pytest.mark.integration
async def test_launch_failure_sets_failed(tmp_output_dir):
    state = await runner.launch(
        stage="test",
        argv=["/bin/sh", "-c", "exit 7"],
        output_dir=str(tmp_output_dir),
        cwd="/tmp",
        env={"PATH": "/bin:/usr/bin"},
        args_snapshot={},
    )
    for _ in range(20):
        await asyncio.sleep(0.2)
        from champollion_sulcal_mcp.job_store import read_job
        s = read_job(str(tmp_output_dir), state.job_id)
        if s.status in ("succeeded", "failed", "cancelled"):
            break
    assert s.status == "failed"
    assert s.returncode == 7
    assert "7" in s.error
    # Error message must not leak raw stderr
    assert "Traceback" not in (s.error or "")


@pytest.mark.integration
async def test_cancel_terminates(tmp_output_dir):
    state = await runner.launch(
        stage="test",
        argv=["/bin/sh", "-c", "sleep 30"],
        output_dir=str(tmp_output_dir),
        cwd="/tmp",
        env={"PATH": "/bin:/usr/bin"},
        args_snapshot={},
    )
    await asyncio.sleep(0.3)
    await runner.cancel(str(tmp_output_dir), state.job_id)
    # Wait for finalization
    for _ in range(25):
        await asyncio.sleep(0.5)
        from champollion_sulcal_mcp.job_store import read_job
        s = read_job(str(tmp_output_dir), state.job_id)
        if s.ended_at is not None:
            break
    assert s.status == "cancelled"


@pytest.mark.integration
async def test_progress_updates(tmp_output_dir):
    state = await runner.launch(
        stage="test",
        argv=["/bin/sh", "-c", "for i in 1 2 3; do echo Fold $i/3; sleep 0.05; done"],
        output_dir=str(tmp_output_dir),
        cwd="/tmp",
        env={"PATH": "/bin:/usr/bin"},
        args_snapshot={},
    )
    for _ in range(30):
        await asyncio.sleep(0.2)
        from champollion_sulcal_mcp.job_store import read_job
        s = read_job(str(tmp_output_dir), state.job_id)
        if s.status == "succeeded":
            break
    assert s.progress.fold_current == 3
    assert s.progress.fold_total == 3
