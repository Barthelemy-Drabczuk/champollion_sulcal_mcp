from __future__ import annotations

import asyncio
import os
import textwrap

import pytest

from champollion_sulcal_mcp import job_store, runner


@pytest.mark.integration
async def test_smoke_fold_progress_and_log(tmp_output_dir):
    """
    Launch a shell script that prints fold progress lines, poll until done,
    then assert progress and log contents are correct.
    """
    script = textwrap.dedent("""\
        #!/bin/sh
        for i in 1 2 3; do
            echo "Processing Fold $i/3"
            sleep 0.05
        done
        exit 0
    """)
    script_path = tmp_output_dir / "fake_stage.sh"
    script_path.write_text(script)
    script_path.chmod(0o755)

    state = await runner.launch(
        stage="embeddings",
        argv=["/bin/sh", str(script_path)],
        output_dir=str(tmp_output_dir),
        cwd=str(tmp_output_dir),
        env=dict(os.environ),
        args_snapshot={"test": True},
    )
    assert state.status == "running"
    assert state.job_id

    # Poll until terminal
    for _ in range(60):
        await asyncio.sleep(0.1)
        current = job_store.read_job(str(tmp_output_dir), state.job_id)
        if current.status in ("succeeded", "failed", "cancelled"):
            break

    assert current.status == "succeeded", f"Expected succeeded, got {current.status}"
    assert current.returncode == 0

    # Progress was tracked
    assert current.progress.fold_total == 3
    assert current.progress.fold_current == 3
    assert current.progress.percent == 100.0

    # Log file contains fold lines
    from pathlib import Path

    log_contents = Path(current.log_path).read_text()
    assert "Fold 1/3" in log_contents
    assert "Fold 3/3" in log_contents


@pytest.mark.integration
async def test_smoke_failed_job(tmp_output_dir):
    """A failing subprocess transitions to failed with a generic error message."""
    state = await runner.launch(
        stage="morphologist",
        argv=["/bin/sh", "-c", "exit 7"],
        output_dir=str(tmp_output_dir),
        cwd=str(tmp_output_dir),
        env=dict(os.environ),
        args_snapshot={},
    )

    for _ in range(30):
        await asyncio.sleep(0.1)
        current = job_store.read_job(str(tmp_output_dir), state.job_id)
        if current.status in ("succeeded", "failed", "cancelled"):
            break

    assert current.status == "failed"
    assert current.returncode == 7
    assert current.error is not None
    assert "7" in current.error
    # No raw stderr/stack traces in user-facing error
    assert "Traceback" not in current.error


@pytest.mark.integration
async def test_smoke_cancel_running_job(tmp_output_dir):
    """Cancelling a running job transitions it to cancelled within 15 seconds."""
    state = await runner.launch(
        stage="combine",
        argv=["/bin/sh", "-c", "sleep 30"],
        output_dir=str(tmp_output_dir),
        cwd=str(tmp_output_dir),
        env=dict(os.environ),
        args_snapshot={},
    )
    assert state.status == "running"

    cancelled = await runner.cancel(str(tmp_output_dir), state.job_id)

    for _ in range(150):
        await asyncio.sleep(0.1)
        current = job_store.read_job(str(tmp_output_dir), state.job_id)
        if current.status in ("succeeded", "failed", "cancelled"):
            break

    assert current.status == "cancelled"


@pytest.mark.integration
async def test_smoke_get_job_log_via_tool(tmp_output_dir):
    """End-to-end: launch → succeed → get_job_log returns lines."""
    state = await runner.launch(
        stage="snapshots",
        argv=["/bin/sh", "-c", "for i in 1 2 3 4 5; do echo line$i; done; exit 0"],
        output_dir=str(tmp_output_dir),
        cwd=str(tmp_output_dir),
        env=dict(os.environ),
        args_snapshot={},
    )

    for _ in range(30):
        await asyncio.sleep(0.1)
        current = job_store.read_job(str(tmp_output_dir), state.job_id)
        if current.status in ("succeeded", "failed", "cancelled"):
            break

    assert current.status == "succeeded"

    from champollion_sulcal_mcp.tools.jobs import get_job_log

    result = await get_job_log(str(tmp_output_dir), state.job_id, tail_lines=3)
    assert len(result["lines"]) == 3
    assert result["lines"][-1] == "line5"
    assert result["total_lines"] == 5
