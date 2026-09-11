from __future__ import annotations

from pathlib import Path

import pytest

from champollion_sulcal_mcp.job_store import JobState, write_job
from champollion_sulcal_mcp.tools.jobs import cancel_job, get_job_log, get_job_status, list_jobs


@pytest.mark.unit
async def test_get_job_status_unknown_raises_toolerror(tmp_output_dir):
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="Unknown job_id"):
        await get_job_status(str(tmp_output_dir), "nonexistent-id")


@pytest.mark.unit
async def test_get_job_status_returns_state(tmp_output_dir):
    state = JobState(stage="embeddings", status="running", output_dir=str(tmp_output_dir), log_path="/tmp/fake.log")
    write_job(state)
    result = await get_job_status(str(tmp_output_dir), state.job_id)
    assert result["job_id"] == state.job_id
    assert result["status"] == "running"
    assert result["stage"] == "embeddings"


@pytest.mark.unit
async def test_list_jobs_filter_by_status(tmp_output_dir):
    s1 = JobState(stage="morphologist", status="succeeded", output_dir=str(tmp_output_dir), log_path="/tmp/a.log")
    s2 = JobState(stage="embeddings", status="failed", output_dir=str(tmp_output_dir), log_path="/tmp/b.log")
    s3 = JobState(stage="combine", status="succeeded", output_dir=str(tmp_output_dir), log_path="/tmp/c.log")
    for s in [s1, s2, s3]:
        write_job(s)

    all_jobs = await list_jobs(str(tmp_output_dir))
    assert len(all_jobs) == 3

    succeeded = await list_jobs(str(tmp_output_dir), status="succeeded")
    assert len(succeeded) == 2
    assert all(j["status"] == "succeeded" for j in succeeded)

    failed = await list_jobs(str(tmp_output_dir), status="failed")
    assert len(failed) == 1
    assert failed[0]["stage"] == "embeddings"


@pytest.mark.unit
async def test_list_jobs_limit(tmp_output_dir):
    for i in range(5):
        s = JobState(stage="morphologist", status="succeeded", output_dir=str(tmp_output_dir), log_path=f"/tmp/{i}.log")
        write_job(s)

    result = await list_jobs(str(tmp_output_dir), limit=3)
    assert len(result) == 3


@pytest.mark.unit
async def test_get_job_log_tail(tmp_output_dir):
    state = JobState(stage="embeddings", status="succeeded", output_dir=str(tmp_output_dir), log_path="")
    log_file = Path(tmp_output_dir) / ".mcp_jobs" / f"{state.job_id}.log"
    state = state.model_copy(update={"log_path": str(log_file)})
    write_job(state)

    lines = [f"line {i}\n" for i in range(500)]
    log_file.write_text("".join(lines))

    result = await get_job_log(str(tmp_output_dir), state.job_id, tail_lines=10)
    assert result["job_id"] == state.job_id
    assert len(result["lines"]) == 10
    assert result["lines"][-1] == "line 499"
    assert result["total_lines"] == 500


@pytest.mark.unit
async def test_get_job_log_cap_at_5000(tmp_output_dir):
    state = JobState(stage="combine", status="succeeded", output_dir=str(tmp_output_dir), log_path="")
    log_file = Path(tmp_output_dir) / ".mcp_jobs" / f"{state.job_id}.log"
    state = state.model_copy(update={"log_path": str(log_file)})
    write_job(state)
    log_file.write_text("\n".join(f"x{i}" for i in range(100)))

    result = await get_job_log(str(tmp_output_dir), state.job_id, tail_lines=9999)
    assert len(result["lines"]) <= 5000


@pytest.mark.unit
async def test_get_job_log_missing_log_raises_toolerror(tmp_output_dir):
    from fastmcp.exceptions import ToolError

    state = JobState(stage="config", status="succeeded", output_dir=str(tmp_output_dir), log_path="/nonexistent/path.log")
    write_job(state)

    with pytest.raises(ToolError, match="not available"):
        await get_job_log(str(tmp_output_dir), state.job_id)


@pytest.mark.unit
async def test_get_job_log_unknown_job_raises_toolerror(tmp_output_dir):
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="Unknown job_id"):
        await get_job_log(str(tmp_output_dir), "no-such-job")


@pytest.mark.unit
async def test_cancel_terminal_job_noop(tmp_output_dir):
    state = JobState(stage="combine", status="succeeded", output_dir=str(tmp_output_dir), log_path="/tmp/noop.log")
    write_job(state)

    result = await cancel_job(str(tmp_output_dir), state.job_id)
    assert result["status"] == "succeeded"
    assert result["job_id"] == state.job_id


@pytest.mark.unit
async def test_cancel_unknown_job_raises_toolerror(tmp_output_dir):
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="Unknown job_id"):
        await cancel_job(str(tmp_output_dir), "ghost-job")
