from __future__ import annotations

import pytest

from champollion_sulcal_mcp.job_store import JobState, list_jobs, read_job, update_job, write_job


@pytest.mark.unit
def test_write_read_roundtrip(tmp_output_dir):
    state = JobState(stage="morphologist", output_dir=str(tmp_output_dir), log_path="/tmp/test.log")
    write_job(state)
    loaded = read_job(str(tmp_output_dir), state.job_id)
    assert loaded.job_id == state.job_id
    assert loaded.stage == "morphologist"
    assert loaded.status == "pending"


@pytest.mark.unit
def test_list_jobs_sorted(tmp_output_dir):
    jobs = [
        JobState(
            stage="morphologist",
            output_dir=str(tmp_output_dir),
            log_path="/tmp/a.log",
            started_at="2024-01-01T00:00:00+00:00",
        ),
        JobState(
            stage="cortical_tiles",
            output_dir=str(tmp_output_dir),
            log_path="/tmp/b.log",
            started_at="2024-01-03T00:00:00+00:00",
        ),
        JobState(
            stage="embeddings",
            output_dir=str(tmp_output_dir),
            log_path="/tmp/c.log",
            started_at="2024-01-02T00:00:00+00:00",
        ),
    ]
    for j in jobs:
        write_job(j)
    result = list_jobs(str(tmp_output_dir))
    assert [r.stage for r in result] == ["cortical_tiles", "embeddings", "morphologist"]


@pytest.mark.unit
async def test_update_job_atomic(tmp_output_dir):
    state = JobState(stage="morphologist", output_dir=str(tmp_output_dir), log_path="/tmp/test.log")
    write_job(state)
    updated = await update_job(str(tmp_output_dir), state.job_id, status="succeeded", returncode=0)
    assert updated.status == "succeeded"
    assert updated.returncode == 0
    reread = read_job(str(tmp_output_dir), state.job_id)
    assert reread.status == "succeeded"


@pytest.mark.unit
def test_jobs_dir_created(tmp_path):
    from champollion_sulcal_mcp.job_store import jobs_dir

    out = str(tmp_path / "newout")
    d = jobs_dir(out)
    assert d.exists()
    assert d.name == ".mcp_jobs"
