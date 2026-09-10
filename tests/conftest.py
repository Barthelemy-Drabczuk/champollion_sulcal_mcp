from __future__ import annotations

import pytest


@pytest.fixture
def tmp_output_dir(tmp_path):
    d = tmp_path / "out"
    d.mkdir()
    return d


@pytest.fixture
def fake_pipeline_dir(tmp_path, monkeypatch):
    pipeline = tmp_path / "champollion_pipeline"
    src = pipeline / "src" / "champollion_pipeline"
    src.mkdir(parents=True)
    # Create stub scripts
    for name in [
        "generate_morphologist_graphs.py",
        "run_cortical_tiles.py",
        "generate_champollion_config.py",
        "generate_embeddings.py",
        "put_together_embeddings.py",
        "generate_snapshots.py",
    ]:
        (src / name).write_text("# stub\n")
    (pipeline / "external" / "champollion_V1").mkdir(parents=True)
    (pipeline / "external" / "cortical_tiles").mkdir(parents=True)
    monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", str(pipeline))
    return pipeline


@pytest.fixture
def recording_runner(monkeypatch):
    """Replace runner.launch with a stub that records calls and returns a dummy JobState."""
    calls = []

    async def fake_launch(stage, argv, output_dir, cwd, env, args_snapshot):
        from champollion_sulcal_mcp.job_store import JobState
        state = JobState(stage=stage, status="running", output_dir=output_dir, log_path="/tmp/fake.log")
        calls.append({"stage": stage, "argv": argv, "output_dir": output_dir})
        from champollion_sulcal_mcp.job_store import write_job
        write_job(state)
        return state

    from champollion_sulcal_mcp import runner
    monkeypatch.setattr(runner, "launch", fake_launch)
    return calls
