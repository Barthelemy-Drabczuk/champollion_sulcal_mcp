from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from champollion_sulcal_mcp import runner
from champollion_sulcal_mcp.job_store import JobState, write_job


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
def mock_roots(tmp_path, monkeypatch):
    """Fixture: ctx with list_roots() returning a single tmp_path root."""
    root = tmp_path / "data_root"
    root.mkdir()

    def _make_root_obj(path: Path):
        obj = MagicMock()
        obj.uri = f"file://{path}"
        return obj

    ctx = MagicMock()
    ctx.list_roots = AsyncMock(return_value=[_make_root_obj(root)])
    ctx.info = AsyncMock()
    return ctx, root


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


# Every script tools/stages.py launches (TASK-042 characterization tests).
_SCRIPTS = [
    "generate_morphologist_graphs.py",
    "run_cortical_tiles.py",
    "generate_champollion_config.py",
    "generate_embeddings.py",
    "put_together_embeddings.py",
    "generate_snapshots.py",
    "run_streaming.py",
    "train_champollion.py",
    "purge_subject.py",
    "prune_failed_subjects.py",
]


@pytest.fixture
def pipeline_dir(tmp_path, monkeypatch):
    """A fake champollion_pipeline with a stub for every script tools/stages.py launches."""
    pipeline = tmp_path / "champollion_pipeline"
    scripts = pipeline / "src" / "champollion_pipeline"
    scripts.mkdir(parents=True)
    for name in _SCRIPTS:
        (scripts / name).write_text("# stub\n")
    monkeypatch.setenv("CHAMPOLLION_PIPELINE_DIR", str(pipeline))
    return pipeline


@pytest.fixture
def launches(monkeypatch):
    """Replace runner.launch with a recorder capturing stage, argv, output_dir and env."""
    calls: list[dict] = []

    async def fake_launch(stage, argv, output_dir, cwd, env, args_snapshot):
        state = JobState(stage=stage, status="running", output_dir=output_dir, log_path="/tmp/fake.log")
        write_job(state)
        calls.append({"stage": stage, "argv": list(argv), "output_dir": output_dir, "env": dict(env)})
        return state

    monkeypatch.setattr(runner, "launch", fake_launch)
    return calls
