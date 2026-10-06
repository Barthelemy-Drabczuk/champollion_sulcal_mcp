"""Characterization tests for tools/pipeline.py orchestration (TASK-042, REQ-MCP-COV-26..34).

start_pipeline is driven end to end with `_launch_stage` replaced by a fake that
persists a child job per stage; the 2 s child-poll sleep is shortened.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp import job_store
from champollion_sulcal_mcp.job_store import JobState, write_job
from champollion_sulcal_mcp.tools import pipeline

pytestmark = pytest.mark.unit

ALL_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]
TERMINAL = ("succeeded", "failed", "cancelled")

_real_sleep = asyncio.sleep


def _pipeline_kwargs(output_dir: Path) -> dict:
    return {
        "input_dir": str(output_dir / "subjects"),
        "output_dir": str(output_dir),
        "path_to_graph": "t1mri/default_acquisition/default_analysis/folds/3.1",
        "path_sk_with_hull": "t1mri/default_acquisition/default_analysis/segmentation",
        "crop_path": str(output_dir / "crops"),
        "dataset": "DEMO01",
        "models_path": str(output_dir / "models"),
        "datasets_root": str(output_dir / "datasets"),
    }


class FakeStages:
    """Stand-in for pipeline._launch_stage: persists one child job per launched stage.

    `outcome[stage]` is the child's status at launch ("succeeded" by default),
    "missing" to launch without persisting a child job file, or an exception
    instance to raise at launch.
    """

    def __init__(self, child_dir: Path):
        self.child_dir = child_dir
        self.outcome: dict[str, object] = {}
        self.launched: list[str] = []
        self.children: dict[str, str] = {}

    async def __call__(self, stage_name: str, umbrella_output_dir: str, **kwargs) -> dict:
        outcome = self.outcome.get(stage_name, "succeeded")
        if isinstance(outcome, BaseException):
            raise outcome
        self.launched.append(stage_name)
        child = JobState(stage=stage_name, status="running", output_dir=str(self.child_dir), log_path="/tmp/x.log")
        if outcome != "missing":
            write_job(child.model_copy(update={"status": outcome}))
        self.children[stage_name] = child.job_id
        return {"job_id": child.job_id, "status": "running", "stage": stage_name, "output_dir": str(self.child_dir)}


@pytest.fixture
def fake_stages(tmp_path, monkeypatch):
    fake = FakeStages(tmp_path / "children")
    monkeypatch.setattr(pipeline, "_launch_stage", fake)
    return fake


@pytest.fixture
def poll_hooks(monkeypatch):
    """Shorten the child-poll sleep; each hook in the list runs once per poll."""
    hooks: list = []

    async def fast_sleep(delay, *args, **kwargs):
        for hook in hooks:
            hook()
        await _real_sleep(0)

    monkeypatch.setattr(pipeline.asyncio, "sleep", fast_sleep)
    return hooks


async def _wait_terminal(output_dir: Path, job_id: str, timeout: float = 5.0) -> JobState:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        state = job_store.read_job(str(output_dir), job_id)
        if state.status in TERMINAL or loop.time() > deadline:
            return state
        await _real_sleep(0.01)


async def _settle(rounds: int = 50) -> None:
    for _ in range(rounds):
        await _real_sleep(0)


# --- REQ-MCP-COV-26: start_pipeline return value and umbrella job ---


class TestStartPipelineReturn:
    async def test_return_value_and_umbrella_job_file(self, tmp_path, fake_stages, poll_hooks):
        """REQ-MCP-COV-26: running/pipeline/active_stages, and job_id names the umbrella job file."""
        out = tmp_path / "ds"
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out), skip_stages=["config", "snapshots"])
        assert result["status"] == "running"
        assert result["stage"] == "pipeline"
        assert result["active_stages"] == ["morphologist", "cortical_tiles", "embeddings", "combine"]
        assert (out / ".mcp_jobs" / f"{result['job_id']}.json").is_file()
        umbrella = job_store.read_job(str(out), result["job_id"])
        assert umbrella.stage == "pipeline"
        assert umbrella.progress.stages_total == 4
        await _wait_terminal(out, result["job_id"])


# --- REQ-MCP-COV-27 / -28: ordering and success ---


class TestStageOrdering:
    async def test_stages_launch_in_order(self, tmp_path, fake_stages, poll_hooks):
        """REQ-MCP-COV-27: stages launch in pipeline order."""
        out = tmp_path / "ds"
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out))
        await _wait_terminal(out, result["job_id"])
        assert fake_stages.launched == ALL_STAGES

    async def test_next_stage_waits_for_previous_child_success(self, tmp_path, fake_stages, poll_hooks):
        """REQ-MCP-COV-27: no stage launches while the previous stage's child job still runs."""
        for stage in ALL_STAGES:
            fake_stages.outcome[stage] = "running"
        launched_when_finished: list[list[str]] = []

        def finish_running_child():
            for job_id in fake_stages.children.values():
                child = job_store.read_job(str(fake_stages.child_dir), job_id)
                if child.status == "running":
                    launched_when_finished.append(list(fake_stages.launched))
                    write_job(child.model_copy(update={"status": "succeeded"}))

        poll_hooks.append(finish_running_child)
        out = tmp_path / "ds"
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out))
        await _wait_terminal(out, result["job_id"])
        assert launched_when_finished == [ALL_STAGES[: i + 1] for i in range(len(ALL_STAGES))]


class TestUmbrellaOutcome:
    async def test_all_succeed_sets_succeeded(self, tmp_path, fake_stages, poll_hooks):
        """REQ-MCP-COV-28: umbrella ends succeeded with stages_done == stages_total."""
        out = tmp_path / "ds"
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out), skip_stages=["morphologist"])
        umbrella = await _wait_terminal(out, result["job_id"])
        assert umbrella.status == "succeeded"
        assert umbrella.progress.stages_done == umbrella.progress.stages_total == 5

    # --- REQ-MCP-COV-29 ---

    @pytest.mark.parametrize("child_status", ["failed", "cancelled"])
    async def test_child_failure_fails_umbrella(self, tmp_path, fake_stages, poll_hooks, child_status):
        """REQ-MCP-COV-29: a failed/cancelled child fails the umbrella, naming the stage; no later stage."""
        fake_stages.outcome["config"] = child_status
        out = tmp_path / "ds"
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out))
        umbrella = await _wait_terminal(out, result["job_id"])
        assert umbrella.status == "failed"
        assert "config" in umbrella.error
        assert fake_stages.launched == ["morphologist", "cortical_tiles", "config"]

    # --- REQ-MCP-COV-30 ---

    async def test_launch_tool_error_fails_umbrella(self, tmp_path, fake_stages, poll_hooks):
        """REQ-MCP-COV-30: a ToolError at launch fails the umbrella with "failed to launch" + stage."""
        fake_stages.outcome["embeddings"] = ToolError("models_path must be an absolute path")
        out = tmp_path / "ds"
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out))
        umbrella = await _wait_terminal(out, result["job_id"])
        assert umbrella.status == "failed"
        assert "failed to launch" in umbrella.error
        assert "embeddings" in umbrella.error
        assert "combine" not in fake_stages.launched

    # --- REQ-MCP-COV-31 ---

    async def test_umbrella_cancel_cancels_child(self, tmp_path, fake_stages, poll_hooks, monkeypatch):
        """REQ-MCP-COV-31: cancelling the umbrella cancels the running child; no later stage."""
        fake_stages.outcome["morphologist"] = "running"
        cancelled: list[tuple[str, str]] = []

        async def fake_cancel(output_dir, job_id):
            cancelled.append((output_dir, job_id))

        monkeypatch.setattr(pipeline.runner, "cancel", fake_cancel)
        out = tmp_path / "ds"
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out))

        def cancel_umbrella():
            umbrella = job_store.read_job(str(out), result["job_id"])
            write_job(umbrella.model_copy(update={"status": "cancelled"}))

        poll_hooks.append(cancel_umbrella)
        await _settle()
        assert cancelled == [(str(fake_stages.child_dir), fake_stages.children["morphologist"])]
        assert fake_stages.launched == ["morphologist"]

    # --- REQ-MCP-COV-32 ---

    async def test_child_job_file_disappears(self, tmp_path, fake_stages, poll_hooks):
        """REQ-MCP-COV-32: a vanished child job file fails the umbrella ("disappeared unexpectedly")."""
        fake_stages.outcome["morphologist"] = "missing"
        out = tmp_path / "ds"
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out))
        umbrella = await _wait_terminal(out, result["job_id"])
        assert umbrella.status == "failed"
        assert "disappeared unexpectedly" in umbrella.error

    # --- REQ-MCP-COV-33 ---

    async def test_unexpected_exception_marks_crashed(self, tmp_path, fake_stages, poll_hooks):
        """REQ-MCP-COV-33: an unexpected exception fails the umbrella with "Pipeline crashed:"."""
        fake_stages.outcome["cortical_tiles"] = RuntimeError("boom")
        out = tmp_path / "ds"
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out))
        umbrella = await _wait_terminal(out, result["job_id"])
        assert umbrella.status == "failed"
        assert umbrella.error.startswith("Pipeline crashed:")


# --- REQ-MCP-COV-34: config stage wiring ---


class TestLaunchStageWiring:
    async def test_config_stage_passes_only_crop_path_and_dataset(self, monkeypatch):
        """REQ-MCP-COV-34: start_config receives crop_path and dataset only (pipeline defaults apply)."""
        captured: dict = {}

        async def fake_start_config(**kwargs):
            captured.update(kwargs)
            return {"job_id": "j", "status": "running", "stage": "config", "output_dir": "/abs"}

        monkeypatch.setattr(pipeline.stages, "start_config", fake_start_config)
        await pipeline._launch_stage(
            "config",
            umbrella_output_dir="/abs/data/DEMO01",
            crop_path="/abs/data/DEMO01/derivatives/cortical_tiles/crops/2mm",
            dataset="DEMO01",
            models_path="/abs/models",
            datasets_root="/abs/data",
        )
        assert captured == {
            "crop_path": "/abs/data/DEMO01/derivatives/cortical_tiles/crops/2mm",
            "dataset": "DEMO01",
        }
