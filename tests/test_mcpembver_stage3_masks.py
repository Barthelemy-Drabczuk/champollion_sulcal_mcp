"""Stage-3 masks on start_config and start_pipeline.

- REQ-MCPEMBVER-BDRABCZUK-B37DC68FCB0E: start_config forwards a supplied masks
  as ``--masks``.
- REQ-MCPEMBVER-BDRABCZUK-4DF15B88BA01: start_pipeline passes a non-empty masks
  to start_config, and only then.

Reference: champollion_pipeline ``generate_champollion_config.py`` accepts
``--masks`` (mask version tag, default ``canonical_25``) and derives the crop
path ``<dataset>/derivatives/<cortical_tiles>/crops/<masks>/2mm`` from it. It
must match the value given to ``run_cortical_tiles``, which ``start_cortical_tiles``
already forwards as ``--masks``; champollion_pipeline ``main.py`` passes
``--masks`` to stage 3.

runner.launch is replaced by the ``recording_runner`` recorder (start_config) or
start_config itself by a recorder (start_pipeline), so no pipeline script runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from champollion_sulcal_mcp import job_store
from champollion_sulcal_mcp.job_store import JobState, write_job
from champollion_sulcal_mcp.tools import pipeline, stages

pytestmark = pytest.mark.unit

ALL_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]

_real_sleep = asyncio.sleep


@pytest.fixture
def crop_path(mock_roots):
    """A crops/2mm directory inside the declared MCP root."""
    _, root = mock_roots
    crops = root / "DS01" / "derivatives" / "cortical_tiles-2026" / "crops" / "2mm"
    crops.mkdir(parents=True)
    return crops


# --- REQ-MCPEMBVER-BDRABCZUK-B37DC68FCB0E: start_config ---


@pytest.mark.parametrize("masks", ["canonical_25", "canonical_corrected_26_1"])
async def test_start_config_forwards_supplied_masks(fake_pipeline_dir, mock_roots, crop_path, recording_runner, masks):
    """A caller-supplied masks reaches generate_champollion_config.py as `--masks <value>`."""
    ctx, _ = mock_roots
    await stages.start_config(crop_path=str(crop_path), dataset="DS01", masks=masks, ctx=ctx)
    argv = recording_runner[0]["argv"]
    assert "--masks" in argv, f"start_config did not forward masks: {argv}"
    assert argv[argv.index("--masks") + 1] == masks


# --- REQ-MCPEMBVER-BDRABCZUK-4DF15B88BA01: start_pipeline -> start_config ---


def _pipeline_kwargs(base: Path) -> dict:
    return {
        "input_dir": str(base / "subjects"),
        "output_dir": str(base / "pipe_out"),
        "path_to_graph": "t1mri/default_acquisition/default_analysis/folds/3.1",
        "path_sk_with_hull": "t1mri/default_acquisition/default_analysis/segmentation",
        "crop_path": str(base / "DEMO01" / "derivatives" / "cortical_tiles" / "crops" / "2mm"),
        "dataset": "DEMO01",
        "models_path": str(base / "models"),
        "datasets_root": str(base / "datasets" / "DEMO01"),
        "skip_stages": [s for s in ALL_STAGES if s != "config"],
    }


async def _config_call_via_pipeline(base: Path, monkeypatch, **options) -> dict:
    """Run start_pipeline with only the config stage active; return the kwargs start_config received.

    The real _launch_stage runs; start_config is a recorder whose child job is
    already succeeded, so the umbrella ends right after the stage.
    """
    calls: list[dict] = []
    child_dir = base / "config_child"

    async def fake_start_config(**kwargs):
        calls.append(kwargs)
        state = JobState(stage="config", status="succeeded", output_dir=str(child_dir), log_path="/tmp/fake.log")
        write_job(state)
        return {
            "job_id": state.job_id,
            "status": state.status,
            "stage": "config",
            "log_path": state.log_path,
            "output_dir": state.output_dir,
        }

    async def fast_sleep(delay, *args, **kwargs):
        await _real_sleep(0)

    monkeypatch.setattr(pipeline.stages, "start_config", fake_start_config)
    monkeypatch.setattr(pipeline.asyncio, "sleep", fast_sleep)

    kwargs = _pipeline_kwargs(base)
    result = await pipeline.start_pipeline(**kwargs, **options)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while loop.time() < deadline:
        umbrella = job_store.read_job(kwargs["output_dir"], result["job_id"])
        if umbrella.status in ("succeeded", "failed", "cancelled"):
            break
        await _real_sleep(0.01)
    assert umbrella.status == "succeeded", f"pipeline did not finish: {umbrella.status} {umbrella.error}"
    assert len(calls) == 1, f"start_config called {len(calls)} times"
    return calls[0]


@pytest.mark.parametrize("masks", ["canonical_25", "canonical_corrected_26_1"])
async def test_start_pipeline_forwards_supplied_masks_to_start_config(tmp_path, monkeypatch, masks):
    """start_pipeline(masks=<value>) calls start_config with masks=<value>."""
    call = await _config_call_via_pipeline(tmp_path, monkeypatch, masks=masks)
    assert call.get("masks") == masks, f"start_config received {call}"


@pytest.mark.parametrize("masks", [None, ""], ids=["none", "empty"])
async def test_start_pipeline_omits_masks_from_start_config_when_unset(tmp_path, monkeypatch, masks):
    """With masks unset or empty, start_config still receives crop_path and dataset only."""
    call = await _config_call_via_pipeline(tmp_path, monkeypatch, masks=masks)
    assert set(call) == {"crop_path", "dataset"}, f"start_config received {call}"
