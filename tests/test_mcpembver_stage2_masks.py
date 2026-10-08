"""start_pipeline forwards masks to stage 2 (REQ-MCPEMBVER-BDRABCZUK-44117031E8D9).

Requirement: start_pipeline shall launch the cortical_tiles stage with --masks
followed by the received masks value exactly when start_pipeline receives a
non-empty masks value.

Stage 2 crops must land under the same mask version stage 4 reads, as
champollion_pipeline's main.py does by passing --masks to both stages. An unset
or empty masks leaves run_cortical_tiles.py on its own default.

runner.launch is replaced by the `launches` recorder (tests/conftest.py), so no
pipeline script runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from champollion_sulcal_mcp import job_store
from champollion_sulcal_mcp.tools import pipeline

pytestmark = pytest.mark.unit

ALL_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]
MASKS = "canonical_corrected_26_1"

_real_sleep = asyncio.sleep


def _contains_run(argv: list[str], tokens: list[str]) -> bool:
    """True when `tokens` appear in `argv` as one contiguous run."""
    n = len(tokens)
    return any(argv[i : i + n] == tokens for i in range(len(argv) - n + 1))


@pytest.fixture
def recording_cortical_tiles(monkeypatch):
    """Replace start_cortical_tiles with a recorder of the kwargs _launch_stage passes."""
    calls: list[dict] = []

    async def _fn(**kwargs):
        calls.append(kwargs)
        return {"stage": "cortical_tiles", "status": "running", "output_dir": kwargs.get("output_dir", "")}

    monkeypatch.setattr(pipeline.stages, "start_cortical_tiles", _fn)
    return calls


def _pipeline_kwargs(base: Path) -> dict:
    """start_pipeline arguments with only the cortical_tiles stage active."""
    return {
        "input_dir": str(base / "DEMO01" / "derivatives" / "morphologist-6.0" / "subjects"),
        "output_dir": str(base / "DEMO01"),
        "path_to_graph": "t1mri/default_acquisition/default_analysis/folds/3.1",
        "path_sk_with_hull": "t1mri/default_acquisition/default_analysis/segmentation",
        "crop_path": str(base / "crops"),
        "dataset": "DEMO01",
        "models_path": str(base / "models"),
        "datasets_root": str(base / "datasets" / "DEMO01"),
        "skip_stages": [s for s in ALL_STAGES if s != "cortical_tiles"],
    }


async def _cortical_tiles_argv_via_pipeline(base: Path, launches: list, monkeypatch, **options) -> list[str]:
    """Run start_pipeline with only cortical_tiles active; return that stage's argv.

    The real _launch_stage and start_cortical_tiles run; runner.launch is the recorder.
    Once the stage is launched its child job is marked succeeded so the umbrella ends.
    """

    async def fast_sleep(delay, *args, **kwargs):
        await _real_sleep(0)

    monkeypatch.setattr(pipeline.asyncio, "sleep", fast_sleep)

    kwargs = _pipeline_kwargs(base)
    result = await pipeline.start_pipeline(**kwargs, **options)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while not launches and loop.time() < deadline:
        await _real_sleep(0.01)
    assert launches, "cortical_tiles stage was never launched"
    assert launches[0]["stage"] == "cortical_tiles", launches[0]

    child_dir = launches[0]["output_dir"]
    for job_file in (Path(child_dir) / ".mcp_jobs").glob("*.json"):
        await job_store.update_job(child_dir, job_file.stem, status="succeeded")

    while loop.time() < deadline:
        if job_store.read_job(kwargs["output_dir"], result["job_id"]).status in ("succeeded", "failed", "cancelled"):
            break
        await _real_sleep(0.01)
    return launches[0]["argv"]


async def test_launch_stage_cortical_tiles_forwards_masks(recording_cortical_tiles):
    """The cortical_tiles branch of _launch_stage hands the pipeline-level masks to start_cortical_tiles."""
    await pipeline._launch_stage(
        "cortical_tiles",
        umbrella_output_dir="/abs/data/DEMO01",
        input_dir="/abs/data/DEMO01/derivatives/morphologist-6.0/subjects",
        path_to_graph="graph/path",
        path_sk_with_hull="sk/path",
        masks=MASKS,
    )

    assert recording_cortical_tiles, "start_cortical_tiles was never called"
    assert recording_cortical_tiles[0].get("masks") == MASKS, recording_cortical_tiles[0]


async def test_start_pipeline_cortical_tiles_argv_carries_masks(pipeline_dir, tmp_path, launches, monkeypatch):
    """start_pipeline(masks=<value>) launches run_cortical_tiles.py with `--masks <value>`."""
    argv = await _cortical_tiles_argv_via_pipeline(tmp_path, launches, monkeypatch, masks=MASKS)

    assert _contains_run(argv, ["--masks", MASKS]), argv


@pytest.mark.parametrize("masks", [None, ""], ids=["unset", "empty"])
async def test_start_pipeline_cortical_tiles_argv_omits_masks_when_unset(
    pipeline_dir, tmp_path, launches, monkeypatch, masks
):
    """Guard: without a non-empty masks, run_cortical_tiles.py gets no `--masks` and keeps its default."""
    argv = await _cortical_tiles_argv_via_pipeline(tmp_path, launches, monkeypatch, masks=masks)

    assert "--masks" not in argv, argv
