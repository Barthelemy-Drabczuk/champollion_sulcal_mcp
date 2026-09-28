from __future__ import annotations

import asyncio

import pytest

from champollion_sulcal_mcp.tools import pipeline


@pytest.fixture
def recording_stages(monkeypatch):
    """Replace every stage launcher with a recorder, so _launch_stage's argument
    wiring can be asserted without running any pipeline script."""
    calls: dict[str, dict] = {}

    def _recorder(name):
        async def _fn(**kwargs):
            calls[name] = kwargs
            return {"stage": name, "status": "running", "output_dir": kwargs.get("output_dir", "")}

        return _fn

    for attr in (
        "start_morphologist",
        "start_cortical_tiles",
        "start_config",
        "start_embeddings",
        "start_combine",
        "start_snapshots",
    ):
        monkeypatch.setattr(pipeline.stages, attr, _recorder(attr))
    return calls


@pytest.mark.unit
async def test_launch_stage_cortical_tiles_targets_derivatives_subdir(recording_stages):
    """REQ-CROPPATH-01: the cortical_tiles stage must be launched against
    <dataset_root>/derivatives, so crops land under
    <dataset_root>/derivatives/cortical_tiles-<version>/crops/... — the exact
    location generate_embeddings.py builds when it looks for 2mm crops.
    """
    dataset_root = "/abs/data/TEST02-verify"

    await pipeline._launch_stage(
        "cortical_tiles",
        umbrella_output_dir=dataset_root,
        input_dir="/abs/data/TEST02-verify/derivatives/morphologist-6.0/subjects",
        path_to_graph="graph/path",
        path_sk_with_hull="sk/path",
    )

    assert recording_stages["start_cortical_tiles"]["output_dir"] == f"{dataset_root}/derivatives"


@pytest.mark.unit
async def test_launch_stage_morphologist_keeps_bare_dataset_root(recording_stages):
    """Guard for REQ-CROPPATH-01: morphologist inserts `derivatives/` itself, so
    its output_dir must stay the bare dataset root and must NOT be rewritten.
    """
    dataset_root = "/abs/data/TEST02-verify"

    await pipeline._launch_stage(
        "morphologist",
        umbrella_output_dir=dataset_root,
        input_dir="/abs/data/TEST02-verify/rawdata",
    )

    assert recording_stages["start_morphologist"]["output_dir"] == dataset_root


@pytest.mark.unit
async def test_launch_stage_cortical_tiles_forwards_labelling_session(recording_stages):
    """REQ-LABELSESSION-02: the cortical_tiles branch of _launch_stage must pass
    the pipeline-level labelling_session through to start_cortical_tiles.
    """
    await pipeline._launch_stage(
        "cortical_tiles",
        umbrella_output_dir="/abs/data/DEMO01",
        input_dir="/abs/data/DEMO01/derivatives/morphologist-6.0/subjects",
        path_to_graph="graph/path",
        path_sk_with_hull="sk/path",
        labelling_session="0_auto",
    )

    assert recording_stages["start_cortical_tiles"].get("labelling_session") == "0_auto"


@pytest.mark.unit
async def test_launch_stage_cortical_tiles_labelling_session_defaults_to_none(recording_stages):
    """Guard for REQ-LABELSESSION-02: when no labelling_session reaches
    _launch_stage, start_cortical_tiles must see None, so run_cortical_tiles.py
    keeps its own default session.
    """
    await pipeline._launch_stage(
        "cortical_tiles",
        umbrella_output_dir="/abs/data/DEMO01",
        input_dir="/abs/data/DEMO01/derivatives/morphologist-6.0/subjects",
        path_to_graph="graph/path",
        path_sk_with_hull="sk/path",
    )

    assert recording_stages["start_cortical_tiles"].get("labelling_session") is None


@pytest.mark.unit
async def test_start_pipeline_hands_labelling_session_to_pipeline_run(monkeypatch, tmp_path):
    """REQ-LABELSESSION-02: start_pipeline must accept labelling_session and hand
    it to the background pipeline run, whose kwargs reach _launch_stage.
    """
    captured: dict = {}

    async def _fake_run_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "_run_pipeline", _fake_run_pipeline)

    await pipeline.start_pipeline(
        input_dir=str(tmp_path / "subjects"),
        output_dir=str(tmp_path),
        path_to_graph="graph/path",
        path_sk_with_hull="sk/path",
        crop_path=str(tmp_path / "crops"),
        dataset="DEMO01",
        models_path=str(tmp_path / "models"),
        datasets_root=str(tmp_path / "datasets"),
        labelling_session="0_auto",
    )
    await asyncio.sleep(0)

    assert captured.get("labelling_session") == "0_auto"
