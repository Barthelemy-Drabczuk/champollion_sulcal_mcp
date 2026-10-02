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


# --- REQ-MCP-OUTLOC-01 / REQ-MCP-OUTLOC-02: combined-embeddings location (TASK-049) ---


def _argv_value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1, f"expected exactly one {flag} in argv: {argv}"
    return argv[argv.index(flag) + 1]


@pytest.mark.unit
async def test_combine_argv_output_path_is_derivatives_champollion_v1_embeddings(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """REQ-MCP-OUTLOC-01: start_pipeline's combine stage, for dataset root R,
    launches put_together_embeddings.py with
    `--output_path R/derivatives/champollion_V1/embeddings`, not bare R.
    """
    dataset_root = tmp_path / "DEMO01"

    await pipeline._launch_stage(
        "combine",
        umbrella_output_dir=str(dataset_root),
        datasets_root=str(dataset_root),
    )

    argv = recording_runner[0]["argv"]
    assert _argv_value(argv, "--output_path") == str(dataset_root / "derivatives" / "champollion_V1" / "embeddings")


@pytest.mark.unit
async def test_snapshots_argv_embeddings_dir_matches_combine_output_path(fake_pipeline_dir, tmp_path, recording_runner):
    """REQ-MCP-OUTLOC-02: start_pipeline's snapshots stage, for dataset root R,
    launches generate_snapshots.py with
    `--embeddings_dir R/derivatives/champollion_V1/embeddings` — the same
    directory the combine stage writes to — so UMAP reads the combined CSVs.
    """
    dataset_root = tmp_path / "DEMO01"
    common = {"umbrella_output_dir": str(dataset_root), "datasets_root": str(dataset_root)}

    await pipeline._launch_stage("combine", **common)
    await pipeline._launch_stage(
        "snapshots",
        input_dir=str(dataset_root / "derivatives" / "morphologist-6.0" / "subjects"),
        **common,
    )

    combine_argv, snapshots_argv = recording_runner[0]["argv"], recording_runner[1]["argv"]
    expected = str(dataset_root / "derivatives" / "champollion_V1" / "embeddings")
    assert _argv_value(snapshots_argv, "--embeddings_dir") == expected
    assert _argv_value(snapshots_argv, "--embeddings_dir") == _argv_value(combine_argv, "--output_path")


# --- REQ-MCP-SNAPREF-02: start_pipeline forwards reference_data_dir to stage 6 (TASK-053) ---


@pytest.mark.unit
async def test_start_pipeline_hands_reference_data_dir_to_pipeline_run(monkeypatch, tmp_path):
    """REQ-MCP-SNAPREF-02: start_pipeline accepts reference_data_dir and hands it to
    the background pipeline run, whose kwargs reach _launch_stage.
    """
    captured: dict = {}

    async def _fake_run_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "_run_pipeline", _fake_run_pipeline)

    reference = str(tmp_path / "reference_data")
    await pipeline.start_pipeline(
        input_dir=str(tmp_path / "subjects"),
        output_dir=str(tmp_path),
        path_to_graph="graph/path",
        path_sk_with_hull="sk/path",
        crop_path=str(tmp_path / "crops"),
        dataset="DEMO01",
        models_path=str(tmp_path / "models"),
        datasets_root=str(tmp_path / "datasets"),
        reference_data_dir=reference,
    )
    await asyncio.sleep(0)

    assert captured.get("reference_data_dir") == reference


@pytest.mark.unit
async def test_snapshots_stage_argv_carries_reference_data_dir(fake_pipeline_dir, tmp_path, recording_runner):
    """REQ-MCP-SNAPREF-02: start_pipeline's snapshots stage, given reference_data_dir,
    launches generate_snapshots.py with `--reference_data_dir <value>`.
    """
    dataset_root = tmp_path / "DEMO01"
    reference = str(tmp_path / "reference_data")

    await pipeline._launch_stage(
        "snapshots",
        umbrella_output_dir=str(dataset_root),
        input_dir=str(dataset_root / "derivatives" / "morphologist-6.0" / "subjects"),
        datasets_root=str(dataset_root),
        reference_data_dir=reference,
    )

    assert _argv_value(recording_runner[0]["argv"], "--reference_data_dir") == reference


@pytest.mark.unit
async def test_snapshots_argv_output_dir_is_derivatives_champollion_v1_snapshots(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """REQ-MCP-SNAPOUT-01: start_pipeline's snapshots stage, for dataset root R,
    launches generate_snapshots.py with
    `--output_dir R/derivatives/champollion_V1/snapshots`, not bare R.
    """
    dataset_root = tmp_path / "DEMO01"

    await pipeline._launch_stage(
        "snapshots",
        umbrella_output_dir=str(dataset_root),
        input_dir=str(dataset_root / "derivatives" / "morphologist-6.0" / "subjects"),
        datasets_root=str(dataset_root),
    )

    argv = recording_runner[0]["argv"]
    assert _argv_value(argv, "--output_dir") == str(dataset_root / "derivatives" / "champollion_V1" / "snapshots")
