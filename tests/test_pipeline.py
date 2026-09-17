from __future__ import annotations

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
