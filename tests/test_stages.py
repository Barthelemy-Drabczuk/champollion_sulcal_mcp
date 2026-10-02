from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp.tools import stages


@pytest.mark.unit
async def test_start_morphologist_builds_correct_argv(fake_pipeline_dir, tmp_output_dir, recording_runner):
    result = await stages.start_morphologist(
        input_dir="/abs/input",
        output_dir=str(tmp_output_dir),
    )
    assert result["stage"] == "morphologist"
    assert result["status"] == "running"
    call = recording_runner[0]
    assert "--parallel" not in call["argv"]
    argv_str = " ".join(call["argv"])
    assert "/abs/input" in argv_str
    assert str(tmp_output_dir) in argv_str


@pytest.mark.unit
async def test_start_morphologist_with_parallel(fake_pipeline_dir, tmp_output_dir, recording_runner):
    await stages.start_morphologist(input_dir="/abs/input", output_dir=str(tmp_output_dir), parallel=True)
    assert "--parallel" in recording_runner[0]["argv"]


@pytest.mark.unit
async def test_start_morphologist_missing_pipeline_dir(monkeypatch, tmp_output_dir):
    monkeypatch.delenv("CHAMPOLLION_PIPELINE_DIR", raising=False)
    monkeypatch.setattr("champollion_sulcal_mcp.preflight.resolve_pipeline_dir",
                        lambda: (_ for _ in ()).throw(FileNotFoundError("not found")))
    with pytest.raises(ToolError):
        await stages.start_morphologist(input_dir="/abs/input", output_dir=str(tmp_output_dir))


@pytest.mark.unit
async def test_start_morphologist_relative_path_rejected():
    with pytest.raises(ToolError, match="absolute"):
        await stages.start_morphologist(input_dir="relative/path", output_dir="/abs/out")


@pytest.mark.unit
async def test_start_cortical_tiles_builds_argv(fake_pipeline_dir, tmp_output_dir, recording_runner):
    await stages.start_cortical_tiles(
        input_dir="/abs/input",
        output_dir=str(tmp_output_dir),
        path_to_graph="sub-*/t1mri/default_acquisition/default_analysis/folds/3.3",
        path_sk_with_hull="sub-*/t1mri/default_acquisition/default_analysis/segmentation",
        njobs=4,
    )
    argv = recording_runner[0]["argv"]
    assert "--path_to_graph" in argv
    assert "--njobs" in argv
    assert "4" in argv


# --- REQ-LABELSESSION-01: optional labelling-session override ---


@pytest.mark.unit
async def test_start_cortical_tiles_forwards_labelling_session(fake_pipeline_dir, tmp_output_dir, recording_runner):
    """REQ-LABELSESSION-01: a supplied labelling_session is forwarded as `--labelling_session <value>`."""
    await stages.start_cortical_tiles(
        input_dir="/abs/input",
        output_dir=str(tmp_output_dir),
        path_to_graph="sub-*/t1mri/default_acquisition/default_analysis/folds/3.3",
        path_sk_with_hull="sub-*/t1mri/default_acquisition/default_analysis/segmentation",
        labelling_session="0_auto",
    )
    argv = recording_runner[0]["argv"]
    assert argv.count("--labelling_session") == 1
    idx = argv.index("--labelling_session")
    assert argv[idx + 1] == "0_auto"


@pytest.mark.unit
async def test_start_cortical_tiles_omits_labelling_session_by_default(
    fake_pipeline_dir, tmp_output_dir, recording_runner
):
    """REQ-LABELSESSION-01: without labelling_session, no flag is added (pipeline default applies)."""
    await stages.start_cortical_tiles(
        input_dir="/abs/input",
        output_dir=str(tmp_output_dir),
        path_to_graph="sub-*/t1mri/default_acquisition/default_analysis/folds/3.3",
        path_sk_with_hull="sub-*/t1mri/default_acquisition/default_analysis/segmentation",
    )
    argv = recording_runner[0]["argv"]
    assert "--labelling_session" not in argv


# --- REQ-OVERWRITE-01: optional overwrite flag ---

_GRAPH = "sub-*/t1mri/default_acquisition/default_analysis/folds/3.3"
_SK_HULL = "sub-*/t1mri/default_acquisition/default_analysis/segmentation"


@pytest.mark.unit
async def test_start_cortical_tiles_forwards_overwrite(fake_pipeline_dir, tmp_output_dir, recording_runner):
    """REQ-OVERWRITE-01: overwrite=True appends exactly one `--overwrite` to argv."""
    await stages.start_cortical_tiles(
        input_dir="/abs/input",
        output_dir=str(tmp_output_dir),
        path_to_graph=_GRAPH,
        path_sk_with_hull=_SK_HULL,
        overwrite=True,
    )
    argv = recording_runner[0]["argv"]
    assert argv.count("--overwrite") == 1


@pytest.mark.unit
async def test_start_cortical_tiles_omits_overwrite_by_default(fake_pipeline_dir, tmp_output_dir, recording_runner):
    """REQ-OVERWRITE-01: without overwrite, no `--overwrite` flag is added."""
    await stages.start_cortical_tiles(
        input_dir="/abs/input",
        output_dir=str(tmp_output_dir),
        path_to_graph=_GRAPH,
        path_sk_with_hull=_SK_HULL,
    )
    argv = recording_runner[0]["argv"]
    assert "--overwrite" not in argv


@pytest.mark.unit
async def test_start_cortical_tiles_omits_overwrite_when_false(fake_pipeline_dir, tmp_output_dir, recording_runner):
    """REQ-OVERWRITE-01: explicit overwrite=False adds no `--overwrite` flag."""
    await stages.start_cortical_tiles(
        input_dir="/abs/input",
        output_dir=str(tmp_output_dir),
        path_to_graph=_GRAPH,
        path_sk_with_hull=_SK_HULL,
        overwrite=False,
    )
    argv = recording_runner[0]["argv"]
    assert "--overwrite" not in argv


# --- root validation ---


@pytest.mark.unit
async def test_start_morphologist_rejects_output_outside_root(fake_pipeline_dir, mock_roots):
    ctx, root = mock_roots
    outside = "/tmp/not_in_root"
    with pytest.raises(ToolError, match="outside declared roots"):
        await stages.start_morphologist(
            input_dir=str(root / "input"),
            output_dir=outside,
            ctx=ctx,
        )


@pytest.mark.unit
async def test_start_morphologist_accepts_paths_inside_root(fake_pipeline_dir, mock_roots, recording_runner):
    ctx, root = mock_roots
    input_d = root / "input"
    input_d.mkdir()
    output_d = root / "output"
    result = await stages.start_morphologist(
        input_dir=str(input_d),
        output_dir=str(output_d),
        ctx=ctx,
    )
    assert result["stage"] == "morphologist"


@pytest.mark.unit
async def test_start_morphologist_no_ctx_skips_validation(fake_pipeline_dir, tmp_output_dir, recording_runner):
    result = await stages.start_morphologist(
        input_dir="/arbitrary/absolute/path",
        output_dir=str(tmp_output_dir),
        ctx=None,
    )
    assert result["stage"] == "morphologist"


# --- REQ-MCP-SWF-01: start_training rejects swf=True before any side effect ---


@pytest.fixture
def training_script(fake_pipeline_dir):
    """Stub train_champollion.py so start_training gets past its script-exists check."""
    script = fake_pipeline_dir / "src" / "champollion_pipeline" / "train_champollion.py"
    script.write_text("# stub\n")
    return script


@pytest.mark.unit
async def test_start_training_rejects_swf_true_with_tool_error(training_script, tmp_path, recording_runner):
    """REQ-MCP-SWF-01: swf=True raises a ToolError whose message names `swf`."""
    with pytest.raises(ToolError, match="swf"):
        await stages.start_training(
            dataset="ds",
            region="S.C.-sylv.",
            output_dir=str(tmp_path / "models" / "S.C.-sylv."),
            swf=True,
        )


@pytest.mark.unit
async def test_start_training_swf_true_creates_no_output_dir(training_script, tmp_path, recording_runner):
    """REQ-MCP-SWF-01: swf=True leaves the requested output directory uncreated."""
    out = tmp_path / "models" / "S.C.-sylv."
    with pytest.raises(ToolError):
        await stages.start_training(dataset="ds", region="S.C.-sylv.", output_dir=str(out), swf=True)
    assert not out.exists()


@pytest.mark.unit
async def test_start_training_swf_true_launches_no_job(training_script, tmp_path, recording_runner):
    """REQ-MCP-SWF-01: swf=True never reaches runner.launch."""
    with pytest.raises(ToolError):
        await stages.start_training(
            dataset="ds",
            region="S.C.-sylv.",
            output_dir=str(tmp_path / "models" / "S.C.-sylv."),
            swf=True,
        )
    assert recording_runner == []


@pytest.mark.unit
async def test_start_training_swf_false_launches_without_swf_flag(training_script, tmp_path, recording_runner):
    """REQ-MCP-SWF-01 regression guard: swf=False still launches one job without `--swf`."""
    out = tmp_path / "models" / "S.C.-sylv."
    result = await stages.start_training(dataset="ds", region="S.C.-sylv.", output_dir=str(out), swf=False)
    assert result["stage"] == "training"
    assert out.is_dir()
    assert len(recording_runner) == 1
    assert "--swf" not in recording_runner[0]["argv"]


# --- REQ-MCP-SNAPREF-01: reference_data_dir passthrough (TASK-053) ---


@pytest.mark.unit
async def test_start_snapshots_forwards_reference_data_dir(fake_pipeline_dir, tmp_output_dir, recording_runner):
    """REQ-MCP-SNAPREF-01: a supplied reference_data_dir reaches generate_snapshots.py as
    `--reference_data_dir <value>`; without it the script's `_run_umap` returns no UMAP plot.
    """
    reference = str(tmp_output_dir / "reference_data")
    await stages.start_snapshots(output_dir=str(tmp_output_dir), reference_data_dir=reference)
    argv = recording_runner[0]["argv"]
    assert argv.count("--reference_data_dir") == 1
    assert argv[argv.index("--reference_data_dir") + 1] == reference


@pytest.mark.unit
async def test_start_snapshots_omits_reference_data_dir_by_default(fake_pipeline_dir, tmp_output_dir, recording_runner):
    """Guard for REQ-MCP-SNAPREF-01: without reference_data_dir, no flag (and no `None` value) is passed."""
    await stages.start_snapshots(output_dir=str(tmp_output_dir))
    argv = recording_runner[0]["argv"]
    assert "--reference_data_dir" not in argv
    assert "None" not in argv
