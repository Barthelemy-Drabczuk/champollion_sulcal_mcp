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
