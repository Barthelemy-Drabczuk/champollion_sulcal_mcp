"""REQ-MCP-ROOTS-01 / REQ-MCP-ROOTS-02: start_config's write paths are path-safe.

``output`` (the configs root) and ``external_config`` (the dataset_localization
YAML, a file or a directory) are write destinations, so like every other write
path they must be absolute and, when the MCP client declares roots, inside one.
A rejection is a ToolError raised before any job is launched.
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp.tools import stages

pytestmark = pytest.mark.unit


@pytest.fixture
def crop_path(mock_roots):
    """A crops/2mm directory inside the declared MCP root."""
    _, root = mock_roots
    crops = root / "DS01" / "derivatives" / "cortical_tiles-2026" / "crops" / "2mm"
    crops.mkdir(parents=True)
    return crops


@pytest.fixture
def outside_dir(tmp_path, mock_roots):
    """A directory that exists but is not inside the declared MCP root."""
    _, root = mock_roots
    d = tmp_path / "elsewhere"
    d.mkdir()
    assert not d.resolve().is_relative_to(root.resolve())
    return d


@pytest.fixture
def scratch_cwd(tmp_path, monkeypatch):
    """Run from a throwaway cwd outside the MCP root, so a relative path never touches the repo."""
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    return cwd


# --- REQ-MCP-ROOTS-01: roots containment (roots declared) ---


async def test_start_config_rejects_external_config_file_outside_roots(
    fake_pipeline_dir, mock_roots, crop_path, outside_dir, recording_runner
):
    """An external_config YAML file path outside the declared roots is rejected, no job launched."""
    ctx, _ = mock_roots
    target = str(outside_dir / "local.yaml")
    with pytest.raises(ToolError, match=r"external_config .*outside declared roots"):
        await stages.start_config(crop_path=str(crop_path), dataset="DS01", external_config=target, ctx=ctx)
    assert recording_runner == []


async def test_start_config_rejects_external_config_dir_outside_roots(
    fake_pipeline_dir, mock_roots, crop_path, outside_dir, recording_runner
):
    """An external_config directory path outside the declared roots is rejected, no job launched."""
    ctx, _ = mock_roots
    with pytest.raises(ToolError, match=r"external_config .*outside declared roots"):
        await stages.start_config(crop_path=str(crop_path), dataset="DS01", external_config=str(outside_dir), ctx=ctx)
    assert recording_runner == []


async def test_start_config_rejects_output_outside_roots(
    fake_pipeline_dir, mock_roots, crop_path, outside_dir, recording_runner
):
    """Guard: an output configs root outside the declared roots is rejected, no job launched."""
    ctx, _ = mock_roots
    with pytest.raises(ToolError, match=r"output .*outside declared roots"):
        await stages.start_config(crop_path=str(crop_path), dataset="DS01", output=str(outside_dir), ctx=ctx)
    assert recording_runner == []


async def test_start_config_forwards_in_root_output_and_external_config(
    fake_pipeline_dir, mock_roots, crop_path, recording_runner
):
    """Guard: in-root output and external_config are forwarded unchanged."""
    ctx, root = mock_roots
    output = str(root / "custom_configs")
    external_config = str(root / "writable" / "local.yaml")
    await stages.start_config(
        crop_path=str(crop_path), dataset="DS01", output=output, external_config=external_config, ctx=ctx
    )
    argv = recording_runner[0]["argv"]
    assert argv[argv.index("--output") + 1] == output
    assert argv[argv.index("--external-config") + 1] == external_config


async def test_start_config_forwards_out_of_root_paths_without_roots(fake_pipeline_dir, tmp_path, recording_runner):
    """Guard: with no roots declared, absolute output/external_config anywhere are forwarded unchanged."""
    crops = tmp_path / "DS01" / "derivatives" / "cortical_tiles-2026" / "crops" / "2mm"
    crops.mkdir(parents=True)
    output = str(tmp_path / "anywhere" / "configs")
    external_config = str(tmp_path / "anywhere" / "local.yaml")
    await stages.start_config(
        crop_path=str(crops), dataset="DS01", output=output, external_config=external_config, ctx=None
    )
    argv = recording_runner[0]["argv"]
    assert argv[argv.index("--output") + 1] == output
    assert argv[argv.index("--external-config") + 1] == external_config


# --- REQ-MCP-ROOTS-02: absolute paths required ---


async def test_start_config_rejects_relative_output_with_roots(
    scratch_cwd, fake_pipeline_dir, mock_roots, crop_path, recording_runner
):
    """A relative output is rejected as non-absolute (roots declared), no job launched."""
    ctx, _ = mock_roots
    with pytest.raises(ToolError, match=r"output must be an absolute path"):
        await stages.start_config(crop_path=str(crop_path), dataset="DS01", output="rel/configs", ctx=ctx)
    assert recording_runner == []


async def test_start_config_rejects_relative_external_config_with_roots(
    scratch_cwd, fake_pipeline_dir, mock_roots, crop_path, recording_runner
):
    """A relative external_config is rejected as non-absolute (roots declared), no job launched."""
    ctx, _ = mock_roots
    with pytest.raises(ToolError, match=r"external_config must be an absolute path"):
        await stages.start_config(crop_path=str(crop_path), dataset="DS01", external_config="rel/local.yaml", ctx=ctx)
    assert recording_runner == []


async def test_start_config_rejects_relative_output_without_roots(
    scratch_cwd, fake_pipeline_dir, tmp_path, recording_runner
):
    """A relative output is rejected as non-absolute even with no roots declared, no job launched."""
    crops = tmp_path / "crops"
    crops.mkdir()
    with pytest.raises(ToolError, match=r"output must be an absolute path"):
        await stages.start_config(crop_path=str(crops), dataset="DS01", output="rel/configs", ctx=None)
    assert recording_runner == []


async def test_start_config_rejects_relative_external_config_without_roots(
    scratch_cwd, fake_pipeline_dir, tmp_path, recording_runner
):
    """A relative external_config is rejected as non-absolute even with no roots declared, no job launched."""
    crops = tmp_path / "crops"
    crops.mkdir()
    with pytest.raises(ToolError, match=r"external_config must be an absolute path"):
        await stages.start_config(crop_path=str(crops), dataset="DS01", external_config="rel/local.yaml", ctx=None)
    assert recording_runner == []
