"""REQ-MCP-CFGLOC-01 / REQ-MCP-CFGLOC-02: config-location flags are caller-owned.

Reference (champollion_pipeline 7adda32, TASK-141):
- ``generate_champollion_config.py`` with no ``--output`` / ``--external-config``
  writes dataset YAMLs and ``dataset_localization/<loc>.yaml`` under
  ``<D>/<dataset>/derivatives/champollion_V1/configs``.
- ``train_champollion.py`` with no ``--config-dir`` reads
  ``<pipeline>/data/<dataset>/derivatives/champollion_V1/configs``.

The MCP tools must therefore forward these flags only when the caller supplies
them, so the pipeline's own defaults apply otherwise.
"""

from __future__ import annotations

import pytest

from champollion_sulcal_mcp.tools import pipeline, stages

pytestmark = pytest.mark.unit


@pytest.fixture
def crop_path(mock_roots):
    """A crops/2mm directory inside the declared MCP root."""
    _, root = mock_roots
    crops = root / "DS01" / "derivatives" / "cortical_tiles-2026" / "crops" / "2mm"
    crops.mkdir(parents=True)
    return crops


@pytest.fixture
def training_script(fake_pipeline_dir):
    """Stub train_champollion.py so start_training gets past its script-exists check."""
    script = fake_pipeline_dir / "src" / "champollion_pipeline" / "train_champollion.py"
    script.write_text("# stub\n")
    return script


# --- REQ-MCP-CFGLOC-01: start_config ---


async def test_start_config_omits_external_config_when_not_supplied(
    fake_pipeline_dir, mock_roots, crop_path, recording_runner
):
    """With MCP roots declared and no external_config, argv carries no --external-config."""
    ctx, _ = mock_roots
    await stages.start_config(crop_path=str(crop_path), dataset="DS01", ctx=ctx)
    argv = recording_runner[0]["argv"]
    assert "--external-config" not in argv, f"start_config injected its own --external-config: {argv}"


async def test_start_config_omits_output_when_not_supplied(fake_pipeline_dir, mock_roots, crop_path, recording_runner):
    """With MCP roots declared and no output, argv carries no --output."""
    ctx, _ = mock_roots
    await stages.start_config(crop_path=str(crop_path), dataset="DS01", ctx=ctx)
    assert "--output" not in recording_runner[0]["argv"]


async def test_start_config_forwards_supplied_external_config(
    fake_pipeline_dir, mock_roots, crop_path, recording_runner
):
    """A caller-supplied external_config is forwarded verbatim as --external-config."""
    ctx, root = mock_roots
    target = str(root / "writable" / "local.yaml")
    await stages.start_config(crop_path=str(crop_path), dataset="DS01", external_config=target, ctx=ctx)
    argv = recording_runner[0]["argv"]
    assert "--external-config" in argv
    assert argv[argv.index("--external-config") + 1] == target


async def test_start_config_forwards_supplied_output(fake_pipeline_dir, mock_roots, crop_path, recording_runner):
    """A caller-supplied output is forwarded verbatim as --output (the configs root)."""
    ctx, root = mock_roots
    configs_root = root / "custom_configs"
    configs_root.mkdir()
    await stages.start_config(crop_path=str(crop_path), dataset="DS01", output=str(configs_root), ctx=ctx)
    argv = recording_runner[0]["argv"]
    assert "--output" in argv
    assert argv[argv.index("--output") + 1] == str(configs_root)


async def test_launch_stage_config_passes_neither_output_nor_external_config(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """start_pipeline's config stage launches generate_champollion_config.py with neither flag."""
    crops = tmp_path / "DS01" / "derivatives" / "cortical_tiles-2026" / "crops" / "2mm"
    crops.mkdir(parents=True)
    await pipeline._launch_stage(
        "config",
        umbrella_output_dir=str(tmp_path / "DS01"),
        crop_path=str(crops),
        dataset="DS01",
    )
    argv = recording_runner[0]["argv"]
    assert "--output" not in argv
    assert "--external-config" not in argv


# --- REQ-MCP-CFGLOC-02 guards: start_training leaves --config-dir to the caller ---


async def test_start_training_omits_config_dir_when_not_supplied(training_script, mock_roots, recording_runner):
    """With no config_dir, argv carries no --config-dir, so train_champollion's default applies."""
    ctx, root = mock_roots
    out = root / "models" / "S.C.-sylv."
    await stages.start_training(dataset="DS01", region="S.C.-sylv.", output_dir=str(out), ctx=ctx)
    assert "--config-dir" not in recording_runner[0]["argv"]


async def test_start_training_forwards_supplied_config_dir(training_script, mock_roots, recording_runner):
    """A caller-supplied config_dir is forwarded verbatim as --config-dir."""
    ctx, root = mock_roots
    out = root / "models" / "S.C.-sylv."
    configs_root = str(root / "DS01" / "derivatives" / "champollion_V1" / "configs")
    await stages.start_training(
        dataset="DS01", region="S.C.-sylv.", output_dir=str(out), config_dir=configs_root, ctx=ctx
    )
    argv = recording_runner[0]["argv"]
    assert "--config-dir" in argv
    assert argv[argv.index("--config-dir") + 1] == configs_root
