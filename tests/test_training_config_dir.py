"""REQ-MCP-CFGLOC-03: start_training defaults --config-dir under the first MCP root.

When the caller omits ``config_dir`` and the MCP client declares at least one
root, ``start_training`` passes
``--config-dir <roots[0]>/data/<dataset>/derivatives/champollion_V1/configs``
to ``train_champollion.py``. That shares its
``<roots[0]>/data/<dataset>/derivatives/champollion_V1`` base with the
roots-declared ``output_dir`` default (``.../models/<region>``), and equals the
stage-3 default configs root ``<D>/<dataset>/derivatives/champollion_V1/configs``
for crops living under ``<roots[0]>/data/``.

Regression guards (kept behaviour): a supplied ``config_dir`` is forwarded
unchanged, and with no roots no ``--config-dir`` is injected, so
``train_champollion``'s own default applies.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from champollion_sulcal_mcp.tools import stages

pytestmark = pytest.mark.unit

DATASET = "DS01"
REGION = "S.C.-sylv."


@pytest.fixture
def training_script(fake_pipeline_dir):
    """Stub train_champollion.py so start_training gets past its script-exists check."""
    script = fake_pipeline_dir / "src" / "champollion_pipeline" / "train_champollion.py"
    script.write_text("# stub\n")
    return script


def _expected_configs_root(root: Path) -> str:
    return str(root.resolve() / "data" / DATASET / "derivatives" / "champollion_V1" / "configs")


def _config_dir_values(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--config-dir"]


def _ctx_with_roots(*roots: Path):
    def _root_obj(path: Path):
        obj = MagicMock()
        obj.uri = f"file://{path}"
        return obj

    ctx = MagicMock()
    ctx.list_roots = AsyncMock(return_value=[_root_obj(r) for r in roots])
    ctx.info = AsyncMock()
    return ctx


# --- REQ-MCP-CFGLOC-03: roots-declared default ---


async def test_start_training_defaults_config_dir_under_first_root(training_script, mock_roots, recording_runner):
    """Roots declared, config_dir and output_dir omitted: --config-dir is the roots[0] configs root."""
    ctx, root = mock_roots
    await stages.start_training(dataset=DATASET, region=REGION, ctx=ctx)
    argv = recording_runner[0]["argv"]
    assert _config_dir_values(argv) == [_expected_configs_root(root)], argv


async def test_start_training_defaults_config_dir_with_explicit_output_dir(
    training_script, mock_roots, recording_runner
):
    """Roots declared, config_dir omitted, output_dir supplied: the roots[0] default still applies."""
    ctx, root = mock_roots
    out = root / "models" / REGION
    await stages.start_training(dataset=DATASET, region=REGION, output_dir=str(out), ctx=ctx)
    argv = recording_runner[0]["argv"]
    assert _config_dir_values(argv) == [_expected_configs_root(root)], argv


async def test_start_training_default_config_dir_shares_output_dir_base(training_script, mock_roots, recording_runner):
    """The defaulted --config-dir and --output_dir hang off the same champollion_V1 derivatives dir."""
    ctx, _ = mock_roots
    await stages.start_training(dataset=DATASET, region=REGION, ctx=ctx)
    argv = recording_runner[0]["argv"]
    config_dirs = _config_dir_values(argv)
    assert len(config_dirs) == 1, argv
    output_dir = Path(argv[argv.index("--output_dir") + 1])
    # output_dir = <base>/models/<region>; config_dir = <base>/configs
    assert Path(config_dirs[0]) == output_dir.parent.parent / "configs"


async def test_start_training_default_config_dir_uses_first_of_several_roots(
    training_script, tmp_path, recording_runner
):
    """With several roots declared, the default --config-dir is built from roots[0], not a later root."""
    first = tmp_path / "root_a"
    second = tmp_path / "root_b"
    first.mkdir()
    second.mkdir()
    ctx = _ctx_with_roots(first, second)
    await stages.start_training(dataset=DATASET, region=REGION, ctx=ctx)
    argv = recording_runner[0]["argv"]
    assert _config_dir_values(argv) == [_expected_configs_root(first)], argv


# --- Regression guards (pass before and after REQ-MCP-CFGLOC-03) ---


async def test_start_training_supplied_config_dir_forwarded_once_with_roots(
    training_script, mock_roots, recording_runner
):
    """Roots declared and config_dir supplied: exactly one --config-dir, carrying the caller's value."""
    ctx, root = mock_roots
    supplied = str(root / "elsewhere" / "configs")
    await stages.start_training(dataset=DATASET, region=REGION, config_dir=supplied, ctx=ctx)
    argv = recording_runner[0]["argv"]
    assert _config_dir_values(argv) == [supplied], argv


async def test_start_training_omits_config_dir_without_roots(training_script, tmp_path, recording_runner):
    """No ctx (no roots) and no config_dir: argv carries no --config-dir, so train_champollion's default applies."""
    out = tmp_path / "models" / REGION
    await stages.start_training(dataset=DATASET, region=REGION, output_dir=str(out))
    assert "--config-dir" not in recording_runner[0]["argv"]
