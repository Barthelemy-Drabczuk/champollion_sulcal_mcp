"""REQ-MCPEMBVER-BDRABCZUK-7B129D469C6A: masks-versioned snapshots output dir.

When start_pipeline runs the snapshots stage, the MCP server passes
``<datasets_root>/derivatives/champollion_V1/<V>/snapshots`` as
generate_snapshots.py's ``--output_dir``, where V is the ``masks`` parameter,
or ``canonical_25`` when ``masks`` is unset.

Reference: champollion_pipeline main.py, whose default stage-6 output is
``<datasets_root>/derivatives/champollion_V1/<masks>/snapshots``, with
``derivatives_layout.DEFAULT_MASKS_VERSION = "canonical_25"``. The base is
``datasets_root``, not start_pipeline's ``output_dir``.

Supersedes REQ-MCP-SNAPOUT-01 (``<output_dir>/derivatives/champollion_V1/snapshots``).
"""

from __future__ import annotations

import pytest

from champollion_sulcal_mcp.tools import pipeline

pytestmark = pytest.mark.unit


def _argv_value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1, f"expected exactly one {flag} in argv: {argv}"
    return argv[argv.index(flag) + 1]


async def test_snapshots_output_dir_is_masks_dir_under_datasets_root_when_masks_set(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """With masks M, --output_dir is <datasets_root>/derivatives/champollion_V1/M/snapshots.

    start_pipeline's output_dir differs from datasets_root here, so the test also
    pins that the base is datasets_root.
    """
    output_dir = tmp_path / "run_output"
    datasets_root = tmp_path / "DEMO01"

    await pipeline._launch_stage(
        "snapshots",
        umbrella_output_dir=str(output_dir),
        input_dir=str(output_dir / "derivatives" / "morphologist-6.0" / "subjects"),
        datasets_root=str(datasets_root),
        masks="canonical_corrected_26_1",
    )

    argv = recording_runner[0]["argv"]
    expected = datasets_root / "derivatives" / "champollion_V1" / "canonical_corrected_26_1" / "snapshots"
    assert _argv_value(argv, "--output_dir") == str(expected)


async def test_snapshots_output_dir_defaults_to_canonical_25_when_masks_unset(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """With masks unset (None, as _execute_pipeline forwards it), V is canonical_25."""
    output_dir = tmp_path / "run_output"
    datasets_root = tmp_path / "DEMO01"

    await pipeline._launch_stage(
        "snapshots",
        umbrella_output_dir=str(output_dir),
        input_dir=str(output_dir / "derivatives" / "morphologist-6.0" / "subjects"),
        datasets_root=str(datasets_root),
        masks=None,
    )

    argv = recording_runner[0]["argv"]
    expected = datasets_root / "derivatives" / "champollion_V1" / "canonical_25" / "snapshots"
    assert _argv_value(argv, "--output_dir") == str(expected)
