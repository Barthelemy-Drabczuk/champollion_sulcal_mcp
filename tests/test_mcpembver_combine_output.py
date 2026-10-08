"""REQ-MCPEMBVER-BDRABCZUK-6418DC8E1DF1: masks-versioned combine output path.

When start_pipeline runs the combine stage, the MCP server passes
``<datasets_root>/derivatives/champollion_V1/<V>/embeddings`` as
put_together_embeddings.py's ``--output_path``, where V is the ``masks``
parameter, or ``canonical_25`` when ``masks`` is unset.

Reference layout (champollion_pipeline main@1745700):
``derivatives_layout.compute_combined_embeddings_dir(datasets_root, masks_version)``
with ``DEFAULT_MASKS_VERSION = "canonical_25"``. The base is ``datasets_root``,
as in champollion_pipeline main.py, not start_pipeline's ``output_dir``.

Supersedes REQ-MCP-OUTLOC-05 (``<output_dir>/derivatives/champollion_V1/embeddings``).
"""

from __future__ import annotations

import pytest

from champollion_sulcal_mcp.tools import pipeline

pytestmark = pytest.mark.unit


def _argv_value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1, f"expected exactly one {flag} in argv: {argv}"
    return argv[argv.index(flag) + 1]


async def test_combine_output_path_is_masks_dir_under_datasets_root_when_masks_set(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """With masks M, --output_path is <datasets_root>/derivatives/champollion_V1/M/embeddings.

    start_pipeline's output_dir differs from datasets_root here, so the test also
    pins that the base is datasets_root.
    """
    output_dir = tmp_path / "run_output"
    datasets_root = tmp_path / "DEMO01"

    await pipeline._launch_stage(
        "combine",
        umbrella_output_dir=str(output_dir),
        datasets_root=str(datasets_root),
        masks="canonical_corrected_26_1",
    )

    argv = recording_runner[0]["argv"]
    expected = datasets_root / "derivatives" / "champollion_V1" / "canonical_corrected_26_1" / "embeddings"
    assert _argv_value(argv, "--output_path") == str(expected)


async def test_combine_output_path_defaults_to_canonical_25_when_masks_unset(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """With masks unset (None, as _execute_pipeline forwards it), V is canonical_25."""
    output_dir = tmp_path / "run_output"
    datasets_root = tmp_path / "DEMO01"

    await pipeline._launch_stage(
        "combine",
        umbrella_output_dir=str(output_dir),
        datasets_root=str(datasets_root),
        masks=None,
    )

    argv = recording_runner[0]["argv"]
    expected = datasets_root / "derivatives" / "champollion_V1" / "canonical_25" / "embeddings"
    assert _argv_value(argv, "--output_path") == str(expected)
