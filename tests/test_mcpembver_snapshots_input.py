"""REQ-MCPEMBVER-BDRABCZUK-5453862A9906: masks-versioned snapshots input dir.

When start_pipeline runs the snapshots stage, the MCP server passes
``<datasets_root>/derivatives/champollion_V1/<V>/embeddings`` as
generate_snapshots.py's ``--embeddings_dir``, where V is the ``masks``
parameter, or ``canonical_25`` when ``masks`` is unset.

This is the directory the combine stage writes
(REQ-MCPEMBVER-BDRABCZUK-6418DC8E1DF1), so UMAP snapshots read the combined
CSVs. The base is ``datasets_root``, as in champollion_pipeline main.py, not
start_pipeline's ``output_dir``.

Supersedes REQ-MCP-OUTLOC-06 (``<output_dir>/derivatives/champollion_V1/embeddings``).
"""

from __future__ import annotations

import pytest

from champollion_sulcal_mcp.tools import pipeline

pytestmark = pytest.mark.unit


def _argv_value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1, f"expected exactly one {flag} in argv: {argv}"
    return argv[argv.index(flag) + 1]


async def test_snapshots_embeddings_dir_is_masks_dir_under_datasets_root_when_masks_set(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """With masks M, --embeddings_dir is <datasets_root>/derivatives/champollion_V1/M/embeddings.

    start_pipeline's output_dir differs from datasets_root here, so the test also
    pins that the base is datasets_root.
    """
    output_dir = tmp_path / "run_output"
    datasets_root = tmp_path / "DEMO01"

    await pipeline._launch_stage(
        "snapshots",
        umbrella_output_dir=str(output_dir),
        datasets_root=str(datasets_root),
        input_dir=str(datasets_root / "derivatives" / "morphologist-6.0" / "subjects"),
        masks="canonical_corrected_26_1",
    )

    argv = recording_runner[0]["argv"]
    expected = datasets_root / "derivatives" / "champollion_V1" / "canonical_corrected_26_1" / "embeddings"
    assert _argv_value(argv, "--embeddings_dir") == str(expected)


async def test_snapshots_embeddings_dir_defaults_to_canonical_25_when_masks_unset(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """With masks unset (None, as _execute_pipeline forwards it), V is canonical_25."""
    output_dir = tmp_path / "run_output"
    datasets_root = tmp_path / "DEMO01"

    await pipeline._launch_stage(
        "snapshots",
        umbrella_output_dir=str(output_dir),
        datasets_root=str(datasets_root),
        input_dir=str(datasets_root / "derivatives" / "morphologist-6.0" / "subjects"),
        masks=None,
    )

    argv = recording_runner[0]["argv"]
    expected = datasets_root / "derivatives" / "champollion_V1" / "canonical_25" / "embeddings"
    assert _argv_value(argv, "--embeddings_dir") == str(expected)
