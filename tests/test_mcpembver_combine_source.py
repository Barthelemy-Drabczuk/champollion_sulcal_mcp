"""REQ-MCPEMBVER-BDRABCZUK-ECBCBA05BD02: masks-versioned combine embeddings_source.

When start_pipeline runs the combine stage, the MCP server passes
``<datasets_root>/derivatives/champollion_V1/<V>/region_embeddings`` as
put_together_embeddings.py's positional embeddings_source argument, where V is
the ``masks`` parameter, or ``canonical_25`` when ``masks`` is unset.

Reference layout (champollion_pipeline main@1745700):
``derivatives_layout.compute_region_embeddings_dir(datasets_root, masks_version)``
with ``DEFAULT_MASKS_VERSION = "canonical_25"``. It is main.py's stage 5 input
and generate_embeddings.py's own default ``--output`` (stage 4 writes there when
start_pipeline passes no ``--output``).

Replaces the old ``<parent of datasets_root>/<name of datasets_root>embeddings``
derivation in ``_launch_stage``.
"""

from __future__ import annotations

import pytest

from champollion_sulcal_mcp.tools import pipeline

pytestmark = pytest.mark.unit

_SCRIPT = "put_together_embeddings.py"


def _embeddings_source(argv: list[str]) -> str:
    """Return the positional argument right after put_together_embeddings.py in argv."""
    scripts = [i for i, a in enumerate(argv) if a.endswith(_SCRIPT)]
    assert len(scripts) == 1, f"expected exactly one {_SCRIPT} in argv: {argv}"
    source = argv[scripts[0] + 1]
    assert not source.startswith("-"), f"expected a positional embeddings_source after {_SCRIPT}: {argv}"
    return source


async def test_combine_embeddings_source_is_region_embeddings_under_datasets_root_when_masks_set(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """With masks M, embeddings_source is <datasets_root>/derivatives/champollion_V1/M/region_embeddings.

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
    expected = datasets_root / "derivatives" / "champollion_V1" / "canonical_corrected_26_1" / "region_embeddings"
    assert _embeddings_source(argv) == str(expected)


async def test_combine_embeddings_source_defaults_to_canonical_25_when_masks_unset(
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
    expected = datasets_root / "derivatives" / "champollion_V1" / "canonical_25" / "region_embeddings"
    assert _embeddings_source(argv) == str(expected)
