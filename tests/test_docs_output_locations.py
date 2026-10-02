"""REQ-MCP-OUTLOC-03 / REQ-MCP-OUTLOC-04 / REQ-MCP-SNAPOUT-02: Output Locations docs match start_pipeline.

Reference (champollion_sulcal_mcp TASK-049; champollion_pipeline WIP):
- start_pipeline's ``output_dir`` is the dataset root R (REQ-CROPPATH-01); the
  combine stage writes ``{region}_embeddings.csv`` to
  ``R/derivatives/champollion_V1/embeddings/`` (REQ-MCP-OUTLOC-01).
- ``generate_embeddings.py`` writes per-region ``full_embeddings.csv`` to
  ``{output}/{region}/``, ``output`` defaulting to
  ``{parent of datasets_root}/{basename of datasets_root}embeddings/`` — never
  to the ``models_cache`` (downloaded-model cache).
- the snapshots stage writes images to ``R/derivatives/champollion_V1/snapshots/``
  (REQ-MCP-SNAPOUT-01; ``generate_snapshots.py`` writes straight into
  ``--output_dir``, matching the pipeline README's
  ``derivatives/champollion_V1/snapshots/`` example).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent

DOCS = [
    REPO_ROOT / "skills" / "run-pipeline" / "SKILL.md",
    REPO_ROOT / "agents" / "champollion-pipeline.md",
]
DOC_IDS = [str(p.relative_to(REPO_ROOT)) for p in DOCS]

COMBINED_PATH = "{output_dir}/derivatives/champollion_V1/embeddings/"
SNAPSHOTS_PATH = "{output_dir}/derivatives/champollion_V1/snapshots/"
PER_FOLD_PATH = "{parent of datasets_root}/{basename of datasets_root}embeddings/{region}/full_embeddings.csv"


def _output_locations_rows(doc: Path, label: str) -> list[str]:
    """Table rows starting with `| <label> |` inside every '## Output Locations' section."""
    sections = re.findall(r"^## Output Locations\b.*?(?=^## |\Z)", doc.read_text(), re.MULTILINE | re.DOTALL)
    assert sections, f"{doc.relative_to(REPO_ROOT)} has no '## Output Locations' section"
    rows = [
        line
        for section in sections
        for line in section.splitlines()
        if re.match(rf"^\|\s*{re.escape(label)}\s*\|", line)
    ]
    assert rows, f"{doc.relative_to(REPO_ROOT)}: no '{label}' row in Output Locations"
    return rows


def _path_cell(row: str) -> str:
    return row.split("|")[2].strip()


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_combined_embeddings_row_names_derivatives_path(doc: Path) -> None:
    """REQ-MCP-OUTLOC-03: the Combined embeddings row names the combine stage's real output dir."""
    for row in _output_locations_rows(doc, "Combined embeddings"):
        assert f"`{COMBINED_PATH}`" in _path_cell(row), row


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_combined_sanity_check_names_derivatives_path(doc: Path) -> None:
    """REQ-MCP-OUTLOC-03: every combined-CSV sanity check lists the combine stage's real output dir."""
    checks = [line.strip() for line in doc.read_text().splitlines() if re.match(r"^\s*ls .*embeddings.*\*\.csv", line)]
    assert checks, f"{doc.relative_to(REPO_ROOT)}: no combined-CSV sanity check (`ls ...*.csv`)"
    for line in checks:
        assert line.startswith(f"ls {COMBINED_PATH}*.csv"), line


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_per_fold_embeddings_row_names_stage4_output(doc: Path) -> None:
    """REQ-MCP-OUTLOC-04: the Per-fold embeddings row names generate_embeddings.py's
    per-region output, not the models cache."""
    for row in _output_locations_rows(doc, "Per-fold embeddings"):
        cell = _path_cell(row)
        assert f"`{PER_FOLD_PATH}`" in cell, row
        assert "models_cache" not in cell, row


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_snapshots_row_names_derivatives_path(doc: Path) -> None:
    """REQ-MCP-SNAPOUT-02: the Snapshots row names the snapshots stage's real output dir."""
    for row in _output_locations_rows(doc, "Snapshots"):
        assert f"`{SNAPSHOTS_PATH}`" in _path_cell(row), row
