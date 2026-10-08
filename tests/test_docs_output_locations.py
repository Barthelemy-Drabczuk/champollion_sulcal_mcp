"""REQ-MCP-OUTDIR-01..07: output-path docs match the tools.

The Per-fold / Combined embeddings and Snapshots rows and the combined-CSV sanity
check (REQ-MCP-OUTLOC-03 / -04, REQ-MCP-SNAPOUT-02) are checked by
tests/test_mcpembver_docs_layout.py against their superseding REQ-MCPEMBVER
requirements (versioned ``{datasets_root}/derivatives/champollion_V1/{masks}/`` tree).

``output_dir`` semantics per tool (champollion_sulcal_mcp TASK-051):
- ``start_pipeline``: dataset root R. It passes R unchanged to
  ``start_morphologist`` and ``R/derivatives`` to ``start_cortical_tiles``.
- ``start_morphologist``: dataset root; ``morphologist-cli`` appends
  ``derivatives/morphologist-6.0/subjects/``.
- ``start_cortical_tiles``: derivatives dir; crops land in
  ``{output_dir}/cortical_tiles-2026/crops/{masks}/2mm/``
  (``masks`` defaults to ``canonical_25``; start_pipeline forwards a non-empty ``masks``).
- ``start_config`` (no ``output``): ``<D>/<dataset>/derivatives/champollion_V1/configs``,
  i.e. ``R/derivatives/...`` under the ``./data/{dataset}/`` layout.
- ``start_snapshots``: images are written straight into ``output_dir``.
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

MORPHOLOGIST_PATH = "{output_dir}/derivatives/morphologist-*/subjects/"
CROPS_PATH = "{output_dir}/derivatives/cortical_tiles-*/crops/canonical_25/2mm/"
CONFIG_PATH = "{output_dir}/derivatives/champollion_V1/configs/dataset/{dataset}/"
CROPS_SANITY_CHECK = "ls {output_dir}/derivatives/cortical_tiles-*/crops/canonical_25/2mm | wc -l"
STAGE_PARAMS = REPO_ROOT / "skills" / "run-pipeline" / "references" / "stage-params.md"


def _output_locations_rows(doc: Path, label: str) -> list[str]:
    """Table rows whose first cell matches the regex `label`, inside every '## Output Locations' section."""
    sections = re.findall(r"^## Output Locations\b.*?(?=^## |\Z)", doc.read_text(), re.MULTILINE | re.DOTALL)
    assert sections, f"{doc.relative_to(REPO_ROOT)} has no '## Output Locations' section"
    rows = [line for section in sections for line in section.splitlines() if re.match(rf"^\|\s*(?:{label})\s*\|", line)]
    assert rows, f"{doc.relative_to(REPO_ROOT)}: no '{label}' row in Output Locations"
    return rows


def _path_cell(row: str) -> str:
    return row.split("|")[2].strip()


# Per-doc first-cell labels of the Output Locations rows (regex alternatives).
MORPHOLOGIST_LABEL = r"Morphologist(?: graphs)?"
CROPS_LABEL = r"Sulcal region crops|Cortical crops"
CONFIG_LABEL = r"Champollion config|Config"


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_morphologist_row_names_derivatives_path(doc: Path) -> None:
    """REQ-MCP-OUTDIR-01: the Morphologist row names morphologist-cli's real output under the dataset root."""
    for row in _output_locations_rows(doc, MORPHOLOGIST_LABEL):
        assert f"`{MORPHOLOGIST_PATH}`" in _path_cell(row), row


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_crops_row_names_derivatives_path(doc: Path) -> None:
    """REQ-MCP-OUTDIR-02: the crops row names start_pipeline's real crops dir (derivatives/, canonical_25)."""
    for row in _output_locations_rows(doc, CROPS_LABEL):
        assert f"`{CROPS_PATH}`" in _path_cell(row), row


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_config_row_names_derivatives_path(doc: Path) -> None:
    """REQ-MCP-OUTDIR-03: the config row names start_config's default configs root under derivatives/."""
    for row in _output_locations_rows(doc, CONFIG_LABEL):
        assert f"`{CONFIG_PATH}`" in _path_cell(row), row


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_crops_sanity_check_names_derivatives_path(doc: Path) -> None:
    """REQ-MCP-OUTDIR-04: the region-folder-count sanity check lists the real crops dir."""
    checks = [line.strip() for line in doc.read_text().splitlines() if re.match(r"^\s*ls .*cortical_tiles", line)]
    assert checks, f"{doc.relative_to(REPO_ROOT)}: no region-folder sanity check (`ls ...cortical_tiles...`)"
    for line in checks:
        assert line == CROPS_SANITY_CHECK, line


def _default_output_dir_block(doc: Path) -> str:
    """Text between '**Default output structure**' and '**Default `path_to_graph`**'."""
    match = re.search(
        r"\*\*Default output structure\*\*(.*?)\*\*Default `path_to_graph`\*\*", doc.read_text(), re.DOTALL
    )
    assert match, f"{doc.relative_to(REPO_ROOT)}: no default output_dir guidance block"
    return match.group(1)


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_default_output_dir_guidance_not_for_all_stages(doc: Path) -> None:
    """REQ-MCP-OUTDIR-05: no single default `output_dir` is recommended for all stages."""
    assert "for all stages" not in doc.read_text(), doc.relative_to(REPO_ROOT)


DATASET_ROOT = "`./data/{dataset}/`"
DERIVATIVES_DIR = "`./data/{dataset}/derivatives/`"
SNAPSHOTS_DIR = "`./data/{dataset}/derivatives/champollion_V1/snapshots/`"
DEFAULT_OUTPUT_DIRS = {
    "start_pipeline": DATASET_ROOT,
    "start_morphologist": DATASET_ROOT,
    "start_cortical_tiles": DERIVATIVES_DIR,
    "start_snapshots": SNAPSHOTS_DIR,
}


@pytest.mark.parametrize("tool", sorted(DEFAULT_OUTPUT_DIRS))
@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_default_output_dir_guidance_per_tool(doc: Path, tool: str) -> None:
    """REQ-MCP-OUTDIR-05: the guidance pairs each tool with its own default `output_dir`.

    The guidance block is split into clauses (newlines, ';', ', ', '. '); some clause must
    name the tool and its value without naming another tool's distinct value.
    """
    expected = DEFAULT_OUTPUT_DIRS[tool]
    others = {value for value in DEFAULT_OUTPUT_DIRS.values() if value != expected}
    clauses = [c for c in re.split(r"\n|;|,\s|\.\s", _default_output_dir_block(doc)) if f"`{tool}`" in c]
    assert clauses, f"{doc.relative_to(REPO_ROOT)}: default output_dir guidance never names `{tool}`"
    assert any(expected in c and not any(o in c for o in others) for c in clauses), (tool, expected, clauses)


MORPHOLOGIST_DOCS = [*DOCS, STAGE_PARAMS]
MORPHOLOGIST_DOC_IDS = [str(p.relative_to(REPO_ROOT)) for p in MORPHOLOGIST_DOCS]


@pytest.mark.parametrize("doc", MORPHOLOGIST_DOCS, ids=MORPHOLOGIST_DOC_IDS)
def test_morphologist_paths_rooted_at_output_dir_use_derivatives(doc: Path) -> None:
    """REQ-MCP-OUTDIR-06: every `{output_dir}/...morphologist-` path goes through derivatives/."""
    paths = re.findall(r"\{output_dir\}/[^`\s|]*?morphologist-", doc.read_text())
    assert paths, f"{doc.relative_to(REPO_ROOT)}: no Morphologist path rooted at {{output_dir}}"
    for path in paths:
        assert path == "{output_dir}/derivatives/morphologist-", path


def test_stage_params_morphologist_output_dir_is_dataset_root() -> None:
    """REQ-MCP-OUTDIR-07: stage-params' start_morphologist output_dir row calls it the dataset root."""
    section = re.search(
        r"^## Stage 1 — `start_morphologist`.*?(?=^## |\Z)", STAGE_PARAMS.read_text(), re.MULTILINE | re.DOTALL
    )
    assert section, "stage-params.md has no Stage 1 start_morphologist section"
    rows = [line for line in section.group(0).splitlines() if line.startswith("| `output_dir` |")]
    assert len(rows) == 1, rows
    assert "dataset root" in rows[0], rows[0]
    assert "`derivatives/morphologist-{version}/subjects/`" in rows[0], rows[0]
