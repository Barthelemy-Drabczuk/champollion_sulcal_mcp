"""REQ-MCP-OUTLOC-03 / -04, REQ-MCP-SNAPOUT-02, REQ-MCP-OUTDIR-01..07: output-path docs match the tools.

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

``output_dir`` semantics per tool (champollion_sulcal_mcp TASK-051):
- ``start_pipeline``: dataset root R. It passes R unchanged to
  ``start_morphologist`` and ``R/derivatives`` to ``start_cortical_tiles``.
- ``start_morphologist``: dataset root; ``morphologist-cli`` appends
  ``derivatives/morphologist-6.0/subjects/``.
- ``start_cortical_tiles``: derivatives dir; crops land in
  ``{output_dir}/cortical_tiles-2026/crops/{masks_version}/2mm/``
  (``masks_version`` defaults to ``canonical_25``; start_pipeline never sets it).
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

COMBINED_PATH = "{output_dir}/derivatives/champollion_V1/embeddings/"
SNAPSHOTS_PATH = "{output_dir}/derivatives/champollion_V1/snapshots/"
MORPHOLOGIST_PATH = "{output_dir}/derivatives/morphologist-*/subjects/"
CROPS_PATH = "{output_dir}/derivatives/cortical_tiles-*/crops/canonical_25/2mm/"
CONFIG_PATH = "{output_dir}/derivatives/champollion_V1/configs/dataset/{dataset}/"
CROPS_SANITY_CHECK = "ls {output_dir}/derivatives/cortical_tiles-*/crops/canonical_25/2mm | wc -l"
STAGE_PARAMS = REPO_ROOT / "skills" / "run-pipeline" / "references" / "stage-params.md"
PER_FOLD_PATH = "{parent of datasets_root}/{basename of datasets_root}embeddings/{region}/full_embeddings.csv"


def _output_locations_rows(doc: Path, label: str) -> list[str]:
    """Table rows whose first cell matches the regex `label`, inside every '## Output Locations' section."""
    sections = re.findall(r"^## Output Locations\b.*?(?=^## |\Z)", doc.read_text(), re.MULTILINE | re.DOTALL)
    assert sections, f"{doc.relative_to(REPO_ROOT)} has no '## Output Locations' section"
    rows = [line for section in sections for line in section.splitlines() if re.match(rf"^\|\s*(?:{label})\s*\|", line)]
    assert rows, f"{doc.relative_to(REPO_ROOT)}: no '{label}' row in Output Locations"
    return rows


def _path_cell(row: str) -> str:
    return row.split("|")[2].strip()


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_combined_embeddings_row_names_derivatives_path(doc: Path) -> None:
    """REQ-MCP-OUTLOC-03: the Combined embeddings row names the combine stage's real output dir."""
    for row in _output_locations_rows(doc, re.escape("Combined embeddings")):
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
    for row in _output_locations_rows(doc, re.escape("Per-fold embeddings")):
        cell = _path_cell(row)
        assert f"`{PER_FOLD_PATH}`" in cell, row
        assert "models_cache" not in cell, row


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_snapshots_row_names_derivatives_path(doc: Path) -> None:
    """REQ-MCP-SNAPOUT-02: the Snapshots row names the snapshots stage's real output dir."""
    for row in _output_locations_rows(doc, re.escape("Snapshots")):
        assert f"`{SNAPSHOTS_PATH}`" in _path_cell(row), row


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
