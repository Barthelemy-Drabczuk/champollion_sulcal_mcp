"""MCPEMBVER docs: the run-pipeline docs describe the versioned champollion_V1 layout.

start_pipeline (``tools/pipeline.py``, ``_compute_masks_version_dir``) now reads and
writes stages 4-6 under ``<datasets_root>/derivatives/champollion_V1/<masks or canonical_25>/``:
stage 4 (champollion_pipeline ``generate_embeddings.py`` default ``output``) writes
``region_embeddings/{region}/full_embeddings.csv``, stage 5 writes ``embeddings/``, stage 6
writes ``snapshots/``. A non-empty ``masks`` is forwarded to stages 2, 3 and 4.

Requirements (supersede REQ-MCPOUTLOC-BDRABCZUK-27B37BC0597A, -EFE8A15B1320 and
REQ-MCPSNAPOUT-BDRABCZUK-1CF725ADFF12):
- REQ-MCPEMBVER-BDRABCZUK-62DB632A8387: Combined embeddings row + combined-CSV sanity check.
- REQ-MCPEMBVER-BDRABCZUK-DF0A668F91B1: Per-fold embeddings row.
- REQ-MCPEMBVER-BDRABCZUK-4AD451D69961: Snapshots row.
- REQ-MCPEMBVER-BDRABCZUK-2126F306B416: ``{masks}`` placeholder defined in Output Locations.
- REQ-MCPEMBVER-BDRABCZUK-0DD8EB95BFDE: full-pipeline section states masks forwarding.
- REQ-MCPEMBVER-BDRABCZUK-F5E791F49719: crops row follows start_pipeline's masks.
- REQ-MCPEMBVER-BDRABCZUK-6D22EDDBD8C7: stage-4 ``output`` default is region_embeddings.
- REQ-MCPEMBVER-BDRABCZUK-1AD33EFD7F66: Default output structure tree nests under ``{masks}/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent

SKILL = REPO_ROOT / "skills" / "run-pipeline" / "SKILL.md"
AGENT = REPO_ROOT / "agents" / "champollion-pipeline.md"
STAGE_PARAMS = REPO_ROOT / "skills" / "run-pipeline" / "references" / "stage-params.md"

DOCS = [SKILL, AGENT]
DOC_IDS = [str(p.relative_to(REPO_ROOT)) for p in DOCS]
STAGE4_DOCS = [SKILL, STAGE_PARAMS, AGENT]
STAGE4_DOC_IDS = [str(p.relative_to(REPO_ROOT)) for p in STAGE4_DOCS]

VERSION_DIR = "{datasets_root}/derivatives/champollion_V1/{masks}/"
COMBINED_PATH = VERSION_DIR + "embeddings/"
REGION_EMBEDDINGS_PATH = VERSION_DIR + "region_embeddings/"
PER_FOLD_PATH = REGION_EMBEDDINGS_PATH + "{region}/full_embeddings.csv"
SNAPSHOTS_PATH = VERSION_DIR + "snapshots/"
OLD_STAGE4_DEFAULTS = ("{basename of datasets_root}embeddings", "TESTXXembeddings")


def _rel(doc: Path) -> str:
    return str(doc.relative_to(REPO_ROOT))


def _output_locations_sections(doc: Path) -> list[str]:
    """Every '## Output Locations' section of `doc` (up to the next '## ' heading)."""
    sections = re.findall(r"^## Output Locations\b.*?(?=^## |\Z)", doc.read_text(), re.MULTILINE | re.DOTALL)
    assert sections, f"{_rel(doc)} has no '## Output Locations' section"
    return sections


def _output_locations_rows(doc: Path, label: str) -> list[str]:
    """Table rows whose first cell matches the regex `label`, inside every Output Locations section."""
    rows = [
        line
        for section in _output_locations_sections(doc)
        for line in section.splitlines()
        if re.match(rf"^\|\s*(?:{label})\s*\|", line)
    ]
    assert rows, f"{_rel(doc)}: no '{label}' row in Output Locations"
    return rows


def _path_cell(row: str) -> str:
    return row.split("|")[2].strip()


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_combined_embeddings_row_names_versioned_path(doc: Path) -> None:
    """REQ-MCPEMBVER-BDRABCZUK-62DB632A8387: Combined embeddings row names the versioned embeddings dir."""
    for row in _output_locations_rows(doc, re.escape("Combined embeddings")):
        assert f"`{COMBINED_PATH}`" in _path_cell(row), row


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_combined_sanity_check_names_versioned_path(doc: Path) -> None:
    """REQ-MCPEMBVER-BDRABCZUK-62DB632A8387: every combined-CSV sanity check lists the versioned embeddings dir."""
    checks = [line.strip() for line in doc.read_text().splitlines() if re.match(r"^\s*ls .*embeddings.*\*\.csv", line)]
    assert checks, f"{_rel(doc)}: no combined-CSV sanity check (`ls ...*.csv`)"
    for line in checks:
        assert line.startswith(f"ls {COMBINED_PATH}*.csv"), line


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_per_fold_embeddings_row_names_region_embeddings_path(doc: Path) -> None:
    """REQ-MCPEMBVER-BDRABCZUK-DF0A668F91B1: Per-fold embeddings row names the versioned region_embeddings dir."""
    for row in _output_locations_rows(doc, re.escape("Per-fold embeddings")):
        cell = _path_cell(row)
        assert f"`{PER_FOLD_PATH}`" in cell, row
        assert "models_cache" not in cell, row


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_snapshots_row_names_versioned_path(doc: Path) -> None:
    """REQ-MCPEMBVER-BDRABCZUK-4AD451D69961: Snapshots row names the versioned snapshots dir."""
    for row in _output_locations_rows(doc, re.escape("Snapshots")):
        assert f"`{SNAPSHOTS_PATH}`" in _path_cell(row), row


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_output_locations_defines_masks_placeholder(doc: Path) -> None:
    """REQ-MCPEMBVER-BDRABCZUK-2126F306B416: Output Locations defines `{masks}` as start_pipeline's
    `masks` value, `canonical_25` when unset or empty.

    Some single line of each Output Locations section must name `{masks}`, `masks`,
    `start_pipeline` and `canonical_25` together.
    """
    for section in _output_locations_sections(doc):
        definitions = [
            line
            for line in section.splitlines()
            if "`{masks}`" in line and "`masks`" in line and "`start_pipeline`" in line and "`canonical_25`" in line
        ]
        assert definitions, f"{_rel(doc)}: Output Locations never defines `{{masks}}`"


def _full_pipeline_section() -> str:
    match = re.search(r"^### Full pipeline\b.*?(?=^#{2,3} |\Z)", SKILL.read_text(), re.MULTILINE | re.DOTALL)
    assert match, "SKILL.md has no '### Full pipeline' section"
    return match.group(0)


def test_full_pipeline_section_states_masks_forwarded_to_stages_2_3_4() -> None:
    """REQ-MCPEMBVER-BDRABCZUK-0DD8EB95BFDE: the full-pipeline section states that `start_pipeline`
    forwards a non-empty `masks` to stages 2, 3 and 4."""
    section = _full_pipeline_section()
    sentences = [s for s in re.split(r"(?<=\.)\s+|\n\n", section) if "`masks`" in s and "`start_pipeline`" in s]
    assert any(re.search(r"stages 2, 3,? and 4", s) for s in sentences), sentences
    assert "Stage 2 always writes the default crops" not in section


CROPS_LABEL = r"Sulcal region crops|Cortical crops"


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_crops_row_states_folder_follows_pipeline_masks(doc: Path) -> None:
    """REQ-MCPEMBVER-BDRABCZUK-F5E791F49719: the crops row says the folder follows a non-empty
    `masks` passed to `start_pipeline` (no longer that `start_pipeline` never sets it)."""
    for row in _output_locations_rows(doc, CROPS_LABEL):
        assert "never sets" not in row, row
        assert "`start_pipeline`" in row, row
        assert "`masks`" in row, row


@pytest.mark.parametrize("doc", STAGE4_DOCS, ids=STAGE4_DOC_IDS)
def test_stage4_output_default_is_region_embeddings(doc: Path) -> None:
    """REQ-MCPEMBVER-BDRABCZUK-6D22EDDBD8C7: the stage-4 `output` default is the versioned
    region_embeddings dir, and the old `{basename}embeddings` default is gone."""
    text = doc.read_text()
    assert REGION_EMBEDDINGS_PATH in text, f"{_rel(doc)} never names `{REGION_EMBEDDINGS_PATH}`"
    for old in OLD_STAGE4_DEFAULTS:
        stale = [line.strip() for line in text.splitlines() if old in line]
        assert not stale, stale


def _default_output_tree(doc: Path) -> list[str]:
    """Lines of the code block that follows '**Default output structure**'."""
    match = re.search(r"\*\*Default output structure\*\*.*?```\n(.*?)```", doc.read_text(), re.DOTALL)
    assert match, f"{_rel(doc)}: no Default output structure tree"
    return match.group(1).splitlines()


def _entry_column(line: str, name: str) -> int | None:
    """Column of tree entry `name` in `line` (after a '── ' connector), else None."""
    match = re.search(rf"── {re.escape(name)}(?:\s|$)", line)
    return match.start() + 3 if match else None


@pytest.mark.parametrize("doc", DOCS, ids=DOC_IDS)
def test_default_output_tree_nests_outputs_under_masks(doc: Path) -> None:
    """REQ-MCPEMBVER-BDRABCZUK-1AD33EFD7F66: the tree shows region_embeddings/, embeddings/ and
    snapshots/ inside a `{masks}/` directory under champollion_V1/."""
    lines = _default_output_tree(doc)
    v1 = [(i, _entry_column(line, "champollion_V1/")) for i, line in enumerate(lines)]
    v1 = [(i, col) for i, col in v1 if col is not None]
    assert v1, f"{_rel(doc)}: tree has no champollion_V1/ entry"
    v1_index, v1_col = v1[0]
    masks = [
        (i, col)
        for i, line in enumerate(lines)
        if i > v1_index and (col := _entry_column(line, "{masks}/")) is not None and col > v1_col
    ]
    assert masks, f"{_rel(doc)}: no {{masks}}/ directory under champollion_V1/ in the tree"
    masks_index, masks_col = masks[0]
    children = lines[masks_index + 1 :]
    for name in ("region_embeddings/", "embeddings/", "snapshots/"):
        cols = [col for line in children if (col := _entry_column(line, name)) is not None]
        assert any(col > masks_col for col in cols), f"{_rel(doc)}: {name} not nested under {{masks}}/"
