"""REQ-MCP-DOCS-01: MCP docs match run_cortical_tiles.py stage-2 CLI defaults.

Reference: champollion_pipeline/src/champollion_pipeline/run_cortical_tiles.py
- ``--input-types`` defaults to skeleton foldlabel (extremities only on request)
- ``--with-distbottom`` is opt-in (off by default)
- ``--skip-distbottom`` is a deprecated no-op kept for backward compatibility
"""

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent

SKILL = REPO_ROOT / "skills" / "run-pipeline" / "SKILL.md"
STAGE_PARAMS = REPO_ROOT / "skills" / "run-pipeline" / "references" / "stage-params.md"
AGENT = REPO_ROOT / "agents" / "champollion-pipeline.md"

ALL_DOCS = [SKILL, STAGE_PARAMS, AGENT]
CLI_ONLY_NOTE_DOCS = [STAGE_PARAMS, AGENT]


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO_ROOT)) for p in paths]


def _lines_mentioning(path: Path, needle: str) -> list[str]:
    return [line for line in path.read_text().splitlines() if needle in line]


@pytest.mark.parametrize("doc", ALL_DOCS, ids=_ids(ALL_DOCS))
def test_input_types_default_not_all(doc: Path) -> None:
    """No line describing --input-types claims its default is 'all'."""
    offending = [line for line in _lines_mentioning(doc, "--input-types") if "default: all" in line.lower()]
    assert offending == [], f"{doc.name} still says --input-types defaults to all: {offending}"


def test_skill_states_input_types_default_skeleton_foldlabel() -> None:
    """SKILL.md's --input-types entry states the skeleton foldlabel default."""
    lines = _lines_mentioning(SKILL, "--input-types")
    assert lines, "SKILL.md no longer documents --input-types"
    pattern = re.compile(r"default\W*skeleton\W+foldlabel", re.IGNORECASE)
    assert any(pattern.search(line) for line in lines), (
        f"SKILL.md --input-types entry does not state 'default ... skeleton foldlabel': {lines}"
    )


@pytest.mark.parametrize("doc", ALL_DOCS, ids=_ids(ALL_DOCS))
def test_skip_distbottom_not_recommended(doc: Path) -> None:
    """No doc recommends enabling --skip-distbottom."""
    offending = [line for line in _lines_mentioning(doc, "--skip-distbottom") if "recommend" in line.lower()]
    assert offending == [], f"{doc.name} still recommends --skip-distbottom: {offending}"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=_ids(ALL_DOCS))
def test_skip_distbottom_described_as_deprecated(doc: Path) -> None:
    """Every line mentioning --skip-distbottom calls it deprecated."""
    offending = [line for line in _lines_mentioning(doc, "--skip-distbottom") if "deprecated" not in line.lower()]
    assert offending == [], f"{doc.name} mentions --skip-distbottom without marking it deprecated: {offending}"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=_ids(ALL_DOCS))
def test_with_distbottom_mentioned(doc: Path) -> None:
    """Each doc mentions the opt-in --with-distbottom flag."""
    assert "--with-distbottom" in doc.read_text(), f"{doc.name} does not mention --with-distbottom"


@pytest.mark.parametrize("doc", CLI_ONLY_NOTE_DOCS, ids=_ids(CLI_ONLY_NOTE_DOCS))
def test_cli_only_note_lists_with_distbottom(doc: Path) -> None:
    """The 'CLI-only flags' note lists --with-distbottom and --input-types."""
    notes = _lines_mentioning(doc, "CLI-only flags")
    assert notes, f"{doc.name} has no 'CLI-only flags' note"
    assert any("--with-distbottom" in n and "--input-types" in n for n in notes), (
        f"{doc.name} CLI-only note does not list --with-distbottom and --input-types: {notes}"
    )
