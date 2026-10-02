"""REQ-MCP-EMBCFG-01 / REQ-MCP-EMBCFG-02: stage-4 docs match generate_embeddings.py.

Reference (champollion_pipeline a41f8cd):
- No MCP tool (``start_embeddings``, ``start_pipeline``, ...) accepts ``config_path``,
  and ``generate_embeddings.py`` has no ``--config_path``: each model folder
  carries its own ``.hydra/config.yaml``.
- Stage 4 runs ``champollion_V1/champollion/evaluate.py`` per region; its
  ``load_model`` does ``glob(<model>/logs/lightning_logs/version_0/checkpoints/*.ckpt)[0]``,
  which raises ``IndexError: list index out of range`` when the model folder has
  neither a Lightning checkpoint nor ``logs/best_model_weights.pt`` (converted by
  ``_ensure_ckpt``). The ``skeleton_all[0].keys()`` frame belongs to
  ``create_datasets.py`` (training path), not stage 4.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent

DOC_DIRS = ("skills", "agents", "docs")
DOCS = sorted(p for d in DOC_DIRS for p in (REPO_ROOT / d).rglob("*.md")) + [REPO_ROOT / "README.md"]

ERROR_PATTERNS = REPO_ROOT / "skills" / "debug" / "references" / "error-patterns.md"

CKPT_GLOB = "logs/lightning_logs/version_0/checkpoints/"
WEIGHTS_FILE = "logs/best_model_weights.pt"
INDEX_ERROR = "IndexError: list index out of range"


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO_ROOT)) for p in paths]


def _stage4_section() -> str:
    """Text of the '## Stage 4' section of error-patterns.md, up to the next '## ' heading."""
    match = re.search(r"^## Stage 4\b.*?(?=^## |\Z)", ERROR_PATTERNS.read_text(), re.MULTILINE | re.DOTALL)
    assert match, "error-patterns.md has no '## Stage 4' section"
    return match.group(0)


def _stage4_entries_with(needle: str) -> list[str]:
    """'### ' entries of the Stage 4 section whose text contains needle."""
    entries = re.split(r"^(?=### )", _stage4_section(), flags=re.MULTILINE)
    return [entry for entry in entries if needle in entry]


@pytest.mark.parametrize("doc", DOCS, ids=_ids(DOCS))
def test_doc_does_not_mention_config_path(doc: Path) -> None:
    """No doc names a `config_path` parameter: no MCP tool or generate_embeddings.py option has one."""
    offending = [line for line in doc.read_text().splitlines() if "config_path" in line]
    assert offending == [], f"{doc.relative_to(REPO_ROOT)} still mentions config_path: {offending}"


def test_stage4_indexerror_attributed_to_missing_checkpoint() -> None:
    """The Stage 4 IndexError entry blames a model folder lacking both a .ckpt and best_model_weights.pt."""
    entries = _stage4_entries_with(INDEX_ERROR)
    assert entries, f"Stage 4 section of error-patterns.md has no '{INDEX_ERROR}' entry"
    assert any(CKPT_GLOB in entry and WEIGHTS_FILE in entry for entry in entries), (
        f"No Stage 4 '{INDEX_ERROR}' entry names both {CKPT_GLOB} and {WEIGHTS_FILE}: {entries}"
    )


def test_stage4_section_does_not_cite_create_datasets_frame() -> None:
    """The Stage 4 section does not show the training-path `skeleton_all[0]` frame."""
    section = _stage4_section()
    assert "skeleton_all[0]" not in section, "Stage 4 section still cites the create_datasets.py skeleton_all[0] frame"
