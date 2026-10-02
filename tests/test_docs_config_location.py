"""REQ-MCP-CFGLOC-02: MCP docs describe the TASK-141 config-location defaults.

Reference (champollion_pipeline 7adda32):
- ``generate_champollion_config.py``: default configs root
  ``<D>/<dataset>/derivatives/champollion_V1/configs`` (``<D>`` = parent of the
  ``<dataset>`` directory in ``crop_path``); ``--output`` *is* a configs root,
  region YAMLs land at ``{output}/dataset/{dataset}/``.
- ``train_champollion.py``: ``--config-dir`` defaults to
  ``<pipeline>/data/<dataset>/derivatives/champollion_V1/configs``.

Placeholders may be written ``<x>`` or ``{x}``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from champollion_sulcal_mcp.tools import stages

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent

SKILL = REPO_ROOT / "skills" / "run-pipeline" / "SKILL.md"
STAGE_PARAMS = REPO_ROOT / "skills" / "run-pipeline" / "references" / "stage-params.md"
AGENT = REPO_ROOT / "agents" / "champollion-pipeline.md"

ALL_DOCS = [SKILL, STAGE_PARAMS, AGENT]

STAGE3_DEFAULT = re.compile(r"[<{]D[>}]/[<{]dataset[>}]/derivatives/champollion_V1/configs")
TRAINING_DEFAULT = re.compile(r"[<{]pipeline(?:_dir)?[>}]/data/[<{]dataset[>}]/derivatives/champollion_V1/configs")
OUTPUT_PARAM = re.compile(r"`output`|--output\b")
CONFIG_DIR_PARAM = re.compile(r"config_dir|--config-dir")
CONFIGS_ROOT = re.compile(r"configs?\s+root", re.IGNORECASE)
DATASET_SUBDIR = re.compile(r"configs/dataset/")


def _ids(paths: list[Path]) -> list[str]:
    return [str(p.relative_to(REPO_ROOT)) for p in paths]


def _lines_matching(path: Path, pattern: re.Pattern[str]) -> list[str]:
    return [line for line in path.read_text().splitlines() if pattern.search(line)]


@pytest.mark.parametrize("doc", ALL_DOCS, ids=_ids(ALL_DOCS))
def test_stage3_default_configs_root_documented(doc: Path) -> None:
    """Each doc states the stage-3 default configs root <D>/<dataset>/derivatives/champollion_V1/configs."""
    assert STAGE3_DEFAULT.search(doc.read_text()), (
        f"{doc.name} does not state the stage-3 default <D>/<dataset>/derivatives/champollion_V1/configs"
    )


@pytest.mark.parametrize("doc", ALL_DOCS, ids=_ids(ALL_DOCS))
def test_output_described_as_configs_root(doc: Path) -> None:
    """Some line describing stage 3's `output` / --output (not start_training's config_dir) calls it a configs root."""
    lines = [line for line in _lines_matching(doc, OUTPUT_PARAM) if not CONFIG_DIR_PARAM.search(line)]
    assert any(CONFIGS_ROOT.search(line) for line in lines), (
        f"{doc.name} never describes `output` as a configs root: {lines}"
    )


@pytest.mark.parametrize("doc", ALL_DOCS, ids=_ids(ALL_DOCS))
def test_output_not_recommended_as_dataset_subdir(doc: Path) -> None:
    """No line gives `output` / --output a .../configs/dataset/<dataset> value (that is below the configs root)."""
    offending = [line for line in _lines_matching(doc, OUTPUT_PARAM) if DATASET_SUBDIR.search(line)]
    assert offending == [], f"{doc.name} still points `output` at a configs/dataset/ subdir: {offending}"


@pytest.mark.parametrize("doc", ALL_DOCS, ids=_ids(ALL_DOCS))
def test_training_config_dir_default_documented(doc: Path) -> None:
    """Some line describing start_training's config_dir states its <pipeline>/data/... default."""
    lines = _lines_matching(doc, CONFIG_DIR_PARAM)
    assert any(TRAINING_DEFAULT.search(line) for line in lines), (
        f"{doc.name} does not state config_dir's default "
        f"<pipeline>/data/<dataset>/derivatives/champollion_V1/configs: {lines}"
    )


@pytest.mark.parametrize("doc", ALL_DOCS, ids=_ids(ALL_DOCS))
def test_config_dir_not_tied_to_submodule(doc: Path) -> None:
    """No line describing config_dir still ties it to configs living outside the champollion_V1 submodule."""
    offending = [line for line in _lines_matching(doc, CONFIG_DIR_PARAM) if "submodule" in line.lower()]
    assert offending == [], f"{doc.name} still ties config_dir to the champollion_V1 submodule: {offending}"


def test_start_config_docstring_states_default_configs_root() -> None:
    """start_config's tool description states the stage-3 default configs root."""
    doc = stages.start_config.__doc__ or ""
    assert STAGE3_DEFAULT.search(doc), f"start_config docstring lacks the default configs root: {doc!r}"


def test_start_training_docstring_states_default_config_dir() -> None:
    """start_training's tool description states config_dir's default and drops the submodule wording."""
    doc = stages.start_training.__doc__ or ""
    assert TRAINING_DEFAULT.search(doc), f"start_training docstring lacks the config_dir default: {doc!r}"
    assert "submodule" not in doc.lower(), f"start_training docstring still mentions the submodule: {doc!r}"
