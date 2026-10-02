"""REQ-MCP-PARAMS-01 / -02 / -03: documented MCP tool parameters match the tool signatures.

The source of truth is each tool function's own signature
(``src/champollion_sulcal_mcp/tools/stages.py``, ``tools/pipeline.py``), minus
the injected ``ctx``. The docs checked here are:

- ``skills/run-pipeline/references/stage-params.md`` — one ``## ... — `start_*` ``
  section per tool, whose ``Parameter`` table must list exactly the tool's
  parameters (REQ-MCP-PARAMS-01).
- ``skills/run-pipeline/SKILL.md`` and ``agents/champollion-pipeline.md`` —
  per-stage parameter tables/lists, the stage-parallelism table and the
  ``start_*(...)`` call examples may only name real tool parameters
  (REQ-MCP-PARAMS-02); the ``start_pipeline(...)`` example must pass every
  parameter ``start_pipeline`` requires (REQ-MCP-PARAMS-03).

CLI-only notes and the agent's "CLI Reference" command blocks describe the
pipeline scripts, not the MCP tools, and are deliberately not parsed.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

import pytest

from champollion_sulcal_mcp.tools import pipeline, stages

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_PARAMS = REPO_ROOT / "skills" / "run-pipeline" / "references" / "stage-params.md"
SKILL = REPO_ROOT / "skills" / "run-pipeline" / "SKILL.md"
AGENT = REPO_ROOT / "agents" / "champollion-pipeline.md"

TOOLS = {
    name: fn
    for module in (stages, pipeline)
    for name, fn in vars(module).items()
    if name.startswith("start_") and inspect.iscoroutinefunction(fn)
}

# Tools that stage-params.md documents with their own section (start_pipeline is
# documented in SKILL.md instead).
STAGE_PARAMS_TOOLS = sorted(
    [
        "start_morphologist",
        "start_cortical_tiles",
        "start_config",
        "start_training",
        "start_embeddings",
        "start_combine",
        "start_snapshots",
        "start_streaming",
    ]
)

# Heading prefix of a per-stage section (SKILL.md "### ...") or bullet label
# (agent "- **...**") -> tool it documents.
STAGE_LABELS = {
    "Stage 1": "start_morphologist",
    "Stage 2": "start_cortical_tiles",
    "Stage 3": "start_config",
    "Training": "start_training",
    "Stage 4": "start_embeddings",
    "Stage 5": "start_combine",
    "Stage 6": "start_snapshots",
    "Streaming Pipeline": "start_streaming",
}

# Stage number in SKILL.md's parallelism table -> tool.
STAGE_NUMBERS = {
    "1": "start_morphologist",
    "2": "start_cortical_tiles",
    "3": "start_config",
    "4": "start_embeddings",
    "5": "start_combine",
    "6": "start_snapshots",
}

IDENTIFIER = re.compile(r"`([a-z][a-z0-9_]*)`")
# Same, but skipping explicit cross-references to another stage's parameter
# (e.g. "the stage-3 `output` configs root" inside the training bullet).
OWN_IDENTIFIER = re.compile(r"(?<!stage-\d )`([a-z][a-z0-9_]*)`")


def _params(tool: str) -> set[str]:
    """Parameter names of a tool function, excluding the injected ``ctx``."""
    return {name for name in inspect.signature(TOOLS[tool]).parameters if name != "ctx"}


def _required_params(tool: str) -> set[str]:
    """Parameters of a tool function that have no default value."""
    return {
        name
        for name, p in inspect.signature(TOOLS[tool]).parameters.items()
        if name != "ctx" and p.default is inspect.Parameter.empty
    }


def _tables(text: str) -> list[list[list[str]]]:
    """Markdown tables in text, each a list of rows (lists of stripped cells), separator rows dropped."""
    tables: list[list[list[str]]] = []
    current: list[list[str]] = []
    for line in [*text.splitlines(), ""]:
        if line.lstrip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                current.append(cells)
        elif current:
            tables.append(current)
            current = []
    return tables


def _parameter_column_names(text: str) -> list[str]:
    """Backticked identifiers in the 'Parameter' column of every table in text."""
    names: list[str] = []
    for table in _tables(text):
        header = table[0]
        if "Parameter" not in header:
            continue
        col = header.index("Parameter")
        for row in table[1:]:
            if col < len(row):
                match = re.fullmatch(IDENTIFIER, row[col])
                if match:
                    names.append(match.group(1))
    return names


def _section(text: str, heading: str) -> str:
    """Body of the first markdown section whose heading line matches the regex `heading`."""
    match = re.search(rf"^{heading}.*?(?=^#{{2,3}} |\Z)", text, re.MULTILINE | re.DOTALL)
    assert match, f"no section matching {heading!r}"
    return match.group(0)


def _stage_params_sections() -> dict[str, str]:
    """stage-params.md '## ... — `start_*`' sections, keyed by tool name."""
    text = STAGE_PARAMS.read_text()
    return {
        m.group(1): m.group(0)
        for m in re.finditer(r"^## [^\n]*`(start_\w+)`[^\n]*\n.*?(?=^## |\Z)", text, re.MULTILINE | re.DOTALL)
    }


def _call_examples(text: str) -> dict[str, list[str]]:
    """Keyword names passed in each ``start_*(...)`` call example, keyed by tool name."""
    examples: dict[str, list[str]] = {}
    for m in re.finditer(r"^(start_\w+)\(\n(.*?)^\)", text, re.MULTILINE | re.DOTALL):
        examples.setdefault(m.group(1), []).extend(re.findall(r"^\s*(\w+)=", m.group(2), re.MULTILINE))
    return examples


def _agent_stage_bullets() -> list[tuple[str, list[str]]]:
    """(tool, identifiers) per required-parameter bullet in the agent's 'Gather parameters' step."""
    text = AGENT.read_text()
    stage_block = re.search(
        r"^For \*\*stage-centric\*\* runs.*?\n(.*?)^For \*\*streaming\*\*", text, re.MULTILINE | re.DOTALL
    )
    streaming_block = re.search(r"^For \*\*streaming\*\* runs.*?\n(.*?)(?=^\s*$)", text, re.MULTILINE | re.DOTALL)
    assert stage_block and streaming_block, "agent doc lost its per-stage / streaming parameter lists"

    bullets: list[tuple[str, list[str]]] = []
    for line in stage_block.group(1).splitlines():
        m = re.match(r"^- \*\*(.+?)\*\*:?(.*)$", line)
        if not m:
            continue
        tool = next((t for label, t in STAGE_LABELS.items() if m.group(1).startswith(label)), None)
        assert tool, f"agent bullet with unknown stage label: {line!r}"
        bullets.append((tool, OWN_IDENTIFIER.findall(m.group(2))))
    for line in streaming_block.group(1).splitlines():
        if line.startswith("- "):
            bullets.append(("start_streaming", OWN_IDENTIFIER.findall(line)))
    return bullets


# --- REQ-MCP-PARAMS-01: stage-params.md tables equal the tool signatures ---------------------


def test_stage_params_documents_expected_tools() -> None:
    """Guard: stage-params.md keeps one section per stage tool, so the table check below is not vacuous."""
    assert sorted(_stage_params_sections()) == STAGE_PARAMS_TOOLS


@pytest.mark.parametrize("tool", STAGE_PARAMS_TOOLS)
def test_stage_params_table_matches_signature(tool: str) -> None:
    """The tool's stage-params.md Parameter table lists exactly its signature parameters (minus ctx)."""
    section = _stage_params_sections().get(tool)
    assert section is not None, f"stage-params.md has no section for {tool}"
    documented = set(_parameter_column_names(section))
    actual = _params(tool)
    assert documented == actual, (
        f"{tool}: documented-but-not-accepted={sorted(documented - actual)}, "
        f"accepted-but-undocumented={sorted(actual - documented)}"
    )


# --- REQ-MCP-PARAMS-02: SKILL.md / agent doc name only real tool parameters ------------------


@pytest.mark.parametrize("label", list(STAGE_LABELS), ids=[STAGE_LABELS[k] for k in STAGE_LABELS])
def test_skill_stage_table_params_exist(label: str) -> None:
    """Each parameter in a SKILL.md per-stage table is a parameter of that stage's tool."""
    tool = STAGE_LABELS[label]
    names = _parameter_column_names(_section(SKILL.read_text(), rf"### {re.escape(label)}\b"))
    assert names, f"SKILL.md '{label}' section has no parameter table"
    unknown = sorted(set(names) - _params(tool))
    assert unknown == [], f"SKILL.md '{label}' documents parameters {tool} does not accept: {unknown}"


def test_skill_parallelism_table_params_exist() -> None:
    """Each `name=` in SKILL.md's stage-parallelism table is a parameter of that row's stage tool."""
    tables = [t for t in _tables(SKILL.read_text()) if "Parallel option" in t[0]]
    assert len(tables) == 1, "SKILL.md must have exactly one stage-parallelism table"
    unknown: dict[str, list[str]] = {}
    for row in tables[0][1:]:
        number = re.match(r"(\d)", row[0])
        assert number, f"unexpected parallelism row {row!r}"
        tool = STAGE_NUMBERS[number.group(1)]
        names = re.findall(r"`(\w+)=", " | ".join(row[1:]))
        bad = sorted(set(names) - _params(tool))
        if bad:
            unknown[tool] = bad
    assert unknown == {}, f"parallelism table names parameters the tools do not accept: {unknown}"


@pytest.mark.parametrize("tool", ["start_pipeline", "start_streaming"])
def test_skill_call_example_params_exist(tool: str) -> None:
    """Each keyword in SKILL.md's start_*(...) call example is a parameter of that tool."""
    examples = _call_examples(SKILL.read_text())
    assert tool in examples, f"SKILL.md has no {tool}(...) call example"
    unknown = sorted(set(examples[tool]) - _params(tool))
    assert unknown == [], f"SKILL.md {tool}(...) example passes parameters it does not accept: {unknown}"


@pytest.mark.parametrize("tool", sorted(set(STAGE_LABELS.values())))
def test_agent_stage_list_params_exist(tool: str) -> None:
    """Each parameter in the agent's per-stage required-parameter lists belongs to that stage's tool."""
    bullets = [names for t, names in _agent_stage_bullets() if t == tool]
    assert bullets, f"agent doc has no parameter list for {tool}"
    unknown = sorted({n for names in bullets for n in names} - _params(tool))
    assert unknown == [], f"agent doc lists parameters {tool} does not accept: {unknown}"


# --- REQ-MCP-PARAMS-03: the start_pipeline example is callable --------------------------------


def test_skill_pipeline_example_passes_required_params() -> None:
    """SKILL.md's start_pipeline(...) example passes every parameter start_pipeline requires."""
    examples = _call_examples(SKILL.read_text())
    assert "start_pipeline" in examples, "SKILL.md has no start_pipeline(...) call example"
    missing = sorted(_required_params("start_pipeline") - set(examples["start_pipeline"]))
    assert missing == [], f"SKILL.md start_pipeline(...) example omits required parameters: {missing}"
