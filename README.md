# champollion-sulcal-mcp

A FastMCP server wrapping the Champollion sulcal embedding pipeline — exposes each pipeline stage as an MCP tool so an agent can run, monitor, and debug the pipeline without shelling out manually.

## Overview

This is a **stdio MCP server** (not a web service). It doesn't run the pipeline in-process — each `start_*` tool launches one of the pipeline's CLI scripts as a subprocess and tracks it as a background job (status + log file on disk), which callers poll until it reaches a terminal state.

It's consumed by the `technician` agent in the sibling [`champollion_agents`](../champollion_agents) repo, and also ships its own Claude Code subagent definition and skills so it can be driven directly from Claude Code.

## How it works

1. A caller (an MCP client, or the Technician agent via `claude-code-sdk`) invokes a tool such as `start_cortical_tiles`.
2. The tool validates its arguments (paths must be absolute), resolves the pipeline's script location via `preflight.detect()`, and builds an `argv` matching that script's real CLI.
3. `runner.launch()` spawns the script as a subprocess with an environment built by `_build_env()` (selectively passes `HF_TOKEN`, injects `BRAINVISA_SHARE`, prepends BrainVISA/pixi bins to `PATH`), and returns immediately with a `job_id`.
4. The subprocess's combined stdout/stderr is streamed line-by-line into a log file; progress lines matching `fold N/M` update the job's progress.
5. The caller polls `get_job_status(output_dir, job_id)` (and can tail `get_job_log`) until the job is `succeeded`, `failed`, or `cancelled`.

## Pipeline stages

| Stage | Tool | Purpose |
|---|---|---|
| 1 | `start_morphologist` | Generate sulcal graphs from T1 MRI using Morphologist |
| 2 | `start_cortical_tiles` | Extract 28 standardized sulcal region crops |
| 3 | `start_config` | Generate Champollion dataset YAML configuration |
| 4 | `start_embeddings` | Compute 56-fold sulcal embeddings (28 regions × 2 hemispheres) |
| 5 | `start_combine` | Collect per-region embedding CSVs into a single output directory |
| 6 | `start_snapshots` | Render sulcal graph meshes, cortical tile masks, and UMAP plots |
| — | `start_pipeline` | Run stages 1–6 sequentially as one umbrella job (stages skippable) |
| — | `start_streaming` | Scan-centric mode: one worker per scan runs stages 2–4 in parallel; combine runs once after all workers drain |
| — | `start_training` | Train a self-supervised `champollion_V1` encoder for one region |

## Project layout

```
src/champollion_sulcal_mcp/
├── server.py            # FastMCP instance, tool registration, entry point (main())
├── preflight.py          # locates the champollion_pipeline repo + its scripts/submodules
├── job_store.py           # JobState/JobProgress models, JSON job file persistence
├── runner.py               # subprocess launch, log streaming, progress parsing, cancel
└── tools/
    ├── stages.py           # one start_<stage> tool per pipeline stage + maintenance tools
    ├── pipeline.py          # start_pipeline composite/umbrella job orchestration
    ├── jobs.py               # get_job_status, list_jobs, cancel_job, get_job_log
    └── utils.py               # get_pipeline_info, preflight_check
agents/
└── champollion-pipeline.md  # Claude Code subagent definition for this MCP server
skills/
├── run-pipeline/             # SKILL.md guiding stage-centric vs streaming execution
├── monitor/                   # SKILL.md for the job-polling loop
└── debug/                      # SKILL.md + known error patterns for failure diagnosis
docs/
└── agents_architecture.md      # early design doc for the champollion_agents repo (historical, superseded — see note below)
tests/                           # pytest suite with fake_pipeline_dir / recording_runner fixtures
```

## MCP tools reference

Registered in `server.py`:

**Stage launchers** — `start_morphologist`, `start_cortical_tiles`, `start_config`, `start_training`, `start_embeddings`, `start_combine`, `start_snapshots`

**Composite pipeline** — `start_pipeline`

**Streaming** — `start_streaming`

**Maintenance** — `purge_subject` (remove a subject's cortical_tiles derivatives), `prune_failed_subjects` (remove outputs for subjects that failed QC)

**Job lifecycle** — `get_job_status`, `list_jobs`, `cancel_job`, `get_job_log`

**Utilities** — `get_pipeline_info` (server/stage metadata), `preflight_check` (verify the pipeline is correctly configured and accessible)

## Requirements

- Python 3.11 or 3.12
- [pixi](https://pixi.sh)
- The sibling **`champollion_pipeline`** repository (not included here) — must contain the stage scripts under `src/` (`generate_morphologist_graphs.py`, `run_cortical_tiles.py`, `generate_champollion_config.py`, `generate_embeddings.py`, `put_together_embeddings.py`, `generate_snapshots.py`, `train_champollion.py`, `run_streaming.py`) and the `external/champollion_V1` and `external/cortical_tiles` submodules.

## Installation

```bash
pixi install
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `CHAMPOLLION_PIPELINE_DIR` | Absolute path to the `champollion_pipeline` repo | Falls back to `../champollion_pipeline` relative to this package |
| `HF_TOKEN` | HuggingFace token | Passed through only to the `embeddings` and `streaming` stages; stripped from every other stage's subprocess environment |
| `BRAINVISA` / `BRAINVISA_SHARE` | BrainVISA install location / share dir | Auto-injected from the pipeline's pixi environment if not already set in the environment |

## Running

This is a stdio server meant to be launched by an MCP client (e.g. the Technician agent's `ClaudeCodeOptions.mcp_servers` config in `champollion_agents`), not run interactively on its own:

```bash
pixi run run                  # python -m champollion_sulcal_mcp.server
# or, once installed:
champollion-sulcal-mcp
```

## Claude Code integration

Beyond the raw MCP tools, this repo ships assets for using the pipeline directly from Claude Code:

- **`agents/champollion-pipeline.md`** — a subagent scoped to exactly the MCP tools it needs, with an operational playbook: always preflight first, never guess paths, default output layout, per-stage required parameters, a known-error-pattern table, and the exact CLI invocation each tool wraps.
- **`skills/run-pipeline`** — guides choosing stage-centric vs. streaming execution and gathering the right parameters.
- **`skills/monitor`** — a poll-every-30-seconds monitoring loop with per-job-type progress reporting.
- **`skills/debug`** — systematic failure diagnosis: read the full log, match against known error patterns, report root cause and fix.

## Job tracking

Each job is persisted as `<output_dir>/.mcp_jobs/<job_id>.json` (atomic write) with its combined stdout/stderr log at `<output_dir>/.mcp_jobs/<job_id>.log`. Status lifecycle: `pending` → `running` → one of `succeeded` / `failed` / `cancelled`. `start_pipeline` additionally writes an "umbrella" job whose progress tracks `current_stage` / `stages_done` / `stages_total` and the currently active child `job_id`.

## Testing

```bash
pixi run test           # full suite
pixi run test-unit        # unit-marked tests only
pixi run test-fast         # stop on first failure
pixi run test-cov           # with coverage report
```

`tests/conftest.py` provides `fake_pipeline_dir` (a temp dir with stub stage scripts and submodule folders) and `recording_runner` (stubs `runner.launch` to record calls instead of spawning real subprocesses), so most tool behavior can be tested without a real `champollion_pipeline` checkout.

## Linting

```bash
pixi run lint        # ruff check
pixi run lint-fix      # ruff check --fix
pixi run format         # ruff format
```

## Note on `docs/agents_architecture.md`

That document is an early architecture proposal for the *champollion_agents* repo (ACP/`acp-sdk`, OpenAI-compatible LLM backend, in-process ChromaDB indexing). It predates and does not reflect the current implementation of either repo — `champollion_agents` now runs its agents through `claude-code-sdk` rather than a custom ACP/LangGraph stack, and this repo has no LLM or ACP code at all. Kept for historical context only.
