---
name: run-pipeline
description: Use this skill to run the Champollion sulcal embedding pipeline. Trigger when the user wants to process T1 MRI data, generate sulcal embeddings, run Morphologist, run cortical tiles, launch the champollion pipeline, or process a neuroimaging dataset through the 6-stage pipeline. Also trigger when the user asks to run specific stages (morphologist, cortical_tiles, config, training, embeddings, combine, snapshots), train a champollion encoder, or use the streaming/scan-centric parallel pipeline via the champollion MCP server.
---

# Champollion Pipeline Runner

This skill guides running the Champollion sulcal embedding pipeline via the `champollion-sulcal` MCP server. The pipeline transforms raw T1 MRI data into sulcal embeddings in 6 stages.

## Step 0 — Preflight

Always call `preflight_check()` first. If it reports issues, stop and help the user resolve them.

---

## Step 1 — Execution Mode

Ask two questions upfront:

**A. Stage-centric or scan-centric?**

Two execution strategies are available:

| Strategy | Tool | Best for |
|----------|------|----------|
| **Stage-centric** (default) | `start_pipeline` or individual `start_<stage>` | Large cohorts where all Morphologist graphs exist upfront |
| **Scan-centric streaming** | `start_streaming` | Online/incremental scenarios; subjects arrive progressively |

- **Stage-centric**: runs each stage across all subjects before moving to the next. Supports all 6 stages including training.
- **Scan-centric streaming**: spawns N workers, each owning one scan (ScanId), running stages 2–4 sequentially with file-presence barriers. Stage 5 (combine) runs once after all workers drain. **Requires `embeddings_only` mode** — training cannot be parallelised per-scan.

If the user mentions "streaming", "scan by scan", "incremental", "subjects arriving progressively", or "one worker per scan" → use `start_streaming`.

**B. Serial or parallel? (stage-centric only)**

Only stages 1, 2, and 4 support intra-stage parallelism. Stages 3, 5, and 6 are always single-threaded.

| Stage | Serial | Parallel option |
|-------|--------|----------------|
| 1 — Morphologist | default | `parallel=True` → soma-workflow (Neurospin cluster only); requires `soma_workflow_gui` configured first |
| 2 — Cortical tiles | default | `njobs=N` → multi-CPU (default: auto) |
| 3 — Config | always | — |
| 4 — Embeddings | `cpu=True` | GPU by default; `nb_jobs=N` for parallel CPU jobs |
| 5 — Combine | always | — |
| 6 — Snapshots | always | — |

If parallel: ask how many cores (for stages 2 and 4), or confirm soma-workflow is configured (for stage 1).

**C. Full pipeline or individual stages? (stage-centric only)**

- **Full pipeline** → `start_pipeline` chains all 6 stages; use `skip_stages` to resume from a checkpoint
- **Individual stages** → `start_<stage>` tools, one at a time

---

## Step 2 — Gather Per-Stage Parameters

Ask for the inputs and outputs of **each stage the user intends to run**. All paths must be **absolute**. Never guess paths.

**Default output structure**: unless the user specifies otherwise, outputs follow this layout under the champollion project root:

```
./data/{dataset}/
├── rawdata/
└── derivatives/
    ├── morphologist-6.0/
    ├── cortical_tiles-2026/
    ├── champollion_V1/
    └── snapshots/
```

Use `./data/{dataset}/derivatives/` as the default `output_dir` for all stages. Stage 3's `output` parameter must always be asked explicitly (common value: `./data/{dataset}/derivatives/champollion_V1/configs/dataset/{dataset}`).

**Default `path_to_graph`**:
- Non-BIDS: `t1mri/default_acquisition/default_analysis/folds/3.1`
- BIDS (`bids=True`): `t1mri/default_acquisition/0/folds/3.1`

---

### Stage 1 — Morphologist
Generates sulcal graphs (`.arg` files) from T1 NIfTI images.

**Ask:**
| Parameter | Description |
|-----------|-------------|
| `input_dir` | Directory containing the raw T1 NIfTI files (`.nii.gz`) |
| `output_dir` | Root output directory — Morphologist writes to `{output_dir}/morphologist-{version}/subjects/` |

**Optional:**
| Parameter | Description |
|-----------|-------------|
| `parallel` | `True` to use soma-workflow (Neurospin cluster; requires prior `soma_workflow_gui` setup) |
| `enable_sulcal_recognition` | `True` to run sulcal labeling (slower; not required for embeddings) |

---

### Stage 2 — Cortical Tiles
Extracts 28 standardized sulcal region crops from the graphs.

**Ask:**
| Parameter | Description |
|-----------|-------------|
| `input_dir` | Morphologist `subjects/` directory (from stage 1: `{output_dir}/morphologist-{version}/subjects/`) |
| `output_dir` | Derivatives parent directory — writes to `{output_dir}/cortical_tiles-{YEAR}/crops/2mm/` |
| `path_to_graph` | Relative path pattern to the `.arg` graph inside each subject folder. Supports `*` wildcards. Common value: `t1mri/default_acquisition/default_analysis/folds/3.1` |
| `path_sk_with_hull` | Relative path to skeleton directory inside each subject folder. Common value: `t1mri/default_acquisition/default_analysis/segmentation` |

**Optional:**
| Parameter | Description |
|-----------|-------------|
| `sk_qc_path` | TSV file with `participant_id` and `qc` (0/1) columns to filter subjects |
| `njobs` | Number of CPU cores (default: auto) |

Less common options (ask only if user requests customisation):
- `--region-file` — custom sulcal region configuration file
- `--input-types` — restrict generated input types (e.g. `skeleton foldlabel extremities`); default: `skeleton foldlabel` (extremities only on request)
- `--with-distbottom` — generate distbottom crops (off by default; unused by champollion_V1 inference); mutually exclusive with the deprecated `--skip-distbottom`
- `--skip-distbottom` — deprecated no-op kept for backward compatibility (distbottom is already skipped by default)
- `--masks` — mask version tag override (e.g. `canonical_25`)
- `--regions` — restrict to a subset of the 28 sulcal regions (space-separated)

> Note: `input_dir` is read-only safe — the script never writes to it.

---

### Stage 3 — Champollion Config
Generates dataset YAML configuration files (`reference.yaml`, `local.yaml`).

**Ask:**
| Parameter | Description |
|-----------|-------------|
| `crop_path` | Path to the `crops/2mm/` directory from stage 2: `{output_dir}/cortical_tiles-{YEAR}/crops/2mm` |
| `dataset` | Short dataset name (e.g. `COHORT_XX`) |
| `output` | **Always ask explicitly.** Where the user wants config files written. Do not infer or default. Common choice: `{output_dir}/champollion_V1/configs/dataset/{dataset}` — but confirm with the user. |

**Optional:**
| Parameter | Description |
|-----------|-------------|
| `champollion_loc` | Override path to `champollion_V1` binaries (default: `external/champollion_V1`) |
| `external_config` | For read-only containers (Apptainer/Docker): write `local.yaml` to a writable path outside the pipeline dir |

> The `output` path here becomes the `config_path` for stage 4 — keep it consistent.

---

### Training — Train a region encoder

Train a champollion_V1 self-supervised encoder on a custom dataset. This is an optional
step between config (stage 3) and embeddings (stage 4): use it when you want to train a
new model rather than using the pre-trained HuggingFace weights.

**Prerequisite:** stage 3 (`start_config`) must have run first, or configs must exist in a
local directory supplied via `config_dir`.

**Ask:**
| Parameter | Description |
|-----------|-------------|
| `dataset` | Dataset name — must match the directory used in stage 3 |
| `region` | Region config name, e.g. `cingulate_left` |

**Optional:**
| Parameter | Description |
|-----------|-------------|
| `mode` | `encoder` (default), `classifier`, or `regresser` |
| `output_dir` | Absolute path for Hydra logs and model checkpoints. Defaults to `data/{dataset}/derivatives/champollion_V1/models/{region}/` |
| `config_dir` | Root of a local Hydra configs directory (contains `dataset/{dataset}/{region}.yaml`). Required when configs were written outside the champollion_V1 submodule (i.e. when `output` was set in stage 3) |
| `njobs` | Number of CPU DataLoader workers |
| `cpu` | `True` to force CPU (disables CUDA) |
| `overwrite` | `True` to re-train even if the output directory already exists |
| `swf` | Not supported: `True` is rejected with an error (training has no soma-workflow mode) |

> **Note:** training operates on the whole dataset — it cannot be parallelised per-scan and is therefore not available in the streaming pipeline.

---

### Stage 4 — Embeddings
Runs inference across all 56 model folds (28 regions × 2 hemispheres).

**Ask:**
| Parameter | Description |
|-----------|-------------|
| `models_path` | Local directory, `.tar.gz` archive, HuggingFace repo ID (`neurospin/Champollion_V1`), or URL |
| `dataset_localization` | Always `local` for local datasets |
| `datasets_root` | Absolute path to the dataset derivatives root (e.g. `/data/TESTXX/derivatives/`) |
| `short_name` | Run tag (e.g. `run01`) — used in output folder names; use different values to avoid overwriting past runs |
| `config_path` | **Always required.** The stage 3 output directory — the folder that contains `reference.yaml` and the per-region YAML files. Without this, the embeddings pipeline crashes with `IndexError: list index out of range`. |

**Optional:**
| Parameter | Description |
|-----------|-------------|
| `embeddings_only` | `True` to skip classifier training (default: `True`) |
| `cpu` | `True` to force CPU (disables CUDA; slower but avoids OOM) |
| `nb_jobs` | Parallel jobs for CPU mode |
| `overwrite` | `True` to recompute and overwrite existing embeddings |

> Models are cached at `{datasets_root}/champollion_V1/models_cache/`. On subsequent runs, pass the cached path directly as `models_path` to skip HuggingFace update checks.

> **HF_TOKEN**: Must be set in the MCP server's environment at startup if downloading from HuggingFace.

---

### Stage 5 — Combine
Collects all 56 per-fold `full_embeddings.csv` files into a single flat directory.

**Ask:**
| Parameter | Description |
|-----------|-------------|
| `embeddings_subpath` | Relative path inside each model fold dir **including filename**. Pattern: `{short_name}_random_embeddings/full_embeddings.csv` (replace `random` with the `split` value used in stage 4, default: `random`) |
| `output_path` | Directory for the 56 collected CSVs (e.g. `{output_dir}/champollion_V1/embeddings/`) |
| `path_models` | (Optional) Override for the models cache dir (`{datasets_root}/champollion_V1/models_cache/Champollion_V1/`) |

---

### Stage 6 — Snapshots
Renders sulcal graph meshes, cortical tile masks, and UMAP scatter plots.

**Ask:**
| Parameter | Description |
|-----------|-------------|
| `output_dir` | Where to write snapshot images |

Then ask which snapshot types they want (at least one source directory is required):

| Ask if... | Parameter | Description |
|-----------|-----------|-------------|
| Sulcal graph images wanted | `morphologist_dir` | Morphologist output directory (e.g. `{output_dir}/morphologist-*/`) |
| Tile mask images wanted | `cortical_tiles_dir` | `crops/2mm/` directory from stage 2 |
| UMAP scatter plots wanted | `embeddings_dir` | Combined embeddings directory from stage 5 |

**Optional:**
| Parameter | Description |
|-----------|-------------|
| `subject` | Subject folder name to visualize (default: first found) |
| `acquisition` | Acquisition tag — required if a subject has multiple segmentations |

---

### Streaming Pipeline
Scan-centric parallel runner: N workers, one per scan, each running stages 2–4 with file-presence barriers.

**Ask:**
| Parameter | Description |
|-----------|-------------|
| `input_dir` | Morphologist `subjects/` directory |
| `output_dir` | Root output directory |
| `dataset` | Short dataset name |
| `path_to_graph` | Relative path to `.arg` graph from subject dir |
| `path_sk_with_hull` | Relative path to skeleton directory |

**Optional:**
| Parameter | Description |
|-----------|-------------|
| `bids` | `True` for BIDS layout (`sub-*/ses-*/...`) |
| `n_workers` | Number of parallel workers (0 = `os.cpu_count()`) |
| `worker_timeout` | Seconds to wait for prerequisite files per worker (default: 7200) |
| `dry_run` | `True` to enumerate scans without processing |
| `models_path` | Models path or HF repo ID |
| `datasets_root` | Dataset derivatives root |
| `short_name` | Run tag for output folder names |

---

## Step 3 — Launch

### Full pipeline

```python
start_pipeline(
  input_dir=<stage 1 input dir>,
  output_dir=<derivatives root>,
  dataset=<dataset name>,
  models_path=<models path or HF repo>,
  short_name=<run tag>,
  config_path=<stage 3 output dir>,         # always required for embeddings
  skip_stages=["morphologist", ...],         # omit if starting fresh
  parallel=<True/False>,                     # stage 1 soma-workflow
  njobs=<N>,                                 # stage 2 CPU cores
  nb_jobs=<N>,                               # stage 4 CPU jobs
  cpu=<True/False>,                          # stage 4 GPU → CPU
)
```

### Individual stages

Call `start_<stage>` with the parameters above. See `references/stage-params.md` for the complete parameter list.

### Streaming pipeline

```python
start_streaming(
  input_dir=<subjects_dir>,
  output_dir=<derivatives root>,
  dataset=<dataset name>,
  path_to_graph=<relative path to .arg>,
  path_sk_with_hull=<relative path to skeleton>,
  bids=<True/False>,
  n_workers=<N>,                      # 0 = os.cpu_count()
  worker_timeout=7200,
  dry_run=<True/False>,               # enumerate scans without processing
  models_path=<models path or HF repo>,
  datasets_root=<dataset derivatives root>,
  short_name=<run tag>,
)
```

For **dry-run**: set `dry_run=True` first — the job log will list all scans that would be processed without running anything.

---

## Step 4 — After Launching

1. Report the `job_id` and relevant output path to the user.
2. Switch to **monitor** skill behavior to poll until completion.
3. On terminal state:
   - **succeeded** → report output locations and suggest sanity checks (below)
   - **failed** → switch to **debug** skill behavior

## Output Locations (full pipeline)

| Stage | Output path |
|-------|-------------|
| Morphologist graphs | `{output_dir}/morphologist-*/subjects/` |
| Sulcal region crops | `{output_dir}/cortical_tiles-*/crops/2mm/` (28 folders) |
| Champollion config | `{output_dir}/champollion_V1/configs/dataset/{dataset}/` |
| Per-fold embeddings | `{datasets_root}/champollion_V1/models_cache/Champollion_V1/` |
| Combined embeddings | `{output_dir}/champollion_V1/embeddings/` (56 CSV files) |
| Snapshots | `{output_dir}/champollion_V1/snapshots/` |
| Streaming worker logs | `{output_dir}/logs/{scan_id}/worker.log` |
| Streaming combined | `{output_dir}/combined_embeddings/` |

Sanity checks:
```bash
# 28 sulcal region folders?
ls {output_dir}/cortical_tiles-*/crops/2mm | wc -l

# 56 combined embedding CSVs?
ls {output_dir}/champollion_V1/embeddings/*.csv | wc -l
```
