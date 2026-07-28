# Stage Parameters Reference

Detailed parameter guide for each `start_<stage>` tool. Use this when running stages individually rather than via `start_pipeline`.

---

## Stage 1 — `start_morphologist`

Calls `morphologist-cli` to generate sulcal graphs (`.arg` files) from T1 NIfTI images.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `input_dir` | str | yes | Directory of T1 NIfTI files (`.nii.gz`) |
| `output_dir` | str | yes | Derivatives root; writes to `morphologist-*/` subdirectory |
| `parallel` | bool | no | Enable soma-workflow parallel scheduling (Neurospin cluster) |
| `enable_sulcal_recognition` | bool | no | Run sulcal labeling (slower, not required for embeddings) |

Output: `{output_dir}/morphologist-{version}/subjects/{subject}/`

---

## Stage 2 — `start_cortical_tiles`

Extracts 28 standardized sulcal region crops from Morphologist's graphs.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `input_dir` | str | yes | Morphologist `subjects/` dir (read-only safe) |
| `output_dir` | str | yes | Derivatives root; writes `cortical_tiles-{YEAR}/crops/2mm/` |
| `path_to_graph` | str | yes | Relative path to `.arg` graph from subject dir. Supports `*` wildcards. E.g. `t1mri/default_acquisition/default_analysis/folds/3.1` |
| `path_sk_with_hull` | str | yes | Relative path to skeleton dir. E.g. `t1mri/default_acquisition/default_analysis/segmentation` |
| `sk_qc_path` | str | no | TSV file with `participant_id` and `qc` columns for filtering |
| `njobs` | int | no | CPU cores (default: auto) |
| `masks` | str | no | Mask version tag override (e.g. `canonical_25`) |
| `regions` | list[str] | no | Restrict to a subset of the 28 sulcal regions (space-separated in CLI; list here) |

Output: `{output_dir}/cortical_tiles-{YEAR}/crops/2mm/` — 28 region folders.

> **CLI-only flags (not exposed via MCP):** `--skip-distbottom`, `--input-types`. Do not attempt to pass these through the MCP tool.

**Default `path_to_graph` by mode:**
- Non-BIDS: `t1mri/default_acquisition/default_analysis/folds/3.1`
- BIDS (`bids=True`): `t1mri/default_acquisition/0/folds/3.1`
- Wildcard fallback: `t1mri/default_acquisition/*/folds/3.1`

---

## Stage 3 — `start_config`

Generates Champollion dataset YAML configuration files (`reference.yaml`, `local.yaml`).

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `crop_path` | str | yes | Path to `crops/2mm/` directory from stage 2 |
| `dataset` | str | yes | Short dataset name (e.g. `TESTXX`) |
| `output` | str | no | Config output dir. Recommended: `{output_dir}/champollion_V1/configs/dataset/{dataset}` |
| `champollion_loc` | str | no | Override path to champollion_V1 (default: `external/champollion_V1`) |
| `external_config` | str | no | For read-only containers: write `local.yaml` to a writable path |
| `external_crops` | bool | no | Set when crops are outside the pipeline dir |

Output: `{output}/reference.yaml` (and related files). This `output` path is the `config_path` for stage 4.

---

## Training — `start_training`

Trains a champollion_V1 self-supervised encoder for a single sulcal region. Use this
instead of (or before) `start_embeddings` when custom model weights are needed.

**Prerequisite:** `start_config` must have run first, or configs must exist in `config_dir`.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `dataset` | str | yes | Dataset name — must match a directory in the config search path |
| `region` | str | yes | Region config name, e.g. `cingulate_left`. Must match a YAML file under `configs/dataset/{dataset}/` |
| `mode` | str | no | `encoder` (default), `classifier`, or `regresser` |
| `output_dir` | str | no | Absolute path for Hydra logs and model checkpoints. Defaults to `{pipeline_dir}/data/{dataset}/derivatives/champollion_V1/models/{region}/` |
| `config_dir` | str | no | Absolute path to a local Hydra configs root containing `dataset/{dataset}/{region}.yaml`. Required when configs live outside the champollion_V1 submodule (e.g. written by `start_config` with `output` set) |
| `njobs` | int | no | CPU DataLoader workers |
| `cpu` | bool | no | Force CPU (disable CUDA) |
| `overwrite` | bool | no | Re-train even if `output_dir` already exists |
| `swf` | bool | no | Submit via soma-workflow (Neurospin HPC only) |

Output: `{output_dir}/` — Hydra run directory containing the trained model checkpoint and config snapshot.

> **Not available in the streaming pipeline.** Training aggregates all subjects and cannot be parallelised per scan.

---

## Stage 4 — `start_embeddings`

Runs inference across 56 model folds (28 regions × 2 hemispheres). This is the longest stage.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `models_path` | str | yes | Local dir, `.tar.gz` archive, or HF repo ID (`neurospin/Champollion_V1`) |
| `dataset_localization` | str | yes | Use `local` for local datasets |
| `datasets_root` | str | yes | Absolute path to the dataset derivatives root |
| `short_name` | str | yes | Run tag (e.g. `run01`). Used in output path names |
| `config_path` | str | **yes** | Stage 3 output dir — the folder containing `reference.yaml` and per-region YAML files. Without this, dataset resolution fails with `IndexError: list index out of range`. |
| `embeddings_only` | bool | no | Skip classifier training (default: `true`) |
| `cpu` | bool | no | Force CPU (disable CUDA) |
| `overwrite` | bool | no | Re-run and overwrite existing embeddings |
| `nb_jobs` | int | no | Parallel jobs |
| `labels` | list[str] | no | Labels for classifiers (default: `['Sex']`) |
| `datasets` | list[str] | no | Dataset names to process (default: all found in `datasets_root`) |

> **CLI-only flags (not exposed via MCP):** `--split` (embedding split strategy), `--classifier_name`. Use the MCP parameters above; do not attempt to pass these.

**HF_TOKEN**: Set in env before starting the MCP server if downloading from HuggingFace.

Output: `{datasets_root}/champollion_V1/models_cache/Champollion_V1/*/` — 56 fold subdirectories, each with `{short_name}_random_embeddings/full_embeddings.csv`.

---

## Stage 5 — `start_combine`

Collects 56 `full_embeddings.csv` files into a single flat directory.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `embeddings_subpath` | str | yes | Relative path within each fold dir **including filename**. Pattern: `{short_name}_random_embeddings/full_embeddings.csv` |
| `output_path` | str | yes | Target directory for the 56 collected CSVs |
| `path_models` | str | no | Override models cache dir (default: auto-detected) |

Output: `{output_path}/*.csv` — 56 files, one per region+hemisphere.

---

## Stage 6 — `start_snapshots`

Renders sulcal graph meshes, cortical tile masks, and UMAP scatter plots.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `output_dir` | str | yes | Where to write snapshot images |
| `morphologist_dir` | str | no | Morphologist output dir (for sulcal graph snapshots) |
| `cortical_tiles_dir` | str | no | `crops/2mm/` dir (for tile mask snapshots) |
| `embeddings_dir` | str | no | Combined embeddings dir from stage 5 (for UMAP plots) |
| `subject` | str | no | Subject folder name to visualize (default: first found) |
| `acquisition` | str | no | Acquisition tag (required if subject has multiple segmentations) |
| `sulcal_only` | bool | no | Render only sulcal graph images (requires `morphologist_dir`) |
| `tiles_only` | bool | no | Render only cortical tile mask images (requires `cortical_tiles_dir`) |
| `umap_only` | bool | no | Render only UMAP scatter plots (requires `embeddings_dir`) |
| `umap_region` | str | no | Restrict UMAP plot to a specific sulcal region (e.g. `SPoC_left`) |

At least one of `morphologist_dir`, `cortical_tiles_dir`, or `embeddings_dir` must be provided.

UMAP plots require a `reference_data/` dir inside the pipeline — this is bundled with `champollion_pipeline` and needs no configuration.

---

## Streaming — `start_streaming`

Scan-centric parallel runner (Strategy 2). Spawns N workers, one per `ScanId`, each running stages 2–4 sequentially with file-presence barriers. Stage 5 (combine) runs once after all workers drain. **Always runs in `embeddings_only` mode** — training requires all subjects and cannot be parallelised per scan.

| Parameter | Type | Required | Notes |
|-----------|------|----------|-------|
| `input_dir` | str | yes | Morphologist `subjects/` directory (absolute path) |
| `output_dir` | str | yes | Root output directory for all pipeline stages |
| `dataset` | str | yes | Short dataset name (e.g. `MY_COHORT`) |
| `path_to_graph` | str | yes | Relative path to `.arg` graph from subject dir (supports `*` wildcards) |
| `path_sk_with_hull` | str | yes | Relative path to skeleton directory from subject dir |
| `bids` | bool | no | Input follows BIDS layout `sub-*/ses-*/...` (default: `False`) |
| `n_workers` | int | no | Number of parallel workers. `0` = `os.cpu_count()` (default: `0`) |
| `worker_timeout` | int | no | Per-worker timeout in seconds waiting for prerequisite files (default: `7200`) |
| `poll_interval` | int | no | Seconds between filesystem polls per worker (default: `10`) |
| `sk_qc_path` | str | no | TSV file with `participant_id` and `qc` columns to filter subjects |
| `models_path` | str | no | Local dir, `.tar.gz` archive, or HF repo ID for model weights |
| `dataset_localization` | str | no | `"local"` for local datasets (default: `"local"`) |
| `datasets_root` | str | no | Absolute path to dataset derivatives root |
| `short_name` | str | no | Run tag for output folder names (default: `"eval"`) |
| `embeddings_path` | str | no | Relative path for combined embeddings (default: `"champollion_V1"`) |
| `dry_run` | bool | no | Enumerate scans and log them without processing (default: `False`) |

**Output locations:**
- Worker logs: `{output_dir}/logs/{scan_id}/worker.log`
- Cortical crops: `{output_dir}/cortical_tiles-{YEAR}/crops/2mm/`
- Config files: `{output_dir}/champollion_V1/configs/dataset/{dataset}/`
- Combined embeddings: `{output_dir}/combined_embeddings/`

**Notes:**
- `njobs_per_worker` is derived automatically: `max(1, cpu_count // n_workers)`
- Resume is built-in: workers skip scans whose `full_embeddings.csv` already exists
- Not compatible with soma-workflow; targets local multiprocessing only
- GPU contention: with multiple workers sharing a GPU, consider `n_workers=1` for the embeddings phase or use CPU mode
