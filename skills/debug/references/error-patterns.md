# Error Patterns Reference

Detailed error categories for diagnosing failed Champollion jobs. Match against log tail output.

---

## Environment Errors (any stage)

### Missing pipeline directory
```
FileNotFoundError: Pipeline directory not found
CHAMPOLLION_PIPELINE_DIR is not set
```
**Fix:** Set `CHAMPOLLION_PIPELINE_DIR` in the MCP server's environment (`.env` or Claude config `env` block). Run `preflight_check()` to verify.

### Missing Python executable
```
.pixi/envs/default/bin/python: No such file or directory
```
**Fix:** Run `pixi install` inside `champollion_pipeline/`. The Pixi environment hasn't been set up.

### Missing BrainVISA tools (morphologist/cortical_tiles)
```
VipSkeleton: command not found
morphologist-cli: not found
aims-config: not found
```
**Fix:** The `BRAINVISA_SHARE` env var is missing or `PATH` doesn't include the BrainVISA bin. `preflight_check()` should flag this. The MCP server injects `BRAINVISA_SHARE` automatically from the pixi env — if it's missing, the pixi install may be incomplete.

---

## Stage 1 — Morphologist

### No NIfTI files found
```
No .nii.gz files found in
ValueError: empty sequence
```
**Fix:** `input_dir` must contain NIfTI files directly (not nested subdirectories unless morphologist-cli supports globbing). Check actual file locations.

### soma-workflow unavailable (parallel mode)
```
ImportError: No module named 'soma_workflow'
soma-workflow server not running
```
**Fix:** Remove `parallel=True` and run in serial mode, or configure soma-workflow GUI first (`soma_workflow_gui`).

---

## Stage 2 — Cortical Tiles

### Graph path pattern matched nothing
```
No graphs found
0 subjects processed
AssertionError: no .arg files
```
**Fix:** The `path_to_graph` pattern doesn't match the actual directory tree under each subject. Inspect a subject folder:
```bash
ls {morphologist_subjects_dir}/{one_subject}/
```
Use `*` wildcards for variable segments, e.g. `t1mri/*/default_analysis/folds/3.1`.

### QC file format error
```
KeyError: 'participant_id'
pandas.errors.ParserError
```
**Fix:** The QC TSV must have `participant_id` and `qc` (0/1) columns, tab-separated. Check encoding and delimiter.

---

## Stage 3 — Champollion Config

### Wrong crop_path
```
reference.yaml: No such file or directory
crops/2mm not found
```
**Fix:** `crop_path` must point to the `crops/2mm/` directory itself, not its parent. Correct path: `{output_dir}/cortical_tiles-{YEAR}/crops/2mm`.

### Read-only pipeline directory
```
PermissionError: [Errno 13] Permission denied: 'external/champollion_V1/local.yaml'
```
**Fix:** Use `external_config` parameter to write `local.yaml` to a writable path:
```python
start_config(crop_path=..., dataset=..., external_config="/writable/path/local.yaml")
```

---

## Stage 4 — Embeddings

### Model folder has no checkpoint — `load_model` finds nothing
```
IndexError: list index out of range
  weights_path = glob.glob(model_path + '/logs/lightning_logs/version_0/checkpoints/*.ckpt')[0]
```
**Root cause:** `champollion/evaluate.py` `load_model` takes the first match of `<model>/logs/lightning_logs/version_0/checkpoints/*.ckpt`. Before each region, `generate_embeddings.py` converts `<model>/logs/best_model_weights.pt` into that checkpoint folder when no native `.ckpt` exists — but if the region's model folder has **neither** `logs/lightning_logs/version_0/checkpoints/*.ckpt` **nor** `logs/best_model_weights.pt`, nothing is converted, the glob is empty and `[0]` fails. Typical causes: an interrupted HuggingFace download, a partially extracted archive, or a `models_path` pointing at an incomplete model set.

**Fix:**
1. Check every region folder under `models_path` for one of the two files: `ls <models_path>/<region>/logs/best_model_weights.pt <models_path>/<region>/logs/lightning_logs/version_0/checkpoints/`.
2. Re-fetch the models (re-run `start_embeddings` after deleting the incomplete cached copy), or, from the CLI, run `generate_embeddings.py --no-cache` to force re-extraction of an archive / re-download (`--no-cache` is not exposed by `start_embeddings`).
3. Or point `models_path` at a complete model set.

### CUDA out of memory
```
RuntimeError: CUDA out of memory
torch.cuda.OutOfMemoryError
```
**Fix:** Use `cpu=True` to run on CPU (slower but reliable), or free the GPU from other processes. `start_embeddings` has no job-count option: regions are embedded one at a time.

### HuggingFace download failure
```
requests.exceptions.ConnectionError
huggingface_hub.utils._errors.RepositoryNotFoundError
401 Unauthorized
```
**Fix:** Set `HF_TOKEN` in the MCP server environment (Claude config `env.HF_TOKEN`). For air-gapped environments, pre-download the model and pass the local path as `models_path`.

### Embeddings already exist (no overwrite)
```
Embeddings already exist, skipping
```
This is not a failure — it's normal behavior. If you want to recompute, add `overwrite=True`.

---

## Stage 5 — Combine

### embeddings_subpath doesn't match
```
No files found matching
FileNotFoundError: full_embeddings.csv
0 files collected
```
**Fix:** The `embeddings_subpath` must exactly match the relative path inside each model fold. Check one fold manually:
```bash
ls {datasets_root}/champollion_V1/models_cache/Champollion_V1/SC-sylv_left/*/
```
The correct `embeddings_subpath` pattern is `{short_name}_{split}_embeddings/full_embeddings.csv` — e.g. `run01_random_embeddings/full_embeddings.csv`.

---

## Stage 6 — Snapshots

### Multiple acquisitions ambiguous
```
Warning: multiple acquisitions found for subject
```
This is a warning, not a crash. But if the script fails immediately after, pass `acquisition=<tag>` explicitly.

### UMAP reference data missing
```
FileNotFoundError: reference_data/umap_*.pkl
```
**Fix:** Pass `reference_data_dir` pointing at a directory holding `umap_{region}_{hemi}.pkl` and `_coords.npy`; it is not shipped, obtain it separately or generate it with `pixi run generate-umap-reference` in `champollion_pipeline` (see the pipeline README "UMAP Visualization" section).

### Anatomist display error (headless)
```
QXcbConnection: Could not connect to display
RuntimeError: Anatomist initialization failed
```
**Fix:** A virtual display (Xvfb) is required for Anatomist. Either set `DISPLAY` to a virtual framebuffer, or run on a machine with a display. On Neurospin clusters, use a display-enabled node.

---

## Training (`start_training`)

### Configs root wrong — dataset config not resolved
```
FileNotFoundError: Dataset config not found in either location
```
**Fix:** `config_dir` must be the stage-3 configs root — default `<D>/<dataset>/derivatives/champollion_V1/configs`, the directory containing `dataset/<dataset>/<region>.yaml` (next to `reference.yaml`) and `dataset_localization/<localization>.yaml`. Do not pass the `dataset/<dataset>/` subfolder or a `local.yaml` file. Omit `config_dir` when stage 3 wrote to the default location under the pipeline's `data/` directory.

---

## Generic Patterns

### Exit code 1 with no traceback
The subprocess failed silently. Request more log lines (`tail_lines=1000`) and look for earlier errors — sometimes the real error is 100+ lines before the final exit.

### Exit code 137
OOM kill by the Linux kernel (not Python). Reduce parallelism or switch to CPU mode.

### Exit code 130
SIGINT (Ctrl-C) or user cancellation — not a real failure. Re-run the stage.
