---
name: debug
description: Use this skill to debug a failed Champollion pipeline job. Trigger when a job status is "failed", when the user reports an error, or when a stage produces unexpected output.
---

# Champollion Pipeline Debugger

Systematic diagnosis of pipeline failures using job logs and known error patterns.

## Step 1 — Read the Log

Call `get_job_log(output_dir, job_id)` and read the **full** output. Do not truncate.

For streaming jobs, also read the per-worker logs:
```bash
cat {output_dir}/logs/{scan_id}/worker.log
```

## Step 2 — Identify the Error

Search for known patterns in `references/error-patterns.md`. Match the error message to a root cause and fix.

## Step 3 — Report and Fix

1. State the **root cause** clearly (not just the error message).
2. Provide the **exact fix** — corrected parameter, missing file, or command to run.
3. Offer to re-launch the stage with the corrected parameters.

## Step 4 — Verify the Fix

After re-launching:
- Switch to monitor skill behavior
- On success: confirm the output files exist
- On repeated failure: escalate to deeper investigation (check `get_pipeline_info`, verify filesystem state)

## Useful Diagnostics

```python
# Check pipeline environment
get_pipeline_info()

# Check if subjects dir has the right structure
# (run from Bash if needed)
ls {subjects_dir}/sub-001/  # should show ses-* or t1mri/

# Check expected graph path
ls {subjects_dir}/{subject}/{path_to_graph}/R*.arg

# Check crops were generated
ls {output_dir}/cortical_tiles-*/crops/2mm/ | wc -l  # expect 28

# Check config was generated
ls {output_dir}/champollion_V1/configs/dataset/{dataset}/
```

## When to Escalate

Stop and ask the user if:
- The error is not in the known patterns
- The fix requires modifying input data
- The same stage fails 3 times with the same error after corrections
- A CUDA/GPU error persists after switching to `cpu=True`
