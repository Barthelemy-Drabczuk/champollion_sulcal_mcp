---
name: monitor
description: Use this skill to monitor a running Champollion pipeline job. Trigger when a job has been launched and the user wants to track progress, when polling get_job_status, or after any start_* tool returns a job_id.
---

# Champollion Job Monitor

Poll and report the status of a running pipeline job.

## Polling Loop

1. Call `get_job_status(output_dir, job_id)` immediately after launch.
2. Report the current state to the user.
3. Wait **30 seconds**, then poll again.
4. Continue until status is `succeeded`, `failed`, or `cancelled`.

## Progress Reporting

### Pipeline job (`stage: "pipeline"` or `stage: "streaming"`)

Report the umbrella progress:
```
Stage {stages_done}/{stages_total}: {current_stage}
```

For pipeline jobs, also report the current child job if available (`current_child_job_id`).

### Embeddings job (`stage: "embeddings"`)

Report fold progress when available:
```
Fold {fold_current}/{fold_total} ({percent}%)
```

### Streaming job (`stage: "streaming"`)

Tail the log with `get_job_log(output_dir, job_id, tail=50)` every 60 seconds to show per-worker activity. Look for lines like:
- `INFO complete` → worker finished its scan
- `INFO skipping ... embeddings already exist` → resume skip
- `ERROR timed out` → worker timed out waiting for prerequisites
- `ERROR failed:` → worker stage failure

### All jobs

Show elapsed time if `started_at` is available. Warn if a job has been running more than 2× the expected duration.

## Terminal States

| Status | Action |
|--------|--------|
| `succeeded` | Report output locations; suggest sanity checks from `run-pipeline` skill |
| `failed` | Switch to **debug** skill — call `get_job_log` and diagnose |
| `cancelled` | Report cancelled; offer to re-launch |

## Cancellation

If the user asks to stop a job: call `cancel_job(output_dir, job_id)`. Confirm the final status.

## Listing Jobs

Use `list_jobs(output_dir)` to show all jobs in an output directory. Useful when the user has lost track of a `job_id`.
