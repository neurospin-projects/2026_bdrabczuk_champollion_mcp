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

### Pipeline job (`stage: "pipeline"`)

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

### All jobs

A job keeps running after the MCP server exits, so polling can resume later in a new session with the same `output_dir` and `job_id`.

Show elapsed time if `started_at` is available. Warn if a job has been running more than 2× the expected duration.

## Terminal States

| Status | Action |
|--------|--------|
| `succeeded` with `outcome` `partial` | The stage (or at least one stage of a pipeline) exited with code 3 and left some subjects out. Report `skipped_subjects` first, then the output locations; suggest sanity checks from `run-pipeline` skill |
| `succeeded` (`outcome` null) | Report output locations; suggest sanity checks from `run-pipeline` skill |
| `failed` | Switch to **debug** skill — call `get_job_log` and diagnose |
| `failed` with error "process gone without exit code" | The job's process ended without writing an exit code (`returncode` is null); call `get_job_log` to see how far it got before relaunching |
| `cancelled` | Report cancelled; offer to re-launch |

## Cancellation

If the user asks to stop a job: call `cancel_job(output_dir, job_id)`. `cancel_job` signals the job's process group (SIGTERM, then SIGKILL after a grace period), so child processes stop too. Confirm the final status.

## Listing Jobs

Use `list_jobs(output_dir)` to show all jobs in an output directory. Useful when the user has lost track of a `job_id`.
