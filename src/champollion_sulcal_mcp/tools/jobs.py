from __future__ import annotations

import logging

from fastmcp import Context
from fastmcp.exceptions import ToolError

from .. import job_store, runner
from ..job_store import JobState

logger = logging.getLogger(__name__)


async def _reconciled(output_dir: str, state: JobState) -> JobState:
    """``state`` reconciled with its process; a running record found dead is persisted under the job lock."""
    result = job_store.reconcile(state)
    if result.status == state.status:
        return result

    def decide(cur: JobState) -> dict | None:
        if cur.status != "running":
            return None  # a terminal or cancelled status written meanwhile wins
        fresh = job_store.reconcile(cur)  # an .rc written since the first read wins over "process gone"
        if fresh.status == "running":
            return None
        return {
            "status": fresh.status,
            "returncode": fresh.returncode,
            "error": fresh.error,
            "ended_at": fresh.ended_at,
        }

    return await job_store.update_job_with(output_dir, state.job_id, decide)


async def _reconciled_or_stored(output_dir: str, job_id: str, state: JobState) -> JobState:
    """Reconciled ``state``; ToolError when the record vanished, the stored state when reconcile fails."""
    try:
        return await _reconciled(output_dir, state)
    except FileNotFoundError:
        raise ToolError(f"Unknown job_id {job_id!r} for output_dir {output_dir!r}") from None
    except Exception as exc:
        logger.warning("reconcile failed for job %s: %s", job_id, exc)
        return state


async def get_job_status(output_dir: str, job_id: str, ctx: Context | None = None) -> dict:
    """Get the current status and progress of a running or completed job."""
    try:
        state = job_store.read_job(output_dir, job_id)
    except FileNotFoundError:
        raise ToolError(f"Unknown job_id {job_id!r} for output_dir {output_dir!r}") from None
    state = await _reconciled_or_stored(output_dir, job_id, state)
    if ctx:
        await ctx.info(f"Job {job_id}: {state.status}")
    return state.model_dump()


async def list_jobs(
    output_dir: str,
    status: str | None = None,
    limit: int = 50,
    ctx: Context | None = None,
) -> list[dict]:
    """List all jobs for a given output directory, optionally filtered by status."""
    jobs = []
    for job in job_store.list_jobs(output_dir):
        try:
            jobs.append(await _reconciled(output_dir, job))
        except Exception as exc:
            logger.warning("reconcile failed for job %s: %s", job.job_id, exc)
            jobs.append(job)
    if status:
        jobs = [j for j in jobs if j.status == status]
    if ctx:
        await ctx.info(f"Found {len(jobs)} jobs in {output_dir}")
    return [j.model_dump() for j in jobs[:limit]]


async def cancel_job(output_dir: str, job_id: str, ctx: Context | None = None) -> dict:
    """Cancel a running job: SIGTERM to its process group, then SIGKILL after a grace period;
    raises a ToolError naming the reason when the recorded process cannot be identified safely."""
    try:
        state = await runner.cancel(output_dir, job_id)
    except FileNotFoundError:
        raise ToolError(f"Unknown job_id {job_id!r} for output_dir {output_dir!r}") from None
    except runner.CancelRefused as exc:
        raise ToolError(f"Job {job_id} not cancelled: {exc.reason}") from None
    if ctx:
        await ctx.info(f"Job {job_id} cancel requested, status: {state.status}")
    return state.model_dump()


async def get_job_log(
    output_dir: str,
    job_id: str,
    tail_lines: int = 200,
    ctx: Context | None = None,
) -> dict:
    """Retrieve the last N lines of a job's log output."""
    tail_lines = max(1, min(tail_lines, 5000))
    try:
        state = job_store.read_job(output_dir, job_id)
    except FileNotFoundError:
        raise ToolError(f"Unknown job_id {job_id!r} for output_dir {output_dir!r}") from None
    state = await _reconciled_or_stored(output_dir, job_id, state)

    from pathlib import Path

    log = Path(state.log_path)
    if not log.exists():
        raise ToolError(f"Log file not available for job {job_id}")

    lines = log.read_text(errors="replace").splitlines()
    tail = lines[-tail_lines:]
    if ctx:
        await ctx.info(f"Returning {len(tail)} log lines for job {job_id}")
    return {"job_id": job_id, "log_path": state.log_path, "lines": tail, "total_lines": len(lines)}
