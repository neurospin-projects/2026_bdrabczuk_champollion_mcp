"""Detached pipeline driver: runs start_pipeline's stage chain in its own process.

start_pipeline launches this module through runner.launch, so the umbrella job is the driver's own job
(pid, pgid, .rc) and the chain reaches a terminal state after the MCP server exits. The umbrella id is
that of the running pipeline record whose process is this driver's job wrapper (its parent); the
output_dir comes from argv, the chain arguments from the umbrella record's ``args``.

Usage: python -P -m champollion_sulcal_mcp.pipeline_driver <output_dir>
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from datetime import UTC, datetime

import champollion_sulcal_mcp
from champollion_sulcal_mcp import runner
from champollion_sulcal_mcp.job_store import JobState, find_job_by_wrapper, read_job, update_job_with
from champollion_sulcal_mcp.tools import pipeline

logger = logging.getLogger(__name__)

STAGE_CANCEL_GRACE_S = 6.0  # driver-process grace for its stage jobs, below the server's 10 s
_STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
_PREFIX = "champollion-mcp pipeline driver"


def main(argv: list[str] | None = None) -> int:
    """Run the chain of the umbrella job this driver's wrapper runs; exit 2 on a usage error."""
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(f"usage: {_PREFIX} <output_dir>", file=sys.stderr)
        return 2
    job_id = find_job_by_wrapper(args[0], "pipeline", os.getppid())
    if job_id is None:
        print(f"{_PREFIX}: no running pipeline job of wrapper {os.getppid()} under {args[0]}", file=sys.stderr)
        return 2
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger.info("pipeline driver for job %s (package %s)", job_id, champollion_sulcal_mcp.__file__)
    runner.set_cancel_grace(STAGE_CANCEL_GRACE_S)
    return asyncio.run(drive(args[0], job_id))


async def run_chain(output_dir: str, job_id: str) -> None:
    """Run the umbrella job's stage chain in this process (the in-process entry)."""
    state = read_job(output_dir, job_id)
    if state.stage != "pipeline":
        raise ValueError(f"job {job_id} is a {state.stage!r} job, not a pipeline job")
    await pipeline._run_pipeline(
        umbrella_job_id=job_id,
        output_dir=output_dir,
        active_stages=state.args["active_stages"],
        **state.args["chain"],
    )


def _stopped_by(signame: str):
    def decide(state: JobState) -> dict | None:
        if state.status in ("running", "pending"):
            return {"status": "cancelled", "error": f"Pipeline driver stopped by {signame}"}
        return None

    return decide


def _install_stop_handlers(loop: asyncio.AbstractEventLoop, output_dir: str, job_id: str, writes: set) -> None:
    """On a stop signal mark the umbrella cancelled; the chain's next poll then cancels its stage and returns."""

    def on_signal(signum: int) -> None:
        write = asyncio.ensure_future(update_job_with(output_dir, job_id, _stopped_by(signal.Signals(signum).name)))
        writes.add(write)
        write.add_done_callback(writes.discard)

    for signum in _STOP_SIGNALS:
        loop.add_signal_handler(signum, on_signal, signum)


def _end_cancelled(state: JobState) -> dict | None:
    if state.status == "cancelled" and state.ended_at is None:
        return {"ended_at": datetime.now(UTC).isoformat()}
    return None


async def _exit_code(output_dir: str, job_id: str) -> int:
    """0 when the record carries the outcome; 1 when it still reads running or cannot be read."""
    try:
        state = await update_job_with(output_dir, job_id, _end_cancelled)
    except (FileNotFoundError, ValueError) as error:
        logger.error("umbrella job %s unreadable at exit: %s", job_id, error)
        return 1
    return 0 if state.status in ("succeeded", "failed", "cancelled") else 1


async def drive(output_dir: str, job_id: str) -> int:
    """Run the chain under stop-signal handlers and return the driver's exit code."""
    loop = asyncio.get_running_loop()
    writes: set[asyncio.Future] = set()
    _install_stop_handlers(loop, output_dir, job_id, writes)
    try:
        await run_chain(output_dir, job_id)
    except (FileNotFoundError, ValueError, KeyError, TypeError) as error:
        logger.error("pipeline job %s not started: %s: %s", job_id, type(error).__name__, error)
        return 2
    finally:
        for signum in _STOP_SIGNALS:
            loop.remove_signal_handler(signum)
        with contextlib.suppress(Exception):
            await asyncio.gather(*writes, return_exceptions=True)
    return await _exit_code(output_dir, job_id)


if __name__ == "__main__":
    sys.exit(main())
