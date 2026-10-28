from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from .job_store import (
    ERROR_PROCESS_GONE,
    JobState,
    jobs_dir,
    log_path_for,
    process_start_time,
    rc_outcome,
    read_job,
    read_rc,
    reconcile,
    release_lock,
    update_job,
    update_job_with,
    write_job,
)

FOLD_RE = re.compile(r"[Ff]old\s+(\d+)\s*(?:/|of)\s*(\d+)")
POLL_S = 0.5
CANCEL_GRACE_S = 10.0
MAX_PARTIAL_LINE = 65536
ERROR_WRAPPER_DIED = "job wrapper exited before the stage started; see log"
# Directory holding the champollion_sulcal_mcp package this server runs; the wrapper imports the same tree.
_PKG_ROOT = str(Path(__file__).resolve().parent.parent)
_WRAPPER_BOOT = (
    "import sys; sys.path.insert(0, sys.argv[1]); "
    "from champollion_sulcal_mcp.job_wrapper import main; sys.exit(main(sys.argv[2:]))"
)

# Watcher task and wrapper process of each job launched by this server
_TASKS: dict[str, asyncio.Task] = {}
_PROCS: dict[str, subprocess.Popen] = {}


def _check_launch_args(argv: object, env: object) -> None:
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        raise ValueError("argv must be a non-empty list of str")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError("env must map str to str")


def _reap_orphans() -> None:
    """Reap the wrappers whose watcher task was cancelled."""
    for job_id, proc in list(_PROCS.items()):
        if job_id not in _TASKS and proc.poll() is not None:
            _PROCS.pop(job_id, None)


def _abort_wrapper(proc: subprocess.Popen) -> None:
    """Close the handshake without a payload, so the wrapper exits without starting the stage."""
    with contextlib.suppress(OSError):
        proc.stdin.close()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=10)


async def launch(
    stage: str,
    argv: list[str],
    output_dir: str,
    cwd: str,
    env: dict,
    args_snapshot: dict,
) -> JobState:
    """Start ``argv`` as a detached job: own session, output in the job log, exit code in the .rc file.

    The stage runs under a job wrapper in its own session, so it outlives this server and the client.
    """
    _check_launch_args(argv, env)
    _reap_orphans()
    jobs_dir(output_dir)  # ensure directory exists
    job_id = str(uuid.uuid4())
    state = JobState(
        job_id=job_id,
        stage=stage,
        status="running",
        output_dir=output_dir,
        log_path=str(log_path_for(output_dir, job_id)),
        args=args_snapshot,
    )
    # The env (HF_TOKEN included) only travels through the handshake pipe, never the record, log or argv.
    payload = json.dumps({"argv": argv, "env": env}).encode()

    log_fd = os.open(state.log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_CLOEXEC, 0o644)
    try:
        proc = subprocess.Popen(  # noqa: S603 — fixed interpreter and boot code; argv goes through the pipe
            [sys.executable, "-I", "-c", _WRAPPER_BOOT, _PKG_ROOT, output_dir, job_id],
            stdin=subprocess.PIPE,
            stdout=log_fd,
            stderr=log_fd,
            cwd=cwd,
            env=dict(os.environ),
            start_new_session=True,
            close_fds=True,
        )
    finally:
        os.close(log_fd)

    try:
        pid = proc.pid  # session and group leader: pgid == pid
        start = process_start_time(pid)
        if start is None:
            _abort_wrapper(proc)
            state = state.model_copy(
                update={
                    "pid": pid,
                    "pgid": pid,
                    "status": "failed",
                    "returncode": None,
                    "ended_at": datetime.now(UTC).isoformat(),
                    "error": ERROR_WRAPPER_DIED,
                }
            )
            write_job(state)
            return state
        state = state.model_copy(update={"pid": pid, "pgid": pid, "pid_start_time": start})
        write_job(state)
    except BaseException:
        _abort_wrapper(proc)
        raise

    try:
        proc.stdin.write(payload)
    except BrokenPipeError:
        pass  # wrapper died in between: no .rc, so the watcher's reconcile fails the job
    finally:
        with contextlib.suppress(BrokenPipeError):
            proc.stdin.close()

    _PROCS[job_id] = proc
    _TASKS[job_id] = asyncio.create_task(_watch(proc, job_id, output_dir, state.log_path))
    return state


def _final_fields(state: JobState) -> dict | None:
    """Terminal fields of a job whose wrapper has been reaped (decided under the job lock)."""
    now = datetime.now(UTC).isoformat()
    if state.status == "running":
        result = reconcile(state)
        if result.status == "running":
            return None  # liveness not provable: left to reconcile-on-read
        return {
            "status": result.status,
            "returncode": result.returncode,
            "error": result.error,
            "ended_at": result.ended_at,
        }
    if state.status == "cancelled" and state.ended_at is None:
        try:
            rc = read_rc(state.output_dir, state.job_id)
        except ValueError:
            rc = None
        return {
            "returncode": rc,
            "ended_at": now,
            "error": rc_outcome(rc, now=now)["error"] if rc is not None else ERROR_PROCESS_GONE,
        }
    return None


async def _watch(proc: subprocess.Popen, job_id: str, output_dir: str, log_path: str) -> None:
    """Reap the wrapper, report fold progress from the job log, then record the job's end."""
    last: tuple[int, int] | None = None
    buf = b""
    log = Path(log_path).open("rb")  # noqa: SIM115 — read across await points, closed in finally
    try:
        while True:
            done = proc.poll() is not None
            data = log.read()
            if data:
                buf += data
                *lines, buf = buf.split(b"\n")
                if len(buf) > MAX_PARTIAL_LINE:
                    buf = b""
                fold = None
                for raw in lines:
                    match = FOLD_RE.search(raw.decode("utf-8", errors="replace"))
                    if match:
                        fold = (int(match.group(1)), int(match.group(2)))
                if fold is not None and fold != last and fold[1] > 0:
                    cur, total = fold
                    try:
                        await update_job(
                            output_dir,
                            job_id,
                            progress={"fold_current": cur, "fold_total": total, "percent": round(100 * cur / total, 1)},
                        )
                    except FileNotFoundError:
                        return
                    last = fold
            if done:
                with contextlib.suppress(FileNotFoundError):
                    await update_job_with(output_dir, job_id, _final_fields)
                return
            await asyncio.sleep(POLL_S)
    finally:
        log.close()
        release_lock(job_id)
        _TASKS.pop(job_id, None)
        if proc.returncode is not None:
            _PROCS.pop(job_id, None)


async def cancel(output_dir: str, job_id: str) -> JobState:
    state = read_job(output_dir, job_id)
    if state.status not in ("running", "pending"):
        return state

    # Mark cancelled first so the watcher keeps the status
    state = await update_job(output_dir, job_id, status="cancelled")

    proc = _PROCS.get(job_id)
    if proc is not None and proc.poll() is None:
        # The wrapper is unreaped, so its pid (== pgid) cannot have been reused.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + CANCEL_GRACE_S
        while loop.time() < deadline:
            await asyncio.sleep(0.1)
            if proc.poll() is not None:
                break
        else:
            if proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
    elif state.pid is not None:
        # Fallback: kill by PID
        with contextlib.suppress(ProcessLookupError):
            os.kill(state.pid, signal.SIGTERM)

    return read_job(output_dir, job_id)
