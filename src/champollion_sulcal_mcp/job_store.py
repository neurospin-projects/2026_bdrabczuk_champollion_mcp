from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import re
import stat
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

_JOBS_SUBDIR = ".mcp_jobs"
JOB_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")
_RC_RE = re.compile(r"-?[0-9]{1,3}")
_RC_MAX_BYTES = 64
_RC_LIMIT = 255
ERROR_PROCESS_GONE = "process gone without exit code (server exited or process killed); see log"


def validate_job_id(job_id: object) -> str:
    """Return job_id when it is a safe file-name stem; raise ValueError otherwise (never touches the filesystem)."""
    if not isinstance(job_id, str) or JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError(f"unsafe job id: {repr(job_id)[:80]}")
    return job_id


class JobProgress(BaseModel):
    fold_current: int | None = None
    fold_total: int | None = None
    percent: float | None = None
    current_stage: str | None = None
    stages_done: int | None = None
    stages_total: int | None = None
    current_child_job_id: str | None = None


class JobState(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    stage: str
    status: Literal["pending", "running", "succeeded", "failed", "cancelled"] = "pending"
    pid: int | None = Field(None, ge=1, strict=True)
    pgid: int | None = Field(None, ge=1, strict=True)
    pid_start_time: int | None = Field(None, ge=0, strict=True)
    output_dir: str
    log_path: str
    started_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    ended_at: str | None = None
    returncode: int | None = None
    progress: JobProgress = Field(default_factory=JobProgress)
    args: dict = Field(default_factory=dict)
    error: str | None = None

    @field_validator("job_id")
    @classmethod
    def _safe_job_id(cls, value: str) -> str:
        return validate_job_id(value)


# Per-job asyncio locks to prevent concurrent write races
_JOB_LOCKS: dict[str, asyncio.Lock] = {}
_LOCKS_LOCK = asyncio.Lock()


async def _get_lock(job_id: str) -> asyncio.Lock:
    async with _LOCKS_LOCK:
        if job_id not in _JOB_LOCKS:
            _JOB_LOCKS[job_id] = asyncio.Lock()
        return _JOB_LOCKS[job_id]


def release_lock(job_id: str) -> None:
    """Remove the per-job lock once the job is terminal. Called from runner._stream finally block."""
    _JOB_LOCKS.pop(job_id, None)


def jobs_dir(output_dir: str) -> Path:
    d = Path(output_dir) / _JOBS_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def job_path(output_dir: str, job_id: str) -> Path:
    validate_job_id(job_id)
    return jobs_dir(output_dir) / f"{job_id}.json"


def log_path_for(output_dir: str, job_id: str) -> Path:
    validate_job_id(job_id)
    return jobs_dir(output_dir) / f"{job_id}.log"


def rc_path_for(output_dir: str, job_id: str) -> Path:
    """Path of the job's exit-code file; creates nothing."""
    validate_job_id(job_id)
    return Path(output_dir) / _JOBS_SUBDIR / f"{job_id}.rc"


def _atomic_write_text(target: Path, text: str) -> None:
    """Write text through a unique temporary file in the target's directory, then rename it over target."""
    fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    try:
        try:
            f = os.fdopen(fd, "w", encoding="utf-8")
        except BaseException:
            os.close(fd)
            raise
        with f:
            os.fchmod(f.fileno(), 0o644)
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_job(state: JobState) -> None:
    _atomic_write_text(job_path(state.output_dir, state.job_id), state.model_dump_json(indent=2))


def write_rc(output_dir: str, job_id: str, returncode: int) -> None:
    """Record the job's exit code atomically in <job>.rc."""
    validate_job_id(job_id)
    if isinstance(returncode, bool) or not isinstance(returncode, int) or abs(returncode) > _RC_LIMIT:
        raise ValueError(f"exit code must be an int in -{_RC_LIMIT}..{_RC_LIMIT}, got {returncode!r}")
    _atomic_write_text(jobs_dir(output_dir) / f"{job_id}.rc", f"{returncode}\n")


def read_rc(output_dir: str, job_id: str) -> int | None:
    """Return the exit code in <job>.rc, None when absent; ValueError when the file is not a valid exit code."""
    path = rc_path_for(output_dir, job_id)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise ValueError(f"exit-code file {path.name} unreadable: {error.strerror}") from None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError(f"exit-code file {path.name} is not a regular file")
        data = os.read(fd, _RC_MAX_BYTES + 1)
    except OSError as error:
        raise ValueError(f"exit-code file {path.name} unreadable: {error.strerror}") from None
    finally:
        os.close(fd)
    if len(data) > _RC_MAX_BYTES or not data.isascii():
        raise ValueError(f"exit-code file {path.name} is not a short ASCII integer")
    text = data.decode("ascii").strip()
    if _RC_RE.fullmatch(text) is None or abs(int(text)) > _RC_LIMIT:
        raise ValueError(f"exit-code file {path.name} is not an exit code in -{_RC_LIMIT}..{_RC_LIMIT}")
    return int(text)


def process_start_time(pid: int) -> int | None:
    """Start time of pid in clock ticks since boot (/proc/<pid>/stat field 22); None when gone or a zombie."""
    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 1:
        raise ValueError(f"pid must be a positive int, got {pid!r}")
    if not os.path.exists("/proc/self/stat"):
        raise RuntimeError("process start time needs Linux /proc")
    try:
        data = Path(f"/proc/{pid}/stat").read_bytes()
    except (FileNotFoundError, ProcessLookupError):
        return None
    try:
        rest = data[data.rindex(b")") + 1 :].split()
        if rest[0] in (b"Z", b"X", b"x"):
            return None
        return int(rest[19])
    except IndexError:
        raise ValueError(f"malformed /proc/{pid}/stat") from None


def rc_outcome(returncode: int, *, now: str) -> dict:
    """Terminal job fields for an exit code."""
    if returncode == 0:
        return {"status": "succeeded", "returncode": 0, "error": None, "ended_at": now}
    return {
        "status": "failed",
        "returncode": returncode,
        "error": f"Process exited with code {returncode}. See log for details.",
        "ended_at": now,
    }


def reconcile(state: JobState) -> JobState:
    """Return the job's true state from its .rc file and /proc; pure: writes nothing."""
    if state.status != "running":
        return state
    now = datetime.now(UTC).isoformat()
    try:
        rc = read_rc(state.output_dir, state.job_id)
    except ValueError:
        return state.model_copy(
            update={
                "status": "failed",
                "returncode": None,
                "ended_at": now,
                "error": f"invalid exit-code file {state.job_id}.rc; see log",
            }
        )
    if rc is not None:
        return state.model_copy(update=rc_outcome(rc, now=now))
    if state.pid is None:
        return state
    try:
        start = process_start_time(state.pid)
    except (RuntimeError, OSError, ValueError) as error:
        logger.warning("Cannot check liveness of job %s (pid %s): %s", state.job_id, state.pid, error)
        return state
    if start is not None and (state.pid_start_time is None or start == state.pid_start_time):
        return state
    return state.model_copy(
        update={"status": "failed", "returncode": None, "ended_at": now, "error": ERROR_PROCESS_GONE}
    )


def read_job(output_dir: str, job_id: str) -> JobState:
    path = job_path(output_dir, job_id)
    if not path.exists():
        raise FileNotFoundError(f"Job {job_id} not found in {output_dir}")
    return JobState.model_validate_json(path.read_text())


def list_jobs(output_dir: str) -> list[JobState]:
    d = jobs_dir(output_dir)
    jobs = []
    for p in d.glob("*.json"):
        try:
            jobs.append(JobState.model_validate_json(p.read_text()))
        except Exception as exc:
            logger.warning("Skipping corrupt job file %s: %s", p, exc)
            continue
    return sorted(jobs, key=lambda j: j.started_at, reverse=True)


async def update_job(output_dir: str, job_id: str, **fields) -> JobState:
    lock = await _get_lock(job_id)
    async with lock:
        state = read_job(output_dir, job_id)
        # Handle nested progress updates
        if "progress" in fields and isinstance(fields["progress"], dict):
            current = state.progress.model_dump()
            current.update(fields.pop("progress"))
            fields["progress"] = JobProgress(**current)
        updated = state.model_copy(update=fields)
        write_job(updated)
        return updated
