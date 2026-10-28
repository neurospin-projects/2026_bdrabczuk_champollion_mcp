"""Job wrapper: runs one stage command detached from the MCP server and records its exit code.

Started by runner.launch as the leader of a new session, with the job log as stdout and stderr.
It reads the stage argv and env from a one-shot handshake on stdin, runs the stage in its own
process group, and writes ``<job_id>.rc`` with the stage's exit code (-N for death by signal N).
It prints nothing on a normal run: its stdout and stderr are the job log.
"""

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import signal
import subprocess
import sys

from champollion_sulcal_mcp.job_store import validate_job_id, write_rc

_FORWARDED_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
_PREFIX = "champollion-mcp job wrapper"
_PR_SET_CHILD_SUBREAPER = 36


def _set_child_subreaper() -> bool:
    """Adopt stage descendants orphaned while this process lives; False (warned) when unavailable."""
    reason = None
    if not sys.platform.startswith("linux"):
        reason = "not Linux"
    else:
        try:
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.prctl(_PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0) != 0:
                reason = os.strerror(ctypes.get_errno())
        except AttributeError:
            reason = "prctl not found"
        except OSError as error:
            reason = str(error)
    if reason is None:
        return True
    print(
        f"{_PREFIX}: warning: child subreaper unavailable ({reason}); orphaned stage processes are not adopted",
        file=sys.stderr,
    )
    return False


def _wait_reaping(stage: subprocess.Popen) -> int:
    """Wait for the stage while reaping the orphans this subreaper adopted; return the stage's exit code."""
    while True:
        try:
            pid, status = os.waitpid(-1, 0)
        except ChildProcessError:
            return stage.wait()
        if pid == stage.pid:
            rc = os.waitstatus_to_exitcode(status)
            stage.returncode = rc
            return rc


def _reap_adopted() -> None:
    """Reap the adopted orphans that already exited, without waiting for live ones."""
    while True:
        try:
            pid, _status = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def _read_payload(stdin_fd: int) -> tuple[list[str], dict[str, str]] | None:
    """argv and env sent by the server on ``stdin_fd`` (read to EOF, then closed); None when malformed."""
    with open(stdin_fd, "rb", closefd=True) as handle:
        raw = handle.read()
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict):
        return None
    argv, env = payload.get("argv"), payload.get("env")
    if not isinstance(argv, list) or not argv or not all(isinstance(item, str) for item in argv):
        return None
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        return None
    return argv, env


def main(args: list[str], stdin_fd: int = 0) -> int:
    """Run the job of ``args`` = [output_dir, job_id]; return the wrapper's exit status.

    ``stdin_fd`` carries the launch handshake (the wrapper's fd 0 when started by runner.launch).
    """
    try:
        output_dir, job_id = args
        validate_job_id(job_id)
    except ValueError as error:
        print(f"{_PREFIX}: {error}", file=sys.stderr)
        return 2

    received: list[int] = []
    stage: subprocess.Popen | None = None

    def forward(signum, _frame) -> None:
        received.append(signum)
        if stage is not None and stage.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.kill(stage.pid, signum)

    for signum in _FORWARDED_SIGNALS:
        signal.signal(signum, forward)

    launch = _read_payload(stdin_fd)
    if launch is None:
        print(f"{_PREFIX}: invalid launch payload", file=sys.stderr)
        return 2
    argv, env = launch

    if received:
        write_rc(output_dir, job_id, -received[0])
        return 0

    adopting = _set_child_subreaper()
    try:
        stage = subprocess.Popen(argv, stdin=subprocess.DEVNULL, env=env, close_fds=True)  # noqa: S603
    except OSError as error:
        print(f"{_PREFIX}: cannot execute {argv[0]!r}: {error.strerror}", file=sys.stderr)
        write_rc(output_dir, job_id, 127 if isinstance(error, FileNotFoundError) else 126)
        return 0
    if received:
        with contextlib.suppress(ProcessLookupError):
            os.kill(stage.pid, received[-1])

    rc = _wait_reaping(stage) if adopting else stage.wait()
    try:
        write_rc(output_dir, job_id, rc)
    except Exception:
        import traceback

        traceback.print_exc()
        return 1
    if adopting:
        _reap_adopted()
    return 0
