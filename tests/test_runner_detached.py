"""Detached runner (TASK-MCPJOBS-BDRABCZUK-B0C778436617).

Root cause: stage processes died with the short-lived MCP client that launched them,
because the server owned their stdout pipe and their life. These tests launch short
real subprocesses (``python -c`` scripts that wait for a "go" file, then exit) in
synthetic tmp dirs, and check that the job keeps running and still records its exit
code once the launching process, or the server-side asyncio task, is gone.

Each test's docstring names the Requirement it verifies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

import champollion_sulcal_mcp
from champollion_sulcal_mcp import runner
from champollion_sulcal_mcp.job_store import JobState, jobs_dir, process_start_time, rc_path_for, read_job, read_rc

pytestmark = pytest.mark.integration

ENV = {"PATH": "/bin:/usr/bin"}
SRC_DIR = str(Path(champollion_sulcal_mcp.__file__).resolve().parent.parent)
WAIT_S = 10.0
TERMINAL = ("succeeded", "failed", "cancelled")

# Stage command body: wait (at most 20 s) for the go file named by argv[1], print a marker
# line, then exit with the code in argv[2].
GO_STAGE = (
    "import os, sys, time\n"
    "deadline = time.monotonic() + 20\n"
    "while not os.path.exists(sys.argv[1]) and time.monotonic() < deadline:\n"
    "    time.sleep(0.05)\n"
    "print('MARKER after go', flush=True)\n"
    "sys.exit(int(sys.argv[2]))\n"
)

# Runs in a separate Python process that exits as soon as runner.launch returns.
LAUNCHER = (
    "import asyncio, json, sys\n"
    "from champollion_sulcal_mcp import runner\n"
    "async def main():\n"
    "    state = await runner.launch(stage='test', argv=json.loads(sys.argv[1]), output_dir=sys.argv[2],\n"
    "                                cwd=sys.argv[3], env=json.loads(sys.argv[4]), args_snapshot={})\n"
    "    print(state.model_dump_json())\n"
    "asyncio.run(main())\n"
)


def _go_argv(go: Path, code: int) -> list[str]:
    return [sys.executable, "-c", GO_STAGE, str(go), str(code)]


def _launch_in_child_process(output_dir: Path, argv: list[str]) -> JobState:
    """Call runner.launch in a separate Python process; return the job once that process has exited."""
    done = subprocess.run(
        [sys.executable, "-c", LAUNCHER, json.dumps(argv), str(output_dir), str(output_dir), json.dumps(ENV)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "PYTHONPATH": SRC_DIR},
        check=False,
    )
    assert done.returncode == 0, done.stderr
    return JobState.model_validate_json(done.stdout.strip().splitlines()[-1])


def _wait(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


async def _await(predicate, timeout: float = WAIT_S) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


def _rc(output_dir: Path, job_id: str) -> int | None:
    try:
        return read_rc(str(output_dir), job_id)
    except ValueError:
        return None


def _log(state: JobState) -> str:
    path = Path(state.log_path)
    return path.read_text(errors="replace") if path.exists() else ""


def _terminal(output_dir: Path, job_id: str) -> bool:
    return read_job(str(output_dir), job_id).status in TERMINAL


def _proc_gone_or_reaped(pid: int) -> bool:
    """True when /proc/<pid> no longer exists (a zombie still has a /proc entry)."""
    return not Path(f"/proc/{pid}").exists()


async def _launch(output_dir: Path, argv: list[str], env: dict | None = None) -> JobState:
    return await runner.launch(
        stage="test",
        argv=argv,
        output_dir=str(output_dir),
        cwd=str(output_dir),
        env=dict(ENV if env is None else env),
        args_snapshot={},
    )


@pytest.fixture
def go(tmp_path):
    """Go file releasing GO_STAGE commands; always created at teardown so no stage outlives the test."""
    path = tmp_path / "go"
    yield path
    path.touch()


# --- the job survives its launcher ---------------------------------------------------------


def test_rc_written_after_launching_process_exits(tmp_output_dir, go):
    """REQ-MCPJOBS-BDRABCZUK-368894B3EE43.

    The wrapper writes the stage exit code to <job>.rc after the launching process exited.
    """
    state = _launch_in_child_process(tmp_output_dir, _go_argv(go, 3))
    go.touch()  # the stage ends only now, after the launching process has exited
    assert _wait(lambda: _rc(tmp_output_dir, state.job_id) is not None), (
        f"no exit-code file {rc_path_for(str(tmp_output_dir), state.job_id).name} after the launcher exited"
    )
    assert read_rc(str(tmp_output_dir), state.job_id) == 3


def test_log_complete_after_launching_process_exits(tmp_output_dir, go):
    """REQ-MCPJOBS-BDRABCZUK-4A66D8A334E9.

    Output the stage writes after the launching process exited reaches <job>.log.
    """
    state = _launch_in_child_process(tmp_output_dir, _go_argv(go, 0))
    go.touch()
    assert _wait(lambda: "MARKER after go" in _log(state)), (
        f"stage output written after the launcher exited is missing from the log: {_log(state)!r}"
    )


async def test_rc_written_after_server_task_cancelled(tmp_output_dir, go):
    """REQ-MCPJOBS-BDRABCZUK-59D4A085B77D.

    Cancelling the server's asyncio task for the job does not stop the .rc being written.
    """
    before = asyncio.all_tasks()
    state = await _launch(tmp_output_dir, _go_argv(go, 4))
    spawned = asyncio.all_tasks() - before - {asyncio.current_task()}
    for task in spawned:
        task.cancel()
    for task in spawned:
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
    go.touch()
    assert await _await(lambda: _rc(tmp_output_dir, state.job_id) is not None), (
        "no exit-code file after the server-side job task was cancelled"
    )
    assert read_rc(str(tmp_output_dir), state.job_id) == 4


# --- the job record --------------------------------------------------------------------------


async def test_record_pgid_equals_pid(tmp_output_dir, go):
    """REQ-MCPJOBS-BDRABCZUK-5485E422F619.

    The returned and stored record hold pgid == pid, and that is the process's real group.
    """
    state = await _launch(tmp_output_dir, _go_argv(go, 0))
    stored = read_job(str(tmp_output_dir), state.job_id)
    assert state.pid is not None
    assert state.pgid == state.pid, f"pgid {state.pgid} != pid {state.pid}"
    assert stored.pgid == stored.pid == state.pid
    assert os.getpgid(state.pid) == state.pgid


async def test_record_pid_start_time_matches_proc(tmp_output_dir, go):
    """REQ-MCPJOBS-BDRABCZUK-603EE25C4134.

    The record's pid_start_time equals the /proc start time of the recorded pid.
    """
    state = await _launch(tmp_output_dir, _go_argv(go, 0))
    stored = read_job(str(tmp_output_dir), state.job_id)
    assert state.pid_start_time is not None, "pid_start_time not recorded"
    assert state.pid_start_time == process_start_time(state.pid)
    assert stored.pid_start_time == state.pid_start_time


# --- the stage's standard streams --------------------------------------------------------


def _report_fd(fd: int, label: str) -> list[str]:
    stream = "sys.stderr" if fd == 2 else "sys.stdout"
    code = f"import os, sys; {stream}.write('{label}=' + os.readlink('/proc/self/fd/{fd}') + '\\n'); {stream}.flush()"
    return [sys.executable, "-c", code]


def _reported(state: JobState, label: str) -> str | None:
    for line in _log(state).splitlines():
        if line.startswith(f"{label}="):
            return line[len(label) + 1 :]
    return None


async def test_stage_stdin_is_devnull(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-F8D483873596: the stage's standard input is /dev/null, not the server's inherited stdin."""
    read_end, write_end = os.pipe()
    saved = os.dup(0)
    try:
        os.dup2(read_end, 0)  # the server's stdin is a pipe (as under an MCP STDIO client)
        state = await _launch(tmp_output_dir, _report_fd(0, "STDIN"))
    finally:
        os.dup2(saved, 0)
        for fd in (saved, read_end, write_end):
            os.close(fd)
    assert await _await(lambda: _reported(state, "STDIN") is not None), _log(state)
    assert _reported(state, "STDIN") == "/dev/null"


async def test_stage_stdout_is_log_file(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-11D09214F72D: the stage's standard output is the job's .log file itself (no pipe)."""
    state = await _launch(tmp_output_dir, _report_fd(1, "STDOUT"))
    assert await _await(lambda: _reported(state, "STDOUT") is not None), _log(state)
    assert _reported(state, "STDOUT") == os.path.realpath(state.log_path)


async def test_stage_stderr_is_log_file(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-8AA28568134B: the stage's standard error is the job's .log file itself (no pipe)."""
    state = await _launch(tmp_output_dir, _report_fd(2, "STDERR"))
    assert await _await(lambda: _reported(state, "STDERR") is not None), _log(state)
    assert _reported(state, "STDERR") == os.path.realpath(state.log_path)


# --- progress and reaping ---------------------------------------------------------------------


async def test_progress_read_from_log_file(tmp_output_dir, go):
    """REQ-MCPJOBS-BDRABCZUK-442351B63A67.

    A fold line appended to the job's .log file sets the record's progress within 5 s.
    """
    state = await _launch(tmp_output_dir, _go_argv(go, 0))
    with Path(state.log_path).open("ab") as log:
        log.write(b"Fold 3/4\n")

    def _progress() -> bool:
        progress = read_job(str(tmp_output_dir), state.job_id).progress
        return progress.fold_current == 3 and progress.fold_total == 4

    assert await _await(_progress, timeout=5.0), read_job(str(tmp_output_dir), state.job_id).progress


async def test_wrapper_reaped_by_server(tmp_output_dir, go):
    """REQ-MCPJOBS-BDRABCZUK-08D541C0272E.

    Once the job's process exits, the running server reaps it (no zombie) within 5 s.
    """
    state = await _launch(tmp_output_dir, _go_argv(go, 0))
    go.touch()
    assert await _await(lambda: _terminal(tmp_output_dir, state.job_id))
    assert await _await(lambda: _proc_gone_or_reaped(state.pid), timeout=5.0), f"pid {state.pid} not reaped"


# --- inputs and environment ---------------------------------------------------------------------


async def test_empty_argv_refused(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-03DFA69CD992: an empty argv raises ValueError and writes no job record."""
    with pytest.raises(ValueError):
        await _launch(tmp_output_dir, [])
    assert list(jobs_dir(str(tmp_output_dir)).glob("*.json")) == []


async def test_hf_token_not_in_job_json(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-38C458D25FB3: the HF_TOKEN value passed in env never appears in the job's JSON record."""
    secret = "hf_TestSecretValue0123456789"
    state = await _launch(tmp_output_dir, ["/bin/sh", "-c", "exit 0"], env={**ENV, "HF_TOKEN": secret})
    assert await _await(lambda: _terminal(tmp_output_dir, state.job_id))
    for record in jobs_dir(str(tmp_output_dir)).glob("*.json"):
        assert secret not in record.read_text()


async def test_stage_environment_equals_env(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-FD247ADCBF06: the stage runs with exactly the env mapping given to runner.launch."""
    env = {"PATH": "/bin:/usr/bin", "CHAMP_MARKER": "a b;c=$HOME"}
    state = await _launch(tmp_output_dir, ["/usr/bin/env"], env=env)
    assert await _await(lambda: _terminal(tmp_output_dir, state.job_id))
    assert sorted(_log(state).splitlines()) == sorted(f"{k}={v}" for k, v in env.items())


async def test_stage_argv_unchanged(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-126D536366A9.

    The stage receives exactly the argv given to runner.launch (no shell, no rewriting).
    """
    extra = ["a b", "--log", "$HOME", "x;y", ""]
    code = "import json, sys; print('ARGV=' + json.dumps(sys.argv[1:]))"
    state = await _launch(tmp_output_dir, [sys.executable, "-c", code, *extra])
    assert await _await(lambda: _reported(state, "ARGV") is not None), _log(state)
    assert json.loads(_reported(state, "ARGV")) == extra
