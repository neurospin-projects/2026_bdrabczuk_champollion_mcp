"""In-process unit tests of job_wrapper.main (TASK-MCPJOBS-BDRABCZUK-B0C778436617).

tests/test_runner_detached.py runs the wrapper as its own process through runner.launch; these
tests call main() directly with a pipe as the handshake, so each branch is checked in isolation.
"""

from __future__ import annotations

import json
import os
import signal
import stat
import sys
from pathlib import Path

import pytest

from champollion_sulcal_mcp import job_wrapper
from champollion_sulcal_mcp.job_store import jobs_dir, rc_path_for, read_rc

JOB_ID = "job-wrapper-unit"
ENV = {"PATH": "/bin:/usr/bin"}


@pytest.fixture(autouse=True)
def _restore_signal_handlers():  # noqa: V103 - autouse fixture, used by pytest
    """main() installs handlers for SIGTERM, SIGINT and SIGHUP; put pytest's back afterwards."""
    saved = {signum: signal.getsignal(signum) for signum in job_wrapper._FORWARDED_SIGNALS}
    yield
    for signum, handler in saved.items():
        signal.signal(signum, handler)


@pytest.fixture(autouse=True)
def _no_child_subreaper(monkeypatch):  # noqa: V103 - autouse fixture, used by pytest
    """In-process main() must not make the pytest process a child subreaper."""
    monkeypatch.setattr(job_wrapper, "_set_child_subreaper", lambda: False)


def _handshake(payload: bytes) -> int:
    """Read end of a pipe that holds ``payload`` and is closed for writing."""
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload)
    os.close(write_fd)
    return read_fd


def _launch_payload(argv: list[str]) -> bytes:
    return json.dumps({"argv": argv, "env": ENV}).encode()


def _run(tmp_path, payload: bytes) -> int:
    jobs_dir(str(tmp_path))
    return job_wrapper.main([str(tmp_path), JOB_ID], stdin_fd=_handshake(payload))


def test_stage_exit_code_is_written_to_rc(tmp_path):
    """The wrapper waits for the stage and writes its exit code to the .rc file."""
    rc = _run(tmp_path, _launch_payload([sys.executable, "-c", "import sys; sys.exit(3)"]))
    assert rc == 0
    assert read_rc(str(tmp_path), JOB_ID) == 3


def test_stage_gets_only_the_handshake_env(tmp_path):
    """The stage runs with the env sent on the handshake, not the wrapper's own environment."""
    out = tmp_path / "env.json"
    code = f"import json, os; open({str(out)!r}, 'w').write(json.dumps(dict(os.environ)))"
    assert _run(tmp_path, _launch_payload([sys.executable, "-c", code])) == 0
    env = json.loads(out.read_text())
    assert env.get("PATH") == ENV["PATH"]
    assert "PYTHONPATH" not in env


@pytest.mark.parametrize(
    "payload",
    [b"", b"not json", b"[]", b'{"argv": [], "env": {}}', b'{"argv": ["x"], "env": {"A": 1}}'],
)
def test_invalid_payload_exits_2_without_rc(tmp_path, capsys, payload):
    """An empty or malformed handshake starts nothing and writes no .rc."""
    assert _run(tmp_path, payload) == 2
    assert "invalid launch payload" in capsys.readouterr().err
    assert not rc_path_for(str(tmp_path), JOB_ID).exists()


@pytest.mark.parametrize("args", [["only-one"], ["dir", "../escape"]])
def test_bad_arguments_exit_2(tmp_path, args):
    """A wrong argument count or an unsafe job id is refused before the handshake is read."""
    read_fd = _handshake(b"")
    try:
        assert job_wrapper.main(args, stdin_fd=read_fd) == 2
    finally:
        os.close(read_fd)


def test_missing_command_writes_127(tmp_path, capsys):
    """A stage command that does not exist gives exit code 127 and one log line."""
    assert _run(tmp_path, _launch_payload([str(tmp_path / "no-such-command")])) == 0
    assert read_rc(str(tmp_path), JOB_ID) == 127
    assert "cannot execute" in capsys.readouterr().err


def test_non_executable_command_writes_126(tmp_path):
    """A stage command that cannot be executed gives exit code 126."""
    script = tmp_path / "not-executable"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(stat.S_IRUSR | stat.S_IWUSR)
    assert _run(tmp_path, _launch_payload([str(script)])) == 0
    assert read_rc(str(tmp_path), JOB_ID) == 126


def test_rc_write_failure_exits_1(tmp_path, monkeypatch, capsys):
    """When the .rc cannot be written the wrapper prints the traceback and exits 1."""

    def fail(*_args):
        raise OSError("disk full")

    monkeypatch.setattr(job_wrapper, "write_rc", fail)
    assert _run(tmp_path, _launch_payload([sys.executable, "-c", "pass"])) == 1
    assert "disk full" in capsys.readouterr().err


def test_signal_before_start_records_negative_signal(tmp_path, monkeypatch):
    """A signal received before the stage starts: no stage, .rc holds -signum."""
    real_read = job_wrapper._read_payload

    def read_then_signal(stdin_fd):
        launch = real_read(stdin_fd)
        os.kill(os.getpid(), signal.SIGHUP)
        return launch

    monkeypatch.setattr(job_wrapper, "_read_payload", read_then_signal)
    marker = tmp_path / "started"
    code = f"open({str(marker)!r}, 'w').close()"
    assert _run(tmp_path, _launch_payload([sys.executable, "-c", code])) == 0
    assert read_rc(str(tmp_path), JOB_ID) == -signal.SIGHUP
    assert not marker.exists()


# --- child subreaper (TASK-MCPJOBS-BDRABCZUK-6E3BE75E16C9) ---------------------------------------
# These run the wrapper as its own process (session leader, as runner.launch starts it), so the
# pytest process itself never becomes a subreaper.

_SRC_DIR = str(Path(job_wrapper.__file__).resolve().parent.parent)
# argv: src dir, prelude code run before the wrapper is imported, then the wrapper's own args.
_WRAPPER_PROCESS_BOOT = (
    "import sys; sys.path.insert(0, sys.argv[1]); exec(sys.argv[2]); "
    "from champollion_sulcal_mcp.job_wrapper import main; sys.exit(main(sys.argv[3:]))"
)
# Makes every ctypes library load fail, as on a platform without a loadable libc / prctl.
_NO_LIBC_PRELUDE = (
    "import ctypes\n"
    "class _NoLib:\n"
    "    def __init__(self, *args, **kwargs):\n"
    "        raise OSError('libc unavailable (test)')\n"
    "ctypes.CDLL = ctypes.PyDLL = _NoLib\n"
    "ctypes.cdll = ctypes.LibraryLoader(_NoLib)\n"
    "ctypes.pydll = ctypes.LibraryLoader(_NoLib)\n"
)
# Stage: fork a middle process that forks a grandchild and exits at once; the orphaned grandchild
# waits to be re-parented, writes its new parent pid to argv[1], and the stage waits for that file.
_ORPHANING_STAGE = (
    "import os, sys, time\n"
    "report = sys.argv[1]\n"
    "middle = os.fork()\n"
    "if middle == 0:\n"
    "    if os.fork() == 0:\n"
    "        first = os.getppid()\n"
    "        deadline = time.monotonic() + 10\n"
    "        while os.getppid() == first and time.monotonic() < deadline:\n"
    "            time.sleep(0.01)\n"
    "        with open(report + '.tmp', 'w') as f:\n"
    "            f.write(str(os.getppid()))\n"
    "        os.replace(report + '.tmp', report)\n"
    "    os._exit(0)\n"
    "os.waitpid(middle, 0)\n"
    "deadline = time.monotonic() + 10\n"
    "while not os.path.exists(report) and time.monotonic() < deadline:\n"
    "    time.sleep(0.01)\n"
)


def _run_wrapper_process(tmp_path, argv: list[str], prelude: str = "pass") -> tuple[int, int, str]:
    """Run the wrapper as a separate session leader; return its pid, exit status and log text."""
    import subprocess

    jobs_dir(str(tmp_path))
    log = tmp_path / "wrapper.log"
    with log.open("wb") as out:
        proc = subprocess.Popen(
            [sys.executable, "-I", "-c", _WRAPPER_PROCESS_BOOT, _SRC_DIR, prelude, str(tmp_path), JOB_ID],
            stdin=subprocess.PIPE,
            stdout=out,
            stderr=out,
            start_new_session=True,
        )
        proc.communicate(_launch_payload(argv), timeout=60)
    return proc.pid, proc.returncode, log.read_text(errors="replace")


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="child subreaper is Linux-only")
def test_orphaned_stage_descendant_adopted_by_wrapper(tmp_path):
    """REQ-MCPJOBS-BDRABCZUK-7FC6B1695928.

    A stage grandchild orphaned while the stage runs is re-parented to the wrapper, not to init.
    """
    report = tmp_path / "adopter"
    wrapper_pid, status, log = _run_wrapper_process(tmp_path, [sys.executable, "-c", _ORPHANING_STAGE, str(report)])
    assert status == 0, log
    assert report.exists(), f"orphaned grandchild never reported its new parent; log: {log!r}"
    assert int(report.read_text()) == wrapper_pid, (
        f"orphan adopted by pid {report.read_text()}, not by the wrapper (pid {wrapper_pid})"
    )


def test_subreaper_unavailable_warns_and_runs_stage(tmp_path):
    """REQ-MCPJOBS-BDRABCZUK-88BE94EC1F2A.

    With no loadable libc (prctl unavailable) the wrapper logs a warning and still runs the stage.
    """
    _, status, log = _run_wrapper_process(tmp_path, [sys.executable, "-c", "import sys; sys.exit(5)"], _NO_LIBC_PRELUDE)
    assert status == 0, log
    assert read_rc(str(tmp_path), JOB_ID) == 5, log
    assert "child subreaper unavailable" in log, f"no subreaper warning in the job log: {log!r}"
