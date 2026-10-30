from __future__ import annotations

import asyncio
import contextlib

import pytest

from champollion_sulcal_mcp import runner
from champollion_sulcal_mcp.runner import FOLD_RE


@pytest.mark.unit
@pytest.mark.parametrize(
    "line",
    [
        "Fold 3/56",
        "fold 12 of 56",
        "Processing fold 1/56",
        "fold 28 of 56 done",
    ],
)
def test_fold_regex_matches(line):
    m = FOLD_RE.search(line)
    assert m is not None
    assert int(m.group(1)) >= 1
    assert int(m.group(2)) >= 1


@pytest.mark.integration
async def test_launch_echo_success(tmp_output_dir):
    state = await runner.launch(
        stage="test",
        argv=["/bin/sh", "-c", "echo hello world; exit 0"],
        output_dir=str(tmp_output_dir),
        cwd="/tmp",
        env={"PATH": "/bin:/usr/bin"},
        args_snapshot={},
    )
    assert state.status == "running"
    # Wait for completion
    for _ in range(30):
        await asyncio.sleep(0.2)
        from champollion_sulcal_mcp.job_store import read_job

        s = read_job(str(tmp_output_dir), state.job_id)
        if s.status in ("succeeded", "failed"):
            break
    assert s.status == "succeeded"
    assert s.returncode == 0
    from pathlib import Path

    log = Path(s.log_path).read_text()
    assert "hello world" in log


@pytest.mark.integration
async def test_launch_failure_sets_failed(tmp_output_dir):
    state = await runner.launch(
        stage="test",
        argv=["/bin/sh", "-c", "exit 7"],
        output_dir=str(tmp_output_dir),
        cwd="/tmp",
        env={"PATH": "/bin:/usr/bin"},
        args_snapshot={},
    )
    for _ in range(20):
        await asyncio.sleep(0.2)
        from champollion_sulcal_mcp.job_store import read_job

        s = read_job(str(tmp_output_dir), state.job_id)
        if s.status in ("succeeded", "failed", "cancelled"):
            break
    assert s.status == "failed"
    assert s.returncode == 7
    assert "7" in s.error
    # Error message must not leak raw stderr
    assert "Traceback" not in (s.error or "")


@pytest.mark.integration
async def test_cancel_terminates(tmp_output_dir):
    state = await runner.launch(
        stage="test",
        argv=["/bin/sh", "-c", "sleep 30"],
        output_dir=str(tmp_output_dir),
        cwd="/tmp",
        env={"PATH": "/bin:/usr/bin"},
        args_snapshot={},
    )
    await asyncio.sleep(0.3)
    await runner.cancel(str(tmp_output_dir), state.job_id)
    # Wait for finalization
    for _ in range(25):
        await asyncio.sleep(0.5)
        from champollion_sulcal_mcp.job_store import read_job

        s = read_job(str(tmp_output_dir), state.job_id)
        if s.ended_at is not None:
            break
    assert s.status == "cancelled"


@pytest.mark.integration
async def test_progress_updates(tmp_output_dir):
    state = await runner.launch(
        stage="test",
        argv=["/bin/sh", "-c", "for i in 1 2 3; do echo Fold $i/3; sleep 0.05; done"],
        output_dir=str(tmp_output_dir),
        cwd="/tmp",
        env={"PATH": "/bin:/usr/bin"},
        args_snapshot={},
    )
    for _ in range(30):
        await asyncio.sleep(0.2)
        from champollion_sulcal_mcp.job_store import read_job

        s = read_job(str(tmp_output_dir), state.job_id)
        if s.status == "succeeded":
            break
    assert s.progress.fold_current == 3
    assert s.progress.fold_total == 3


# --- cancel of a job launched by another server process (TASK-MCPJOBS-BDRABCZUK-6E3BE75E16C9) ---
#
# Such a job is absent from this server's process table, so runner.cancel works from the
# record alone. Each test spawns a throwaway process group (a leader in its own session plus
# one child) and writes a running job record for it, as a previous server process would have.

GRACE_S = 0.3
DEATH_WAIT_S = 5.0
CANCEL_TIMEOUT_S = 30.0

# One member of the throwaway group: argv = mode ("term" or "ignore"), marks dir, role, own code.
# On SIGTERM ("term" mode) it writes <marks>/<role>.term and exits; the leader starts the child.
GROUP_MEMBER = r"""
import os, signal, subprocess, sys, time
mode, marks, role, code = sys.argv[1:5]
def on_term(signum, frame):
    with open(os.path.join(marks, role + ".term"), "w") as f:
        f.write(str(os.getpid()))
    os._exit(143)
signal.signal(signal.SIGTERM, signal.SIG_IGN if mode == "ignore" else on_term)
if role == "leader":
    subprocess.Popen([sys.executable, "-c", code, mode, marks, "child", code])
else:
    tmp = os.path.join(marks, "child.pid.tmp")
    with open(tmp, "w") as f:
        f.write(str(os.getpid()))
    os.replace(tmp, os.path.join(marks, "child.pid"))
deadline = time.monotonic() + 60
while time.monotonic() < deadline:
    time.sleep(0.05)
"""


def _alive(pid: int) -> bool:
    from champollion_sulcal_mcp.job_store import process_start_time

    return process_start_time(pid) is not None


async def _until(predicate, timeout: float) -> bool:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.05)
    return predicate()


def _write_running_record(output_dir, pid: int, pgid: int | None, start: int | None) -> str:
    """Write the running job record a previous server process would have left; return its job id."""
    import uuid

    from champollion_sulcal_mcp.job_store import JobState, log_path_for, write_job

    job_id = f"foreign-{uuid.uuid4().hex[:12]}"
    write_job(
        JobState(
            job_id=job_id,
            stage="test",
            status="running",
            pid=pid,
            pgid=pgid,
            pid_start_time=start,
            output_dir=str(output_dir),
            log_path=str(log_path_for(str(output_dir), job_id)),
        )
    )
    return job_id


@pytest.fixture
def foreign_group(tmp_path):
    """Factory spawning a leader in its own session plus one child; both are SIGKILLed at teardown."""
    import os
    import signal
    import subprocess
    import sys
    import threading
    import time

    from champollion_sulcal_mcp.job_store import process_start_time

    members: list[tuple[int, int | None]] = []
    leaders: list[subprocess.Popen] = []

    def spawn(mode: str) -> dict:
        marks = tmp_path / f"marks-{len(leaders)}"
        marks.mkdir()
        leader = subprocess.Popen(
            [sys.executable, "-c", GROUP_MEMBER, mode, str(marks), "leader", GROUP_MEMBER],
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        leaders.append(leader)
        # Reap the leader as soon as it dies, as init would for a job left by a previous server.
        threading.Thread(target=leader.wait, daemon=True).start()
        members.append((leader.pid, process_start_time(leader.pid)))
        child_pid_file = marks / "child.pid"
        deadline = time.monotonic() + 15
        while not child_pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert child_pid_file.exists(), "throwaway group child did not start"
        child = int(child_pid_file.read_text())
        members.append((child, process_start_time(child)))
        assert os.getpgid(child) == leader.pid
        return {"leader": leader.pid, "child": child, "marks": marks}

    yield spawn

    for pid, start in members:
        if start is not None and process_start_time(pid) == start:
            with contextlib.suppress(ProcessLookupError):
                os.kill(pid, signal.SIGKILL)
    for leader in leaders:
        with contextlib.suppress(Exception):
            leader.wait(timeout=10)


@pytest.fixture
def signal_spy(monkeypatch):
    """Record os.kill / os.killpg calls; never forward one aimed at this process, its group or a blocked target."""
    import os

    calls: list[tuple[str, int, int]] = []
    blocked: set[int] = set()
    real_kill, real_killpg = os.kill, os.killpg

    def kill(pid, sig):
        calls.append(("kill", pid, sig))
        if pid <= 0 or pid == os.getpid() or abs(pid) in blocked:
            return
        real_kill(pid, sig)

    def killpg(pgid, sig):
        calls.append(("killpg", pgid, sig))
        if pgid <= 0 or pgid == os.getpgrp() or pgid in blocked:
            return
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, "kill", kill)
    monkeypatch.setattr(os, "killpg", killpg)
    return {"calls": calls, "blocked": blocked}


def _signals_to(calls: list[tuple[str, int, int]], pid: int, pgid: int) -> list[tuple[str, int, int]]:
    return [
        call
        for call in calls
        if (call[0] == "kill" and abs(call[1]) in (pid, pgid)) or (call[0] == "killpg" and call[1] == pgid)
    ]


async def _cancel_tolerating_refusal(output_dir, job_id: str) -> None:
    """runner.cancel for a job it must not signal; a refusal may raise (test_jobs_tools checks the error)."""
    with contextlib.suppress(Exception):
        await asyncio.wait_for(runner.cancel(str(output_dir), job_id), CANCEL_TIMEOUT_S)


@pytest.mark.integration
async def test_cancel_foreign_job_sends_sigterm_to_group(tmp_output_dir, foreign_group, monkeypatch):
    """REQ-MCPJOBS-BDRABCZUK-6750870DEB69.

    Cancelling a live job launched by another server process sends SIGTERM to every member of its
    recorded group: the child, not only the leader, receives SIGTERM, and both die.
    """
    from champollion_sulcal_mcp.job_store import process_start_time

    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    group = foreign_group("term")
    leader, child = group["leader"], group["child"]
    job_id = _write_running_record(tmp_output_dir, leader, leader, process_start_time(leader))

    await asyncio.wait_for(runner.cancel(str(tmp_output_dir), job_id), CANCEL_TIMEOUT_S)

    assert await _until(lambda: not _alive(leader) and not _alive(child), DEATH_WAIT_S), (
        f"group survived cancel: leader alive={_alive(leader)}, child alive={_alive(child)}"
    )
    assert (group["marks"] / "child.term").exists(), "the child of the recorded process group received no SIGTERM"
    assert (group["marks"] / "leader.term").exists(), "the group leader received no SIGTERM"


@pytest.mark.integration
async def test_cancel_foreign_job_escalates_to_sigkill(tmp_output_dir, foreign_group, monkeypatch):
    """REQ-MCPJOBS-BDRABCZUK-3461750BAB09.

    A group whose members ignore SIGTERM is SIGKILLed once CANCEL_GRACE_S has elapsed: both die.
    """
    from champollion_sulcal_mcp.job_store import process_start_time

    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    group = foreign_group("ignore")
    leader, child = group["leader"], group["child"]
    job_id = _write_running_record(tmp_output_dir, leader, leader, process_start_time(leader))

    await asyncio.wait_for(runner.cancel(str(tmp_output_dir), job_id), CANCEL_TIMEOUT_S)

    assert await _until(lambda: not _alive(leader) and not _alive(child), DEATH_WAIT_S), (
        f"SIGTERM-ignoring group survived cancel: leader alive={_alive(leader)}, child alive={_alive(child)}"
    )


@pytest.mark.integration
async def test_cancel_foreign_job_reused_pid_not_signalled(tmp_output_dir, foreign_group, signal_spy, monkeypatch):
    """REQ-MCPJOBS-BDRABCZUK-DD7F66265C68.

    A live recorded pid whose start time differs from pid_start_time (pid reuse) gets no signal,
    nor does its group: both processes stay alive.
    """
    from champollion_sulcal_mcp.job_store import process_start_time

    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    group = foreign_group("term")
    leader, child = group["leader"], group["child"]
    job_id = _write_running_record(tmp_output_dir, leader, leader, process_start_time(leader) + 1)

    await _cancel_tolerating_refusal(tmp_output_dir, job_id)
    await asyncio.sleep(GRACE_S + 0.2)

    assert _signals_to(signal_spy["calls"], leader, leader) == []
    assert _alive(leader) and _alive(child), "a process whose pid was reused was signalled"


@pytest.mark.integration
async def test_cancel_foreign_job_dead_pid_not_signalled(tmp_output_dir, signal_spy, monkeypatch):
    """REQ-MCPJOBS-BDRABCZUK-3691E662E6E0.

    A recorded pid naming no live process gets no signal, nor does its recorded group.
    """
    import subprocess
    import sys

    from champollion_sulcal_mcp.job_store import process_start_time

    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    gone = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(0.5)"], stdin=subprocess.DEVNULL, start_new_session=True
    )
    start = process_start_time(gone.pid)
    gone.wait(timeout=10)
    # Should the pid be reused meanwhile, the spy keeps any signal from reaching the new process.
    signal_spy["blocked"].add(gone.pid)
    job_id = _write_running_record(tmp_output_dir, gone.pid, gone.pid, start)

    await _cancel_tolerating_refusal(tmp_output_dir, job_id)

    assert _signals_to(signal_spy["calls"], gone.pid, gone.pid) == []


@pytest.mark.integration
async def test_cancel_job_in_server_group_not_signalled(tmp_output_dir, signal_spy, monkeypatch):
    """REQ-MCPJOBS-BDRABCZUK-1FB794D9A5D0.

    A job recorded in the server's own process group gets no signal: neither the group (the spy
    never forwards a signal to this process's group) nor the recorded pid, which stays alive.
    """
    import os
    import subprocess
    import sys

    from champollion_sulcal_mcp.job_store import process_start_time

    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    same_group = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], stdin=subprocess.DEVNULL)
    try:
        assert os.getpgid(same_group.pid) == os.getpgrp()
        job_id = _write_running_record(tmp_output_dir, same_group.pid, os.getpgrp(), process_start_time(same_group.pid))

        await _cancel_tolerating_refusal(tmp_output_dir, job_id)
        await asyncio.sleep(GRACE_S + 0.2)

        assert _signals_to(signal_spy["calls"], same_group.pid, os.getpgrp()) == []
        assert same_group.poll() is None, "the recorded pid in the server's own group was signalled"
    finally:
        same_group.kill()
        same_group.wait(timeout=10)


@pytest.mark.integration
async def test_cancel_foreign_job_records_cancelled_terminal_state(tmp_output_dir, foreign_group, monkeypatch):
    """REQ-MCPJOBS-BDRABCZUK-8C2A1B2F88F6.

    Once runner.cancel returns after signalling the group, the stored record is cancelled and ended.
    """
    from champollion_sulcal_mcp.job_store import process_start_time, read_job

    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    group = foreign_group("term")
    leader = group["leader"]
    job_id = _write_running_record(tmp_output_dir, leader, leader, process_start_time(leader))

    await asyncio.wait_for(runner.cancel(str(tmp_output_dir), job_id), CANCEL_TIMEOUT_S)

    stored = read_job(str(tmp_output_dir), job_id)
    assert stored.status == "cancelled"
    assert stored.ended_at is not None, "cancelled job record has no ended_at"


@pytest.mark.integration
async def test_cancel_foreign_job_pid_not_pgid_not_signalled(tmp_output_dir, foreign_group, signal_spy, monkeypatch):
    """REQ-MCPJOBS-BDRABCZUK-6A1270FAEBB7.

    A record whose pgid differs from its pid gets no signal, neither to the pid nor to either group.
    """
    from champollion_sulcal_mcp.job_store import process_start_time

    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    first, second = foreign_group("term"), foreign_group("term")
    pid, pgid = first["leader"], second["leader"]
    job_id = _write_running_record(tmp_output_dir, pid, pgid, process_start_time(pid))

    await _cancel_tolerating_refusal(tmp_output_dir, job_id)
    await asyncio.sleep(GRACE_S + 0.2)

    assert _signals_to(signal_spy["calls"], pid, pid) == []
    assert _signals_to(signal_spy["calls"], pgid, pgid) == []
    assert all(_alive(member[role]) for member in (first, second) for role in ("leader", "child"))


@pytest.mark.integration
async def test_cancel_legacy_job_without_pgid_not_signalled(tmp_output_dir, foreign_group, signal_spy, monkeypatch):
    """REQ-MCPJOBS-BDRABCZUK-88AE35B2025C: a legacy record with no pgid gets no signal to its live pid."""
    from champollion_sulcal_mcp.job_store import process_start_time

    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    leader = foreign_group("term")["leader"]
    job_id = _write_running_record(tmp_output_dir, leader, None, process_start_time(leader))

    await _cancel_tolerating_refusal(tmp_output_dir, job_id)
    await asyncio.sleep(GRACE_S + 0.2)

    assert _signals_to(signal_spy["calls"], leader, leader) == []
    assert _alive(leader), "the recorded pid of a record without pgid was signalled"


@pytest.mark.integration
async def test_cancel_legacy_job_without_start_time_not_signalled(
    tmp_output_dir, foreign_group, signal_spy, monkeypatch
):
    """REQ-MCPJOBS-BDRABCZUK-E64EC07425C7: a legacy record with no pid_start_time gets no signal, nor does its group."""
    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    group = foreign_group("term")
    leader = group["leader"]
    job_id = _write_running_record(tmp_output_dir, leader, leader, None)

    await _cancel_tolerating_refusal(tmp_output_dir, job_id)
    await asyncio.sleep(GRACE_S + 0.2)

    assert _signals_to(signal_spy["calls"], leader, leader) == []
    assert _alive(leader) and _alive(group["child"]), "a record without pid_start_time was signalled"


@pytest.mark.integration
@pytest.mark.parametrize("missing", ["pgid", "pid_start_time"])
async def test_cancel_legacy_job_records_cancelled_without_signal(
    tmp_output_dir, foreign_group, signal_spy, monkeypatch, missing
):
    """REQ-MCPJOBS-BDRABCZUK-A9EF221C7647.

    A legacy running record (no pgid, or no pid_start_time) is stored cancelled, ended, with an
    error saying no process was signalled.
    """
    from champollion_sulcal_mcp.job_store import process_start_time, read_job

    monkeypatch.setattr(runner, "CANCEL_GRACE_S", GRACE_S)
    leader = foreign_group("term")["leader"]
    signal_spy["blocked"].add(leader)  # keep the throwaway alive whatever cancel does
    pgid = None if missing == "pgid" else leader
    start = None if missing == "pid_start_time" else process_start_time(leader)
    job_id = _write_running_record(tmp_output_dir, leader, pgid, start)

    await asyncio.wait_for(runner.cancel(str(tmp_output_dir), job_id), CANCEL_TIMEOUT_S)

    stored = read_job(str(tmp_output_dir), job_id)
    assert stored.status == "cancelled"
    assert stored.ended_at is not None, "legacy cancelled record has no ended_at"
    assert "no process was signalled" in (stored.error or ""), f"error is {stored.error!r}"
