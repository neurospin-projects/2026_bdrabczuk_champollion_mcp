from __future__ import annotations

from pathlib import Path

import pytest

from champollion_sulcal_mcp.job_store import JobState, write_job
from champollion_sulcal_mcp.tools.jobs import cancel_job, get_job_log, get_job_status, list_jobs


@pytest.mark.unit
async def test_get_job_status_unknown_raises_toolerror(tmp_output_dir):
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="Unknown job_id"):
        await get_job_status(str(tmp_output_dir), "nonexistent-id")


@pytest.mark.unit
async def test_get_job_status_returns_state(tmp_output_dir):
    state = JobState(stage="embeddings", status="running", output_dir=str(tmp_output_dir), log_path="/tmp/fake.log")
    write_job(state)
    result = await get_job_status(str(tmp_output_dir), state.job_id)
    assert result["job_id"] == state.job_id
    assert result["status"] == "running"
    assert result["stage"] == "embeddings"


@pytest.mark.unit
async def test_list_jobs_filter_by_status(tmp_output_dir):
    s1 = JobState(stage="morphologist", status="succeeded", output_dir=str(tmp_output_dir), log_path="/tmp/a.log")
    s2 = JobState(stage="embeddings", status="failed", output_dir=str(tmp_output_dir), log_path="/tmp/b.log")
    s3 = JobState(stage="combine", status="succeeded", output_dir=str(tmp_output_dir), log_path="/tmp/c.log")
    for s in [s1, s2, s3]:
        write_job(s)

    all_jobs = await list_jobs(str(tmp_output_dir))
    assert len(all_jobs) == 3

    succeeded = await list_jobs(str(tmp_output_dir), status="succeeded")
    assert len(succeeded) == 2
    assert all(j["status"] == "succeeded" for j in succeeded)

    failed = await list_jobs(str(tmp_output_dir), status="failed")
    assert len(failed) == 1
    assert failed[0]["stage"] == "embeddings"


@pytest.mark.unit
async def test_list_jobs_limit(tmp_output_dir):
    for i in range(5):
        s = JobState(stage="morphologist", status="succeeded", output_dir=str(tmp_output_dir), log_path=f"/tmp/{i}.log")
        write_job(s)

    result = await list_jobs(str(tmp_output_dir), limit=3)
    assert len(result) == 3


@pytest.mark.unit
async def test_get_job_log_tail(tmp_output_dir):
    state = JobState(stage="embeddings", status="succeeded", output_dir=str(tmp_output_dir), log_path="")
    log_file = Path(tmp_output_dir) / ".mcp_jobs" / f"{state.job_id}.log"
    state = state.model_copy(update={"log_path": str(log_file)})
    write_job(state)

    lines = [f"line {i}\n" for i in range(500)]
    log_file.write_text("".join(lines))

    result = await get_job_log(str(tmp_output_dir), state.job_id, tail_lines=10)
    assert result["job_id"] == state.job_id
    assert len(result["lines"]) == 10
    assert result["lines"][-1] == "line 499"
    assert result["total_lines"] == 500


@pytest.mark.unit
async def test_get_job_log_cap_at_5000(tmp_output_dir):
    state = JobState(stage="combine", status="succeeded", output_dir=str(tmp_output_dir), log_path="")
    log_file = Path(tmp_output_dir) / ".mcp_jobs" / f"{state.job_id}.log"
    state = state.model_copy(update={"log_path": str(log_file)})
    write_job(state)
    log_file.write_text("\n".join(f"x{i}" for i in range(100)))

    result = await get_job_log(str(tmp_output_dir), state.job_id, tail_lines=9999)
    assert len(result["lines"]) <= 5000


@pytest.mark.unit
async def test_get_job_log_missing_log_raises_toolerror(tmp_output_dir):
    from fastmcp.exceptions import ToolError

    state = JobState(
        stage="config", status="succeeded", output_dir=str(tmp_output_dir), log_path="/nonexistent/path.log"
    )
    write_job(state)

    with pytest.raises(ToolError, match="not available"):
        await get_job_log(str(tmp_output_dir), state.job_id)


@pytest.mark.unit
async def test_get_job_log_unknown_job_raises_toolerror(tmp_output_dir):
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="Unknown job_id"):
        await get_job_log(str(tmp_output_dir), "no-such-job")


@pytest.mark.unit
async def test_cancel_terminal_job_noop(tmp_output_dir):
    state = JobState(stage="combine", status="succeeded", output_dir=str(tmp_output_dir), log_path="/tmp/noop.log")
    write_job(state)

    result = await cancel_job(str(tmp_output_dir), state.job_id)
    assert result["status"] == "succeeded"
    assert result["job_id"] == state.job_id


@pytest.mark.unit
async def test_cancel_unknown_job_raises_toolerror(tmp_output_dir):
    from fastmcp.exceptions import ToolError

    with pytest.raises(ToolError, match="Unknown job_id"):
        await cancel_job(str(tmp_output_dir), "ghost-job")


# --- Reconcile on every read (TASK-MCPJOBS-BDRABCZUK-BA67581C97D8) -----------------------------
# Liveness is faked through job_store.process_start_time: no real process is started or probed.

_LIVE_START = 123456


def _running_job(output_dir, *, pid: int, pid_start_time: int | None = _LIVE_START) -> JobState:
    state = JobState(
        stage="morphologist",
        status="running",
        pid=pid,
        pid_start_time=pid_start_time,
        output_dir=str(output_dir),
        log_path="",
    )
    log_file = Path(output_dir) / ".mcp_jobs" / f"{state.job_id}.log"
    state = state.model_copy(update={"log_path": str(log_file)})
    write_job(state)
    log_file.write_text("line 0\nline 1\n")
    return state


def _fake_liveness(monkeypatch, dead_pids: set[int]) -> None:
    """Pids in dead_pids are gone; any other pid is alive with the recorded start time."""
    from champollion_sulcal_mcp import job_store

    monkeypatch.setattr(job_store, "process_start_time", lambda pid: None if pid in dead_pids else _LIVE_START)


def _spy_reconcile(monkeypatch, fail_for: set[str] | None = None) -> list[str]:
    """Wrap job_store.reconcile; record the job ids it sees; raise for the ids in fail_for."""
    from champollion_sulcal_mcp import job_store

    calls: list[str] = []
    real = job_store.reconcile

    def spy(state):
        calls.append(state.job_id)
        if fail_for and state.job_id in fail_for:
            raise RuntimeError("fake reconcile failure")
        return real(state)

    monkeypatch.setattr(job_store, "reconcile", spy)
    return calls


def _record_fingerprint(output_dir, job_id: str) -> tuple:
    from champollion_sulcal_mcp import job_store

    path = job_store.job_path(str(output_dir), job_id)
    st = path.stat()
    return (st.st_ino, st.st_mtime_ns, path.read_bytes())


@pytest.mark.unit
async def test_get_job_status_returns_reconciled_state(tmp_output_dir, monkeypatch):
    from champollion_sulcal_mcp import job_store

    state = _running_job(tmp_output_dir, pid=900001)

    def fake_reconcile(s):
        return s.model_copy(update={"status": "failed", "error": "FAKE-RECONCILED", "ended_at": "2026-01-01T00:00:00"})

    monkeypatch.setattr(job_store, "reconcile", fake_reconcile)
    result = await get_job_status(str(tmp_output_dir), state.job_id)
    assert result["status"] == "failed"
    assert result["error"] == "FAKE-RECONCILED"


@pytest.mark.unit
async def test_get_job_status_dead_pid_reports_failed(tmp_output_dir, monkeypatch):
    from champollion_sulcal_mcp.job_store import ERROR_PROCESS_GONE

    state = _running_job(tmp_output_dir, pid=900002)
    _fake_liveness(monkeypatch, dead_pids={900002})
    result = await get_job_status(str(tmp_output_dir), state.job_id)
    assert result["status"] == "failed"
    assert result["returncode"] is None
    assert result["ended_at"] is not None
    assert result["error"] == ERROR_PROCESS_GONE


@pytest.mark.unit
async def test_get_job_status_persists_terminal_state(tmp_output_dir, monkeypatch):
    from champollion_sulcal_mcp import job_store

    state = _running_job(tmp_output_dir, pid=900003)
    _fake_liveness(monkeypatch, dead_pids={900003})
    await get_job_status(str(tmp_output_dir), state.job_id)
    stored = job_store.read_job(str(tmp_output_dir), state.job_id)
    assert stored.status == "failed"
    assert stored.error == job_store.ERROR_PROCESS_GONE
    assert stored.ended_at is not None


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["terminal", "live_running"])
async def test_get_job_status_unchanged_state_writes_nothing(tmp_output_dir, monkeypatch, kind):
    if kind == "terminal":
        state = JobState(stage="combine", status="succeeded", output_dir=str(tmp_output_dir), log_path="/tmp/x.log")
        write_job(state)
    else:
        state = _running_job(tmp_output_dir, pid=900004)
    _fake_liveness(monkeypatch, dead_pids=set())
    calls = _spy_reconcile(monkeypatch)
    before = _record_fingerprint(tmp_output_dir, state.job_id)
    await get_job_status(str(tmp_output_dir), state.job_id)
    assert calls == [state.job_id]
    assert _record_fingerprint(tmp_output_dir, state.job_id) == before


@pytest.mark.unit
async def test_list_jobs_returns_reconciled_states(tmp_output_dir, monkeypatch):
    from champollion_sulcal_mcp.job_store import ERROR_PROCESS_GONE

    dead = _running_job(tmp_output_dir, pid=900011)
    live = _running_job(tmp_output_dir, pid=900012)
    _fake_liveness(monkeypatch, dead_pids={900011})
    by_id = {j["job_id"]: j for j in await list_jobs(str(tmp_output_dir))}
    assert by_id[dead.job_id]["status"] == "failed"
    assert by_id[dead.job_id]["error"] == ERROR_PROCESS_GONE
    assert by_id[live.job_id]["status"] == "running"


@pytest.mark.unit
async def test_list_jobs_persists_terminal_states(tmp_output_dir, monkeypatch):
    from champollion_sulcal_mcp import job_store

    dead = _running_job(tmp_output_dir, pid=900013)
    live = _running_job(tmp_output_dir, pid=900014)
    _fake_liveness(monkeypatch, dead_pids={900013})
    await list_jobs(str(tmp_output_dir))
    stored = job_store.read_job(str(tmp_output_dir), dead.job_id)
    assert stored.status == "failed"
    assert stored.error == job_store.ERROR_PROCESS_GONE
    assert job_store.read_job(str(tmp_output_dir), live.job_id).status == "running"


@pytest.mark.unit
async def test_list_jobs_unchanged_states_write_nothing(tmp_output_dir, monkeypatch):
    done = JobState(stage="combine", status="succeeded", output_dir=str(tmp_output_dir), log_path="/tmp/x.log")
    write_job(done)
    live = _running_job(tmp_output_dir, pid=900015)
    _fake_liveness(monkeypatch, dead_pids=set())
    calls = _spy_reconcile(monkeypatch)
    before = {j: _record_fingerprint(tmp_output_dir, j) for j in (done.job_id, live.job_id)}
    await list_jobs(str(tmp_output_dir))
    assert sorted(calls) == sorted([done.job_id, live.job_id])
    assert {j: _record_fingerprint(tmp_output_dir, j) for j in before} == before


@pytest.mark.unit
async def test_list_jobs_filter_uses_reconciled_status(tmp_output_dir, monkeypatch):
    dead = _running_job(tmp_output_dir, pid=900016)
    _fake_liveness(monkeypatch, dead_pids={900016})
    running = await list_jobs(str(tmp_output_dir), status="running")
    assert dead.job_id not in [j["job_id"] for j in running]
    failed = await list_jobs(str(tmp_output_dir), status="failed")
    assert dead.job_id in [j["job_id"] for j in failed]


@pytest.mark.unit
async def test_list_jobs_reconcile_error_keeps_other_jobs(tmp_output_dir, monkeypatch):
    broken = _running_job(tmp_output_dir, pid=900017)
    dead = _running_job(tmp_output_dir, pid=900018)
    _fake_liveness(monkeypatch, dead_pids={900017, 900018})
    _spy_reconcile(monkeypatch, fail_for={broken.job_id})
    by_id = {j["job_id"]: j for j in await list_jobs(str(tmp_output_dir))}
    assert by_id[dead.job_id]["status"] == "failed"


@pytest.mark.unit
async def test_list_jobs_reconcile_error_returns_stored_state(tmp_output_dir, monkeypatch):
    broken = _running_job(tmp_output_dir, pid=900019)
    _running_job(tmp_output_dir, pid=900020)
    _fake_liveness(monkeypatch, dead_pids={900019, 900020})
    calls = _spy_reconcile(monkeypatch, fail_for={broken.job_id})
    by_id = {j["job_id"]: j for j in await list_jobs(str(tmp_output_dir))}
    assert broken.job_id in calls
    assert by_id[broken.job_id] == broken.model_dump()


@pytest.mark.unit
async def test_get_job_log_persists_terminal_state(tmp_output_dir, monkeypatch):
    from champollion_sulcal_mcp import job_store

    state = _running_job(tmp_output_dir, pid=900021)
    _fake_liveness(monkeypatch, dead_pids={900021})
    result = await get_job_log(str(tmp_output_dir), state.job_id)
    assert result["lines"] == ["line 0", "line 1"]
    stored = job_store.read_job(str(tmp_output_dir), state.job_id)
    assert stored.status == "failed"
    assert stored.error == job_store.ERROR_PROCESS_GONE


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["terminal", "live_running"])
async def test_get_job_log_unchanged_state_writes_nothing(tmp_output_dir, monkeypatch, kind):
    state = _running_job(tmp_output_dir, pid=900022)
    if kind == "terminal":
        state = state.model_copy(update={"status": "succeeded", "returncode": 0})
        write_job(state)
    _fake_liveness(monkeypatch, dead_pids=set())
    calls = _spy_reconcile(monkeypatch)
    before = _record_fingerprint(tmp_output_dir, state.job_id)
    await get_job_log(str(tmp_output_dir), state.job_id)
    assert calls == [state.job_id]
    assert _record_fingerprint(tmp_output_dir, state.job_id) == before


# --- cancel_job refusals (TASK-MCPJOBS-BDRABCZUK-6E3BE75E16C9) ----------------------------------
# Each case writes the running record a previous server process would have left, for throwaway
# processes. os.kill and os.killpg are replaced by recorders that never deliver a signal, so a
# wrong cancel cannot hit this process, its group or a reused pid.

_REAL_KILL = __import__("os").kill
_REFUSAL_REASONS = {
    "reused_pid": "pid reused, not signalled",
    "dead_pid": "process gone, not signalled",
    "server_group": "server process group, not signalled",
    "pid_not_pgid": "pid differs from pgid, not signalled",
}


@pytest.fixture
def throwaway_procs():
    """Factory for sleeping processes (own session unless told otherwise); all killed at teardown."""
    import signal
    import subprocess
    import sys

    procs: list[subprocess.Popen] = []

    def spawn(*, new_session: bool = True, seconds: float = 60) -> subprocess.Popen:
        proc = subprocess.Popen(
            [sys.executable, "-c", f"import time; time.sleep({seconds})"],
            stdin=subprocess.DEVNULL,
            start_new_session=new_session,
        )
        procs.append(proc)
        return proc

    yield spawn

    for proc in procs:
        if proc.poll() is None:
            _REAL_KILL(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)


@pytest.fixture
def no_signals(monkeypatch):
    """Record os.kill / os.killpg calls without delivering any signal."""
    import os

    calls: list[tuple[str, int, int]] = []
    monkeypatch.setattr(os, "kill", lambda pid, sig: calls.append(("kill", pid, sig)))
    monkeypatch.setattr(os, "killpg", lambda pgid, sig: calls.append(("killpg", pgid, sig)))
    return calls


def _refusal_record(case: str, output_dir, spawn) -> str:
    """Running record (pgid and pid_start_time set) for one refusal case; return its job id."""
    import os
    import uuid

    from champollion_sulcal_mcp.job_store import log_path_for, process_start_time

    if case == "reused_pid":
        proc = spawn()
        pid, pgid, start = proc.pid, proc.pid, process_start_time(proc.pid) + 1
    elif case == "dead_pid":
        proc = spawn(seconds=0.3)
        pid, pgid, start = proc.pid, proc.pid, process_start_time(proc.pid)
        proc.wait(timeout=10)
    elif case == "server_group":
        proc = spawn(new_session=False)
        pid, pgid, start = proc.pid, os.getpgrp(), process_start_time(proc.pid)
    elif case == "pid_not_pgid":
        proc, other = spawn(), spawn()
        pid, pgid, start = proc.pid, other.pid, process_start_time(proc.pid)
    else:
        raise AssertionError(case)
    assert start is not None
    job_id = f"refusal-{case.replace('_', '-')}-{uuid.uuid4().hex[:8]}"
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


async def _assert_refusal_error(output_dir, case: str, spawn) -> None:
    import re

    from fastmcp.exceptions import ToolError

    job_id = _refusal_record(case, output_dir, spawn)
    with pytest.raises(ToolError, match=re.escape(_REFUSAL_REASONS[case])):
        await cancel_job(str(output_dir), job_id)


@pytest.mark.integration
async def test_cancel_job_reused_pid_raises_toolerror(tmp_output_dir, throwaway_procs, no_signals):
    """REQ-MCPJOBS-BDRABCZUK-A0B8DED58C3B: a live pid with another start time raises "pid reused, not signalled"."""
    await _assert_refusal_error(tmp_output_dir, "reused_pid", throwaway_procs)


@pytest.mark.integration
async def test_cancel_job_dead_pid_raises_toolerror(tmp_output_dir, throwaway_procs, no_signals):
    """REQ-MCPJOBS-BDRABCZUK-50774AEA5442: a dead recorded pid raises "process gone, not signalled"."""
    await _assert_refusal_error(tmp_output_dir, "dead_pid", throwaway_procs)


@pytest.mark.integration
async def test_cancel_job_server_group_raises_toolerror(tmp_output_dir, throwaway_procs, no_signals):
    """REQ-MCPJOBS-BDRABCZUK-4794BEF4CE61: a pgid equal to the server's group raises "server process group, ..."."""
    await _assert_refusal_error(tmp_output_dir, "server_group", throwaway_procs)


@pytest.mark.integration
async def test_cancel_job_pid_not_pgid_raises_toolerror(tmp_output_dir, throwaway_procs, no_signals):
    """REQ-MCPJOBS-BDRABCZUK-4A38A43A5E25: a pgid differing from the pid raises "pid differs from pgid, ..."."""
    await _assert_refusal_error(tmp_output_dir, "pid_not_pgid", throwaway_procs)


@pytest.mark.integration
@pytest.mark.parametrize("case", sorted(_REFUSAL_REASONS))
async def test_cancel_job_refusal_keeps_reconciled_status(tmp_output_dir, throwaway_procs, no_signals, case):
    """REQ-MCPJOBS-BDRABCZUK-A2DD38343F7A.

    A cancel that signals nothing leaves the stored status equal to what reconcile returns for the
    record (failed for a reused or dead pid, running otherwise), never cancelled.
    """
    import contextlib

    from fastmcp.exceptions import ToolError

    from champollion_sulcal_mcp import job_store

    job_id = _refusal_record(case, tmp_output_dir, throwaway_procs)
    expected = job_store.reconcile(job_store.read_job(str(tmp_output_dir), job_id)).status

    with contextlib.suppress(ToolError):
        await cancel_job(str(tmp_output_dir), job_id)

    assert job_store.read_job(str(tmp_output_dir), job_id).status == expected
