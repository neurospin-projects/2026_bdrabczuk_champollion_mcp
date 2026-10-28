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
