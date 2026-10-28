from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from champollion_sulcal_mcp import job_store as js
from champollion_sulcal_mcp.job_store import JobState, list_jobs, read_job, update_job, write_job


@pytest.mark.unit
def test_write_read_roundtrip(tmp_output_dir):
    state = JobState(stage="morphologist", output_dir=str(tmp_output_dir), log_path="/tmp/test.log")
    write_job(state)
    loaded = read_job(str(tmp_output_dir), state.job_id)
    assert loaded.job_id == state.job_id
    assert loaded.stage == "morphologist"
    assert loaded.status == "pending"


@pytest.mark.unit
def test_list_jobs_sorted(tmp_output_dir):
    jobs = [
        JobState(
            stage="morphologist",
            output_dir=str(tmp_output_dir),
            log_path="/tmp/a.log",
            started_at="2024-01-01T00:00:00+00:00",
        ),
        JobState(
            stage="cortical_tiles",
            output_dir=str(tmp_output_dir),
            log_path="/tmp/b.log",
            started_at="2024-01-03T00:00:00+00:00",
        ),
        JobState(
            stage="embeddings",
            output_dir=str(tmp_output_dir),
            log_path="/tmp/c.log",
            started_at="2024-01-02T00:00:00+00:00",
        ),
    ]
    for j in jobs:
        write_job(j)
    result = list_jobs(str(tmp_output_dir))
    assert [r.stage for r in result] == ["cortical_tiles", "embeddings", "morphologist"]


@pytest.mark.unit
async def test_update_job_atomic(tmp_output_dir):
    state = JobState(stage="morphologist", output_dir=str(tmp_output_dir), log_path="/tmp/test.log")
    write_job(state)
    updated = await update_job(str(tmp_output_dir), state.job_id, status="succeeded", returncode=0)
    assert updated.status == "succeeded"
    assert updated.returncode == 0
    reread = read_job(str(tmp_output_dir), state.job_id)
    assert reread.status == "succeeded"


@pytest.mark.unit
def test_jobs_dir_created(tmp_path):
    from champollion_sulcal_mcp.job_store import jobs_dir

    out = str(tmp_path / "newout")
    d = jobs_dir(out)
    assert d.exists()
    assert d.name == ".mcp_jobs"


# --- TASK-MCPJOBS-BDRABCZUK-690C7A8D9A8C: pid/start time/pgid, <job>.rc, reconcile ---
# New job_store names are reached through the module (js.<name>) so this file still
# collects before they exist; a missing name fails only the tests that need it.

GONE = "process gone without exit code"

linux_only = pytest.mark.skipif(not Path("/proc/self/stat").exists(), reason="needs Linux /proc")

UNSAFE_JOB_IDS = [
    "../escape",
    "a/b",
    "..",
    "",
    "x" * 65,
    "a\x00b",
    "/abs",
    ".hidden",
    "-lead",
    "a b",
]


def _start_ticks(pid: int) -> int:
    """Field 22 (starttime) of /proc/<pid>/stat; the comm field may contain spaces or parentheses."""
    after_comm = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(after_comm[19])


def _dead_pid() -> int:
    """Pid of a child that has exited and been reaped: no process holds it any more."""
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _running(out, **fields) -> JobState:
    return JobState(
        stage="morphologist",
        status=fields.pop("status", "running"),
        output_dir=str(out),
        log_path=str(Path(out) / "job.log"),
        **fields,
    )


def _write_rc_file(out, job_id: str, content: str | bytes) -> Path:
    path = js.jobs_dir(str(out)) / f"{job_id}.rc"
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)
    return path


def _snapshot(out) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in sorted(js.jobs_dir(str(out)).iterdir())}


def _assert_utc_iso(stamp: str | None) -> None:
    assert stamp is not None
    parsed = datetime.fromisoformat(stamp)
    assert parsed.utcoffset() == timedelta(0)


# REQ #0: pgid and pid_start_time persisted
@pytest.mark.unit
def test_record_roundtrips_pgid_and_start_time(tmp_output_dir):
    state = _running(tmp_output_dir, pid=4242, pgid=4242, pid_start_time=987654)
    write_job(state)
    loaded = read_job(str(tmp_output_dir), state.job_id)
    assert loaded.pgid == 4242
    assert loaded.pid_start_time == 987654
    assert loaded.pid == 4242


# REQ #1: pid / pgid below 1 rejected
@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, -4242])
def test_record_rejects_non_positive_pid(tmp_output_dir, bad):
    with pytest.raises(ValidationError):
        _running(tmp_output_dir, pid=bad)


@pytest.mark.unit
@pytest.mark.parametrize("bad", [0, -1, -4242])
def test_record_rejects_non_positive_pgid(tmp_output_dir, bad):
    with pytest.raises(ValidationError):
        _running(tmp_output_dir, pid=4242, pgid=bad)


# REQ #2: negative process start time rejected
@pytest.mark.unit
@pytest.mark.parametrize("bad", [-1, -987654])
def test_record_rejects_negative_start_time(tmp_output_dir, bad):
    with pytest.raises(ValidationError):
        _running(tmp_output_dir, pid=4242, pid_start_time=bad)


# REQ #3: status keeps the five values (no "lost"; user decision 2026-10-28)
@pytest.mark.unit
@pytest.mark.parametrize("bad", ["lost", "orphaned", "unknown", "RUNNING"])
def test_record_rejects_unknown_status(tmp_output_dir, bad):
    with pytest.raises(ValidationError):
        _running(tmp_output_dir, status=bad)


# REQ #4: job ids cannot escape the jobs directory
@pytest.mark.unit
@pytest.mark.parametrize("bad_id", UNSAFE_JOB_IDS)
@pytest.mark.parametrize("helper", ["job_path", "log_path_for", "rc_path_for"])
def test_path_helpers_reject_unsafe_job_id(tmp_path, helper, bad_id):
    fresh = tmp_path / "fresh_out"
    func = getattr(js, helper)
    with pytest.raises(ValueError):
        func(str(fresh), bad_id)
    assert not fresh.exists(), "no file access may happen before the job id is validated"


@pytest.mark.unit
@pytest.mark.parametrize("bad_id", UNSAFE_JOB_IDS)
def test_read_job_rejects_unsafe_job_id(tmp_path, bad_id):
    fresh = tmp_path / "fresh_out"
    with pytest.raises(ValueError):
        read_job(str(fresh), bad_id)
    assert not fresh.exists()


@pytest.mark.unit
@pytest.mark.parametrize("bad_id", UNSAFE_JOB_IDS)
def test_record_rejects_unsafe_job_id(tmp_output_dir, bad_id):
    with pytest.raises(ValueError):
        _running(tmp_output_dir, job_id=bad_id)


@pytest.mark.unit
def test_uuid_job_id_accepted(tmp_output_dir):
    state = _running(tmp_output_dir)
    write_job(state)
    assert read_job(str(tmp_output_dir), state.job_id).job_id == state.job_id
    assert js.job_path(str(tmp_output_dir), state.job_id).parent == js.jobs_dir(str(tmp_output_dir))


# REQ #5: exit code written atomically to <job>.rc
@pytest.mark.unit
@pytest.mark.parametrize("code", [0, 1, 2, -9, 137])
def test_write_rc_writes_decimal_exit_code(tmp_output_dir, code):
    state = _running(tmp_output_dir)
    js.write_rc(str(tmp_output_dir), state.job_id, code)
    rc_file = js.jobs_dir(str(tmp_output_dir)) / f"{state.job_id}.rc"
    assert rc_file.read_text().strip() == str(code)


@pytest.mark.unit
def test_write_rc_renames_temporary_file_over_target(tmp_output_dir, monkeypatch):
    state = _running(tmp_output_dir)
    rc_file = js.jobs_dir(str(tmp_output_dir)) / f"{state.job_id}.rc"
    calls: list[tuple[Path, Path]] = []
    real_replace = os.replace

    def spy(src, dst, *args, **kwargs):
        src_p, dst_p = Path(src), Path(dst)
        if dst_p == rc_file:
            # the temporary file is complete before the rename
            assert src_p.read_text().strip() == "3"
        calls.append((src_p, dst_p))
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", spy)
    js.write_rc(str(tmp_output_dir), state.job_id, 3)
    renames = [c for c in calls if c[1] == rc_file]
    assert len(renames) == 1
    src, _ = renames[0]
    assert src != rc_file
    assert src.parent == rc_file.parent
    assert sorted(p.name for p in rc_file.parent.iterdir()) == [rc_file.name]


# REQ #6: read_rc returns a valid exit code
@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "expected"),
    [("0", 0), ("0\n", 0), (" 3 \n", 3), ("-9\n", -9), ("255", 255), ("-255", -255), ("137\n", 137)],
)
def test_read_rc_returns_valid_exit_code(tmp_output_dir, content, expected):
    state = _running(tmp_output_dir)
    _write_rc_file(tmp_output_dir, state.job_id, content)
    assert js.read_rc(str(tmp_output_dir), state.job_id) == expected


# REQ #7: read_rc returns None without an .rc file
@pytest.mark.unit
def test_read_rc_returns_none_when_absent(tmp_output_dir):
    state = _running(tmp_output_dir)
    assert js.read_rc(str(tmp_output_dir), state.job_id) is None


# REQ #8: read_rc rejects corrupt content
@pytest.mark.unit
@pytest.mark.parametrize(
    "content",
    ["", "\n", "abc", "1.5", "256", "-256", "1 2", "0x1", "1_0", "+-1", "٣", "9" * 10000, b"\xff\xfe", b"\x00"],
)
def test_read_rc_rejects_corrupt_content(tmp_output_dir, content):
    state = _running(tmp_output_dir)
    _write_rc_file(tmp_output_dir, state.job_id, content)
    with pytest.raises(ValueError):
        js.read_rc(str(tmp_output_dir), state.job_id)


# REQ #9 / #10: process start time
@pytest.mark.unit
@linux_only
def test_process_start_time_of_live_process():
    pid = os.getpid()
    assert js.process_start_time(pid) == _start_ticks(pid)


@pytest.mark.unit
@linux_only
def test_process_start_time_of_dead_pid():
    assert js.process_start_time(_dead_pid()) is None


# REQ #11: rc 0 -> succeeded
@pytest.mark.unit
@linux_only
def test_reconcile_rc_zero_succeeds(tmp_output_dir):
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1)
    _write_rc_file(tmp_output_dir, state.job_id, "0\n")
    result = js.reconcile(state)
    assert result.status == "succeeded"
    assert result.returncode == 0
    assert result.error is None
    _assert_utc_iso(result.ended_at)
    assert state.status == "running", "reconcile returns a copy"


@pytest.mark.unit
@linux_only
def test_reconcile_rc_wins_over_live_pid(tmp_output_dir):
    pid = os.getpid()
    state = _running(tmp_output_dir, pid=pid, pid_start_time=_start_ticks(pid))
    _write_rc_file(tmp_output_dir, state.job_id, "0")
    result = js.reconcile(state)
    assert result.status == "succeeded"
    assert result.returncode == 0


# REQ #12: nonzero rc -> failed
@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("code", [1, 2, -9, 137])
def test_reconcile_rc_nonzero_fails(tmp_output_dir, code):
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1)
    _write_rc_file(tmp_output_dir, state.job_id, f"{code}\n")
    result = js.reconcile(state)
    assert result.status == "failed"
    assert result.returncode == code
    assert result.ended_at is not None
    assert result.error == f"Process exited with code {code}. See log for details."


# REQ #13: dead pid without rc -> failed (not a new status)
@pytest.mark.unit
@linux_only
def test_reconcile_dead_pid_without_rc_fails(tmp_output_dir):
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=12345, pgid=4242)
    result = js.reconcile(state)
    assert result.status == "failed"
    assert result.returncode is None
    assert result.ended_at is not None
    assert GONE in (result.error or "")


@pytest.mark.unit
@linux_only
def test_reconcile_legacy_dead_pid_fails(tmp_output_dir):
    """A record written before the fix: pid only, no start time, no pgid (the stale rosette jobs)."""
    state = _running(tmp_output_dir, pid=_dead_pid())
    result = js.reconcile(state)
    assert result.status == "failed"
    assert result.returncode is None
    assert result.ended_at is not None
    assert GONE in (result.error or "")


# REQ #14: reused pid without rc -> failed
@pytest.mark.unit
@linux_only
def test_reconcile_reused_pid_fails(tmp_output_dir):
    pid = os.getpid()
    state = _running(tmp_output_dir, pid=pid, pid_start_time=_start_ticks(pid) + 1)
    result = js.reconcile(state)
    assert result.status == "failed"
    assert GONE in (result.error or "")


# REQ #15: live matching process -> unchanged
@pytest.mark.unit
@linux_only
def test_reconcile_live_matching_process_unchanged(tmp_output_dir):
    pid = os.getpid()
    state = _running(tmp_output_dir, pid=pid, pid_start_time=_start_ticks(pid), pgid=os.getpgid(pid))
    assert js.reconcile(state) == state


# REQ #16: legacy record with live pid -> unchanged
@pytest.mark.unit
@linux_only
def test_reconcile_legacy_live_pid_unchanged(tmp_output_dir):
    state = _running(tmp_output_dir, pid=os.getpid())
    assert js.reconcile(state) == state


# REQ #17: running job without pid -> unchanged
@pytest.mark.unit
def test_reconcile_running_without_pid_unchanged(tmp_output_dir):
    state = _running(tmp_output_dir)
    assert js.reconcile(state) == state


# REQ #18: non-running jobs -> unchanged, even with a dead pid and an rc file
@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("status", ["pending", "succeeded", "failed", "cancelled"])
def test_reconcile_non_running_unchanged(tmp_output_dir, status):
    state = _running(tmp_output_dir, status=status, pid=_dead_pid())
    _write_rc_file(tmp_output_dir, state.job_id, "1\n")
    assert js.reconcile(state) == state


# REQ #19: corrupt rc -> failed
@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("content", ["garbage", "", "1.5", "9" * 10000, b"\xff"])
def test_reconcile_corrupt_rc_fails(tmp_output_dir, live, content):
    pid = os.getpid() if live else _dead_pid()
    start = _start_ticks(pid) if live else 1
    state = _running(tmp_output_dir, pid=pid, pid_start_time=start)
    _write_rc_file(tmp_output_dir, state.job_id, content)
    result = js.reconcile(state)
    assert result.status == "failed"
    assert result.returncode is None
    assert "invalid exit-code file" in (result.error or "")


# REQ #20: reconcile writes nothing
@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("rc_content", [None, "0\n", "2\n", "garbage"])
def test_reconcile_writes_nothing(tmp_output_dir, rc_content):
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1)
    write_job(state)
    if rc_content is not None:
        _write_rc_file(tmp_output_dir, state.job_id, rc_content)
    before = _snapshot(tmp_output_dir)
    js.reconcile(state)
    assert _snapshot(tmp_output_dir) == before
