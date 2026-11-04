from __future__ import annotations

import json
import logging
import os
import socket
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


# REQ #12: nonzero rc -> failed, except exit 3 (partial, user decision 2026-11-04); -3 is death by signal 3
@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("code", [1, 2, 4, -3, -9, 137, 255])
def test_reconcile_rc_nonzero_fails(tmp_output_dir, code):
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1)
    _write_rc_file(tmp_output_dir, state.job_id, f"{code}\n")
    result = js.reconcile(state)
    assert result.status == "failed"
    assert result.returncode == code
    assert result.ended_at is not None
    assert result.error == f"Process exited with code {code}. See log for details."


# --- TASK-MCPJOBS-BDRABCZUK-3B8937480E8F: a stage exit 3 is a partial outcome, not a failure ---

ALL_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]
LOST_SIDE_LINE = (
    "ERROR:generate_sulcal_regions.py: Run summary: side R lost: NoValidSubjectError raised, no skip entry\n"
)


def _summary_line(side: str, step: str, names: list[str], causes: str = "ValueError: no graph, no mask") -> str:
    """One run-summary record as cortical_tiles generate_sulcal_regions logs it to stderr (the job log)."""
    subjects = ", ".join(names)
    return (
        f"ERROR:generate_sulcal_regions.py: Run summary: side {side}, step {step}: "
        f"{len(names)} skipped subject(s): {subjects}; causes: {causes}\n"
    )


def _exit_3_job(out, stage: str = "cortical_tiles", log: str | bytes | None = None) -> JobState:
    state = _running(out, pid=_dead_pid(), pid_start_time=1).model_copy(update={"stage": stage})
    _write_rc_file(out, state.job_id, "3\n")
    if isinstance(log, str):
        Path(state.log_path).write_text(log)
    elif log is not None:
        Path(state.log_path).write_bytes(log)
    return state


@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("stage", ALL_STAGES)
def test_reconcile_rc_3_is_partial(tmp_output_dir, stage):
    """REQ-MCPJOBS-BDRABCZUK-1531D2654281: exit 3 of any stage is succeeded with outcome partial, no error."""
    result = js.reconcile(_exit_3_job(tmp_output_dir, stage=stage))
    assert result.status == "succeeded"
    assert result.model_dump().get("outcome") == "partial"
    assert result.returncode == 3
    assert result.error is None
    _assert_utc_iso(result.ended_at)


@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("code", [0, 1, 4, -3, 137])
def test_reconcile_rc_other_than_3_has_no_outcome(tmp_output_dir, code):
    """REQ-MCPJOBS-BDRABCZUK-E2CE5413E75D: an exit code other than 3 gives outcome None."""
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1)
    _write_rc_file(tmp_output_dir, state.job_id, f"{code}\n")
    assert js.reconcile(state).model_dump().get("outcome", "absent") is None


@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("before", [b"", b"\xff\xfe binary stage output\n"], ids=["text", "undecodable_bytes"])
def test_reconcile_rc_3_lists_skipped_subjects_of_cortical_tiles(tmp_output_dir, before):
    """REQ-MCPJOBS-BDRABCZUK-D39EEC205C86: the sorted distinct names of the run-summary lines, over sides and steps."""
    log = "".join(
        [
            "INFO:generate_sulcal_regions.py: 2 skipped subject(s) so far\n",
            _summary_line("L", "generate_skeletons", ["sub-02", "sub-01_ses-1"]),
            _summary_line("R", "generate_crops", ["sub-03", "sub-02"], causes="KeyError: sub-99, sub-98"),
            LOST_SIDE_LINE,
        ]
    )
    result = js.reconcile(_exit_3_job(tmp_output_dir, log=before + log.encode()))
    assert result.model_dump().get("skipped_subjects") == ["sub-01_ses-1", "sub-02", "sub-03"]


@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("log", ["", "Running command:\n", LOST_SIDE_LINE], ids=["empty", "unrelated", "lost_side"])
def test_reconcile_rc_3_without_summary_line_lists_no_subject(tmp_output_dir, log):
    """REQ-MCPJOBS-BDRABCZUK-D43F636F6079: no line naming skipped subjects: partial, with an empty list."""
    dumped = js.reconcile(_exit_3_job(tmp_output_dir, log=log)).model_dump()
    assert dumped.get("outcome") == "partial"
    assert dumped.get("skipped_subjects") == []


@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_reconcile_rc_3_with_unreadable_log_lists_no_subject(tmp_output_dir, kind):
    """REQ-MCPJOBS-BDRABCZUK-3E77CF926916: an unreadable job log leaves the job partial, with an empty list."""
    state = _exit_3_job(tmp_output_dir)
    if kind == "directory":
        Path(state.log_path).mkdir()
    dumped = js.reconcile(state).model_dump()
    assert dumped.get("outcome") == "partial"
    assert dumped.get("skipped_subjects") == []


@pytest.mark.unit
@linux_only
@pytest.mark.parametrize("stage", [stage for stage in ALL_STAGES if stage != "cortical_tiles"])
def test_reconcile_rc_3_of_other_stage_lists_no_subject(tmp_output_dir, stage):
    """REQ-MCPJOBS-BDRABCZUK-E3B874F45AC3: only a cortical_tiles job reads skipped subjects from its log."""
    log = _summary_line("L", "generate_crops", ["sub-01"])
    dumped = js.reconcile(_exit_3_job(tmp_output_dir, stage=stage, log=log)).model_dump()
    assert dumped.get("outcome") == "partial"
    assert dumped.get("skipped_subjects") == []


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["lost", "failed", "PARTIAL", ""])
def test_record_rejects_unknown_outcome(tmp_output_dir, bad):
    """REQ-MCPJOBS-BDRABCZUK-EE27D7F781B2: outcome is None or partial."""
    with pytest.raises(ValidationError):
        _running(tmp_output_dir, outcome=bad)


@pytest.mark.unit
def test_read_job_without_outcome_keys(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-8275FF99F8AE: a record written before the two keys existed stays readable."""
    state = _running(tmp_output_dir, status="succeeded")
    record = json.loads(state.model_dump_json())
    record.pop("outcome", None)
    record.pop("skipped_subjects", None)
    js.job_path(str(tmp_output_dir), state.job_id).write_text(json.dumps(record))
    dumped = read_job(str(tmp_output_dir), state.job_id).model_dump()
    assert dumped.get("outcome", "absent") is None
    assert dumped.get("skipped_subjects") == []


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


# --- TASK-MCPJOBS-BDRABCZUK-0DDDE42A965C: host and boot id on the record; reconcile guard ---
# Records carry the host name and the boot id of the launch; reconcile probes the local /proc only
# when both match the current ones. The current values are the real ones of this test process.

FOREIGN_HOST = "other-host.invalid"
OTHER_BOOT = "00000000-0000-4000-8000-000000000000"


def _current_boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def _job_store_warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == js.logger.name and r.levelno == logging.WARNING]


# REQ-MCPJOBS-BDRABCZUK-F74963C2850E: host and boot_id survive write_job + read_job
@pytest.mark.unit
def test_record_roundtrips_host_and_boot_id(tmp_output_dir):
    state = _running(tmp_output_dir, pid=4242, host="node-a.example", boot_id=OTHER_BOOT)
    write_job(state)
    loaded = read_job(str(tmp_output_dir), state.job_id).model_dump()
    assert loaded.get("host") == "node-a.example"
    assert loaded.get("boot_id") == OTHER_BOOT


# REQ-MCPJOBS-BDRABCZUK-61037DA49688: a legacy record without the fields reads them as None
@pytest.mark.unit
def test_legacy_record_reads_host_and_boot_id_none(tmp_output_dir):
    state = _running(tmp_output_dir, pid=4242)
    raw = state.model_dump(mode="json")
    raw.pop("host", None)
    raw.pop("boot_id", None)
    js.job_path(str(tmp_output_dir), state.job_id).write_text(json.dumps(raw))
    loaded = read_job(str(tmp_output_dir), state.job_id).model_dump()
    assert "host" in loaded and loaded["host"] is None
    assert "boot_id" in loaded and loaded["boot_id"] is None


# REQ-MCPJOBS-BDRABCZUK-0B2428B9C153: foreign host, no .rc -> unchanged
@pytest.mark.unit
@linux_only
def test_reconcile_foreign_host_unchanged(tmp_output_dir):
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1, host=FOREIGN_HOST, boot_id=_current_boot_id())
    assert js.reconcile(state) == state


# REQ-MCPJOBS-BDRABCZUK-05979502429A: other boot, no .rc -> unchanged
@pytest.mark.unit
@linux_only
def test_reconcile_other_boot_unchanged(tmp_output_dir):
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1, host=socket.gethostname(), boot_id=OTHER_BOOT)
    assert js.reconcile(state) == state


# REQ-MCPJOBS-BDRABCZUK-A61BAA7660BE: foreign host -> process_start_time never called
@pytest.mark.unit
@linux_only
def test_reconcile_foreign_host_skips_proc_probe(tmp_output_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(js, "process_start_time", lambda pid: calls.append(pid))
    pid = os.getpid()
    state = _running(tmp_output_dir, pid=pid, pid_start_time=1, host=FOREIGN_HOST, boot_id=_current_boot_id())
    js.reconcile(state)
    assert calls == []


# REQ-MCPJOBS-BDRABCZUK-89DD6A88FD4C: other boot -> process_start_time never called
@pytest.mark.unit
@linux_only
def test_reconcile_other_boot_skips_proc_probe(tmp_output_dir, monkeypatch):
    calls = []
    monkeypatch.setattr(js, "process_start_time", lambda pid: calls.append(pid))
    pid = os.getpid()
    state = _running(tmp_output_dir, pid=pid, pid_start_time=1, host=socket.gethostname(), boot_id=OTHER_BOOT)
    js.reconcile(state)
    assert calls == []


# REQ-MCPJOBS-BDRABCZUK-DD534F69B54E: foreign host -> one warning naming job, recorded and current host
@pytest.mark.unit
@linux_only
def test_reconcile_foreign_host_logs_one_warning(tmp_output_dir, caplog):
    caplog.set_level(logging.WARNING, logger=js.logger.name)
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1, host=FOREIGN_HOST, boot_id=_current_boot_id())
    js.reconcile(state)
    warnings = _job_store_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert state.job_id in warnings[0]
    assert FOREIGN_HOST in warnings[0]
    assert socket.gethostname() in warnings[0]


# REQ-MCPJOBS-BDRABCZUK-B49E30CF1C4A: same host, other boot -> one warning naming job, recorded and current boot id
@pytest.mark.unit
@linux_only
def test_reconcile_other_boot_logs_one_warning(tmp_output_dir, caplog):
    caplog.set_level(logging.WARNING, logger=js.logger.name)
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1, host=socket.gethostname(), boot_id=OTHER_BOOT)
    js.reconcile(state)
    warnings = _job_store_warnings(caplog)
    assert len(warnings) == 1, warnings
    assert state.job_id in warnings[0]
    assert OTHER_BOOT in warnings[0]
    assert _current_boot_id() in warnings[0]


# REQ-MCPJOBS-BDRABCZUK-6C1F80F835B0: same host and boot, dead pid, no .rc -> failed copy of the job
@pytest.mark.unit
@linux_only
def test_reconcile_same_host_and_boot_dead_pid_fails(tmp_output_dir):
    host, boot = socket.gethostname(), _current_boot_id()
    state = _running(tmp_output_dir, pid=_dead_pid(), pid_start_time=1, host=host, boot_id=boot)
    result = js.reconcile(state)
    assert result.status == "failed"
    assert GONE in (result.error or "")
    copied = result.model_dump()
    assert copied.get("host") == host
    assert copied.get("boot_id") == boot
