"""A stage exit 3 reaches the job record by both writers (TASK-MCPJOBS-BDRABCZUK-3B8937480E8F).

reconcile itself writes nothing: runner's watcher and the job tools copy its result into the record,
field by field. These tests read the record each of them wrote.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from champollion_sulcal_mcp import job_store, runner
from champollion_sulcal_mcp.job_store import JobState, log_path_for, read_job, write_job, write_rc
from champollion_sulcal_mcp.tools.jobs import get_job_status

SUMMARY = (
    "ERROR:generate_sulcal_regions.py: Run summary: side L, step generate_crops: "
    "2 skipped subject(s): sub-02, sub-01; causes: ValueError: no graph"
)
STAGE = f"import sys; print({SUMMARY!r}, file=sys.stderr); sys.exit(3)"


async def _stored_once(out: str, job_id: str, ended) -> JobState:
    """The job record once `ended(record)` is true, or as it stands after 10 s."""
    for _ in range(100):
        await asyncio.sleep(0.1)
        stored = read_job(out, job_id)
        if ended(stored):
            break
    return stored


async def _stage_exits_3_after_cancel(out: str) -> JobState:
    """The record of a cortical_tiles job marked cancelled while its stage runs, once that stage has exited 3."""
    go = Path(out) / "go"
    wait_for_go = f"import pathlib, time\nwhile not pathlib.Path({str(go)!r}).exists(): time.sleep(0.02)\n"
    state = await runner.launch(
        stage="cortical_tiles",
        argv=[sys.executable, "-c", wait_for_go + STAGE],
        output_dir=out,
        cwd=out,
        env=dict(os.environ),
        args_snapshot={},
    )
    await job_store.update_job(out, state.job_id, status="cancelled")
    go.touch()
    return await _stored_once(out, state.job_id, lambda stored: stored.ended_at is not None)


@pytest.mark.integration
async def test_watcher_records_partial_outcome_with_skipped_subjects(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-B89A874C185A: the record the watcher writes for a stage exit 3."""
    state = await runner.launch(
        stage="cortical_tiles",
        argv=[sys.executable, "-c", STAGE],
        output_dir=str(tmp_output_dir),
        cwd=str(tmp_output_dir),
        env=dict(os.environ),
        args_snapshot={},
    )
    stored = await _stored_once(str(tmp_output_dir), state.job_id, lambda record: record.status != "running")
    assert stored.status == "succeeded"
    assert stored.returncode == 3
    assert stored.model_dump().get("outcome") == "partial"
    assert stored.model_dump().get("skipped_subjects") == ["sub-01", "sub-02"]


@pytest.mark.integration
async def test_cancelled_job_exit_3_keeps_exit_code_and_error(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-8DC90E4DEA09: a stage exit 3 on a cancelled job keeps today's record."""
    stored = await _stage_exits_3_after_cancel(str(tmp_output_dir))
    assert stored.status == "cancelled"
    assert stored.returncode == 3
    assert stored.error == "Process exited with code 3. See log for details."


@pytest.mark.integration
async def test_cancelled_job_exit_3_has_no_partial_outcome(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-ED01D3ED0665: a stage exit 3 on a cancelled job is not partial."""
    stored = await _stage_exits_3_after_cancel(str(tmp_output_dir))
    assert stored.returncode == 3, "the stage did not end before the deadline"
    dumped = stored.model_dump()
    assert dumped.get("outcome", "absent") is None
    assert dumped.get("skipped_subjects") == []


@pytest.mark.unit
async def test_get_job_status_stores_partial_outcome_with_skipped_subjects(tmp_output_dir):
    """REQ-MCPJOBS-BDRABCZUK-2BD3EE749EC5: the record get_job_status writes for a job whose .rc holds 3."""
    out = str(tmp_output_dir)
    state = JobState(stage="cortical_tiles", status="running", output_dir=out, log_path="unset")
    state = state.model_copy(update={"log_path": str(log_path_for(out, state.job_id))})
    write_job(state)
    Path(state.log_path).write_text(f"{SUMMARY}\n")
    write_rc(out, state.job_id, 3)

    result = await get_job_status(out, state.job_id)

    assert result.get("outcome") == "partial"
    assert result.get("skipped_subjects") == ["sub-01", "sub-02"]
    stored = job_store.read_job(out, state.job_id).model_dump()
    assert stored.get("outcome") == "partial"
    assert stored.get("skipped_subjects") == ["sub-01", "sub-02"]
