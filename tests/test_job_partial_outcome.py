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
    for _ in range(100):
        await asyncio.sleep(0.1)
        stored = read_job(str(tmp_output_dir), state.job_id)
        if stored.status != "running":
            break
    assert stored.status == "succeeded"
    assert stored.returncode == 3
    assert stored.model_dump().get("outcome") == "partial"
    assert stored.model_dump().get("skipped_subjects") == ["sub-01", "sub-02"]


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
