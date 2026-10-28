"""Detached pipeline driver (TASK-MCPJOBS-BDRABCZUK-0376C2AEC946).

Root cause: start_pipeline ran the six-stage chain as an asyncio task inside the MCP server, so the
chain stopped, and the umbrella job stayed "running", as soon as the server process exited. These tests
call start_pipeline from a separate Python process that exits right after start_pipeline returns, then
watch the umbrella job and the stage jobs from the test process. Every stage script is a fake (a short
Python file in a synthetic champollion_pipeline tree) that records its start and end in a trace file and
then succeeds, fails, waits for a "go" file or kills its job wrapper, as the test's control file says.
No real pipeline stage runs.

Each test's docstring names the Requirement it verifies.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

import champollion_sulcal_mcp
from champollion_sulcal_mcp import job_store, pipeline_driver
from champollion_sulcal_mcp.job_store import (
    JobState,
    find_job_by_wrapper,
    process_start_time,
    read_job,
    write_job,
)
from champollion_sulcal_mcp.tools import jobs, pipeline

SRC_DIR = str(Path(champollion_sulcal_mcp.__file__).resolve().parent.parent)
TERMINAL = ("succeeded", "failed", "cancelled")
FIRST_STAGE_S = 15.0
CHAIN_S = 90.0
# No stage started or ended for this long: the chain has stopped.
STALL_S = 20.0
CANCEL_S = 15.0

# Fake script of each stage, in pipeline order.
STAGE_SCRIPTS = {
    "morphologist": "generate_morphologist_graphs",
    "cortical_tiles": "run_cortical_tiles",
    "config": "generate_champollion_config",
    "embeddings": "generate_embeddings",
    "combine": "put_together_embeddings",
    "snapshots": "generate_snapshots",
}

# Body of every fake stage script. It reads its mode from <pipeline>/control/<script>.mode:
# "ok" (default) exits 0, "fail" exits 3, "wait" waits (at most 60 s) for <pipeline>/control/go,
# "kill_wrapper" SIGKILLs its parent (the job wrapper) so no exit code is ever written,
# "delete_record <stage>" / "corrupt_record <stage>" delete / corrupt the running job record of
# <stage> under <root>/ds, then wait like "wait".
FAKE_STAGE = """\
import json, os, signal, sys, time
from pathlib import Path

here = Path(__file__).resolve()
ctl = here.parents[2] / "control"
name = here.stem


def trace(event):
    with open(ctl / "trace.log", "a") as handle:
        handle.write(f"{event} {name} {os.getpid()}\\n")


trace("start")
mode_file = ctl / f"{name}.mode"
mode = mode_file.read_text().strip() if mode_file.exists() else "ok"
kind, _, target = mode.partition(" ")
if kind in ("delete_record", "corrupt_record"):
    for record in (here.parents[3] / "ds").rglob(".mcp_jobs/*.json"):
        try:
            data = json.loads(record.read_text())
        except ValueError:
            continue
        if data.get("stage") == target and data.get("status") == "running":
            if kind == "delete_record":
                record.unlink()
            else:
                record.write_text("{corrupt")
if kind in ("wait", "delete_record", "corrupt_record"):
    deadline = time.monotonic() + 60
    while not (ctl / "go").exists() and time.monotonic() < deadline:
        time.sleep(0.05)
elif mode == "fail":
    trace("end")
    sys.exit(3)
elif mode == "kill_wrapper":
    os.kill(os.getppid(), signal.SIGKILL)
trace("end")
sys.exit(0)
"""

# Runs in a separate Python process that exits as soon as start_pipeline returns.
LAUNCHER = (
    "import asyncio, json, sys\n"
    "from champollion_sulcal_mcp.tools import pipeline\n"
    "async def main():\n"
    "    result = await pipeline.start_pipeline(**json.loads(sys.argv[1]))\n"
    "    print(json.dumps(result))\n"
    "asyncio.run(main())\n"
)


def _pipeline_kwargs(out: Path) -> dict:
    return {
        "input_dir": str(out / "subjects"),
        "output_dir": str(out),
        "path_to_graph": "t1mri/default_acquisition/default_analysis/folds/3.1",
        "path_sk_with_hull": "t1mri/default_acquisition/default_analysis/segmentation",
        "crop_path": str(out / "crops"),
        "dataset": "DEMO01",
        "models_path": str(out / "models"),
        "datasets_root": str(out),
        "log": False,
    }


class Chain:
    """A synthetic dataset and champollion_pipeline tree whose stage scripts are fakes."""

    def __init__(self, root: Path):
        self.pipeline = root / "champollion_pipeline"
        self.scripts = self.pipeline / "src" / "champollion_pipeline"
        self.control = self.pipeline / "control"
        self.out = root / "ds"
        self.scripts.mkdir(parents=True)
        self.control.mkdir()
        self.out.mkdir()
        for script in STAGE_SCRIPTS.values():
            (self.scripts / f"{script}.py").write_text(FAKE_STAGE)

    def mode(self, stage: str, mode: str) -> None:
        (self.control / f"{STAGE_SCRIPTS[stage]}.mode").write_text(mode)

    def remove_script(self, stage: str) -> None:
        (self.scripts / f"{STAGE_SCRIPTS[stage]}.py").unlink()

    def go(self) -> None:
        (self.control / "go").write_text("go")

    def trace(self) -> list[tuple[str, str]]:
        path = self.control / "trace.log"
        if not path.exists():
            return []
        return [tuple(line.split()[:2]) for line in path.read_text().splitlines() if line.strip()]

    def started(self, stage: str) -> bool:
        return ("start", STAGE_SCRIPTS[stage]) in self.trace()

    def records(self) -> list[JobState]:
        found = []
        for path in self.out.rglob(".mcp_jobs/*.json"):
            with contextlib.suppress(ValueError, OSError):
                found.append(JobState.model_validate_json(path.read_text()))
        return found

    def launch_and_exit(self) -> str:
        """Call start_pipeline in a separate Python process; return the umbrella job id once it has exited."""
        done = subprocess.run(
            [sys.executable, "-c", LAUNCHER, json.dumps(_pipeline_kwargs(self.out))],
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
            env={**os.environ, "PYTHONPATH": SRC_DIR, "CHAMPOLLION_PIPELINE_DIR": str(self.pipeline)},
            check=False,
        )
        assert done.returncode == 0, done.stderr
        return json.loads(done.stdout.strip().splitlines()[-1])["job_id"]

    def umbrella(self, job_id: str) -> JobState:
        return read_job(str(self.out), job_id)

    def stage_job(self, job_id: str) -> JobState:
        matches = [record for record in self.records() if record.job_id == job_id]
        assert matches, f"no stage job record {job_id} under {self.out}"
        return matches[0]


def _wait(predicate, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def _group_alive(pgid: int) -> bool:
    """True while a live, non-zombie process belongs to process group ``pgid``."""
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            data = Path(f"/proc/{name}/stat").read_bytes()
        except OSError:
            continue
        rest = data[data.rindex(b")") + 1 :].split()
        if rest[0] not in (b"Z", b"X", b"x") and int(rest[2]) == pgid:
            return True
    return False


def _wait_first_stage(chain: Chain) -> None:
    assert _wait(lambda: chain.started("morphologist"), FIRST_STAGE_S), (
        f"no stage started within {FIRST_STAGE_S:.0f} s after the process that called start_pipeline exited"
    )


def _wait_umbrella_terminal(chain: Chain, job_id: str) -> JobState:
    """The umbrella job once terminal, or as it stands after CHAIN_S, or after STALL_S without a stage event."""
    _wait_first_stage(chain)
    deadline = time.monotonic() + CHAIN_S
    events, changed = len(chain.trace()), time.monotonic()
    while time.monotonic() < deadline:
        umbrella = chain.umbrella(job_id)
        if umbrella.status in TERMINAL:
            return umbrella
        now = len(chain.trace())
        if now != events:
            events, changed = now, time.monotonic()
        elif time.monotonic() - changed > STALL_S:
            break
        time.sleep(0.1)
    return chain.umbrella(job_id)


def _running_child(chain: Chain, job_id: str) -> str:
    assert _wait(lambda: chain.umbrella(job_id).progress.current_child_job_id is not None, FIRST_STAGE_S), (
        "the umbrella job never named its running stage job"
    )
    return chain.umbrella(job_id).progress.current_child_job_id


@pytest.fixture
def chain(tmp_path):
    """A fake chain; on teardown releases waiting stages and SIGKILLs the job groups still running."""
    fake = Chain(tmp_path)
    yield fake
    fake.go()
    for record in fake.records():
        if record.status != "running" or None in (record.pid, record.pgid, record.pid_start_time):
            continue
        with contextlib.suppress(OSError, ValueError, RuntimeError):
            if process_start_time(record.pid) == record.pid_start_time:
                os.killpg(record.pgid, signal.SIGKILL)


# --- REQ-MCPJOBS-BDRABCZUK-F7C3597735B8: driver launched through runner.launch ---


@pytest.mark.unit
async def test_start_pipeline_launches_driver_module_through_runner_launch(pipeline_dir, launches, tmp_path):
    """REQ-MCPJOBS-BDRABCZUK-F7C3597735B8: start_pipeline starts the pipeline driver through
    runner.launch with stage name "pipeline" and a command running a champollion_sulcal_mcp
    module through ``python -m``; the returned job_id is the launched job's id.
    """
    out = tmp_path / "ds"
    try:
        result = await pipeline.start_pipeline(**_pipeline_kwargs(out))
        for _ in range(20):
            await asyncio.sleep(0)
        driver = [call for call in launches if call["stage"] == "pipeline"]
        assert len(driver) == 1, f"expected one runner.launch with stage 'pipeline', got {launches}"
        argv = driver[0]["argv"]
        assert "-m" in argv, f"driver command does not use python -m: {argv}"
        module = argv[argv.index("-m") + 1]
        assert module.startswith("champollion_sulcal_mcp."), f"driver module {module!r} is not in the package"
        assert (out / ".mcp_jobs" / f"{result['job_id']}.json").is_file()
        assert read_job(str(out), result["job_id"]).stage == "pipeline"
    finally:
        for task in asyncio.all_tasks() - {asyncio.current_task()}:
            task.cancel()


# --- REQ-MCPJOBS-BDRABCZUK-D6ABF4893302: the chain completes after the server exits ---


@pytest.mark.integration
def test_umbrella_succeeds_after_launching_process_exits(chain):
    """REQ-MCPJOBS-BDRABCZUK-D6ABF4893302: when the process that called start_pipeline exits right
    after start_pipeline returns, the umbrella job reaches status succeeded once the last active
    stage job succeeds.
    """
    job_id = chain.launch_and_exit()
    umbrella = _wait_umbrella_terminal(chain, job_id)
    assert umbrella.status == "succeeded", f"umbrella {umbrella.status}: {umbrella.error}"
    assert chain.started("snapshots")


# --- REQ-MCPCOV-BDRABCZUK-1007D756AD03: order and sequencing, with the server gone ---


@pytest.mark.integration
def test_driver_runs_stages_one_after_another_in_order(chain):
    """REQ-MCPCOV-BDRABCZUK-1007D756AD03: the stages run in pipeline order, each starting only after
    the previous stage's job ended, also after the process that called start_pipeline exited.
    """
    job_id = chain.launch_and_exit()
    _wait_umbrella_terminal(chain, job_id)
    expected = []
    for script in STAGE_SCRIPTS.values():
        expected += [("start", script), ("end", script)]
    assert chain.trace() == expected


# --- REQ-MCPCOV-BDRABCZUK-F12D3909336C: a failed stage fails the umbrella, no later stage ---


@pytest.mark.integration
def test_failed_stage_fails_umbrella_naming_stage(chain):
    """REQ-MCPCOV-BDRABCZUK-F12D3909336C: a stage job that ends failed sets the umbrella job to
    failed with an error naming that stage, with the server gone.
    """
    chain.mode("cortical_tiles", "fail")
    job_id = chain.launch_and_exit()
    umbrella = _wait_umbrella_terminal(chain, job_id)
    assert umbrella.status == "failed"
    assert "cortical_tiles" in (umbrella.error or "")


@pytest.mark.integration
def test_failed_stage_starts_no_later_stage(chain):
    """REQ-MCPCOV-BDRABCZUK-F12D3909336C: after a stage job ends failed, no later stage starts,
    with the server gone.
    """
    chain.mode("cortical_tiles", "fail")
    job_id = chain.launch_and_exit()
    _wait_umbrella_terminal(chain, job_id)
    time.sleep(3)
    assert [script for event, script in chain.trace() if event == "start"] == [
        STAGE_SCRIPTS["morphologist"],
        STAGE_SCRIPTS["cortical_tiles"],
    ]


# --- REQ-MCPCOV-BDRABCZUK-199A2F34559F / -E6FADE5EE94F / -733D33D0319F, with the server gone ---


@pytest.mark.integration
def test_vanished_stage_record_fails_umbrella(chain):
    """REQ-MCPCOV-BDRABCZUK-199A2F34559F: when a running stage's job file disappears, the umbrella job
    ends failed with an error containing "disappeared unexpectedly", with the server gone.
    """
    chain.mode("morphologist", "delete_record morphologist")
    job_id = chain.launch_and_exit()
    umbrella = _wait_umbrella_terminal(chain, job_id)
    assert umbrella.status == "failed", f"umbrella {umbrella.status}"
    assert "disappeared unexpectedly" in (umbrella.error or "")


@pytest.mark.integration
def test_driver_exception_marks_umbrella_crashed(chain):
    """REQ-MCPCOV-BDRABCZUK-E6FADE5EE94F: when the background pipeline run raises an unexpected
    exception (here: a corrupt stage job record), the umbrella job ends failed with an error beginning
    "Pipeline crashed:", with the server gone.
    """
    chain.mode("morphologist", "corrupt_record morphologist")
    job_id = chain.launch_and_exit()
    umbrella = _wait_umbrella_terminal(chain, job_id)
    assert umbrella.status == "failed", f"umbrella {umbrella.status}"
    assert (umbrella.error or "").startswith("Pipeline crashed:"), umbrella.error


@pytest.mark.integration
def test_stage_launch_error_fails_umbrella(chain):
    """REQ-MCPCOV-BDRABCZUK-733D33D0319F: when a stage tool raises a ToolError at launch (here: its
    script is missing), the umbrella job ends failed with an error containing "failed to launch" and
    the stage name, with the server gone.
    """
    chain.remove_script("cortical_tiles")
    job_id = chain.launch_and_exit()
    umbrella = _wait_umbrella_terminal(chain, job_id)
    assert umbrella.status == "failed", f"umbrella {umbrella.status}"
    assert "failed to launch" in (umbrella.error or "")
    assert "cortical_tiles" in (umbrella.error or "")


# --- REQ-MCPJOBS-BDRABCZUK-FC7ACE1B885D: stage wrapper dead without exit code ---


@pytest.mark.integration
def test_stage_wrapper_killed_fails_umbrella_naming_stage(chain):
    """REQ-MCPJOBS-BDRABCZUK-FC7ACE1B885D: when a stage job's wrapper dies without writing an exit
    code, the pipeline driver sets the umbrella job to failed with an error naming that stage.
    """
    chain.mode("morphologist", "kill_wrapper")
    job_id = chain.launch_and_exit()
    umbrella = _wait_umbrella_terminal(chain, job_id)
    assert umbrella.status == "failed", f"umbrella {umbrella.status}"
    assert "morphologist" in (umbrella.error or "")
    assert not chain.started("cortical_tiles")


# --- REQ-MCPJOBS-BDRABCZUK-4E64CF822D7E: dead driver reported failed ---


@pytest.mark.integration
def test_killed_driver_reports_umbrella_failed(chain):
    """REQ-MCPJOBS-BDRABCZUK-4E64CF822D7E: when the pipeline driver dies without writing an exit
    code, get_job_status reports the umbrella job with status failed.
    """
    chain.mode("morphologist", "wait")
    job_id = chain.launch_and_exit()
    _wait_first_stage(chain)
    umbrella = chain.umbrella(job_id)
    assert umbrella.pgid is not None, "umbrella job records no driver process group"
    os.killpg(umbrella.pgid, signal.SIGKILL)
    assert _wait(lambda: not _group_alive(umbrella.pgid), 10.0)
    status = asyncio.run(jobs.get_job_status(str(chain.out), job_id))
    assert status["status"] == "failed"


# --- cancel_job on the umbrella job, with the server gone ---


def _cancel_running_umbrella(chain: Chain) -> tuple[str, JobState, str, float]:
    """Start a chain whose first stage waits, then cancel_job its umbrella.

    Returns the umbrella job id, the umbrella record before the cancel, the running stage job id and
    the monotonic time at which cancel_job was called.
    """
    chain.mode("morphologist", "wait")
    job_id = chain.launch_and_exit()
    _wait_first_stage(chain)
    child_id = _running_child(chain, job_id)
    before = chain.umbrella(job_id)
    assert before.pgid is not None, "umbrella job records no driver process group"
    called = time.monotonic()
    asyncio.run(jobs.cancel_job(str(chain.out), job_id))
    return job_id, before, child_id, called


def _remaining(called: float) -> float:
    return max(CANCEL_S - (time.monotonic() - called), 0.0)


@pytest.mark.integration
def test_cancel_umbrella_ends_driver_process(chain):
    """REQ-MCPJOBS-BDRABCZUK-C81CF7656357: when cancel_job cancels a running pipeline umbrella job,
    the pipeline driver process exits within 15 seconds.
    """
    _job_id, before, _child, called = _cancel_running_umbrella(chain)
    assert _wait(lambda: not _group_alive(before.pgid), _remaining(called)), (
        "driver process group still alive 15 s after cancel_job"
    )


@pytest.mark.integration
def test_cancel_umbrella_cancels_running_stage_job(chain):
    """REQ-MCPCOV-BDRABCZUK-79BAC3C8CAF6: cancelling the umbrella job while a stage job runs cancels
    that stage job (status cancelled within 15 seconds), with the server gone.
    """
    _job_id, _before, child_id, called = _cancel_running_umbrella(chain)
    _wait(lambda: chain.stage_job(child_id).status in TERMINAL, _remaining(called))
    assert chain.stage_job(child_id).status == "cancelled"


@pytest.mark.integration
def test_cancel_umbrella_starts_no_later_stage(chain):
    """REQ-MCPCOV-BDRABCZUK-79BAC3C8CAF6: after the umbrella job is cancelled, no later stage starts,
    even once the stage that was running could have finished.
    """
    _job_id, before, _child, called = _cancel_running_umbrella(chain)
    _wait(lambda: not _group_alive(before.pgid), _remaining(called))
    chain.go()
    time.sleep(5)
    assert not chain.started("cortical_tiles")


@pytest.mark.integration
def test_cancelled_umbrella_stays_cancelled_after_driver_exits(chain):
    """REQ-MCPJOBS-BDRABCZUK-82A936C1B53B: after cancel_job cancels a running pipeline umbrella job,
    the umbrella job status remains cancelled once the pipeline driver process has exited.
    """
    job_id, before, _child, called = _cancel_running_umbrella(chain)
    assert _wait(lambda: not _group_alive(before.pgid), _remaining(called)), "driver process group still alive"
    time.sleep(1)
    assert chain.umbrella(job_id).status == "cancelled"
    status = asyncio.run(jobs.get_job_status(str(chain.out), job_id))
    assert status["status"] == "cancelled"


# --- in-process coverage of the driver entry points (green, TASK-MCPJOBS-BDRABCZUK-CCA117D50465) ---


def _own_record(out: Path, stage: str = "pipeline", **fields) -> JobState:
    """A running record whose process is this test process, as runner.launch writes it for a wrapper."""
    values = {
        "stage": stage,
        "status": "running",
        "output_dir": str(out),
        "log_path": str(out / "driver.log"),
        "pid": os.getpid(),
        "pgid": os.getpgrp(),
        "pid_start_time": process_start_time(os.getpid()),
        "args": {"active_stages": ["morphologist"], "chain": {"input_dir": str(out / "subjects")}},
    }
    state = JobState(**{**values, **fields})
    write_job(state)
    return state


@pytest.mark.unit
def test_find_job_by_wrapper_matches_only_one_running_record(tmp_path):
    """find_job_by_wrapper returns the one running record of the stage whose pid and start time are the wrapper's."""
    state = _own_record(tmp_path)
    _own_record(tmp_path, stage="morphologist")
    (tmp_path / ".mcp_jobs" / "corrupt.json").write_text("{corrupt")
    assert find_job_by_wrapper(str(tmp_path), "pipeline", os.getpid()) == state.job_id
    _own_record(tmp_path)
    assert find_job_by_wrapper(str(tmp_path), "pipeline", os.getpid()) is None


@pytest.mark.unit
def test_find_job_by_wrapper_ignores_reused_pid(tmp_path):
    """A record whose start time differs from the live process's is not the wrapper's."""
    _own_record(tmp_path, pid_start_time=process_start_time(os.getpid()) + 1)
    assert find_job_by_wrapper(str(tmp_path), "pipeline", os.getpid()) is None


@pytest.mark.unit
@pytest.mark.parametrize("argv", [[], ["a", "b"]])
def test_driver_main_usage_error(argv, capsys):
    """The driver takes exactly one argument, the output directory."""
    assert pipeline_driver.main(argv) == 2
    assert "usage" in capsys.readouterr().err


@pytest.mark.unit
def test_driver_main_without_wrapper_record_exits_2(tmp_path, capsys):
    """Run outside a job wrapper (no record of its parent), the driver starts nothing."""
    assert pipeline_driver.main([str(tmp_path)]) == 2
    assert "no running pipeline job" in capsys.readouterr().err


@pytest.mark.unit
async def test_drive_returns_0_when_chain_ends_the_record(tmp_path, monkeypatch):
    """drive runs the chain of the record and exits 0 once the record is terminal."""
    state = _own_record(tmp_path)
    seen = {}

    async def chain(umbrella_job_id, output_dir, active_stages, **kwargs):
        seen.update(kwargs, active_stages=active_stages)
        await job_store.update_job(output_dir, umbrella_job_id, status="succeeded")

    monkeypatch.setattr(pipeline, "_run_pipeline", chain)
    assert await pipeline_driver.drive(str(tmp_path), state.job_id) == 0
    assert seen == {"active_stages": ["morphologist"], "input_dir": str(tmp_path / "subjects")}


@pytest.mark.unit
async def test_drive_returns_1_when_record_still_running(tmp_path, monkeypatch):
    """A chain that leaves the record running makes the driver exit 1, so reconcile fails the job."""
    state = _own_record(tmp_path)

    async def chain(**_kwargs):
        return None

    monkeypatch.setattr(pipeline, "_run_pipeline", chain)
    assert await pipeline_driver.drive(str(tmp_path), state.job_id) == 1


@pytest.mark.unit
async def test_drive_refuses_a_non_pipeline_record(tmp_path):
    """A record of another stage is never run as a pipeline chain."""
    state = _own_record(tmp_path, stage="morphologist")
    assert await pipeline_driver.drive(str(tmp_path), state.job_id) == 2


@pytest.mark.unit
async def test_drive_stop_signal_cancels_the_record(tmp_path, monkeypatch):
    """SIGTERM to the driver marks the umbrella cancelled with ended_at set; the driver exits 0."""
    state = _own_record(tmp_path)

    async def chain(**_kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        for _ in range(100):
            await asyncio.sleep(0.01)
            if read_job(str(tmp_path), state.job_id).status != "running":
                return

    monkeypatch.setattr(pipeline, "_run_pipeline", chain)
    assert await pipeline_driver.drive(str(tmp_path), state.job_id) == 0
    ended = read_job(str(tmp_path), state.job_id)
    assert ended.status == "cancelled"
    assert "SIGTERM" in (ended.error or "")
    assert ended.ended_at is not None
