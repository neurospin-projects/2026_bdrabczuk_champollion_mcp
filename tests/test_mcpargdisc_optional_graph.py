"""path_to_graph optional on start_cortical_tiles and start_pipeline (TASK-MCPARGDISC-BDRABCZUK-C8A1BFF90009).

When path_to_graph is unset, the MCP server passes no --path_to_graph to run_cortical_tiles.py, so
champollion_pipeline discovers each subject's Morphologist graph itself (pipeline Task
TASK-ARGDISC-BDRABCZUK-1601F53514C5). The start_pipeline driver's graph check then asks the pipeline's own
FileIndexStageIndexer.discover_graphs, run in the pipeline interpreter, for the number of graphs below the
checked directory; when that subprocess fails, times out or cannot start, the count is unknown and the check fails
(no second counting rule).

No pipeline code runs: the pipeline interpreter (preflight.detect().python_exe, i.e.
<CHAMPOLLION_PIPELINE_DIR>/.pixi/envs/default/bin/python) is a fake that records its argv and prints a
configured graph count as its last stdout line, or exits non-zero, or sleeps.
"""

from __future__ import annotations

import asyncio
import json
import re
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from champollion_sulcal_mcp import job_store
from champollion_sulcal_mcp.job_store import JobState, write_job
from champollion_sulcal_mcp.server import mcp
from champollion_sulcal_mcp.tools import pipeline, stages
from tests.stage_files import install_fake_stages, launch_reply

pytestmark = pytest.mark.unit

SK = "t1mri/default_acquisition/default_analysis/segmentation"
GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.1"
ALL_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]
ONLY_CORTICAL_TILES = [s for s in ALL_STAGES if s != "cortical_tiles"]
TERMINAL = ("succeeded", "failed", "cancelled")
REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE_PARAMS = REPO_ROOT / "skills" / "run-pipeline" / "references" / "stage-params.md"
SKILL = REPO_ROOT / "skills" / "run-pipeline" / "SKILL.md"

_real_sleep = asyncio.sleep

# The fake pipeline interpreter: records its argv, then answers from a JSON config.
_FAKE_PYTHON = """
import json, sys, time
cfg = json.load(open({cfg!r}))
with open({record!r}, "a") as out:
    out.write(json.dumps(sys.argv[1:]) + "\\n")
time.sleep(cfg.get("sleep", 0))
if cfg.get("exit", 0):
    sys.stderr.write("ModuleNotFoundError: champollion_pipeline.file_indexer\\n")
    sys.exit(cfg["exit"])
joined = " ".join(sys.argv[1:])
count = cfg.get("default", 0)
for directory, n in cfg.get("counts", {{}}).items():
    if directory in joined:
        count = n
print(count)
"""


@dataclass
class FakeDiscovery:
    """Controls the fake pipeline interpreter: per-directory graph counts, exit status, delay."""

    cfg_path: Path
    record_path: Path
    counts: dict[str, int] = field(default_factory=dict)
    default: int = 0
    exit: int = 0
    sleep: float = 0.0

    def save(self) -> None:
        cfg = {"counts": self.counts, "default": self.default, "exit": self.exit, "sleep": self.sleep}
        self.cfg_path.write_text(json.dumps(cfg))

    def calls(self) -> list[list[str]]:
        if not self.record_path.exists():
            return []
        return [json.loads(line) for line in self.record_path.read_text().splitlines() if line]


@pytest.fixture
def discovery(pipeline_dir, tmp_path) -> FakeDiscovery:
    """A fake pixi interpreter in the fake pipeline checkout (preflight.detect() prefers it)."""
    fake = FakeDiscovery(cfg_path=tmp_path / "discovery.json", record_path=tmp_path / "discovery_calls.jsonl")
    fake.save()
    code = _FAKE_PYTHON.format(cfg=str(fake.cfg_path), record=str(fake.record_path))
    (tmp_path / "fake_python.py").write_text(code)
    python = pipeline_dir / ".pixi" / "envs" / "default" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{tmp_path / "fake_python.py"}" "$@"\n')
    python.chmod(python.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return fake


@pytest.fixture
def fast_poll(monkeypatch):
    async def fast_sleep(delay, *args, **kwargs):
        await _real_sleep(0)

    monkeypatch.setattr(pipeline.asyncio, "sleep", fast_sleep)


class FakeStages:
    """Stand-in for pipeline._launch_stage: records the stage and its kwargs, writes a succeeded child job."""

    def __init__(self, child_dir: Path):
        self.child_dir = child_dir
        self.launched: list[str] = []
        self.kwargs: dict[str, dict] = {}

    async def __call__(self, stage_name: str, umbrella_output_dir: str, **kwargs) -> dict:
        self.launched.append(stage_name)
        self.kwargs[stage_name] = kwargs
        child = JobState(stage=stage_name, status="succeeded", output_dir=str(self.child_dir), log_path="/tmp/x.log")
        write_job(child)
        return launch_reply(child.job_id, stage_name, self.child_dir)


@pytest.fixture
def fake_stages(tmp_path, monkeypatch):
    return install_fake_stages(FakeStages, tmp_path, monkeypatch)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Distinct input and output directories; neither path is a substring of the other."""
    input_dir = tmp_path / "data" / "raw_subjects"
    output_dir = tmp_path / "data" / "derivs_out"
    input_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    return input_dir, output_dir


def _pipeline_kwargs(input_dir: Path, output_dir: Path, **overrides) -> dict:
    """start_pipeline arguments without path_to_graph."""
    kwargs = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "path_sk_with_hull": SK,
        "crop_path": str(output_dir / "crops"),
        "dataset": "DS",
        "models_path": str(output_dir / "models"),
        "datasets_root": str(output_dir),
        "skip_stages": ONLY_CORTICAL_TILES,
    }
    kwargs.update(overrides)
    return kwargs


async def _run(**kwargs) -> JobState:
    result = await pipeline.start_pipeline(**kwargs)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10.0
    while True:
        state = job_store.read_job(kwargs["output_dir"], result["job_id"])
        if state.status in TERMINAL or loop.time() > deadline:
            return state
        await _real_sleep(0.01)


def _no_t1mri_graph(directory: Path, subject: str = "sub-01") -> None:
    """A Morphologist graph in a layout without t1mri/ (morphologist-cli style)."""
    _touch(directory / subject / "ses-1" / "anat" / "folds" / f"R{subject}.arg")


# --- schema: path_to_graph optional ---


async def test_cortical_tiles_schema_path_to_graph_optional():
    """REQ-MCPARGDISC-BDRABCZUK-8CEFE4901F37: start_cortical_tiles lists path_to_graph as an optional input property."""
    tool = await mcp.get_tool("start_cortical_tiles")
    assert "path_to_graph" in tool.parameters["properties"]
    assert "path_to_graph" not in tool.parameters.get("required", [])


async def test_pipeline_schema_path_to_graph_optional():
    """REQ-MCPARGDISC-BDRABCZUK-1BAE790788EB: start_pipeline lists path_to_graph as an optional input property."""
    tool = await mcp.get_tool("start_pipeline")
    assert "path_to_graph" in tool.parameters["properties"]
    assert "path_to_graph" not in tool.parameters.get("required", [])


# --- start_cortical_tiles argv and roots ---


async def test_cortical_tiles_omitted_graph_has_no_graph_flag(pipeline_dir, launches, tmp_path):
    """REQ-MCPARGDISC-BDRABCZUK-9F043F8B65D9 (and REQ-MCPARGDISC-BDRABCZUK-47D54C45AA8C).

    Path_to_graph omitted, run_cortical_tiles.py gets no --path_to_graph.
    """
    input_dir, output_dir = _dirs(tmp_path)
    result = await stages.start_cortical_tiles(
        input_dir=str(input_dir), output_dir=str(output_dir), path_sk_with_hull=SK
    )
    assert result["stage"] == "cortical_tiles"
    argv = launches[0]["argv"]
    assert "--path_to_graph" not in argv, argv
    assert "--path_sk_with_hull" in argv, argv


async def test_cortical_tiles_given_graph_passed_unchanged(pipeline_dir, launches, tmp_path):
    """Guard for REQ-MCPARGDISC-BDRABCZUK-9F043F8B65D9: a given path_to_graph is passed unchanged."""
    input_dir, output_dir = _dirs(tmp_path)
    await stages.start_cortical_tiles(
        input_dir=str(input_dir), output_dir=str(output_dir), path_to_graph=GRAPH, path_sk_with_hull=SK
    )
    argv = launches[0]["argv"]
    index = argv.index("--path_to_graph")
    assert argv[index + 1] == GRAPH


async def test_cortical_tiles_omitted_graph_launches_under_roots(pipeline_dir, mock_roots, launches):
    """REQ-MCPARGDISC-BDRABCZUK-799B90031395.

    Roots declared, path_to_graph omitted, the job launches (no roots check on path_to_graph).
    """
    ctx, root = mock_roots
    result = await stages.start_cortical_tiles(
        input_dir=str(root / "subjects"), output_dir=str(root / "derivatives"), path_sk_with_hull=SK, ctx=ctx
    )
    assert result["stage"] == "cortical_tiles"
    assert len(launches) == 1
    assert "--path_to_graph" not in launches[0]["argv"]


# --- start_pipeline: stage 2 argv ---


async def test_pipeline_omitted_graph_stage2_has_no_graph_flag(
    in_process_driver, launches, discovery, fast_poll, tmp_path
):
    """REQ-MCPARGDISC-BDRABCZUK-62D28084F999 (and REQ-MCPARGDISC-BDRABCZUK-47D54C45AA8C).

    Start_pipeline without path_to_graph launches run_cortical_tiles.py without it.
    """
    input_dir, output_dir = _dirs(tmp_path)
    _no_t1mri_graph(input_dir)
    discovery.counts = {str(input_dir): 1}
    discovery.save()
    result = await pipeline.start_pipeline(**_pipeline_kwargs(input_dir, output_dir))
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 10.0
    while not launches and loop.time() < deadline:
        state = job_store.read_job(str(output_dir), result["job_id"])
        assert state.status == "running", state.error
        await _real_sleep(0.01)
    assert launches, "cortical_tiles was never launched"
    argv = launches[0]["argv"]
    assert launches[0]["stage"] == "cortical_tiles"
    assert "--path_to_graph" not in argv, argv
    child_dir = launches[0]["output_dir"]
    for job_file in (Path(child_dir) / ".mcp_jobs").glob("*.json"):
        await job_store.update_job(child_dir, job_file.stem, status="succeeded")
    while loop.time() < deadline:
        if job_store.read_job(str(output_dir), result["job_id"]).status in TERMINAL:
            break
        await _real_sleep(0.01)


# --- preflight graph check with path_to_graph unset ---


async def test_preflight_runs_pipeline_discover_graphs(in_process_driver, fake_stages, discovery, fast_poll, tmp_path):
    """REQ-MCPARGDISC-BDRABCZUK-BDBAB826CFAE.

    The graph count comes from discover_graphs in the pipeline interpreter, not from a glob.

    No .arg file exists, yet the fake discovery counts two graphs: cortical_tiles launches.
    """
    input_dir, output_dir = _dirs(tmp_path)
    discovery.counts = {str(input_dir): 2}
    discovery.save()
    state = await _run(**_pipeline_kwargs(input_dir, output_dir))
    assert "cortical_tiles" in fake_stages.launched, state.error
    assert fake_stages.kwargs["cortical_tiles"].get("path_to_graph") is None
    calls = discovery.calls()
    assert calls, "the pipeline interpreter was never run"
    joined = " ".join(calls[0])
    assert "discover_graphs" in joined, calls[0]
    assert str(input_dir) in joined, calls[0]


async def test_preflight_given_graph_runs_no_discovery(in_process_driver, fake_stages, discovery, fast_poll, tmp_path):
    """Guard for REQ-MCPARGDISC-BDRABCZUK-BDBAB826CFAE.

    A given path_to_graph keeps the */<path_to_graph>/R*.arg glob and runs no subprocess.
    """
    input_dir, output_dir = _dirs(tmp_path)
    _touch(input_dir / "sub-01" / GRAPH / "Rsub-01.arg")
    state = await _run(**_pipeline_kwargs(input_dir, output_dir, path_to_graph=GRAPH))
    assert "cortical_tiles" in fake_stages.launched, state.error
    assert discovery.calls() == []


async def test_preflight_discovery_none_in_input_dir_fails(
    in_process_driver, fake_stages, discovery, fast_poll, tmp_path
):
    """REQ-MCPARGDISC-BDRABCZUK-159BBEC6CA95.

    Discovery counts no graph below input_dir, the umbrella fails before cortical_tiles.

    A graph file is present, so a glob would pass: discovery is the one rule.
    """
    input_dir, output_dir = _dirs(tmp_path)
    _no_t1mri_graph(input_dir)
    discovery.counts = {str(input_dir): 0}
    discovery.save()
    state = await _run(**_pipeline_kwargs(input_dir, output_dir))
    assert state.status == "failed", state.error
    assert "cortical_tiles" not in fake_stages.launched
    assert (state.error or "").startswith("Stage 'cortical_tiles' not started: missing "), state.error
    assert f" at {input_dir}" in (state.error or ""), state.error


async def test_preflight_discovery_none_in_morphologist_output_fails(
    in_process_driver, fake_stages, discovery, fast_poll, tmp_path
):
    """REQ-MCPARGDISC-BDRABCZUK-E159E4E3A915.

    Morphologist ran in this job; discovery counts no graph below output_dir; cortical_tiles never launches.
    """
    input_dir, output_dir = _dirs(tmp_path)
    _touch(input_dir / "sub-01_T1w.nii.gz")
    _no_t1mri_graph(input_dir)
    discovery.counts = {str(input_dir): 3, str(output_dir): 0}
    discovery.save()
    kwargs = _pipeline_kwargs(input_dir, output_dir, skip_stages=["config", "embeddings", "combine", "snapshots"])
    state = await _run(**kwargs)
    assert fake_stages.launched == ["morphologist"], state.error
    assert state.status == "failed", state.error
    assert (state.error or "").startswith("Stage 'cortical_tiles' not started: missing "), state.error
    assert f" at {output_dir}" in (state.error or ""), state.error


@pytest.mark.parametrize("graph_present", [True, False])
async def test_preflight_discovery_error_fails_graph_count_unknown(
    in_process_driver, fake_stages, discovery, fast_poll, tmp_path, graph_present
):
    """REQ-MCPARGDISC-BDRABCZUK-DDF2C77E5418.

    Discovery exits non-zero; the count is unknown and the umbrella fails, whatever *.arg files exist.
    """
    input_dir, output_dir = _dirs(tmp_path)
    if graph_present:
        _no_t1mri_graph(input_dir)
    discovery.exit = 1
    discovery.save()
    state = await _run(**_pipeline_kwargs(input_dir, output_dir))
    assert discovery.calls(), "the pipeline interpreter was never run"
    assert state.status == "failed", state.error
    assert "cortical_tiles" not in fake_stages.launched
    assert "graph count unknown" in (state.error or ""), state.error


@pytest.mark.parametrize("graph_present", [True, False])
async def test_preflight_discovery_timeout_fails_graph_count_unknown(
    in_process_driver, fake_stages, discovery, fast_poll, tmp_path, monkeypatch, graph_present
):
    """REQ-MCPARGDISC-BDRABCZUK-DE684557D01A: discovery runs past GRAPH_DISCOVERY_TIMEOUT_S; the count is unknown.

    The fake would report 5 graphs after 4 s: the umbrella still fails, and the subprocess is stopped at the timeout.
    """
    monkeypatch.setattr(pipeline, "GRAPH_DISCOVERY_TIMEOUT_S", 0.5, raising=False)
    input_dir, output_dir = _dirs(tmp_path)
    if graph_present:
        _no_t1mri_graph(input_dir)
    discovery.default = 5
    discovery.sleep = 4.0
    discovery.save()
    loop = asyncio.get_running_loop()
    started = loop.time()
    state = await _run(**_pipeline_kwargs(input_dir, output_dir))
    assert discovery.calls(), "the pipeline interpreter was never run"
    assert state.status == "failed", state.error
    assert "cortical_tiles" not in fake_stages.launched
    assert "graph count unknown" in (state.error or ""), state.error
    assert loop.time() - started < 3.5, "the discovery subprocess was not stopped at its timeout"


@pytest.mark.parametrize("failure", ["exit", "timeout", "cannot_start"])
async def test_preflight_unknown_count_error_names_reason(
    in_process_driver, fake_stages, discovery, fast_poll, tmp_path, monkeypatch, pipeline_dir, failure
):
    """REQ-MCPARGDISC-BDRABCZUK-C860E47D526A.

    The error keeps the 'Stage ... not started: missing' form, contains 'graph count unknown' and the reason.
    """
    input_dir, output_dir = _dirs(tmp_path)
    if failure == "exit":
        discovery.exit = 1
        reason = "ModuleNotFoundError"
    elif failure == "timeout":
        monkeypatch.setattr(pipeline, "GRAPH_DISCOVERY_TIMEOUT_S", 0.5, raising=False)
        discovery.sleep = 4.0
        reason = "timed out"
    else:
        python = pipeline_dir / ".pixi" / "envs" / "default" / "bin" / "python"
        python.chmod(0o644)
        reason = "could not start"
    discovery.save()
    state = await _run(**_pipeline_kwargs(input_dir, output_dir))
    assert state.status == "failed", state.error
    assert "cortical_tiles" not in fake_stages.launched
    error = state.error or ""
    assert error.startswith("Stage 'cortical_tiles' not started: missing "), error
    assert "graph count unknown" in error, error
    assert reason in error, error


# --- docs ---


def _table_row(text: str, section: str, name: str) -> list[str]:
    """Cells of the `name` row in the first table after the heading line containing `section`."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("#") and section in line)
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and cells[0] == f"`{name}`":
            return cells
    raise AssertionError(f"no `{name}` row under {section!r}")


def test_stage_params_path_to_graph_not_required():
    """REQ-MCPARGDISC-BDRABCZUK-F532353B872B.

    The start_cortical_tiles Parameter table marks path_to_graph as not required.
    """
    cells = _table_row(STAGE_PARAMS.read_text(), "start_cortical_tiles", "path_to_graph")
    assert cells[2] == "no", cells


def test_skill_path_to_graph_row_names_graph_discovery():
    """REQ-MCPARGDISC-BDRABCZUK-1EC3D4695573.

    SKILL.md's path_to_graph row says an unset value leaves graph finding to graph discovery.
    """
    rows = [line for line in SKILL.read_text().splitlines() if re.match(r"^\|\s*`path_to_graph`\s*\|", line)]
    assert rows, "no path_to_graph row in SKILL.md"
    assert any("graph discovery" in row for row in rows), rows
