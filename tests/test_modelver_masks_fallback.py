"""start_pipeline: one model version drives the whole run (TASK-MODELVER-BDRABCZUK-3C642461FB75, round 2).

User decision 2026-11-17: when ``masks`` is unset, the model version (``model_version``, or its
deprecated ``masks_version`` alias) is also passed to stages 2 and 3 as ``masks``, as
champollion_pipeline ``main.py`` does once ``--masks`` follows ``--model-version``. Stage 2 then
writes ``crops/<V>/2mm``, stage 4 reads it, and stages 4-6 use ``champollion_V1/<V>/``; every
pipeline-driver file check follows the same V:

V = the first non-empty start_pipeline value among masks, model_version and masks_version,
else canonical_corrected_26_1 (TASK-MODELVER-BDRABCZUK-2D826D2F0C5C).

Stage-2 argv: the real _launch_stage and start_cortical_tiles run, runner.launch is the
``launches`` recorder. Stage-3: start_config is a recorder. File checks: _launch_stage is a fake
that writes each stage's output under V, as the real stages would.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from champollion_sulcal_mcp import job_store
from champollion_sulcal_mcp.job_store import JobState, write_job
from champollion_sulcal_mcp.tools import pipeline
from tests.stage_files import (
    CT_FOLDER,
    _real_sleep,
    _touch,
    install_fake_stages,
    launch_reply,
    stage_argv_via_pipeline,
)

pytestmark = pytest.mark.unit

ALL_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]
TERMINAL = ("succeeded", "failed", "cancelled")
VERSION = "canonical_corrected_26_1"
OTHER = "canonical_25"
VERSION_PARAMS = ["model_version", "masks_version"]
GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.1"
SK = "t1mri/default_acquisition/default_analysis/segmentation"
DATASET = "DS"
REGION = "SC-sylv"
MODEL = "SC-sylv_left"


def _contains_run(argv: list[str], tokens: list[str]) -> bool:
    """True when `tokens` appear in `argv` as one contiguous run."""
    n = len(tokens)
    return any(argv[i : i + n] == tokens for i in range(len(argv) - n + 1))


@pytest.fixture
def fast_poll(monkeypatch):
    """Shorten the driver's child-poll sleep."""

    async def fast_sleep(delay, *args, **kwargs):
        await _real_sleep(0)

    monkeypatch.setattr(pipeline.asyncio, "sleep", fast_sleep)


async def _wait_terminal(output_dir: str, job_id: str) -> JobState:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while True:
        state = job_store.read_job(output_dir, job_id)
        if state.status in TERMINAL or loop.time() > deadline:
            return state
        await _real_sleep(0.01)


# --- stage 2: run_cortical_tiles.py --masks follows the model version ---


def _cortical_tiles_kwargs(base: Path) -> dict:
    return {
        "input_dir": str(base / "DEMO01" / "derivatives" / "morphologist-6.0" / "subjects"),
        "output_dir": str(base / "DEMO01"),
        "path_to_graph": GRAPH,
        "path_sk_with_hull": SK,
        "crop_path": str(base / "crops"),
        "dataset": "DEMO01",
        "models_path": str(base / "models"),
        "datasets_root": str(base / "datasets" / "DEMO01"),
        "skip_stages": [s for s in ALL_STAGES if s != "cortical_tiles"],
    }


@pytest.mark.parametrize("param", VERSION_PARAMS)
@pytest.mark.usefixtures("in_process_driver")
async def test_cortical_tiles_masks_follows_version(pipeline_dir, tmp_path, launches, monkeypatch, param):
    """Stage-2 --masks rule (supersedes REQ-MCPEMBVER-BDRABCZUK-44117031E8D9): masks unset, version V: `--masks V`."""
    argv = await stage_argv_via_pipeline(
        "cortical_tiles", _cortical_tiles_kwargs(tmp_path), launches, monkeypatch, **{param: VERSION}
    )
    assert _contains_run(argv, ["--masks", VERSION]), argv


@pytest.mark.usefixtures("in_process_driver")
async def test_cortical_tiles_masks_prefers_masks_over_model_version(pipeline_dir, tmp_path, launches, monkeypatch):
    """Stage-2 --masks rule: masks given explicitly wins over model_version."""
    argv = await stage_argv_via_pipeline(
        "cortical_tiles", _cortical_tiles_kwargs(tmp_path), launches, monkeypatch, masks=OTHER, model_version=VERSION
    )
    assert _contains_run(argv, ["--masks", OTHER]), argv
    assert VERSION not in argv, argv


# --- stage 3: start_config receives the model version as masks ---


async def _config_call(base: Path, monkeypatch, **options) -> dict:
    """Run start_pipeline with only the config stage active; return the kwargs start_config received."""
    calls: list[dict] = []
    child_dir = base / "config_child"

    async def fake_start_config(**kwargs):
        calls.append(kwargs)
        state = JobState(stage="config", status="succeeded", output_dir=str(child_dir), log_path="/tmp/fake.log")
        write_job(state)
        return launch_reply(state.job_id, "config", child_dir)

    monkeypatch.setattr(pipeline.stages, "start_config", fake_start_config)
    crop_path = base / DATASET / "derivatives" / CT_FOLDER / "crops" / VERSION / "2mm"
    _touch(crop_path / REGION / "mask" / "Rskeleton.npy")
    kwargs = {
        "input_dir": str(base / "subjects"),
        "output_dir": str(base / "pipe_out"),
        "path_to_graph": GRAPH,
        "path_sk_with_hull": SK,
        "crop_path": str(crop_path),
        "dataset": DATASET,
        "models_path": str(base / "models"),
        "datasets_root": str(base / DATASET),
        "skip_stages": [s for s in ALL_STAGES if s != "config"],
    }
    result = await pipeline.start_pipeline(**kwargs, **options)
    umbrella = await _wait_terminal(kwargs["output_dir"], result["job_id"])
    assert umbrella.status == "succeeded", f"pipeline did not finish: {umbrella.status} {umbrella.error}"
    assert len(calls) == 1, f"start_config called {len(calls)} times"
    return calls[0]


@pytest.mark.parametrize("param", VERSION_PARAMS)
@pytest.mark.usefixtures("in_process_driver", "fast_poll")
async def test_config_masks_follows_version(tmp_path, monkeypatch, param):
    """Stage-3 masks rule (supersedes REQ-MCPEMBVER-BDRABCZUK-4DF15B88BA01): masks unset, version V: masks=V."""
    call = await _config_call(tmp_path, monkeypatch, **{param: VERSION})
    assert call.get("masks") == VERSION, f"start_config received {call}"


# --- pipeline-driver file checks follow the model version ---


class VersionedStages:
    """Stand-in for pipeline._launch_stage: a succeeded child job plus the stage's output under V."""

    def __init__(self, child_dir: Path):
        self.child_dir = child_dir
        self.root: Path | None = None
        self.launched: list[str] = []

    def _write_output(self, stage_name: str) -> None:
        root = self.root
        assert root is not None
        version_dir = _version_dir(root)
        if stage_name == "cortical_tiles":
            _touch(_crops_dir(root) / REGION / "mask" / "Rskeleton.npy")
        elif stage_name == "config":
            _touch(root / "derivatives" / "champollion_V1" / "configs" / "dataset" / DATASET / f"{MODEL}.yaml")
        elif stage_name == "embeddings":
            _touch(version_dir / "region_embeddings" / MODEL / "full_embeddings.csv")
        elif stage_name == "combine":
            _touch(version_dir / "embeddings" / f"{MODEL}_embeddings.csv")
        elif stage_name == "snapshots":
            _touch(version_dir / "snapshots" / "umap.png")

    async def __call__(self, stage_name: str, umbrella_output_dir: str, **kwargs) -> dict:
        self.launched.append(stage_name)
        self._write_output(stage_name)
        child = JobState(stage=stage_name, status="succeeded", output_dir=str(self.child_dir), log_path="/tmp/x.log")
        write_job(child)
        return launch_reply(child.job_id, stage_name, self.child_dir)


@pytest.fixture
def versioned_stages(tmp_path, monkeypatch):
    return install_fake_stages(VersionedStages, tmp_path, monkeypatch)


def _dataset_root(tmp_path: Path) -> Path:
    """input_dir == output_dir == datasets_root, a chained run on one dataset root."""
    return tmp_path / "data" / DATASET


def _crops_dir(root: Path) -> Path:
    return root / "derivatives" / CT_FOLDER / "crops" / VERSION / "2mm"


def _version_dir(root: Path) -> Path:
    return root / "derivatives" / "champollion_V1" / VERSION


async def _run_only(fake: VersionedStages, tmp_path: Path, active: list[str], param: str) -> JobState:
    """start_pipeline with only `active` stages, masks unset and `param` = VERSION; wait for the umbrella."""
    root = _dataset_root(tmp_path)
    fake.root = root
    _touch(root / "sub-01" / GRAPH / "Rsub-01.arg")
    kwargs = {
        "input_dir": str(root),
        "output_dir": str(root),
        "path_to_graph": GRAPH,
        "path_sk_with_hull": SK,
        "crop_path": str(_crops_dir(root)),
        "dataset": DATASET,
        "models_path": str(tmp_path / "models"),
        "datasets_root": str(root),
        "skip_stages": [s for s in ALL_STAGES if s not in active],
        param: VERSION,
    }
    result = await pipeline.start_pipeline(**kwargs)
    return await _wait_terminal(str(root), result["job_id"])


@pytest.mark.parametrize("param", VERSION_PARAMS)
@pytest.mark.usefixtures("in_process_driver", "fast_poll")
async def test_config_check_reads_crops_under_version(tmp_path, versioned_stages, param):
    """cortical_tiles output check (supersedes REQ-MCPJOBS-BDRABCZUK-B8570836E740): crops/<V>/2mm satisfies it."""
    state = await _run_only(versioned_stages, tmp_path, ["cortical_tiles", "config"], param)
    assert versioned_stages.launched == ["cortical_tiles", "config"], state.error


@pytest.mark.parametrize("param", VERSION_PARAMS)
@pytest.mark.usefixtures("in_process_driver", "fast_poll")
async def test_embeddings_check_reads_crops_under_version(tmp_path, versioned_stages, param):
    """embeddings input check (supersedes REQ-MCPJOBS-BDRABCZUK-9E4881E055BE): crops/<V>/2mm satisfies it."""
    _touch(_crops_dir(_dataset_root(tmp_path)) / REGION / "mask" / "Rskeleton.npy")
    state = await _run_only(versioned_stages, tmp_path, ["embeddings"], param)
    assert versioned_stages.launched == ["embeddings"], state.error


@pytest.mark.parametrize("param", VERSION_PARAMS)
@pytest.mark.usefixtures("in_process_driver", "fast_poll")
async def test_combine_check_reads_region_embeddings_under_version(tmp_path, versioned_stages, param):
    """combine input check (supersedes REQ-MCPJOBS-BDRABCZUK-744BD223E318): champollion_V1/<V>/region_embeddings."""
    _touch(_version_dir(_dataset_root(tmp_path)) / "region_embeddings" / MODEL / "full_embeddings.csv")
    state = await _run_only(versioned_stages, tmp_path, ["combine"], param)
    assert versioned_stages.launched == ["combine"], state.error


@pytest.mark.parametrize("param", VERSION_PARAMS)
@pytest.mark.usefixtures("in_process_driver", "fast_poll")
async def test_snapshots_input_check_reads_combined_under_version(tmp_path, versioned_stages, param):
    """snapshots input check (supersedes REQ-MCPJOBS-BDRABCZUK-53A8D2400525): champollion_V1/<V>/embeddings."""
    _touch(_version_dir(_dataset_root(tmp_path)) / "embeddings" / f"{MODEL}_embeddings.csv")
    state = await _run_only(versioned_stages, tmp_path, ["snapshots"], param)
    assert versioned_stages.launched == ["snapshots"], state.error


@pytest.mark.parametrize("param", VERSION_PARAMS)
@pytest.mark.usefixtures("in_process_driver", "fast_poll")
async def test_snapshots_output_check_reads_png_under_version(tmp_path, versioned_stages, param):
    """snapshots output check (supersedes REQ-MCPJOBS-BDRABCZUK-DC30C76E32E8): champollion_V1/<V>/snapshots/*.png."""
    _touch(_version_dir(_dataset_root(tmp_path)) / "embeddings" / f"{MODEL}_embeddings.csv")
    state = await _run_only(versioned_stages, tmp_path, ["snapshots"], param)
    assert state.status == "succeeded", state.error
