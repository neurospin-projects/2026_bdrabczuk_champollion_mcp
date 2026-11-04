"""start_pipeline driver: pre-launch stage checks and the post-snapshots output check.

TASK-MCPJOBS-BDRABCZUK-280700F3166D. Before launching each stage the pipeline driver
checks the files the stage will read and, when the previous stage ran in the same job,
the files that stage wrote; after snapshots it checks the snapshot PNGs. A failed check
fails the umbrella job and launches nothing further.

start_pipeline runs through the in-process driver; ``_launch_stage`` is replaced by fake
stages that persist a succeeded child job and write the stage's expected output (or not).
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

from champollion_sulcal_mcp import job_store
from champollion_sulcal_mcp.job_store import JobState, write_job
from champollion_sulcal_mcp.tools import pipeline
from tests.stage_files import install_fake_stages, launch_reply

pytestmark = [pytest.mark.unit, pytest.mark.usefixtures("in_process_driver")]

ALL_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]
TERMINAL = ("succeeded", "failed", "cancelled")
GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.3/base"
SK = "t1mri/default_acquisition/default_analysis/segmentation"
DATASET = "DS"
CT_FOLDER = "cortical_tiles-2026"  # champollion_pipeline utils/lib.py DERIVATIVES_FOLDER
MASKS = "canonical_25"
REGION = "SC-sylv"
MODEL = "SC-sylv_left"

_real_sleep = asyncio.sleep


@dataclass
class Layout:
    """Paths of one run; by default input_dir == output_dir == datasets_root (a valid chained run)."""

    root: Path
    input_dir: Path
    output_dir: Path
    datasets_root: Path
    crop_path: Path

    def kwargs(self, **overrides) -> dict:
        kwargs = {
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "path_to_graph": GRAPH,
            "path_sk_with_hull": SK,
            "crop_path": str(self.crop_path),
            "dataset": DATASET,
            "models_path": str(self.root / "models"),
            "datasets_root": str(self.datasets_root),
        }
        kwargs.update(overrides)
        return kwargs

    @property
    def ct_crops(self) -> Path:
        """Where cortical_tiles writes its crops (output_dir/derivatives/<ct folder>/crops/<masks>/2mm)."""
        return self.output_dir / "derivatives" / CT_FOLDER / "crops" / MASKS / "2mm"

    @property
    def embeddings_crops(self) -> Path:
        """Where embeddings reads crops (datasets_root/derivatives/<ct folder>/crops/<masks>/2mm)."""
        return self.datasets_root / "derivatives" / CT_FOLDER / "crops" / MASKS / "2mm"

    @property
    def configs(self) -> Path:
        """Where config writes region YAMLs: <D>/<dataset>/derivatives/champollion_V1/configs/dataset/<dataset>."""
        parts = self.crop_path.parts
        parent = Path(*parts[: parts.index(DATASET)])
        return parent / DATASET / "derivatives" / "champollion_V1" / "configs" / "dataset" / DATASET

    @property
    def version_dir(self) -> Path:
        return self.datasets_root / "derivatives" / "champollion_V1" / MASKS

    @property
    def region_embeddings(self) -> Path:
        return self.version_dir / "region_embeddings"

    @property
    def combined(self) -> Path:
        return self.version_dir / "embeddings"

    @property
    def snapshots(self) -> Path:
        return self.version_dir / "snapshots"


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def _nifti(directory: Path) -> None:
    _touch(directory / "sub-01_T1w.nii.gz")


def _graphs(directory: Path) -> None:
    _touch(directory / "sub-01" / GRAPH / "Rsub-01.arg")


def _crops(crops_dir: Path) -> None:
    _touch(crops_dir / REGION / "mask" / "Rskeleton.npy")


def _layout(tmp_path: Path) -> Layout:
    root = tmp_path / "data"
    ds = root / DATASET
    ds.mkdir(parents=True)
    _nifti(ds)
    crop_path = ds / "derivatives" / CT_FOLDER / "crops" / MASKS / "2mm"
    return Layout(root=root, input_dir=ds, output_dir=ds, datasets_root=ds, crop_path=crop_path)


class FakeStages:
    """Stand-in for pipeline._launch_stage: a succeeded child job plus the stage's expected output.

    A stage named in ``silent`` writes no output.
    """

    def __init__(self, child_dir: Path):
        self.child_dir = child_dir
        self.layout: Layout | None = None
        self.silent: set[str] = set()
        self.launched: list[str] = []

    def _write_output(self, stage_name: str) -> None:
        lay = self.layout
        assert lay is not None
        if stage_name == "morphologist":
            _graphs(lay.output_dir)
        elif stage_name == "cortical_tiles":
            _crops(lay.ct_crops)
        elif stage_name == "config":
            _touch(lay.configs / f"{MODEL}.yaml")
        elif stage_name == "embeddings":
            _touch(lay.region_embeddings / MODEL / "full_embeddings.csv")
        elif stage_name == "combine":
            _touch(lay.combined / f"{MODEL}_embeddings.csv")
        elif stage_name == "snapshots":
            _touch(lay.snapshots / "umap.png")

    async def __call__(self, stage_name: str, umbrella_output_dir: str, **kwargs) -> dict:
        self.launched.append(stage_name)
        if stage_name not in self.silent:
            self._write_output(stage_name)
        child = JobState(stage=stage_name, status="succeeded", output_dir=str(self.child_dir), log_path="/tmp/x.log")
        write_job(child)
        return launch_reply(child.job_id, stage_name, self.child_dir)


@pytest.fixture
def fake_stages(tmp_path, monkeypatch):
    return install_fake_stages(FakeStages, tmp_path, monkeypatch)


@pytest.fixture
def fast_poll(monkeypatch):
    """Shorten the driver's child-poll sleep."""

    async def fast_sleep(delay, *args, **kwargs):
        await _real_sleep(0)

    monkeypatch.setattr(pipeline.asyncio, "sleep", fast_sleep)


async def _run(fake: FakeStages, layout: Layout, **overrides) -> JobState:
    fake.layout = layout
    kwargs = layout.kwargs(**overrides)
    result = await pipeline.start_pipeline(**kwargs)
    output_dir = kwargs["output_dir"]
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while True:
        state = job_store.read_job(output_dir, result["job_id"])
        if state.status in TERMINAL or loop.time() > deadline:
            return state
        await _real_sleep(0.01)


def _assert_not_started(state: JobState, fake: FakeStages, stage: str, path: Path) -> None:
    """Umbrella failed by the pre-launch check of ``stage``, naming ``path``; ``stage`` never launched."""
    assert state.status == "failed", f"umbrella {state.status}, error {state.error!r}"
    assert stage not in fake.launched
    assert state.error is not None
    assert state.error.startswith(f"Stage '{stage}' not started: missing "), state.error
    assert f" at {path}" in state.error, state.error


# --- guards: a valid run passes every check; a skipped stage's output is not checked ---


async def test_valid_run_with_expected_outputs_succeeds(tmp_path, fake_stages, fast_poll):
    """Guard: when each stage leaves its expected output, the umbrella job succeeds after six launches."""
    state = await _run(fake_stages, _layout(tmp_path))
    assert state.status == "succeeded", state.error
    assert fake_stages.launched == ALL_STAGES


async def test_skipped_morphologist_output_is_not_checked(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-B1FC28B27BE6: morphologist skipped, graphs in input_dir alone start cortical_tiles."""
    layout = _layout(tmp_path)
    layout.input_dir = layout.root / "subjects"
    _graphs(layout.input_dir)
    state = await _run(fake_stages, layout, skip_stages=["morphologist"])
    assert "cortical_tiles" in fake_stages.launched, state.error


# --- REQ-MCPJOBS-BDRABCZUK-C7B49DD7C164: error text of a failed pre-launch check ---


async def test_check_error_text_names_stage_what_and_path(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-C7B49DD7C164: "Stage '<stage>' not started: missing <what> at <path>"."""
    layout = _layout(tmp_path)
    layout.input_dir = layout.root / "raw"
    layout.input_dir.mkdir()
    state = await _run(fake_stages, layout)
    assert state.status == "failed"
    pattern = rf"^Stage 'morphologist' not started: missing \S.* at {re.escape(str(layout.input_dir))}"
    assert re.match(pattern, state.error or ""), state.error


# --- REQ-MCPJOBS-BDRABCZUK-83FF2FED5275: a failed check launches no further stage ---


async def test_failed_check_launches_no_later_stage(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-83FF2FED5275: config's input check fails; config and later stages never launch."""
    layout = _layout(tmp_path)
    layout.crop_path = layout.root / DATASET / "empty_crops"
    state = await _run(fake_stages, layout)
    assert state.status == "failed", state.error
    assert fake_stages.launched == ["morphologist", "cortical_tiles"]


# --- per-stage checks ---


async def test_morphologist_input_without_nifti_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-EEA79D0B763F: no *.nii / *.nii.gz in input_dir fails the umbrella before morphologist."""
    layout = _layout(tmp_path)
    layout.input_dir = layout.root / "raw"
    _touch(layout.input_dir / "notes.txt")
    state = await _run(fake_stages, layout)
    _assert_not_started(state, fake_stages, "morphologist", layout.input_dir)
    assert fake_stages.launched == []


async def test_cortical_tiles_input_without_graphs_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-5C5D5454530A: no <input_dir>/*/<path_to_graph>/R*.arg fails the umbrella."""
    layout = _layout(tmp_path)
    layout.input_dir = layout.root / "subjects"
    _touch(layout.input_dir / "sub-01" / GRAPH / "Lsub-01.arg")  # left graph only: no R*.arg
    state = await _run(fake_stages, layout, skip_stages=["morphologist"])
    _assert_not_started(state, fake_stages, "cortical_tiles", layout.input_dir)


async def test_cortical_tiles_check_of_morphologist_output_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-B1FC28B27BE6: morphologist ran but left no <output_dir>/*/<path_to_graph>/R*.arg."""
    layout = _layout(tmp_path)
    layout.input_dir = layout.root / "subjects"
    _nifti(layout.input_dir)
    _graphs(layout.input_dir)  # the input check passes; only morphologist's own output is missing
    fake_stages.silent.add("morphologist")
    state = await _run(fake_stages, layout)
    _assert_not_started(state, fake_stages, "cortical_tiles", layout.output_dir)
    assert fake_stages.launched == ["morphologist"]


async def test_config_input_without_crops_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-B65534359629: no <crop_path>/*/mask/*skeleton.npy fails the umbrella."""
    layout = _layout(tmp_path)
    _touch(layout.crop_path / REGION / "mask" / "Rmask_cropped.nii.gz")  # a crop dir, no skeleton.npy
    state = await _run(fake_stages, layout, skip_stages=["morphologist", "cortical_tiles"])
    _assert_not_started(state, fake_stages, "config", layout.crop_path)


async def test_config_check_of_cortical_tiles_output_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-B8570836E740: cortical_tiles ran but left no crops under output_dir/derivatives."""
    layout = _layout(tmp_path)
    layout.crop_path = layout.root / DATASET / "user_crops" / "2mm"
    _crops(layout.crop_path)  # the input check passes; only cortical_tiles' own output is missing
    fake_stages.silent.add("cortical_tiles")
    state = await _run(fake_stages, layout)
    _assert_not_started(state, fake_stages, "config", layout.ct_crops)
    assert fake_stages.launched == ["morphologist", "cortical_tiles"]


async def test_embeddings_input_without_crops_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-9E4881E055BE: no crops under datasets_root/derivatives/<ct>/crops/<masks>/2mm."""
    layout = _layout(tmp_path)
    layout.crop_path = layout.root / DATASET / "user_crops" / "2mm"
    _crops(layout.crop_path)  # crops exist elsewhere, not where embeddings reads them
    state = await _run(fake_stages, layout, skip_stages=["morphologist", "cortical_tiles", "config"])
    _assert_not_started(state, fake_stages, "embeddings", layout.embeddings_crops)


async def test_embeddings_check_of_config_output_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-7D59509D1566: config ran but left no region YAML under the dataset configs root."""
    layout = _layout(tmp_path)
    fake_stages.silent.add("config")
    state = await _run(fake_stages, layout)
    _assert_not_started(state, fake_stages, "embeddings", layout.configs)
    assert fake_stages.launched == ["morphologist", "cortical_tiles", "config"]


async def test_combine_input_without_region_embeddings_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-744BD223E318: no region_embeddings/*/full_embeddings.csv fails the umbrella."""
    layout = _layout(tmp_path)
    _touch(layout.region_embeddings / MODEL / "pca.csv")  # a region dir, no full_embeddings.csv
    skip = ["morphologist", "cortical_tiles", "config", "embeddings"]
    state = await _run(fake_stages, layout, skip_stages=skip)
    _assert_not_started(state, fake_stages, "combine", layout.region_embeddings)


async def test_snapshots_input_without_combined_embeddings_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-53A8D2400525: no embeddings/*_embeddings.csv fails the umbrella before snapshots."""
    layout = _layout(tmp_path)
    _touch(layout.combined / "README.txt")
    skip = ["morphologist", "cortical_tiles", "config", "embeddings", "combine"]
    state = await _run(fake_stages, layout, skip_stages=skip)
    _assert_not_started(state, fake_stages, "snapshots", layout.combined)


# --- post-snapshots output check ---


async def test_snapshots_without_png_fails_umbrella(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-DC30C76E32E8: snapshots succeeded but left no snapshots/*.png: umbrella failed."""
    layout = _layout(tmp_path)
    fake_stages.silent.add("snapshots")
    _touch(layout.snapshots / "log.txt")
    state = await _run(fake_stages, layout)
    assert fake_stages.launched == ALL_STAGES
    assert state.status == "failed", state.error


async def test_snapshots_output_error_text(tmp_path, fake_stages, fast_poll):
    """REQ-MCPJOBS-BDRABCZUK-13317D55A7C2: "Stage 'snapshots' output missing: snapshot PNG at <snapshots dir>"."""
    layout = _layout(tmp_path)
    fake_stages.silent.add("snapshots")
    state = await _run(fake_stages, layout)
    assert (state.error or "").startswith("Stage 'snapshots' output missing: snapshot PNG at "), state.error
    assert str(layout.snapshots) in (state.error or ""), state.error
