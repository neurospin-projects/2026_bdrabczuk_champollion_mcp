"""Shared helpers of the start_pipeline chain tests: seed the checked files, fake the stages, run one stage.

Not a test module (no test_ prefix). seed_stage_files writes one matching file per checked path, derived from
start_pipeline kwargs; it uses its own layout constants, not the production ones, so a layout change in
tools/pipeline.py still shows up in tests/test_pipeline_stage_checks.py.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from champollion_sulcal_mcp import job_store
from champollion_sulcal_mcp.tools import pipeline

CT_FOLDER = "cortical_tiles-2026"
MASKS = "canonical_25"
RESOLUTION = "2mm"
SUBJECT = "sub-0001"
REGION = "S.C.-sylv._left"

_real_sleep = asyncio.sleep


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _crop(crops_dir: Path) -> None:
    _touch(crops_dir / REGION / "mask" / "Lskeleton.npy")


def seed_stage_files(kwargs: dict) -> None:
    """Write one file matching each stage check (input and output) for these start_pipeline kwargs."""
    masks = kwargs.get("masks") or MASKS
    graph = Path(SUBJECT) / kwargs["path_to_graph"] / f"R{SUBJECT}.arg"
    _touch(Path(kwargs["input_dir"]) / f"{SUBJECT}.nii.gz")
    _touch(Path(kwargs["input_dir"]) / graph)
    _touch(Path(kwargs["output_dir"]) / graph)
    _crop(Path(kwargs["crop_path"]))
    _crop(Path(kwargs["output_dir"]) / "derivatives" / CT_FOLDER / "crops" / masks / RESOLUTION)
    folder = kwargs.get("cortical_version") or CT_FOLDER
    _crop(Path(kwargs["datasets_root"]) / "derivatives" / folder / "crops" / masks / RESOLUTION)
    parts = Path(kwargs["crop_path"]).parts
    dataset = kwargs["dataset"]
    if dataset in parts:
        root = Path(*parts[: parts.index(dataset)]) / dataset
        _touch(root / "derivatives" / "champollion_V1" / "configs" / "dataset" / dataset / f"{REGION}.yaml")
    version_dir = Path(kwargs["datasets_root"]) / "derivatives" / "champollion_V1" / masks
    _touch(version_dir / "region_embeddings" / REGION / "full_embeddings.csv")
    _touch(version_dir / "embeddings" / f"{REGION}_embeddings.csv")
    _touch(version_dir / "snapshots" / "umap.png")


def launch_reply(job_id: str, stage_name: str, child_dir: Path) -> dict:
    """The reply of a fake pipeline._launch_stage for a child job in `child_dir`."""
    return {"job_id": job_id, "status": "running", "stage": stage_name, "output_dir": str(child_dir)}


def install_fake_stages(fake_class: type, tmp_path: Path, monkeypatch):
    """Replace pipeline._launch_stage with a `fake_class` writing child jobs under tmp_path/children."""
    fake = fake_class(tmp_path / "children")
    monkeypatch.setattr(pipeline, "_launch_stage", fake)
    return fake


async def stage_argv_via_pipeline(stage: str, kwargs: dict, launches: list, monkeypatch, **options) -> list[str]:
    """Run start_pipeline on `kwargs`; return the argv of the first launch recorded in `launches`.

    The real _launch_stage and start_<stage> run; runner.launch is the recorder.
    Once the stage is launched its child job is marked succeeded so the umbrella ends.
    """

    async def fast_sleep(delay, *args, **kwargs):
        await _real_sleep(0)

    monkeypatch.setattr(pipeline.asyncio, "sleep", fast_sleep)

    seed_stage_files({**kwargs, **options})
    result = await pipeline.start_pipeline(**kwargs, **options)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while not launches and loop.time() < deadline:
        await _real_sleep(0.01)
    assert launches, f"{stage} stage was never launched"

    child_dir = launches[0]["output_dir"]
    for job_file in (Path(child_dir) / ".mcp_jobs").glob("*.json"):
        await job_store.update_job(child_dir, job_file.stem, status="succeeded")

    while loop.time() < deadline:
        if job_store.read_job(kwargs["output_dir"], result["job_id"]).status in ("succeeded", "failed", "cancelled"):
            break
        await _real_sleep(0.01)
    return launches[0]["argv"]
