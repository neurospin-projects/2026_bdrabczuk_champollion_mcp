"""Seed the files start_pipeline's stage checks look for, so chain tests on synthetic trees pass the checks.

Not a test module (no test_ prefix). seed_stage_files writes one matching file per checked path, derived from
start_pipeline kwargs; it uses its own layout constants, not the production ones, so a layout change in
tools/pipeline.py still shows up in tests/test_pipeline_stage_checks.py.
"""

from __future__ import annotations

from pathlib import Path

CT_FOLDER = "cortical_tiles-2026"
MASKS = "canonical_25"
RESOLUTION = "2mm"
SUBJECT = "sub-0001"
REGION = "S.C.-sylv._left"


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
