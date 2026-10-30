from __future__ import annotations

import asyncio
import contextlib
import glob
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastmcp import Context
from fastmcp.exceptions import ToolError

from .. import job_store, runner
from ..job_store import JobState
from . import stages

logger = logging.getLogger(__name__)

DERIVATIVES_SUBDIR = "derivatives"
CHAMPOLLION_SUBDIR = "champollion_V1"
EMBEDDINGS_SUBDIR = "embeddings"
SNAPSHOTS_SUBDIR = "snapshots"
REGION_EMBEDDINGS_SUBDIR = "region_embeddings"
DEFAULT_MASKS_VERSION = "canonical_25"

DRIVER_MODULE = "champollion_sulcal_mcp.pipeline_driver"
CHILD_POLL_S = 2.0

STAGE_ORDER = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]


def _compute_masks_version_dir(datasets_root: str, masks: str | None) -> Path:
    """Return the champollion_V1 directory of one mask version, shared by stages 5 and 6.

    Mirrors champollion_pipeline's derivatives_layout.compute_masks_version_dir, so
    start_pipeline reads and writes the same tree as the pipeline CLI.

    Args:
        datasets_root: dataset root (the directory holding ``derivatives/``).
        masks: mask version tag passed to start_pipeline; empty or None means the
            pipeline default, DEFAULT_MASKS_VERSION. Used for paths only.

    Returns:
        Path(datasets_root) / "derivatives" / "champollion_V1" / (masks or "canonical_25").

    Complexity: O(1).
    """
    return Path(datasets_root) / DERIVATIVES_SUBDIR / CHAMPOLLION_SUBDIR / (masks or DEFAULT_MASKS_VERSION)


# --- Stage file checks: each stage's input before launch, the previous stage's output when it ran in this job,
# --- and snapshots' own output after its job succeeds. Present = at least one file matching a pattern.

CORTICAL_TILES_FOLDER = "cortical_tiles-2026"  # mirrors champollion_pipeline utils/lib.py DERIVATIVES_FOLDER
CROPS_SUBDIR = "crops"
CROP_RESOLUTION = "2mm"
CONFIGS_SUBDIR = "configs"
DATASET_CONFIGS_SUBDIR = "dataset"
_POST_RUN_STAGES = ("snapshots",)

_T1_LABEL = "T1 MRI (*.nii, *.nii.gz)"
_CROPS_LABEL = "skeleton crops (*/mask/*skeleton.npy)"
_CROPS_PATTERNS = ("*/mask/*skeleton.npy",)


@dataclass(frozen=True)
class FileCheck:
    """One expected-file check: ``what`` names it in the error, ``directory`` is the path the error gives."""

    what: str
    directory: Path
    patterns: tuple[str, ...]

    def present(self) -> bool:
        """True iff directory is a directory holding at least one regular file matching a pattern; never raises."""
        try:
            if not self.directory.is_dir():
                return False
            return any(match.is_file() for pattern in self.patterns for match in self.directory.glob(pattern))
        except (OSError, ValueError, NotImplementedError):
            return False


CheckBuilder = Callable[[str, Mapping], FileCheck]


def _graphs_label(kwargs: Mapping) -> str:
    return f"Morphologist graphs (*/{kwargs['path_to_graph']}/R*.arg)"


def _graphs_patterns(kwargs: Mapping) -> tuple[str, ...]:
    return (f"*/{glob.escape(kwargs['path_to_graph'])}/R*.arg",)


def _crops_dir(root: str, folder: str, masks: str | None) -> Path:
    return Path(root) / DERIVATIVES_SUBDIR / folder / CROPS_SUBDIR / (masks or DEFAULT_MASKS_VERSION) / CROP_RESOLUTION


def _configs_dir(crop_path: str, dataset: str) -> Path | None:
    """Region configs directory of ``dataset``, rooted at the first ``dataset`` component of crop_path; else None."""
    parts = Path(crop_path).parts
    if dataset not in parts:
        return None
    root = Path(*parts[: parts.index(dataset)])
    return root / dataset / DERIVATIVES_SUBDIR / CHAMPOLLION_SUBDIR / CONFIGS_SUBDIR / DATASET_CONFIGS_SUBDIR / dataset


def _morphologist_input(output_dir: str, kwargs: Mapping) -> FileCheck:
    return FileCheck(_T1_LABEL, Path(kwargs["input_dir"]), ("*.nii", "*.nii.gz"))


def _morphologist_output(output_dir: str, kwargs: Mapping) -> FileCheck:
    return FileCheck(_graphs_label(kwargs), Path(output_dir), _graphs_patterns(kwargs))


def _cortical_tiles_input(output_dir: str, kwargs: Mapping) -> FileCheck:
    return FileCheck(_graphs_label(kwargs), Path(kwargs["input_dir"]), _graphs_patterns(kwargs))


def _cortical_tiles_output(output_dir: str, kwargs: Mapping) -> FileCheck:
    return FileCheck(_CROPS_LABEL, _crops_dir(output_dir, CORTICAL_TILES_FOLDER, kwargs.get("masks")), _CROPS_PATTERNS)


def _config_input(output_dir: str, kwargs: Mapping) -> FileCheck:
    return FileCheck(_CROPS_LABEL, Path(kwargs["crop_path"]), _CROPS_PATTERNS)


def _config_output(output_dir: str, kwargs: Mapping) -> FileCheck:
    directory = _configs_dir(kwargs["crop_path"], kwargs["dataset"])
    if directory is None:
        return FileCheck(f"dataset directory {kwargs['dataset']!r} in crop_path", Path(kwargs["crop_path"]), ())
    return FileCheck("region configs (*.yaml)", directory, ("*.yaml",))


def _embeddings_input(output_dir: str, kwargs: Mapping) -> FileCheck:
    folder = kwargs.get("cortical_version") or CORTICAL_TILES_FOLDER
    return FileCheck(_CROPS_LABEL, _crops_dir(kwargs["datasets_root"], folder, kwargs.get("masks")), _CROPS_PATTERNS)


def _combine_input(output_dir: str, kwargs: Mapping) -> FileCheck:
    version_dir = _compute_masks_version_dir(kwargs["datasets_root"], kwargs.get("masks"))
    return FileCheck(
        "region embeddings (*/full_embeddings.csv)", version_dir / REGION_EMBEDDINGS_SUBDIR, ("*/full_embeddings.csv",)
    )


def _snapshots_input(output_dir: str, kwargs: Mapping) -> FileCheck:
    version_dir = _compute_masks_version_dir(kwargs["datasets_root"], kwargs.get("masks"))
    return FileCheck("combined embeddings (*_embeddings.csv)", version_dir / EMBEDDINGS_SUBDIR, ("*_embeddings.csv",))


def _snapshots_output(output_dir: str, kwargs: Mapping) -> FileCheck:
    version_dir = _compute_masks_version_dir(kwargs["datasets_root"], kwargs.get("masks"))
    return FileCheck("snapshot PNG", version_dir / SNAPSHOTS_SUBDIR, ("*.png",))


_INPUT_CHECKS: dict[str, CheckBuilder] = {
    "morphologist": _morphologist_input,
    "cortical_tiles": _cortical_tiles_input,
    "config": _config_input,
    "embeddings": _embeddings_input,
    "combine": _combine_input,
    "snapshots": _snapshots_input,
}
_OUTPUT_CHECKS: dict[str, CheckBuilder] = {
    "morphologist": _morphologist_output,
    "cortical_tiles": _cortical_tiles_output,
    "config": _config_output,
    "snapshots": _snapshots_output,
}


def _pre_launch_checks(stage_name: str, output_dir: str, done: Sequence[str], kwargs: Mapping) -> list[FileCheck]:
    """The previous stage's output (when it ran in this job), then this stage's input; [] for an unknown stage."""
    if stage_name not in STAGE_ORDER:
        return []
    index = STAGE_ORDER.index(stage_name)
    previous = STAGE_ORDER[index - 1] if index else None
    checks = []
    if previous in done and previous in _OUTPUT_CHECKS:
        checks.append(_OUTPUT_CHECKS[previous](output_dir, kwargs))
    if stage_name in _INPUT_CHECKS:
        checks.append(_INPUT_CHECKS[stage_name](output_dir, kwargs))
    return checks


def _post_run_checks(stage_name: str, output_dir: str, kwargs: Mapping) -> list[FileCheck]:
    return [_OUTPUT_CHECKS[stage_name](output_dir, kwargs)] if stage_name in _POST_RUN_STAGES else []


def _first_missing(checks: Sequence[FileCheck]) -> FileCheck | None:
    return next((check for check in checks if not check.present()), None)


async def _files_ok(output_dir: str, umbrella_job_id: str, checks: Sequence[FileCheck], prefix: str) -> bool:
    """True when every check finds its files; else fail the umbrella with "<prefix> <what> at <directory>"."""
    if not checks:
        return True
    missing = await asyncio.to_thread(_first_missing, list(checks))
    if missing is None:
        return True
    await _finish(output_dir, umbrella_job_id, status="failed", error=f"{prefix} {missing.what} at {missing.directory}")
    return False


async def start_pipeline(
    input_dir: str,
    output_dir: str,
    path_to_graph: str,
    path_sk_with_hull: str,
    crop_path: str,
    dataset: str,
    models_path: str,
    datasets_root: str,
    sk_qc_path: str | None = None,
    njobs: int | None = None,
    parallel: bool = False,
    cpu: bool = False,
    skip_stages: list[str] | None = None,
    labelling_session: str | None = None,
    reference_data_dir: str | None = None,
    overwrite: bool = False,
    run_cka: bool = False,
    regions: list[str] | None = None,
    masks: str | None = None,
    masks_version: str | None = None,
    cortical_version: str | None = None,
    log: bool = True,
    ctx: Context | None = None,
) -> dict:
    """Launch the full Champollion pipeline (all 6 stages sequentially). Returns a pipeline job_id immediately.

    ``reference_data_dir`` is optional and forwarded to stage 6 (snapshots) for UMAP plots.

    A non-empty ``masks`` is forwarded to stages 2, 3 and 4, and names the
    ``<datasets_root>/derivatives/champollion_V1/<masks or canonical_25>/`` directory that
    stages 5 and 6 read and write (``region_embeddings``, ``embeddings``, ``snapshots``).
    ``overwrite``, ``run_cka``, ``regions``, ``masks_version`` and ``cortical_version``
    apply to stage 4 (embeddings) only. ``regions`` are model names (e.g. ``SCsylv_left``);
    ``cortical_version`` must name where the stage-2 crops actually are.
    """
    skip = set(skip_stages or [])
    active_stages = [s for s in STAGE_ORDER if s not in skip]
    chain = {
        "input_dir": input_dir,
        "path_to_graph": path_to_graph,
        "path_sk_with_hull": path_sk_with_hull,
        "crop_path": crop_path,
        "dataset": dataset,
        "models_path": models_path,
        "datasets_root": datasets_root,
        "sk_qc_path": sk_qc_path,
        "njobs": njobs,
        "parallel": parallel,
        "cpu": cpu,
        "labelling_session": labelling_session,
        "reference_data_dir": reference_data_dir,
        "overwrite": overwrite,
        "run_cka": run_cka,
        "regions": regions,
        "masks": masks,
        "masks_version": masks_version,
        "cortical_version": cortical_version,
        "log": log,
    }
    snapshot = {
        "input_dir": input_dir,
        "output_dir": output_dir,
        "skip_stages": sorted(skip),
        "active_stages": active_stages,
        "chain": chain,
    }
    state = await runner.launch(
        stage="pipeline",
        argv=_driver_argv(output_dir),
        output_dir=output_dir,
        cwd=output_dir,
        env=_driver_env(),
        args_snapshot=snapshot,
    )
    if state.status == "failed":
        raise ToolError(f"Pipeline driver failed to start: {state.error}")
    await _initial_progress(output_dir, state.job_id, active_stages)

    if ctx:
        await ctx.info(f"Launching pipeline: {len(active_stages)} stages, job_id={state.job_id}")

    return {"job_id": state.job_id, "status": "running", "stage": "pipeline", "active_stages": active_stages}


def _driver_argv(output_dir: str) -> list[str]:
    """Command of the detached pipeline driver; ``-P`` keeps the cwd off sys.path."""
    return [sys.executable, "-P", "-m", DRIVER_MODULE, output_dir]


def _driver_env() -> dict[str, str]:
    """The server environment, with the server's package tree first on PYTHONPATH."""
    paths = [runner.PKG_ROOT]
    if os.environ.get("PYTHONPATH"):
        paths.append(os.environ["PYTHONPATH"])
    return {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}


async def _initial_progress(output_dir: str, job_id: str, active_stages: list[str]) -> None:
    """Write the initial progress unless the driver already wrote progress or ended the job."""

    def decide(state: JobState) -> dict | None:
        if state.status != "running" or state.progress.stages_total is not None:
            return None
        current = active_stages[0] if active_stages else None
        return {"progress": {"stages_done": 0, "stages_total": len(active_stages), "current_stage": current}}

    with contextlib.suppress(FileNotFoundError):
        await job_store.update_job_with(output_dir, job_id, decide)


def _while_running(fields: dict) -> Callable[[JobState], dict | None]:
    return lambda state: fields if state.status == "running" else None


async def _finish(output_dir: str, umbrella_job_id: str, **fields) -> None:
    """End the umbrella job, unless it is no longer running (a cancel is never overwritten)."""
    fields["ended_at"] = datetime.now(UTC).isoformat()
    await job_store.update_job_with(output_dir, umbrella_job_id, _while_running(fields))


async def _progress(output_dir: str, umbrella_job_id: str, **progress) -> None:
    await job_store.update_job_with(output_dir, umbrella_job_id, _while_running({"progress": progress}))


async def _run_pipeline(
    umbrella_job_id: str,
    output_dir: str,
    active_stages: list[str],
    **kwargs,
) -> None:
    try:
        await _execute_pipeline(umbrella_job_id, output_dir, active_stages, **kwargs)
    except Exception as exc:
        logger.exception("Pipeline task crashed unexpectedly")
        with contextlib.suppress(Exception):
            await _finish(output_dir, umbrella_job_id, status="failed", error=f"Pipeline crashed: {exc}")


def _umbrella_running(output_dir: str, umbrella_job_id: str) -> bool:
    try:
        return job_store.read_job(output_dir, umbrella_job_id).status == "running"
    except FileNotFoundError:
        return False


async def _execute_pipeline(
    umbrella_job_id: str,
    output_dir: str,
    active_stages: list[str],
    **kwargs,
) -> None:
    total = len(active_stages)
    for i, stage_name in enumerate(active_stages):
        if not _umbrella_running(output_dir, umbrella_job_id):
            return
        await _progress(output_dir, umbrella_job_id, current_stage=stage_name, stages_done=i, stages_total=total)
        checks = _pre_launch_checks(stage_name, output_dir, active_stages[:i], kwargs)
        if not await _files_ok(output_dir, umbrella_job_id, checks, f"Stage {stage_name!r} not started: missing"):
            return
        try:
            result = await _launch_stage(stage_name, umbrella_output_dir=output_dir, **kwargs)
        except ToolError as e:
            await _finish(
                output_dir, umbrella_job_id, status="failed", error=f"Stage {stage_name!r} failed to launch: {e}"
            )
            return
        status = await _await_child(output_dir, umbrella_job_id, stage_name, result)
        if status is None:
            return
        if status != "succeeded":
            error = f"Stage {stage_name!r} failed (child job {result['job_id']}). Status: {status}."
            await _finish(output_dir, umbrella_job_id, status="failed", error=error)
            return
        checks = _post_run_checks(stage_name, output_dir, kwargs)
        if not await _files_ok(output_dir, umbrella_job_id, checks, f"Stage {stage_name!r} output missing:"):
            return
    await _finish(
        output_dir, umbrella_job_id, status="succeeded", progress={"stages_done": total, "stages_total": total}
    )


async def _await_child(output_dir: str, umbrella_job_id: str, stage_name: str, result: dict) -> str | None:
    """Final status of the stage's child job, or None when the chain must stop (umbrella ended or child gone)."""
    child_id = result["job_id"]
    child_output_dir = result["output_dir"]  # use child's actual output_dir for polling
    await _progress(output_dir, umbrella_job_id, current_stage=stage_name, current_child_job_id=child_id)
    while True:
        await asyncio.sleep(CHILD_POLL_S)
        try:
            child = job_store.read_job(child_output_dir, child_id)
        except FileNotFoundError:
            error = f"Stage {stage_name!r} job file disappeared unexpectedly."
            await _finish(output_dir, umbrella_job_id, status="failed", error=error)
            return None
        if child.status in ("succeeded", "failed", "cancelled"):
            return child.status
        if not _umbrella_running(output_dir, umbrella_job_id):
            # Propagate cancellation
            try:
                await runner.cancel(child_output_dir, child_id)
            except (FileNotFoundError, runner.CancelRefused) as error:
                logger.warning("cancel of stage job %s: %s", child_id, error)
            return None


async def _launch_stage(stage_name: str, umbrella_output_dir: str, **kwargs) -> dict:
    log = kwargs.get("log", True)
    if stage_name == "morphologist":
        return await stages.start_morphologist(
            input_dir=kwargs["input_dir"],
            output_dir=umbrella_output_dir,
            parallel=kwargs.get("parallel", False),
            log=log,
        )
    elif stage_name == "cortical_tiles":
        return await stages.start_cortical_tiles(
            input_dir=kwargs["input_dir"],
            output_dir=str(Path(umbrella_output_dir) / DERIVATIVES_SUBDIR),
            path_to_graph=kwargs["path_to_graph"],
            path_sk_with_hull=kwargs["path_sk_with_hull"],
            sk_qc_path=kwargs.get("sk_qc_path"),
            njobs=kwargs.get("njobs"),
            labelling_session=kwargs.get("labelling_session"),
            masks=kwargs.get("masks"),
            log=log,
        )
    elif stage_name == "config":
        masks = kwargs.get("masks")
        return await stages.start_config(
            crop_path=kwargs["crop_path"],
            dataset=kwargs["dataset"],
            **({"masks": masks} if masks else {}),
            log=log,
        )
    elif stage_name == "embeddings":
        return await stages.start_embeddings(
            models_path=kwargs["models_path"],
            datasets_root=kwargs["datasets_root"],
            cpu=kwargs.get("cpu", False),
            overwrite=kwargs.get("overwrite", False),
            run_cka=kwargs.get("run_cka", False),
            regions=kwargs.get("regions"),
            masks=kwargs.get("masks"),
            masks_version=kwargs.get("masks_version"),
            cortical_version=kwargs.get("cortical_version"),
            log=log,
        )
    elif stage_name == "combine":
        version_dir = _compute_masks_version_dir(kwargs["datasets_root"], kwargs.get("masks"))
        return await stages.start_combine(
            embeddings_source=str(version_dir / REGION_EMBEDDINGS_SUBDIR),
            output_path=str(version_dir / EMBEDDINGS_SUBDIR),
            log=log,
        )
    elif stage_name == "snapshots":
        version_dir = _compute_masks_version_dir(kwargs["datasets_root"], kwargs.get("masks"))
        return await stages.start_snapshots(
            output_dir=str(version_dir / SNAPSHOTS_SUBDIR),
            morphologist_dir=kwargs.get("input_dir"),
            embeddings_dir=str(version_dir / EMBEDDINGS_SUBDIR),
            reference_data_dir=kwargs.get("reference_data_dir"),
            log=log,
        )
    else:
        raise ToolError(f"Unknown stage: {stage_name!r}")
