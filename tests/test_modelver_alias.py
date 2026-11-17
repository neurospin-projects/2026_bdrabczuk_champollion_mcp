"""masks_version renamed model_version on start_embeddings / start_pipeline (TASK-MODELVER-BDRABCZUK-3C642461FB75).

champollion_pipeline WIP (a1cefdf) generate_embeddings.py takes ``--model-version`` (the Hugging Face
model-set subfolder); ``--masks-version`` is its deprecated alias (a FutureWarning) and ``--masks``
defaults to the ``--model-version`` value (models/version_options.resolve_masks_default). So:

- the MCP ``model_version`` parameter is forwarded as ``--model-version``; ``masks_version`` stays a
  deprecated alias that logs a warning, is forwarded as ``--model-version`` (never ``--masks-version``)
  and is refused when it differs from ``model_version``;
- start_pipeline's versioned ``champollion_V1/<V>/`` directories of stages 5 and 6 fall back to the
  model version when ``masks`` is unset, as stage 4 now writes there.

runner.launch is replaced by the ``launches`` recorder, so no pipeline script runs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp.tools import pipeline, stages
from tests.stage_files import seed_stage_files, stage_argv_via_pipeline

pytestmark = pytest.mark.unit

ALL_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]
MODEL_VERSION = "canonical_corrected_26_1"
OTHER_VERSION = "canonical_25"


def _embeddings_kwargs(base: Path) -> dict:
    return {"models_path": str(base / "models"), "datasets_root": str(base / "ds")}


def _pipeline_kwargs(base: Path, only: str) -> dict:
    return {
        "input_dir": str(base / "subjects"),
        "output_dir": str(base / "pipe_out"),
        "path_to_graph": "t1mri/default_acquisition/default_analysis/folds/3.1",
        "path_sk_with_hull": "t1mri/default_acquisition/default_analysis/segmentation",
        "crop_path": str(base / "crops"),
        "dataset": "DEMO01",
        "models_path": str(base / "models"),
        "datasets_root": str(base / "datasets" / "DEMO01"),
        "skip_stages": [s for s in ALL_STAGES if s != only],
    }


def _contains_run(argv: list[str], tokens: list[str]) -> bool:
    """True when `tokens` appear in `argv` as one contiguous run."""
    n = len(tokens)
    return any(argv[i : i + n] == tokens for i in range(len(argv) - n + 1))


def _argv_value(argv: list[str], flag: str) -> str:
    assert argv.count(flag) == 1, f"expected exactly one {flag} in argv: {argv}"
    return argv[argv.index(flag) + 1]


def _positional_after(argv: list[str], script: str) -> str:
    scripts = [i for i, a in enumerate(argv) if a.endswith(script)]
    assert len(scripts) == 1, f"expected exactly one {script} in argv: {argv}"
    return argv[scripts[0] + 1]


def _deprecation_records(caplog) -> list[logging.LogRecord]:
    """WARNING-or-higher records naming masks_version as a deprecated alias of model_version."""
    return [
        r
        for r in caplog.records
        if r.levelno >= logging.WARNING
        and "masks_version" in r.getMessage()
        and "model_version" in r.getMessage()
        and "deprecat" in r.getMessage().lower()
    ]


def _version_dir(datasets_root: Path, version: str) -> Path:
    return datasets_root / "derivatives" / "champollion_V1" / version


# --- start_embeddings: the masks_version alias ---


async def test_embeddings_masks_version_forwarded_as_model_version(pipeline_dir, tmp_path, launches):
    """REQ-MODELVER-BDRABCZUK-9208E2907923: masks_version alone puts `--model-version <value>` in the argv."""
    await stages.start_embeddings(**_embeddings_kwargs(tmp_path), masks_version=MODEL_VERSION)
    argv = launches[0]["argv"]
    assert _contains_run(argv, ["--model-version", MODEL_VERSION]), argv


async def test_embeddings_masks_version_omits_masks_version_flag(pipeline_dir, tmp_path, launches):
    """REQ-MODELVER-BDRABCZUK-11993F7756A6: masks_version never reaches the argv as `--masks-version`."""
    await stages.start_embeddings(**_embeddings_kwargs(tmp_path), masks_version=MODEL_VERSION)
    argv = launches[0]["argv"]
    assert "--masks-version" not in argv, argv


async def test_embeddings_masks_version_logs_deprecation_warning(pipeline_dir, tmp_path, launches, caplog):
    """REQ-MODELVER-BDRABCZUK-59E6ED8506C4: masks_version logs a WARNING naming it a deprecated model_version alias."""
    with caplog.at_level(logging.WARNING):
        await stages.start_embeddings(**_embeddings_kwargs(tmp_path), masks_version=MODEL_VERSION)
    assert _deprecation_records(caplog), [r.getMessage() for r in caplog.records]


async def test_embeddings_conflicting_versions_raise_without_launch(pipeline_dir, tmp_path, launches):
    """REQ-MODELVER-BDRABCZUK-98D765E64A9E: differing masks_version and model_version raise a ToolError; no launch."""
    with pytest.raises(ToolError):
        await stages.start_embeddings(
            **_embeddings_kwargs(tmp_path), masks_version=OTHER_VERSION, model_version=MODEL_VERSION
        )
    assert launches == []


# --- start_pipeline: the masks_version alias ---


@pytest.mark.usefixtures("in_process_driver")
async def test_pipeline_masks_version_forwarded_as_model_version(pipeline_dir, tmp_path, launches, monkeypatch):
    """REQ-MODELVER-BDRABCZUK-B84C0B22B0BE: start_pipeline(masks_version=V) gives stage 4 `--model-version V`."""
    kwargs = _pipeline_kwargs(tmp_path, "embeddings")
    seed_stage_files({**kwargs, "masks": MODEL_VERSION})
    argv = await stage_argv_via_pipeline("embeddings", kwargs, launches, monkeypatch, masks_version=MODEL_VERSION)
    assert _contains_run(argv, ["--model-version", MODEL_VERSION]), argv


async def test_pipeline_masks_version_logs_deprecation_warning(pipeline_dir, tmp_path, launches, caplog):
    """REQ-MODELVER-BDRABCZUK-330DB5551345: start_pipeline(masks_version=...) logs the deprecation WARNING."""
    with caplog.at_level(logging.WARNING):
        await pipeline.start_pipeline(**_pipeline_kwargs(tmp_path, "embeddings"), masks_version=MODEL_VERSION)
    assert _deprecation_records(caplog), [r.getMessage() for r in caplog.records]


async def test_pipeline_conflicting_versions_raise_without_launch(pipeline_dir, tmp_path, launches):
    """REQ-MODELVER-BDRABCZUK-9705BF79F7BC: differing masks_version and model_version: ToolError, no driver launch."""
    with pytest.raises(ToolError):
        await pipeline.start_pipeline(
            **_pipeline_kwargs(tmp_path, "embeddings"), masks_version=OTHER_VERSION, model_version=MODEL_VERSION
        )
    assert launches == []


@pytest.mark.usefixtures("in_process_driver")
async def test_pipeline_masks_version_names_combine_source(pipeline_dir, tmp_path, launches, monkeypatch):
    """REQ-MODELVER-BDRABCZUK-D7A899A974D2: masks_version alone names the combine embeddings_source version dir."""
    kwargs = _pipeline_kwargs(tmp_path, "combine")
    seed_stage_files({**kwargs, "masks": MODEL_VERSION})
    argv = await stage_argv_via_pipeline("combine", kwargs, launches, monkeypatch, masks_version=MODEL_VERSION)
    expected = _version_dir(Path(kwargs["datasets_root"]), MODEL_VERSION) / "region_embeddings"
    assert _positional_after(argv, "put_together_embeddings.py") == str(expected)


# --- start_pipeline stages 5 and 6: the version dir falls back to model_version ---


async def _launch(stage: str, tmp_path: Path, **kwargs) -> Path:
    datasets_root = tmp_path / "DEMO01"
    await pipeline._launch_stage(
        stage,
        umbrella_output_dir=str(tmp_path / "run_output"),
        datasets_root=str(datasets_root),
        input_dir=str(datasets_root / "derivatives" / "morphologist-6.0" / "subjects"),
        **kwargs,
    )
    return datasets_root


async def test_combine_source_follows_model_version_when_masks_unset(fake_pipeline_dir, tmp_path, recording_runner):
    """REQ-MCPEMBVER-BDRABCZUK-C24A1D9DDA82: masks unset, model_version M: source is .../M/region_embeddings."""
    datasets_root = await _launch("combine", tmp_path, masks=None, model_version=MODEL_VERSION)
    expected = _version_dir(datasets_root, MODEL_VERSION) / "region_embeddings"
    assert _positional_after(recording_runner[0]["argv"], "put_together_embeddings.py") == str(expected)


async def test_combine_source_prefers_masks_over_model_version(fake_pipeline_dir, tmp_path, recording_runner):
    """REQ-MCPEMBVER-BDRABCZUK-C24A1D9DDA82: masks set, model_version different: V is masks."""
    datasets_root = await _launch("combine", tmp_path, masks=OTHER_VERSION, model_version=MODEL_VERSION)
    expected = _version_dir(datasets_root, OTHER_VERSION) / "region_embeddings"
    assert _positional_after(recording_runner[0]["argv"], "put_together_embeddings.py") == str(expected)


async def test_combine_output_follows_model_version_when_masks_unset(fake_pipeline_dir, tmp_path, recording_runner):
    """REQ-MCPEMBVER-BDRABCZUK-A190045778B4: masks unset, model_version M: --output_path is .../M/embeddings."""
    datasets_root = await _launch("combine", tmp_path, masks=None, model_version=MODEL_VERSION)
    expected = _version_dir(datasets_root, MODEL_VERSION) / "embeddings"
    assert _argv_value(recording_runner[0]["argv"], "--output_path") == str(expected)


async def test_snapshots_output_dir_follows_model_version_when_masks_unset(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """REQ-MCPEMBVER-BDRABCZUK-302C63F966E8: masks unset, model_version M: --output_dir is .../M/snapshots."""
    datasets_root = await _launch("snapshots", tmp_path, masks=None, model_version=MODEL_VERSION)
    expected = _version_dir(datasets_root, MODEL_VERSION) / "snapshots"
    assert _argv_value(recording_runner[0]["argv"], "--output_dir") == str(expected)


async def test_snapshots_embeddings_dir_follows_model_version_when_masks_unset(
    fake_pipeline_dir, tmp_path, recording_runner
):
    """REQ-MCPEMBVER-BDRABCZUK-439C4F48FF43: masks unset, model_version M: --embeddings_dir is .../M/embeddings."""
    datasets_root = await _launch("snapshots", tmp_path, masks=None, model_version=MODEL_VERSION)
    expected = _version_dir(datasets_root, MODEL_VERSION) / "embeddings"
    assert _argv_value(recording_runner[0]["argv"], "--embeddings_dir") == str(expected)
