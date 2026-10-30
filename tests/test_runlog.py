"""Run logs for every launched pipeline script (TASK-RUNLOG-BDRABCZUK-930932521C06).

- REQ-RUNLOG-BDRABCZUK-F8C9BBAE2000: with log true or omitted, each stage tool ends the
  launched script's argv with exactly one bare ``--log``. champollion_pipeline (main
  f7e1c6e, champollion_utils v0.3.0 ScriptBuilder) declares ``--log`` with ``nargs="?"``,
  so a bare ``--log`` must come last: before a positional it would take it as its PATH.
- REQ-RUNLOG-BDRABCZUK-6D9B26684958: with log false, no ``--log`` is added.
- REQ-RUNLOG-BDRABCZUK-3DDA6C8E63A3: start_pipeline hands its log value (true when
  omitted) to the stage tool of each stage it launches.

The MCP's own ``.mcp_jobs`` job logs are unchanged. runner.launch is replaced by the
``launches`` recorder, so no pipeline script runs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from champollion_sulcal_mcp.tools import pipeline, stages

pytestmark = pytest.mark.unit

_GRAPH = "t1mri/default_acquisition/default_analysis/folds/3.1"
_SK_HULL = "t1mri/default_acquisition/default_analysis/segmentation"

STAGE_TOOLS = [
    "start_morphologist",
    "start_cortical_tiles",
    "start_config",
    "start_embeddings",
    "start_combine",
    "start_snapshots",
    "start_training",
    "purge_subject",
    "prune_failed_subjects",
]

PIPELINE_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]


def _tool_kwargs(tool: str, base: Path) -> dict:
    """Minimal valid arguments for `tool`, every path absolute and under `base`."""
    return {
        "start_morphologist": {"input_dir": str(base / "in"), "output_dir": str(base / "out")},
        "start_cortical_tiles": {
            "input_dir": str(base / "in"),
            "output_dir": str(base / "out"),
            "path_to_graph": _GRAPH,
            "path_sk_with_hull": _SK_HULL,
        },
        "start_config": {"crop_path": str(base / "crops"), "dataset": "ds"},
        "start_embeddings": {"models_path": str(base / "models"), "datasets_root": str(base / "ds")},
        "start_combine": {"embeddings_source": str(base / "emb"), "output_path": str(base / "combined")},
        "start_snapshots": {"output_dir": str(base / "snap")},
        "start_training": {"dataset": "ds", "region": "S.C.-sylv.", "output_dir": str(base / "models" / "r")},
        "purge_subject": {"derivatives": str(base / "deriv"), "subject": "sub-01"},
        "prune_failed_subjects": {"output": str(base / "deriv"), "qc": str(base / "qc.tsv")},
    }[tool]


def _log_tokens(argv: list[str]) -> list[str]:
    """Every argv token that is --log or --log=<value>."""
    return [a for a in argv if a == "--log" or a.startswith("--log=")]


# --- REQ-RUNLOG-BDRABCZUK-F8C9BBAE2000: bare --log last, by default and with log=True ---


@pytest.mark.parametrize("tool", STAGE_TOOLS)
async def test_bare_log_ends_argv_by_default(pipeline_dir, tmp_path, launches, tool):
    """REQ-RUNLOG-BDRABCZUK-F8C9BBAE2000: with log omitted, argv ends with exactly one bare --log."""
    await getattr(stages, tool)(**_tool_kwargs(tool, tmp_path))

    argv = launches[0]["argv"]
    assert argv[-1] == "--log", argv
    assert _log_tokens(argv) == ["--log"], argv


@pytest.mark.parametrize("tool", STAGE_TOOLS)
async def test_log_true_ends_argv_with_one_bare_log(pipeline_dir, tmp_path, launches, tool):
    """REQ-RUNLOG-BDRABCZUK-F8C9BBAE2000: with log=True, argv ends with exactly one bare --log."""
    await getattr(stages, tool)(**_tool_kwargs(tool, tmp_path), log=True)

    argv = launches[0]["argv"]
    assert argv[-1] == "--log", argv
    assert _log_tokens(argv) == ["--log"], argv


@pytest.mark.parametrize(
    ("tool", "extra"),
    [
        ("start_cortical_tiles", {"regions": ["S.C.-sylv.", "S.T.s."], "overwrite": True}),
        ("start_embeddings", {"regions": ["SC-sylv_left"], "checkpoint_epoch": 80, "legacy": True}),
        ("start_snapshots", {"subject": "sub-01", "umap_only": True}),
    ],
)
async def test_bare_log_follows_optional_flags(pipeline_dir, tmp_path, launches, tool, extra):
    """REQ-RUNLOG-BDRABCZUK-F8C9BBAE2000: --log stays last after optional and multi-value flags."""
    await getattr(stages, tool)(**_tool_kwargs(tool, tmp_path), **extra)

    argv = launches[0]["argv"]
    assert argv[-1] == "--log", argv
    assert _log_tokens(argv) == ["--log"], argv


# --- REQ-RUNLOG-BDRABCZUK-6D9B26684958: log=False adds no --log ---


@pytest.mark.parametrize("tool", STAGE_TOOLS)
async def test_log_false_omits_log(pipeline_dir, tmp_path, launches, tool):
    """REQ-RUNLOG-BDRABCZUK-6D9B26684958: with log=False, argv holds no --log token."""
    await getattr(stages, tool)(**_tool_kwargs(tool, tmp_path), log=False)

    assert len(launches) == 1
    assert _log_tokens(launches[0]["argv"]) == [], launches[0]["argv"]


# --- REQ-RUNLOG-BDRABCZUK-3DDA6C8E63A3: start_pipeline forwards log to each stage tool ---


@pytest.fixture
def recording_stages(monkeypatch):
    """Replace every stage launcher with a recorder of its keyword arguments."""
    calls: dict[str, dict] = {}

    def _recorder(name):
        async def _fn(**kwargs):
            calls[name] = kwargs
            return {"stage": name, "status": "running", "output_dir": kwargs.get("output_dir", "")}

        return _fn

    for stage in PIPELINE_STAGES:
        monkeypatch.setattr(pipeline.stages, f"start_{stage}", _recorder(f"start_{stage}"))
    return calls


def _pipeline_kwargs(base: Path) -> dict:
    return {
        "input_dir": str(base / "subjects"),
        "output_dir": str(base),
        "path_to_graph": _GRAPH,
        "path_sk_with_hull": _SK_HULL,
        "crop_path": str(base / "crops"),
        "dataset": "DEMO01",
        "models_path": str(base / "models"),
        "datasets_root": str(base / "datasets"),
    }


@pytest.mark.parametrize(("supplied", "expected"), [({}, True), ({"log": True}, True), ({"log": False}, False)])
@pytest.mark.usefixtures("in_process_driver")
async def test_start_pipeline_hands_log_to_pipeline_run(monkeypatch, tmp_path, supplied, expected):
    """REQ-RUNLOG-BDRABCZUK-3DDA6C8E63A3: start_pipeline hands its log value, true when
    omitted, to the background pipeline run, whose kwargs reach _launch_stage."""
    captured: dict = {}

    async def _fake_run_pipeline(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(pipeline, "_run_pipeline", _fake_run_pipeline)

    await pipeline.start_pipeline(**_pipeline_kwargs(tmp_path), **supplied)
    await asyncio.sleep(0)

    assert "log" in captured, sorted(captured)
    assert captured["log"] is expected


@pytest.mark.parametrize("value", [True, False])
@pytest.mark.parametrize("stage", PIPELINE_STAGES)
async def test_launch_stage_forwards_log(recording_stages, tmp_path, stage, value):
    """REQ-RUNLOG-BDRABCZUK-3DDA6C8E63A3: _launch_stage passes the pipeline's log value
    as the log argument of the stage tool it calls."""
    kwargs = _pipeline_kwargs(tmp_path)
    kwargs.pop("output_dir")

    await pipeline._launch_stage(stage, umbrella_output_dir=str(tmp_path), log=value, **kwargs)

    tool_kwargs = recording_stages[f"start_{stage}"]
    assert "log" in tool_kwargs, sorted(tool_kwargs)
    assert tool_kwargs["log"] is value
