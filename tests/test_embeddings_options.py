"""Stage-4 options on start_embeddings and start_pipeline (TASK-050, REQ-MCP-EMBOPT-01..03, -05..07).

REQ-MCP-EMBOPT-04 (use_last_checkpoint) is superseded by REQ-MCPEMBOPT-BDRABCZUK-53C7DE141356
and REQ-MCPEMBOPT-BDRABCZUK-09CD41362839: champollion_pipeline removed --use_last_checkpoint and
evaluates the epoch given by --checkpoint-epoch (TASK-EPOCH80, REQ-MCPEMBOPT-BDRABCZUK-3F8CD6EEBA46,
-3221B3A11FE2, -E29F84095D7A).

runner.launch is replaced by the `launches` recorder, so no pipeline script runs.
The parser check (REQ-MCP-EMBOPT-05) parses the recorded argv with the real
champollion_pipeline GenerateEmbeddings parser, in that checkout's own pixi
interpreter (generate_embeddings.py imports torch at module level).
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp import job_store
from champollion_sulcal_mcp.server import mcp
from champollion_sulcal_mcp.tools import pipeline, stages

ALL_STAGES = ["morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"]

# Real checkout named by CHAMPOLLION_PIPELINE_DIR, read at import time (the pipeline_dir
# fixture later points the variable at a fake one); None when the variable is unset.
_ENV_PIPELINE_DIR = os.environ.get("CHAMPOLLION_PIPELINE_DIR")
_REAL_PIPELINE_DIR = Path(_ENV_PIPELINE_DIR) if _ENV_PIPELINE_DIR else None

_real_sleep = asyncio.sleep


def _embeddings_kwargs(base: Path) -> dict:
    return {"models_path": str(base / "models"), "datasets_root": str(base / "ds")}


def _contains_run(argv: list[str], tokens: list[str]) -> bool:
    """True when `tokens` appear in `argv` as one contiguous run."""
    n = len(tokens)
    return any(argv[i : i + n] == tokens for i in range(len(argv) - n + 1))


# --- REQ-MCP-EMBOPT-01..03: new boolean flags on start_embeddings ---


@pytest.mark.unit
class TestEmbeddingsNewFlags:
    @pytest.mark.parametrize(
        ("name", "flag"),
        [
            ("profiling", "--profiling"),  # REQ-MCP-EMBOPT-01
            ("no_cache", "--no-cache"),  # REQ-MCP-EMBOPT-02
            ("legacy", "--legacy"),  # REQ-MCP-EMBOPT-03
        ],
        ids=["profiling", "no_cache", "legacy"],
    )
    async def test_appends_flag(self, pipeline_dir, tmp_path, launches, name, flag):
        """REQ-MCP-EMBOPT-01..03: `<name>=True` appends its generate_embeddings.py flag."""
        await stages.start_embeddings(**_embeddings_kwargs(tmp_path), **{name: True})
        assert flag in launches[0]["argv"][2:], launches[0]["argv"]


# --- TASK-EPOCH80: use_last_checkpoint removed (the pipeline no longer accepts the flag) ---


@pytest.mark.unit
class TestEmbeddingsNoLastCheckpoint:
    async def test_schema_omits_use_last_checkpoint(self):
        """REQ-MCPEMBOPT-BDRABCZUK-53C7DE141356: the tool input schema has no use_last_checkpoint property."""
        tool = await mcp.get_tool("start_embeddings")
        assert "use_last_checkpoint" not in tool.parameters["properties"]

    async def test_call_with_use_last_checkpoint_errors_without_launch(self, pipeline_dir, tmp_path, launches):
        """REQ-MCPEMBOPT-BDRABCZUK-09CD41362839: use_last_checkpoint is a validation error; nothing launches."""
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="use_last_checkpoint"):
                await client.call_tool(
                    "start_embeddings", {**_embeddings_kwargs(tmp_path), "use_last_checkpoint": True}
                )
        assert launches == []


# --- TASK-EPOCH80: checkpoint_epoch forwarded as --checkpoint-epoch ---


@pytest.mark.unit
class TestEmbeddingsCheckpointEpoch:
    @pytest.mark.parametrize("epoch", [0, 80, 12], ids=["epoch_0", "epoch_80", "epoch_12"])
    async def test_forwards_checkpoint_epoch(self, pipeline_dir, tmp_path, launches, epoch):
        """REQ-MCPEMBOPT-BDRABCZUK-3F8CD6EEBA46: checkpoint_epoch=N puts `--checkpoint-epoch N` in the argv."""
        async with Client(mcp) as client:
            await client.call_tool("start_embeddings", {**_embeddings_kwargs(tmp_path), "checkpoint_epoch": epoch})
        assert len(launches) == 1
        assert _contains_run(launches[0]["argv"][2:], ["--checkpoint-epoch", str(epoch)]), launches[0]["argv"]

    @pytest.mark.parametrize(
        "extra",
        [{}, {"checkpoint_epoch": None}],
        ids=["omitted", "null"],
    )
    async def test_no_checkpoint_epoch_omits_flag(self, pipeline_dir, tmp_path, launches, extra):
        """REQ-MCPEMBOPT-BDRABCZUK-3221B3A11FE2: no checkpoint_epoch value leaves --checkpoint-epoch out."""
        async with Client(mcp) as client:
            await client.call_tool("start_embeddings", {**_embeddings_kwargs(tmp_path), **extra})
        assert len(launches) == 1
        assert "--checkpoint-epoch" not in launches[0]["argv"], launches[0]["argv"]

    async def test_negative_checkpoint_epoch_errors_without_launch(self, pipeline_dir, tmp_path, launches):
        """REQ-MCPEMBOPT-BDRABCZUK-E29F84095D7A: a negative checkpoint_epoch is a validation error; nothing launches.

        The error must reject the value, not the argument name (an unknown argument also fails validation).
        """
        async with Client(mcp) as client:
            with pytest.raises(ToolError, match="checkpoint_epoch") as excinfo:
                await client.call_tool("start_embeddings", {**_embeddings_kwargs(tmp_path), "checkpoint_epoch": -1})
        assert "unexpected_keyword_argument" not in str(excinfo.value), str(excinfo.value)
        assert launches == []


# --- REQ-MCP-EMBOPT-05: argv accepted by the real pipeline parser ---


_ALL_OPTIONS: dict = {
    "cpu": True,
    "overwrite": True,
    "masks": "canonical_25",
    "masks_version": "canonical_25",
    "output": "/abs/emb_out",
    "subjects": "/abs/subjects.csv",
    "regions": ["SC-sylv_left", "FIP_right"],
    "run_cka": True,
    "cortical_version": "cortical_tiles-2027",
    "profiling": True,
    "no_cache": True,
    "legacy": True,
    "checkpoint_epoch": 12,
}

_PARSE_SNIPPET = """
import json, sys
from champollion_pipeline.generate_embeddings import GenerateEmbeddings
args = GenerateEmbeddings().parse_args(json.loads(sys.argv[1]))
print("PARSED=" + json.dumps(vars(args)))
"""


@pytest.mark.integration
class TestEmbeddingsArgvMatchesPipelineParser:
    async def test_full_argv_parses_into_supplied_values(self, pipeline_dir, tmp_path, launches):
        """REQ-MCP-EMBOPT-05: argv with each optional parameter parses into the supplied values."""
        if _REAL_PIPELINE_DIR is None:
            pytest.skip("CHAMPOLLION_PIPELINE_DIR is not set")
        real_python = _REAL_PIPELINE_DIR / ".pixi" / "envs" / "default" / "bin" / "python"
        if not real_python.exists():
            pytest.skip(f"champollion_pipeline pixi interpreter not found: {real_python}")
        kwargs = _embeddings_kwargs(tmp_path)
        await stages.start_embeddings(**kwargs, **_ALL_OPTIONS)
        script_args = launches[0]["argv"][2:]

        proc = subprocess.run(
            [str(real_python), "-c", _PARSE_SNIPPET, json.dumps(script_args)],
            cwd=str(_REAL_PIPELINE_DIR),
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, f"parser rejected {script_args}:\n{proc.stderr[-2000:]}"
        line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("PARSED="))
        parsed = json.loads(line[len("PARSED=") :])

        assert parsed["models_path"] == kwargs["models_path"]
        assert parsed["datasets_root"] == kwargs["datasets_root"]
        for name, value in _ALL_OPTIONS.items():
            assert parsed[name] == value, f"{name}: parsed {parsed[name]!r}, supplied {value!r}"


# --- REQ-MCP-EMBOPT-06 / -07: start_pipeline forwards stage-4 options ---


def _pipeline_kwargs(base: Path) -> dict:
    return {
        "input_dir": str(base / "subjects"),
        "output_dir": str(base / "pipe_out"),
        "path_to_graph": "t1mri/default_acquisition/default_analysis/folds/3.1",
        "path_sk_with_hull": "t1mri/default_acquisition/default_analysis/segmentation",
        "crop_path": str(base / "crops"),
        "dataset": "DEMO01",
        "models_path": str(base / "models"),
        "datasets_root": str(base / "datasets" / "DEMO01"),
        "skip_stages": [s for s in ALL_STAGES if s != "embeddings"],
    }


async def _embeddings_argv_via_pipeline(base: Path, launches: list, monkeypatch, **options) -> list[str]:
    """Run start_pipeline with only the embeddings stage active; return that stage's argv.

    The real _launch_stage and start_embeddings run; runner.launch is the recorder.
    Once the stage is launched its child job is marked succeeded so the umbrella ends.
    """

    async def fast_sleep(delay, *args, **kwargs):
        await _real_sleep(0)

    monkeypatch.setattr(pipeline.asyncio, "sleep", fast_sleep)

    kwargs = _pipeline_kwargs(base)
    result = await pipeline.start_pipeline(**kwargs, **options)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + 5.0
    while not launches and loop.time() < deadline:
        await _real_sleep(0.01)
    assert launches, "embeddings stage was never launched"

    child_dir = launches[0]["output_dir"]
    for job_file in (Path(child_dir) / ".mcp_jobs").glob("*.json"):
        await job_store.update_job(child_dir, job_file.stem, status="succeeded")

    while loop.time() < deadline:
        if job_store.read_job(kwargs["output_dir"], result["job_id"]).status in ("succeeded", "failed", "cancelled"):
            break
        await _real_sleep(0.01)
    return launches[0]["argv"]


@pytest.mark.unit
class TestPipelineForwardsEmbeddingsOptions:
    @pytest.mark.parametrize(
        ("name", "flag"),
        [("overwrite", "--overwrite"), ("run_cka", "--run-cka")],
        ids=["overwrite", "run_cka"],
    )
    async def test_forwards_bool_flag(self, pipeline_dir, tmp_path, launches, monkeypatch, name, flag):
        """REQ-MCP-EMBOPT-06: start_pipeline(<name>=True) puts its flag in the embeddings argv."""
        argv = await _embeddings_argv_via_pipeline(tmp_path, launches, monkeypatch, **{name: True})
        assert flag in argv[2:], argv

    @pytest.mark.parametrize(
        ("name", "value", "expected"),
        [
            ("regions", ["SC-sylv_left", "FIP_right"], ["--regions", "SC-sylv_left", "FIP_right"]),
            ("masks", "canonical_25", ["--masks", "canonical_25"]),
            ("masks_version", "canonical_25", ["--masks-version", "canonical_25"]),
            ("cortical_version", "cortical_tiles-2027", ["--cortical_version", "cortical_tiles-2027"]),
        ],
        ids=["regions", "masks", "masks_version", "cortical_version"],
    )
    async def test_forwards_valued_option(self, pipeline_dir, tmp_path, launches, monkeypatch, name, value, expected):
        """REQ-MCP-EMBOPT-07: start_pipeline(<name>=value) puts the option and value in the embeddings argv."""
        argv = await _embeddings_argv_via_pipeline(tmp_path, launches, monkeypatch, **{name: value})
        assert _contains_run(argv, expected), argv
