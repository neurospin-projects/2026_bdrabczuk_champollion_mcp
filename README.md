# champollion-sulcal-mcp

A FastMCP server wrapping the Champollion sulcal embedding pipeline — exposes each pipeline stage as an MCP tool so an agent can run, monitor, and debug the pipeline without shelling out manually.

## Overview

This is a **stdio MCP server** (not a web service). It doesn't run the pipeline in-process — each `start_*` tool launches one of the pipeline's CLI scripts as a subprocess and tracks it as a background job (status + log file on disk), which callers poll until it reaches a terminal state.

Any MCP client can drive it. The repository also ships its own Claude Code subagent definition and skills, so it can be driven directly from Claude Code.

## How it works

1. A caller (an MCP client) invokes a tool such as `start_cortical_tiles`.
2. The tool validates its arguments (paths must be absolute), resolves the pipeline's script location via `preflight.detect()`, and builds an `argv` matching that script's real CLI.
3. `runner.launch()` spawns the script as a subprocess with an environment built by `_build_env()` (selectively passes `HF_TOKEN`, injects `BRAINVISA_SHARE`, prepends BrainVISA/pixi bins to `PATH`), and returns immediately with a `job_id`.
4. The subprocess's combined stdout/stderr is streamed line-by-line into a log file; progress lines matching `fold N/M` update the job's progress.
5. The caller polls `get_job_status(output_dir, job_id)` (and can tail `get_job_log`) until the job is `succeeded`, `failed`, or `cancelled`.

## Pipeline stages

| Stage | Tool | Purpose |
|---|---|---|
| 1 | `start_morphologist` | Generate sulcal graphs from T1 MRI using Morphologist |
| 2 | `start_cortical_tiles` | Extract 28 standardized sulcal region crops |
| 3 | `start_config` | Generate Champollion dataset YAML configuration |
| 4 | `start_embeddings` | Compute 56-fold sulcal embeddings (28 regions × 2 hemispheres) |
| 5 | `start_combine` | Collect per-region embedding CSVs into a single output directory |
| 6 | `start_snapshots` | Render sulcal graph meshes, cortical tile masks, and UMAP plots |
| — | `start_pipeline` | Run stages 1–6 sequentially as one umbrella job (stages skippable) |
| — | `start_training` | Train a self-supervised `champollion_V1` encoder for one region |

## Project layout

```
src/champollion_sulcal_mcp/
├── server.py            # FastMCP instance, tool registration, entry point (main())
├── preflight.py          # locates the champollion_pipeline repo + its scripts/submodules
├── job_store.py           # JobState/JobProgress models, JSON job file persistence
├── runner.py               # subprocess launch, log capture, progress parsing, cancel
└── tools/
    ├── stages.py           # one start_<stage> tool per pipeline stage + maintenance tools
    ├── pipeline.py          # start_pipeline composite/umbrella job orchestration
    ├── jobs.py               # get_job_status, list_jobs, cancel_job, get_job_log
    └── utils.py               # get_pipeline_info, preflight_check
agents/
└── champollion-pipeline.md  # Claude Code subagent definition for this MCP server
skills/
├── run-pipeline/             # SKILL.md guiding pipeline runs and their parameters
├── monitor/                   # SKILL.md for the job-polling loop
└── debug/                      # SKILL.md + known error patterns for failure diagnosis
tests/                           # pytest suite with fake_pipeline_dir / recording_runner fixtures
```

## MCP tools reference

Registered in `server.py`:

**Stage launchers** — `start_morphologist`, `start_cortical_tiles`, `start_config`, `start_training`, `start_embeddings`, `start_combine`, `start_snapshots`

**Composite pipeline** — `start_pipeline`

**Maintenance** — `purge_subject` (remove a subject's cortical_tiles derivatives), `prune_failed_subjects` (remove outputs for subjects that failed QC)

**Job lifecycle** — `get_job_status`, `list_jobs`, `cancel_job`, `get_job_log`

**Utilities** — `get_pipeline_info` (server/stage metadata), `preflight_check` (verify the pipeline is correctly configured and accessible)

## Requirements

- Python 3.11 or 3.12
- [pixi](https://pixi.sh)
- A checkout of the **[`champollion_pipeline`](https://github.com/neurospin/champollion_pipeline)** repository (not included here) — must contain the stage scripts under `src/` (`generate_morphologist_graphs.py`, `run_cortical_tiles.py`, `generate_champollion_config.py`, `generate_embeddings.py`, `put_together_embeddings.py`, `generate_snapshots.py`, `train_champollion.py`) and the `external/champollion_V1` and `external/cortical_tiles` submodules.

## Installation

```bash
pixi install
```

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `CHAMPOLLION_PIPELINE_DIR` | Absolute path to the `champollion_pipeline` repo | Required |
| `HF_TOKEN` | HuggingFace token | Passed through only to the `embeddings` stage; stripped from every other stage's subprocess environment |
| `BRAINVISA` / `BRAINVISA_SHARE` | BrainVISA install location / share dir | Auto-injected from the pipeline's pixi environment if not already set in the environment |

## Running

This is a stdio server meant to be launched by an MCP client or coding agent (see [Coding agents integration](#coding-agents-integration) for ready-to-paste configurations), not run interactively on its own:

```bash
pixi run run                  # python -m champollion_sulcal_mcp.server
# or, once installed:
champollion-sulcal-mcp
```

## Coding agents integration

Any coding agent or MCP client that can launch a local stdio server can use this server. Every configuration
below starts it with `pixi run --manifest-path /path/to/champollion_sulcal_mcp/pixi.toml run`, so it works from
any working directory. Replace the `/path/to/...` placeholders with absolute paths.

- `CHAMPOLLION_PIPELINE_DIR` is required: the absolute path of your `champollion_pipeline` checkout.
- `HF_TOKEN` is optional: set it only to download models from Hugging Face. Prefer an environment reference over
  writing the token in a configuration file, and never commit a token.
- The first start can take a few seconds while pixi resolves the environment, so the snippets raise the client's
  start-up timeout.

### Claude Code

Register it with the CLI (user scope shown; `--scope project` writes a shared `.mcp.json` instead):

```bash
claude mcp add --transport stdio --scope user \
  --env CHAMPOLLION_PIPELINE_DIR=/path/to/champollion_pipeline \
  champollion-sulcal \
  -- pixi run --manifest-path /path/to/champollion_sulcal_mcp/pixi.toml run
```

or add it to `.mcp.json` at the root of your project:

```json
{
  "mcpServers": {
    "champollion-sulcal": {
      "command": "pixi",
      "args": ["run", "--manifest-path", "/path/to/champollion_sulcal_mcp/pixi.toml", "run"],
      "env": {
        "CHAMPOLLION_PIPELINE_DIR": "/path/to/champollion_pipeline",
        "HF_TOKEN": "${HF_TOKEN:-}"
      }
    }
  }
}
```

This repository also ships Claude Code assets:

- **`agents/champollion-pipeline.md`** — a subagent scoped to exactly the MCP tools it needs, with an operational playbook: always preflight first, never guess paths, default output layout, per-stage required parameters, a known-error-pattern table, and the exact CLI invocation each tool wraps.
- **`skills/run-pipeline`** — guides choosing between the full pipeline and individual stages, and gathering the right parameters.
- **`skills/monitor`** — a poll-every-30-seconds monitoring loop with per-job-type progress reporting.
- **`skills/debug`** — systematic failure diagnosis: read the full log, match against known error patterns, report root cause and fix.

They are Markdown files: other coding agents can read them as plain instructions, for example by pointing their
own agent or rules file at them.

### OpenCode

In `opencode.json` (project) or `~/.config/opencode/opencode.json` (global):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "champollion-sulcal": {
      "type": "local",
      "command": ["pixi", "run", "--manifest-path", "/path/to/champollion_sulcal_mcp/pixi.toml", "run"],
      "enabled": true,
      "timeout": 30000,
      "environment": {
        "CHAMPOLLION_PIPELINE_DIR": "/path/to/champollion_pipeline",
        "HF_TOKEN": "{env:HF_TOKEN}"
      }
    }
  }
}
```

### Mistral Vibe

In `./.vibe/config.toml` (project) or `~/.vibe/config.toml` (global). Vibe does not expand environment
variables in this file; add `"HF_TOKEN" = "<your token>"` to `env` only in a private, uncommitted config.

```toml
[[mcp_servers]]
name = "champollion-sulcal"
transport = "stdio"
command = "pixi"
args = ["run", "--manifest-path", "/path/to/champollion_sulcal_mcp/pixi.toml", "run"]
startup_timeout_sec = 60
env = { "CHAMPOLLION_PIPELINE_DIR" = "/path/to/champollion_pipeline" }
```

### Codex

In `~/.codex/config.toml` (global) or `.codex/config.toml` (trusted project); `env_vars` forwards `HF_TOKEN`
from your shell without storing it in the file:

```toml
[mcp_servers.champollion-sulcal]
command = "pixi"
args = ["run", "--manifest-path", "/path/to/champollion_sulcal_mcp/pixi.toml", "run"]
startup_timeout_sec = 60
env_vars = ["HF_TOKEN"]

[mcp_servers.champollion-sulcal.env]
CHAMPOLLION_PIPELINE_DIR = "/path/to/champollion_pipeline"
```

or from the command line:

```bash
codex mcp add champollion-sulcal \
  --env CHAMPOLLION_PIPELINE_DIR=/path/to/champollion_pipeline \
  -- pixi run --manifest-path /path/to/champollion_sulcal_mcp/pixi.toml run
```

### Other MCP clients

Most other clients (Claude Desktop, Cursor, and others) read the common `mcpServers` format, for example
Claude Desktop's `claude_desktop_config.json` or Cursor's `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "champollion-sulcal": {
      "command": "pixi",
      "args": ["run", "--manifest-path", "/path/to/champollion_sulcal_mcp/pixi.toml", "run"],
      "env": {
        "CHAMPOLLION_PIPELINE_DIR": "/path/to/champollion_pipeline"
      }
    }
  }
}
```

If your client has no `env` setting, export `CHAMPOLLION_PIPELINE_DIR` (and optionally `HF_TOKEN`) in the
environment the client is started from.

## Job tracking

Each job is persisted as `<output_dir>/.mcp_jobs/<job_id>.json` (atomic write) with its combined stdout/stderr log at `<output_dir>/.mcp_jobs/<job_id>.log`. Status lifecycle: `pending` → `running` → one of `succeeded` / `failed` / `cancelled`. `start_pipeline` additionally writes an "umbrella" job whose progress tracks `current_stage` / `stages_done` / `stages_total` and the currently active child `job_id`.

## Testing

```bash
pixi run test           # full suite
pixi run test-unit        # unit-marked tests only
pixi run test-fast         # stop on first failure
pixi run test-cov           # with coverage report
```

`tests/conftest.py` provides `fake_pipeline_dir` (a temp dir with stub stage scripts and submodule folders) and `recording_runner` (stubs `runner.launch` to record calls instead of spawning real subprocesses), so most tool behavior can be tested without a real `champollion_pipeline` checkout.

## Linting

```bash
pixi run lint        # ruff check
pixi run lint-fix      # ruff check --fix
pixi run format         # ruff format
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, checks, branch model and how to report issues.

## License

Released under the CeCILL-B free software licence, the same as `champollion_pipeline`. See [LICENSE](LICENSE).
