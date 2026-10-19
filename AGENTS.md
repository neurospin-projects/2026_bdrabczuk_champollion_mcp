# AGENTS.md

Guidance for coding agents working in this repository. Humans should start with [README.md](README.md) and
[CONTRIBUTING.md](CONTRIBUTING.md).

## What this repository is

`champollion_sulcal_mcp` is a FastMCP server using the stdio transport. It registers 16 tools that drive the
Champollion sulcal embedding pipeline:

- stage launchers: `start_morphologist`, `start_cortical_tiles`, `start_config`, `start_training`,
  `start_embeddings`, `start_combine`, `start_snapshots`;
- the composite `start_pipeline`, which chains the stages as one umbrella job;
- maintenance: `purge_subject`, `prune_failed_subjects`;
- job lifecycle: `get_job_status`, `list_jobs`, `cancel_job`, `get_job_log`;
- utilities: `get_pipeline_info`, `preflight_check`.

Each launcher runs a champollion_pipeline stage script as a background subprocess and returns a job id at once.
Job state is persisted atomically to `{output_dir}/.mcp_jobs/{job_id}.json` and the subprocess output to
`{output_dir}/.mcp_jobs/{job_id}.log`, so jobs survive a server restart.

## Environment

- `CHAMPOLLION_PIPELINE_DIR` is required: it is the absolute path of the champollion_pipeline checkout whose
  scripts the server runs. The server refuses to start a stage without it.
- The champollion_pipeline repository is a separate project: it is never edited from this repository. Change
  only this server's code, tests and docs here.
- Use pixi for everything; never install packages with bare pip or conda.

## Commands

```bash
pixi install          # set up the environment
pixi run format       # ruff format
pixi run lint         # ruff check
pixi run lint-fix     # ruff check --fix
pixi run test         # full test suite
pixi run test-cov     # tests with coverage (85% minimum)
pixi run run          # start the stdio server
```

Run `pixi run format`, `pixi run lint` and `pixi run test` before proposing any change.

## Branches and commits

- Create one task branch from `dev` for each change, and merge it back into `dev` when its tests pass.
- `dev` is merged into `main` as a separate, deliberate release step; `main` holds releases only.
- Task work is never committed on `main`.
- Write commit messages in the imperative mood ("Add mask version option", not "Added ...").
- Keep one logical change per commit: refactoring, documentation and features are separate commits.

## Security

- `HF_TOKEN` is passed to the subprocess environment only by `start_embeddings`; every other tool strips it.
- `HF_TOKEN` is never written to job JSON files or job log files, and must never appear in code, tests or docs.
- Errors are masked (`mask_error_details=True`) so clients do not see internal paths or secrets.
- All paths passed to tools must be absolute; when the client declares MCP roots, paths must stay inside them.

## More

- [README.md](README.md): tools reference, configuration and ready-to-paste coding agent configurations.
- [CONTRIBUTING.md](CONTRIBUTING.md): development setup, checks, branch model and issue reporting.
