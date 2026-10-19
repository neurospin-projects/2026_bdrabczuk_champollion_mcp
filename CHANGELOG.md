# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-19

First public release of the Champollion sulcal MCP server.

### Added

- 16 MCP tools that drive the [Champollion pipeline](https://github.com/neurospin/champollion_pipeline): one
  launcher per stage (Morphologist, cortical tiles, config, training, embeddings, combine, snapshots), a
  composite `start_pipeline`, subject clean-up tools, job lifecycle tools and preflight utilities.
- Background jobs with restart-safe, atomic job files and per-job logs with fold progress.
- Versioned embeddings output layout: embeddings are written under a mask-version folder
  (`champollion_V1/<masks>/`), so runs with different masks never overwrite each other.
- Stage-4 embedding options exposed on `start_embeddings` and `start_pipeline`.
- Ready-to-paste configurations so the MCP tools can be used from any coding agent: Claude Code, OpenCode,
  Mistral Vibe, Codex CLI and any client reading the common `mcpServers` format.
- A Claude Code subagent and skills (run, monitor, debug) that other agents can read as plain instructions.
- [CONTRIBUTING.md](https://github.com/neurospin-projects/2026_bdrabczuk_champollion_mcp/blob/main/CONTRIBUTING.md)
  as the contributor guide, and an AGENTS.md guide for coding agents.

### Changed

- `CHAMPOLLION_PIPELINE_DIR` is now required to locate the pipeline checkout.

### Removed

- The `start_streaming` tool was removed; use `start_pipeline` or the individual stage tools.

### Security

- `HF_TOKEN` is passed only to the embeddings stage and never written to job files or logs.

### License

- Released under the CeCILL-B free software licence
  ([LICENSE](https://github.com/neurospin-projects/2026_bdrabczuk_champollion_mcp/blob/main/LICENSE)).
