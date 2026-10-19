# Contributing to champollion_sulcal_mcp

Thank you for your interest in improving this MCP server. This guide explains how to set up a development
environment, how to check your changes, and how to propose them.

## Development setup

1. Install [pixi](https://pixi.sh).
2. Clone the repository and install the environment:

   ```bash
   git clone https://github.com/neurospin-projects/2026_bdrabczuk_champollion_mcp.git
   cd 2026_bdrabczuk_champollion_mcp
   pixi install
   ```

3. Get a checkout of [`champollion_pipeline`](https://github.com/neurospin/champollion_pipeline). The
   `CHAMPOLLION_PIPELINE_DIR` environment variable is required: set it to the absolute path of that checkout.
   Tests that need a real pipeline checkout are skipped when it is unset.

   ```bash
   export CHAMPOLLION_PIPELINE_DIR=/absolute/path/to/champollion_pipeline
   ```

## Checking your changes

Run these before opening a pull request:

```bash
pixi run format     # format the code with ruff
pixi run lint       # lint with ruff
pixi run lint-fix   # apply ruff's automatic fixes
pixi run test       # run the test suite
```

`pixi run test-cov` reports coverage; the suite enforces a minimum coverage of 85%. Add or update tests with
every behaviour change.

## Branches and pull requests

- `main` holds releases only; each release is tagged (`v0.1.0`, ...).
- `dev` is the integration branch.
- Create feature branches from `dev`, one branch per change.
- Pull requests target `dev`. Maintainers merge `dev` into `main` when cutting a release.

## Commit messages

- Write the commit subject in the imperative mood: "Add mask version option", not "Added mask version option".
- Keep one logical change per commit: refactoring, documentation and features go in separate commits.
- Explain the why in the body when it is not obvious from the diff.

## Reporting issues

Report bugs and request features on the
[GitHub issues page](https://github.com/neurospin-projects/2026_bdrabczuk_champollion_mcp/issues). Include
the tool you called, its arguments, the job log (`get_job_log`) and what you expected. Never paste access
tokens or personal data.

## License

This project is distributed under the CeCILL-B free software license (see [LICENSE](LICENSE)). By submitting
a contribution, you agree that your contribution is licensed under CeCILL-B as well.
