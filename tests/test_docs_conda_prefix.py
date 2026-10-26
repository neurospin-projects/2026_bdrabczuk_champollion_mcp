"""The docs state that stage subprocesses get CONDA_PREFIX set to the pipeline pixi environment.

REQ-MCPENV-BDRABCZUK-615C1E2DA408: README.md states that `_build_env()` sets CONDA_PREFIX to the
pipeline pixi environment directory in stage subprocess environments.
REQ-MCPENV-BDRABCZUK-4773331BC77C: skills/debug/references/error-patterns.md states that the MCP
server sets CONDA_PREFIX to the pipeline pixi environment directory in stage subprocess environments.

Each statement is looked for inside one chunk of text (a paragraph, a table row or a list item),
as in tests/test_contributing_doc.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_contributing_doc import _chunks

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
ERROR_PATTERNS = REPO_ROOT / "skills" / "debug" / "references" / "error-patterns.md"


def _chunk_matching(path: Path, *patterns: str) -> str | None:
    """First chunk of the file matching every pattern (case-insensitive), else None."""
    for chunk in _chunks(path.read_text(encoding="utf-8")):
        if all(re.search(p, chunk, re.IGNORECASE) for p in patterns):
            return chunk
    return None


def test_readme_build_env_names_conda_prefix():
    """REQ-MCPENV-BDRABCZUK-615C1E2DA408."""
    assert _chunk_matching(README, r"_build_env", r"\bCONDA_PREFIX\b", r"pixi"), (
        "README.md has no paragraph, row or item naming _build_env, CONDA_PREFIX and the pixi env"
    )


def test_error_patterns_names_conda_prefix():
    """REQ-MCPENV-BDRABCZUK-4773331BC77C."""
    assert _chunk_matching(ERROR_PATTERNS, r"MCP server", r"\bCONDA_PREFIX\b", r"pixi"), (
        "error-patterns.md has no paragraph, row or item stating the MCP server sets CONDA_PREFIX"
    )
