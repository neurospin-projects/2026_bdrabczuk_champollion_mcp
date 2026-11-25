"""Docs of the start_embeddings profiling removal (TASK-MCPEMBOPT-BDRABCZUK-F275ADAC384B).

REQ-MCPEMBOPT-BDRABCZUK-02B6F35AAC65: the stage-4 CLI-only flags note of agents/champollion-pipeline.md
describes --profiling as deprecated in favour of the main.py --profile DIR option.

REQ-MCPEMBOPT-BDRABCZUK-349CCCDD80F3: the '## [Unreleased]' entry of CHANGELOG.md contains a BREAKING item
stating that start_embeddings no longer accepts the profiling parameter.

The parameter tables and lists themselves are pinned by tests/test_docs_tool_params.py
(REQ-MCPPARAMS-BDRABCZUK-B78F7F6A6C4A, REQ-MCPPARAMS-BDRABCZUK-BE2220F82C7B).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_contributing_doc import _chunks

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT = REPO_ROOT / "agents" / "champollion-pipeline.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

CLI_ONLY_NOTE = re.compile(r"^>\s*\*\*CLI-only flags\b.*$", re.MULTILINE)


def _stage4_cli_only_note() -> str:
    notes = [m.group(0) for m in CLI_ONLY_NOTE.finditer(AGENT.read_text(encoding="utf-8"))]
    stage4 = [n for n in notes if "(stage 4)" in n]
    assert len(stage4) == 1, f"expected one stage-4 CLI-only flags note, found {len(stage4)}: {notes}"
    return stage4[0]


def _unreleased_entry() -> str:
    match = re.search(r"^## \[Unreleased\][ \t]*$.*?(?=^## |\Z)", CHANGELOG.read_text(encoding="utf-8"), re.M | re.S)
    assert match, "CHANGELOG.md has no '## [Unreleased]' entry"
    return match.group(0)


def test_agent_cli_only_note_marks_profiling_deprecated() -> None:
    """REQ-MCPEMBOPT-BDRABCZUK-02B6F35AAC65: --profiling is deprecated in favour of main.py --profile DIR."""
    note = _stage4_cli_only_note()
    assert re.search(r"--profiling`?[^,;]*?\bdeprecated\b", note, re.IGNORECASE), note
    assert "main.py --profile DIR" in note, note


def test_changelog_unreleased_names_profiling_removal() -> None:
    """REQ-MCPEMBOPT-BDRABCZUK-349CCCDD80F3: a BREAKING Unreleased item says start_embeddings drops profiling."""
    patterns = (r"\bBREAKING\b", r"\bstart_embeddings\b", r"\bprofiling\b", r"\bno longer accepts\b|\bremoved\b")
    chunks = [c for c in _chunks(_unreleased_entry()) if all(re.search(p, c) for p in patterns)]
    assert chunks, "no '## [Unreleased]' item marked BREAKING names the removed start_embeddings profiling parameter"
