"""CHANGELOG.md follows Keep a Changelog; its 0.1.0 entry is reused as the GitHub release notes.

REQ-MCPREL-BDRABCZUK-177EACA5EBF1: CHANGELOG.md contains a Markdown link to
https://keepachangelog.com/en/1.1.0/.
REQ-MCPREL-BDRABCZUK-20C9951F6FB4: each level-two heading except '## [Unreleased]' has the form
'## [<version>] - <YYYY-MM-DD>'.
REQ-MCPREL-BDRABCZUK-C545E69F6D54: the first version heading names the version declared in the
[project] table of pyproject.toml.
REQ-MCPREL-BDRABCZUK-CA6D6200B76F: it contains the heading '## [0.1.0] - 2026-10-19'.
REQ-MCPREL-BDRABCZUK-B62F1CFF10AF: the 0.1.0 entry names Claude Code, OpenCode, Mistral Vibe and Codex
CLI as coding agents that can use the MCP tools.
REQ-MCPREL-BDRABCZUK-A9B7E306FD1F: the 0.1.0 entry states that the start_streaming tool was removed.
REQ-MCPREL-BDRABCZUK-171C4EEC3A35: the 0.1.0 entry names the versioned embeddings output layout.
REQ-MCPREL-BDRABCZUK-5C60B7D7E9FA: the 0.1.0 entry names CeCILL-B as the licence.
REQ-MCPREL-BDRABCZUK-2CB750187C8C: the 0.1.0 entry names CONTRIBUTING.md as the added contributor guide.
REQ-MCPREL-BDRABCZUK-47FBDA5FD163: each Markdown link target in the 0.1.0 entry is an absolute https URL
(relative links do not resolve in GitHub release notes).

The 0.1.0 entry runs from its heading to the next level-two heading or the end of the file, so link
reference definitions at the bottom of the file belong to the last entry.
"""

from __future__ import annotations

import datetime
import re
import tomllib
from pathlib import Path

import pytest

from tests.test_contributing_doc import _chunks

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
PYPROJECT = REPO_ROOT / "pyproject.toml"

RELEASE_HEADING = "## [0.1.0] - 2026-10-19"
UNRELEASED_HEADING = "## [Unreleased]"
VERSION_HEADING = re.compile(r"^## \[(?P<version>\d+\.\d+\.\d+[^\]]*)\] - (?P<date>\d{4}-\d{2}-\d{2})$")
KEEP_A_CHANGELOG_LINK = re.compile(r"\]\(\s*https://keepachangelog\.com/en/1\.1\.0/?\s*\)")
CODING_AGENTS = ("Claude Code", "OpenCode", "Mistral Vibe", "Codex CLI")
INLINE_TARGET = re.compile(r"\]\(\s*<?([^)\s>]+)")
REFERENCE_TARGET = re.compile(r"^\s*\[[^\]]+\]:\s*<?(\S+?)>?(?:\s|$)", re.MULTILINE)


def _changelog() -> str:
    assert CHANGELOG.is_file(), f"missing changelog: {CHANGELOG}"
    return CHANGELOG.read_text(encoding="utf-8")


def _level_two_headings() -> list[str]:
    return [line.rstrip() for line in _changelog().splitlines() if line.startswith("## ")]


def _release_entry() -> str:
    """Text of the 0.1.0 entry: its heading up to the next level-two heading or the end of the file."""
    match = re.search(rf"^{re.escape(RELEASE_HEADING)}[ \t]*$.*?(?=^## |\Z)", _changelog(), re.MULTILINE | re.DOTALL)
    assert match, f"CHANGELOG.md has no '{RELEASE_HEADING}' entry"
    return match.group(0)


def _entry_chunk_matching(*patterns: str) -> str | None:
    """First chunk of the 0.1.0 entry matching every pattern (case-insensitive), else None."""
    for chunk in _chunks(_release_entry()):
        if all(re.search(p, chunk, re.IGNORECASE) for p in patterns):
            return chunk
    return None


def test_changelog_links_keep_a_changelog() -> None:
    assert KEEP_A_CHANGELOG_LINK.search(_changelog()), "CHANGELOG.md must link Keep a Changelog 1.1.0"


def test_changelog_version_headings_are_dated() -> None:
    headings = [h for h in _level_two_headings() if h != UNRELEASED_HEADING]
    assert headings, "CHANGELOG.md has no version heading"
    malformed = []
    for heading in headings:
        match = VERSION_HEADING.match(heading)
        try:
            datetime.date.fromisoformat(match.group("date")) if match else None
        except ValueError:
            match = None
        if match is None:
            malformed.append(heading)
    assert malformed == [], f"CHANGELOG.md headings not of the form '## [<version>] - <YYYY-MM-DD>': {malformed}"


def test_changelog_newest_version_matches_pyproject() -> None:
    versions = [m.group("version") for h in _level_two_headings() if (m := VERSION_HEADING.match(h))]
    assert versions, "CHANGELOG.md has no version heading"
    declared = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]["version"]
    assert versions[0] == declared, f"newest CHANGELOG.md version {versions[0]} != pyproject.toml {declared}"


def test_changelog_has_0_1_0_entry() -> None:
    assert RELEASE_HEADING in _level_two_headings(), f"CHANGELOG.md must contain '{RELEASE_HEADING}'"


def test_changelog_0_1_0_names_coding_agents() -> None:
    entry = _release_entry()
    missing = [name for name in CODING_AGENTS if not re.search(rf"\b{re.escape(name)}\b", entry)]
    assert missing == [], f"the 0.1.0 entry does not name the coding agents: {missing}"


def test_changelog_0_1_0_start_streaming_removed() -> None:
    assert _entry_chunk_matching(r"\bstart_streaming\b", r"\bremov"), (
        "the 0.1.0 entry must state that the start_streaming tool was removed"
    )


def test_changelog_0_1_0_versioned_embeddings() -> None:
    assert _entry_chunk_matching(r"\bversioned\b", r"\bembeddings?\b"), (
        "the 0.1.0 entry must name the versioned embeddings output layout"
    )


def test_changelog_0_1_0_cecill_b() -> None:
    assert _entry_chunk_matching(r"\bCeCILL-B\b", r"\blicen[cs]e"), "the 0.1.0 entry must name the CeCILL-B licence"


def test_changelog_0_1_0_contributing() -> None:
    assert _entry_chunk_matching(r"\bCONTRIBUTING\.md\b", r"\bguide\b"), (
        "the 0.1.0 entry must name CONTRIBUTING.md as the contributor guide"
    )


def test_changelog_0_1_0_links_are_absolute() -> None:
    entry = _release_entry()
    targets = INLINE_TARGET.findall(entry) + REFERENCE_TARGET.findall(entry)
    relative = [t for t in targets if not t.startswith("https://")]
    assert relative == [], f"the 0.1.0 entry has non-https link targets: {relative}"
