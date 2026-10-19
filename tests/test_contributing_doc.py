"""CONTRIBUTING.md guides external contributors of the public champollion_sulcal_mcp repository.

REQ-MCPREL-BDRABCZUK-5B7032B2F231: CONTRIBUTING.md at the repository root gives `pixi install` as the
development environment setup step.
REQ-MCPREL-BDRABCZUK-6D90495708E1: it states that CHAMPOLLION_PIPELINE_DIR is required to locate the
champollion_pipeline checkout.
REQ-MCPREL-BDRABCZUK-1BDA533272B7: it contains `pixi run format`, `pixi run lint`, `pixi run lint-fix`
and `pixi run test`.
REQ-MCPREL-BDRABCZUK-BDC4D4ACD2BF: each `pixi run` command it names is a task of pixi.toml's [tasks].
REQ-MCPREL-BDRABCZUK-DDA06EF1D7EE: it tells contributors to create feature branches from dev.
REQ-MCPREL-BDRABCZUK-E581B6E3B039: it states that pull requests target dev.
REQ-MCPREL-BDRABCZUK-749A640B894C: it states that the main branch holds releases.
REQ-MCPREL-BDRABCZUK-21F8AAA5BF48: it asks for commit messages in the imperative mood.
REQ-MCPREL-BDRABCZUK-B5D11406CACD: it asks for one logical change per commit.
REQ-MCPREL-BDRABCZUK-5F2E8E48B892: it states that contributions are licensed under CeCILL-B.
REQ-MCPREL-BDRABCZUK-30548661CE95: it links the LICENSE file at the repository root.
REQ-MCPREL-BDRABCZUK-E3FCDEC76D86: it tells contributors to report issues through a link to the
GitHub issues page of the repository (owner neurospin-projects).
REQ-MCPREL-BDRABCZUK-ED7C12C328DD: it contains none of the words ALM, Claude, Anthropic or
Co-Authored-By, compared case-insensitively.
REQ-MCPREL-BDRABCZUK-17E236EDFDF7: README.md links CONTRIBUTING.md at the repository root.

Statements that pair two facts (a variable and "required", "pull request" and dev, ...) are looked
for inside one chunk of text: a paragraph (lines between blank lines), except that each table row and
each list item is a chunk of its own. The self-containment rule of REQ-MCPSELF-BDRABCZUK-C7125EF3AE31
is also applied to CONTRIBUTING.md here, before the file is tracked by git.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from tests.test_docs_self_contained import _forbidden_in

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRIBUTING = REPO_ROOT / "CONTRIBUTING.md"
README = REPO_ROOT / "README.md"
PIXI_TOML = REPO_ROOT / "pixi.toml"

DEV_COMMANDS = ("format", "lint", "lint-fix", "test")
PIXI_RUN = re.compile(r"\bpixi run ([A-Za-z0-9_-]+)")
ITEM_START = re.compile(r"^\s*(?:\||[-*+] |\d+[.)] )")
DEV = r"`?dev`?(?![\w-])"
INTERNAL_WORDS = re.compile(r"(?<![\w-])(?:alm|claude|anthropic|co-authored-by)(?![\w-])", re.IGNORECASE)


def _contributing() -> str:
    assert CONTRIBUTING.is_file(), f"missing contributor guide: {CONTRIBUTING}"
    return CONTRIBUTING.read_text(encoding="utf-8")


def _chunks(text: str) -> list[str]:
    """Paragraphs of the text; each table row and each list item starts a chunk of its own."""
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in text.splitlines():
        if not line.strip() or ITEM_START.match(line):
            if current:
                chunks.append(current)
            current = [line] if line.strip() else []
        else:
            current.append(line)
    if current:
        chunks.append(current)
    return [" ".join(chunk) for chunk in chunks]


def _chunk_matching(*patterns: str) -> str | None:
    """First chunk of CONTRIBUTING.md matching every pattern (case-insensitive), else None."""
    for chunk in _chunks(_contributing()):
        if all(re.search(p, chunk, re.IGNORECASE) for p in patterns):
            return chunk
    return None


def _has_command(text: str, command: str) -> bool:
    return re.search(rf"\bpixi run {re.escape(command)}(?![\w-])", text) is not None


def test_chunks_split_paragraphs_rows_and_items() -> None:
    """The chunker keeps wrapped prose together and separates table rows and list items."""
    text = "one\ntwo\n\n| a |\n| b |\n- x\n  y\n1. z"
    assert _chunks(text) == ["one two", "| a |", "| b |", "- x   y", "1. z"]


def test_contributing_gives_pixi_install() -> None:
    assert re.search(r"\bpixi install(?![\w-])", _contributing()), "CONTRIBUTING.md must give `pixi install`"


def test_contributing_requires_pipeline_dir() -> None:
    assert _chunk_matching(r"\bCHAMPOLLION_PIPELINE_DIR\b", r"\brequired\b"), (
        "CONTRIBUTING.md must state that CHAMPOLLION_PIPELINE_DIR is required"
    )


def test_contributing_names_dev_commands() -> None:
    text = _contributing()
    missing = [c for c in DEV_COMMANDS if not _has_command(text, c)]
    assert missing == [], f"CONTRIBUTING.md lacks `pixi run` for: {missing}"


def test_contributing_pixi_commands_exist() -> None:
    tasks = set(tomllib.loads(PIXI_TOML.read_text(encoding="utf-8"))["tasks"])
    named = set(PIXI_RUN.findall(_contributing()))
    assert named, "CONTRIBUTING.md names no `pixi run` command"
    assert named <= tasks, f"CONTRIBUTING.md names pixi tasks absent from pixi.toml: {sorted(named - tasks)}"


def test_contributing_branches_from_dev() -> None:
    assert _chunk_matching(r"\bfeature branch", rf"\bfrom\s+(?:the\s+)?{DEV}"), (
        "CONTRIBUTING.md must tell contributors to create feature branches from dev"
    )


def test_contributing_pull_requests_target_dev() -> None:
    assert _chunk_matching(r"\bpull requests?\b", rf"\b(?:target|targets|against|into)\s+(?:the\s+)?{DEV}"), (
        "CONTRIBUTING.md must state that pull requests target dev"
    )


def test_contributing_main_holds_releases() -> None:
    assert _chunk_matching(r"`?main`?(?![\w-])", r"\breleases?\b"), (
        "CONTRIBUTING.md must state that the main branch holds releases"
    )


def test_contributing_imperative_commits() -> None:
    assert _chunk_matching(r"\bcommit", r"\bimperative\b"), "CONTRIBUTING.md must ask for imperative commit messages"


def test_contributing_one_logical_change() -> None:
    assert re.search(r"one logical change per commit", _contributing(), re.IGNORECASE), (
        "CONTRIBUTING.md must ask for one logical change per commit"
    )


def test_contributing_licence_is_cecill_b() -> None:
    assert _chunk_matching(r"\bcontribution", r"CeCILL-B"), (
        "CONTRIBUTING.md must state that contributions are licensed under CeCILL-B"
    )


def test_contributing_links_license_file() -> None:
    assert re.search(r"\]\(\s*(?:\./)?LICENSE\s*\)", _contributing()), "CONTRIBUTING.md must link to LICENSE"


def test_contributing_links_issue_tracker() -> None:
    link = r"\]\(\s*https://github\.com/neurospin-projects/[A-Za-z0-9_.-]+/issues/?\s*\)"
    assert _chunk_matching(r"\bissues?\b", link), "CONTRIBUTING.md must link the repository's GitHub issues page"


def test_contributing_names_no_internal_tooling() -> None:
    found = sorted({m.group(0) for m in INTERNAL_WORDS.finditer(_contributing())})
    assert found == [], f"CONTRIBUTING.md names internal tooling: {found}"


def test_contributing_names_no_client_project() -> None:
    offending = [n for n, line in enumerate(_contributing().splitlines(), 1) if _forbidden_in(line)]
    assert offending == [], f"CONTRIBUTING.md names a client project at lines: {offending}"


def test_readme_links_contributing() -> None:
    text = README.read_text(encoding="utf-8")
    assert re.search(r"\]\(\s*(?:\./)?CONTRIBUTING\.md\s*\)", text), "README.md must link to CONTRIBUTING.md"
