"""AGENTS.md is the tracked, self-contained guide for coding agents working in champollion_sulcal_mcp.

REQ-MCPREL-BDRABCZUK-38613BE51A2B: AGENTS.md describes the server as a FastMCP server using the stdio
transport.
REQ-MCPREL-BDRABCZUK-717D26165399: it states 16 as the number of MCP tools the server registers, with
no other tool count.
REQ-MCPREL-BDRABCZUK-925942FB6466: it states that the server runs champollion_pipeline stage scripts as
background subprocesses.
REQ-MCPREL-BDRABCZUK-2DDC93970A73: it states that CHAMPOLLION_PIPELINE_DIR is required.
REQ-MCPREL-BDRABCZUK-61C0E6671978: it states that work in this repository does not edit the
champollion_pipeline repository.
REQ-MCPREL-BDRABCZUK-D7FFF5704BD4: it contains `pixi run format`, `pixi run lint` and `pixi run test`.
REQ-MCPREL-BDRABCZUK-EC3C572A2E46: each `pixi run` command it names is a task of pixi.toml's [tasks].
REQ-MCPREL-BDRABCZUK-1C6BF8DC7548: it instructs agents to create task branches from dev.
REQ-MCPREL-BDRABCZUK-1DFD4BA9BD10: it states that dev is merged into main as a separate step.
REQ-MCPREL-BDRABCZUK-F51A888DA78B: it states that task work is not committed on main.
REQ-MCPREL-BDRABCZUK-80E5C70AAD65: it asks for commit messages in the imperative mood.
REQ-MCPREL-BDRABCZUK-B2BF04A28905: it asks for one logical change per commit.
REQ-MCPREL-BDRABCZUK-2DD80918C633: it names start_embeddings as the only MCP tool that passes HF_TOKEN
to the subprocess environment.
REQ-MCPREL-BDRABCZUK-0CD7EF0CC308: it states that HF_TOKEN is not written to job JSON files or job log
files.
REQ-MCPREL-BDRABCZUK-C3D0DE1DF6D0: it links README.md at the repository root.
REQ-MCPREL-BDRABCZUK-62C1D796185C: it links CONTRIBUTING.md at the repository root.
REQ-MCPREL-BDRABCZUK-05040EC92FBD: it contains none of a set of out-of-repository texts, compared
case-insensitively. The client project among them is matched through the hashed detector of
tests/test_docs_self_contained.py, so this module does not spell it out (it is a tracked file).

Statements pairing two facts are looked for inside one chunk of text (a paragraph, a table row or a
list item), as in tests/test_contributing_doc.py.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest

from tests.test_contributing_doc import _chunks
from tests.test_docs_self_contained import _forbidden_in

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENTS_MD = REPO_ROOT / "AGENTS.md"
PIXI_TOML = REPO_ROOT / "pixi.toml"

PIXI_COMMANDS = ("format", "lint", "test")
PIXI_RUN = re.compile(r"\bpixi run ([A-Za-z0-9_-]+)")
TOOL_COUNT = re.compile(r"\b(\d+)\s+(?:MCP\s+)?tools\b", re.IGNORECASE)
DEV = r"`?dev`?(?![\w-])"
MAIN = r"`?main`?(?![\w-])"
NEGATION = r"\b(?:never|not)\b"
# Literal out-of-repository texts; the client project is checked by _forbidden_in instead.
FORBIDDEN_TEXTS = ("champollion_utils", "../", "master", "start_streaming")


def _agents_md() -> str:
    assert AGENTS_MD.is_file(), f"missing agent guide: {AGENTS_MD}"
    return AGENTS_MD.read_text(encoding="utf-8")


def _chunk_matching(*patterns: str) -> str | None:
    """First chunk of AGENTS.md matching every pattern (case-insensitive), else None."""
    for chunk in _chunks(_agents_md()):
        if all(re.search(p, chunk, re.IGNORECASE) for p in patterns):
            return chunk
    return None


def _links_to(name: str) -> bool:
    target = rf"\]\(\s*(?:\./)?{re.escape(name)}(?:#[^)\s]*)?\s*\)"
    return re.search(target, _agents_md()) is not None


def test_agents_md_fastmcp_stdio() -> None:
    assert _chunk_matching(r"\bFastMCP\b", r"\bstdio\b"), "AGENTS.md must describe a FastMCP stdio server"


def test_agents_md_states_16_tools() -> None:
    counts = set(TOOL_COUNT.findall(_agents_md()))
    assert counts == {"16"}, f"AGENTS.md must state 16 tools and no other tool count, found: {sorted(counts)}"


def test_agents_md_background_subprocesses() -> None:
    assert _chunk_matching(r"\bchampollion_pipeline\b", r"\bbackground\b", r"\bsubprocess(?:es)?\b"), (
        "AGENTS.md must state that champollion_pipeline stage scripts run as background subprocesses"
    )


def test_agents_md_requires_pipeline_dir() -> None:
    assert _chunk_matching(r"\bCHAMPOLLION_PIPELINE_DIR\b", r"\brequired\b"), (
        "AGENTS.md must state that CHAMPOLLION_PIPELINE_DIR is required"
    )


def test_agents_md_pipeline_not_edited() -> None:
    assert _chunk_matching(r"\bchampollion_pipeline\b", rf"{NEGATION}\s+(?:be\s+)?(?:edit|modif)\w*"), (
        "AGENTS.md must state that the champollion_pipeline repository is not edited from here"
    )


def test_agents_md_names_pixi_commands() -> None:
    text = _agents_md()
    missing = [c for c in PIXI_COMMANDS if not re.search(rf"\bpixi run {re.escape(c)}(?![\w-])", text)]
    assert missing == [], f"AGENTS.md lacks `pixi run` for: {missing}"


def test_agents_md_pixi_commands_exist() -> None:
    tasks = set(tomllib.loads(PIXI_TOML.read_text(encoding="utf-8"))["tasks"])
    named = set(PIXI_RUN.findall(_agents_md()))
    assert named, "AGENTS.md names no `pixi run` command"
    assert named <= tasks, f"AGENTS.md names pixi tasks absent from pixi.toml: {sorted(named - tasks)}"


def test_agents_md_task_branches_from_dev() -> None:
    assert _chunk_matching(r"\btask branch", rf"\bfrom\s+(?:the\s+)?{DEV}"), (
        "AGENTS.md must tell agents to create task branches from dev"
    )


def test_agents_md_dev_merged_into_main_separately() -> None:
    assert _chunk_matching(DEV, rf"\bmerged?\s+into\s+(?:the\s+)?{MAIN}", r"\bseparate"), (
        "AGENTS.md must state that dev is merged into main as a separate step"
    )


def test_agents_md_no_task_work_on_main() -> None:
    assert _chunk_matching(r"\btask work\b", rf"{NEGATION}[^.]*\bon\s+(?:the\s+)?{MAIN}"), (
        "AGENTS.md must state that task work is not committed on main"
    )


def test_agents_md_imperative_commits() -> None:
    assert _chunk_matching(r"\bcommit", r"\bimperative\b"), "AGENTS.md must ask for imperative commit messages"


def test_agents_md_one_logical_change() -> None:
    assert re.search(r"one logical change per commit", _agents_md(), re.IGNORECASE), (
        "AGENTS.md must ask for one logical change per commit"
    )


def test_agents_md_hf_token_start_embeddings_only() -> None:
    assert _chunk_matching(r"\bHF_TOKEN\b", r"\bstart_embeddings\b", r"\bonly\b"), (
        "AGENTS.md must name start_embeddings as the only tool that passes HF_TOKEN to the subprocess"
    )


def test_agents_md_hf_token_not_persisted() -> None:
    assert _chunk_matching(
        r"\bHF_TOKEN\b",
        rf"{NEGATION}\s+(?:be\s+)?(?:written|stored|persisted|logged)\b",
        r"\bJSON\b",
        r"\blogs?\b",
    ), "AGENTS.md must state that HF_TOKEN is not written to job JSON or log files"


def test_agents_md_links_readme() -> None:
    assert _links_to("README.md"), "AGENTS.md must link to README.md"


def test_agents_md_links_contributing() -> None:
    assert _links_to("CONTRIBUTING.md"), "AGENTS.md must link to CONTRIBUTING.md"


def test_agents_md_has_no_forbidden_text() -> None:
    lines = _agents_md().splitlines()
    found = sorted({t for t in FORBIDDEN_TEXTS for line in lines if t in line.lower()})
    client_lines = [n for n, line in enumerate(lines, 1) if _forbidden_in(line)]
    assert found == [], f"AGENTS.md contains out-of-repository text: {found}"
    assert client_lines == [], f"AGENTS.md names a client project at lines: {client_lines}"
