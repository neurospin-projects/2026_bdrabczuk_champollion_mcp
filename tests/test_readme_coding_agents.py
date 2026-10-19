"""README.md documents the server for any coding agent, not only Claude Code.

REQ-MCPREL-BDRABCZUK-7B217C9AB3E6: README.md contains the level-two heading "## Coding agents integration".
REQ-MCPREL-BDRABCZUK-85349CA04178: README.md contains no heading whose text is "Claude Code integration".
REQ-MCPREL-BDRABCZUK-B4207716ADF8: the section holds one level-three subsection with a configuration
snippet for each of Claude Code, OpenCode, Mistral Vibe, Codex and other MCP clients.
REQ-MCPREL-BDRABCZUK-56FCF1D10E1E: its snippets launch the server with
"pixi run --manifest-path /path/to/champollion_sulcal_mcp/pixi.toml run".
REQ-MCPREL-BDRABCZUK-0D590CC8D05E: its snippets set CHAMPOLLION_PIPELINE_DIR.
REQ-MCPREL-BDRABCZUK-08F8FE27E115: it states that HF_TOKEN is optional.
REQ-MCPREL-BDRABCZUK-083259D0F661: the OpenCode snippet sets "timeout" to 30000.
REQ-MCPREL-BDRABCZUK-3848DB551D4E: the Mistral Vibe and Codex snippets set startup_timeout_sec to 60.
REQ-MCPREL-BDRABCZUK-F8EF9D252022: its json and jsonc blocks parse as JSON once // comment lines go.
REQ-MCPREL-BDRABCZUK-442F378D5B2E: its toml blocks parse with tomllib.
REQ-MCPREL-BDRABCZUK-D8C02AF37529: the snippet for other MCP clients is a JSON object whose only
top-level key is "mcpServers".
REQ-MCPREL-BDRABCZUK-F0BBFD228515: README.md contains no text matching hf_[A-Za-z0-9]{4,}.
REQ-MCPREL-BDRABCZUK-A742EDFF47AE: the section lists the Claude Code subagent and the three skills.
REQ-MCPREL-BDRABCZUK-F23F8C4F2DC8: it states that other coding agents can read them as plain instructions.
REQ-MCPREL-BDRABCZUK-4557544C361F: the "Running" section links to #coding-agents-integration.

A section runs from its heading to the next heading of the same or a higher level; headings inside
fenced code blocks do not count. A snippet is a fenced code block. A prose statement is looked for in one
chunk: a paragraph (lines between blank lines), except that each list item and each table row is a
chunk of its own.
"""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"

SECTION_TITLE = "Coding agents integration"
OLD_TITLE = "Claude Code integration"
LAUNCH_ARGS = ["run", "--manifest-path", "/path/to/champollion_sulcal_mcp/pixi.toml", "run"]
CLIENT_HEADINGS = {
    "Claude Code": re.compile(r"\bClaude Code\b"),
    "OpenCode": re.compile(r"\bOpenCode\b"),
    "Mistral Vibe": re.compile(r"\bMistral Vibe\b"),
    "Codex": re.compile(r"\bCodex\b"),
    "other MCP clients": re.compile(r"\bMCP clients?\b", re.IGNORECASE),
}
CLAUDE_CODE_ASSETS = ("agents/champollion-pipeline.md", "skills/run-pipeline", "skills/monitor", "skills/debug")
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)\s*([\w+-]*)")
ITEM_START = re.compile(r"^\s*(?:\||[-*+] |\d+[.)] )")
TOKEN = re.compile(r'"([^"]*)"|([^\s"\[\]{}:,=\\]+)')
HF_TOKEN_VALUE = re.compile(r"hf_[A-Za-z0-9]{4,}")


@dataclass
class Block:
    lang: str
    text: str


@dataclass
class Section:
    level: int
    title: str
    prose: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    children: list[Section] = field(default_factory=list)

    def all_blocks(self) -> list[Block]:
        return self.blocks + [b for child in self.children for b in child.all_blocks()]

    def all_prose(self) -> list[str]:
        return self.prose + [line for child in self.children for line in child.all_prose()]


def _readme() -> str:
    assert README.is_file(), f"missing {README}"
    return README.read_text(encoding="utf-8")


def _headings(text: str) -> list[tuple[int, str]]:
    """Level and text of every Markdown heading outside fenced code blocks."""
    found: list[tuple[int, str]] = []
    in_fence = False
    for line in text.splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        match = None if in_fence else HEADING.match(line)
        if match:
            found.append((len(match.group(1)), match.group(2)))
    return found


def _sections(text: str) -> list[Section]:
    """Flat list of sections; each one also holds its deeper sections as children."""
    roots: list[Section] = []
    stack: list[Section] = []
    fence: Block | None = None
    fence_lines: list[str] = []
    for line in text.splitlines():
        fence_match = FENCE.match(line)
        if fence is not None:
            if fence_match and not fence_match.group(2):
                fence.text = "\n".join(fence_lines)
                if stack:
                    stack[-1].blocks.append(fence)
                fence = None
            else:
                fence_lines.append(line)
            continue
        if fence_match:
            fence, fence_lines = Block(fence_match.group(2).lower(), ""), []
            continue
        heading = HEADING.match(line)
        if heading:
            section = Section(len(heading.group(1)), heading.group(2))
            while stack and stack[-1].level >= section.level:
                stack.pop()
            if stack:
                stack[-1].children.append(section)
            roots.append(section)
            stack.append(section)
        elif stack:
            stack[-1].prose.append(line)
    return roots


def _section(title: str, level: int = 2) -> Section:
    matches = [s for s in _sections(_readme()) if s.level == level and s.title == title]
    assert matches, f"README.md has no {'#' * level} {title} section"
    return matches[0]


def _chunks(lines: list[str]) -> list[str]:
    """Paragraphs; each list item and each table row starts a chunk of its own."""
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        if not line.strip():
            if current:
                chunks.append(current)
            current = []
        elif ITEM_START.match(line) and current:
            chunks.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        chunks.append(current)
    return [" ".join(chunk) for chunk in chunks]


def _client_subsections(name: str) -> list[Section]:
    pattern = CLIENT_HEADINGS[name]
    return [
        child
        for child in _section(SECTION_TITLE).children
        if child.level == 3 and pattern.search(child.title) and child.all_blocks()
    ]


def _one_client(name: str) -> Section:
    found = _client_subsections(name)
    assert len(found) == 1, f"expected one ### subsection with a snippet for {name}, got {[s.title for s in found]}"
    return found[0]


def _snippets() -> list[Block]:
    blocks = _section(SECTION_TITLE).all_blocks()
    assert blocks, f"the {SECTION_TITLE} section has no fenced code block"
    return blocks


def _parse_json(block: Block) -> object:
    kept = [line for line in block.text.splitlines() if not line.lstrip().startswith("//")]
    return json.loads("\n".join(kept))


def _values(data: object, key: str) -> list[object]:
    """Every value stored under ``key`` at any depth of parsed JSON or TOML."""
    if isinstance(data, dict):
        own = [data[key]] if key in data else []
        return own + [v for value in data.values() for v in _values(value, key)]
    if isinstance(data, list):
        return [v for item in data for v in _values(item, key)]
    return []


def _contains_run(tokens: list[str], run: list[str]) -> bool:
    return any(tokens[i : i + len(run)] == run for i in range(len(tokens) - len(run) + 1))


def test_new_heading_present() -> None:
    assert (2, SECTION_TITLE) in _headings(_readme())


def test_old_heading_absent() -> None:
    titles = [title for _, title in _headings(_readme())]
    assert OLD_TITLE not in titles


@pytest.mark.parametrize("client", list(CLIENT_HEADINGS))
def test_one_subsection_per_client(client: str) -> None:
    _one_client(client)


def test_snippets_use_manifest_path() -> None:
    for block in _snippets():
        tokens = [a or b for a, b in TOKEN.findall(block.text)]
        assert "pixi" in tokens, f"snippet does not launch pixi:\n{block.text}"
        assert _contains_run(tokens, LAUNCH_ARGS), f"snippet lacks {' '.join(LAUNCH_ARGS)}:\n{block.text}"


def test_snippets_set_pipeline_dir() -> None:
    for block in _snippets():
        assert "CHAMPOLLION_PIPELINE_DIR" in block.text, f"snippet lacks CHAMPOLLION_PIPELINE_DIR:\n{block.text}"


def test_hf_token_optional() -> None:
    chunks = _chunks(_section(SECTION_TITLE).all_prose())
    assert any("HF_TOKEN" in c and "optional" in c.lower() for c in chunks)


def test_opencode_timeout() -> None:
    opencode = _one_client("OpenCode")
    timeouts = [
        v for b in opencode.all_blocks() if b.lang in ("json", "jsonc") for v in _values(_parse_json(b), "timeout")
    ]
    assert timeouts == [30000]


@pytest.mark.parametrize("client", ["Mistral Vibe", "Codex"])
def test_vibe_and_codex_startup_timeout(client: str) -> None:
    sub = _one_client(client)
    tables = [tomllib.loads(b.text) for b in sub.all_blocks() if b.lang == "toml"]
    assert tables, f"no toml snippet for {client}"
    assert [v for t in tables for v in _values(t, "startup_timeout_sec")] == [60]


def test_json_snippets_parse() -> None:
    blocks = [b for b in _snippets() if b.lang in ("json", "jsonc")]
    assert blocks, "no json or jsonc snippet"
    for block in blocks:
        _parse_json(block)


def test_toml_snippets_parse() -> None:
    blocks = [b for b in _snippets() if b.lang == "toml"]
    assert blocks, "no toml snippet"
    for block in blocks:
        tomllib.loads(block.text)


def test_generic_snippet_mcpservers() -> None:
    generic = _one_client("other MCP clients")
    parsed = [_parse_json(b) for b in generic.all_blocks() if b.lang in ("json", "jsonc")]
    assert parsed, "no JSON snippet for other MCP clients"
    assert all(isinstance(p, dict) and list(p) == ["mcpServers"] for p in parsed)


def test_no_literal_hf_token() -> None:
    assert HF_TOKEN_VALUE.findall(_readme()) == []


def test_claude_code_assets_listed() -> None:
    section = _section(SECTION_TITLE)
    text = "\n".join(section.all_prose())
    missing = [asset for asset in CLAUDE_CODE_ASSETS if asset not in text]
    assert not missing, f"{SECTION_TITLE} section does not list {missing}"


def test_assets_readable_as_plain_instructions() -> None:
    chunks = _chunks(_section(SECTION_TITLE).all_prose())
    assert any("plain instructions" in c.lower() and "agent" in c.lower() for c in chunks)


def test_running_links_integration() -> None:
    text = "\n".join(_section("Running").prose)
    assert "(#coding-agents-integration)" in text
