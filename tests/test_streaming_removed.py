"""The start_streaming tool is removed (user decision 2026-10-19).

- REQ-MCPSTREAM-BDRABCZUK-9B83FBB8EF84: ``tools.stages`` defines no ``start_streaming``.
- REQ-MCPSTREAM-BDRABCZUK-9B34F6383ACD: no Markdown file under ``skills/`` or
  ``agents/``, nor ``README.md``, contains the word ``streaming`` (case-insensitive).
- REQ-MCPSTREAM-BDRABCZUK-68A67F6C9337: the (gitignored) CLAUDE.md Security
  statement names ``start_embeddings`` as the only tool passing ``HF_TOKEN``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from champollion_sulcal_mcp.tools import stages

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"

DOCS = sorted([*(REPO_ROOT / "skills").rglob("*.md"), *(REPO_ROOT / "agents").rglob("*.md"), REPO_ROOT / "README.md"])


def test_stages_module_has_no_start_streaming() -> None:
    """REQ-MCPSTREAM-BDRABCZUK-9B83FBB8EF84: tools.stages defines no attribute named start_streaming."""
    assert not hasattr(stages, "start_streaming")


@pytest.mark.parametrize("doc", DOCS, ids=[str(p.relative_to(REPO_ROOT)) for p in DOCS])
def test_doc_does_not_mention_streaming(doc: Path) -> None:
    """REQ-MCPSTREAM-BDRABCZUK-9B34F6383ACD: the doc contains no case-insensitive `streaming`."""
    hits = [
        f"{n}: {line.strip()}"
        for n, line in enumerate(doc.read_text().splitlines(), start=1)
        if re.search("streaming", line, re.IGNORECASE)
    ]
    assert hits == [], f"{doc.relative_to(REPO_ROOT)} mentions streaming:\n" + "\n".join(hits)


@pytest.mark.skipif(not CLAUDE_MD.is_file(), reason="CLAUDE.md is gitignored and absent from this checkout")
def test_claude_md_security_names_only_start_embeddings() -> None:
    """REQ-MCPSTREAM-BDRABCZUK-68A67F6C9337: Security names start_embeddings as the only HF_TOKEN tool."""
    security = next(
        (line for line in CLAUDE_MD.read_text().splitlines() if line.startswith("**Security:**")),
        None,
    )
    assert security is not None, "CLAUDE.md has no **Security:** statement"
    assert "HF_TOKEN" in security
    assert "start_embeddings" in security
    tools = set(re.findall(r"`(start_\w+)`", security))
    assert tools == {"start_embeddings"}, f"Security statement names HF_TOKEN tools {sorted(tools)}"
