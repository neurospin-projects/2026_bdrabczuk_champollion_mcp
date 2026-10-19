"""The MCP server repository is self-contained: no client project is referenced.

REQ-MCPSELF-BDRABCZUK-C7125EF3AE31: apart from this test module, no file tracked by git in the
champollion_sulcal_mcp repository names a client project of this server, in its path or its text
(case-insensitive). The forbidden names are stored only as SHA-256 digests, so this module does not
spell them out either. A name matches when it is a word of the text: a run of letters and digits joined
by "_" or "-" (any contiguous part of such a run counts), or two such words separated by whitespace.

REQ-MCPSELF-BDRABCZUK-73BADD41F11A: the '## Requirements' section of README.md shall name
champollion_pipeline without using the word sibling.

``agents/champollion-pipeline.md`` (this repo's own Claude Code subagent) and generic words such as
"agent" are deliberately not forbidden. The README Configuration table row for
``CHAMPOLLION_PIPELINE_DIR`` belongs to another Task and is outside the '## Requirements' section.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
THIS_MODULE = Path(__file__).resolve().relative_to(REPO_ROOT).as_posix()
README = REPO_ROOT / "README.md"

# SHA-256 of the six forbidden lowercase names (five single words and one two-word name).
FORBIDDEN_SHA256 = frozenset(
    {
        "aff133a73d1257e637001d74524d728eba62a262884e6717031848e06ca410ee",
        "cbdcdb11708ca75b772093999b2bdf927a539a6454e26b0a1d16468b66af926c",
        "29ba9d9cef5a66461116a24938bb9307e005c35aa1bb909f16aa5e85bd767480",
        "22c009485a6d8f139582719426054c126f7f8b426351dbfb5681cddb42ae180d",
        "8671e007946eda1fe3398a887e08ccadf1667802c2a3466d203d9910ca05371c",
        "71410c63c3ccd9555d945314ff9e2bef73d3a686ffa064aa89695aeabbe8c65f",
    }
)
WORD = re.compile(r"[a-z0-9]+(?:[_-][a-z0-9]+)*")
PART = re.compile(r"[a-z0-9]+|[_-]")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _candidates(line: str) -> set[str]:
    """Every word of a lowercase line, every contiguous part of a joined word, and every word pair."""
    words = WORD.findall(line)
    found = set()
    for word in words:
        parts = PART.findall(word)
        for i in range(0, len(parts), 2):
            for j in range(i + 1, len(parts) + 1, 2):
                found.add("".join(parts[i:j]))
    found.update(f"{a} {b}" for a, b in zip(words, words[1:], strict=False))
    return found


def _forbidden_in(line: str) -> bool:
    return any(_sha256(c) in FORBIDDEN_SHA256 for c in _candidates(line.lower()))


def _tracked_files() -> list[str]:
    """Repository-relative paths of every file in the git index, this module excluded."""
    if shutil.which("git") is None:
        pytest.skip("git executable not available")
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    paths = [p for p in result.stdout.decode().split("\0") if p]
    return [p for p in paths if p != THIS_MODULE]


def _readme_requirements_section() -> str:
    """Text of README.md's '## Requirements' section, up to the next '## ' heading."""
    match = re.search(r"^## Requirements\b.*?(?=^## |\Z)", README.read_text(), re.MULTILINE | re.DOTALL)
    assert match, "README.md has no '## Requirements' section"
    return match.group(0)


def test_forbidden_name_detector_matches_words_parts_and_pairs() -> None:
    """The detector flags a joined word containing a forbidden name, and ignores generic words."""
    assert _candidates("x-foo-bar-y") >= {"foo-bar", "x-foo-bar-y", "foo", "y"}
    assert not _forbidden_in("an agent drives the mcp server")


def test_tracked_file_text_has_no_client_reference() -> None:
    """No tracked file's text names a client project."""
    offending = []
    for rel in _tracked_files():
        path = REPO_ROOT / rel
        if not path.is_file():
            continue
        text = path.read_bytes().decode("utf-8", errors="ignore")
        offending.extend(f"{rel}:{n}" for n, line in enumerate(text.splitlines(), 1) if _forbidden_in(line))
    assert offending == [], f"tracked files name a client project at: {offending}"


def test_tracked_path_has_no_client_reference() -> None:
    """No tracked file's path names a client project."""
    offending = [rel for rel in _tracked_files() if _forbidden_in(rel)]
    assert offending == [], f"tracked paths name a client project: {offending}"


def test_docs_folder_holds_no_client_design_document() -> None:
    """No file under docs/ (tracked or not) names a client project in its file name."""
    docs = REPO_ROOT / "docs"
    offending = [p.name for p in docs.rglob("*") if _forbidden_in(p.name)] if docs.is_dir() else []
    assert offending == [], f"docs/ holds a client design document: {offending}"


def test_readme_requirements_names_pipeline() -> None:
    """The README '## Requirements' section names champollion_pipeline as a dependency."""
    assert "champollion_pipeline" in _readme_requirements_section()


def test_readme_requirements_has_no_sibling_wording() -> None:
    """The README '## Requirements' section does not call champollion_pipeline a sibling."""
    offending = [line for line in _readme_requirements_section().splitlines() if "sibling" in line.lower()]
    assert offending == [], f"README '## Requirements' still uses 'sibling': {offending}"
