"""champollion-sulcal-mcp is distributed under the CeCILL-B licence.

REQ-MCPREL-BDRABCZUK-2F27A910E1C2: the root directory of the repository shall contain a file named
LICENSE whose bytes are identical to the CeCILL-B Free Software License Agreement version 1.0 text of
champollion_pipeline/LICENSE (SHA-256 pinned below).

REQ-MCPREL-BDRABCZUK-57E72B2A1708: the [project] table of pyproject.toml shall declare the package
licence as the text CeCILL-B.

REQ-MCPREL-BDRABCZUK-E5903D51535A: README.md shall contain a level-two section titled License that
names CeCILL-B as the licence of champollion-sulcal-mcp.

REQ-MCPREL-BDRABCZUK-7427AC685522: the License section of README.md shall contain a Markdown link
whose target is the LICENSE file at the repository root.

The repository is self-contained, so the reference text is pinned by its SHA-256 digest, its title
line and its newline count instead of being read from a champollion_pipeline checkout.
"""

from __future__ import annotations

import hashlib
import re
import tomllib
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
LICENSE = REPO_ROOT / "LICENSE"
PYPROJECT = REPO_ROOT / "pyproject.toml"
README = REPO_ROOT / "README.md"

CECILL_B_TITLE = "CeCILL-B FREE SOFTWARE LICENSE AGREEMENT"
CECILL_B_SHA256 = "f1018669f184667cb73b3e6e8e0b7e1b05e6d08a0ab473911f3055fccf614b19"
CECILL_B_NEWLINES = 514

LINK_TARGET = re.compile(r"\]\(\s*(?:\./)?LICENSE\s*\)")


def _license_bytes() -> bytes:
    assert LICENSE.is_file(), f"missing licence file: {LICENSE}"
    return LICENSE.read_bytes()


def _readme_license_section() -> str:
    """Return the body of README's '## License' section (up to the next level-one or level-two heading)."""
    lines = README.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "## License"]
    assert len(starts) == 1, f"README.md must contain exactly one '## License' heading, found {len(starts)}"
    body: list[str] = []
    in_fence = False
    for line in lines[starts[0] + 1 :]:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and re.match(r"#{1,2} ", line):
            break
        body.append(line)
    return "\n".join(body)


def test_license_file_title_is_cecill_b() -> None:
    text = _license_bytes().decode("utf-8")
    first = next((line.strip() for line in text.splitlines() if line.strip()), "")
    assert first == CECILL_B_TITLE


def test_license_file_matches_pipeline_cecill_b_text() -> None:
    data = _license_bytes()
    assert data.count(b"\n") == CECILL_B_NEWLINES
    assert hashlib.sha256(data).hexdigest() == CECILL_B_SHA256


def test_pyproject_declares_cecill_b() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert project.get("license") == {"text": "CeCILL-B"}


def test_readme_license_section_names_cecill_b() -> None:
    assert "CeCILL-B" in _readme_license_section()


def test_readme_license_section_links_license_file() -> None:
    assert LINK_TARGET.search(_readme_license_section()), "the License section must link to LICENSE"
