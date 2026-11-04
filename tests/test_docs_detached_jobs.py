"""Detached jobs are documented: they outlive the MCP server, and a vanished process is a failed job.

README.md, "Job tracking" section:
REQ-MCPJOBS-BDRABCZUK-C18E24EFF3B2: it states that a job keeps running after the MCP server exits.
REQ-MCPJOBS-BDRABCZUK-908B7FB2D56E: it states that start_pipeline runs the stage chain in a detached
driver process.
REQ-MCPJOBS-BDRABCZUK-2828D5402338: it states that get_job_status, list_jobs and get_job_log reconcile a
running job with the job's process before answering.
REQ-MCPJOBS-BDRABCZUK-A36C61A507FB: it states that cancel_job sends SIGTERM to the job's process group,
then SIGKILL to the surviving processes after a grace period.
REQ-MCPJOBS-BDRABCZUK-0334BD1D58E8: it names the error text "process gone without exit code".
REQ-MCPJOBS-BDRABCZUK-D938DBEF426C: the README paragraph naming that error states that the returncode is
null.
REQ-MCPJOBS-BDRABCZUK-F0DA13C1D435: the README paragraph naming that error directs the reader to
get_job_log.
REQ-MCPJOBS-BDRABCZUK-AB5D292336CC: README.md, CHANGELOG.md, skills/monitor/SKILL.md,
skills/debug/SKILL.md and agents/champollion-pipeline.md contain no backticked `lost` token (there is no
status lost).

skills/monitor/SKILL.md:
REQ-MCPJOBS-BDRABCZUK-4C839552553B: it states that a job keeps running after the MCP server exits.
REQ-MCPJOBS-BDRABCZUK-C79C8D424395: its Terminal States table holds a row naming
"process gone without exit code" whose action names get_job_log.
REQ-MCPJOBS-BDRABCZUK-F02A2ED255A8: its Cancellation section states that cancel_job signals the job's
process group.

skills/debug/SKILL.md:
REQ-MCPJOBS-BDRABCZUK-FB2D6F69C6DA: it names the error text "process gone without exit code".
REQ-MCPJOBS-BDRABCZUK-CC66EEBE43FC: it names the error text "Process exited with code".
REQ-MCPJOBS-BDRABCZUK-9E9AA3057B58: the paragraph naming "process gone without exit code" directs the
reader to get_job_log before relaunching the job.

agents/champollion-pipeline.md:
REQ-MCPJOBS-BDRABCZUK-A82F414A7B08: its "Step 3 — Monitor" section states that a job keeps running after
the MCP server exits.
REQ-MCPJOBS-BDRABCZUK-0B85AB981247: its "Debug Mode" section names "process gone without exit code".

CHANGELOG.md:
REQ-MCPJOBS-BDRABCZUK-D40C0DCB7297: it contains '## [Unreleased]' above '## [0.1.0] - 2026-10-19'.
REQ-MCPJOBS-BDRABCZUK-569A68D1D4A0: the Unreleased entry states that a job keeps running after the MCP
server exits.
REQ-MCPJOBS-BDRABCZUK-3C800D8B7BF6: the Unreleased entry names "process gone without exit code".
REQ-MCPJOBS-BDRABCZUK-F2C5C1934735: the Unreleased entry states that cancel_job signals the job's
process group.
REQ-MCPJOBS-BDRABCZUK-E3492A2168BB: the Unreleased entry states that start_pipeline runs the stage
chain in a detached driver process.

A section runs from its heading to the next heading of the same or a higher level. A "paragraph" is a
chunk as in tests/test_contributing_doc.py: lines between blank lines, except that each table row and
each list item is a chunk of its own. Statements are matched case-insensitively with whitespace runs
collapsed, so a sentence may wrap across lines.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.test_contributing_doc import _chunks

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parent.parent
README = REPO_ROOT / "README.md"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"
MONITOR = REPO_ROOT / "skills" / "monitor" / "SKILL.md"
DEBUG = REPO_ROOT / "skills" / "debug" / "SKILL.md"
AGENT = REPO_ROOT / "agents" / "champollion-pipeline.md"

PROCESS_GONE = "process gone without exit code"
EXIT_CODE_ERROR = "Process exited with code"
OUTLIVES_SERVER = r"keeps running after the MCP server exits"
DETACHED_DRIVER = r"detached driver"
RELEASE_HEADING = "## [0.1.0] - 2026-10-19"
UNRELEASED_HEADING = "## [Unreleased]"
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*$")


def _read(path: Path) -> str:
    assert path.is_file(), f"missing document: {path}"
    return path.read_text(encoding="utf-8")


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text)


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, _flat(text), re.IGNORECASE) is not None


def _section(path: Path, title: str) -> str:
    """Text under the heading titled ``title`` up to the next heading of the same or a higher level."""
    lines = _read(path).splitlines()
    for start, line in enumerate(lines):
        match = HEADING.match(line)
        if match and match.group(2) == title:
            level = len(match.group(1))
            body: list[str] = []
            for nxt in lines[start + 1 :]:
                other = HEADING.match(nxt)
                if other and len(other.group(1)) <= level:
                    break
                body.append(nxt)
            return "\n".join(body)
    pytest.fail(f"{path.name}: no heading {title!r}")


def _chunk_naming(text: str, needle: str) -> str:
    for chunk in _chunks(text):
        if needle.lower() in _flat(chunk).lower():
            return _flat(chunk)
    pytest.fail(f"no paragraph names {needle!r}")


def _unreleased_entry() -> str:
    match = re.search(
        rf"^{re.escape(UNRELEASED_HEADING)}[ \t]*$(.*?)(?=^## |\Z)", _read(CHANGELOG), re.MULTILINE | re.DOTALL
    )
    assert match, f"CHANGELOG.md has no {UNRELEASED_HEADING!r} heading"
    return match.group(1)


def _job_tracking() -> str:
    return _section(README, "Job tracking")


# README.md, "Job tracking" section


def test_readme_job_outlives_server() -> None:
    assert _has(_job_tracking(), OUTLIVES_SERVER), "Job tracking does not say a job outlives the MCP server"


def test_readme_pipeline_detached_driver() -> None:
    section = _job_tracking()
    assert "start_pipeline" in section and _has(section, DETACHED_DRIVER), (
        "Job tracking does not say start_pipeline runs its chain in a detached driver process"
    )


@pytest.mark.parametrize("tool", ["get_job_status", "list_jobs", "get_job_log"])
def test_readme_reconcile_on_read(tool: str) -> None:
    section = _job_tracking()
    assert _has(section, r"reconcil"), "Job tracking does not describe reconcile-on-read"
    assert tool in section, f"Job tracking does not name {tool} as reconciling on read"


@pytest.mark.parametrize("term", ["cancel_job", "process group", "SIGTERM", "SIGKILL", "grace"])
def test_readme_cancel_process_group(term: str) -> None:
    assert _has(_job_tracking(), re.escape(term)), f"Job tracking does not name {term!r} for cancellation"


def test_readme_names_process_gone_error() -> None:
    assert PROCESS_GONE in _flat(_job_tracking()), f"Job tracking does not name {PROCESS_GONE!r}"


def test_readme_process_gone_returncode_null() -> None:
    chunk = _chunk_naming(_read(README), PROCESS_GONE)
    assert "returncode" in chunk and _has(chunk, r"\bnull\b"), f"no null returncode in: {chunk}"


def test_readme_process_gone_points_to_log() -> None:
    chunk = _chunk_naming(_read(README), PROCESS_GONE)
    assert "get_job_log" in chunk, f"no get_job_log in: {chunk}"


@pytest.mark.parametrize("path", [README, CHANGELOG, MONITOR, DEBUG, AGENT], ids=lambda p: p.name)
def test_docs_name_no_lost_status(path: Path) -> None:
    assert "`lost`" not in _read(path), f"{path.relative_to(REPO_ROOT)} names a `lost` status"


# skills/monitor/SKILL.md


def test_monitor_job_outlives_server() -> None:
    assert _has(_read(MONITOR), OUTLIVES_SERVER), "monitor skill does not say a job outlives the MCP server"


def test_monitor_terminal_row_process_gone() -> None:
    rows = [line for line in _section(MONITOR, "Terminal States").splitlines() if line.lstrip().startswith("|")]
    hits = [row for row in rows if PROCESS_GONE in row and "get_job_log" in row]
    assert hits, f"no Terminal States row names {PROCESS_GONE!r} with get_job_log"


def test_monitor_cancel_process_group() -> None:
    section = _section(MONITOR, "Cancellation")
    assert "cancel_job" in section and _has(section, r"process group"), "Cancellation does not name the process group"


# skills/debug/SKILL.md


def test_debug_names_process_gone_error() -> None:
    assert PROCESS_GONE in _flat(_read(DEBUG)), f"debug skill does not name {PROCESS_GONE!r}"


def test_debug_names_exit_code_error() -> None:
    assert EXIT_CODE_ERROR in _flat(_read(DEBUG)), f"debug skill does not name {EXIT_CODE_ERROR!r}"


def test_debug_process_gone_reads_log_first() -> None:
    chunk = _chunk_naming(_read(DEBUG), PROCESS_GONE)
    log = chunk.find("get_job_log")
    relaunch = re.search(r"re-?launch", chunk, re.IGNORECASE)
    assert log >= 0 and relaunch and log < relaunch.start(), f"get_job_log not before relaunch in: {chunk}"


# agents/champollion-pipeline.md


def test_agent_monitor_job_outlives_server() -> None:
    assert _has(_section(AGENT, "Step 3 — Monitor"), OUTLIVES_SERVER), (
        "Step 3 — Monitor does not say a job outlives the MCP server"
    )


def test_agent_debug_names_process_gone_error() -> None:
    assert PROCESS_GONE in _flat(_section(AGENT, "Debug Mode")), f"Debug Mode does not name {PROCESS_GONE!r}"


# CHANGELOG.md


def test_changelog_unreleased_heading() -> None:
    lines = [line.rstrip() for line in _read(CHANGELOG).splitlines()]
    assert UNRELEASED_HEADING in lines, f"CHANGELOG.md has no {UNRELEASED_HEADING!r} heading"
    assert lines.index(UNRELEASED_HEADING) < lines.index(RELEASE_HEADING), "Unreleased is not above 0.1.0"


def test_changelog_unreleased_job_outlives_server() -> None:
    assert _has(_unreleased_entry(), OUTLIVES_SERVER), "Unreleased does not say a job outlives the MCP server"


def test_changelog_unreleased_process_gone() -> None:
    assert PROCESS_GONE in _flat(_unreleased_entry()), f"Unreleased does not name {PROCESS_GONE!r}"


def test_changelog_unreleased_cancel_process_group() -> None:
    entry = _unreleased_entry()
    assert "cancel_job" in entry and _has(entry, r"process group"), "Unreleased does not name the process group"


def test_changelog_unreleased_pipeline_driver() -> None:
    entry = _unreleased_entry()
    assert "start_pipeline" in entry and _has(entry, DETACHED_DRIVER), "Unreleased does not name the detached driver"
