"""Tool registration of the champollion-sulcal MCP server.

REQ-MCPSTREAM-BDRABCZUK-B1F9F6EF027A: the server registers exactly 16 tools,
start_streaming not among them.
"""

from __future__ import annotations

import asyncio

import pytest

EXPECTED_TOOLS = {
    "start_morphologist",
    "start_cortical_tiles",
    "start_config",
    "start_training",
    "start_embeddings",
    "start_combine",
    "start_snapshots",
    "start_pipeline",
    "purge_subject",
    "prune_failed_subjects",
    "get_job_status",
    "list_jobs",
    "cancel_job",
    "get_job_log",
    "get_pipeline_info",
    "preflight_check",
}


def _registered_tool_names() -> list[str]:
    from champollion_sulcal_mcp.server import mcp

    return [tool.name for tool in asyncio.run(mcp.list_tools())]


@pytest.mark.unit
def test_server_registers_exactly_16_tools():
    """REQ-MCPSTREAM-BDRABCZUK-B1F9F6EF027A: the registered tool set equals the 16 named tools."""
    names = _registered_tool_names()
    assert len(names) == 16, f"Expected 16 tools, got {len(names)}: {sorted(names)}"
    assert set(names) == EXPECTED_TOOLS


@pytest.mark.unit
def test_server_registers_no_start_streaming():
    """REQ-MCPSTREAM-BDRABCZUK-B1F9F6EF027A: start_streaming is not a registered tool."""
    assert "start_streaming" not in _registered_tool_names()


@pytest.mark.unit
def test_server_imports_cleanly():
    import champollion_sulcal_mcp.server  # noqa: F401

    assert True
