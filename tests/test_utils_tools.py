from __future__ import annotations

import pytest

from champollion_sulcal_mcp.tools.utils import get_pipeline_info, preflight_check


@pytest.mark.unit
async def test_get_pipeline_info_shape():
    result = await get_pipeline_info()
    assert "version" in result
    assert "stages" in result
    assert "tools" in result
    assert "transport" in result


@pytest.mark.unit
async def test_get_pipeline_info_stages_complete():
    result = await get_pipeline_info()
    expected_stages = {"morphologist", "cortical_tiles", "config", "embeddings", "combine", "snapshots"}
    assert set(result["stages"].keys()) == expected_stages


async def _registered_tool_names() -> set[str]:
    from champollion_sulcal_mcp.server import mcp

    return {tool.name for tool in await mcp.list_tools()}


@pytest.mark.unit
async def test_get_pipeline_info_tools_complete():
    """REQ-MODELVER-BDRABCZUK-6859789716FD: tools equals the server's registered tool set."""
    result = await get_pipeline_info()
    registered = await _registered_tool_names()
    assert set(result["tools"]) == registered, (
        f"missing {sorted(registered - set(result['tools']))}, extra {sorted(set(result['tools']) - registered)}"
    )
    assert len(result["tools"]) == len(registered)


@pytest.mark.unit
async def test_get_pipeline_info_tools_follow_a_newly_registered_tool():
    """REQ-MODELVER-BDRABCZUK-6859789716FD: a tool registered later appears without editing tools/utils.py."""
    from champollion_sulcal_mcp.server import mcp

    async def alm_probe_extra_tool() -> dict:
        """Probe tool registered only for this test."""
        return {}

    mcp.tool(alm_probe_extra_tool)
    try:
        result = await get_pipeline_info()
        assert "alm_probe_extra_tool" in result["tools"]
        assert set(result["tools"]) == await _registered_tool_names()
    finally:
        mcp.local_provider.remove_tool("alm_probe_extra_tool")


@pytest.mark.unit
async def test_get_pipeline_info_transport_is_stdio():
    result = await get_pipeline_info()
    assert result["transport"] == "stdio"


@pytest.mark.unit
async def test_preflight_check_missing_pipeline_ok_false(monkeypatch):
    from champollion_sulcal_mcp import preflight

    def fake_summarize():
        raise FileNotFoundError("no pipeline dir")

    monkeypatch.setattr(preflight, "summarize", fake_summarize)

    result = await preflight_check()
    assert result["ok"] is False
    assert "error" in result


@pytest.mark.unit
async def test_preflight_check_returns_dict_not_exception(monkeypatch):
    from champollion_sulcal_mcp import preflight

    monkeypatch.setattr(preflight, "summarize", lambda: {"ok": False, "error": "missing"})
    result = await preflight_check()
    assert isinstance(result, dict)


@pytest.mark.unit
async def test_preflight_check_with_fake_pipeline(fake_pipeline_dir):
    result = await preflight_check()
    assert isinstance(result, dict)
    assert "ok" in result
    assert "scripts" in result
    assert result["ok"] is True
