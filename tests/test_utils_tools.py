from __future__ import annotations

import pytest

from champollion_sulcal_mcp.tools.utils import get_pipeline_info, preflight_check, TOOL_NAMES, STAGE_DESCRIPTIONS


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


@pytest.mark.unit
async def test_get_pipeline_info_tools_complete():
    result = await get_pipeline_info()
    assert set(result["tools"]) == set(TOOL_NAMES)
    assert len(result["tools"]) == 13


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
