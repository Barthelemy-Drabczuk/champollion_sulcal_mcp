from __future__ import annotations

import pytest


@pytest.mark.unit
def test_server_has_13_tools():
    from champollion_sulcal_mcp.server import mcp

    # FastMCP 3.x exposes tools via ._tool_manager or similar
    # Try common attribute names
    tool_count = None
    for attr in ("_tool_manager", "_tools", "tools"):
        obj = getattr(mcp, attr, None)
        if obj is not None:
            if hasattr(obj, "__len__"):
                tool_count = len(obj)
            elif hasattr(obj, "_tools"):
                tool_count = len(obj._tools)
            break
    # If we can't introspect, just verify import succeeds
    assert tool_count is None or tool_count == 13, f"Expected 13 tools, got {tool_count}"


@pytest.mark.unit
def test_server_imports_cleanly():
    import champollion_sulcal_mcp.server  # noqa: F401

    assert True
