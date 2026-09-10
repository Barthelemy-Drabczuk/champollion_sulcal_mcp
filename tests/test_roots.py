from __future__ import annotations

import pytest

from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp.roots import get_roots, validate_within_roots


# --- validate_within_roots ---


@pytest.mark.unit
def test_validate_within_roots_no_roots_passes(tmp_path):
    validate_within_roots(str(tmp_path / "anywhere"), [], "param")


@pytest.mark.unit
def test_validate_within_roots_inside_root_passes(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    validate_within_roots(str(root / "subdir" / "file.txt"), [root], "param")


@pytest.mark.unit
def test_validate_within_roots_exact_root_passes(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    validate_within_roots(str(root), [root], "param")


@pytest.mark.unit
def test_validate_within_roots_outside_root_raises(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "other"
    outside.mkdir()
    with pytest.raises(ToolError, match="outside declared roots"):
        validate_within_roots(str(outside), [root], "output_dir")


@pytest.mark.unit
def test_validate_within_roots_multiple_roots_matches_second(tmp_path):
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()
    validate_within_roots(str(root_b / "subdir"), [root_a, root_b], "param")


@pytest.mark.unit
def test_validate_within_roots_error_message_names_roots(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    outside = tmp_path / "outside"
    with pytest.raises(ToolError, match=str(root)):
        validate_within_roots(str(outside), [root], "input_dir")


# --- get_roots ---


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_roots_no_ctx_returns_empty():
    result = await get_roots(None)
    assert result == []


@pytest.mark.unit
@pytest.mark.asyncio
async def test_get_roots_ctx_list_roots_exception_returns_empty():
    class BrokenCtx:
        async def list_roots(self):
            raise RuntimeError("no roots support")

    result = await get_roots(BrokenCtx())  # type: ignore[arg-type]
    assert result == []
