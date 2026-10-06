"""Roots checks on start_streaming path arguments (TASK-058, REQ-MCP-STREAM-01..04).

Mirrors tests/test_stages_subject_subpaths.py (REQ-MCP-COV-25/35/36) for
start_streaming: path_to_graph and path_sk_with_hull are joined inside each subject
folder, so under declared roots they are checked as subject sub-paths, not resolved
against the server's working directory.
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp.tools import stages
from tests.test_stages_characterization import _GRAPH, _SK_HULL, _contains_run


def _base_kwargs(root) -> dict:
    return {
        "input_dir": str(root / "subjects"),
        "output_dir": str(root / "derivatives"),
        "dataset": "ds",
        "path_to_graph": _GRAPH,
        "path_sk_with_hull": _SK_HULL,
    }


# --- REQ-MCP-STREAM-01: relative sub-paths with a '..' segment refused under roots ---


class TestStreamingDotDotSubPathsUnderRoots:
    @pytest.mark.parametrize("param", ["path_to_graph", "path_sk_with_hull"])
    async def test_refuses_dotdot_subpath_under_roots(self, pipeline_dir, mock_roots, launches, monkeypatch, param):
        """REQ-MCP-STREAM-01: with roots declared, a relative sub-path containing a '..'
        segment is refused with a ToolError and no job is launched.

        The server cwd is the declared root, so the '..' value still resolves inside
        it: only a '..'-segment rule (not a cwd-resolved roots check) refuses it.
        """
        ctx, root = mock_roots
        monkeypatch.chdir(root)
        kwargs = _base_kwargs(root)
        kwargs[param] = "t1mri/../" + kwargs[param]
        with pytest.raises(ToolError, match=param):
            await stages.start_streaming(ctx=ctx, **kwargs)
        assert launches == []


# --- REQ-MCP-STREAM-02: absolute sub-paths outside declared roots refused ---


class TestStreamingAbsoluteSubPathsOutsideRoots:
    @pytest.mark.parametrize("param", ["path_to_graph", "path_sk_with_hull"])
    async def test_refuses_absolute_subpath_outside_roots(self, pipeline_dir, mock_roots, launches, tmp_path, param):
        """REQ-MCP-STREAM-02: with roots declared, an absolute sub-path outside the
        roots is refused with a ToolError and no job is launched.
        """
        ctx, root = mock_roots
        kwargs = _base_kwargs(root)
        # Sibling sub-path is absolute and inside the root, so only `param` can be refused.
        kwargs["path_to_graph"] = str(root / _GRAPH)
        kwargs["path_sk_with_hull"] = str(root / _SK_HULL)
        kwargs[param] = str(tmp_path / "elsewhere" / ("graph" if param == "path_to_graph" else "sk"))
        with pytest.raises(ToolError, match=param):
            await stages.start_streaming(ctx=ctx, **kwargs)
        assert launches == []


# --- REQ-MCP-STREAM-03: relative subject sub-paths under declared roots accepted ---


class TestStreamingRelativeSubPathsUnderRoots:
    async def test_accepts_relative_graph_subpaths_under_roots(self, pipeline_dir, mock_roots, launches):
        """REQ-MCP-STREAM-03: with roots declared, relative path_to_graph /
        path_sk_with_hull sub-paths are launched, not refused.
        """
        ctx, root = mock_roots
        result = await stages.start_streaming(ctx=ctx, **_base_kwargs(root))
        assert result["stage"] == "streaming"
        argv = launches[0]["argv"]
        assert _contains_run(argv, ["--path-to-graph", _GRAPH])
        assert _contains_run(argv, ["--path-sk-with-hull", _SK_HULL])


# --- REQ-MCP-STREAM-04: input_dir / output_dir outside declared roots refused ---


class TestStreamingDirsOutsideRoots:
    @pytest.mark.parametrize("param", ["input_dir", "output_dir"])
    async def test_refuses_dir_outside_roots(self, pipeline_dir, mock_roots, launches, tmp_path, param):
        """REQ-MCP-STREAM-04: with roots declared, an input_dir or output_dir outside
        the roots is refused with a ToolError and no job is launched.
        """
        ctx, root = mock_roots
        kwargs = _base_kwargs(root)
        kwargs[param] = str(tmp_path / "elsewhere" / param)
        with pytest.raises(ToolError, match=param):
            await stages.start_streaming(ctx=ctx, **kwargs)
        assert launches == []
