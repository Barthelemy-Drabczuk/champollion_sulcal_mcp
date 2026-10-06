"""Roots checks on start_cortical_tiles subject sub-paths (TASK-042, REQ-MCP-COV-25, -35, -36).

path_to_graph and path_sk_with_hull are joined inside each subject folder, so under
declared roots they are checked by roots.validate_subject_subpath, not resolved
against the server's working directory.
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

from champollion_sulcal_mcp.tools import stages
from tests.test_stages_characterization import _GRAPH, _SK_HULL, _contains_run

# --- REQ-MCP-COV-25: relative subject sub-paths under declared roots ---


class TestCorticalTilesRelativeSubPathsUnderRoots:
    async def test_accepts_relative_graph_subpaths_under_roots(self, pipeline_dir, mock_roots, launches):
        """REQ-MCP-COV-25: with roots declared, relative path_to_graph / path_sk_with_hull
        sub-paths (as documented in skills/run-pipeline) are launched, not refused.
        """
        ctx, root = mock_roots
        result = await stages.start_cortical_tiles(
            input_dir=str(root / "subjects"),
            output_dir=str(root / "derivatives"),
            path_to_graph=_GRAPH,
            path_sk_with_hull=_SK_HULL,
            ctx=ctx,
        )
        assert result["stage"] == "cortical_tiles"
        assert _contains_run(launches[0]["argv"], ["--path_to_graph", _GRAPH])


# --- REQ-MCP-COV-35: relative sub-paths with a '..' segment refused under roots ---


class TestCorticalTilesDotDotSubPathsUnderRoots:
    @pytest.mark.parametrize("param", ["path_to_graph", "path_sk_with_hull"])
    async def test_refuses_dotdot_subpath_under_roots(self, pipeline_dir, mock_roots, launches, monkeypatch, param):
        """REQ-MCP-COV-35: with roots declared, a relative sub-path containing a '..'
        segment is refused with a ToolError and no job is launched.

        The server cwd is the declared root, so the '..' value still resolves inside
        it: only a '..'-segment rule (not a cwd-resolved roots check) refuses it.
        """
        ctx, root = mock_roots
        monkeypatch.chdir(root)
        kwargs = {"path_to_graph": _GRAPH, "path_sk_with_hull": _SK_HULL}
        kwargs[param] = "t1mri/../" + kwargs[param]
        with pytest.raises(ToolError, match=param):
            await stages.start_cortical_tiles(
                input_dir=str(root / "subjects"),
                output_dir=str(root / "derivatives"),
                ctx=ctx,
                **kwargs,
            )
        assert launches == []


# --- REQ-MCP-COV-36: absolute sub-paths outside declared roots still refused ---


class TestCorticalTilesAbsoluteSubPathsOutsideRoots:
    @pytest.mark.parametrize("param", ["path_to_graph", "path_sk_with_hull"])
    async def test_refuses_absolute_subpath_outside_roots(self, pipeline_dir, mock_roots, launches, tmp_path, param):
        """REQ-MCP-COV-36: with roots declared, an absolute sub-path outside the roots
        is refused with a ToolError and no job is launched.
        """
        ctx, root = mock_roots
        # Sibling sub-path is absolute and inside the root, so only `param` can be refused.
        kwargs = {"path_to_graph": str(root / _GRAPH), "path_sk_with_hull": str(root / _SK_HULL)}
        kwargs[param] = str(tmp_path / "elsewhere" / ("graph" if param == "path_to_graph" else "sk"))
        with pytest.raises(ToolError, match=param):
            await stages.start_cortical_tiles(
                input_dir=str(root / "subjects"),
                output_dir=str(root / "derivatives"),
                ctx=ctx,
                **kwargs,
            )
        assert launches == []
