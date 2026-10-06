"""REQ-MCP-COV-01: the coverage run enforces an 85 percent floor.

`pixi run test-cov` runs pytest-cov over the champollion_sulcal_mcp package;
pytest-cov fails the run when total coverage is below
`[tool.coverage.report] fail_under` from pyproject.toml.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.unit


def test_pyproject_sets_coverage_fail_under_85():
    """REQ-MCP-COV-01: pyproject.toml sets `[tool.coverage.report] fail_under = 85`."""
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    report = config.get("tool", {}).get("coverage", {}).get("report", {})
    assert report.get("fail_under") == 85, f"[tool.coverage.report] fail_under is {report.get('fail_under')!r}"


def test_test_cov_task_measures_champollion_sulcal_mcp():
    """REQ-MCP-COV-01: `pixi run test-cov` measures the champollion_sulcal_mcp package."""
    tasks = tomllib.loads((REPO_ROOT / "pixi.toml").read_text())["tasks"]
    test_cov = tasks["test-cov"]
    command = test_cov if isinstance(test_cov, str) else test_cov["cmd"]
    assert "--cov=champollion_sulcal_mcp" in command
