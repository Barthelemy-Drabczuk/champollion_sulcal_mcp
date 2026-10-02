"""REQ-MCP-RUFF-01 / REQ-MCP-RUFF-02: src/ and tests/ are ruff format- and lint-clean.

Both checks run the pixi default environment's ruff from the repository root,
so the configuration in pyproject.toml ([tool.ruff]) is the one applied.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _ruff_binary() -> str | None:
    """Return the ruff next to the running interpreter (pixi env), else PATH."""
    candidate = Path(sys.executable).parent / "ruff"
    if candidate.is_file():
        return str(candidate)
    return shutil.which("ruff")


RUFF = _ruff_binary()

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(RUFF is None, reason="ruff binary not available"),
]


def _run_ruff(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [RUFF, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_src_and_tests_pass_ruff_format_check():
    """REQ-MCP-RUFF-01: `ruff format --check src tests` exits 0."""
    result = _run_ruff("format", "--check", "src", "tests")
    assert result.returncode == 0, f"ruff format --check src tests failed:\n{result.stdout}{result.stderr}"


def test_src_and_tests_pass_ruff_check():
    """REQ-MCP-RUFF-02: `ruff check src tests` exits 0."""
    result = _run_ruff("check", "src", "tests")
    assert result.returncode == 0, f"ruff check src tests failed:\n{result.stdout}{result.stderr}"
