"""The MCP server's root .gitignore must ignore a root .gitattributes file.

REQ-MCPCHORE-BDRABCZUK-624A8DCA0194: the root ``.gitignore`` file of
champollion_sulcal_mcp shall contain a pattern that makes git ignore the
``.gitattributes`` file at the root of champollion_sulcal_mcp.

alm 6 writes ``.alm/*.jsonl merge=union`` into a root ``.gitattributes``
for an in-repo ``.alm/``. Here ``.alm/`` is its own git repository, which
holds that rule itself, so the root file must stay out of the code repo.

``git check-ignore`` on the live repository would also honour
``$GIT_DIR/info/exclude`` and ``core.excludesFile``, so a local-only
exclude could satisfy it. Instead, the root ``.gitignore`` alone is copied
into a scratch repository (global excludes disabled) and real git pattern
matching is run there.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

MCP_ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.unit


def _assert_gitignore_ignores_root_gitattributes(gitignore: Path, tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git executable not available")
    assert gitignore.is_file(), f"{gitignore} does not exist"

    scratch = tmp_path / "repo"
    scratch.mkdir()
    subprocess.run(["git", "init", "-q", str(scratch)], check=True)
    shutil.copyfile(gitignore, scratch / ".gitignore")
    (scratch / ".gitattributes").write_text(".alm/*.jsonl merge=union\n")

    result = subprocess.run(
        [
            "git",
            "-c",
            f"core.excludesFile={tmp_path / 'no-global-excludes'}",
            "-C",
            str(scratch),
            "check-ignore",
            "-q",
            ".gitattributes",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"{gitignore} has no pattern ignoring the root .gitattributes file "
        f"(git check-ignore rc={result.returncode}, "
        f"stderr={result.stderr.strip()!r})"
    )


def test_mcp_gitignore_ignores_root_gitattributes(tmp_path):
    """REQ-MCPCHORE-BDRABCZUK-624A8DCA0194: root .gitignore ignores /.gitattributes."""
    _assert_gitignore_ignores_root_gitattributes(MCP_ROOT / ".gitignore", tmp_path)
