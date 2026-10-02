from __future__ import annotations

from pathlib import Path

from fastmcp import Context
from fastmcp.exceptions import ToolError


async def get_roots(ctx: Context | None) -> list[Path]:
    """Return client-declared roots as resolved Paths. Empty list = no restriction."""
    if ctx is None:
        return []
    try:
        roots = await ctx.list_roots()
        return [Path(r.uri.removeprefix("file://")).resolve() for r in roots]
    except Exception:
        return []


def validate_within_roots(path: str, roots: list[Path], param_name: str) -> None:
    """Raise ToolError if path is not under any declared root (no-op when roots is empty)."""
    if not roots:
        return
    p = Path(path).resolve()
    for root in roots:
        try:
            p.relative_to(root)
            return
        except ValueError:
            continue
    raise ToolError(f"{param_name} {path!r} is outside declared roots: " + ", ".join(str(r) for r in roots))
