"""Tool access control — config-driven capability model.

Replaces the old RBAC permission system with path resolution,
directory confinement, and access configuration used at tool
instantiation time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolAccessConfig:
    """Configuration for tool access control.

    Injected at tool instantiation time to control what the tool can access.
    """

    restrict_to_workspace: bool = False
    workspace: Path | None = None
    allowed_dir: Path | None = None
    extra_allowed_dirs: list[Path] = field(default_factory=list)
    deny_patterns: list[str] = field(default_factory=list)
    allow_patterns: list[str] = field(default_factory=list)


def _is_under(path: Path, directory: Path) -> bool:
    """Check if *path* is under *directory*."""
    try:
        path.relative_to(directory.resolve())
        return True
    except ValueError:
        return False


def _resolve_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
) -> Path:
    """Resolve *path* against workspace and enforce directory restriction.

    Raises :class:`PermissionError` if the resolved path falls outside
    the allowed directory tree.
    """
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()

    if allowed_dir:
        all_dirs = [allowed_dir] + (extra_allowed_dirs or [])
        if not any(_is_under(resolved, d) for d in all_dirs):
            raise PermissionError(
                f"Path {path} is outside allowed directory {allowed_dir}"
            )

    return resolved
