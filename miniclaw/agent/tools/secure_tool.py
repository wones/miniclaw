"""Secure tool implementations — file and memory tools with access control.

Based on miniclaw's filesystem.py design with _FsTool base class,
path resolution, and device path blocking.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from miniclaw.agent.tools.base import Tool, tool_parameters
from miniclaw.agent.tools.schema import StringSchema, tool_parameters_schema
from miniclaw.agent.tools.permission import _resolve_path


# ---------------------------------------------------------------------------
# Blocked device paths (prevent hanging on infinite-output devices)
# ---------------------------------------------------------------------------

_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/stdout", "/dev/stderr",
    "/dev/tty", "/dev/console",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})


def _is_blocked_device(path: str | Path) -> bool:
    """Check if path is a blocked device that could hang or produce infinite output."""
    raw = str(path)
    if raw in _BLOCKED_DEVICE_PATHS:
        return True
    if re.match(r"/proc/\d+/fd/[012]$", raw) or re.match(r"/proc/self/fd/[012]$", raw):
        return True
    return False


# ---------------------------------------------------------------------------
# Shared filesystem base
# ---------------------------------------------------------------------------

class _FsTool(Tool):
    """Shared base for filesystem tools — common init and path resolution."""

    def __init__(
        self,
        workspace: Path | None = None,
        allowed_dir: Path | None = None,
        extra_allowed_dirs: list[Path] | None = None,
    ):
        self._workspace = workspace
        self._allowed_dir = allowed_dir
        self._extra_allowed_dirs = extra_allowed_dirs or []

    def _resolve(self, path: str) -> Path:
        return _resolve_path(
            path, self._workspace, self._allowed_dir, self._extra_allowed_dirs
        )


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------

@tool_parameters(
    tool_parameters_schema(
        file=StringSchema(
            "Which memory file to read",
            enum=["memory", "soul", "user"],
        ),
        required=["file"],
    )
)
class ReadMemoryTool(Tool):
    """Read content from a memory file (MEMORY.md, SOUL.md, USER.md)."""

    def __init__(self, memory_store: Any) -> None:
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "read_memory"

    @property
    def description(self) -> str:
        return "Read content from a memory file (MEMORY.md, SOUL.md, USER.md)"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, file: str, **kwargs: Any) -> str:
        try:
            if file == "memory":
                content = self.memory_store.read_memory()
            elif file == "soul":
                content = self.memory_store.read_soul()
            elif file == "user":
                content = self.memory_store.read_user()
            else:
                return f"Error: Unknown memory file: {file}"
            return content if content else "(empty)"
        except Exception as e:
            return f"Error reading {file}: {e}"


@tool_parameters(
    tool_parameters_schema(
        file=StringSchema(
            "Which memory file to write",
            enum=["memory", "soul", "user"],
        ),
        content=StringSchema("Content to write"),
        mode=StringSchema(
            "Write mode: overwrite or append",
            enum=["overwrite", "append"],
        ),
        required=["file", "content"],
    )
)
class WriteMemoryTool(Tool):
    """Write content to a memory file (MEMORY.md, SOUL.md, USER.md)."""

    def __init__(self, memory_store: Any) -> None:
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "write_memory"

    @property
    def description(self) -> str:
        return "Write content to a memory file (MEMORY.md, SOUL.md, USER.md)"

    async def execute(
        self, file: str, content: str, mode: str = "append", **kwargs: Any
    ) -> str:
        try:
            if file == "memory":
                if mode == "append":
                    self.memory_store.append_memory(content)
                else:
                    self.memory_store.write_memory(content)
            elif file == "soul":
                self.memory_store.write_soul(content)
            elif file == "user":
                self.memory_store.write_user(content)
            else:
                return f"Error: Unknown memory file: {file}"
            return f"Successfully wrote to {file}"
        except Exception as e:
            return f"Error writing to {file}: {e}"


# ---------------------------------------------------------------------------
# File tools
# ---------------------------------------------------------------------------

@tool_parameters(
    tool_parameters_schema(
        path=StringSchema("File path to read (relative to workspace or absolute)"),
        required=["path"],
    )
)
class ReadFileTool(_FsTool):
    """Read file contents from workspace."""

    _MAX_CHARS = 128_000

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read content from a file in workspace"

    @property
    def read_only(self) -> bool:
        return True

    async def execute(self, path: str, **kwargs: Any) -> str:
        # Block dangerous device paths
        if _is_blocked_device(path):
            return f"Error: Cannot read blocked device path: {path}"

        try:
            file_path = self._resolve(path)
        except PermissionError as e:
            return f"Error: {e}"

        if not file_path.exists():
            return f"Error: File not found: {path}"
        if not file_path.is_file():
            return f"Error: Not a file: {path}"

        try:
            raw = file_path.read_bytes()
            # Simple binary detection
            if b"\x00" in raw[:8192]:
                return f"Error: File appears to be binary: {path}"
            content = raw.decode("utf-8", errors="replace")
        except Exception as e:
            return f"Error reading file: {e}"

        # Truncate if too large
        if len(content) > self._MAX_CHARS:
            half = self._MAX_CHARS // 2
            content = (
                content[:half]
                + f"\n\n... ({len(content) - self._MAX_CHARS:,} chars truncated) ...\n\n"
                + content[-half:]
            )

        # Add line numbers
        lines = content.splitlines()
        width = len(str(len(lines)))
        numbered = "\n".join(
            f"{i + 1:>{width}}| {line}" for i, line in enumerate(lines)
        )
        return numbered


@tool_parameters(
    tool_parameters_schema(
        path=StringSchema("File path to write (relative to workspace or absolute)"),
        content=StringSchema("Content to write to the file"),
        required=["path", "content"],
    )
)
class WriteFileTool(_FsTool):
    """Write content to a file in workspace."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file in workspace"

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            file_path = self._resolve(path)
        except PermissionError as e:
            return f"Error: {e}"

        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            byte_count = len(content.encode("utf-8"))
            return f"Successfully wrote {byte_count} bytes to {path}"
        except Exception as e:
            return f"Error writing file: {e}"
