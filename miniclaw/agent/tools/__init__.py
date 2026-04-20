"""Tool system for miniclaw.

This module provides the tool system including:
- Tool base classes with schema-driven validation
- Tool registry for dynamic management
- Secure tool implementations (file, memory, shell)
- Access control via capability model
"""

from .base import Tool, Schema, tool_parameters
from .schema import (
    StringSchema,
    IntegerSchema,
    NumberSchema,
    BooleanSchema,
    ArraySchema,
    ObjectSchema,
    tool_parameters_schema,
)
from .registry import ToolRegistry
from .permission import ToolAccessConfig, _resolve_path
from .secure_tool import (
    ReadMemoryTool,
    WriteMemoryTool,
    ReadFileTool,
    WriteFileTool,
)
from .memory_tool import MemoryTool
from .exec_tool import ExecTool


def setup_tools(
    memory_store,
    workspace,
    restrict_to_workspace: bool = False,
) -> ToolRegistry:
    """Setup and register all tools.

    Args:
        memory_store: The memory store instance.
        workspace: Path to the workspace directory.
        restrict_to_workspace: If True, restrict file/exec access to workspace.
    """
    registry = ToolRegistry()

    allowed_dir = workspace if restrict_to_workspace else None

    # Create and register tools
    tools = [
        ReadMemoryTool(memory_store),
        WriteMemoryTool(memory_store),
        ReadFileTool(
            workspace=workspace,
            allowed_dir=allowed_dir,
        ),
        WriteFileTool(
            workspace=workspace,
            allowed_dir=allowed_dir,
        ),
        MemoryTool(memory_store),
        ExecTool(
            working_dir=str(workspace),
            restrict_to_workspace=True,
        ),
    ]

    for tool in tools:
        registry.register(tool)

    return registry


__all__ = [
    "Tool",
    "Schema",
    "tool_parameters",
    "StringSchema",
    "IntegerSchema",
    "NumberSchema",
    "BooleanSchema",
    "ArraySchema",
    "ObjectSchema",
    "tool_parameters_schema",
    "ToolRegistry",
    "ToolAccessConfig",
    "setup_tools",
    "ReadMemoryTool",
    "WriteMemoryTool",
    "ReadFileTool",
    "WriteFileTool",
    "MemoryTool",
    "ExecTool",
]
