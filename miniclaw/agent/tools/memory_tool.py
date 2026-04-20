"""Memory tool for miniclaw — composite memory operations."""

from typing import Any

from miniclaw.agent.tools.base import Tool, tool_parameters
from miniclaw.agent.tools.schema import (
    ArraySchema,
    ObjectSchema,
    StringSchema,
    tool_parameters_schema,
)


@tool_parameters(
    tool_parameters_schema(
        action=StringSchema(
            "Memory action to perform",
            enum=["archive", "compact", "clear", "update"],
        ),
        messages=ArraySchema(
            items=ObjectSchema(),
            description="Messages to archive (for archive action)",
        ),
        file=StringSchema(
            "Which memory file to update (for update action)",
            enum=["memory", "soul", "user"],
        ),
        content=StringSchema(
            "Content to write to the memory file (for update action)"
        ),
        mode=StringSchema(
            "Write mode (for update action)",
            enum=["overwrite", "append"],
        ),
        required=["action"],
    )
)
class MemoryTool(Tool):
    """Tool for memory-related operations: archive, compact, clear, update."""

    def __init__(self, memory_store: Any) -> None:
        self.memory_store = memory_store

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Memory operations: archive, compact, clear, update."

    async def execute(
        self,
        action: str,
        messages: list[dict] | None = None,
        file: str | None = None,
        content: str | None = None,
        mode: str = "append",
        **kwargs: Any,
    ) -> str:
        try:
            if action == "archive":
                if not messages:
                    return "Error: Messages required for archive action"
                result = self.memory_store.raw_archive(messages)
                return f"Archived {len(messages)} messages: {result}"

            elif action == "update":
                if not file or not content:
                    return "Error: File and content required for update action"
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
                    return f"Error: Unknown file: {file}"
                return f"Successfully updated {file} file"

            elif action == "compact":
                self.memory_store.compact_history()
                return "History compacted successfully"

            elif action == "clear":
                self.memory_store.write_memory("")
                return "Memory cleared successfully"

            else:
                return f"Error: Unknown action: {action}"
        except Exception as e:
            return f"Error in memory operation: {e}"
