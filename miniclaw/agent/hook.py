"""Agent hook system for extensibility.

Based on miniclaw's design.
"""

from dataclasses import dataclass, field
from typing import Any


class AgentHookContext:
    """Context for agent hook calls."""

    def __init__(self, iteration: int, messages: list):
        self.iteration = iteration
        self.messages = messages
        self.response = None
        self.usage = None
        self.tool_calls = []
        self.tool_results = []
        self.tool_events = []
        self.final_content = None
        self.error = None
        self.stop_reason = None


class AgentHook:
    """Base class for agent hooks."""

    def wants_streaming(self) -> bool:
        """Return True if this hook wants streaming updates."""
        return False

    async def before_iteration(self, context: AgentHookContext) -> None:
        """Called before each iteration of the agent loop."""
        pass

    async def on_stream(self, context: AgentHookContext, delta: str) -> None:
        """Called when a streaming delta is received."""
        pass

    async def on_stream_end(self, context: AgentHookContext, resuming: bool) -> None:
        """Called when streaming ends."""
        pass

    async def before_execute_tools(self, context: AgentHookContext) -> None:
        """Called before executing tools."""
        pass

    async def after_iteration(self, context: AgentHookContext) -> None:
        """Called after each iteration of the agent loop."""
        pass

    def finalize_content(self, context: AgentHookContext, content: str) -> str:
        """Finalize the content before returning it."""
        return content or ""
