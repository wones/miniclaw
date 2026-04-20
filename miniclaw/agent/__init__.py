"""Agent module for miniclaw.

Based on miniclaw's design.
"""

from .loop import AgentLoop
from .memory import MemoryStore, Consolidator, Dream
from .context import ContextBuilder
from .autocompact import AutoCompact
from .runner import AgentRunner, AgentRunSpec, AgentRunResult
from .hook import AgentHook, AgentHookContext

__all__ = [
    "AgentLoop",
    "MemoryStore",
    "Consolidator",
    "Dream",
    "ContextBuilder",
    "AutoCompact",
    "AgentRunner",
    "AgentRunSpec",
    "AgentRunResult",
    "AgentHook",
    "AgentHookContext",
]
