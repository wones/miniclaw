"""Base classes for LLM providers.

Based on miniclaw's design.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from typing import List, Dict, Any, Optional


@dataclass
class ToolCallRequest:
    """Request to call a tool."""

    id: str
    name: str
    arguments: Dict[str, Any]

    def to_openai_tool_call(self) -> Dict[str, Any]:
        """Convert to OpenAI tool call format."""
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass
class LLMResponse:
    """LLM response."""

    content: Optional[str]
    tool_calls: List[ToolCallRequest]
    finish_reason: str
    usage: Optional[Dict[str, Any]]
    reasoning_content: Optional[str] = None
    thinking_blocks: Optional[List[Dict[str, Any]]] = None

    @property
    def has_tool_calls(self) -> bool:
        """Return True if the response has tool calls."""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> LLMResponse:
        """Chat with the LLM.
        
        Args:
            messages: List of messages
            tools: List of tool definitions
            
        Returns:
            LLMResponse object
        """
        pass

    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
    ) -> Any:
        """Generate a response from the LLM.
        
        Args:
            messages: List of messages
            tools: List of tool definitions
            
        Returns:
            Response content or dict with tool_calls
        """
        pass
