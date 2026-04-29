"""Context builder for assembling agent prompts.

Based on miniclaw's design.
"""

from pathlib import Path
from typing import List, Optional, Any
from datetime import datetime

class ContextBuilder:
    """Builds the context (system prompt + messages) for the agent."""
    BOOTSTRAP_FILES = ["AGENTS.md","USER.md","SOUL.md","TOOLS.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context]"

    def __init__(self,workspace:Path,timezone:Optional[str]=None,disabled_skills:Optional[List[str]] = None):
        self.workspace = workspace
        self._timezone = timezone
        self.memory = None

    def set_memory_store(self,memory_store):
        self.memory = memory_store

    def build_system_prompt(self,channel:Optional[str]=None) -> str:
        """Build the system prompt from identity, bootstrap files, memory, and history."""
        parts = []
        # Add identity section
        parts.append(self._get_identity(channel=channel))
        # Load bootstrap files
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)
        # Add memory context
        if self.memory:
            memory_ctx = self.memory.get_memory_context()
            if memory_ctx:
                parts.append(memory_ctx)
        return "\n\n---\n\n".join(parts)
    
    def _get_identity(self,channel:Optional[str]=None) -> str:
        """Get the identity section of the context."""
        workspace_path = str(self.workspace.expanduser().resolve())
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "# Identity",
            f"- Workspace: {workspace_path}",
            f"- Current Time: {current_time}",
        ]
        if channel:
            lines.append(f"- Channel: {channel}")
        
        return "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """Load all bootstrap files from workspace."""
        parts = []
        
        for filename in self.BOOTSTRAP_FILES:
            content = self._load_file(filename)
            if content:
                parts.append(f"## {filename}\n\n{content}")
        
        return "\n\n".join(parts) if parts else ""

    def _load_file(self, filename: str) -> str:
        """Load a file from workspace if it exists."""
        file_path = self.workspace / filename
        if file_path.exists():
            try:
                return file_path.read_text(encoding="utf-8")
            except Exception:
                pass
        return ""

    @staticmethod
    def _build_runtime_context(
        channel: Optional[str],
        chat_id: Optional[str],
        session_summary: Optional[str] = None,
    ) -> str:
        """Build runtime metadata block for injection before user message."""
        lines = [f"Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"]
        if channel and chat_id:
            lines.extend([f"Channel: {channel}", f"Chat ID: {chat_id}"])
        if session_summary:
            lines.extend(["", "[Resumed Session]", session_summary])
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines) + "\n[/Runtime Context]"

    def build_messages(
        self,
        history: List[dict],
        current_message: str,
        channel: Optional[str] = None,
        chat_id: Optional[str] = None,
        current_role: str = "user",
        session_summary: Optional[str] = None,
    ) -> List[dict]:
        """Build the complete message list for an LLM call."""
        runtime_ctx = self._build_runtime_context(channel, chat_id, session_summary)
        
        # Merge runtime context with user message
        merged = f"{runtime_ctx}\n\n{current_message}"
        
        # Build messages
        messages = [
            {"role": "system", "content": self.build_system_prompt(channel=channel)},
            *history,
        ]
        
        # Append user message, merging with last if same role
        if messages and messages[-1].get("role") == current_role:
            last = dict(messages[-1])
            last["content"] = f"{last.get('content', '')}\n\n{merged}"
            messages[-1] = last
        else:
            messages.append({"role": current_role, "content": merged})

        return messages
    
    def add_tool_result(
        self,
        messages: List[dict],
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> List[dict]:
        """Add a tool result to the message list."""
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "name": tool_name,
            "content": str(result)
        })
        return messages
    
    def add_assistant_message(
        self,
        messages: List[dict],
        content: Optional[str],
        tool_calls: Optional[List[dict]] = None,
    ) -> List[dict]:
        """Add an assistant message to the message list."""
        msg = {"role": "assistant", "content": content or ""}
        if tool_calls:
            msg["tool_calls"] = tool_calls
        messages.append(msg)
        return messages
    
