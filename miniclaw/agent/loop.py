"""Agent loop for handling conversations and tool calls.

Based on miniclaw's design.
"""

from miniclaw.session.manager import SessionManager
from miniclaw.providers.base import LLMProvider
from miniclaw.bus.queue import MessageBus
from miniclaw.agent.runner import AgentRunner, AgentRunSpec
from miniclaw.agent.skills import SkillsLoader
from pathlib import Path



class AgentLoop:
    """Agent loop for handling conversations and tool calls."""

    def __init__(self, bus, provider, session_manager, context_builder, consolidator, memory_store, tool_registry=None, tool_calling_strategy="react_prompt"):
        self.bus = bus
        self.provider = provider
        self.session_manager = session_manager
        self.context_builder = context_builder
        self.consolidator = consolidator
        self.memory_store = memory_store
        self.tool_registry = tool_registry
        self.tool_calling_strategy = tool_calling_strategy
        self.runner = AgentRunner(provider)
        self.skills_loader = SkillsLoader(Path("."))
        # 启动技能热加载
        self.skills_loader.watch_skills()

    async def process_message(self, message, session_key='default'):
        """Process a message and return a response."""
        session = self.session_manager.get_or_create(session_key)
        session.add_message("user", message)
        if self.consolidator and session.messages:
            await self.consolidator.maybe_consolidate_by_tokens(session)

        # Build context
        context = self.context_builder.build_messages(
            history=session.get_history(max_messages=50),
            current_message=message,
            channel="cli",
            chat_id=session_key
        )

        # Create run spec
        model = getattr(self.provider, "default_model", "gpt-4o")
        spec = AgentRunSpec(
            initial_messages=context,
            tools=self.tool_registry,
            model=model,
            max_iterations=10,
            max_tool_result_chars=10000,
            workspace=None,
            session_key=session_key,
            context_window_tokens=128000,
            tool_calling_strategy=self.tool_calling_strategy,
            skills_loader=self.skills_loader,
            enable_skills=["weather"],
            skill_summary=False,
            skill_priority=True
        )

        # Run the agent
        result = await self.runner.run(spec)

        # Add assistant message to session
        if result.final_content:
            session.add_message("assistant", result.final_content)
            self.memory_store.append_history(f"User: {message}\nAssistant: {result.final_content}")

        # Save session
        self.session_manager.save(session)

        return result.final_content
