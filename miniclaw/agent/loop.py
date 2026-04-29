"""Agent loop for handling conversations and tool calls.

Based on miniclaw's design.
"""

from pathlib import Path

from miniclaw.agent.runner import AgentRunSpec, AgentRunner
from miniclaw.agent.skills import SkillsLoader


class AgentLoop:
    """Agent loop for handling conversations and tool calls."""

    def __init__(
        self,
        bus,
        provider,
        session_manager,
        context_builder,
        consolidator,
        memory_store,
        tool_registry=None,
        tool_calling_strategy="react_prompt",
        autocompact=None,
        workspace: Path | None = None,
    ):
        self.bus = bus
        self.provider = provider
        self.session_manager = session_manager
        self.context_builder = context_builder
        self.consolidator = consolidator
        self.memory_store = memory_store
        self.tool_registry = tool_registry
        self.tool_calling_strategy = tool_calling_strategy
        self.autocompact = autocompact
        self.workspace = workspace
        self.runner = AgentRunner(provider)
        self.skills_loader = SkillsLoader(workspace or Path("."))
        self.skills_loader.watch_skills()

    async def process_message(self, message, session_key="default"):
        """Process a message and return a response."""
        session = self.session_manager.get_or_create(session_key)
        session_summary = None
        if self.autocompact:
            session, session_summary = self.autocompact.prepare_session(session, session_key)

        if self.consolidator and session.messages:
            await self.consolidator.maybe_consolidate_by_tokens(session)

        context = self.context_builder.build_messages(
            history=session.get_history(max_messages=0),
            current_message=message,
            channel="cli",
            chat_id=session_key,
            session_summary=session_summary,
        )

        model = getattr(self.provider, "default_model", "gpt-4o")
        spec = AgentRunSpec(
            initial_messages=context,
            tools=self.tool_registry,
            model=model,
            max_iterations=10,
            max_tool_result_chars=10000,
            workspace=self.workspace,
            session_key=session_key,
            context_window_tokens=128000,
            tool_calling_strategy=self.tool_calling_strategy,
            skills_loader=self.skills_loader,
            enable_skills=["weather"],
            skill_summary=False,
            skill_priority=True,
        )

        result = await self.runner.run(spec)

        session.add_message("user", message)
        if result.final_content:
            session.add_message("assistant", result.final_content)
            self.memory_store.append_history(
                f"User: {message}\nAssistant: {result.final_content}",
                session_key=session_key,
                kind="dialogue",
            )

        self.session_manager.save(session)
        return result.final_content
