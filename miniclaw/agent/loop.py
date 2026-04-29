"""Agent loop for handling conversations and tool calls.

Based on miniclaw's design.
"""

from pathlib import Path

from miniclaw.agent.runner import AgentRunSpec, AgentRunner
from miniclaw.agent.skills import SkillsLoader


class AgentLoop:
    """Agent loop for handling conversations and tool calls."""

    _DEFAULT_CONTEXT_WINDOW_TOKENS = 128000

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

    def _resolve_context_window_tokens(self) -> int:
        """Resolve the context window budget from configured components."""
        provider_budget = getattr(self.provider, "context_window_tokens", None)
        if isinstance(provider_budget, int) and provider_budget > 0:
            return provider_budget

        consolidator_budget = getattr(self.consolidator, "context_window_tokens", None)
        if isinstance(consolidator_budget, int) and consolidator_budget > 0:
            return consolidator_budget

        return self._DEFAULT_CONTEXT_WINDOW_TOKENS

    @staticmethod
    def _merge_session_summaries(
        resumed_summary: str | None,
        rolling_summary: str | None,
    ) -> str | None:
        """Merge archived session summary and rolling in-session summary."""
        parts: list[str] = []
        if rolling_summary:
            parts.append("[Conversation Memory]\n" + rolling_summary)
        if resumed_summary:
            parts.append(resumed_summary)
        if not parts:
            return None
        return "\n\n".join(parts)

    async def process_message(self, message, session_key="default"):
        """Process a message and return a response."""
        session = self.session_manager.get_or_create(session_key)
        session_summary = None
        if self.autocompact:
            session, session_summary = self.autocompact.prepare_session(session, session_key)

        if self.consolidator and session.messages:
            await self.consolidator.maybe_consolidate_by_tokens(session)

        rolling_summary = session.metadata.get("_rolling_summary")
        combined_summary = self._merge_session_summaries(session_summary, rolling_summary)

        context = self.context_builder.build_messages(
            history=session.get_history(max_messages=0),
            current_message=message,
            channel="cli",
            chat_id=session_key,
            session_summary=combined_summary,
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
            context_window_tokens=self._resolve_context_window_tokens(),
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
