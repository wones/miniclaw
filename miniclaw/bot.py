from pathlib import Path
import asyncio

import uvicorn

from miniclaw.agent.autocompact import AutoCompact
from miniclaw.agent.context import ContextBuilder
from miniclaw.agent.loop import AgentLoop
from miniclaw.agent.memory import Consolidator, Dream, MemoryStore
from miniclaw.agent.tools import setup_tools
from miniclaw.bus.queue import MessageBus
from miniclaw.config.loader import load_config
from miniclaw.providers.openai_compat_provider import OpenAICompatProvider
from miniclaw.server.webhook import app
from miniclaw.session.manager import SessionManager



class miniclaw:
    _active_sessions = set()

    def __init__(self, loop, context_builder: ContextBuilder):
        self.loop = loop
        self.context_builder = context_builder

    @classmethod
    def from_config(cls, config_path=None, workspace=None):
        config = load_config(config_path)
        api_key = config.providers.openai.get("apikey")
        env_baseurl = config.providers.openai.get("baseurl")
        env_model = config.providers.openai.get("model")

        if not api_key:
            raise ValueError("OpenAI API key is required")
        if not env_baseurl:
            raise ValueError("OpenAI base URL is required")
        if not env_model:
            raise ValueError("OpenAI model is required")
        if workspace is None:
            workspace = Path.home() / ".miniclaw"
        workspace.mkdir(parents=True, exist_ok=True)

        provider = OpenAICompatProvider(api_key, env_baseurl, env_model)
        session_manager = SessionManager(workspace=workspace)
        memory_store = MemoryStore(workspace)
        context_builder = ContextBuilder(workspace, timezone=None)
        context_builder.set_memory_store(memory_store)
        tool_registry = setup_tools(memory_store, workspace)
        consolidator = Consolidator(
            store=memory_store,
            provider=provider,
            model=env_model,
            sessions=session_manager,
            context_window_tokens=128000,
            max_completion_tokens=4096,
        )
        autocompact = AutoCompact(
            sessions=session_manager,
            consolidator=consolidator,
            session_ttl_minutes=30,
        )
        bus = MessageBus()
        loop = AgentLoop(
            bus,
            provider,
            session_manager,
            context_builder,
            consolidator,
            memory_store,
            tool_registry,
            autocompact=autocompact,
            workspace=workspace,
        )
        dream = Dream(
            store=memory_store,
            provider=provider,
            model=env_model,
            tools=tool_registry,
            runner=None,
        )

        async def start_scheduled_tasks():
            async def run_dream_periodically():
                while True:
                    await asyncio.sleep(3600)
                    print("Running Dream...")
                    try:
                        await dream.process_dream()
                    except Exception as e:
                        print(f"Dream failed: {e}")

            def get_active_session_keys():
                return cls._active_sessions if hasattr(cls, "_active_sessions") else set()

            async def run_autocompact_periodically():
                while True:
                    await asyncio.sleep(600)
                    print("Running AutoCompact...")
                    try:
                        autocompact.check_expired(
                            schedule_background=asyncio.create_task,
                            active_session_keys=get_active_session_keys(),
                        )
                    except Exception as e:
                        print(f"AutoCompact failed: {e}")

            asyncio.create_task(run_dream_periodically())
            asyncio.create_task(run_autocompact_periodically())

        asyncio.create_task(start_scheduled_tasks())
        return cls(loop, context_builder)

    async def run(self, message, session_key="default"):
        self._active_sessions.add(session_key)
        try:
            return await self.loop.process_message(message, session_key)
        finally:
            self._active_sessions.discard(session_key)

    @classmethod
    def run_server(cls, host="0.0.0.0", port=8765):
        uvicorn.run(app, host=host, port=port)
