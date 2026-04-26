"""Shared execution loop for tool-using agents.

Based on miniclaw's design.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass, field
import inspect
from pathlib import Path
from typing import Any, List, Dict, Optional, Callable
import json
from loguru import logger

from miniclaw.agent.hook import AgentHook, AgentHookContext
from miniclaw.agent.tools.registry import ToolRegistry
from miniclaw.providers.base import LLMProvider, ToolCallRequest, LLMResponse
from miniclaw.skills import SkillsLoader

_DEFAULT_ERROR_MESSAGE = "Sorry, I encountered an error calling the AI model."
_PERSISTED_MODEL_ERROR_PLACEHOLDER = "[Assistant reply unavailable due to model error.]"
_MAX_EMPTY_RETRIES = 2
_MAX_LENGTH_RECOVERIES = 3
_MAX_INJECTIONS_PER_TURN = 3
_MAX_INJECTION_CYCLES = 5
_SNIP_SAFETY_BUFFER = 1024
_MICROCOMPACT_KEEP_RECENT = 10
_MICROCOMPACT_MIN_CHARS = 500
_COMPACTABLE_TOOLS = frozenset({
    "read_file", "exec", "grep", "glob",
    "web_search", "web_fetch", "list_dir",
})
_BACKFILL_CONTENT = "[Tool result unavailable — call was interrupted or lost]"


@dataclass(slots=True)
class AgentRunSpec:
    """Configuration for a single agent execution."""

    initial_messages: list[dict]
    tools: ToolRegistry
    model: str
    max_iterations: int
    max_tool_result_chars: int
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    reasoning_effort: Optional[str] = None
    hook: Optional[AgentHook] = None
    error_message: Optional[str] = _DEFAULT_ERROR_MESSAGE
    max_iterations_message: Optional[str] = None
    concurrent_tools: bool = False
    fail_on_tool_error: bool = False
    workspace: Optional[Path] = None
    session_key: Optional[str] = None
    context_window_tokens: Optional[int] = None
    context_block_limit: Optional[int] = None
    provider_retry_mode: str = "standard"
    progress_callback: Optional[Callable] = None
    checkpoint_callback: Optional[Callable] = None
    injection_callback: Optional[Callable] = None
    tool_calling_strategy: str = "function_calling"  # function_calling or react_prompt
    skills_loader: Optional[SkillsLoader] = None
    enable_skills: Optional[list[str]] = None

@dataclass(slots=True)
class AgentRunResult:
    """Outcome of a shared agent execution."""

    final_content: Optional[str]
    messages: list[dict]
    tools_used: list[str] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = "completed"
    error: Optional[str] = None
    tool_events: list[dict[str, str]] = field(default_factory=list)
    had_injections: bool = False


class AgentRunner:
    """Run a tool-capable LLM loop without product-layer concerns."""

    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self._tool_semaphore: asyncio.Semaphore | None = None

    def _get_tool_semaphore(self, max_concurrent: int = 5) -> asyncio.Semaphore:
        """Get or create a semaphore for tool concurrency control."""
        if self._tool_semaphore is None:
            self._tool_semaphore = asyncio.Semaphore(max_concurrent)
        return self._tool_semaphore

    def _get_tool_semaphore(self, max_concurrent: int = 10) -> asyncio.Semaphore:
        """Get or create a semaphore for tool concurrency control."""
        if self._tool_semaphore is None:
            self._tool_semaphore = asyncio.Semaphore(max_concurrent)
        return self._tool_semaphore

    @staticmethod
    def _merge_message_content(left: Any, right: Any) -> str | list[dict]:
        if isinstance(left, str) and isinstance(right, str):
            return f"{left}\n\n{right}" if left else right

        def _to_blocks(value: Any) -> list[dict]:
            if isinstance(value, list):
                return [
                    item if isinstance(item, dict) else {"type": "text", "text": str(item)}
                    for item in value
                ]
            if value is None:
                return []
            return [{"type": "text", "text": str(value)}]

        return _to_blocks(left) + _to_blocks(right)

    @classmethod
    def _append_injected_messages(
        cls,
        messages: list[dict],
        injections: list[dict],
    ) -> None:
        """Append injected user messages while preserving role alternation."""
        for injection in injections:
            if (
                messages
                and injection.get("role") == "user"
                and messages[-1].get("role") == "user"
            ):
                merged = dict(messages[-1])
                merged["content"] = cls._merge_message_content(
                    merged.get("content"),
                    injection.get("content"),
                )
                messages[-1] = merged
                continue
            messages.append(injection)

    async def _drain_injections(self, spec: AgentRunSpec) -> list[dict]:
        """Drain pending user messages via the injection callback."""
        if spec.injection_callback is None:
            return []
        try:
            signature = inspect.signature(spec.injection_callback)
            accepts_limit = (
                "limit" in signature.parameters
                or any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in signature.parameters.values()
                )
            )
            if accepts_limit:
                items = await spec.injection_callback(limit=_MAX_INJECTIONS_PER_TURN)
            else:
                items = await spec.injection_callback()
        except Exception:
            logger.exception("injection_callback failed")
            return []
        if not items:
            return []
        injected_messages: list[dict] = []
        for item in items:
            if isinstance(item, dict) and item.get("role") == "user" and "content" in item:
                injected_messages.append(item)
                continue
            text = getattr(item, "content", str(item))
            if text.strip():
                injected_messages.append({"role": "user", "content": text})
        if len(injected_messages) > _MAX_INJECTIONS_PER_TURN:
            dropped = len(injected_messages) - _MAX_INJECTIONS_PER_TURN
            logger.warning(
                "Injection callback returned {} messages, capping to {} ({} dropped)",
                len(injected_messages), _MAX_INJECTIONS_PER_TURN, dropped,
            )
            injected_messages = injected_messages[:_MAX_INJECTIONS_PER_TURN]
        return injected_messages

    def _build_react_prompt(self, messages: list[dict], tools: ToolRegistry) -> list[dict]:
        """Build ReAct prompt for models that don't support function calling."""
        tool_descriptions = []
        for tool_name in tools.tool_names:
            tool = tools.get(tool_name)
            if tool:
                params = tool.parameters or {}
                props = params.get("properties", {})
                required = params.get("required", [])
                param_info = []
                for pname, pschema in props.items():
                    ptype = pschema.get("type", "string")
                    pdesc = pschema.get("description", "")
                    req_mark = " (required)" if pname in required else ""
                    param_info.append(f"    {pname}: {ptype}{req_mark} — {pdesc}")
                param_str = "\n".join(param_info) if param_info else "    (no parameters)"
                tool_descriptions.append(
                    f"- {tool.name}: {tool.description}\n  Parameters:\n{param_str}"
                )
        # 保留原始系统提示中的重要信息
        original_system_prompt = ""
        if messages and messages[0].get("role") == "system":
            original_system_prompt = messages[0].get("content", "")
        react_prompt = f"""{original_system_prompt}
            可用工具：
            {chr(10).join(tool_descriptions)}

            调用格式：
            调用：{{"name":"工具名","parameters":{{"参数名":"参数值"}}}}

            重要：严格使用上述参数名称，不要使用其他名称。

            示例：
            调用：{{"name":"write_file","parameters":{{"path":"test.txt","content":"hello"}}}}
        """

        react_messages = [
            {"role": "system", "content": react_prompt}
        ]
        react_messages.extend(messages[1:])  # Skip the original system message

        return react_messages

    @staticmethod
    def _extract_balanced_json(text: str) -> Optional[str]:
        """Extract the first balanced JSON object from text."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                if in_string:
                    escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
        return None

    def _parse_react_tool_call(self, content: str) -> Optional[ToolCallRequest]:
        """Parse tool call from ReAct format using balanced-brace JSON extraction."""
        if content is None:
            return None
        match = re.search(r'调用[：:]\s*', content)
        if not match:
            return None

        json_str = self._extract_balanced_json(content[match.end() - 1:])
        if not json_str:
            return None

        try:
            tool_call_data = json.loads(json_str)
            if isinstance(tool_call_data, dict) and "name" in tool_call_data:
                return ToolCallRequest(
                    id=str(uuid.uuid4()),
                    name=tool_call_data["name"],
                    arguments=tool_call_data.get("parameters", {})
                )
        except Exception as e:
            logger.debug("Failed to parse ReAct tool call: {}", e)

        return None

    async def run(self, spec: AgentRunSpec) -> AgentRunResult:
        hook = spec.hook or AgentHook()
        messages = list(spec.initial_messages)
        final_content: Optional[str] = None
        tools_used: list[str] = []
        usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
        error: Optional[str] = None
        stop_reason = "completed"
        tool_events: list[dict[str, str]] = []
        external_lookup_counts: dict[str, int] = {}
        empty_content_retries = 0
        length_recovery_count = 0
        had_injections = False
        injection_cycles = 0
        #加载技能
        skills = [] 
        if spec.skills_loader:
            always_skills = spec.skills_loader.get_always_skills()
            skills.extend(always_skills)
            if spec.enable_skills:
                skills.extend([s for s in spec.enable_skills if s not in skills])
            
            if skills:
                skills_content = spec.skills_loader.load_skills_for_context(skills)
                if skills_content and messages and messages[0].get("role") == "system":
                    messages[0]["content"] = f"{messages[0]['content']}\n\n{skills_content}"
                    print(messages[0]["content"])
        for iteration in range(spec.max_iterations):
            try:
                # Context governance
                messages_for_model = self._drop_orphan_tool_results(messages)
                messages_for_model = self._backfill_missing_tool_results(messages_for_model)
                messages_for_model = self._microcompact(messages_for_model)
                messages_for_model = self._apply_tool_result_budget(spec, messages_for_model)
                messages_for_model = self._snip_history(spec, messages_for_model)
                messages_for_model = self._drop_orphan_tool_results(messages_for_model)
                messages_for_model = self._backfill_missing_tool_results(messages_for_model)
            except Exception as exc:
                logger.warning(
                    "Context governance failed on turn {} for {}: {}; applying minimal repair",
                    iteration,
                    spec.session_key or "default",
                    exc,
                )
                try:
                    messages_for_model = self._drop_orphan_tool_results(messages)
                    messages_for_model = self._backfill_missing_tool_results(messages_for_model)
                except Exception:
                    messages_for_model = messages
            
            context = AgentHookContext(iteration=iteration, messages=messages)
            await hook.before_iteration(context)
            
            # Get tools definitions
            tools_defs = spec.tools.get_definitions() if spec.tools else []
            
            # Handle tool calling strategy
            if spec.tool_calling_strategy == "react_prompt":
                # Use ReAct format for models that don't support function calling
                react_messages = self._build_react_prompt(messages_for_model, spec.tools)
                response = await self.provider.chat(react_messages, tools=tools_defs)
                
                # 增加空值检查
                if response is None:
                    logger.error("Provider returned None response")
                    # 处理错误，例如使用默认响应
                    response = LLMResponse(
                        content="Error: Model returned no response",
                        tool_calls=[],
                        finish_reason="error",
                        usage=None
                    )   
                else:
                    # Parse ReAct format
                    if response.content:
                        tool_call = self._parse_react_tool_call(response.content)
                        if tool_call:
                            # Create synthetic tool call response
                            synthetic_response = LLMResponse(
                                content=response.content,
                                tool_calls=[tool_call],
                                finish_reason="tool_calls",
                                usage=response.usage
                            )
                            response = synthetic_response
            else:
                # Use native function calling
                response = await self.provider.chat(messages_for_model, tools=tools_defs)
            
            raw_usage = self._usage_dict(response.usage)
            context.response = response
            context.usage = dict(raw_usage)
            context.tool_calls = list(response.tool_calls)
            self._accumulate_usage(usage, raw_usage)

            if response.has_tool_calls:
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=True)

                assistant_message = self._build_assistant_message(
                    response.content or "",
                    tool_calls=[tc.to_openai_tool_call() for tc in response.tool_calls],
                )
                messages.append(assistant_message)
                tools_used.extend(tc.name for tc in response.tool_calls)
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "awaiting_tools",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": [],
                        "pending_tool_calls": [tc.to_openai_tool_call() for tc in response.tool_calls],
                    },
                )

                await hook.before_execute_tools(context)

                results, new_events, fatal_error = await self._execute_tools(
                    spec,
                    response.tool_calls,
                    external_lookup_counts,
                )
                tool_events.extend(new_events)
                context.tool_results = list(results)
                context.tool_events = list(new_events)
                completed_tool_results: list[dict] = []
                for tool_call, result in zip(response.tool_calls, results):
                    content = self._normalize_tool_result(
                        spec, tool_call.id, tool_call.name, result
                    )
                    if not isinstance(content, str):
                        content = str(content)
                    tool_message = {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.name,
                        "content": content,
                    }
                    messages.append(tool_message)
                    completed_tool_results.append(tool_message)
                if fatal_error is not None:
                    error = f"Error: {type(fatal_error).__name__}: {fatal_error}"
                    final_content = error
                    stop_reason = "tool_error"
                    self._append_final_message(messages, final_content)
                    context.final_content = final_content
                    context.error = error
                    context.stop_reason = stop_reason
                    await hook.after_iteration(context)
                    break
                await self._emit_checkpoint(
                    spec,
                    {
                        "phase": "tools_completed",
                        "iteration": iteration,
                        "model": spec.model,
                        "assistant_message": assistant_message,
                        "completed_tool_results": completed_tool_results,
                        "pending_tool_calls": [],
                    },
                )
                empty_content_retries = 0
                length_recovery_count = 0
                
                # Checkpoint 1: drain injections after tools, before next LLM call
                if injection_cycles < _MAX_INJECTION_CYCLES:
                    injections = await self._drain_injections(spec)
                    if injections:
                        had_injections = True
                        injection_cycles += 1
                        self._append_injected_messages(messages, injections)
                        logger.info(
                            "Injected {} follow-up message(s) after tool execution ({}/{})",
                            len(injections), injection_cycles, _MAX_INJECTION_CYCLES,
                        )
                await hook.after_iteration(context)
                continue

            clean = hook.finalize_content(context, response.content)
            if response.finish_reason != "error" and self._is_blank_text(clean):
                empty_content_retries += 1
                if empty_content_retries < _MAX_EMPTY_RETRIES:
                    logger.warning(
                        "Empty response on turn {} for {} ({}/{}); retrying",
                        iteration,
                        spec.session_key or "default",
                        empty_content_retries,
                        _MAX_EMPTY_RETRIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=False)
                    await hook.after_iteration(context)
                    continue
                logger.warning(
                    "Empty response on turn {} for {} after {} retries; attempting finalization",
                    iteration,
                    spec.session_key or "default",
                    empty_content_retries,
                )
                if hook.wants_streaming():
                    await hook.on_stream_end(context, resuming=False)
                response = await self._request_finalization_retry(spec, messages_for_model)
                retry_usage = self._usage_dict(response.usage)
                self._accumulate_usage(usage, retry_usage)
                raw_usage = self._merge_usage(raw_usage, retry_usage)
                context.response = response
                context.usage = dict(raw_usage)
                context.tool_calls = list(response.tool_calls)
                clean = hook.finalize_content(context, response.content)

            if response.finish_reason == "length" and not self._is_blank_text(clean):
                length_recovery_count += 1
                if length_recovery_count <= _MAX_LENGTH_RECOVERIES:
                    logger.info(
                        "Output truncated on turn {} for {} ({}/{}); continuing",
                        iteration,
                        spec.session_key or "default",
                        length_recovery_count,
                        _MAX_LENGTH_RECOVERIES,
                    )
                    if hook.wants_streaming():
                        await hook.on_stream_end(context, resuming=True)
                    messages.append(self._build_assistant_message(
                        clean,
                    ))
                    messages.append({"role": "user", "content": "Please continue from where you left off."})
                    await hook.after_iteration(context)
                    continue

            assistant_message: Optional[dict] = None
            if response.finish_reason != "error" and not self._is_blank_text(clean):
                assistant_message = self._build_assistant_message(
                    clean,
                )

            # Check for mid-turn injections BEFORE signaling stream end
            _injected_after_final = False
            if injection_cycles < _MAX_INJECTION_CYCLES:
                injections = await self._drain_injections(spec)
                if injections:
                    had_injections = True
                    injection_cycles += 1
                    _injected_after_final = True
                    if assistant_message is not None:
                        messages.append(assistant_message)
                        await self._emit_checkpoint(
                            spec,
                            {
                                "phase": "final_response",
                                "iteration": iteration,
                                "model": spec.model,
                                "assistant_message": assistant_message,
                                "completed_tool_results": [],
                                "pending_tool_calls": [],
                            },
                        )
                    self._append_injected_messages(messages, injections)
                    logger.info(
                        "Injected {} follow-up message(s) after final response ({}/{})",
                        len(injections), injection_cycles, _MAX_INJECTION_CYCLES,
                    )

            if hook.wants_streaming():
                await hook.on_stream_end(context, resuming=_injected_after_final)

            if _injected_after_final:
                await hook.after_iteration(context)
                continue

            if response.finish_reason == "error":
                final_content = clean or spec.error_message or _DEFAULT_ERROR_MESSAGE
                stop_reason = "error"
                error = final_content
                self._append_model_error_placeholder(messages)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break
            if self._is_blank_text(clean):
                final_content = "I apologize, but I couldn't generate a response. Please try again."
                stop_reason = "empty_final_response"
                error = final_content
                self._append_final_message(messages, final_content)
                context.final_content = final_content
                context.error = error
                context.stop_reason = stop_reason
                await hook.after_iteration(context)
                break

            messages.append(assistant_message or self._build_assistant_message(
                clean,
            ))
            await self._emit_checkpoint(
                spec,
                {
                    "phase": "final_response",
                    "iteration": iteration,
                    "model": spec.model,
                    "assistant_message": messages[-1],
                    "completed_tool_results": [],
                    "pending_tool_calls": [],
                },
            )
            final_content = clean
            context.final_content = final_content
            context.stop_reason = stop_reason
            await hook.after_iteration(context)
            break
        else:
            stop_reason = "max_iterations"
            final_content = f"I've reached the maximum number of iterations ({spec.max_iterations}). Please try to be more specific with your request."
            self._append_final_message(messages, final_content)

        return AgentRunResult(
            final_content=final_content,
            messages=messages,
            tools_used=tools_used,
            usage=usage,
            stop_reason=stop_reason,
            error=error,
            tool_events=tool_events,
            had_injections=had_injections,
        )

    async def _request_model(
        self,
        spec: AgentRunSpec,
        messages: list[dict],
        hook: AgentHook,
        context: AgentHookContext,
    ) -> LLMResponse:
        """Request model response."""
        tools_defs = spec.tools.get_definitions() if spec.tools else []
        return await self.provider.chat(messages, tools=tools_defs)

    async def _request_finalization_retry(
        self,
        spec: AgentRunSpec,
        messages: list[dict],
    ) -> LLMResponse:
        """Request finalization retry."""
        retry_messages = list(messages)
        retry_messages.append({"role": "user", "content": "Please provide a complete response."})
        return await self.provider.chat(retry_messages, tools=None)

    @staticmethod
    def _usage_dict(usage: Optional[dict]) -> dict[str, int]:
        """Convert usage to dict."""
        if not usage:
            return {}
        result: dict[str, int] = {}
        for key, value in usage.items():
            try:
                result[key] = int(value or 0)
            except (TypeError, ValueError):
                continue
        return result

    @staticmethod
    def _accumulate_usage(target: dict[str, int], addition: dict[str, int]) -> None:
        """Accumulate usage."""
        for key, value in addition.items():
            target[key] = target.get(key, 0) + value

    @staticmethod
    def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        """Merge usage."""
        merged = dict(left)
        for key, value in right.items():
            merged[key] = merged.get(key, 0) + value
        return merged

    async def _execute_tools(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
        external_lookup_counts: dict[str, int],
    ) -> tuple[list[Any], list[dict[str, str]], Optional[BaseException]]:
        """Execute tools with optimized concurrency control."""
        batches = self._partition_tool_batches(spec, tool_calls)
        tool_results: list[tuple[Any, dict[str, str], Optional[BaseException]]] = []
        semaphore = self._get_tool_semaphore(max_concurrent=5)
        
        async def run_with_semaphore(tool_call: ToolCallRequest) -> tuple[Any, dict[str, str], Optional[BaseException]]:
            async with semaphore:
                return await self._run_tool(spec, tool_call, external_lookup_counts)
        
        for batch in batches:
            if spec.concurrent_tools and len(batch) > 1:
                results = await asyncio.gather(*(
                    run_with_semaphore(tool_call) for tool_call in batch
                ), return_exceptions=True)
                for result in results:
                    if isinstance(result, BaseException):
                        tool_results.append((
                            f"Error: {type(result).__name__}: {str(result)}",
                            {"name": "unknown", "status": "error", "detail": str(result)},
                            result
                        ))
                    else:
                        tool_results.append(result)
            else:
                for tool_call in batch:
                    tool_results.append(await run_with_semaphore(tool_call))

        results: list[Any] = []
        events: list[dict[str, str]] = []
        fatal_error: Optional[BaseException] = None
        for result, event, error in tool_results:
            results.append(result)
            events.append(event)
            if error is not None and fatal_error is None:
                fatal_error = error
        return results, events, fatal_error

    async def _run_tool(
        self,
        spec: AgentRunSpec,
        tool_call: ToolCallRequest,
        external_lookup_counts: dict[str, int],
    ) -> tuple[Any, dict[str, str], Optional[BaseException]]:
        """Run a single tool."""
        _HINT = "\n\n[Analyze the error above and try a different approach.]"
        lookup_error = self._repeated_external_lookup_error(
            tool_call.name,
            tool_call.arguments,
            external_lookup_counts,
        )
        if lookup_error:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": "repeated external lookup blocked",
            }
            if spec.fail_on_tool_error:
                return lookup_error + _HINT, event, RuntimeError(lookup_error)
            return lookup_error + _HINT, event, None
        
        try:
            result = await spec.tools.execute(tool_call.name, tool_call.arguments)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            event = {
                "name": tool_call.name,
                "status": "error",
                "detail": str(exc),
            }
            if spec.fail_on_tool_error:
                return f"Error: {type(exc).__name__}: {exc}", event, exc
            return f"Error: {type(exc).__name__}: {exc}", event, None

        detail = "" if result is None else str(result)
        detail = detail.replace("\n", " ").strip()
        if not detail:
            detail = "(empty)"
        elif len(detail) > 120:
            detail = detail[:120] + "..."
        return result, {"name": tool_call.name, "status": "ok", "detail": detail}, None

    async def _emit_checkpoint(
        self,
        spec: AgentRunSpec,
        payload: dict,
    ) -> None:
        """Emit checkpoint."""
        callback = spec.checkpoint_callback
        if callback is not None:
            await callback(payload)

    @staticmethod
    def _append_final_message(messages: list[dict], content: Optional[str]) -> None:
        """Append final message."""
        if not content:
            return
        if (
            messages
            and messages[-1].get("role") == "assistant"
            and not messages[-1].get("tool_calls")
        ):
            if messages[-1].get("content") == content:
                return
            messages[-1] = AgentRunner._build_assistant_message(content)
            return
        messages.append(AgentRunner._build_assistant_message(content))

    @staticmethod
    def _append_model_error_placeholder(messages: list[dict]) -> None:
        """Append model error placeholder."""
        if messages and messages[-1].get("role") == "assistant" and not messages[-1].get("tool_calls"):
            return
        messages.append(AgentRunner._build_assistant_message(_PERSISTED_MODEL_ERROR_PLACEHOLDER))

    
    def _normalize_tool_result(
        self,
        spec: AgentRunSpec,
        tool_call_id: str,
        tool_name: str,
        result: Any,
    ) -> Any:
        """Normalize tool result."""
        result = self._ensure_nonempty_tool_result(tool_name, result)
        content = result if isinstance(result, str) else str(result)
        try:
            content = self._maybe_persist_tool_result(
                spec.workspace,
                spec.session_key,
                tool_call_id,
                content,
                max_chars=spec.max_tool_result_chars,
            )
        except Exception as exc:
            logger.warning(
                "Tool result persist failed for {} in {}: {}; using raw result",
                tool_call_id,
                spec.session_key or "default",
                exc,
            )
            content = result
        if isinstance(content, str) and len(content) > spec.max_tool_result_chars:
            return self._truncate_text(content, spec.max_tool_result_chars)
        return content

    @staticmethod
    def _drop_orphan_tool_results(
        messages: list[dict],
    ) -> list[dict]:
        """Drop tool results that have no matching assistant tool_call."""
        declared: set[str] = set()
        updated: Optional[list[dict]] = None
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        declared.add(str(tc["id"]))
            if role == "tool":
                tid = msg.get("tool_call_id")
                if tid and str(tid) not in declared:
                    if updated is None:
                        updated = [dict(m) for m in messages[:idx]]
                    continue
            if updated is not None:
                updated.append(dict(msg))

        if updated is None:
            return messages
        return updated

    @staticmethod
    def _backfill_missing_tool_results(
        messages: list[dict],
    ) -> list[dict]:
        """Insert synthetic error results for orphaned tool_use blocks."""
        declared: list[tuple[int, str, str]] = []  # (assistant_idx, call_id, name)
        fulfilled: set[str] = set()
        for idx, msg in enumerate(messages):
            role = msg.get("role")
            if role == "assistant":
                for tc in msg.get("tool_calls") or []:
                    if isinstance(tc, dict) and tc.get("id"):
                        name = ""
                        func = tc.get("function")
                        if isinstance(func, dict):
                            name = func.get("name", "")
                        declared.append((idx, str(tc["id"]), name))
            elif role == "tool":
                tid = msg.get("tool_call_id")
                if tid:
                    fulfilled.add(str(tid))

        missing = [(ai, cid, name) for ai, cid, name in declared if cid not in fulfilled]
        if not missing:
            return messages

        updated = list(messages)
        offset = 0
        for assistant_idx, call_id, name in missing:
            insert_at = assistant_idx + 1 + offset
            while insert_at < len(updated) and updated[insert_at].get("role") == "tool":
                insert_at += 1
            updated.insert(insert_at, {
                "role": "tool",
                "tool_call_id": call_id,
                "name": name,
                "content": _BACKFILL_CONTENT,
            })
            offset += 1
        return updated

    @staticmethod
    def _microcompact(messages: list[dict]) -> list[dict]:
        """Replace old compactable tool results with one-line summaries."""
        compactable_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.get("role") == "tool" and msg.get("name") in _COMPACTABLE_TOOLS:
                compactable_indices.append(idx)

        if len(compactable_indices) <= _MICROCOMPACT_KEEP_RECENT:
            return messages

        stale = compactable_indices[: len(compactable_indices) - _MICROCOMPACT_KEEP_RECENT]
        updated: Optional[list[dict]] = None
        for idx in stale:
            msg = messages[idx]
            content = msg.get("content")
            if not isinstance(content, str) or len(content) < _MICROCOMPACT_MIN_CHARS:
                continue
            name = msg.get("name", "tool")
            summary = f"[{name} result omitted from context]"
            if updated is None:
                updated = [dict(m) for m in messages]
            updated[idx]["content"] = summary

        return updated if updated is not None else messages

    def _apply_tool_result_budget(
        self,
        spec: AgentRunSpec,
        messages: list[dict],
    ) -> list[dict]:
        """Apply tool result budget."""
        updated = messages
        for idx, message in enumerate(messages):
            if message.get("role") != "tool":
                continue
            normalized = self._normalize_tool_result(
                spec,
                str(message.get("tool_call_id") or f"tool_{idx}"),
                str(message.get("name") or "tool"),
                message.get("content"),
            )
            if normalized != message.get("content"):
                if updated is messages:
                    updated = [dict(m) for m in messages]
                updated[idx]["content"] = normalized
        return updated

    def _snip_history(
        self,
        spec: AgentRunSpec,
        messages: list[dict],
    ) -> list[dict]:
        """Snip history to fit context window."""
        if not messages or not spec.context_window_tokens:
            return messages

        provider_max_tokens = getattr(getattr(self.provider, "generation", None), "max_tokens", 4096)
        max_output = spec.max_tokens if isinstance(spec.max_tokens, int) else (
            provider_max_tokens if isinstance(provider_max_tokens, int) else 4096
        )
        budget = spec.context_block_limit or (
            spec.context_window_tokens - max_output - _SNIP_SAFETY_BUFFER
        )
        if budget <= 0:
            return messages

        estimate = self._estimate_prompt_tokens_chain(
            self.provider,
            spec.model,
            messages,
            spec.tools.get_definitions() if spec.tools else [],
        )
        if estimate <= budget:
            return messages

        system_messages = [dict(msg) for msg in messages if msg.get("role") == "system"]
        non_system = [dict(msg) for msg in messages if msg.get("role") != "system"]
        if not non_system:
            return messages

        system_tokens = sum(self._estimate_message_tokens(msg) for msg in system_messages)
        remaining_budget = max(128, budget - system_tokens)
        kept: list[dict] = []
        kept_tokens = 0
        for message in reversed(non_system):
            msg_tokens = self._estimate_message_tokens(message)
            if kept and kept_tokens + msg_tokens > remaining_budget:
                break
            kept.append(message)
            kept_tokens += msg_tokens
        kept.reverse()

        if kept:
            for i, message in enumerate(kept):
                if message.get("role") == "user":
                    kept = kept[i:]
                    break
            start = self._find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        if not kept:
            kept = non_system[-min(len(non_system), 4) :]
            start = self._find_legal_message_start(kept)
            if start:
                kept = kept[start:]
        return system_messages + kept

    def _partition_tool_batches(
        self,
        spec: AgentRunSpec,
        tool_calls: list[ToolCallRequest],
    ) -> list[list[ToolCallRequest]]:
        """Partition tool calls into batches."""
        if not spec.concurrent_tools:
            return [[tool_call] for tool_call in tool_calls]

        batches: list[list[ToolCallRequest]] = []
        current: list[ToolCallRequest] = []
        for tool_call in tool_calls:
            get_tool = getattr(spec.tools, "get", None)
            tool = get_tool(tool_call.name) if callable(get_tool) else None
            can_batch = bool(tool and getattr(tool, "concurrency_safe", False))
            if can_batch:
                current.append(tool_call)
                continue
            if current:
                batches.append(current)
                current = []
            batches.append([tool_call])
        if current:
            batches.append(current)
        return batches

    @staticmethod
    def _build_assistant_message(content: str, tool_calls: Optional[list[dict]] = None) -> dict:
        """Build an assistant message."""
        message = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return message

    @staticmethod
    def _is_blank_text(text: Optional[str]) -> bool:
        """Check if text is blank."""
        return not text or not text.strip()

    @staticmethod
    def _ensure_nonempty_tool_result(tool_name: str, result: Any) -> Any:
        """Ensure tool result is not empty."""
        if result is None:
            return f"No result from {tool_name}"
        if isinstance(result, str) and not result.strip():
            return f"No result from {tool_name}"
        return result

    @staticmethod
    def _maybe_persist_tool_result(workspace: Optional[Path], session_key: Optional[str], tool_call_id: str, result: Any, max_chars: int) -> Any:
        """Maybe persist tool result to file."""
        return result

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        """Truncate text to max_chars."""
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "..."

    @staticmethod
    def _estimate_prompt_tokens_chain(provider: LLMProvider, model: str, messages: list[dict], tools: list[dict]) -> int:
        """Estimate prompt tokens."""
        return 0

    @staticmethod
    def _estimate_message_tokens(message: dict) -> int:
        """Estimate message tokens."""
        return len(str(message)) // 4

    @staticmethod
    def _find_legal_message_start(messages: list[dict]) -> Optional[int]:
        """Find legal message start."""
        for i, message in enumerate(messages):
            if message.get("role") == "user":
                return i
        return None

    @staticmethod
    def _repeated_external_lookup_error(tool_name: str, arguments: dict, external_lookup_counts: dict[str, int]) -> Optional[str]:
        """Check for repeated external lookup."""
        return None
