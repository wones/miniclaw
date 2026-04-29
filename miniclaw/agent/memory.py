"""Memory system: MemoryStore (file I/O) and Consolidator (token-budget triggered).

Based on miniclaw design.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import tiktoken
import asyncio

class MemoryStore:
    """Pure file I/O for memory files: MEMORY.md, history.jsonl, SOUL.md, USER.md."""
    _DEFAULT_MAX_HISTORY = 1000
    def __init__(self,workspace:Path,max_history_entries: int = _DEFAULT_MAX_HISTORY):
        self.workspace = workspace
        self.max_history_entries = max_history_entries
        self.memory_dir = workspace / "memory"
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "history.jsonl"
        self.soul_file = workspace / "SOUL.md"
        self.user_file = workspace / "USER.md"
        self._cursor_file = self.memory_dir / ".cursor"
        self._dream_cursor_file = self.memory_dir / ".dream_cursor"
        self._ensure_directories()

    def _ensure_directories(self):
        self.memory_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def read_file(path:Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        
    def read_memory(self) -> str:
        return self.read_file(self.memory_file)
    
    def write_memory(self,content:str) -> None:
        self.memory_file.write_text(content,encoding="utf-8")

    def append_memory(self,new_content: str) -> None:
        existing = self.read_memory()
        content = existing + "\n\n" + new_content if existing else new_content
        self.write_memory(content)

    def read_soul(self) -> str:
        return self.read_file(self.soul_file)
    
    def write_soul(self,content:str) -> None:
        self.soul_file.write_text(content,encoding="utf-8")

    def read_user(self) -> str:
        return self.read_file(self.user_file)
    
    def write_user(self,content:str) -> None:
        self.user_file.write_text(content,encoding="utf-8")
    
    def get_memory_context(self) -> str:
        long_term = self.read_memory()
        return f"# Memory\n\n{long_term}" if long_term else ""
    
    def append_history(
        self,
        entry: str,
        session_key: str | None = None,
        kind: str = "dialogue",
    ) -> int:
        cursor = self._next_cursor()
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        record = {
            "cursor": cursor,
            "timestamp": ts,
            "content": entry.rstrip(),
            "kind": kind,
        }
        if session_key:
            record["session_key"] = session_key
        with open(self.history_file,"a",encoding="utf-8") as f:
            f.write(json.dumps(record,ensure_ascii=False) + "\n")
        self._cursor_file.write_text(str(cursor))
        return cursor

    def _next_cursor(self) -> int:
        if self._cursor_file.exists():
            try:
                return int(self._cursor_file.read_text(encoding='utf-8').strip())+1
            except (ValueError,OSError):
                pass
        last = self._read_last_entry()
        if last:
            return last["cursor"]+1
        return 1

    def read_unprocessed_history(self,since_cursor:int) -> List[Dict[str,Any]]:
        return [e for e in self._read_entries() if e["cursor"] > since_cursor]

    def compact_history(self) -> None:
        if self.max_history_entries <= 0:
            return
        entries = self._read_entries()
        if len(entries) <= self.max_history_entries:
            return
        kept = entries[-self.max_history_entries:]
        self._write_entries(kept)

    def _read_entries(self) -> List[Dict[str,Any]]:
        entries = []
        try:
            with open(self.history_file,"r",encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except FileNotFoundError:
            pass
        return entries

    def _read_last_entry(self) -> Optional[Dict[str,Any]]:
        try:
            with open(self.history_file,'rb') as f:
                f.seek(0,2)
                size = f.tell()
                if size == 0:
                    return None
                read_size = min(size,4096)
                f.seek(size - read_size)
                data = f.read().decode("utf-8")
                lines = [l for l in data.split("\n") if l.strip()]
                if not lines:
                    return None
                return json.loads(lines[-1])
        except (FileNotFoundError,json.JSONDecodeError,UnicodeDecodeError):
            return None

    def _write_entries(self,entries:List[Dict[str,Any]]) -> None:
        with open(self.history_file,"w",encoding="utf-8") as f:
            for entry in entries:
                f.write(json.dumps(entry,ensure_ascii=False) + "\n")

    def get_last_dream_cursor(self) -> int:
        """Get the last processed dream cursor."""
        if self._dream_cursor_file.exists():
            try:
                return int(self._dream_cursor_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
        return 0
    
    def set_last_dream_cursor(self, cursor: int) -> None:
        """Set the last processed dream cursor."""
        self._dream_cursor_file.write_text(str(cursor), encoding="utf-8")

    @staticmethod
    def _format_messages(messages:List[dict]) -> str:
        lines = []
        for message in messages:
            if not message.get("content"):
                continue
            lines.append(f"[{message.get('timestamp','?')[:16]}] {message['role'].upper()}: {message['content']}")
        return "\n".join(lines)

    def raw_archive(self,messages:List[dict], session_key: str | None = None) -> str:
        formatted = self._format_messages(messages)
        self.append_history(
            f"[RAW] {len(messages)} messages\n{formatted}",
            session_key=session_key,
            kind="raw_archive",
        )
        return f"Archived {len(messages)} messages (raw mode)"
       


# ---------------------------------------------------------------------------
# Consolidator — token-budget triggered consolidation
# ---------------------------------------------------------------------------
class Consolidator:
    """Lightweight consolidation: summarizes evicted messages into history.jsonl."""
    _MAX_CONSOLIDATION_ROUNDS = 5
    _MAX_CHUNK_MESSAGES = 60
    _SAFETY_BUFFER = 1024
    _ROLLING_SUMMARY_KEY = "_rolling_summary"
    _MAX_ROLLING_SUMMARY_CHARS = 4000
    _TOKENIZER_CACHE = None  # 类变量缓存编码器

    def __init__(
        self,
        store:MemoryStore,
        provider,
        model:str,
        sessions,
        context_window_tokens: int = 128000,
        max_completion_tokens: int = 4096
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self.sessions = sessions
        self.context_window_tokens = context_window_tokens
        self.max_completion_tokens = max_completion_tokens
        self._locks:Dict[str,asyncio.Lock] = {}
    
    def get_lock(self,session_key:str) -> Any:
        if session_key not in self._locks:
            import asyncio
            self._locks[session_key] = asyncio.Lock()
        return self._locks[session_key]

    def pick_consolidation_boundary(
        self,
        session,
        tokens_to_remove: int
    ) -> Optional[tuple[int,int]]:
        """Pick the boundary for consolidation."""
        start = getattr(session,'last_consolidated',0)
        if start >= len(session.messages) or tokens_to_remove <= 0:
            return None

        removed_tokens = 0
        last_boundary = None
        for idx in range(start,len(session.messages)):
            message = session.messages[idx]
            if idx > start and message.get("role") == "user":
                last_boundary = (idx, removed_tokens)
                if removed_tokens >= tokens_to_remove:
                    return last_boundary
            removed_tokens += self._estimate_message_tokens(message)
        
        return last_boundary

    def _get_tokenizer(self):
        """获取 tiktoken 编码器，使用缓存避免重复创建"""
        if Consolidator._TOKENIZER_CACHE is None:
            try:
                Consolidator._TOKENIZER_CACHE = tiktoken.get_encoding("cl100k_base")
            except Exception:
                return None
        return Consolidator._TOKENIZER_CACHE

    def _estimate_message_tokens(self,message:dict) -> int:
        """Estimate the token count for a message using tiktoken."""
        content = message.get("content","")
        if isinstance(content,str):
            tokenizer = self._get_tokenizer()
            if tokenizer:
                try:
                    return len(tokenizer.encode(content))
                except Exception:
                    pass
            return len(content) // 4
        elif isinstance(content,list):
            return sum(self._estimate_message_tokens(msg) for msg in content)
        return 1
    

    def _cap_consolidation_boundary(
        self,session,end_idx: int) -> Optional[int]:
        start = getattr(session,'last_consolidated',0)
        if end_idx - start <= self._MAX_CHUNK_MESSAGES:
            return end_idx
        
        capped_end = start + self._MAX_CHUNK_MESSAGES
        for idx in range(capped_end,start,-1):
            if session.messages[idx].get("role") == "user":
                return idx
        return None

    def estimate_session_prompt_tokens(self,session) -> tuple[int,str]:
        total_tokens = 0
        for msg in session.messages:
            total_tokens += self._estimate_message_tokens(msg)
        return total_tokens,"simple_estimate"

    @classmethod
    def _merge_summary_text(cls, existing: str | None, addition: str | None) -> str | None:
        """Merge rolling summaries while keeping the newest summary visible."""
        existing = (existing or "").strip()
        addition = (addition or "").strip()
        if not addition:
            return existing or None
        if not existing:
            merged = addition
        else:
            merged = f"{existing}\n\n{addition}"
        if len(merged) <= cls._MAX_ROLLING_SUMMARY_CHARS:
            return merged

        keep = max(256, cls._MAX_ROLLING_SUMMARY_CHARS // 2)
        return (
            merged[:keep]
            + "\n\n[... older rolling summary omitted ...]\n\n"
            + merged[-keep:]
        )

    async def archive(self,messages:list, session_key: str | None = None) -> Optional[str]:
        if not messages:
            return None
        try:
            formatted = MemoryStore._format_messages(messages)
            response = self.provider.generate(
                [
                {"role":"system","content":"请总结以下对话的要点，生成简洁的摘要，保留重要信息。"},
                {"role":"user","content":formatted}
                ],
                model=self.model,
            )
            summary = response if response else "[no summary]"
            self.store.append_history(summary, session_key=session_key, kind="summary")
            return summary
        except Exception as e:
            self.store.raw_archive(messages, session_key=session_key)
            return None
    
    async def maybe_consolidate_by_tokens(self,session) -> None:
        """Maybe consolidate the session."""
        if not session.messages or self.context_window_tokens <=0:
            return
        
        import asyncio
        lock = self.get_lock(session.key)
        async with lock:
            budget = self.context_window_tokens - self.max_completion_tokens - self._SAFETY_BUFFER 
            target = budget // 2

            try:
                estimated,source = self.estimate_session_prompt_tokens(session)
            except Exception as e:
                estimated,source = 0,"error"
            
            if estimated <= 0:
                return
            
            if estimated < budget:
                return
            
            for round_num in range(self._MAX_CONSOLIDATION_ROUNDS):
                if estimated <= target:
                    return
                boundary = self.pick_consolidation_boundary(session,max(1,estimated - target))
                if boundary is None:
                    return
                end_idx = boundary[0]
                end_idx = self._cap_consolidation_boundary(session,end_idx)
                if end_idx is None:
                    return  
                
                chunk = session.messages[getattr(session,'last_consolidated',0):end_idx]
                if not chunk:
                    return
                
                summary = await self.archive(chunk, session_key=session.key)
                if summary and summary != "[no summary]":
                    session.metadata[self._ROLLING_SUMMARY_KEY] = self._merge_summary_text(
                        session.metadata.get(self._ROLLING_SUMMARY_KEY),
                        summary,
                    )
                session.messages = session.messages[end_idx:]
                if hasattr(session,'last_consolidated'):
                    session.last_consolidated = 0
                estimated,_ = self.estimate_session_prompt_tokens(session)


class Dream:
    """
    - Phase 1 ：分析历史记录，识别重要信息
    - Phase 2 ：使用工具更新记忆文件（如 MEMORY.md）
    """
    _DEFAULT_MAX_ITERATIONS = 8
    _DEFAULT_MAX_TOOL_RESULT_CHARS = 4096

    def __init__(
        self,
        store: MemoryStore,
        provider,
        model: str,
        tools,
        runner,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        max_tool_result_chars: int = _DEFAULT_MAX_TOOL_RESULT_CHARS,
    ):
        self.store = store
        self.provider = provider
        self.model = model
        self._tools = tools
        self._runner = runner
        self.max_iterations = max_iterations
        self.max_tool_result_chars = max_tool_result_chars

    async def process_dream(self) -> bool:
        """处理 Dream 两阶段流程"""
        try:
            batch, cursor = await self._dream_phase1()
            if not batch:
                return False
            
            analysis = await self._analyze_batch(batch)
            if not analysis:
                self.store.set_last_dream_cursor(cursor)
                return False
            
            file_context = self._get_file_context()
            skills_section = self._get_skills_section()
        
            await self._dream_phase2(analysis, file_context, skills_section)
            self.store.set_last_dream_cursor(cursor)
            self.store.compact_history()
            return True
            
        except Exception as e:
            print(f"Dream processing failed: {e}")
            return False

    async def _dream_phase1(self) -> tuple[list[dict], int]:
        """第一阶段：分析历史记录"""
        last_cursor = self.store.get_last_dream_cursor()
        batch = self.store.read_unprocessed_history(last_cursor)
        
        if not batch:
            return [], last_cursor
        
        return batch, batch[-1]["cursor"]


    async def _analyze_batch(self, batch: list[dict]) -> str:
        """分析历史记录批次"""
        content = "\n\n".join([f"[{entry['timestamp']}] {entry['content']}" for entry in batch])
        
        try:
            response = self.provider.generate(
                [
                {"role": "system", "content": "请分析以下对话历史，识别需要提取到长期记忆的重要信息，包括事实、关系、事件等。"},
                {"role": "user", "content": content}
                ],
                model=self.model,
            )
            return response
        except Exception as e:
            print(f"Analysis failed: {e}")
            return ""
    
    def _get_file_context(self) -> str:
        """获取文件上下文"""
        memory = self.store.read_memory()
        soul = self.store.read_soul()
        user = self.store.read_user()
        
        context = []
        if memory:
            context.append(f"## Current Memory\n{memory[:1000]}...")
        if soul:
            context.append(f"## Current Soul\n{soul[:500]}...")
        if user:
            context.append(f"## Current User Info\n{user[:500]}...")
        
        return "\n\n".join(context)

    def _get_skills_section(self) -> str:
        """获取技能部分"""
        return "## Available Skills\n- 文件操作：读写记忆文件\n- 工具调用：执行各种操作"
    
    async def _dream_phase2(self, analysis: str, file_context: str, skills_section: str):
        """第二阶段：更新记忆文件"""
        changelog: list[str] = []
        try:
            prompt = f"""## Analysis Result\n{analysis}\n\n{file_context}\n\n{skills_section}\n\n
                请根据分析结果，更新相关记忆文件。请用以下JSON格式回复：
                {{"memory": "要添加到MEMORY.md的内容（如果没有则为空）",
                    "soul": "要更新到SOUL.md的内容（如果没有则为空）",
                    "user": "要更新到USER.md的内容（如果没有则为空）"}}"""
            
            response = self.provider.generate(
                [
                {"role": "system", "content": "你是一个记忆管理系统助手。请根据分析结果，用JSON格式提供需要更新的记忆内容。只返回JSON，不要其他内容。"},
                {"role": "user", "content": prompt}
                ],
                model=self.model,
            )
            
            # 这里可以根据响应更新记忆文件
            import json
            try:
                # 尝试提取JSON
                import re
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    updates = json.loads(json_match.group())
                else:
                    updates = json.loads(response)
            except (json.JSONDecodeError, Exception):
                updates = {"memory": response[:500] if response else ""}
            # 使用工具系统更新记忆文件
            if self._tools:
                # 更新 MEMORY.md
                if updates.get("memory"):
                    result = await self._tools.execute(
                        "memory",
                        {
                            "action": "update",
                            "file": "memory",
                            "content": updates["memory"],
                            "mode": "append"
                        }
                    )
                    if isinstance(result, str) and not result.startswith("Error"):
                        changelog.append(f"memory: 添加新记忆")
                
                # 更新 SOUL.md
                if updates.get("soul"):
                    result = await self._tools.execute(
                        "memory",
                        {
                            "action": "update",
                            "file": "soul",
                            "content": updates["soul"],
                            "mode": "overwrite"
                        }
                    )
                    if isinstance(result, str) and not result.startswith("Error"):
                        changelog.append("soul: 更新身份信息")
                
                # 更新 USER.md
                if updates.get("user"):
                    result = await self._tools.execute(
                        "memory",
                        {
                            "action": "update",
                            "file": "user",
                            "content": updates["user"],
                            "mode": "overwrite"
                        }
                    )
                    if isinstance(result, str) and not result.startswith("Error"):
                        changelog.append("user: 更新用户信息")
            else:
                # 降级到直接调用 MemoryStore
                # 更新 MEMORY.md
                if updates.get("memory"):
                    current_memory = self.store.read_memory()
                    new_memory = current_memory + "\n\n" + updates["memory"] if current_memory else updates["memory"]
                    self.store.write_memory(new_memory)
                    changelog.append(f"memory: 添加新记忆")
            
                # 更新 SOUL.md
                if updates.get("soul"):
                    self.store.write_soul(updates["soul"])
                    changelog.append("soul: 更新身份信息")
            
                # 更新 USER.md
                if updates.get("user"):
                    self.store.write_user(updates["user"])
                    changelog.append("user: 更新用户信息")
            
            print(f"Dream完成: {len(changelog)} 项更新")
            for change in changelog:
                print(f"  - {change}")
            
            return changelog
            
        except Exception as e:
            print(f"Dream Phase 2 failed: {e}")
            return []
