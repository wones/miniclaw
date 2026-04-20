"""AutoCompact: proactive compression of idle sessions to reduce token cost.

Based on miniclaw's design.
"""

from collections.abc import Collection
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable, Coroutine, Optional

if TYPE_CHECKING:
    from miniclaw.agent.memory import Consolidator
    from miniclaw.session.manager import Session, SessionManager


class AutoCompact:
    """Proactive compression of idle sessions."""
    
    _RECENT_SUFFIX_MESSAGES = 8
    
    def __init__(
        self,
        sessions: "SessionManager",
        consolidator: "Consolidator",
        session_ttl_minutes: int = 0,
    ):
        self.sessions = sessions
        self.consolidator = consolidator
        self._ttl = session_ttl_minutes
        self._archiving: set = set()
        self._summaries: dict = {}
    
    def _is_expired(self, ts: Optional[datetime | str], now: Optional[datetime] = None) -> bool:
        """Check if timestamp is expired based on TTL."""
        if self._ttl <= 0 or not ts:
            return False
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return ((now or datetime.now()) - ts).total_seconds() >= self._ttl * 60
    
    @staticmethod
    def _format_summary(text: str, last_active: datetime) -> str:
        """Format session summary for resumed session."""
        idle_min = int((datetime.now() - last_active).total_seconds() / 60)
        return f"Inactive for {idle_min} minutes.\nPrevious conversation summary: {text}"
    
    def _split_unconsolidated(
        self, session: "Session",
    ) -> tuple[list, list]:
        """Split live session tail into archiveable prefix and retained recent suffix."""
        tail = list(session.messages[getattr(session, 'last_consolidated', 0):])
        if not tail:
            return [], []
        
        # Keep recent messages
        kept_count = min(self._RECENT_SUFFIX_MESSAGES, len(tail))
        kept = tail[-kept_count:]
        cut = len(tail) - len(kept)
        
        return tail[:cut], kept
    
    def check_expired(
        self,
        schedule_background: Callable[[Coroutine], None],
        active_session_keys: Collection[str] = (),
    ) -> None:
        """Schedule archival for idle sessions."""
        now = datetime.now()
        for info in self.sessions.list_sessions():
            key = info.get("key", "")
            if not key or key in self._archiving:
                continue
            if key in active_session_keys:
                continue
            if self._is_expired(info.get("updated_at"), now):
                self._archiving.add(key)
                schedule_background(self._archive(key))
    
    async def _archive(self, key: str) -> None:
        """Archive idle session: summarize old messages, keep recent."""
        try:
            session = self.sessions.get_or_create(key)
            archive_msgs, kept_msgs = self._split_unconsolidated(session)
            
            if not archive_msgs and not kept_msgs:
                session.updated_at = datetime.now()
                self.sessions.save(session)
                return
            
            last_active = session.updated_at
            summary = ""
            
            if archive_msgs:
                summary = await self.consolidator.archive(archive_msgs) or ""
            
            if summary and summary != "(nothing)":
                self._summaries[key] = (summary, last_active)
                session.metadata["_last_summary"] = {
                    "text": summary,
                    "last_active": last_active.isoformat()
                }
            
            session.messages = kept_msgs
            if hasattr(session, 'last_consolidated'):
                session.last_consolidated = 0
            session.updated_at = datetime.now()
            self.sessions.save(session)
            
        except Exception:
            pass
        finally:
            self._archiving.discard(key)
    
    def prepare_session(
        self, session: "Session", key: str
    ) -> tuple["Session", Optional[str]]:
        """Prepare session for use: check if needs reload, return summary if available."""
        if key in self._archiving or self._is_expired(session.updated_at):
            session = self.sessions.get_or_create(key)
        
        # Check in-memory summary cache
        entry = self._summaries.pop(key, None)
        if entry:
            session.metadata.pop("_last_summary", None)
            return session, self._format_summary(entry[0], entry[1])
        
        # Check disk metadata
        if "_last_summary" in session.metadata:
            meta = session.metadata.pop("_last_summary")
            self.sessions.save(session)
            return session, self._format_summary(
                meta["text"],
                datetime.fromisoformat(meta["last_active"])
            )
        
        return session, None