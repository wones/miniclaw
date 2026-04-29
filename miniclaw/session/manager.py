from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

@dataclass
class Session:
    key: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    last_consolidated: int = 0


    def add_message(self,role:str,content:str):
        self.messages.append({"role": role, "content": content,"timestamp":datetime.now().isoformat()})
        self.updated_at = datetime.now()

    def get_history(self,max_messages:int =0):
        if max_messages <= 0:
            return self.messages
        return self.messages[-max_messages:]

class SessionManager:
    def __init__(self, workspace: Path | None = None):
        self.sessions : Dict[str,Session] = {}
        self.workspace = workspace
        self.sessions_dir = workspace / "sessions" if workspace else None
        if self.sessions_dir is not None:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def _session_path(self, key: str) -> Path | None:
        if self.sessions_dir is None:
            return None
        safe_name = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in key)
        return self.sessions_dir / f"{safe_name}.json"

    @staticmethod
    def _serialize_session(session: Session) -> dict[str, Any]:
        payload = asdict(session)
        payload["created_at"] = session.created_at.isoformat()
        payload["updated_at"] = session.updated_at.isoformat()
        return payload

    @staticmethod
    def _deserialize_session(payload: dict[str, Any]) -> Session:
        return Session(
            key=payload["key"],
            messages=payload.get("messages", []),
            metadata=payload.get("metadata", {}),
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            last_consolidated=payload.get("last_consolidated", 0),
        )

    def _load_from_disk(self, key: str) -> Session | None:
        path = self._session_path(key)
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, KeyError, ValueError):
            return None
        return self._deserialize_session(payload)

    def _load_all_disk_metadata(self) -> list[dict[str, Any]]:
        if self.sessions_dir is None or not self.sessions_dir.exists():
            return []

        infos: list[dict[str, Any]] = []
        for path in self.sessions_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                session = self._deserialize_session(payload)
            except (OSError, json.JSONDecodeError, KeyError, ValueError):
                continue
            infos.append(
                {
                    "key": session.key,
                    "created_at": session.created_at,
                    "updated_at": session.updated_at,
                    "message_count": len(session.messages),
                }
            )
        return infos
        
    def get_or_create(self,key):
        if key not in self.sessions:
            self.sessions[key] = self._load_from_disk(key) or Session(key)
        return self.sessions[key]

    def save(self,session):
        self.sessions[session.key] = session
        path = self._session_path(session.key)
        if path is None:
            return
        payload = self._serialize_session(session)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_sessions(self) -> List[Dict[str,Any]]:
        """List all sessions with their metadata."""
        in_memory = [
            {
                "key":key,
                "created_at":session.created_at,
                "updated_at":session.updated_at,
                "message_count":len(session.messages)
            }
            for key,session in self.sessions.items()
        ]
        known_keys = {item["key"] for item in in_memory}
        disk_only = [item for item in self._load_all_disk_metadata() if item["key"] not in known_keys]
        return in_memory + disk_only

    def invalidate(self,key:str):
        """Invalidate/clear a session."""
        session = self.get_or_create(key)
        session.messages=[]
        session.last_consolidated=0
        session.updated_at = datetime.now()
        self.save(session)
