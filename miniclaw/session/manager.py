from dataclasses import dataclass,field
from typing import List, Dict, Any
from datetime import datetime

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
    def __init__(self):
        self.sessions : Dict[str,Session] = {}
        
    def get_or_create(self,key):
        if key not in self.sessions:
            self.sessions[key] = Session(key)
        return self.sessions[key]

    def save(self,session):
        pass

    def list_sessions(self) -> List[Dict[str,Any]]:
        """List all sessions with their metadata."""
        return [
            {
                "key":key,
                "created_at":session.created_at,
                "updated_at":session.updated_at,
                "message_count":len(session.messages)
            }
            for key,session in self.sessions.items()
        ]

    def invalidate(self,key:str):
        """Invalidate/clear a session."""
        if key in self.sessions:
            self.sessions[key].messages=[]
            self.sessions[key].last_consolidated=0
