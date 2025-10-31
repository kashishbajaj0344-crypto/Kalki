#!/usr/bin/env python3
"""
SessionAgent: Manages user sessions and context persistence
Phase 4: Persistent Memory & Session Management
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from cryptography.fernet import Fernet

from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from ...config import ROOT


def now_ts() -> str:
    """Get current timestamp as ISO string"""
    return datetime.utcnow().isoformat()


def load_json(filepath: Path, default=None):
    """Load JSON from file with default fallback"""
    try:
        if filepath.exists():
            return json.loads(filepath.read_text())
        return default if default is not None else {}
    except Exception:
        return default if default is not None else {}


def save_json(filepath: Path, data: Dict[str, Any]):
    """Save data as JSON to file"""
    try:
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(json.dumps(data, indent=2))
    except Exception as e:
        print(f"Failed to save JSON to {filepath}: {e}")


class SessionAgent(BaseAgent):
    """
    Manages user sessions with context persistence and encryption
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="SessionAgent",
            capabilities=[AgentCapability.MEMORY],
            description="Manages user sessions with context persistence and encryption",
            config=config or {}
        )
        self.sessions_dir = Path(ROOT) / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_index = self.sessions_dir / "sessions_index.json"
        self.encryption_key = self.config.get("encryption_key") if self.config else None
        self.cipher = None
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)

    async def initialize(self) -> bool:
        """Initialize session storage"""
        try:
            if not self.sessions_index.exists():
                save_json(self.sessions_index, {})
            self.update_status(AgentStatus.READY)
            self.logger.info("SessionAgent initialized")
            return True
        except Exception as e:
            self.logger.exception(f"SessionAgent initialization failed: {e}")
            self.update_status(AgentStatus.ERROR)
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute session management tasks"""
        action = task.get("action")

        if action == "create":
            session_id = await self.create_session(task["user_id"], task.get("metadata"))
            return {"status": "success", "session_id": session_id}
        elif action == "get":
            session_data = await self.get_session(task["session_id"])
            return {"status": "success", "data": session_data}
        elif action == "update":
            await self.update_session(task["session_id"], task["context_update"])
            return {"status": "success"}
        elif action == "close":
            await self.close_session(task["session_id"])
            return {"status": "success"}
        elif action == "list":
            sessions = await self.list_user_sessions(task["user_id"])
            return {"status": "success", "sessions": sessions}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    async def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create a new session for a user"""
        try:
            session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": now_ts(),
                "updated_at": now_ts(),
                "metadata": metadata or {},
                "context": [],
                "state": "active"
            }

            session_file = self.sessions_dir / f"{session_id}.json"
            await self._save_session(session_file, session_data)

            # Update index
            index = load_json(self.sessions_index, {})
            if user_id not in index:
                index[user_id] = []
            index[user_id].append(session_id)
            save_json(self.sessions_index, index)

            self.logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
        except Exception as e:
            self.logger.exception(f"Failed to create session: {e}")
            raise

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                return None
            return await self._load_session(session_file)
        except Exception as e:
            self.logger.exception(f"Failed to get session {session_id}: {e}")
            return None

    async def update_session(self, session_id: str, context_update: Dict[str, Any]):
        """Update session context"""
        try:
            session_data = await self.get_session(session_id)
            if not session_data:
                raise ValueError(f"Session {session_id} not found")

            session_data["context"].append({
                "timestamp": now_ts(),
                "data": context_update
            })
            session_data["updated_at"] = now_ts()

            session_file = self.sessions_dir / f"{session_id}.json"
            await self._save_session(session_file, session_data)
            self.logger.debug(f"Updated session {session_id}")
        except Exception as e:
            self.logger.exception(f"Failed to update session: {e}")
            raise

    async def close_session(self, session_id: str):
        """Close a session"""
        try:
            session_data = await self.get_session(session_id)
            if session_data:
                session_data["state"] = "closed"
                session_data["closed_at"] = now_ts()
                session_file = self.sessions_dir / f"{session_id}.json"
                await self._save_session(session_file, session_data)
                self.logger.info(f"Closed session {session_id}")
        except Exception as e:
            self.logger.exception(f"Failed to close session: {e}")

    async def list_user_sessions(self, user_id: str) -> List[str]:
        """List all sessions for a user"""
        index = load_json(self.sessions_index, {})
        return index.get(user_id, [])

    async def _save_session(self, filepath: Path, data: Dict[str, Any]):
        """Save session data with optional encryption"""
        if self.cipher:
            encrypted_data = self.cipher.encrypt(json.dumps(data).encode())
            filepath.write_bytes(encrypted_data)
        else:
            save_json(filepath, data)

    async def _load_session(self, filepath: Path) -> Dict[str, Any]:
        """Load session data with optional decryption"""
        if self.cipher:
            encrypted_data = filepath.read_bytes()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        else:
            return load_json(filepath, {})

    async def shutdown(self) -> bool:
        """Shutdown the session agent"""
        self.logger.info("SessionAgent shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True