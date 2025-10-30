#!/usr/bin/env python3
"""
Phase 4: Persistent Memory & Session Management
- SessionAgent: Manages user sessions and context persistence
- MemoryAgent: Episodic and semantic memory storage
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from cryptography.fernet import Fernet
from .base_agent import BaseAgent
from ..config import ROOT_DIR
from ..utils import now_ts, load_json, save_json

logger = logging.getLogger("kalki.agents.phase4")


class SessionAgent(BaseAgent):
    """
    Manages user sessions with context persistence and encryption
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="SessionAgent", config=config)
        self.sessions_dir = Path(ROOT_DIR) / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        self.sessions_index = self.sessions_dir / "sessions_index.json"
        self.encryption_key = config.get("encryption_key") if config else None
        self.cipher = None
        if self.encryption_key:
            self.cipher = Fernet(self.encryption_key.encode() if isinstance(self.encryption_key, str) else self.encryption_key)
    
    def initialize(self) -> bool:
        """Initialize session storage"""
        try:
            if not self.sessions_index.exists():
                save_json(self.sessions_index, {})
            self.state = "ready"
            self.logger.info("SessionAgent initialized")
            return True
        except Exception as e:
            self.logger.exception(f"SessionAgent initialization failed: {e}")
            self.state = "error"
            return False
    
    def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
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
            self._save_session(session_file, session_data)
            
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
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data"""
        try:
            session_file = self.sessions_dir / f"{session_id}.json"
            if not session_file.exists():
                return None
            return self._load_session(session_file)
        except Exception as e:
            self.logger.exception(f"Failed to get session {session_id}: {e}")
            return None
    
    def update_session(self, session_id: str, context_update: Dict[str, Any]):
        """Update session context"""
        try:
            session_data = self.get_session(session_id)
            if not session_data:
                raise ValueError(f"Session {session_id} not found")
            
            session_data["context"].append({
                "timestamp": now_ts(),
                "data": context_update
            })
            session_data["updated_at"] = now_ts()
            
            session_file = self.sessions_dir / f"{session_id}.json"
            self._save_session(session_file, session_data)
            self.logger.debug(f"Updated session {session_id}")
        except Exception as e:
            self.logger.exception(f"Failed to update session: {e}")
            raise
    
    def close_session(self, session_id: str):
        """Close a session"""
        try:
            session_data = self.get_session(session_id)
            if session_data:
                session_data["state"] = "closed"
                session_data["closed_at"] = now_ts()
                session_file = self.sessions_dir / f"{session_id}.json"
                self._save_session(session_file, session_data)
                self.logger.info(f"Closed session {session_id}")
        except Exception as e:
            self.logger.exception(f"Failed to close session: {e}")
    
    def list_user_sessions(self, user_id: str) -> List[str]:
        """List all sessions for a user"""
        index = load_json(self.sessions_index, {})
        return index.get(user_id, [])
    
    def _save_session(self, filepath: Path, data: Dict[str, Any]):
        """Save session data with optional encryption"""
        if self.cipher:
            encrypted_data = self.cipher.encrypt(json.dumps(data).encode())
            filepath.write_bytes(encrypted_data)
        else:
            save_json(filepath, data)
    
    def _load_session(self, filepath: Path) -> Dict[str, Any]:
        """Load session data with optional decryption"""
        if self.cipher:
            encrypted_data = filepath.read_bytes()
            decrypted_data = self.cipher.decrypt(encrypted_data)
            return json.loads(decrypted_data.decode())
        else:
            return load_json(filepath, {})
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute session management tasks"""
        action = task.get("action")
        
        if action == "create":
            session_id = self.create_session(task["user_id"], task.get("metadata"))
            return {"status": "success", "session_id": session_id}
        elif action == "get":
            session_data = self.get_session(task["session_id"])
            return {"status": "success", "data": session_data}
        elif action == "update":
            self.update_session(task["session_id"], task["context_update"])
            return {"status": "success"}
        elif action == "close":
            self.close_session(task["session_id"])
            return {"status": "success"}
        elif action == "list":
            sessions = self.list_user_sessions(task["user_id"])
            return {"status": "success", "sessions": sessions}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class MemoryAgent(BaseAgent):
    """
    Manages episodic and semantic memory storage
    - Episodic: Event-based, time-ordered memories
    - Semantic: Concept-based, structured knowledge
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="MemoryAgent", config=config)
        self.memory_dir = Path(ROOT_DIR) / "memory"
        self.episodic_dir = self.memory_dir / "episodic"
        self.semantic_dir = self.memory_dir / "semantic"
        self.episodic_dir.mkdir(parents=True, exist_ok=True)
        self.semantic_dir.mkdir(parents=True, exist_ok=True)
        self.episodic_index = self.memory_dir / "episodic_index.json"
        self.semantic_index = self.memory_dir / "semantic_index.json"
    
    def initialize(self) -> bool:
        """Initialize memory storage"""
        try:
            if not self.episodic_index.exists():
                save_json(self.episodic_index, [])
            if not self.semantic_index.exists():
                save_json(self.semantic_index, {})
            self.state = "ready"
            self.logger.info("MemoryAgent initialized")
            return True
        except Exception as e:
            self.logger.exception(f"MemoryAgent initialization failed: {e}")
            self.state = "error"
            return False
    
    def store_episodic(self, event: Dict[str, Any]) -> str:
        """
        Store episodic memory (event-based)
        """
        try:
            memory_id = f"episodic_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            memory_data = {
                "memory_id": memory_id,
                "timestamp": now_ts(),
                "event": event,
                "type": "episodic"
            }
            
            memory_file = self.episodic_dir / f"{memory_id}.json"
            save_json(memory_file, memory_data)
            
            # Update index
            index = load_json(self.episodic_index, [])
            index.append({
                "memory_id": memory_id,
                "timestamp": memory_data["timestamp"],
                "summary": event.get("summary", "")
            })
            save_json(self.episodic_index, index)
            
            self.logger.debug(f"Stored episodic memory {memory_id}")
            return memory_id
        except Exception as e:
            self.logger.exception(f"Failed to store episodic memory: {e}")
            raise
    
    def store_semantic(self, concept: str, knowledge: Dict[str, Any]) -> str:
        """
        Store semantic memory (concept-based knowledge)
        """
        try:
            memory_id = f"semantic_{concept}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            memory_data = {
                "memory_id": memory_id,
                "concept": concept,
                "timestamp": now_ts(),
                "knowledge": knowledge,
                "type": "semantic"
            }
            
            memory_file = self.semantic_dir / f"{memory_id}.json"
            save_json(memory_file, memory_data)
            
            # Update index
            index = load_json(self.semantic_index, {})
            if concept not in index:
                index[concept] = []
            index[concept].append(memory_id)
            save_json(self.semantic_index, index)
            
            self.logger.debug(f"Stored semantic memory {memory_id} for concept '{concept}'")
            return memory_id
        except Exception as e:
            self.logger.exception(f"Failed to store semantic memory: {e}")
            raise
    
    def recall_episodic(self, limit: int = 10, start_time: Optional[str] = None, end_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recall episodic memories within time range
        """
        try:
            index = load_json(self.episodic_index, [])
            filtered = index
            
            if start_time:
                filtered = [m for m in filtered if m["timestamp"] >= start_time]
            if end_time:
                filtered = [m for m in filtered if m["timestamp"] <= end_time]
            
            # Sort by timestamp descending
            filtered.sort(key=lambda x: x["timestamp"], reverse=True)
            filtered = filtered[:limit]
            
            # Load full memory data
            memories = []
            for entry in filtered:
                memory_file = self.episodic_dir / f"{entry['memory_id']}.json"
                if memory_file.exists():
                    memories.append(load_json(memory_file))
            
            return memories
        except Exception as e:
            self.logger.exception(f"Failed to recall episodic memories: {e}")
            return []
    
    def recall_semantic(self, concept: str) -> List[Dict[str, Any]]:
        """
        Recall semantic memories for a concept
        """
        try:
            index = load_json(self.semantic_index, {})
            memory_ids = index.get(concept, [])
            
            memories = []
            for memory_id in memory_ids:
                memory_file = self.semantic_dir / f"{memory_id}.json"
                if memory_file.exists():
                    memories.append(load_json(memory_file))
            
            return memories
        except Exception as e:
            self.logger.exception(f"Failed to recall semantic memories: {e}")
            return []
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute memory management tasks"""
        action = task.get("action")
        memory_type = task.get("type", "episodic")
        
        if action == "store":
            if memory_type == "episodic":
                memory_id = self.store_episodic(task["event"])
                return {"status": "success", "memory_id": memory_id}
            elif memory_type == "semantic":
                memory_id = self.store_semantic(task["concept"], task["knowledge"])
                return {"status": "success", "memory_id": memory_id}
        elif action == "recall":
            if memory_type == "episodic":
                memories = self.recall_episodic(
                    limit=task.get("limit", 10),
                    start_time=task.get("start_time"),
                    end_time=task.get("end_time")
                )
                return {"status": "success", "memories": memories}
            elif memory_type == "semantic":
                memories = self.recall_semantic(task["concept"])
                return {"status": "success", "memories": memories}
        
        return {"status": "error", "message": f"Unknown action or type: {action}/{memory_type}"}
