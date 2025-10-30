"""
modules/session.py
KALKI v2.3 — Session persistence module.
Handles runtime session state (ID, timestamp, metadata) with robust I/O, logging, and version registration.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from modules.config import CONFIG, register_module_version
from modules.logger import get_logger

__version__ = "Kalki v2.3 - modules/session.py - v0.3"
register_module_version("session.py", __version__)

logger = get_logger("session")

# Resolve and ensure session file path exists
try:
    SESSION_FILE = Path(CONFIG["session_file"]).resolve()
except Exception:
    SESSION_FILE = Path("./session.json").resolve()
SESSION_FILE.parent.mkdir(parents=True, exist_ok=True)


class Session:
    """
    Manages runtime session state for Kalki.
    Each session has a unique ID, creation time, and metadata dictionary.
    """

    def __init__(self, session_id: str, created_at: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.session_id = session_id
        self.created_at = created_at
        self.metadata = metadata or {}

    # ------------------------------
    # Session lifecycle management
    # ------------------------------
    @classmethod
    def load_or_create(cls) -> "Session":
        """
        Loads an existing session from SESSION_FILE, or creates a new one if missing or invalid.
        Returns:
            Session: The loaded or newly created session object.
        """
        if SESSION_FILE.exists():
            try:
                with SESSION_FILE.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict) or "session_id" not in data:
                    raise ValueError("Invalid session structure")
                logger.info(f"Loaded existing session {data['session_id']} created at {data['created_at']}")
                return cls(
                    session_id=data["session_id"],
                    created_at=data["created_at"],
                    metadata=data.get("metadata", {})
                )
            except Exception as e:
                logger.warning(f"Failed to load session file ({e}), creating new one.")

        # Create new session if not found or corrupted
        sid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        logger.info(f"Creating new session {sid}")
        session = cls(sid, now)
        session.save()
        return session

    # ------------------------------
    # Persistence methods
    # ------------------------------
    def save(self) -> None:
        """
        Persists the current session state to SESSION_FILE as JSON.
        """
        payload = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "metadata": self.metadata
        }
        try:
            with SESSION_FILE.open("w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            logger.debug(f"Session {self.session_id} saved to {SESSION_FILE}")
        except Exception as e:
            logger.error(f"Failed to save session {self.session_id}: {e}")

    def update_metadata(self, key: str, value: Any) -> None:
        """
        Update or add a metadata key for this session and save changes.
        """
        self.metadata[key] = value
        self.save()
        logger.debug(f"Session metadata updated: {key}={value}")

    # ------------------------------
    # Representation methods
    # ------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the session for serialization/testing.
        """
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "metadata": self.metadata
        }

    def __repr__(self) -> str:
        return f"<Session id={self.session_id[:8]} created_at={self.created_at}>"
