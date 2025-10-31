"""
Kalki Memory Module - Phases 13-14
Long-term memory persistence with in-memory and SQLite implementations,
plus episodic and semantic memory layers.
"""

from typing import Dict, Any, List
from .base import MemoryStore, MemoryEntry, MemoryQuery
from .in_memory import InMemoryStore
from .sqlite_store import SQLiteMemoryStore
from .layered import EpisodicMemory, SemanticMemory, EpisodeEvent

__all__ = [
    'MemoryStore',
    'MemoryEntry',
    'MemoryQuery',
    'InMemoryStore',
    'SQLiteMemoryStore',
    'EpisodicMemory',
    'SemanticMemory',
    'EpisodeEvent',
]


def initialize_default_memory():
    """
    Initialize standard memory layers for Kalki agents.
    
    Returns:
        Tuple of (episodic_memory, semantic_memory, transient_memory)
    """
    episodic = EpisodicMemory(SQLiteMemoryStore("data/episodic.db"))
    semantic = SemanticMemory(SQLiteMemoryStore("data/semantic.db"))
    transient = InMemoryStore()
    return episodic, semantic, transient


class MemoryMonitor:
    """
    Memory monitoring and analytics stub for future meta-core integration.
    """
    
    def summarize(self, memory_store: MemoryStore) -> Dict[str, Any]:
        """Generate summary statistics for a memory store."""
        count = memory_store.count()
        return {
            "total_entries": count,
            "store_type": type(memory_store).__name__,
            "status": "operational" if count >= 0 else "error"
        }
    
    def detect_patterns(self, episodic: EpisodicMemory) -> List[Dict[str, Any]]:
        """Detect patterns in episodic memory (stub for future implementation)."""
        recent = episodic.get_recent_episodes(limit=100)
        # Simple pattern detection - group by event type
        patterns = {}
        for event in recent:
            patterns[event.event_type] = patterns.get(event.event_type, 0) + 1
        
        return [
            {"pattern": event_type, "frequency": count, "confidence": count / len(recent)}
            for event_type, count in patterns.items()
        ]


# [Kalki v2.3 — memory/__init__.py v1.0]