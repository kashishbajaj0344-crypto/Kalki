"""
In-memory implementation of MemoryStore for testing and CI.
"""

import asyncio
import logging
from typing import Any, Optional, List, Dict
from datetime import datetime
from .base import MemoryStore, MemoryEntry, MemoryQuery

logger = logging.getLogger('Kalki.Memory.InMemory')


class InMemoryStore(MemoryStore):
    """In-memory implementation of memory storage."""
    
    def __init__(self):
        """Initialize empty in-memory store."""
        self._store: Dict[str, MemoryEntry] = {}
        self._lock = asyncio.Lock()
        logger.info("[Kalki.Memory] InMemoryStore initialized")
    
    async def put_async(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of put for thread safety."""
        async with self._lock:
            try:
                entry = MemoryEntry(
                    key=key,
                    value=value,
                    metadata=metadata or {},
                    timestamp=datetime.now()
                )
                self._store[key] = entry
                logger.debug(f"[Kalki.Memory] Stored entry: {key}")
                return True
            except Exception as e:
                logger.error(f"[Kalki.Memory] Failed to store entry {key}: {e}")
                return False
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a key-value pair in memory."""
        try:
            entry = MemoryEntry(
                key=key,
                value=value,
                metadata=metadata or {},
                timestamp=datetime.now()
            )
            self._store[key] = entry
            logger.debug(f"[Kalki.Memory] Stored entry: {key}")
            return True
        except Exception as e:
            logger.error(f"[Kalki.Memory] Failed to store entry {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by key."""
        entry = self._store.get(key)
        if entry:
            logger.debug(f"[Kalki.Memory] Retrieved entry: {key}")
        else:
            logger.debug(f"[Kalki.Memory] Entry not found: {key}")
        return entry
    
    def query(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Query memory entries based on filters."""
        results = list(self._store.values())
        logger.debug(f"[Kalki.Memory] Querying {len(results)} entries")
        
        # Filter by keys if specified
        if query.keys:
            results = [e for e in results if e.key in query.keys]
        
        # Filter by timestamp range
        if query.since:
            results = [e for e in results if e.timestamp >= query.since]
        if query.until:
            results = [e for e in results if e.timestamp <= query.until]
        
        # Filter by metadata
        if query.filter:
            results = [
                e for e in results
                if all(e.metadata.get(k) == v for k, v in query.filter.items())
            ]
        
        # Sort by timestamp (newest first)
        results.sort(key=lambda e: e.timestamp, reverse=True)
        
        # Apply limit
        if query.limit:
            results = results[:query.limit]
        
        logger.debug(f"[Kalki.Memory] Query returned {len(results)} results")
        return results
    
    def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        if key in self._store:
            del self._store[key]
            logger.debug(f"[Kalki.Memory] Deleted entry: {key}")
            return True
        logger.debug(f"[Kalki.Memory] Entry not found for deletion: {key}")
        return False
    
    def compact(self, limit: Optional[int] = None) -> int:
        """Remove oldest entries to keep storage size manageable."""
        if limit is None or len(self._store) <= limit:
            logger.debug("[Kalki.Memory] No compaction needed")
            return 0
        
        # Sort entries by timestamp (oldest first)
        sorted_entries = sorted(self._store.values(), key=lambda e: e.timestamp)
        
        # Calculate how many to remove
        to_remove = len(sorted_entries) - limit
        
        # Remove oldest entries
        for entry in sorted_entries[:to_remove]:
            del self._store[entry.key]
        
        logger.info(f"[Kalki.Memory] Compacted {to_remove} entries")
        return to_remove
    
    def clear(self) -> None:
        """Remove all entries from the store."""
        count = len(self._store)
        self._store.clear()
        logger.info(f"[Kalki.Memory] Cleared {count} entries")
    
    def count(self) -> int:
        """Return the total number of entries."""
        return len(self._store)


# [Kalki v2.3 — memory/in_memory.py v1.0]