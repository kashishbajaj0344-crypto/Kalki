"""
SQLite-backed persistent memory storage implementation.
"""

import asyncio
import sqlite3
import json
import logging
from typing import Any, Optional, List, Dict
from datetime import datetime
from pathlib import Path
from .base import MemoryStore, MemoryEntry, MemoryQuery

logger = logging.getLogger('Kalki.Memory.SQLite')


class SQLiteMemoryStore(MemoryStore):
    """SQLite-backed persistent memory storage."""
    
    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize SQLite memory store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._lock = asyncio.Lock()
        self._init_db()
        logger.info(f"[Kalki.Memory] SQLiteMemoryStore initialized: {db_path}")
    
    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    metadata TEXT,
                    timestamp TEXT NOT NULL
                )
            """)
            # Add additional indexes for performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_key ON memory (key)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_timestamp ON memory (timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_memory_metadata_type ON memory (json_extract(metadata, '$.type'))
            """)
            conn.commit()
    
    async def put_async(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Async version of put for thread safety."""
        async with self._lock:
            return self._put_sync(key, value, metadata)
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a key-value pair in SQLite."""
        return self._put_sync(key, value, metadata)
    
    def _put_sync(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Internal synchronous put operation."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO memory (key, value, metadata, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (
                    key,
                    json.dumps(value),
                    json.dumps(metadata or {}),
                    datetime.now().isoformat()
                ))
                conn.commit()
            logger.debug(f"[Kalki.Memory] Stored entry: {key}")
            return True
        except Exception as e:
            logger.error(f"[Kalki.Memory] Failed to store entry {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[MemoryEntry]:
        """Retrieve a memory entry by key."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    SELECT key, value, metadata, timestamp
                    FROM memory
                    WHERE key = ?
                """, (key,))
                row = cursor.fetchone()
                
                if row:
                    entry = MemoryEntry(
                        key=row[0],
                        value=json.loads(row[1]),
                        metadata=json.loads(row[2]),
                        timestamp=datetime.fromisoformat(row[3])
                    )
                    logger.debug(f"[Kalki.Memory] Retrieved entry: {key}")
                    return entry
                else:
                    logger.debug(f"[Kalki.Memory] Entry not found: {key}")
        except Exception as e:
            logger.error(f"[Kalki.Memory] Failed to retrieve entry {key}: {e}")
        return None
    
    def query(self, query: MemoryQuery) -> List[MemoryEntry]:
        """Query memory entries based on filters."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                sql_parts = ["SELECT key, value, metadata, timestamp FROM memory WHERE 1=1"]
                params = []
                
                # Filter by keys
                if query.keys:
                    placeholders = ','.join('?' * len(query.keys))
                    sql_parts.append(f"AND key IN ({placeholders})")
                    params.extend(query.keys)
                
                # Filter by timestamp range
                if query.since:
                    sql_parts.append("AND timestamp >= ?")
                    params.append(query.since.isoformat())
                
                if query.until:
                    sql_parts.append("AND timestamp <= ?")
                    params.append(query.until.isoformat())
                
                # Order by timestamp (newest first)
                sql_parts.append("ORDER BY timestamp DESC")
                
                # Apply limit
                if query.limit:
                    sql_parts.append("LIMIT ?")
                    params.append(query.limit)
                
                sql = " ".join(sql_parts)
                cursor = conn.execute(sql, params)
                
                results = []
                for row in cursor.fetchall():
                    entry = MemoryEntry(
                        key=row[0],
                        value=json.loads(row[1]),
                        metadata=json.loads(row[2]),
                        timestamp=datetime.fromisoformat(row[3])
                    )
                    
                    # Apply metadata filter in Python (more flexible)
                    if query.filter:
                        if all(entry.metadata.get(k) == v for k, v in query.filter.items()):
                            results.append(entry)
                    else:
                        results.append(entry)
                
                logger.debug(f"[Kalki.Memory] Query returned {len(results)} results")
                return results
        except Exception as e:
            logger.error(f"[Kalki.Memory] Query failed: {e}")
            return []
    
    def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("DELETE FROM memory WHERE key = ?", (key,))
                conn.commit()
                deleted = cursor.rowcount > 0
                if deleted:
                    logger.debug(f"[Kalki.Memory] Deleted entry: {key}")
                else:
                    logger.debug(f"[Kalki.Memory] Entry not found for deletion: {key}")
                return deleted
        except Exception as e:
            logger.error(f"[Kalki.Memory] Failed to delete entry {key}: {e}")
            return False
    
    def compact(self, limit: Optional[int] = None) -> int:
        """Remove oldest entries to keep storage size manageable."""
        if limit is None:
            logger.debug("[Kalki.Memory] No compaction needed")
            return 0
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get current count
                cursor = conn.execute("SELECT COUNT(*) FROM memory")
                count = cursor.fetchone()[0]
                
                if count <= limit:
                    logger.debug("[Kalki.Memory] No compaction needed")
                    return 0
                
                # Delete oldest entries
                to_remove = count - limit
                conn.execute("""
                    DELETE FROM memory
                    WHERE key IN (
                        SELECT key FROM memory
                        ORDER BY timestamp ASC
                        LIMIT ?
                    )
                """, (to_remove,))
                conn.commit()
                logger.info(f"[Kalki.Memory] Compacted {to_remove} entries")
                return to_remove
        except Exception as e:
            logger.error(f"[Kalki.Memory] Compaction failed: {e}")
            return 0
    
    def clear(self) -> None:
        """Remove all entries from the store."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memory")
                count = cursor.fetchone()[0]
                conn.execute("DELETE FROM memory")
                conn.commit()
            logger.info(f"[Kalki.Memory] Cleared {count} entries")
        except Exception as e:
            logger.error(f"[Kalki.Memory] Clear failed: {e}")
    
    def count(self) -> int:
        """Return the total number of entries."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memory")
                return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"[Kalki.Memory] Count failed: {e}")
            return 0


# [Kalki v2.3 — memory/sqlite_store.py v1.0]