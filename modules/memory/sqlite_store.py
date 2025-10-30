"""
SQLite-backed persistent memory storage implementation.
"""

import sqlite3
import json
from typing import Any, Optional, List, Dict
from datetime import datetime
from pathlib import Path
from .base import MemoryStore, MemoryEntry, MemoryQuery


class SQLiteMemoryStore(MemoryStore):
    """SQLite-backed persistent memory storage."""
    
    def __init__(self, db_path: str = "memory.db"):
        """
        Initialize SQLite memory store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._init_db()
    
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
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON memory(timestamp)
            """)
            conn.commit()
    
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a key-value pair in SQLite."""
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
            return True
        except Exception:
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
                    return MemoryEntry(
                        key=row[0],
                        value=json.loads(row[1]),
                        metadata=json.loads(row[2]),
                        timestamp=datetime.fromisoformat(row[3])
                    )
        except Exception:
            pass
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
                
                return results
        except Exception:
            return []
    
    def delete(self, key: str) -> bool:
        """Delete a memory entry by key."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("DELETE FROM memory WHERE key = ?", (key,))
                conn.commit()
                return cursor.rowcount > 0
        except Exception:
            return False
    
    def compact(self, limit: Optional[int] = None) -> int:
        """Remove oldest entries to keep storage size manageable."""
        if limit is None:
            return 0
        
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                # Get current count
                cursor = conn.execute("SELECT COUNT(*) FROM memory")
                count = cursor.fetchone()[0]
                
                if count <= limit:
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
                return to_remove
        except Exception:
            return 0
    
    def clear(self) -> None:
        """Remove all entries from the store."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("DELETE FROM memory")
            conn.commit()
    
    def count(self) -> int:
        """Return the total number of entries."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM memory")
                return cursor.fetchone()[0]
        except Exception:
            return 0
