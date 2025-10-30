"""
Unit tests for Phase 13 - Long-term Memory Persistence
"""

import unittest
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path

from modules.memory import (
    MemoryStore, MemoryEntry, MemoryQuery,
    InMemoryStore, SQLiteMemoryStore
)


class TestMemoryStore(unittest.TestCase):
    """Base test class for memory store implementations."""
    
    def get_store(self) -> MemoryStore:
        """Override in subclasses to return specific store implementation."""
        raise NotImplementedError
    
    def test_put_and_get(self):
        """Test basic put and get operations."""
        store = self.get_store()
        
        # Put a value
        result = store.put("test_key", {"data": "test_value"}, {"type": "test"})
        self.assertTrue(result)
        
        # Get the value
        entry = store.get("test_key")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.key, "test_key")
        self.assertEqual(entry.value, {"data": "test_value"})
        self.assertEqual(entry.metadata, {"type": "test"})
    
    def test_get_nonexistent(self):
        """Test getting a non-existent key."""
        store = self.get_store()
        entry = store.get("nonexistent")
        self.assertIsNone(entry)
    
    def test_update_value(self):
        """Test updating an existing key."""
        store = self.get_store()
        
        store.put("key1", "value1")
        store.put("key1", "value2")
        
        entry = store.get("key1")
        self.assertEqual(entry.value, "value2")
    
    def test_delete(self):
        """Test deleting entries."""
        store = self.get_store()
        
        store.put("key1", "value1")
        self.assertTrue(store.delete("key1"))
        self.assertIsNone(store.get("key1"))
        
        # Delete non-existent key
        self.assertFalse(store.delete("nonexistent"))
    
    def test_query_all(self):
        """Test querying all entries."""
        store = self.get_store()
        
        store.put("key1", "value1")
        store.put("key2", "value2")
        store.put("key3", "value3")
        
        query = MemoryQuery()
        results = store.query(query)
        
        self.assertEqual(len(results), 3)
    
    def test_query_with_limit(self):
        """Test querying with limit."""
        store = self.get_store()
        
        for i in range(10):
            store.put(f"key{i}", f"value{i}")
        
        query = MemoryQuery(limit=5)
        results = store.query(query)
        
        self.assertEqual(len(results), 5)
    
    def test_query_with_keys(self):
        """Test querying specific keys."""
        store = self.get_store()
        
        store.put("key1", "value1")
        store.put("key2", "value2")
        store.put("key3", "value3")
        
        query = MemoryQuery(keys=["key1", "key3"])
        results = store.query(query)
        
        self.assertEqual(len(results), 2)
        keys = [e.key for e in results]
        self.assertIn("key1", keys)
        self.assertIn("key3", keys)
    
    def test_query_with_metadata_filter(self):
        """Test querying with metadata filter."""
        store = self.get_store()
        
        store.put("key1", "value1", {"type": "A"})
        store.put("key2", "value2", {"type": "B"})
        store.put("key3", "value3", {"type": "A"})
        
        query = MemoryQuery(filter={"type": "A"})
        results = store.query(query)
        
        self.assertEqual(len(results), 2)
        for entry in results:
            self.assertEqual(entry.metadata["type"], "A")
    
    def test_query_with_time_range(self):
        """Test querying with time range."""
        store = self.get_store()
        
        now = datetime.now()
        
        # Manually create entries with specific timestamps
        store.put("old", "value_old")
        store.put("new", "value_new")
        
        # Query since 1 second ago
        query = MemoryQuery(since=now - timedelta(seconds=1))
        results = store.query(query)
        
        # Both should be included as they were just created
        self.assertGreaterEqual(len(results), 2)
    
    def test_compact(self):
        """Test compacting the store."""
        store = self.get_store()
        
        # Add 10 entries
        for i in range(10):
            store.put(f"key{i}", f"value{i}")
        
        # Compact to 5
        removed = store.compact(limit=5)
        
        self.assertEqual(removed, 5)
        self.assertEqual(store.count(), 5)
    
    def test_compact_below_limit(self):
        """Test compact when already below limit."""
        store = self.get_store()
        
        store.put("key1", "value1")
        store.put("key2", "value2")
        
        removed = store.compact(limit=10)
        self.assertEqual(removed, 0)
        self.assertEqual(store.count(), 2)
    
    def test_clear(self):
        """Test clearing all entries."""
        store = self.get_store()
        
        store.put("key1", "value1")
        store.put("key2", "value2")
        
        store.clear()
        self.assertEqual(store.count(), 0)
    
    def test_count(self):
        """Test counting entries."""
        store = self.get_store()
        
        self.assertEqual(store.count(), 0)
        
        store.put("key1", "value1")
        self.assertEqual(store.count(), 1)
        
        store.put("key2", "value2")
        self.assertEqual(store.count(), 2)
        
        store.delete("key1")
        self.assertEqual(store.count(), 1)


class TestInMemoryStore(TestMemoryStore):
    """Tests for InMemoryStore implementation."""
    
    def get_store(self) -> MemoryStore:
        """Return InMemoryStore instance."""
        return InMemoryStore()


class TestSQLiteMemoryStore(TestMemoryStore):
    """Tests for SQLiteMemoryStore implementation."""
    
    def setUp(self):
        """Create temporary database file."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_memory.db")
    
    def tearDown(self):
        """Clean up temporary database file."""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        os.rmdir(self.temp_dir)
    
    def get_store(self) -> MemoryStore:
        """Return SQLiteMemoryStore instance."""
        return SQLiteMemoryStore(self.db_path)
    
    def test_persistence(self):
        """Test that data persists across store instances."""
        store1 = SQLiteMemoryStore(self.db_path)
        store1.put("persist_key", "persist_value", {"persist": True})
        
        # Create new store instance with same database
        store2 = SQLiteMemoryStore(self.db_path)
        entry = store2.get("persist_key")
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry.value, "persist_value")
        self.assertEqual(entry.metadata, {"persist": True})


if __name__ == '__main__':
    unittest.main()
