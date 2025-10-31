# Kalki Memory Module - Phases 13-14

## Overview

The Kalki Memory Module provides long-term memory persistence with in-memory and SQLite implementations, plus episodic and semantic memory layers. This module enables agents to maintain state, learn from experiences, and perform similarity-based retrieval across different types of knowledge.

## Architecture

### Core Components

- **`MemoryStore`**: Abstract base class defining the memory interface
- **`MemoryEntry`**: Data structure for memory entries with metadata and timestamps
- **`MemoryQuery`**: Query parameters for filtering and retrieving memories

### Storage Implementations

- **`InMemoryStore`**: Fast, volatile storage for temporary/session data
- **`SQLiteMemoryStore`**: Persistent, disk-based storage with indexing

### Memory Layers

- **`EpisodicMemory`**: Time-ordered event storage and retrieval
- **`SemanticMemory`**: Similarity-based document storage using TF-IDF vectors

### Utilities

- **`MemoryMonitor`**: Analytics and monitoring for memory stores
- **`initialize_default_memory()`**: Helper for quick memory layer setup

## Features

### ✅ Implemented

- **Thread Safety**: Async locks for concurrent access
- **Metadata Standardization**: Consistent metadata structure across all entries
- **Performance Indexing**: SQLite indexes on key, timestamp, and metadata type
- **Logging Integration**: Comprehensive logging with Kalki.Logger style messages
- **Version Tracking**: Kalki v2.3 version headers on all modules
- **Cross-Layer Integration**: Helper functions for easy agent integration

### 🔄 Key Capabilities

- **Episodic Memory**: Store and retrieve time-ordered events
- **Semantic Search**: Cosine similarity search over document collections
- **Persistent Storage**: SQLite backend with automatic schema management
- **Memory Compaction**: Automatic cleanup of old entries
- **Query Filtering**: Flexible filtering by time, keys, and metadata

## Usage Examples

### Basic Memory Operations

```python
from modules.agents.memory import InMemoryStore, SQLiteMemoryStore

# In-memory store for session data
transient = InMemoryStore()
transient.put("user_session", {"user_id": "123", "last_action": "query"})

# Persistent store for long-term data
persistent = SQLiteMemoryStore("data/agent_memory.db")
persistent.put("learned_fact", "AI safety is important", {"importance": 0.9})
```

### Episodic Memory

```python
from modules.agents.memory import EpisodicMemory, SQLiteMemoryStore

episodic = EpisodicMemory(SQLiteMemoryStore("data/episodic.db"))

# Add events
event_id = episodic.add_event("user_query", {
    "query": "What is machine learning?",
    "response_quality": 0.85
})

# Retrieve recent events
recent = episodic.get_recent_episodes(limit=10, event_type="user_query")
```

### Semantic Memory

```python
from modules.agents.memory import SemanticMemory, SQLiteMemoryStore

semantic = SemanticMemory(SQLiteMemoryStore("data/semantic.db"))

# Add documents
doc_id = semantic.add_document(
    "Machine learning is a subset of artificial intelligence",
    {"domain": "ai", "source": "textbook"}
)

# Search similar content
results = semantic.search_similar("artificial intelligence techniques", limit=5)
for result in results:
    print(f"Score: {result['score']:.3f} - {result['text'][:50]}...")
```

### Quick Integration Setup

```python
from modules.agents.memory import initialize_default_memory

# Get standard memory layers
episodic, semantic, transient = initialize_default_memory()

# Ready to use in agents
episodic.add_event("agent_action", {"action": "processed_query"})
semantic.add_document("New knowledge about AI safety")
transient.put("temp_data", {"session_id": "abc123"})
```

## Integration with Kalki Agents

### Agent Memory Usage

```python
class MyAgent:
    def __init__(self):
        self.episodic = EpisodicMemory(SQLiteMemoryStore("data/agent_memory.db"))
        self.semantic = SemanticMemory(SQLiteMemoryStore("data/agent_knowledge.db"))

    def learn_from_interaction(self, user_query, response):
        # Store the interaction
        self.episodic.add_event("user_interaction", {
            "query": user_query,
            "response": response,
            "timestamp": datetime.now()
        })

        # Extract and store knowledge
        if len(response) > 100:  # Substantial responses
            self.semantic.add_document(response, {
                "source": "agent_response",
                "query": user_query
            })
```

### Memory Monitoring

```python
from modules.agents.memory import MemoryMonitor

monitor = MemoryMonitor()

# Get store statistics
stats = monitor.summarize(my_memory_store)
print(f"Total entries: {stats['total_entries']}")

# Analyze episodic patterns
patterns = monitor.detect_patterns(episodic_memory)
for pattern in patterns:
    print(f"{pattern['pattern']}: {pattern['frequency']} times")
```

## Configuration

### SQLite Database Paths

- Default episodic memory: `data/episodic.db`
- Default semantic memory: `data/semantic.db`
- Custom paths: `SQLiteMemoryStore("custom/path.db")`

### Memory Compaction

```python
# Keep only the 1000 most recent entries
store.compact(limit=1000)

# Clear all entries
store.clear()
```

## Performance Characteristics

- **InMemoryStore**: O(1) operations, limited by RAM
- **SQLiteMemoryStore**: O(log n) queries with indexing, persistent storage
- **EpisodicMemory**: Fast event addition, timestamp-ordered retrieval
- **SemanticMemory**: O(n) search over documents, cosine similarity scoring

## Future Extensions (Phases 15-18)

- **Meta-Core Integration**: Memory summarization for self-modeling
- **Advanced Indexing**: Vector embeddings for semantic search
- **Memory Consolidation**: Automatic knowledge distillation
- **Distributed Memory**: Multi-agent memory synchronization

## API Reference

### MemoryStore Interface

- `put(key, value, metadata) -> bool`: Store entry
- `get(key) -> MemoryEntry`: Retrieve entry
- `query(query) -> List[MemoryEntry]`: Filter entries
- `delete(key) -> bool`: Remove entry
- `compact(limit) -> int`: Remove old entries
- `clear() -> None`: Remove all entries
- `count() -> int`: Get entry count

### MemoryEntry Fields

- `key: str`: Unique identifier
- `value: Any`: Stored data
- `metadata: Dict`: Standardized metadata
- `timestamp: datetime`: Creation time

### Metadata Structure

```python
{
    "type": "episodic" | "semantic" | "system",
    "source": "agent_name",
    "phase": int,
    "tags": [str],
    "importance": float
}
```

## Testing

Run the demo to verify functionality:

```bash
python demo_memory.py
```

## Dependencies

- Python 3.8+
- sqlite3 (built-in)
- asyncio (built-in)
- typing (built-in)
- math, collections (built-in)

---

**Kalki v2.3 — Memory Module v1.0**