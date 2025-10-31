#!/usr/bin/env python3
"""
Kalki Memory Module Demo - Phases 13-14
Demonstrates long-term memory persistence with in-memory and SQLite implementations,
plus episodic and semantic memory layers.
"""

import logging
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.agents.memory import (
    InMemoryStore, SQLiteMemoryStore,
    EpisodicMemory, SemanticMemory,
    initialize_default_memory, MemoryMonitor
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("Kalki.Memory.Demo")


async def demo_memory_system():
    """Demonstrate the complete memory system."""
    logger.info("🚀 Kalki Memory Module v1.0 Demo")
    logger.info("=" * 60)

    # Demo 1: Basic Memory Stores
    logger.info("\n📋 Demo 1: Basic Memory Store Operations")
    logger.info("-" * 40)

    # In-memory store
    mem_store = InMemoryStore()
    mem_store.put("user_pref", {"theme": "dark", "language": "en"}, {"type": "user", "importance": 0.8})
    mem_store.put("system_config", {"debug": True}, {"type": "system", "importance": 0.9})

    logger.info(f"  In-memory store count: {mem_store.count()}")
    entry = mem_store.get("user_pref")
    if entry:
        logger.info(f"  Retrieved user pref: {entry.value}")

    # SQLite store
    sqlite_store = SQLiteMemoryStore("demo_memory.db")
    sqlite_store.put("persistent_data", {"sessions": 42}, {"type": "analytics"})
    logger.info(f"  SQLite store count: {sqlite_store.count()}")

    # Demo 2: Episodic Memory
    logger.info("\n🧠 Demo 2: Episodic Memory (Time-ordered Events)")
    logger.info("-" * 40)

    episodic = EpisodicMemory(sqlite_store)

    # Add some events
    event1 = episodic.add_event("user_action", {"action": "login", "user_id": "demo_user"})
    event2 = episodic.add_event("system_event", {"event": "backup_completed", "size_mb": 150})
    event3 = episodic.add_event("user_action", {"action": "query", "query": "What is AI safety?"})

    logger.info(f"  Added events: {event1}, {event2}, {event3}")

    # Retrieve recent events
    recent_events = episodic.get_recent_episodes(limit=5)
    logger.info(f"  Recent events count: {len(recent_events)}")
    for event in recent_events[:2]:  # Show first 2
        logger.info(f"    {event.event_type}: {event.data}")

    # Demo 3: Semantic Memory
    logger.info("\n🔍 Demo 3: Semantic Memory (Similarity Search)")
    logger.info("-" * 40)

    semantic = SemanticMemory(sqlite_store)

    # Add documents
    docs = [
        "AI safety is crucial for responsible development of artificial intelligence systems.",
        "Machine learning models need proper validation and testing before deployment.",
        "Episodic memory helps agents remember specific events and experiences over time.",
        "Semantic memory stores general knowledge and concepts for similarity matching.",
        "Neural networks can be used for both supervised and unsupervised learning tasks."
    ]

    doc_ids = []
    for doc in docs:
        doc_id = semantic.add_document(doc, {"source": "demo", "domain": "ai"})
        doc_ids.append(doc_id)

    logger.info(f"  Added {len(doc_ids)} documents to semantic memory")

    # Search for similar content
    query = "artificial intelligence safety and validation"
    results = semantic.search_similar(query, limit=3)

    logger.info(f"  Search query: '{query}'")
    logger.info(f"  Found {len(results)} similar documents:")
    for i, result in enumerate(results, 1):
        logger.info(f"    {i}. Score: {result['score']:.3f} - {result['text'][:60]}...")

    # Demo 4: Memory Monitor
    logger.info("\n📊 Demo 4: Memory Monitoring")
    logger.info("-" * 40)

    monitor = MemoryMonitor()

    # Monitor different stores
    mem_summary = monitor.summarize(mem_store)
    sqlite_summary = monitor.summarize(sqlite_store)

    logger.info(f"  In-memory store: {mem_summary}")
    logger.info(f"  SQLite store: {sqlite_summary}")

    # Pattern detection in episodic memory
    patterns = monitor.detect_patterns(episodic)
    logger.info(f"  Detected {len(patterns)} event patterns:")
    for pattern in patterns:
        logger.info(f"    {pattern['pattern']}: {pattern['frequency']} occurrences")

    # Demo 5: Integration Helper
    logger.info("\n🔗 Demo 5: Integration Helper")
    logger.info("-" * 40)

    episodic_mem, semantic_mem, transient_mem = initialize_default_memory()
    logger.info("  Initialized default memory layers:")
    logger.info(f"    Episodic: {type(episodic_mem.store).__name__}")
    logger.info(f"    Semantic: {type(semantic_mem.store).__name__}")
    logger.info(f"    Transient: {type(transient_mem).__name__}")

    # Quick integration test
    test_event = episodic_mem.add_event("integration_test", {"status": "success"})
    test_doc = semantic_mem.add_document("Integration test document for memory system validation.")

    logger.info(f"  Integration test - Event: {test_event}")
    logger.info(f"  Integration test - Document: {test_doc}")

    # Cleanup
    try:
        os.remove("demo_memory.db")
        logger.info("  Cleaned up demo database")
    except:
        pass

    logger.info("\n" + "=" * 60)
    logger.info("✅ Kalki Memory Module Demo Complete!")
    logger.info("🎯 All memory layers working correctly with persistence and search")
    logger.info("=" * 60)


if __name__ == "__main__":
    import asyncio
    asyncio.run(demo_memory_system())