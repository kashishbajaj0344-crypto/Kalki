"""
Unit tests for Phase 14 - Episodic & Semantic Memory Layers
"""

import unittest
from datetime import datetime, timedelta

from modules.memory import (
    InMemoryStore,
    EpisodicMemory,
    SemanticMemory,
    EpisodeEvent
)


class TestEpisodicMemory(unittest.TestCase):
    """Tests for episodic memory layer."""
    
    def setUp(self):
        """Create fresh episodic memory for each test."""
        self.store = InMemoryStore()
        self.episodic = EpisodicMemory(self.store)
    
    def test_add_event(self):
        """Test adding episodic events."""
        event_id = self.episodic.add_event(
            "user_action",
            {"action": "click", "target": "button1"},
            {"user": "test_user"}
        )
        
        self.assertIsNotNone(event_id)
        self.assertTrue(event_id.startswith("event_"))
    
    def test_get_recent_episodes(self):
        """Test retrieving recent episodes."""
        # Add multiple events
        self.episodic.add_event("action", {"type": "click"})
        self.episodic.add_event("action", {"type": "scroll"})
        self.episodic.add_event("navigation", {"page": "home"})
        
        # Get all recent events
        recent = self.episodic.get_recent_episodes(limit=10)
        self.assertEqual(len(recent), 3)
        
        # Check they're EpisodeEvent objects
        for event in recent:
            self.assertIsInstance(event, EpisodeEvent)
    
    def test_get_recent_episodes_with_limit(self):
        """Test limit on recent episodes."""
        for i in range(10):
            self.episodic.add_event("test", {"index": i})
        
        recent = self.episodic.get_recent_episodes(limit=5)
        self.assertEqual(len(recent), 5)
    
    def test_get_recent_episodes_filtered_by_type(self):
        """Test filtering episodes by event type."""
        self.episodic.add_event("action", {"data": "a"})
        self.episodic.add_event("navigation", {"data": "n"})
        self.episodic.add_event("action", {"data": "b"})
        
        actions = self.episodic.get_recent_episodes(event_type="action")
        self.assertEqual(len(actions), 2)
        
        for event in actions:
            self.assertEqual(event.event_type, "action")


class TestSemanticMemory(unittest.TestCase):
    """Tests for semantic memory layer."""
    
    def setUp(self):
        """Create fresh semantic memory for each test."""
        self.store = InMemoryStore()
        self.semantic = SemanticMemory(self.store)
    
    def test_add_document(self):
        """Test adding documents to semantic memory."""
        doc_id = self.semantic.add_document(
            "This is a test document about Python programming",
            {"category": "programming"}
        )
        
        self.assertIsNotNone(doc_id)
        self.assertTrue(doc_id.startswith("doc_"))
    
    def test_search_similar_basic(self):
        """Test basic similarity search."""
        # Add documents
        self.semantic.add_document("Python is a programming language")
        self.semantic.add_document("Java is also a programming language")
        self.semantic.add_document("The weather is nice today")
        
        # Search for programming-related content
        results = self.semantic.search_similar("programming language", limit=2)
        
        self.assertEqual(len(results), 2)
        
        # Check that programming documents have higher scores
        for result in results:
            self.assertIn('doc_id', result)
            self.assertIn('text', result)
            self.assertIn('score', result)
            self.assertGreater(result['score'], 0)
    
    def test_search_similar_ordering(self):
        """Test that results are ordered by similarity."""
        self.semantic.add_document("machine learning artificial intelligence")
        self.semantic.add_document("machine learning")
        self.semantic.add_document("cooking recipes")
        
        results = self.semantic.search_similar("machine learning")
        
        # Results should be ordered by score (highest first)
        scores = [r['score'] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        
        # Most similar should be exact match
        self.assertIn("machine learning", results[0]['text'])
    
    def test_tokenization(self):
        """Test tokenization is case-insensitive."""
        tokens1 = self.semantic._tokenize("Python Programming")
        tokens2 = self.semantic._tokenize("python programming")
        
        self.assertEqual(tokens1, tokens2)
    
    def test_cosine_similarity(self):
        """Test cosine similarity calculation."""
        vec1 = {"hello": 1, "world": 1}
        vec2 = {"hello": 1, "world": 1}
        vec3 = {"goodbye": 1, "world": 1}
        vec4 = {"totally": 1, "different": 1}
        
        # Identical vectors should have similarity 1.0
        sim_identical = self.semantic._cosine_similarity(vec1, vec2)
        self.assertAlmostEqual(sim_identical, 1.0, places=5)
        
        # Partial overlap should have intermediate similarity
        sim_partial = self.semantic._cosine_similarity(vec1, vec3)
        self.assertGreater(sim_partial, 0)
        self.assertLess(sim_partial, 1.0)
        
        # No overlap should have similarity 0.0
        sim_none = self.semantic._cosine_similarity(vec1, vec4)
        self.assertEqual(sim_none, 0.0)
    
    def test_empty_query(self):
        """Test handling of empty query."""
        self.semantic.add_document("test document")
        
        results = self.semantic.search_similar("")
        # Empty query should return low scores
        if results:
            self.assertEqual(results[0]['score'], 0.0)


if __name__ == '__main__':
    unittest.main()
