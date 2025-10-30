"""
Phase 14 - Episodic & Semantic Memory Layers
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field
import math
from collections import Counter

from .base import MemoryStore, MemoryEntry, MemoryQuery


@dataclass
class EpisodeEvent:
    """Represents a time-ordered episodic event."""
    event_id: str
    event_type: str
    data: Any
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Wrapper for episodic (time-ordered) memory."""
    
    def __init__(self, store: MemoryStore):
        """
        Initialize episodic memory with a memory store.
        
        Args:
            store: MemoryStore instance to use for storage
        """
        self.store = store
        self._event_counter = 0
    
    def add_event(self, event_type: str, data: Any, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add an episodic event.
        
        Args:
            event_type: Type/category of the event
            data: Event data
            metadata: Optional metadata
            
        Returns:
            Event ID (unique identifier)
        """
        self._event_counter += 1
        event_id = f"event_{self._event_counter}_{datetime.now().timestamp()}"
        
        event = EpisodeEvent(
            event_id=event_id,
            event_type=event_type,
            data=data,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        # Store as JSON-serializable dict
        event_data = {
            'event_id': event.event_id,
            'event_type': event.event_type,
            'data': event.data,
            'timestamp': event.timestamp.isoformat(),
            'metadata': event.metadata
        }
        
        self.store.put(
            event_id,
            event_data,
            metadata={'type': 'episodic', 'event_type': event_type}
        )
        
        return event_id
    
    def get_recent_episodes(self, limit: int = 10, event_type: Optional[str] = None) -> List[EpisodeEvent]:
        """
        Get recent episodic events.
        
        Args:
            limit: Maximum number of events to return
            event_type: Optional filter by event type
            
        Returns:
            List of EpisodeEvent objects
        """
        query_filter = {'type': 'episodic'}
        if event_type:
            query_filter['event_type'] = event_type
        
        query = MemoryQuery(filter=query_filter, limit=limit)
        entries = self.store.query(query)
        
        events = []
        for entry in entries:
            data = entry.value
            events.append(EpisodeEvent(
                event_id=data['event_id'],
                event_type=data['event_type'],
                data=data['data'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                metadata=data['metadata']
            ))
        
        return events


class SemanticMemory:
    """Wrapper for semantic (vector-similarity) memory using TF-IDF-lite."""
    
    def __init__(self, store: MemoryStore):
        """
        Initialize semantic memory with a memory store.
        
        Args:
            store: MemoryStore instance to use for storage
        """
        self.store = store
        self._doc_counter = 0
    
    def add_document(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a document to semantic memory.
        
        Args:
            text: Document text
            metadata: Optional metadata
            
        Returns:
            Document ID
        """
        self._doc_counter += 1
        doc_id = f"doc_{self._doc_counter}_{datetime.now().timestamp()}"
        
        # Store document with tokens for similarity search
        tokens = self._tokenize(text)
        doc_data = {
            'doc_id': doc_id,
            'text': text,
            'tokens': tokens,
            'metadata': metadata or {}
        }
        
        self.store.put(
            doc_id,
            doc_data,
            metadata={'type': 'semantic'}
        )
        
        return doc_id
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar documents using cosine similarity over token counts.
        
        Args:
            query: Query text
            limit: Maximum number of results
            
        Returns:
            List of results with 'doc_id', 'text', 'score', and 'metadata'
        """
        query_tokens = self._tokenize(query)
        query_vector = self._tokens_to_vector(query_tokens)
        
        # Get all semantic documents
        mem_query = MemoryQuery(filter={'type': 'semantic'})
        entries = self.store.query(mem_query)
        
        # Calculate similarities
        results = []
        for entry in entries:
            doc_data = entry.value
            doc_tokens = doc_data['tokens']
            doc_vector = self._tokens_to_vector(doc_tokens)
            
            similarity = self._cosine_similarity(query_vector, doc_vector)
            
            results.append({
                'doc_id': doc_data['doc_id'],
                'text': doc_data['text'],
                'score': similarity,
                'metadata': doc_data['metadata']
            })
        
        # Sort by similarity (highest first) and apply limit
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:limit]
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase and split on whitespace."""
        return text.lower().split()
    
    def _tokens_to_vector(self, tokens: List[str]) -> Dict[str, int]:
        """Convert token list to frequency vector."""
        return dict(Counter(tokens))
    
    def _cosine_similarity(self, vec1: Dict[str, int], vec2: Dict[str, int]) -> float:
        """
        Calculate cosine similarity between two token frequency vectors.
        
        Args:
            vec1: First vector (token -> count)
            vec2: Second vector (token -> count)
            
        Returns:
            Cosine similarity score (0 to 1)
        """
        if not vec1 or not vec2:
            return 0.0
        
        # Calculate dot product
        dot_product = sum(vec1.get(token, 0) * vec2.get(token, 0) for token in set(vec1) | set(vec2))
        
        # Calculate magnitudes
        mag1 = math.sqrt(sum(count ** 2 for count in vec1.values()))
        mag2 = math.sqrt(sum(count ** 2 for count in vec2.values()))
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
