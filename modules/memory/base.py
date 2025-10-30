"""
Base classes and interfaces for Kalki memory system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, List, Dict


@dataclass
class MemoryEntry:
    """Represents a single memory entry with key, value, metadata, and timestamp."""
    key: str
    value: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary format."""
        return {
            'key': self.key,
            'value': self.value,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryEntry':
        """Create entry from dictionary format."""
        return cls(
            key=data['key'],
            value=data['value'],
            metadata=data.get('metadata', {}),
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class MemoryQuery:
    """Query parameters for memory search."""
    filter: Optional[Dict[str, Any]] = None
    since: Optional[datetime] = None
    until: Optional[datetime] = None
    limit: Optional[int] = None
    keys: Optional[List[str]] = None


class MemoryStore(ABC):
    """Abstract base class for memory storage implementations."""
    
    @abstractmethod
    def put(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Store a key-value pair with optional metadata.
        
        Args:
            key: Unique identifier for the memory entry
            value: The value to store
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get(self, key: str) -> Optional[MemoryEntry]:
        """
        Retrieve a memory entry by key.
        
        Args:
            key: The key to retrieve
            
        Returns:
            MemoryEntry if found, None otherwise
        """
        pass
    
    @abstractmethod
    def query(self, query: MemoryQuery) -> List[MemoryEntry]:
        """
        Query memory entries based on filters.
        
        Args:
            query: MemoryQuery object with filter criteria
            
        Returns:
            List of matching MemoryEntry objects
        """
        pass
    
    @abstractmethod
    def delete(self, key: str) -> bool:
        """
        Delete a memory entry by key.
        
        Args:
            key: The key to delete
            
        Returns:
            True if deleted, False if not found
        """
        pass
    
    @abstractmethod
    def compact(self, limit: Optional[int] = None) -> int:
        """
        Remove old entries to keep storage size manageable.
        
        Args:
            limit: Maximum number of entries to keep (oldest removed first)
            
        Returns:
            Number of entries removed
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all entries from the store."""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Return the total number of entries in the store."""
        pass
