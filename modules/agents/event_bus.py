"""
Event Bus for inter-agent communication
Provides pub/sub pattern for agent coordination
"""
import asyncio
import logging
from typing import Dict, Any, Callable, List, Optional
from collections import defaultdict
from datetime import datetime


class EventBus:
    """
    Event bus for asynchronous inter-agent communication
    Supports pub/sub pattern with topic-based routing
    """
    
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.event_history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self.logger = logging.getLogger("kalki.eventbus")
        self._lock = asyncio.Lock()
    
    async def publish(self, event_type: str, data: Dict[str, Any]):
        """
        Publish an event to all subscribers
        
        Args:
            event_type: Type of event (topic)
            data: Event data dictionary
        """
        event = {
            "event_type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        
        # Store in history
        async with self._lock:
            self.event_history.append(event)
            if len(self.event_history) > self.max_history:
                self.event_history.pop(0)
        
        # Notify subscribers
        subscribers = self.subscribers.get(event_type, [])
        if subscribers:
            self.logger.debug(f"Publishing event {event_type} to {len(subscribers)} subscribers")
            tasks = [asyncio.create_task(callback(event)) for callback in subscribers]
            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            self.logger.debug(f"No subscribers for event {event_type}")
    
    async def subscribe(self, event_type: str, callback: Callable):
        """
        Subscribe to an event type
        
        Args:
            event_type: Type of event to subscribe to
            callback: Async callback function to handle events
        """
        async with self._lock:
            if callback not in self.subscribers[event_type]:
                self.subscribers[event_type].append(callback)
                self.logger.debug(f"Added subscriber to {event_type}")
    
    async def unsubscribe(self, event_type: str, callback: Callable):
        """
        Unsubscribe from an event type
        
        Args:
            event_type: Type of event to unsubscribe from
            callback: Callback function to remove
        """
        async with self._lock:
            if callback in self.subscribers[event_type]:
                self.subscribers[event_type].remove(callback)
                self.logger.debug(f"Removed subscriber from {event_type}")
    
    def get_event_history(self, event_type: Optional[str] = None, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get event history
        
        Args:
            event_type: Optional filter by event type
            limit: Maximum number of events to return
            
        Returns:
            List of events
        """
        events = self.event_history
        if event_type:
            events = [e for e in events if e["event_type"] == event_type]
        return events[-limit:]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get event bus statistics"""
        return {
            "total_subscribers": sum(len(subs) for subs in self.subscribers.values()),
            "event_types": list(self.subscribers.keys()),
            "history_size": len(self.event_history),
            "max_history": self.max_history
        }
    
    async def clear_history(self):
        """Clear event history"""
        async with self._lock:
            self.event_history.clear()
            self.logger.info("Event history cleared")