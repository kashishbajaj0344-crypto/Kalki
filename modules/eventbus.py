"""
modules/eventbus.py
KALKI v2.3 — Robust asyncio-friendly in-process event bus.
Supports async handlers, safe publish scheduling, logging, and version registry.
Enhanced for Phase 3: Better error handling, message validation, routing guarantees.
"""

import asyncio
import time
import json
from collections import deque
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from modules.config import register_module_version
from modules.logger import get_logger

__version__ = "Kalki v2.3 - modules/eventbus.py - v0.4"
register_module_version("eventbus.py", __version__)

logger = get_logger("eventbus")

# Type alias for event handlers
EventHandler = Callable[[Any], Awaitable[None]]


class EventBus:
    """
    Enhanced in-process event bus supporting async handler subscriptions and payload delivery.
    Phase 3 enhancements: Guaranteed routing, error isolation, message validation, telemetry.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[EventHandler]] = {}
        self._handler_metadata: Dict[str, Dict[str, Any]] = {}  # Track handler info
        self._loop = self._get_or_create_loop()
        # Enhanced instrumentation
        self._event_history = deque(maxlen=1000)
        self._published_events = 0
        self._delivered_events = 0
        self._failed_deliveries = 0
        self._last_event_ts: Optional[float] = None
        self._active_subscriptions: Set[str] = set()
        self._event_routing_errors: Dict[str, int] = {}

    # ------------------------------
    # Loop management
    # ------------------------------
    @staticmethod
    def _get_or_create_loop() -> asyncio.AbstractEventLoop:
        """
        Ensures a valid running event loop exists for scheduling tasks.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        return loop

    # ------------------------------
    # Enhanced subscription management
    # ------------------------------
    def subscribe(self, topic: str, handler: EventHandler, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Subscribe an async handler to a topic with metadata tracking.

        Args:
            topic: Event topic to subscribe to
            handler: Async handler function
            metadata: Optional metadata about the handler (agent_name, priority, etc.)

        Returns:
            Subscription ID for potential future unsubscription
        """
        if not callable(handler):
            raise ValueError(f"Handler must be callable, got {type(handler)}")

        self._handlers.setdefault(topic, []).append(handler)

        # Generate subscription ID
        subscription_id = f"{topic}:{handler.__name__}:{id(handler)}"
        self._active_subscriptions.add(subscription_id)

        # Store metadata
        self._handler_metadata[subscription_id] = {
            "topic": topic,
            "handler_name": handler.__name__,
            "handler_id": id(handler),
            "subscribed_at": time.time(),
            "metadata": metadata or {}
        }

        logger.info("Handler '%s' subscribed to topic '%s' (subscription: %s)",
                   handler.__name__, topic, subscription_id)
        return subscription_id

    def unsubscribe(self, topic: str, handler: EventHandler) -> bool:
        """
        Unsubscribe a handler from a topic.

        Returns:
            True if handler was found and removed, False otherwise
        """
        if topic in self._handlers and handler in self._handlers[topic]:
            self._handlers[topic].remove(handler)

            # Remove from metadata
            subscription_id = f"{topic}:{handler.__name__}:{id(handler)}"
            self._active_subscriptions.discard(subscription_id)
            self._handler_metadata.pop(subscription_id, None)

            logger.info("Handler '%s' unsubscribed from topic '%s'", handler.__name__, topic)
            return True

        logger.warning("Handler '%s' not found for topic '%s'", handler.__name__, topic)
        return False

    def unsubscribe_by_id(self, subscription_id: str) -> bool:
        """
        Unsubscribe using subscription ID.

        Returns:
            True if subscription was found and removed, False otherwise
        """
        if subscription_id not in self._handler_metadata:
            return False

        metadata = self._handler_metadata[subscription_id]
        topic = metadata["topic"]
        handler_name = metadata["handler_name"]

        # Find and remove handler
        if topic in self._handlers:
            for handler in self._handlers[topic]:
                if handler.__name__ == handler_name and id(handler) == metadata["handler_id"]:
                    self._handlers[topic].remove(handler)
                    break

        self._active_subscriptions.discard(subscription_id)
        del self._handler_metadata[subscription_id]

        logger.info("Subscription '%s' removed", subscription_id)
        return True

    def clear_topic(self, topic: str) -> int:
        """
        Remove all handlers for a specific topic.

        Returns:
            Number of handlers removed
        """
        if topic not in self._handlers:
            return 0

        removed_count = len(self._handlers[topic])

        # Remove from active subscriptions and metadata
        for handler in self._handlers[topic]:
            subscription_id = f"{topic}:{handler.__name__}:{id(handler)}"
            self._active_subscriptions.discard(subscription_id)
            self._handler_metadata.pop(subscription_id, None)

        del self._handlers[topic]
        logger.info("Cleared %d handlers for topic '%s'", removed_count, topic)
        return removed_count

    def clear(self) -> None:
        """
        Remove all registered handlers (useful for teardown/testing).
        """
        total_handlers = sum(len(handlers) for handlers in self._handlers.values())
        self._handlers.clear()
        self._handler_metadata.clear()
        self._active_subscriptions.clear()
        logger.info("All event handlers cleared (%d total)", total_handlers)

    # ------------------------------
    # Enhanced event publishing with guaranteed routing
    # ------------------------------
    async def publish(self, topic: str, payload: Any, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Publish a payload to all async handlers for the topic with guaranteed routing.

        Args:
            topic: Event topic
            payload: Event payload (must be JSON serializable for validation)
            timeout: Maximum time to wait for all handlers to complete

        Returns:
            Delivery report with success/failure statistics
        """
        handlers = list(self._handlers.get(topic, []))
        if not handlers:
            logger.debug("No handlers for topic '%s'", topic)
            return {
                "topic": topic,
                "handlers_count": 0,
                "delivered": 0,
                "failed": 0,
                "status": "no_handlers"
            }

        # Validate payload
        try:
            json.dumps(payload, default=str)  # Test JSON serialization
        except Exception as e:
            logger.error("Invalid payload for topic '%s': %s", topic, e)
            return {
                "topic": topic,
                "handlers_count": len(handlers),
                "delivered": 0,
                "failed": len(handlers),
                "status": "invalid_payload",
                "error": str(e)
            }

        # Record telemetry
        self._published_events += 1
        event_id = f"{topic}:{self._published_events}:{time.time()}"
        self._event_history.append((time.time(), topic, event_id))
        self._last_event_ts = time.time()

        logger.info("Publishing event '%s' to %d handler(s) (ID: %s)",
                   topic, len(handlers), event_id)

        # Execute all handlers with timeout and error isolation
        delivery_results = await self._deliver_to_handlers(topic, handlers, payload, timeout, event_id)

        # Update counters
        successful_deliveries = sum(1 for r in delivery_results if r["success"])
        failed_deliveries = len(delivery_results) - successful_deliveries

        self._delivered_events += successful_deliveries
        self._failed_deliveries += failed_deliveries

        # Track routing errors per topic
        if failed_deliveries > 0:
            self._event_routing_errors[topic] = self._event_routing_errors.get(topic, 0) + failed_deliveries

        result = {
            "topic": topic,
            "event_id": event_id,
            "handlers_count": len(handlers),
            "delivered": successful_deliveries,
            "failed": failed_deliveries,
            "status": "completed" if failed_deliveries == 0 else "partial_failure",
            "delivery_details": delivery_results
        }

        if failed_deliveries > 0:
            logger.warning("Event '%s' delivery incomplete: %d/%d handlers failed",
                          topic, failed_deliveries, len(handlers))
        else:
            logger.debug("Event '%s' delivered successfully to all %d handlers",
                        topic, len(handlers))

        return result

    async def _deliver_to_handlers(self, topic: str, handlers: List[EventHandler],
                                 payload: Any, timeout: float, event_id: str) -> List[Dict[str, Any]]:
        """
        Deliver event to all handlers with individual timeout and error isolation.
        """
        async def deliver_single(handler: EventHandler) -> Dict[str, Any]:
            handler_name = handler.__name__
            try:
                # Create task with timeout
                task = asyncio.create_task(handler(payload))
                await asyncio.wait_for(task, timeout=timeout)

                return {
                    "handler": handler_name,
                    "success": True,
                    "duration": None,  # Could add timing if needed
                    "error": None
                }

            except asyncio.TimeoutError:
                logger.error("Handler '%s' timed out for event '%s'", handler_name, event_id)
                return {
                    "handler": handler_name,
                    "success": False,
                    "duration": timeout,
                    "error": "timeout"
                }

            except Exception as e:
                logger.exception("Handler '%s' raised exception for event '%s': %s",
                               handler_name, event_id, e)
                return {
                    "handler": handler_name,
                    "success": False,
                    "duration": None,
                    "error": str(e)
                }

        # Execute all deliveries concurrently
        delivery_tasks = [deliver_single(handler) for handler in handlers]
        return await asyncio.gather(*delivery_tasks, return_exceptions=True)

    def publish_sync(self, topic: str, payload: Any, timeout: float = 30.0) -> Dict[str, Any]:
        """
        Synchronous convenience wrapper with guaranteed routing.
        Ensures coroutine scheduling even outside async contexts.

        Returns:
            Delivery report
        """
        try:
            if self._loop.is_running():
                # Create task and wait for completion
                task = self._loop.create_task(self.publish(topic, payload, timeout))
                future = asyncio.run_coroutine_threadsafe(task, self._loop)
                return future.result(timeout=timeout + 1.0)  # Add buffer for scheduling
            else:
                return self._loop.run_until_complete(self.publish(topic, payload, timeout))

        except Exception as e:
            logger.error("Failed to publish sync event '%s': %s", topic, e)
            return {
                "topic": topic,
                "status": "sync_error",
                "error": str(e)
            }

    # ------------------------------
    # Enhanced introspection and monitoring
    # ------------------------------
    def topics(self) -> List[str]:
        """
        Returns a list of all topics with registered handlers.
        """
        return list(self._handlers.keys())

    def get_topic_info(self, topic: str) -> Dict[str, Any]:
        """
        Get detailed information about a topic and its handlers.
        """
        handlers = self._handlers.get(topic, [])
        handler_info = []

        for handler in handlers:
            subscription_id = f"{topic}:{handler.__name__}:{id(handler)}"
            metadata = self._handler_metadata.get(subscription_id, {})
            handler_info.append({
                "name": handler.__name__,
                "subscription_id": subscription_id,
                "metadata": metadata
            })

        return {
            "topic": topic,
            "handler_count": len(handlers),
            "handlers": handler_info,
            "routing_errors": self._event_routing_errors.get(topic, 0)
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Return comprehensive event-bus statistics for monitoring.
        """
        topics_list = list(self._handlers.keys())
        handlers_count = sum(len(h) for h in self._handlers.values())

        stats = {
            # Schema and version
            "schema_version": "2.0",
            "eventbus_version": __version__,

            # Handler statistics
            "total_subscribers": handlers_count,
            "active_subscriptions": len(self._active_subscriptions),
            "topics_count": len(topics_list),
            "topics": topics_list,

            # Event statistics
            "published_events": self._published_events,
            "delivered_events": self._delivered_events,
            "failed_deliveries": self._failed_deliveries,
            "delivery_success_rate": (self._delivered_events / max(self._published_events, 1)) * 100,

            # History and timing
            "history_size": len(self._event_history),
            "last_event_timestamp": self._last_event_ts,

            # Error tracking
            "topics_with_errors": list(self._event_routing_errors.keys()),
            "total_routing_errors": sum(self._event_routing_errors.values()),

            # Legacy compatibility
            "event_types": topics_list,
            "handlers": handlers_count,
        }

        return stats

    async def clear_history(self) -> None:
        """
        Async-friendly wrapper to clear event handler history.
        """
        self._event_history.clear()
        self._event_routing_errors.clear()
        logger.info("Event history and routing errors cleared")

    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on the event bus.
        """
        issues = []

        if not self._handlers:
            issues.append("No event handlers registered")

        if self._failed_deliveries > self._published_events * 0.1:  # More than 10% failure rate
            issues.append(f"High failure rate: {self._failed_deliveries}/{self._published_events}")

        if len(self._event_routing_errors) > len(self._handlers) * 0.5:  # Many topics with errors
            issues.append("Many topics experiencing routing errors")

        return {
            "healthy": len(issues) == 0,
            "issues": issues,
            "stats": self.get_stats()
        }

