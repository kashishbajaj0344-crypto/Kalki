"""
modules/eventbus.py
KALKI v2.3 — Robust asyncio-friendly in-process event bus.
Supports async handlers, safe publish scheduling, logging, and version registry.
"""

import asyncio
import time
from collections import deque
from typing import Any, Awaitable, Callable, Dict, List, Optional
from modules.config import register_module_version
from modules.logger import get_logger

__version__ = "Kalki v2.3 - modules/eventbus.py - v0.3"
register_module_version("eventbus.py", __version__)

logger = get_logger("eventbus")

Handler = Callable[[Any], Awaitable[None]]


class EventBus:
    """
    In-process event bus supporting async handler subscriptions and payload delivery.
    """

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Handler]] = {}
        self._loop = self._get_or_create_loop()
        # instrumentation
        self._event_history = deque(maxlen=1000)
        self._published_events = 0
        self._last_event_ts: Optional[float] = None

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
    # Subscription management
    # ------------------------------
    def subscribe(self, topic: str, handler: Handler) -> None:
        """
        Subscribe an async handler to a topic.
        """
        self._handlers.setdefault(topic, []).append(handler)
        logger.debug("Handler subscribed to topic '%s': %s", topic, handler.__name__)

    def unsubscribe(self, topic: str, handler: Handler) -> None:
        """
        Unsubscribe a handler from a topic.
        """
        if topic in self._handlers and handler in self._handlers[topic]:
            self._handlers[topic].remove(handler)
            logger.debug("Handler unsubscribed from topic '%s': %s", topic, handler.__name__)

    def clear(self) -> None:
        """
        Remove all registered handlers (useful for teardown/testing).
        """
        self._handlers.clear()
        logger.debug("All event handlers cleared.")

    # ------------------------------
    # Event publishing
    # ------------------------------
    async def publish(self, topic: str, payload: Any) -> None:
        """
        Publish a payload to all async handlers for the topic.
        """
        handlers = list(self._handlers.get(topic, []))
        if not handlers:
            logger.debug("No handlers for topic '%s'", topic)
            return
        # record basic telemetry
        try:
            self._published_events += 1
            self._event_history.append((time.time(), topic))
            self._last_event_ts = time.time()
        except Exception:
            # instrumentation must not break event delivery
            logger.debug("Failed to record event telemetry for topic %s", topic)

        logger.debug("Publishing event '%s' to %d handler(s)", topic, len(handlers))
        for handler in handlers:
            try:
                await handler(payload)
            except Exception as e:
                logger.exception("Event handler '%s' raised: %s", handler.__name__, e)

    def publish_sync(self, topic: str, payload: Any) -> None:
        """
        Synchronous convenience wrapper (fire-and-forget).
        Ensures coroutine scheduling even outside async contexts.
        """
        try:
            if self._loop.is_running():
                self._loop.create_task(self.publish(topic, payload))
            else:
                self._loop.run_until_complete(self.publish(topic, payload))
            keys = list(getattr(payload, "keys", lambda: [])())
            logger.debug("Published sync event '%s' (payload keys: %s)", topic, keys)
        except Exception as e:
            logger.error(f"Failed to publish sync event '{topic}': {e}")
        else:
            # reflect in telemetry for sync calls as well
            try:
                self._published_events += 1
                self._event_history.append((time.time(), topic))
                self._last_event_ts = time.time()
            except Exception:
                pass

    # ------------------------------
    # Introspection
    # ------------------------------
    def topics(self) -> List[str]:
        """
        Returns a list of all topics with registered handlers.
        """
        return list(self._handlers.keys())

    async def clear_history(self) -> None:
        """
        Async-friendly wrapper to clear event handler history.
        Kept for compatibility with callers expecting an awaitable cleanup method.
        """
        self.clear()

    def get_stats(self) -> Dict[str, int]:
        """
        Return a small dict with event-bus statistics for monitoring.
        """
        # canonical values
        topics_list = list(self._handlers.keys())
        handlers_count = sum(len(h) for h in self._handlers.values())

        stats: Dict[str, Any] = {
            # new, explicit fields
            "schema_version": "1.0",
            "total_subscribers": handlers_count,
            "event_types": topics_list,
            "history_size": len(self._event_history),
            "published_events": self._published_events,
            "last_event_timestamp": self._last_event_ts,
            # legacy aliases for older consumers
            "topics": len(topics_list),
            "handlers": handlers_count,
        }

        return stats
