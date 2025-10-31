#!/usr/bin/env python3
"""
DreamModeAgent - Specialized agent for creative dream generation

Manages dream sessions, resource allocation, and creative exploration with
safety controls and performance monitoring.
"""

import asyncio
import logging
import time
import random
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from collections import defaultdict

from ..base_agent import BaseAgent, AgentCapability
from ..memory.memory_agent import MemoryAgent
from ..cognitive.performance_monitor import PerformanceMonitorAgent
from ...eventbus import EventBus

logger = logging.getLogger("kalki.agents.dream_mode")


class DreamSession:
    """Represents a dream generation session"""

    def __init__(self, session_id: str, theme: Optional[str] = None,
                 max_dreams: int = 10, ttl_seconds: int = 3600):
        self.session_id = session_id
        self.theme = theme
        self.max_dreams = max_dreams
        self.ttl_seconds = ttl_seconds
        self.created_at = datetime.now()
        self.expires_at = self.created_at + timedelta(seconds=ttl_seconds)
        self.dreams_generated = 0
        self.dreams = []
        self.status = "active"
        self.resource_usage = {
            "cpu_time": 0.0,
            "memory_peak": 0,
            "api_calls": 0
        }

    def is_expired(self) -> bool:
        """Check if session has expired"""
        return datetime.now() > self.expires_at

    def can_generate_more(self) -> bool:
        """Check if more dreams can be generated"""
        return (self.dreams_generated < self.max_dreams and
                not self.is_expired() and
                self.status == "active")

    def add_dream(self, dream: Dict[str, Any]):
        """Add a generated dream to the session"""
        self.dreams.append(dream)
        self.dreams_generated += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        return {
            "session_id": self.session_id,
            "theme": self.theme,
            "dreams_generated": self.dreams_generated,
            "max_dreams": self.max_dreams,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "time_remaining_seconds": max(0, (self.expires_at - datetime.now()).total_seconds()),
            "status": self.status,
            "resource_usage": self.resource_usage
        }


class DreamModeAgent(BaseAgent):
    """
    Specialized agent for managing creative dream generation sessions.

    Features:
    - Session-based dream generation with resource controls
    - Rate limiting and quota management
    - Performance monitoring and analytics
    - Safety controls and content filtering
    - Async dream generation with concurrency limits
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="DreamModeAgent",
            capabilities=[AgentCapability.CREATIVE_SYNTHESIS],
            description="Manages creative dream generation with resource controls and safety",
            config=config
        )

        # Configuration
        self.max_concurrent_dreams = self.config.get('max_concurrent_dreams', 5)
        self.max_dreams_per_session = self.config.get('max_dreams_per_session', 20)
        self.default_session_ttl = self.config.get('default_session_ttl', 3600)  # 1 hour
        self.rate_limit_per_minute = self.config.get('rate_limit_per_minute', 10)
        self.enable_safety_filter = self.config.get('enable_safety_filter', True)
        self.enable_performance_monitoring = self.config.get('enable_performance_monitoring', True)

        # State
        self.active_sessions = {}
        self.session_history = []
        self.rate_limiter = defaultdict(list)  # user -> timestamps
        self.performance_stats = {
            "total_sessions": 0,
            "total_dreams": 0,
            "avg_dreams_per_session": 0.0,
            "session_success_rate": 0.0,
            "avg_session_duration": 0.0
        }

        # Concurrency control
        self.dream_semaphore = asyncio.Semaphore(self.max_concurrent_dreams)

        # Dependencies (set by orchestrator)
        self.creative_agent = None
        self.safety_policy_engine = None
        self.event_bus = None
        self.memory_agent = None
        self.performance_monitor = None

    def set_creative_agent(self, creative_agent):
        """Set the creative agent for dream generation"""
        self.creative_agent = creative_agent

    def set_safety_policy_engine(self, safety_engine):
        """Set the safety policy engine"""
        self.safety_policy_engine = safety_engine

    def set_event_bus(self, event_bus):
        """Set the event bus for broadcasting"""
        self.event_bus = event_bus

    def set_memory_agent(self, memory_agent: MemoryAgent):
        """Set the memory agent for episodic storage"""
        self.memory_agent = memory_agent

    def set_performance_monitor(self, performance_monitor: PerformanceMonitorAgent):
        """Set the performance monitor for metrics tracking"""
        self.performance_monitor = performance_monitor

    async def initialize(self) -> bool:
        """Initialize the dream mode agent"""
        try:
            logger.info("DreamModeAgent initializing")

            # Start cleanup task
            asyncio.create_task(self._cleanup_expired_sessions())

            logger.info("DreamModeAgent initialized successfully")
            return True

        except Exception as e:
            logger.exception(f"DreamModeAgent initialization failed: {e}")
            return False

    async def _cleanup_expired_sessions(self):
        """Periodically clean up expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes

                expired_sessions = []
                for session_id, session in self.active_sessions.items():
                    if session.is_expired():
                        expired_sessions.append(session_id)
                        session.status = "expired"

                for session_id in expired_sessions:
                    session = self.active_sessions.pop(session_id)
                    self.session_history.append(session)
                    logger.info(f"Cleaned up expired session: {session_id}")

                # Update performance stats
                if self.enable_performance_monitoring:
                    self._update_performance_stats()

            except Exception as e:
                logger.exception(f"Session cleanup failed: {e}")

    def _check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limits"""
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Clean old timestamps
        self.rate_limiter[user_id] = [
            ts for ts in self.rate_limiter[user_id] if ts > window_start
        ]

        # Check limit
        if len(self.rate_limiter[user_id]) >= self.rate_limit_per_minute:
            return False

        return True

    def _record_rate_limit(self, user_id: str):
        """Record a rate-limited operation"""
        now = time.time()
        self.rate_limiter[user_id].append(now)

    async def create_dream_session(self, user_id: str, theme: Optional[str] = None,
                                  max_dreams: Optional[int] = None,
                                  ttl_seconds: Optional[int] = None) -> Dict[str, Any]:
        """Create a new dream generation session"""
        try:
            # Check rate limit
            if not self._check_rate_limit(user_id):
                return {
                    "status": "rate_limited",
                    "error": f"Rate limit exceeded ({self.rate_limit_per_minute}/minute)"
                }

            # Record operation
            self._record_rate_limit(user_id)

            # Create session
            session_id = f"dream_session_{user_id}_{int(time.time())}"
            max_dreams = min(max_dreams or self.max_dreams_per_session, self.max_dreams_per_session)
            ttl_seconds = ttl_seconds or self.default_session_ttl

            session = DreamSession(session_id, theme, max_dreams, ttl_seconds)
            self.active_sessions[session_id] = session

            # Update stats
            self.performance_stats["total_sessions"] += 1

            # Emit event
            if self.event_bus:
                await self.event_bus.publish("dream_session.created", {
                    "agent": "DreamModeAgent",
                    "session_id": session_id,
                    "user_id": user_id,
                    "theme": theme,
                    "max_dreams": max_dreams,
                    "ttl_seconds": ttl_seconds
                })

            logger.info(f"Created dream session {session_id} for user {user_id}")
            return {
                "status": "success",
                "session_id": session_id,
                "session_info": session.get_stats()
            }

        except Exception as e:
            logger.exception(f"Failed to create dream session: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def generate_dream(self, session_id: str, domain: Optional[str] = None,
                           constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generate a dream within a session"""
        start_time = time.time()
        try:
            # Get session
            session = self.active_sessions.get(session_id)
            if not session:
                return {
                    "status": "error",
                    "error": f"Session {session_id} not found or expired"
                }

            # Check session status
            if not session.can_generate_more():
                reason = "expired" if session.is_expired() else "limit_reached"
                return {
                    "status": "session_limit",
                    "error": f"Session {reason}",
                    "session_stats": session.get_stats()
                }

            # Safety check
            if self.enable_safety_filter and self.safety_policy_engine:
                domain = domain or random.choice([
                    "technology", "art", "science", "business", "healthcare", "education"
                ])
                safety_result = await self.safety_policy_engine.assess_creative_content(domain, constraints)
                if not safety_result["approved"]:
                    return {
                        "status": "blocked",
                        "error": f"Content blocked by safety policy: {safety_result['reason']}",
                        "safety_violation": True
                    }

            # Generate dream with concurrency control
            async with self.dream_semaphore:
                start_time = time.time()

                # Use creative agent if available
                if self.creative_agent:
                    seed = random.randint(0, 2**32 - 1)
                    result = await self.creative_agent.generate_idea(
                        domain or random.choice([
                            "technology", "art", "science", "business", "healthcare", "education"
                        ]),
                        constraints,
                        seed
                    )

                    if result.get("status") == "success":
                        dream = result["idea"]
                        dream["session_id"] = session_id
                        dream["dream_theme"] = session.theme

                        # Record performance
                        generation_time = time.time() - start_time
                        session.resource_usage["cpu_time"] += generation_time

                        # Add to session
                        session.add_dream(dream)

                        # Update global stats
                        self.performance_stats["total_dreams"] += 1

                        # Store in memory agent
                        if self.memory_agent:
                            await self.memory_agent.store_episodic({
                                "event_type": "dream_generated",
                                "session_id": session_id,
                                "dream_id": dream["idea_id"],
                                "domain": dream["domain"],
                                "theme": session.theme,
                                "generation_time": generation_time,
                                "timestamp": datetime.now().isoformat()
                            })

                        # Record performance metrics
                        if self.performance_monitor:
                            self.performance_monitor.record_metric(
                                "dream_generation_duration",
                                generation_time,
                                {
                                    "session_id": session_id,
                                    "domain": dream["domain"],
                                    "theme": session.theme
                                }
                            )

                        # Emit event
                        if self.event_bus:
                            await self.event_bus.publish("dream.generated", {
                                "agent": "DreamModeAgent",
                                "session_id": session_id,
                                "dream_id": dream["idea_id"],
                                "domain": dream["domain"],
                                "generation_time": generation_time
                            })

                        return {
                            "status": "success",
                            "dream": dream,
                            "session_stats": session.get_stats()
                        }
                    else:
                        return result
                else:
                    return {
                        "status": "error",
                        "error": "Creative agent not available"
                    }

        except Exception as e:
            logger.exception(f"Dream generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a dream session"""
        try:
            session = self.active_sessions.get(session_id)
            if session:
                return {
                    "status": "success",
                    "session_info": session.get_stats(),
                    "dreams": session.dreams
                }
            else:
                # Check history
                for historical_session in self.session_history:
                    if historical_session.session_id == session_id:
                        return {
                            "status": "success",
                            "session_info": historical_session.get_stats(),
                            "dreams": historical_session.dreams,
                            "historical": True
                        }

                return {
                    "status": "not_found",
                    "error": f"Session {session_id} not found"
                }

        except Exception as e:
            logger.exception(f"Failed to get session status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a dream session"""
        try:
            session = self.active_sessions.pop(session_id, None)
            if session:
                session.status = "ended"
                self.session_history.append(session)

                # Update performance stats
                if self.enable_performance_monitoring:
                    self._update_performance_stats()

                # Emit event
                if self.event_bus:
                    await self.event_bus.publish("dream_session.ended", {
                        "agent": "DreamModeAgent",
                        "session_id": session_id,
                        "dreams_generated": session.dreams_generated,
                        "duration_seconds": (datetime.now() - session.created_at).total_seconds()
                    })

                logger.info(f"Ended dream session {session_id}")
                return {
                    "status": "success",
                    "session_stats": session.get_stats()
                }
            else:
                return {
                    "status": "not_found",
                    "error": f"Session {session_id} not found"
                }

        except Exception as e:
            logger.exception(f"Failed to end session: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _update_performance_stats(self):
        """Update performance statistics"""
        try:
            total_sessions = len(self.session_history)
            if total_sessions > 0:
                total_dreams = sum(s.dreams_generated for s in self.session_history)
                self.performance_stats["total_dreams"] = total_dreams
                self.performance_stats["avg_dreams_per_session"] = total_dreams / total_sessions

                # Session success rate (sessions that generated at least one dream)
                successful_sessions = sum(1 for s in self.session_history if s.dreams_generated > 0)
                self.performance_stats["session_success_rate"] = successful_sessions / total_sessions

                # Average session duration
                total_duration = sum(
                    (s.expires_at - s.created_at).total_seconds()
                    for s in self.session_history
                    if s.status == "ended"
                )
                ended_sessions = sum(1 for s in self.session_history if s.status == "ended")
                if ended_sessions > 0:
                    self.performance_stats["avg_session_duration"] = total_duration / ended_sessions

        except Exception as e:
            logger.exception(f"Failed to update performance stats: {e}")

    async def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            # Update stats first
            self._update_performance_stats()

            return {
                "status": "success",
                "stats": self.performance_stats.copy(),
                "active_sessions": len(self.active_sessions),
                "total_sessions_ever": len(self.session_history) + len(self.active_sessions)
            }

        except Exception as e:
            logger.exception(f"Failed to get performance stats: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute dream mode tasks"""
        action = task.get("action")

        try:
            if action == "create_session":
                return await self.create_dream_session(
                    task["user_id"],
                    task.get("theme"),
                    task.get("max_dreams"),
                    task.get("ttl_seconds")
                )
            elif action == "generate_dream":
                return await self.generate_dream(
                    task["session_id"],
                    task.get("domain"),
                    task.get("constraints")
                )
            elif action == "get_session_status":
                return await self.get_session_status(task["session_id"])
            elif action == "end_session":
                return await self.end_session(task["session_id"])
            elif action == "get_performance_stats":
                return await self.get_performance_stats()
            elif action == "list_active_sessions":
                return {
                    "status": "success",
                    "sessions": [s.get_stats() for s in self.active_sessions.values()]
                }
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception(f"Task execution failed: {e}")
            return {"status": "error", "message": str(e)}

    async def shutdown(self) -> bool:
        """Shutdown the dream mode agent"""
        try:
            logger.info("DreamModeAgent shutting down")

            # End all active sessions
            session_ids = list(self.active_sessions.keys())
            for session_id in session_ids:
                await self.end_session(session_id)

            # Clear state
            self.active_sessions.clear()
            self.session_history.clear()
            self.rate_limiter.clear()

            logger.info("DreamModeAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"DreamModeAgent shutdown failed: {e}")
            return False