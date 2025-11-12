#!/usr/bin/env python3
"""
Emotional Intelligence Agent (Phase 15)
Implements empathy analytics, emotional maturity modeling, and persistence.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from ..base_agent import AgentCapability, AgentStatus, BaseAgent
from ...utils.eventbus import EventBus
from ...analytics import SentimentAnalyzer
from ..memory.memory_agent import MemoryAgent

logger = logging.getLogger("kalki.agent.emotional")


@dataclass
class EmotionalSnapshot:
    timestamp: datetime
    sentiment: float
    primary_emotion: str
    intensity: float
    empathy: float
    stress: float
    curiosity: float
    notes: str = ""


@dataclass
class EmotionalProfile:
    user_id: str
    empathy: float = 50.0
    resilience: float = 50.0
    curiosity: float = 50.0
    stress: float = 25.0
    joy: float = 50.0
    maturity: float = 50.0
    history: List[EmotionalSnapshot] = field(default_factory=list)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_from_snapshot(self, snapshot: EmotionalSnapshot, guardrails: Dict[str, Any]) -> None:
        """Update emotional profile using exponential smoothing with guardrails."""
        alpha = guardrails.get("smoothing_factor", 0.15)
        self.empathy = self._bounded_update(self.empathy, snapshot.empathy, alpha, guardrails)
        self.resilience = self._bounded_update(
            self.resilience,
            (1.0 - snapshot.stress) * 100.0,
            alpha,
            guardrails,
        )
        self.curiosity = self._bounded_update(self.curiosity, snapshot.curiosity, alpha, guardrails)
        self.stress = self._bounded_update(self.stress, snapshot.stress * 100.0, alpha, guardrails)
        self.joy = self._bounded_update(self.joy, max(snapshot.sentiment * 100.0, 0.0), alpha, guardrails)

        # Emotional maturity as composite of empathy, resilience, and joy
        target_maturity = (self.empathy + self.resilience + self.joy) / 3.0
        self.maturity = self._bounded_update(self.maturity, target_maturity, alpha / 2.0, guardrails)

        self.updated_at = datetime.utcnow()
        self.history.append(snapshot)

        history_limit = guardrails.get("history_limit", 200)
        if len(self.history) > history_limit:
            self.history = self.history[-history_limit:]

    def _bounded_update(self, current: float, target: float, alpha: float, guardrails: Dict[str, Any]) -> float:
        delta = (target - current) * alpha
        max_delta = guardrails.get("max_delta", 8.0)
        bounded_delta = max(-max_delta, min(max_delta, delta))
        new_value = current + bounded_delta
        lower = guardrails.get("min_value", 0.0)
        upper = guardrails.get("max_value", 100.0)
        return max(lower, min(upper, new_value))


class EmotionalProfileStore:
    """Persists emotional profiles in dedicated storage."""

    def __init__(self, memory_agent: MemoryAgent):
        self.memory_agent = memory_agent
        self._profiles: Dict[str, EmotionalProfile] = {}
        self._lock = asyncio.Lock()

    async def load_profile(self, user_id: str) -> EmotionalProfile:
        async with self._lock:
            if user_id in self._profiles:
                return self._profiles[user_id]

            profile_data = await self._load_profile_from_storage(user_id)
            if profile_data:
                profile = EmotionalProfile(**profile_data)
            else:
                profile = EmotionalProfile(user_id=user_id)

            self._profiles[user_id] = profile
            return profile

    async def save_profile(self, profile: EmotionalProfile) -> None:
        async with self._lock:
            await self._persist_profile(profile)
            self._profiles[profile.user_id] = profile

    async def _load_profile_from_storage(self, user_id: str) -> Optional[Dict[str, Any]]:
        task = {
            "action": "recall",
            "type": "semantic",
            "concept": f"emotional_profile::{user_id}",
        }
        result = await self.memory_agent.execute(task)
        memories = result.get("memories", [])
        if not memories:
            return None

        # Use most recent memory
        latest = max(memories, key=lambda m: m.get("timestamp", ""))
        return latest.get("knowledge", {})

    async def _persist_profile(self, profile: EmotionalProfile) -> None:
        knowledge = {
            "user_id": profile.user_id,
            "empathy": profile.empathy,
            "resilience": profile.resilience,
            "curiosity": profile.curiosity,
            "stress": profile.stress,
            "joy": profile.joy,
            "maturity": profile.maturity,
            "updated_at": profile.updated_at.isoformat(),
            "history": [
                {
                    "timestamp": snapshot.timestamp.isoformat(),
                    "sentiment": snapshot.sentiment,
                    "primary_emotion": snapshot.primary_emotion,
                    "intensity": snapshot.intensity,
                    "empathy": snapshot.empathy,
                    "stress": snapshot.stress,
                    "curiosity": snapshot.curiosity,
                    "notes": snapshot.notes,
                }
                for snapshot in profile.history
            ],
        }

        task = {
            "action": "store",
            "type": "semantic",
            "concept": f"emotional_profile::{profile.user_id}",
            "knowledge": knowledge,
        }
        await self.memory_agent.execute(task)


class EmotionalIntelligenceAgent(BaseAgent):
    """Phase 15 agent providing empathy analytics and emotional maturity modeling."""

    DEFAULT_GUARDRAILS = {
        "min_value": 0.0,
        "max_value": 100.0,
        "max_delta": 5.0,
        "smoothing_factor": 0.2,
        "history_limit": 100,
        "event_thresholds": {
            "stress_high": 0.75,
            "stress_critical": 0.9,
            "empathy_drop": -0.15,
            "curiosity_spike": 0.2,
        },
    }

    def __init__(self, memory_agent: MemoryAgent, event_bus: Optional[EventBus] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="EmotionalIntelligenceAgent",
            capabilities=[AgentCapability.EMOTIONAL_STATE, AgentCapability.PERSONA],
            description="Analyzes user emotion, maintains Kalki's emotional maturity profile, and publishes empathy insights",
            config=config or {},
        )
        self.memory_agent = memory_agent
        self.event_bus = event_bus or EventBus()
        self.guardrails = {**self.DEFAULT_GUARDRAILS, **(config or {}).get("guardrails", {})}
        self.profile_store = EmotionalProfileStore(memory_agent)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.active_profiles: Dict[str, EmotionalProfile] = {}
        self.current_session_profiles: Dict[str, EmotionalProfile] = {}
        self.profile_cache_ttl = self.config.get("profile_cache_ttl", 3600)
        self.session_only_mode = self.config.get("session_only_mode", False)

    async def initialize(self) -> bool:
        self.update_status(AgentStatus.READY)
        return True

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        action = task.get("action")
        if action == "analyze" or action == "ingest":
            return await self._handle_analysis(task)
        if action == "get_state":
            return await self._handle_get_state(task)
        if action == "reset":
            return await self._handle_reset(task)
        return {"status": "error", "message": f"Unknown action: {action}"}

    async def shutdown(self) -> bool:
        self.update_status(AgentStatus.TERMINATED)
        return True

    async def _handle_analysis(self, task: Dict[str, Any]) -> Dict[str, Any]:
        user_id = task.get("user_id", "default")
        text = task.get("text", "")
        metadata = task.get("metadata", {})
        emit_events = task.get("emit_events", True)

        if not text:
            return {"status": "error", "message": "Text is required for emotional analysis"}

        profile = await self._get_profile(user_id)

        # Analyze sentiment/emotion
        analysis = await self.sentiment_analyzer.analyze(text)
        snapshot = self._build_snapshot(analysis, metadata)

        # Update profile and persist if required
        profile.update_from_snapshot(snapshot, self.guardrails)
        self.current_session_profiles[user_id] = profile

        if not self.session_only_mode:
            await self.profile_store.save_profile(profile)

        if emit_events:
            await self._emit_events(user_id, profile, snapshot)

        return {
            "status": "success",
            "analysis": {
                "sentiment": snapshot.sentiment,
                "emotion": snapshot.primary_emotion,
                "intensity": snapshot.intensity,
                "stress": snapshot.stress,
                "curiosity": snapshot.curiosity,
                "empathy": snapshot.empathy,
            },
            "profile": self._profile_to_dict(profile),
        }

    async def _handle_get_state(self, task: Dict[str, Any]) -> Dict[str, Any]:
        user_id = task.get("user_id", "default")
        profile = await self._get_profile(user_id)
        return {"status": "success", "profile": self._profile_to_dict(profile)}

    async def _handle_reset(self, task: Dict[str, Any]) -> Dict[str, Any]:
        user_id = task.get("user_id", "default")
        reset_persistent = task.get("reset_persistent", False)

        profile = EmotionalProfile(user_id=user_id)
        self.current_session_profiles[user_id] = profile

        if reset_persistent and not self.session_only_mode:
            await self.profile_store.save_profile(profile)

        return {"status": "success", "profile": self._profile_to_dict(profile)}

    async def _get_profile(self, user_id: str) -> EmotionalProfile:
        if user_id in self.current_session_profiles:
            return self.current_session_profiles[user_id]

        if user_id in self.active_profiles:
            return self.active_profiles[user_id]

        profile = await self.profile_store.load_profile(user_id)
        self.active_profiles[user_id] = profile
        return profile

    def _build_snapshot(self, analysis: Dict[str, Any], metadata: Dict[str, Any]) -> EmotionalSnapshot:
        sentiment_score = analysis.get("sentiment_score", 0.0)
        primary_emotion = analysis.get("primary_emotion", "neutral")
        intensity = analysis.get("emotion_intensity", 0.5)
        stress_level = analysis.get("stress_level", 0.2)
        curiosity_level = analysis.get("curiosity_level", 0.5)
        empathy_score = analysis.get("empathy_alignment", 0.6) * 100.0

        return EmotionalSnapshot(
            timestamp=datetime.utcnow(),
            sentiment=sentiment_score,
            primary_emotion=primary_emotion,
            intensity=intensity,
            empathy=empathy_score,
            stress=stress_level,
            curiosity=curiosity_level * 100.0,
            notes=metadata.get("notes", ""),
        )

    async def _emit_events(self, user_id: str, profile: EmotionalProfile, snapshot: EmotionalSnapshot) -> None:
        thresholds = self.guardrails.get("event_thresholds", {})

        payload = {
            "user_id": user_id,
            "timestamp": snapshot.timestamp.isoformat(),
            "analysis": {
                "sentiment": snapshot.sentiment,
                "primary_emotion": snapshot.primary_emotion,
                "intensity": snapshot.intensity,
                "stress": snapshot.stress,
                "curiosity": snapshot.curiosity,
                "empathy": snapshot.empathy,
            },
            "profile": self._profile_to_dict(profile),
        }

        await self.emit_event("emotional.update", payload)

        if snapshot.stress >= thresholds.get("stress_high", 0.75):
            await self.emit_event("emotional.alert.stress", {**payload, "level": "high"})
        if snapshot.stress >= thresholds.get("stress_critical", 0.9):
            await self.emit_event("emotional.alert.stress", {**payload, "level": "critical"})
        if snapshot.empathy - profile.empathy <= thresholds.get("empathy_drop", -0.15):
            await self.emit_event("emotional.alert.empathy_drop", payload)
        if snapshot.curiosity - profile.curiosity >= thresholds.get("curiosity_spike", 0.2):
            await self.emit_event("emotional.alert.curiosity", payload)

    def _profile_to_dict(self, profile: EmotionalProfile) -> Dict[str, Any]:
        return {
            "user_id": profile.user_id,
            "empathy": profile.empathy,
            "resilience": profile.resilience,
            "curiosity": profile.curiosity,
            "stress": profile.stress,
            "joy": profile.joy,
            "maturity": profile.maturity,
            "updated_at": profile.updated_at.isoformat(),
            "history": [
                {
                    "timestamp": snap.timestamp.isoformat(),
                    "sentiment": snap.sentiment,
                    "primary_emotion": snap.primary_emotion,
                    "intensity": snap.intensity,
                    "empathy": snap.empathy,
                    "stress": snap.stress,
                    "curiosity": snap.curiosity,
                    "notes": snap.notes,
                }
                for snap in profile.history
            ],
        }
