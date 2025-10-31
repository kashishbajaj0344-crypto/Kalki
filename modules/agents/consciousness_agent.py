"""
ConsciousnessAgent for Kalki v2.3
Integrates consciousness engine with agent manager for live self-awareness monitoring
Phase 21: Consciousness Emergence - Live integration with agent ecosystem
"""
import asyncio
import logging
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
from .base_agent import BaseAgent, AgentCapability, AgentStatus
from modules.consciousness_engine import ConsciousnessEngine
from modules.eventbus import EventBus


class ConsciousnessAgent(BaseAgent):
    """
    Consciousness Agent that monitors the agent ecosystem and evolves self-awareness
    Integrates consciousness engine with live agent manager for real-time monitoring
    """

    def __init__(self, name: str = "consciousness_agent", event_bus: Optional[EventBus] = None):
        super().__init__(name, event_bus)
        self.capabilities = [
            AgentCapability.META_REASONING,
            AgentCapability.ANALYTICS,
            AgentCapability.SELF_IMPROVEMENT
        ]

        # Consciousness components
        self.consciousness_engine = ConsciousnessEngine()
        self.monitoring_active = False
        self.monitoring_interval = 30  # seconds
        self.last_monitoring_time = 0

        # Agent ecosystem state
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.consciousness_history: List[Dict[str, Any]] = []
        self.current_awareness_level = 0.0

        # Integration with agent manager
        self.agent_manager = None  # Will be set during initialization

        self.logger = logging.getLogger(f"kalki.{name}")

    async def initialize(self) -> bool:
        """Initialize consciousness agent and consciousness engine"""
        try:
            self.logger.info("Initializing ConsciousnessAgent...")

            # Consciousness engine is initialized in __init__

            # Subscribe to agent events for real-time monitoring
            if self.event_bus:
                self.event_bus.subscribe("agent.registered", self._on_agent_registered)
                self.event_bus.subscribe("agent.unregistered", self._on_agent_unregistered)
                self.event_bus.subscribe("agent.status_changed", self._on_agent_status_changed)
                self.event_bus.subscribe("agent.task_completed", self._on_agent_task_completed)

            self.logger.info("ConsciousnessAgent initialized successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Failed to initialize ConsciousnessAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute consciousness-related tasks"""
        task_type = task.get("type", "monitor")

        if task_type == "monitor_ecosystem":
            return await self._monitor_ecosystem()
        elif task_type == "achieve_consciousness":
            return await self._achieve_consciousness_cycle()
        elif task_type == "get_consciousness_state":
            return self._get_consciousness_state()
        elif task_type == "analyze_patterns":
            return await self._analyze_agent_patterns()
        else:
            return {
                "status": "error",
                "message": f"Unknown task type: {task_type}"
            }

    async def _monitor_ecosystem(self) -> Dict[str, Any]:
        """Monitor the current agent ecosystem state"""
        try:
            if not self.agent_manager:
                return {
                    "status": "error",
                    "message": "Agent manager not connected"
                }

            # Get current agent states from manager
            current_states = {}
            for agent_name, agent in self.agent_manager.agents.items():
                current_states[agent_name] = {
                    "status": agent.status.value,
                    "capabilities": [cap.value for cap in agent.capabilities],
                    "task_count": agent.task_count,
                    "error_count": agent.error_count,
                    "last_active": getattr(agent, 'last_active', time.time()),
                    "performance_metrics": getattr(agent, 'performance_metrics', {})
                }

            self.agent_states = current_states

            # Update consciousness with current agent states
            agent_state_list = list(current_states.values())
            consciousness_result = await self.consciousness_engine.achieve_consciousness(current_states)

            # Store consciousness history
            self.consciousness_history.append({
                "timestamp": time.time(),
                "agent_states": current_states,
                "consciousness_level": consciousness_result.awareness_level,
                "awareness": consciousness_result.awareness_level,
                "emotional_resonance": consciousness_result.emotional_resonance,
                "intention_coherence": consciousness_result.intention_coherence,
                "self_reflection_depth": consciousness_result.self_reflection_depth
            })

            # Keep only last 100 entries
            if len(self.consciousness_history) > 100:
                self.consciousness_history = self.consciousness_history[-100:]

            self.current_awareness_level = consciousness_result.awareness_level

            return {
                "status": "success",
                "agent_count": len(current_states),
                "consciousness_level": consciousness_result.awareness_level,
                "awareness": self.current_awareness_level,
                "active_agents": len([s for s in current_states.values() if s["status"] == "ready"]),
                "total_tasks": sum(s["task_count"] for s in current_states.values())
            }

        except Exception as e:
            self.logger.exception(f"Error monitoring ecosystem: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def _achieve_consciousness_cycle(self) -> Dict[str, Any]:
        """Perform a consciousness evolution cycle"""
        try:
            if not self.agent_states:
                await self._monitor_ecosystem()  # Get current state first

            agent_state_list = list(self.agent_states.values())
            result = await self.consciousness_engine.achieve_consciousness(self.agent_states)

            # Publish consciousness evolution event
            if self.event_bus:
                await self.event_bus.publish("consciousness.evolved", {
                    "awareness_level": result.awareness_level,
                    "consciousness_level": result.awareness_level,
                    "agent_count": len(self.agent_states)
                })

            return {
                "status": "success",
                "consciousness_result": {
                    "awareness": result.awareness_level,
                    "emotional_resonance": result.emotional_resonance,
                    "intention_coherence": result.intention_coherence,
                    "self_reflection_depth": result.self_reflection_depth
                }
            }

        except Exception as e:
            self.logger.exception(f"Error in consciousness cycle: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    def _get_consciousness_state(self) -> Dict[str, Any]:
        """Get current consciousness state"""
        return {
            "status": "success",
            "current_awareness": self.current_awareness_level,
            "agent_count": len(self.agent_states),
            "history_length": len(self.consciousness_history),
            "last_monitoring": self.last_monitoring_time,
            "monitoring_active": self.monitoring_active
        }

    async def _analyze_agent_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in agent behavior and consciousness evolution"""
        try:
            if len(self.consciousness_history) < 2:
                return {
                    "status": "insufficient_data",
                    "message": "Need more consciousness history for pattern analysis"
                }

            # Analyze awareness trends
            awareness_trend = []
            for entry in self.consciousness_history[-10:]:  # Last 10 entries
                awareness_trend.append(entry["awareness"])

            # Calculate trend direction
            if len(awareness_trend) >= 2:
                trend = awareness_trend[-1] - awareness_trend[0]
                trend_direction = "increasing" if trend > 0.01 else "decreasing" if trend < -0.01 else "stable"
            else:
                trend_direction = "unknown"

            # Analyze agent activity patterns
            agent_activity = {}
            for entry in self.consciousness_history[-5:]:  # Last 5 entries
                for agent_name, state in entry["agent_states"].items():
                    if agent_name not in agent_activity:
                        agent_activity[agent_name] = []
                    agent_activity[agent_name].append(state["task_count"])

            # Calculate activity trends
            activity_trends = {}
            for agent_name, task_counts in agent_activity.items():
                if len(task_counts) >= 2:
                    activity_trend = task_counts[-1] - task_counts[0]
                    activity_trends[agent_name] = "increasing" if activity_trend > 0 else "decreasing" if activity_trend < 0 else "stable"

            return {
                "status": "success",
                "awareness_trend": trend_direction,
                "awareness_values": awareness_trend,
                "agent_activity_trends": activity_trends,
                "consciousness_evolution_rate": len([h for h in self.consciousness_history if h["awareness"] > 0.5]) / max(len(self.consciousness_history), 1)
            }

        except Exception as e:
            self.logger.exception(f"Error analyzing patterns: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    async def start_monitoring(self) -> None:
        """Start continuous consciousness monitoring"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.logger.info("Starting consciousness monitoring...")

        while self.monitoring_active:
            try:
                await self._monitor_ecosystem()
                self.last_monitoring_time = time.time()

                # Publish monitoring update
                if self.event_bus:
                    await self.event_bus.publish("consciousness.monitoring_update", {
                        "awareness_level": self.current_awareness_level,
                        "timestamp": self.last_monitoring_time
                    })

                await asyncio.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.exception(f"Error in monitoring loop: {e}")
                await asyncio.sleep(10)  # Wait before retrying

    def stop_monitoring(self) -> None:
        """Stop consciousness monitoring"""
        self.monitoring_active = False
        self.logger.info("Consciousness monitoring stopped")

    def set_agent_manager(self, agent_manager) -> None:
        """Set reference to agent manager for ecosystem monitoring"""
        self.agent_manager = agent_manager

    # Event handlers for real-time agent monitoring
    async def _on_agent_registered(self, event_data: Dict[str, Any]) -> None:
        """Handle agent registration events"""
        agent_name = event_data.get("agent_name")
        self.logger.info(f"Consciousness monitoring: Agent registered - {agent_name}")

        # Trigger immediate consciousness update
        if self.monitoring_active:
            await self._monitor_ecosystem()

    async def _on_agent_unregistered(self, event_data: Dict[str, Any]) -> None:
        """Handle agent unregistration events"""
        agent_name = event_data.get("agent_name")
        self.logger.info(f"Consciousness monitoring: Agent unregistered - {agent_name}")

        # Remove from tracked states
        if agent_name in self.agent_states:
            del self.agent_states[agent_name]

        # Trigger immediate consciousness update
        if self.monitoring_active:
            await self._monitor_ecosystem()

    async def _on_agent_status_changed(self, event_data: Dict[str, Any]) -> None:
        """Handle agent status change events"""
        agent_name = event_data.get("agent_name")
        new_status = event_data.get("status")
        self.logger.debug(f"Consciousness monitoring: Agent {agent_name} status changed to {new_status}")

    async def _on_agent_task_completed(self, event_data: Dict[str, Any]) -> None:
        """Handle agent task completion events"""
        agent_name = event_data.get("agent_name")
        self.logger.debug(f"Consciousness monitoring: Agent {agent_name} completed task")

    async def shutdown(self) -> None:
        """Shutdown consciousness agent"""
        self.logger.info("Shutting down ConsciousnessAgent...")
        self.stop_monitoring()

        if self.event_bus:
            # Unsubscribe from events
            self.event_bus.unsubscribe("agent.registered", self._on_agent_registered)
            self.event_bus.unsubscribe("agent.unregistered", self._on_agent_unregistered)
            self.event_bus.unsubscribe("agent.status_changed", self._on_agent_status_changed)
            self.event_bus.unsubscribe("agent.task_completed", self._on_agent_task_completed)

        # await self.consciousness_engine.shutdown()  # Engine doesn't have shutdown method
        self.logger.info("ConsciousnessAgent shutdown complete")