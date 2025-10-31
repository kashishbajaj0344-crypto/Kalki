"""
Temporal Paradox Engine (Phase 14)
----------------------------------
Implements counterfactual reasoning and multi-future scenario simulation.
Resolves temporal paradoxes and analyzes cascading effects.
"""

import asyncio
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from modules.logging_config import get_logger

from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.TemporalParadox")


@dataclass
class TimelineEvent:
    """Represents an event in a timeline"""
    event_id: str
    timestamp: datetime
    description: str
    event_type: str
    probability: float = 1.0
    causal_links: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalChain:
    """Represents a chain of causally linked events"""
    chain_id: str
    events: List[TimelineEvent]
    paradox_score: float = 0.0
    stability_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CounterfactualScenario:
    """Represents a counterfactual 'what-if' scenario"""
    scenario_id: str
    premise: str
    changed_events: List[TimelineEvent]
    resulting_timeline: List[TimelineEvent]
    paradox_level: float
    probability: float
    impact_score: float


class TemporalParadoxEngine(BaseAgent):
    """
    Temporal paradox engine for counterfactual reasoning and multi-future simulation.
    Analyzes temporal paradoxes, causal chains, and alternative timeline scenarios.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="TemporalParadoxEngine",
            capabilities=[
                AgentCapability.TEMPORAL_ANALYSIS,
                AgentCapability.SIMULATION,
                AgentCapability.EXPERIMENTATION
            ],
            description="Counterfactual reasoning and temporal paradox resolution",
            config=config or {}
        )

        # Temporal modeling parameters
        self.max_timeline_depth = self.config.get('max_depth', 100)
        self.paradox_threshold = self.config.get('paradox_threshold', 0.7)
        self.causal_decay_factor = self.config.get('causal_decay', 0.9)

        # Initialize causal graph
        self.causal_graph = nx.DiGraph()

        # Sample historical timeline for demonstration
        self.historical_timeline = self._initialize_historical_timeline()

    async def initialize(self) -> bool:
        """Initialize temporal paradox analysis environment"""
        try:
            # Test causal analysis
            test_events = self.historical_timeline[:5]
            chains = self._analyze_causal_chains(test_events)
            logger.info(f"TemporalParadoxEngine initialized with causal analysis capability")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize TemporalParadoxEngine: {e}")
            return False

    def _initialize_historical_timeline(self) -> List[TimelineEvent]:
        """Initialize sample historical timeline for demonstration"""
        base_date = datetime(2000, 1, 1)

        events = [
            TimelineEvent(
                event_id="ai_research_start",
                timestamp=base_date + timedelta(days=365*5),
                description="Initial AI research programs established",
                event_type="research",
                probability=1.0,
                causal_links=[],
                metadata={"field": "AI", "impact": "foundation"}
            ),
            TimelineEvent(
                event_id="deep_learning_breakthrough",
                timestamp=base_date + timedelta(days=365*12),
                description="Deep learning algorithms achieve breakthrough performance",
                event_type="technological",
                probability=0.9,
                causal_links=["ai_research_start"],
                metadata={"field": "AI", "impact": "major"}
            ),
            TimelineEvent(
                event_id="gpu_acceleration",
                timestamp=base_date + timedelta(days=365*15),
                description="GPU acceleration enables large-scale neural network training",
                event_type="technological",
                probability=0.95,
                causal_links=["deep_learning_breakthrough"],
                metadata={"field": "hardware", "impact": "enabling"}
            ),
            TimelineEvent(
                event_id="ai_winter_delay",
                timestamp=base_date + timedelta(days=365*18),
                description="AI winter delays progress by 3 years",
                event_type="setback",
                probability=0.6,
                causal_links=["gpu_acceleration"],
                metadata={"field": "AI", "impact": "delaying"}
            ),
            TimelineEvent(
                event_id="transformer_architecture",
                timestamp=base_date + timedelta(days=365*20),
                description="Transformer architecture revolutionizes NLP",
                event_type="technological",
                probability=0.85,
                causal_links=["gpu_acceleration", "ai_winter_delay"],
                metadata={"field": "AI", "impact": "revolutionary"}
            ),
            TimelineEvent(
                event_id="gpt_release",
                timestamp=base_date + timedelta(days=365*22),
                description="Large language models become publicly available",
                event_type="deployment",
                probability=0.9,
                causal_links=["transformer_architecture"],
                metadata={"field": "AI", "impact": "transformative"}
            )
        ]

        return events

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute temporal paradox analysis tasks"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "analyze_paradox":
            return await self._analyze_temporal_paradox(params)
        elif action == "counterfactual":
            return await self._generate_counterfactual(params)
        elif action == "causal_chains":
            return await self._analyze_causal_chains_async(params)
        elif action == "future_scenarios":
            return await self._simulate_future_scenarios(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _analyze_temporal_paradox(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal paradoxes in event sequences"""
        try:
            events = params.get("events", self.historical_timeline)
            paradox_types = params.get("paradox_types", ["causality", "bootstrap", "predestination"])

            # Analyze for paradoxes
            paradoxes = self._detect_temporal_paradoxes(events, paradox_types)

            return {
                "status": "success",
                "paradoxes_detected": paradoxes,
                "total_events": len(events),
                "paradox_types_analyzed": paradox_types
            }
        except Exception as e:
            logger.exception(f"Temporal paradox analysis error: {e}")
            return {"status": "error", "error": str(e)}

    def _detect_temporal_paradoxes(self, events: List[TimelineEvent],
                                 paradox_types: List[str]) -> List[Dict[str, Any]]:
        """Detect various types of temporal paradoxes"""
        paradoxes = []

        # Build causal graph
        self._build_causal_graph(events)

        for paradox_type in paradox_types:
            if paradox_type == "causality":
                causality_paradoxes = self._detect_causality_paradoxes(events)
                paradoxes.extend(causality_paradoxes)
            elif paradox_type == "bootstrap":
                bootstrap_paradoxes = self._detect_bootstrap_paradoxes(events)
                paradoxes.extend(bootstrap_paradoxes)
            elif paradox_type == "predestination":
                predestination_paradoxes = self._detect_predestination_paradoxes(events)
                paradoxes.extend(predestination_paradoxes)

        return paradoxes

    def _build_causal_graph(self, events: List[TimelineEvent]) -> None:
        """Build directed graph of causal relationships"""
        self.causal_graph.clear()

        # Add nodes
        for event in events:
            self.causal_graph.add_node(event.event_id, event=event)

        # Add edges
        for event in events:
            for link_id in event.causal_links:
                if self.causal_graph.has_node(link_id):
                    self.causal_graph.add_edge(link_id, event.event_id)

    def _detect_causality_paradoxes(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Detect causality violation paradoxes"""
        paradoxes = []

        # Check for cycles in causal graph (causality loops)
        try:
            cycles = list(nx.simple_cycles(self.causal_graph))
            for cycle in cycles:
                paradoxes.append({
                    "type": "causality_violation",
                    "description": f"Causal cycle detected: {' -> '.join(cycle)}",
                    "severity": "high",
                    "involved_events": cycle,
                    "resolution_suggestion": "Break causal link or introduce time delay"
                })
        except Exception as e:
            logger.warning(f"Cycle detection failed: {e}")

        # Check for reverse causality (effect before cause)
        event_dict = {event.event_id: event for event in events}
        for event in events:
            for link_id in event.causal_links:
                if link_id in event_dict:
                    cause_event = event_dict[link_id]
                    if cause_event.timestamp > event.timestamp:
                        paradoxes.append({
                            "type": "reverse_causality",
                            "description": f"Effect '{event.event_id}' occurs before cause '{link_id}'",
                            "severity": "critical",
                            "involved_events": [event.event_id, link_id],
                            "time_difference_days": (cause_event.timestamp - event.timestamp).days
                        })

        return paradoxes

    def _detect_bootstrap_paradoxes(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Detect bootstrap paradoxes (self-creating information)"""
        paradoxes = []

        # Look for events that create their own causal prerequisites
        for event in events:
            # Check if event creates information that enables its own causes
            if self._is_self_referential(event, events):
                paradoxes.append({
                    "type": "bootstrap_paradox",
                    "description": f"Event '{event.event_id}' creates information needed for its own existence",
                    "severity": "high",
                    "involved_events": [event.event_id],
                    "resolution_suggestion": "Information must come from external source"
                })

        return paradoxes

    def _is_self_referential(self, event: TimelineEvent, all_events: List[TimelineEvent]) -> bool:
        """Check if an event creates information needed for its own causal chain"""
        # Simplified check: if event is linked to events that depend on similar information
        event_info = event.description.lower() + str(event.metadata)

        for cause_id in event.causal_links:
            cause_event = next((e for e in all_events if e.event_id == cause_id), None)
            if cause_event:
                cause_info = cause_event.description.lower() + str(cause_event.metadata)
                # Check for information overlap (simplified)
                if len(set(event_info.split()) & set(cause_info.split())) > 3:
                    return True

        return False

    def _detect_predestination_paradoxes(self, events: List[TimelineEvent]) -> List[Dict[str, Any]]:
        """Detect predestination paradoxes"""
        paradoxes = []

        # Look for events that attempt to change the past but are predetermined
        for event in events:
            if event.event_type == "intervention" and self._is_predetermined(event, events):
                paradoxes.append({
                    "type": "predestination_paradox",
                    "description": f"Intervention '{event.event_id}' attempts to change predetermined events",
                    "severity": "medium",
                    "involved_events": [event.event_id],
                    "resolution_suggestion": "Accept predetermination or change intervention approach"
                })

        return paradoxes

    def _is_predetermined(self, event: TimelineEvent, all_events: List[TimelineEvent]) -> bool:
        """Check if an event's outcome was predetermined"""
        # Simplified: check if multiple causal chains lead to same outcome
        incoming_links = [e for e in all_events if event.event_id in e.causal_links]
        return len(incoming_links) > 2  # Multiple causes suggest predetermination

    async def _generate_counterfactual(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate counterfactual scenarios"""
        try:
            premise = params.get("premise", "")
            changed_events = params.get("changed_events", [])
            timeline_depth = params.get("timeline_depth", 10)

            # Generate counterfactual scenario
            counterfactual = self._generate_counterfactual_scenario(
                premise, changed_events, timeline_depth
            )

            return {
                "status": "success",
                "counterfactual": {
                    "scenario_id": counterfactual.scenario_id,
                    "premise": counterfactual.premise,
                    "changed_events": [event.event_id for event in counterfactual.changed_events],
                    "resulting_events": len(counterfactual.resulting_timeline),
                    "paradox_level": counterfactual.paradox_level,
                    "probability": counterfactual.probability,
                    "impact_score": counterfactual.impact_score
                }
            }
        except Exception as e:
            logger.exception(f"Counterfactual generation error: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_counterfactual_scenario(self, premise: str,
                                        changed_events: List[Dict],
                                        timeline_depth: int) -> CounterfactualScenario:
        """Generate a counterfactual scenario with changed events"""
        scenario_id = f"cf_{np.random.randint(1000, 9999)}"

        # Convert changed events to TimelineEvent objects
        counterfactual_events = []
        for event_dict in changed_events:
            event = TimelineEvent(
                event_id=event_dict.get("event_id", f"modified_{len(counterfactual_events)}"),
                timestamp=datetime.now() + timedelta(days=event_dict.get("delay_days", 0)),
                description=event_dict.get("new_description", "Modified event"),
                event_type=event_dict.get("new_type", "intervention"),
                probability=event_dict.get("new_probability", 0.5),
                causal_links=event_dict.get("new_links", []),
                metadata=event_dict.get("metadata", {})
            )
            counterfactual_events.append(event)

        # Simulate resulting timeline
        resulting_timeline = self._simulate_counterfactual_timeline(
            counterfactual_events, timeline_depth
        )

        # Calculate paradox level
        paradox_level = self._calculate_paradox_level(counterfactual_events, resulting_timeline)

        # Estimate probability and impact
        probability = np.prod([event.probability for event in counterfactual_events])
        impact_score = self._calculate_impact_score(resulting_timeline)

        return CounterfactualScenario(
            scenario_id=scenario_id,
            premise=premise,
            changed_events=counterfactual_events,
            resulting_timeline=resulting_timeline,
            paradox_level=paradox_level,
            probability=probability,
            impact_score=impact_score
        )

    def _simulate_counterfactual_timeline(self, changed_events: List[TimelineEvent],
                                        depth: int) -> List[TimelineEvent]:
        """Simulate how timeline evolves after counterfactual changes"""
        timeline = changed_events.copy()
        current_time = max(event.timestamp for event in changed_events)

        # Generate cascading effects
        for i in range(depth):
            new_events = self._generate_cascading_events(timeline, current_time)
            if not new_events:
                break

            timeline.extend(new_events)
            current_time = max(event.timestamp for event in new_events)

        return timeline

    def _generate_cascading_events(self, current_timeline: List[TimelineEvent],
                                 current_time: datetime) -> List[TimelineEvent]:
        """Generate events that cascade from current timeline state"""
        new_events = []

        # Analyze current state and generate plausible follow-on events
        recent_events = [e for e in current_timeline
                        if (current_time - e.timestamp).days < 365]

        # Generate technological breakthrough if research events exist
        research_events = [e for e in recent_events if e.event_type == "research"]
        if research_events and np.random.random() < 0.3:
            breakthrough = TimelineEvent(
                event_id=f"breakthrough_{len(current_timeline)}",
                timestamp=current_time + timedelta(days=np.random.randint(180, 365)),
                description="Technological breakthrough from ongoing research",
                event_type="technological",
                probability=0.7,
                causal_links=[e.event_id for e in research_events[-2:]],
                metadata={"field": "technology", "impact": "major"}
            )
            new_events.append(breakthrough)

        # Generate market response to technological events
        tech_events = [e for e in recent_events if e.event_type == "technological"]
        if tech_events and np.random.random() < 0.4:
            market_response = TimelineEvent(
                event_id=f"market_{len(current_timeline)}",
                timestamp=current_time + timedelta(days=np.random.randint(90, 180)),
                description="Market response to technological development",
                event_type="economic",
                probability=0.8,
                causal_links=[e.event_id for e in tech_events[-1:]],
                metadata={"field": "economics", "impact": "moderate"}
            )
            new_events.append(market_response)

        return new_events

    def _calculate_paradox_level(self, changed_events: List[TimelineEvent],
                               resulting_timeline: List[TimelineEvent]) -> float:
        """Calculate paradox level of counterfactual scenario"""
        paradox_score = 0.0

        # Check for causality violations
        for event in resulting_timeline:
            for cause_id in event.causal_links:
                cause_event = next((e for e in resulting_timeline if e.event_id == cause_id), None)
                if cause_event and cause_event.timestamp > event.timestamp:
                    paradox_score += 0.5  # Reverse causality

        # Check for information loops
        for changed_event in changed_events:
            if self._creates_information_loop(changed_event, resulting_timeline):
                paradox_score += 0.3

        return min(paradox_score, 1.0)  # Cap at 1.0

    def _creates_information_loop(self, changed_event: TimelineEvent,
                                timeline: List[TimelineEvent]) -> bool:
        """Check if changed event creates information causality loop"""
        # Simplified check: if changed event is prerequisite for its own effects
        event_chain = self._get_causal_chain(changed_event.event_id, timeline)
        return changed_event.event_id in event_chain

    def _get_causal_chain(self, start_event_id: str, timeline: List[TimelineEvent]) -> Set[str]:
        """Get all events in causal chain starting from given event"""
        chain = set()
        to_visit = [start_event_id]

        while to_visit:
            current_id = to_visit.pop()
            if current_id not in chain:
                chain.add(current_id)
                # Find events that have this as a cause
                for event in timeline:
                    if current_id in event.causal_links:
                        to_visit.append(event.event_id)

        return chain

    def _calculate_impact_score(self, timeline: List[TimelineEvent]) -> float:
        """Calculate overall impact score of timeline"""
        impact_sum = 0.0
        for event in timeline:
            # Impact based on event type and probability
            base_impact = {
                "technological": 1.0,
                "economic": 0.8,
                "social": 0.7,
                "research": 0.6,
                "deployment": 0.9,
                "setback": -0.5
            }.get(event.event_type, 0.5)

            impact_sum += base_impact * event.probability

        return impact_sum / len(timeline) if timeline else 0.0

    async def _analyze_causal_chains_async(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causal chains in event sequences"""
        try:
            events = params.get("events", self.historical_timeline)
            max_chain_length = params.get("max_chain_length", 10)

            # Analyze causal chains
            chains = self._analyze_causal_chains(events)

            # Filter and rank chains
            significant_chains = [chain for chain in chains
                                if len(chain.events) <= max_chain_length
                                and chain.stability_score > 0.5]

            significant_chains.sort(key=lambda x: x.stability_score, reverse=True)

            return {
                "status": "success",
                "causal_chains": [
                    {
                        "chain_id": chain.chain_id,
                        "length": len(chain.events),
                        "events": [event.event_id for event in chain.events],
                        "paradox_score": chain.paradox_score,
                        "stability_score": chain.stability_score
                    }
                    for chain in significant_chains[:10]  # Top 10
                ],
                "total_chains_analyzed": len(chains)
            }
        except Exception as e:
            logger.exception(f"Causal chain analysis error: {e}")
            return {"status": "error", "error": str(e)}

    def _analyze_causal_chains(self, events: List[TimelineEvent]) -> List[CausalChain]:
        """Analyze causal chains in timeline"""
        chains = []

        # Build causal graph
        self._build_causal_graph(events)
        event_dict = {event.event_id: event for event in events}

        # Find all causal paths
        for start_node in self.causal_graph.nodes():
            for end_node in self.causal_graph.nodes():
                if start_node != end_node:
                    try:
                        paths = list(nx.all_simple_paths(self.causal_graph, start_node, end_node, cutoff=5))
                        for path in paths:
                            if len(path) > 2:  # At least 3 events in chain
                                chain_events = [event_dict[node_id] for node_id in path if node_id in event_dict]
                                if len(chain_events) == len(path):
                                    chain = CausalChain(
                                        chain_id=f"chain_{len(chains)}",
                                        events=chain_events,
                                        paradox_score=self._calculate_chain_paradox_score(chain_events),
                                        stability_score=self._calculate_chain_stability(chain_events)
                                    )
                                    chains.append(chain)
                    except Exception:
                        continue  # Skip problematic paths

        return chains

    def _calculate_chain_paradox_score(self, events: List[TimelineEvent]) -> float:
        """Calculate paradox score for a causal chain"""
        paradox_score = 0.0

        for i, event in enumerate(events):
            # Check for reverse causality within chain
            for j in range(i + 1, len(events)):
                if events[j].timestamp < event.timestamp and events[j].event_id in event.causal_links:
                    paradox_score += 0.2

        return min(paradox_score, 1.0)

    def _calculate_chain_stability(self, events: List[TimelineEvent]) -> float:
        """Calculate stability score for a causal chain"""
        if len(events) < 2:
            return 1.0

        # Stability based on probability consistency and temporal ordering
        stability = 1.0

        for i in range(len(events) - 1):
            current_event = events[i]
            next_event = events[i + 1]

            # Temporal ordering should be maintained
            if next_event.timestamp < current_event.timestamp:
                stability *= 0.8

            # Probability should be reasonable
            if next_event.probability < 0.1:
                stability *= 0.9

        return stability

    async def _simulate_future_scenarios(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate multiple future scenarios"""
        try:
            base_timeline = params.get("base_timeline", self.historical_timeline)
            num_scenarios = params.get("num_scenarios", 5)
            simulation_depth = params.get("simulation_depth", 20)

            # Generate multiple future scenarios
            scenarios = self._generate_future_scenarios(base_timeline, num_scenarios, simulation_depth)

            return {
                "status": "success",
                "scenarios": [
                    {
                        "scenario_id": scenario.scenario_id,
                        "description": scenario.premise,
                        "events_count": len(scenario.resulting_timeline),
                        "paradox_level": scenario.paradox_level,
                        "probability": scenario.probability,
                        "impact_score": scenario.impact_score
                    }
                    for scenario in scenarios
                ],
                "simulation_parameters": {
                    "base_events": len(base_timeline),
                    "num_scenarios": num_scenarios,
                    "simulation_depth": simulation_depth
                }
            }
        except Exception as e:
            logger.exception(f"Future scenario simulation error: {e}")
            return {"status": "error", "error": str(e)}

    def _generate_future_scenarios(self, base_timeline: List[TimelineEvent],
                                 num_scenarios: int, depth: int) -> List[CounterfactualScenario]:
        """Generate multiple future scenarios from base timeline"""
        scenarios = []
        current_time = max(event.timestamp for event in base_timeline)

        for i in range(num_scenarios):
            # Generate random intervention point
            intervention_event = self._generate_random_intervention(base_timeline, current_time)

            scenario = self._generate_counterfactual_scenario(
                f"Scenario {i+1}: {intervention_event.description}",
                [intervention_event.__dict__],
                depth
            )

            scenarios.append(scenario)

        return scenarios

    def _generate_random_intervention(self, base_timeline: List[TimelineEvent],
                                    current_time: datetime) -> TimelineEvent:
        """Generate a random intervention event"""
        intervention_types = [
            ("accelerate_research", "Accelerated research funding leads to breakthrough"),
            ("regulatory_change", "New regulations change technology landscape"),
            ("market_disruption", "Market disruption creates new opportunities"),
            ("technological_accident", "Unexpected technological incident"),
            ("collaboration", "Major collaboration between organizations")
        ]

        intervention_type, description = random.choice(intervention_types)

        return TimelineEvent(
            event_id=f"intervention_{np.random.randint(1000, 9999)}",
            timestamp=current_time + timedelta(days=np.random.randint(30, 365)),
            description=description,
            event_type="intervention",
            probability=np.random.uniform(0.3, 0.9),
            causal_links=[random.choice(base_timeline).event_id],
            metadata={"intervention_type": intervention_type, "generated": True}
        )

    async def shutdown(self) -> bool:
        """Clean up temporal paradox analysis resources"""
        logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True