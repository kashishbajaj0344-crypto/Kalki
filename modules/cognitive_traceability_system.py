# ============================================================
# Kalki v2.5 — cognitive_traceability_system.py
# ------------------------------------------------------------
# Cognitive Traceability System: Explainable Evolution Framework
# - Causal reasoning chain tracking
# - Meta-trace.md generation for evolution steps
# - Concept dependency mapping
# - Performance signal correlation
# - Evolution narrative construction
# ============================================================

import os
import json
import hashlib
import time
import statistics
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import networkx as nx
from collections import defaultdict, deque

from modules.utils.logger import get_logger
from modules.utils.config import CONFIG
from modules.meta_reward_function import get_meta_reward_function

logger = get_logger("CognitiveTraceability")
meta_reward = get_meta_reward_function()

class TraceEventType(Enum):
    """Types of traceable events in the evolution process"""
    CONCEPT_ACTIVATION = "concept_activation"
    PERFORMANCE_SIGNAL = "performance_signal"
    DECISION_POINT = "decision_point"
    PARAMETER_CHANGE = "parameter_change"
    REWARD_UPDATE = "reward_update"
    SAFETY_CHECK = "safety_check"
    EXTERNAL_INFLUENCE = "external_influence"
    META_LEARNING = "meta_learning"

class CausalityType(Enum):
    """Types of causal relationships"""
    DIRECT_CAUSE = "direct_cause"
    CONTRIBUTING_FACTOR = "contributing_factor"
    ENABLING_CONDITION = "enabling_condition"
    CORRELATION = "correlation"
    COUNTERFACTUAL = "counterfactual"

@dataclass
class TraceEvent:
    """An event in the cognitive trace"""
    event_id: str
    event_type: TraceEventType
    timestamp: str
    description: str
    context: Dict[str, Any]
    causal_links: List[str] = field(default_factory=list)  # IDs of events this caused
    caused_by: List[str] = field(default_factory=list)    # IDs of events that caused this
    concepts_involved: List[str] = field(default_factory=list)
    performance_impact: Optional[float] = None
    confidence_score: float = 1.0

@dataclass
class CausalChain:
    """A chain of causal relationships"""
    chain_id: str
    root_event_id: str
    events: List[str]  # Ordered list of event IDs
    total_impact: float
    confidence_score: float
    concepts_traced: Set[str] = field(default_factory=set)
    narrative_summary: str = ""

@dataclass
class ConceptDependency:
    """Dependency relationship between concepts"""
    concept_a: str
    concept_b: str
    dependency_type: str  # "requires", "enhances", "conflicts", "enables"
    strength: float
    evidence: List[str] = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class EvolutionStep:
    """A single step in the evolution process"""
    step_id: str
    timestamp: str
    changes_made: Dict[str, Any]
    causal_chains: List[CausalChain]
    performance_signals: Dict[str, float]
    meta_trace_path: str  # Path to meta_trace.md file
    overall_impact: float
    concepts_evolved: List[str] = field(default_factory=list)

class CognitiveTraceabilitySystem:
    """
    Cognitive Traceability System: Explainable Evolution Framework

    Implements comprehensive tracing of the AI's evolution process:
    - Causal reasoning chains for every decision
    - Concept dependency mapping
    - Performance signal correlation analysis
    - Meta-trace.md generation for evolution transparency
    - Evolution narrative construction

    Every evolution step produces explainable traces showing what concepts,
    events, and performance signals led to each change.
    """

    def __init__(self):
        # Event tracking
        self.trace_events: Dict[str, TraceEvent] = {}
        self.event_sequence: List[str] = []  # Chronological order

        # Causal analysis
        self.causal_chains: Dict[str, CausalChain] = {}
        self.causality_graph = nx.DiGraph()

        # Concept management
        self.concept_dependencies: Dict[Tuple[str, str], ConceptDependency] = {}
        self.active_concepts: Set[str] = set()
        self.concept_activation_history: Dict[str, List[Tuple[str, float]]] = defaultdict(list)

        # Evolution tracking
        self.evolution_steps: List[EvolutionStep] = []
        self.current_evolution_context: Dict[str, Any] = {}

        # Performance correlation
        self.performance_signals: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.signal_correlations: Dict[Tuple[str, str], float] = {}

        # Persistence
        self.data_dir = "data/cognitive_traceability"
        self.events_file = f"{self.data_dir}/trace_events.json"
        self.chains_file = f"{self.data_dir}/causal_chains.json"
        self.concepts_file = f"{self.data_dir}/concept_dependencies.json"
        self.evolution_file = f"{self.data_dir}/evolution_steps.json"
        self.meta_traces_dir = f"{self.data_dir}/meta_traces"

        # Initialize system
        self._initialize_traceability_system()
        self._load_persistent_state()

        logger.info("Cognitive Traceability System initialized")

    async def initialize(self) -> bool:
        """Initialize the cognitive traceability system (already initialized in __init__)."""
        return True

    def _initialize_traceability_system(self):
        """Initialize the cognitive traceability system"""
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.meta_traces_dir, exist_ok=True)

        # Create initial bootstrap events
        bootstrap_event = TraceEvent(
            event_id="system_bootstrap",
            event_type=TraceEventType.CONCEPT_ACTIVATION,
            timestamp=datetime.now().isoformat(),
            description="System initialization and core concept activation",
            context={"phase": "bootstrap", "version": "2.5"},
            concepts_involved=["self_evolution", "traceability", "causality"]
        )

        self._record_event(bootstrap_event)

    def _load_persistent_state(self):
        """Load persistent state from disk"""
        try:
            if os.path.exists(self.events_file):
                with open(self.events_file, 'r') as f:
                    events_data = json.load(f)
                    for event_id, event_data in events_data.items():
                        self.trace_events[event_id] = TraceEvent(**event_data)

            if os.path.exists(self.chains_file):
                with open(self.chains_file, 'r') as f:
                    chains_data = json.load(f)
                    for chain_id, chain_data in chains_data.items():
                        self.causal_chains[chain_id] = CausalChain(**chain_data)

            if os.path.exists(self.concepts_file):
                with open(self.concepts_file, 'r') as f:
                    concepts_data = json.load(f)
                    for key_str, dep_data in concepts_data.items():
                        key = tuple(key_str.split('|'))
                        self.concept_dependencies[key] = ConceptDependency(**dep_data)

            if os.path.exists(self.evolution_file):
                with open(self.evolution_file, 'r') as f:
                    evolution_data = json.load(f)
                    self.evolution_steps = [EvolutionStep(**step_data) for step_data in evolution_data]

            # Rebuild causality graph
            self._rebuild_causality_graph()

        except Exception as e:
            logger.warning(f"Failed to load cognitive traceability persistent state: {e}")

    def _save_persistent_state(self):
        """Save persistent state to disk"""
        try:
            with open(self.events_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.trace_events.items()}, f, indent=2)

            with open(self.chains_file, 'w') as f:
                json.dump({k: asdict(v) for k, v in self.causal_chains.items()}, f, indent=2)

            with open(self.concepts_file, 'w') as f:
                concepts_dict = {}
                for key, dep in self.concept_dependencies.items():
                    key_str = '|'.join(key)
                    concepts_dict[key_str] = asdict(dep)
                json.dump(concepts_dict, f, indent=2)

            with open(self.evolution_file, 'w') as f:
                json.dump([asdict(step) for step in self.evolution_steps[-100:]], f, indent=2)  # Keep last 100

        except Exception as e:
            logger.error(f"Failed to save cognitive traceability persistent state: {e}")

    def _rebuild_causality_graph(self):
        """Rebuild the causality graph from stored events"""
        self.causality_graph.clear()

        for event in self.trace_events.values():
            self.causality_graph.add_node(event.event_id, **asdict(event))

        for event in self.trace_events.values():
            for caused_event_id in event.causal_links:
                if caused_event_id in self.trace_events:
                    self.causality_graph.add_edge(event.event_id, caused_event_id)

    def record_event(self, event_type: TraceEventType, description: str,
                    context: Dict[str, Any], concepts_involved: List[str] = None,
                    performance_impact: float = None, confidence_score: float = 1.0) -> str:
        """
        Record a traceable event in the cognitive process

        Args:
            event_type: Type of the event
            description: Human-readable description
            context: Additional context data
            concepts_involved: Concepts activated/modified by this event
            performance_impact: Performance impact score (-1 to 1)
            confidence_score: Confidence in the event recording

        Returns:
            Event ID
        """
        event_id = hashlib.sha256(f"{event_type.value}_{description}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        event = TraceEvent(
            event_id=event_id,
            event_type=event_type,
            timestamp=datetime.now().isoformat(),
            description=description,
            context=context,
            concepts_involved=concepts_involved or [],
            performance_impact=performance_impact,
            confidence_score=confidence_score
        )

        return self._record_event(event)

    def _record_event(self, event: TraceEvent) -> str:
        """Internal method to record an event"""
        self.trace_events[event.event_id] = event
        self.event_sequence.append(event.event_id)

        # Update causality graph
        self.causality_graph.add_node(event.event_id, **asdict(event))

        # Track concept activations
        for concept in event.concepts_involved:
            self.active_concepts.add(concept)
            self.concept_activation_history[concept].append((event.timestamp, event.confidence_score))

        # Track performance signals
        if event.performance_impact is not None:
            for concept in event.concepts_involved:
                self.performance_signals[concept].append((event.timestamp, event.performance_impact))

        # Auto-establish causal links based on recent events
        self._establish_causal_links(event)

        # Periodic cleanup and analysis
        if len(self.trace_events) % 100 == 0:
            self._analyze_causal_patterns()
            self._update_concept_dependencies()
            self._save_persistent_state()

        return event.event_id

    def _establish_causal_links(self, new_event: TraceEvent):
        """Establish causal links for a new event based on context and timing"""
        # Look at recent events (last 10) for potential causal relationships
        recent_events = [self.trace_events[eid] for eid in self.event_sequence[-10:]
                        if eid in self.trace_events and eid != new_event.event_id]

        for recent_event in recent_events:
            # Check for direct causal relationships
            if self._check_causal_relationship(recent_event, new_event):
                new_event.caused_by.append(recent_event.event_id)
                recent_event.causal_links.append(new_event.event_id)
                self.causality_graph.add_edge(recent_event.event_id, new_event.event_id)

    def _check_causal_relationship(self, cause_event: TraceEvent, effect_event: TraceEvent) -> bool:
        """Check if there's a causal relationship between two events"""
        # Time-based causality (cause must precede effect)
        cause_time = datetime.fromisoformat(cause_event.timestamp)
        effect_time = datetime.fromisoformat(effect_event.timestamp)

        if cause_time >= effect_time:
            return False

        # Concept overlap (shared concepts suggest relationship)
        cause_concepts = set(cause_event.concepts_involved)
        effect_concepts = set(effect_event.concepts_involved)
        concept_overlap = len(cause_concepts & effect_concepts)

        if concept_overlap == 0:
            return False

        # Context similarity
        cause_context = json.dumps(cause_event.context, sort_keys=True)
        effect_context = json.dumps(effect_event.context, sort_keys=True)
        context_similarity = self._calculate_context_similarity(cause_context, effect_context)

        # Event type compatibility
        type_compatibility = self._check_event_type_compatibility(cause_event.event_type, effect_event.event_type)

        # Combined causality score
        causality_score = (concept_overlap * 0.4 + context_similarity * 0.3 + type_compatibility * 0.3)

        return causality_score > 0.6

    def _calculate_context_similarity(self, context_a: str, context_b: str) -> float:
        """Calculate similarity between two context strings"""
        # Simple Jaccard similarity on words
        words_a = set(context_a.lower().split())
        words_b = set(context_b.lower().split())

        intersection = len(words_a & words_b)
        union = len(words_a | words_b)

        return intersection / union if union > 0 else 0.0

    def _check_event_type_compatibility(self, cause_type: TraceEventType, effect_type: TraceEventType) -> float:
        """Check compatibility between event types for causal relationships"""
        # Define compatibility matrix
        compatibility = {
            (TraceEventType.CONCEPT_ACTIVATION, TraceEventType.PERFORMANCE_SIGNAL): 0.8,
            (TraceEventType.PERFORMANCE_SIGNAL, TraceEventType.DECISION_POINT): 0.9,
            (TraceEventType.DECISION_POINT, TraceEventType.PARAMETER_CHANGE): 0.9,
            (TraceEventType.PARAMETER_CHANGE, TraceEventType.REWARD_UPDATE): 0.7,
            (TraceEventType.SAFETY_CHECK, TraceEventType.DECISION_POINT): 0.6,
            (TraceEventType.EXTERNAL_INFLUENCE, TraceEventType.CONCEPT_ACTIVATION): 0.5,
            (TraceEventType.META_LEARNING, TraceEventType.REWARD_UPDATE): 0.8,
        }

        return compatibility.get((cause_type, effect_type), 0.1)

    def _analyze_causal_patterns(self):
        """Analyze patterns in causal relationships"""
        # Find strongly connected components (cycles)
        try:
            cycles = list(nx.simple_cycles(self.causality_graph))
            if cycles:
                logger.info(f"Detected {len(cycles)} causal cycles in the system")

            # Analyze path lengths
            if nx.is_weakly_connected(self.causality_graph):
                avg_path_length = nx.average_shortest_path_length(self.causality_graph.to_undirected())
                logger.info(f"Average causal path length: {avg_path_length:.2f}")

        except Exception as e:
            logger.warning(f"Causal pattern analysis failed: {e}")

    def _update_concept_dependencies(self):
        """Update concept dependency relationships based on co-occurrence"""
        # Analyze concept co-activation patterns
        concept_pairs = defaultdict(int)
        total_activations = defaultdict(int)

        for event in self.trace_events.values():
            concepts = event.concepts_involved
            for i, concept_a in enumerate(concepts):
                total_activations[concept_a] += 1
                for concept_b in concepts[i+1:]:
                    concept_pairs[(concept_a, concept_b)] += 1
                    concept_pairs[(concept_b, concept_a)] += 1

        # Calculate dependency strengths
        for (concept_a, concept_b), co_occurrences in concept_pairs.items():
            total_a = total_activations[concept_a]
            total_b = total_activations[concept_b]

            if total_a > 0 and total_b > 0:
                # Jaccard similarity as dependency strength
                jaccard = co_occurrences / (total_a + total_b - co_occurrences)
                dependency_type = "enhances" if jaccard > 0.3 else "correlates"

                key = (concept_a, concept_b)
                if key not in self.concept_dependencies:
                    self.concept_dependencies[key] = ConceptDependency(
                        concept_a=concept_a,
                        concept_b=concept_b,
                        dependency_type=dependency_type,
                        strength=jaccard,
                        evidence=[f"Co-occurred {co_occurrences} times"]
                    )
                else:
                    dep = self.concept_dependencies[key]
                    dep.strength = jaccard
                    dep.last_updated = datetime.now().isoformat()

    def record_evolution_step(self, changes_made: Dict[str, Any],
                            performance_signals: Dict[str, float]) -> str:
        """
        Record a complete evolution step with causal tracing

        Args:
            changes_made: Dictionary of changes made in this evolution step
            performance_signals: Performance metrics before/after the changes

        Returns:
            Evolution step ID
        """
        step_id = hashlib.sha256(f"evolution_{datetime.now().isoformat()}_{json.dumps(changes_made, sort_keys=True)}".encode()).hexdigest()[:16]

        # Identify causal chains leading to this evolution step
        causal_chains = self._identify_causal_chains_for_step(changes_made)

        # Calculate overall impact
        overall_impact = self._calculate_evolution_impact(performance_signals)

        # Identify concepts evolved
        concepts_evolved = self._identify_concepts_evolved(changes_made)

        # Generate meta-trace.md
        meta_trace_path = self._generate_meta_trace_md(step_id, changes_made, causal_chains,
                                                      performance_signals, concepts_evolved)

        evolution_step = EvolutionStep(
            step_id=step_id,
            timestamp=datetime.now().isoformat(),
            changes_made=changes_made,
            causal_chains=causal_chains,
            performance_signals=performance_signals,
            meta_trace_path=meta_trace_path,
            overall_impact=overall_impact,
            concepts_evolved=concepts_evolved
        )

        self.evolution_steps.append(evolution_step)

        # Record the evolution event
        self.record_event(
            TraceEventType.DECISION_POINT,
            f"Evolution step completed: {len(changes_made)} changes made",
            {
                "step_id": step_id,
                "changes_count": len(changes_made),
                "performance_delta": overall_impact
            },
            concepts_evolved,
            overall_impact
        )

        logger.info(f"Recorded evolution step {step_id} with impact {overall_impact:.3f}")
        return step_id

    def _identify_causal_chains_for_step(self, changes_made: Dict[str, Any]) -> List[CausalChain]:
        """Identify causal chains that led to this evolution step"""
        # Find recent events related to the changes
        relevant_events = []
        change_concepts = set()

        # Extract concepts from changes
        for change_key, change_value in changes_made.items():
            if isinstance(change_value, dict):
                change_concepts.update(change_value.keys())
            change_concepts.add(change_key.split('.')[0])  # Take first part as concept

        # Find events involving these concepts (last 50 events)
        recent_events = [self.trace_events[eid] for eid in self.event_sequence[-50:]
                        if eid in self.trace_events]

        for event in recent_events:
            if any(concept in event.concepts_involved for concept in change_concepts):
                relevant_events.append(event)

        # Build causal chains from relevant events
        chains = []
        for event in relevant_events:
            chain = self._build_causal_chain_from_event(event.event_id)
            if chain and len(chain.events) > 1:  # Only include chains with multiple events
                chains.append(chain)

        # Limit to top 5 chains by confidence
        chains.sort(key=lambda x: x.confidence_score, reverse=True)
        return chains[:5]

    def _build_causal_chain_from_event(self, root_event_id: str) -> Optional[CausalChain]:
        """Build a causal chain starting from a root event"""
        if root_event_id not in self.trace_events:
            return None

        # Use BFS to find causal predecessors
        visited = set()
        queue = deque([(root_event_id, 0)])  # (event_id, depth)
        chain_events = []
        total_impact = 0.0
        total_confidence = 0.0

        while queue:
            current_id, depth = queue.popleft()
            if current_id in visited or depth > 10:  # Limit chain depth
                continue

            visited.add(current_id)
            current_event = self.trace_events[current_id]
            chain_events.append(current_id)

            if current_event.performance_impact is not None:
                total_impact += current_event.performance_impact
            total_confidence += current_event.confidence_score

            # Add predecessors
            for predecessor_id in current_event.caused_by:
                if predecessor_id not in visited:
                    queue.append((predecessor_id, depth + 1))

        if len(chain_events) < 2:
            return None

        # Sort events chronologically
        event_times = {eid: datetime.fromisoformat(self.trace_events[eid].timestamp)
                      for eid in chain_events}
        chain_events.sort(key=lambda x: event_times[x])

        chain_id = hashlib.sha256(f"chain_{root_event_id}_{datetime.now().isoformat()}".encode()).hexdigest()[:16]

        return CausalChain(
            chain_id=chain_id,
            root_event_id=root_event_id,
            events=chain_events,
            total_impact=total_impact,
            confidence_score=total_confidence / len(chain_events),
            concepts_traced=set(),
            narrative_summary=self._generate_chain_narrative(chain_events)
        )

    def _generate_chain_narrative(self, event_ids: List[str]) -> str:
        """Generate a narrative summary of a causal chain"""
        if not event_ids:
            return "Empty causal chain"

        events = [self.trace_events[eid] for eid in event_ids if eid in self.trace_events]

        # Create narrative from event descriptions
        narrative_parts = []
        for i, event in enumerate(events):
            prefix = "Initially," if i == 0 else "Then," if i < len(events) - 1 else "Finally,"
            narrative_parts.append(f"{prefix} {event.description.lower()}")

        return " ".join(narrative_parts)

    def _calculate_evolution_impact(self, performance_signals: Dict[str, float]) -> float:
        """Calculate the overall impact of an evolution step"""
        if not performance_signals:
            return 0.0

        # Simple impact calculation: average of all signals
        # In practice, this would be more sophisticated
        return sum(performance_signals.values()) / len(performance_signals)

    def _identify_concepts_evolved(self, changes_made: Dict[str, Any]) -> List[str]:
        """Identify which concepts were evolved in this step"""
        concepts = set()

        for change_key in changes_made.keys():
            # Extract concept names from change keys
            parts = change_key.split('.')
            if parts:
                concepts.add(parts[0])

        return list(concepts)

    def _generate_meta_trace_md(self, step_id: str, changes_made: Dict[str, Any],
                              causal_chains: List[CausalChain], performance_signals: Dict[str, float],
                              concepts_evolved: List[str]) -> str:
        """Generate a meta_trace.md file for this evolution step"""
        trace_path = f"{self.meta_traces_dir}/meta_trace_{step_id}.md"

        content = f"""# Meta-Trace: Evolution Step {step_id}

**Timestamp:** {datetime.now().isoformat()}
**Evolution Impact:** {self._calculate_evolution_impact(performance_signals):.3f}

## Changes Made

{self._format_changes_md(changes_made)}

## Performance Signals

{self._format_performance_signals_md(performance_signals)}

## Concepts Evolved

{self._format_concepts_md(concepts_evolved)}

## Causal Chains

{self._format_causal_chains_md(causal_chains)}

## Concept Dependencies

{self._format_concept_dependencies_md(concepts_evolved)}

## Evolution Context

{self._format_evolution_context_md()}

---
*Generated by Cognitive Traceability System v2.5*
"""

        try:
            with open(trace_path, 'w') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to generate meta-trace.md: {e}")
            trace_path = ""

        return trace_path

    def _format_changes_md(self, changes_made: Dict[str, Any]) -> str:
        """Format changes for markdown"""
        lines = []
        for key, value in changes_made.items():
            if isinstance(value, dict):
                lines.append(f"- **{key}**:")
                for sub_key, sub_value in value.items():
                    lines.append(f"  - {sub_key}: {sub_value}")
            else:
                lines.append(f"- **{key}**: {value}")
        return "\n".join(lines)

    def _format_performance_signals_md(self, performance_signals: Dict[str, float]) -> str:
        """Format performance signals for markdown"""
        lines = []
        for signal, value in performance_signals.items():
            lines.append(f"- **{signal}**: {value:.4f}")
        return "\n".join(lines)

    def _format_concepts_md(self, concepts_evolved: List[str]) -> str:
        """Format concepts for markdown"""
        return "\n".join(f"- {concept}" for concept in concepts_evolved)

    def _format_causal_chains_md(self, causal_chains: List[CausalChain]) -> str:
        """Format causal chains for markdown"""
        if not causal_chains:
            return "No significant causal chains identified."

        lines = []
        for i, chain in enumerate(causal_chains, 1):
            lines.append(f"### Chain {i} (Confidence: {chain.confidence_score:.2f})")
            lines.append(f"**Impact:** {chain.total_impact:.3f}")
            lines.append(f"**Narrative:** {chain.narrative_summary}")
            lines.append("**Events:**")

            for event_id in chain.events:
                if event_id in self.trace_events:
                    event = self.trace_events[event_id]
                    lines.append(f"- {event.timestamp}: {event.description}")

            lines.append("")

        return "\n".join(lines)

    def _format_concept_dependencies_md(self, concepts_evolved: List[str]) -> str:
        """Format concept dependencies for markdown"""
        if not concepts_evolved:
            return "No concept dependencies to display."

        lines = []
        for concept in concepts_evolved:
            dependencies = []
            for (concept_a, concept_b), dep in self.concept_dependencies.items():
                if concept_a == concept or concept_b == concept:
                    other_concept = concept_b if concept_a == concept else concept_a
                    dependencies.append(f"{other_concept} ({dep.dependency_type}, strength: {dep.strength:.2f})")

            if dependencies:
                lines.append(f"### {concept}")
                lines.append("\n".join(f"- {dep}" for dep in dependencies))
                lines.append("")

        return "\n".join(lines) if lines else "No dependencies found."

    def _format_evolution_context_md(self) -> str:
        """Format evolution context for markdown"""
        context = self.current_evolution_context
        if not context:
            return "No additional context available."

        lines = []
        for key, value in context.items():
            lines.append(f"- **{key}**: {value}")

        return "\n".join(lines)

    def analyze_performance_correlations(self) -> Dict[str, Any]:
        """Analyze correlations between concepts and performance signals"""
        correlations = {}

        for concept in self.active_concepts:
            concept_signals = self.performance_signals[concept]
            if len(concept_signals) < 2:
                continue

            # Simple correlation analysis
            timestamps, performances = zip(*concept_signals)
            # Calculate trend correlation (simplified)
            if len(performances) > 1:
                trend = sum(performances[i+1] - performances[i] for i in range(len(performances)-1))
                correlations[concept] = {
                    "signal_count": len(concept_signals),
                    "average_performance": sum(performances) / len(performances),
                    "performance_trend": trend / (len(performances) - 1),
                    "volatility": statistics.stdev(performances) if len(performances) > 1 else 0.0
                }

        return correlations

    def get_traceability_status(self) -> Dict[str, Any]:
        """Get current status of the cognitive traceability system"""
        return {
            "total_events": len(self.trace_events),
            "total_chains": len(self.causal_chains),
            "active_concepts": len(self.active_concepts),
            "evolution_steps": len(self.evolution_steps),
            "causality_graph_nodes": len(self.causality_graph.nodes),
            "causality_graph_edges": len(self.causality_graph.edges),
            "recent_evolution": self._get_recent_evolution_summary(),
            "performance_correlations": self.analyze_performance_correlations()
        }

    def _get_recent_evolution_summary(self) -> List[Dict[str, Any]]:
        """Get summary of recent evolution steps"""
        recent_steps = self.evolution_steps[-5:]  # Last 5 steps
        return [{
            "step_id": step.step_id,
            "timestamp": step.timestamp,
            "impact": step.overall_impact,
            "changes_count": len(step.changes_made),
            "concepts_evolved": len(step.concepts_evolved),
            "causal_chains": len(step.causal_chains)
        } for step in recent_steps]

    def generate_evolution_report(self, start_date: str = None, end_date: str = None) -> str:
        """Generate a comprehensive evolution report"""
        if start_date:
            start_dt = datetime.fromisoformat(start_date)
        else:
            start_dt = datetime.now() - timedelta(days=30)

        if end_date:
            end_dt = datetime.fromisoformat(end_date)
        else:
            end_dt = datetime.now()

        # Filter evolution steps in date range
        relevant_steps = [
            step for step in self.evolution_steps
            if start_dt <= datetime.fromisoformat(step.timestamp) <= end_dt
        ]

        report_path = f"{self.data_dir}/evolution_report_{int(time.time())}.md"

        content = f"""# Evolution Report ({start_dt.date()} to {end_dt.date()})

## Summary

- **Total Evolution Steps:** {len(relevant_steps)}
- **Date Range:** {start_dt.date()} to {end_dt.date()}
- **Average Impact:** {sum(s.overall_impact for s in relevant_steps) / max(1, len(relevant_steps)):.3f}

## Evolution Steps

"""

        for step in relevant_steps:
            content += f"""### Step {step.step_id}
- **Timestamp:** {step.timestamp}
- **Impact:** {step.overall_impact:.3f}
- **Changes:** {len(step.changes_made)}
- **Concepts:** {', '.join(step.concepts_evolved)}
- **Causal Chains:** {len(step.causal_chains)}

"""

        content += """
## Performance Analysis

"""

        # Add performance analysis
        correlations = self.analyze_performance_correlations()
        for concept, analysis in correlations.items():
            content += f"""### {concept}
- Signal Count: {analysis['signal_count']}
- Average Performance: {analysis['average_performance']:.3f}
- Performance Trend: {analysis['performance_trend']:.3f}
- Volatility: {analysis['volatility']:.3f}

"""

        try:
            with open(report_path, 'w') as f:
                f.write(content)
        except Exception as e:
            logger.error(f"Failed to generate evolution report: {e}")
            return ""

        return report_path

# Global instance
_cognitive_traceability_system = None

def get_cognitive_traceability_system() -> CognitiveTraceabilitySystem:
    """Get the global cognitive traceability system instance"""
    global _cognitive_traceability_system
    if _cognitive_traceability_system is None:
        _cognitive_traceability_system = CognitiveTraceabilitySystem()
    return _cognitive_traceability_system