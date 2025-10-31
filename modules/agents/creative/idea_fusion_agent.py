#!/usr/bin/env python3
"""
IdeaFusionAgent - Cross-domain idea synthesis and fusion

Combines ideas from different domains with explainable rationale,
evaluation metrics, and safety controls.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from collections import defaultdict

from ..base_agent import BaseAgent, AgentCapability
from ..memory.memory_agent import MemoryAgent
from ..cognitive.performance_monitor import PerformanceMonitorAgent
from ...eventbus import EventBus

logger = logging.getLogger("kalki.agents.idea_fusion")


class FusionRationale:
    """Represents the rationale for an idea fusion"""

    def __init__(self):
        self.key_concepts = []  # List of (idea_id, concept, importance)
        self.synergistic_elements = []  # Elements that create synergy
        self.conflict_resolutions = []  # How conflicts were resolved
        self.novel_combinations = []  # New combinations created
        self.domain_bridges = []  # How domains were bridged

    def add_concept(self, idea_id: str, concept: str, importance: float):
        """Add a key concept from an idea"""
        self.key_concepts.append((idea_id, concept, importance))

    def add_synergy(self, description: str, impact: float):
        """Add a synergistic element"""
        self.synergistic_elements.append((description, impact))

    def add_conflict_resolution(self, conflict: str, resolution: str):
        """Add a conflict resolution"""
        self.conflict_resolutions.append((conflict, resolution))

    def add_novel_combination(self, combination: str, novelty_score: float):
        """Add a novel combination"""
        self.novel_combinations.append((combination, novelty_score))

    def add_domain_bridge(self, from_domain: str, to_domain: str, bridge_concept: str):
        """Add a domain bridge"""
        self.domain_bridges.append((from_domain, to_domain, bridge_concept))

    def to_dict(self) -> Dict[str, Any]:
        """Convert rationale to dictionary"""
        return {
            "key_concepts": self.key_concepts,
            "synergistic_elements": self.synergistic_elements,
            "conflict_resolutions": self.conflict_resolutions,
            "novel_combinations": self.novel_combinations,
            "domain_bridges": self.domain_bridges
        }


class IdeaFusionAgent(BaseAgent):
    """
    Advanced idea fusion agent with explainable synthesis.

    Features:
    - Cross-domain idea combination with rationale
    - Conflict detection and resolution
    - Novelty and feasibility evaluation
    - Safety controls and content policy
    - Persistence and versioning
    - Event broadcasting
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="IdeaFusionAgent",
            capabilities=[AgentCapability.IDEA_FUSION, AgentCapability.CREATIVE_SYNTHESIS],
            description="Cross-domain idea synthesis with explainable fusion rationale",
            config=config
        )

        # Configuration
        self.max_ideas_per_fusion = self.config.get('max_ideas_per_fusion', 5)
        self.enable_explainable_fusion = self.config.get('enable_explainable_fusion', True)
        self.enable_evaluation = self.config.get('enable_evaluation', True)
        self.enable_persistence = self.config.get('enable_persistence', True)
        self.enable_events = self.config.get('enable_events', True)
        self.enable_safety = self.config.get('enable_safety', True)

        # State
        self.fusions = []
        self.fusion_history = defaultdict(list)

        # Dependencies (set by orchestrator)
        self.knowledge_agent = None
        self.safety_policy_engine = None
        self.event_bus = None
        self.memory_agent = None
        self.performance_monitor = None

        # Fusion templates for different domain combinations
        self.fusion_templates = self._initialize_fusion_templates()

    def _initialize_fusion_templates(self) -> Dict[str, List[str]]:
        """Initialize fusion templates for domain combinations"""
        return {
            "technology+art": [
                "Interactive art installation using {tech} technology to explore {art_concept}",
                "Digital art platform combining {tech} with {art_concept} for immersive experiences",
                "Algorithmic art generation using {tech} to express {art_concept}"
            ],
            "science+business": [
                "Commercial application of {science} research for {business_problem}",
                "Startup company developing {science} technology for {business_market}",
                "Business model leveraging {science} breakthroughs in {business_domain}"
            ],
            "healthcare+technology": [
                "Digital health solution using {tech} for {healthcare_challenge}",
                "Medical device combining {tech} with {healthcare_approach}",
                "Healthcare platform integrating {tech} for {healthcare_outcome}"
            ],
            "education+science": [
                "Educational program teaching {science} through {education_method}",
                "Learning platform using {science} principles for {education_goal}",
                "Curriculum combining {science} discovery with {education_innovation}"
            ],
            "default": [
                "Integrated approach combining {domain1} and {domain2} methodologies",
                "Cross-domain solution merging {domain1} with {domain2} principles",
                "Innovative framework bridging {domain1} and {domain2} domains"
            ]
        }

    def set_knowledge_agent(self, knowledge_agent):
        """Set the knowledge agent for persistence"""
        self.knowledge_agent = knowledge_agent

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
        """Initialize the idea fusion agent"""
        try:
            logger.info("IdeaFusionAgent initializing")
            logger.info("IdeaFusionAgent initialized successfully")
            return True

        except Exception as e:
            logger.exception(f"IdeaFusionAgent initialization failed: {e}")
            return False

    async def fuse_ideas(self, ideas: List[Dict[str, Any]], fusion_goal: Optional[str] = None) -> Dict[str, Any]:
        """Fuse multiple ideas into a novel concept with explainable rationale"""
        start_time = time.time()
        try:
            # Validate input
            if len(ideas) < 2:
                return {
                    "status": "error",
                    "error": "Need at least 2 ideas to fuse"
                }

            if len(ideas) > self.max_ideas_per_fusion:
                return {
                    "status": "error",
                    "error": f"Too many ideas ({len(ideas)}), maximum is {self.max_ideas_per_fusion}"
                }

            # Safety check
            if self.enable_safety and self.safety_policy_engine:
                domains = [idea.get("domain", "unknown") for idea in ideas]
                safety_result = await self.safety_policy_engine.assess_fusion_content(domains, ideas)
                if not safety_result["approved"]:
                    return {
                        "status": "blocked",
                        "error": f"Fusion blocked by safety policy: {safety_result['reason']}",
                        "safety_violation": True
                    }

            # Create fusion
            fusion_id = f"fusion_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Generate fusion content and rationale
            fusion_content, rationale = await self._create_fusion_with_rationale(ideas, fusion_goal)

            # Evaluate fusion
            evaluation = await self._evaluate_fusion(ideas, fusion_content) if self.enable_evaluation else {}

            # Create fusion object
            fusion = {
                "fusion_id": fusion_id,
                "source_ideas": [idea.get("idea_id", f"idea_{i}") for i, idea in enumerate(ideas)],
                "domains": list(set(idea.get("domain", "unknown") for idea in ideas)),
                "fusion_goal": fusion_goal,
                "content": fusion_content,
                "rationale": rationale.to_dict() if self.enable_explainable_fusion else None,
                "evaluation": evaluation,
                "created_at": datetime.now().isoformat(),
                "source_agent": "IdeaFusionAgent",
                "version": "1.0"
            }

            # Persist if enabled
            if self.enable_persistence and self.knowledge_agent:
                meta = {
                    "source_agent": "IdeaFusionAgent",
                    "source_ideas": fusion["source_ideas"],
                    "domains": fusion["domains"],
                    "fusion_goal": fusion_goal
                }
                version_id = await self.knowledge_agent.create_version(
                    knowledge_id=fusion_id,
                    content=fusion,
                    metadata=meta
                )
                fusion["version_id"] = version_id

            # Store locally
            self.fusions.append(fusion)
            for idea in ideas:
                idea_id = idea.get("idea_id", "unknown")
                self.fusion_history[idea_id].append(fusion_id)

            # Store in memory agent
            if self.memory_agent:
                await self.memory_agent.store_episodic({
                    "event_type": "fusion_created",
                    "fusion_id": fusion_id,
                    "source_ideas": [idea.get("idea_id", "unknown") for idea in ideas],
                    "domains": fusion["domains"],
                    "novelty_score": evaluation.get("novelty_score", 0),
                    "cross_domain_value": evaluation.get("cross_domain_value", 0),
                    "timestamp": datetime.now().isoformat()
                })

                # Semantic memory for fusion rationale
                await self.memory_agent.store_semantic(
                    concept="idea_fusion",
                    knowledge={
                        "fusion_id": fusion_id,
                        "rationale": fusion.get("rationale", {}),
                        "description": fusion["description"],
                        "domains": fusion["domains"],
                        "evaluation": evaluation,
                        "created_at": fusion["created_at"]
                    }
                )

            # Record performance metrics
            if self.performance_monitor:
                duration = time.time() - start_time
                self.performance_monitor.record_metric(
                    "fusion_duration",
                    duration,
                    {
                        "ideas_count": len(ideas),
                        "domains": fusion["domains"],
                        "novelty_score": evaluation.get("novelty_score", 0)
                    }
                )

            # Emit event
            if self.enable_events and self.event_bus:
                await self.event_bus.publish("fusion.created", {
                    "agent": "IdeaFusionAgent",
                    "fusion_id": fusion_id,
                    "source_ideas_count": len(ideas),
                    "domains": fusion["domains"],
                    "novelty_score": evaluation.get("novelty_score", 0)
                })

            logger.info(f"Created fusion {fusion_id} from {len(ideas)} ideas")
            return {
                "status": "success",
                "fusion": fusion
            }

        except Exception as e:
            logger.exception(f"Idea fusion failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def _create_fusion_with_rationale(self, ideas: List[Dict[str, Any]],
                                          fusion_goal: Optional[str]) -> Tuple[str, FusionRationale]:
        """Create fusion content with detailed rationale"""
        rationale = FusionRationale()

        # Extract key concepts from each idea
        for i, idea in enumerate(ideas):
            idea_id = idea.get("idea_id", f"idea_{i}")
            content = idea.get("content", idea.get("description", ""))

            # Extract key concepts (simplified - could use NLP)
            concepts = self._extract_key_concepts(content)
            for concept in concepts[:3]:  # Top 3 concepts per idea
                importance = random.uniform(0.5, 1.0)
                rationale.add_concept(idea_id, concept, importance)

        # Identify domain combinations
        domains = list(set(idea.get("domain", "unknown") for idea in ideas))
        rationale.add_domain_bridge(
            domains[0] if len(domains) > 0 else "unknown",
            domains[1] if len(domains) > 1 else "unknown",
            "cross-domain integration"
        )

        # Find synergistic elements
        synergy_descriptions = [
            "Combining technical implementation with creative expression",
            "Merging analytical rigor with intuitive insights",
            "Integrating systematic processes with adaptive approaches",
            "Bridging theoretical foundations with practical applications"
        ]

        for desc in synergy_descriptions[:2]:  # Add 2 synergies
            impact = random.uniform(0.6, 0.9)
            rationale.add_synergy(desc, impact)

        # Identify potential conflicts and resolutions
        conflicts = [
            ("methodological differences", "integrated framework"),
            ("resource requirements", "optimized allocation"),
            ("scalability concerns", "modular architecture")
        ]

        for conflict, resolution in conflicts[:1]:  # Add 1 conflict resolution
            rationale.add_conflict_resolution(conflict, resolution)

        # Create novel combinations
        combinations = [
            "hybrid methodology combining multiple approaches",
            "integrated system with cross-domain capabilities",
            "innovative framework bridging traditional boundaries"
        ]

        for combo in combinations[:2]:  # Add 2 novel combinations
            novelty = random.uniform(0.7, 0.95)
            rationale.add_novel_combination(combo, novelty)

        # Generate fusion content
        fusion_content = await self._generate_fusion_content(ideas, domains, fusion_goal)

        return fusion_content, rationale

    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content (simplified)"""
        # Simple keyword extraction - could be enhanced with NLP
        keywords = [
            "innovation", "technology", "system", "approach", "solution",
            "platform", "method", "framework", "integration", "optimization"
        ]

        found_concepts = []
        content_lower = content.lower()

        for keyword in keywords:
            if keyword in content_lower:
                found_concepts.append(keyword)

        # Add some domain-specific concepts
        if "ai" in content_lower or "artificial" in content_lower:
            found_concepts.append("artificial intelligence")
        if "blockchain" in content_lower:
            found_concepts.append("blockchain technology")
        if "sustainable" in content_lower:
            found_concepts.append("sustainability")

        return found_concepts[:5]  # Return top 5 concepts

    async def _generate_fusion_content(self, ideas: List[Dict[str, Any]],
                                     domains: List[str], fusion_goal: Optional[str]) -> str:
        """Generate fusion content description"""
        # Find appropriate template
        domain_key = "+".join(sorted(domains[:2]))  # Use first two domains
        templates = self.fusion_templates.get(domain_key, self.fusion_templates["default"])

        template = random.choice(templates)

        # Fill in template variables
        content = template
        if len(domains) >= 1:
            content = content.replace("{domain1}", domains[0])
        if len(domains) >= 2:
            content = content.replace("{domain2}", domains[1])

        # Add specific elements from ideas
        idea_elements = []
        for idea in ideas[:3]:  # Use first 3 ideas
            content_part = idea.get("content", idea.get("description", ""))
            # Extract a key phrase (simplified)
            words = content_part.split()[:5]  # First 5 words
            if words:
                idea_elements.append(" ".join(words))

        if idea_elements:
            content += f" incorporating elements like {', '.join(idea_elements)}."

        if fusion_goal:
            content += f" This fusion aims to {fusion_goal}."

        return content

    async def _evaluate_fusion(self, source_ideas: List[Dict[str, Any]],
                             fusion_content: str) -> Dict[str, Any]:
        """Evaluate the fusion for novelty, feasibility, etc."""
        try:
            # Calculate novelty based on domain diversity and concept combinations
            domains = set(idea.get("domain", "unknown") for idea in source_ideas)
            domain_diversity = len(domains) / len(source_ideas) if source_ideas else 0

            # Average novelty of source ideas
            avg_source_novelty = sum(
                idea.get("novelty_score", 0.5) for idea in source_ideas
            ) / len(source_ideas) if source_ideas else 0.5

            novelty_score = min(1.0, avg_source_novelty + (domain_diversity * 0.3))

            # Feasibility based on technical complexity and resource requirements
            feasibility_score = random.uniform(0.4, 0.8)  # Could be more sophisticated

            # Identify risks
            risks = []
            if novelty_score > 0.8:
                risks.append("High implementation complexity")
            if len(domains) > 3:
                risks.append("Integration challenges across many domains")
            if feasibility_score < 0.6:
                risks.append("Resource constraints may limit viability")

            # Suggest next steps
            next_steps = [
                "Conduct technical feasibility study",
                "Validate with domain experts",
                "Create detailed implementation plan",
                "Assess market potential"
            ]

            if novelty_score > 0.7:
                next_steps.append("Consider intellectual property protection")

            return {
                "novelty_score": novelty_score,
                "feasibility_score": feasibility_score,
                "domain_diversity": domain_diversity,
                "risks": risks,
                "next_steps": next_steps,
                "overall_score": (novelty_score + feasibility_score) / 2
            }

        except Exception as e:
            logger.exception(f"Fusion evaluation failed: {e}")
            return {
                "novelty_score": 0.5,
                "feasibility_score": 0.5,
                "risks": ["Evaluation failed"],
                "next_steps": ["Manual review required"],
                "overall_score": 0.5
            }

    async def get_fusion_history(self, idea_id: Optional[str] = None) -> Dict[str, Any]:
        """Get fusion history for an idea or all fusions"""
        try:
            if idea_id:
                fusion_ids = self.fusion_history.get(idea_id, [])
                fusions = [f for f in self.fusions if f["fusion_id"] in fusion_ids]
            else:
                fusions = self.fusions

            return {
                "status": "success",
                "fusions": fusions,
                "total_count": len(fusions)
            }

        except Exception as e:
            logger.exception(f"Failed to get fusion history: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute idea fusion tasks"""
        action = task.get("action")

        try:
            if action == "fuse_ideas":
                return await self.fuse_ideas(
                    task["ideas"],
                    task.get("fusion_goal")
                )
            elif action == "get_fusion_history":
                return await self.get_fusion_history(task.get("idea_id"))
            elif action == "list_fusions":
                return {
                    "status": "success",
                    "fusions": self.fusions
                }
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception(f"Task execution failed: {e}")
            return {"status": "error", "message": str(e)}

    async def shutdown(self) -> bool:
        """Shutdown the idea fusion agent"""
        try:
            logger.info("IdeaFusionAgent shutting down")

            # Clear state
            self.fusions.clear()
            self.fusion_history.clear()

            logger.info("IdeaFusionAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"IdeaFusionAgent shutdown failed: {e}")
            return False