#!/usr/bin/env python3
"""
CreativeAgent - Phase 10: Creative Cognition & Synthetic Intuition

Generates creative ideas and inventions with dream mode capabilities.
Enhanced with persistence, async operations, event broadcasting, and safety controls.
"""

import asyncio
import logging
import random
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from ..base_agent import BaseAgent, AgentCapability
from ..knowledge.rollback_manager import RollbackManager
from ..memory.memory_agent import MemoryAgent
from ..core.planner import PlannerAgent
from ..cognitive.performance_monitor import PerformanceMonitorAgent
from ...eventbus import EventBus

logger = logging.getLogger("kalki.agents.creative")


class CreativeBackend(ABC):
    """Pluggable backend interface for creativity generation"""

    @abstractmethod
    async def generate(self, domain: str, constraints: Optional[Dict[str, Any]] = None,
                      seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate creative content"""
        pass

    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get backend capabilities"""
        pass


class RuleBasedCreativeBackend(CreativeBackend):
    """Default rule-based creative backend"""

    def get_capabilities(self) -> List[str]:
        return ["rule_based", "template_based"]

    async def generate(self, domain: str, constraints: Optional[Dict[str, Any]] = None,
                      seed: Optional[int] = None) -> Dict[str, Any]:
        """Generate creative content using rule-based templates"""
        if seed is not None:
            random.seed(seed)

        templates = {
            "technology": [
                "A novel approach to {domain} that combines emerging technologies like AI and quantum computing",
                "Revolutionary {domain} solution using blockchain for decentralized coordination",
                "Next-generation {domain} platform with immersive AR/VR integration",
                "Sustainable {domain} innovation powered by renewable energy systems"
            ],
            "art": [
                "Interactive art installation exploring {domain} through generative algorithms",
                "Cross-cultural artistic fusion blending {domain} with traditional techniques",
                "Digital art series examining {domain} through algorithmic composition",
                "Immersive art experience combining {domain} with sensory technology"
            ],
            "science": [
                "Breakthrough hypothesis in {domain} research challenging current paradigms",
                "Interdisciplinary approach combining {domain} with computational modeling",
                "Novel experimental methodology for {domain} investigation",
                "Theoretical framework unifying {domain} with emerging scientific fields"
            ],
            "business": [
                "Disruptive business model for {domain} leveraging platform economics",
                "Social enterprise approach to {domain} addressing global challenges",
                "Decentralized marketplace for {domain} using smart contracts",
                "Circular economy solution for {domain} with sustainable value chains"
            ],
            "healthcare": [
                "Personalized {domain} solution using AI-driven diagnostics",
                "Preventive care platform for {domain} with predictive analytics",
                "Telemedicine innovation in {domain} improving access to care",
                "Digital therapeutics for {domain} combining software and medical devices"
            ],
            "education": [
                "Adaptive learning platform for {domain} using cognitive science",
                "Peer-to-peer knowledge sharing network for {domain} expertise",
                "Gamified education system for {domain} skill development",
                "Lifelong learning ecosystem for {domain} professional development"
            ]
        }

        domain_templates = templates.get(domain, [
            f"A creative concept in {domain} combining multiple perspectives",
            f"Innovative approach to {domain} challenges",
            f"Transformative solution for {domain} opportunities"
        ])

        description = random.choice(domain_templates).format(domain=domain)

        return {
            "content": description,
            "method": "rule_based",
            "confidence": random.uniform(0.7, 0.95)
        }


class CreativeAgent(BaseAgent):
    """
    Advanced creative agent with dream mode, persistence, and pluggable backends.

    Features:
    - Creative idea generation with domain expertise
    - Dream mode with resource controls and TTL
    - Persistence with provenance tracking
    - Async operations and event broadcasting
    - Pluggable backends for different generation methods
    - Safety controls and content policy
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="CreativeAgent",
            capabilities=[
                AgentCapability.CREATIVE_SYNTHESIS,
                AgentCapability.PATTERN_RECOGNITION,
                AgentCapability.IDEA_FUSION
            ],
            description="Advanced creative cognition with dream mode and synthesis capabilities",
            config=config
        )

        # Configuration
        self.max_dreams = self.config.get('max_dreams', 10)
        self.dream_rate_limit = self.config.get('dream_rate_limit', 5)  # dreams per minute
        self.enable_dream_mode = self.config.get('enable_dream_mode', False)
        self.dream_ttl_seconds = self.config.get('dream_ttl_seconds', 3600)  # 1 hour
        self.enable_persistence = self.config.get('enable_persistence', True)
        self.enable_events = self.config.get('enable_events', True)
        self.enable_safety = self.config.get('enable_safety', True)

        # State
        self.ideas = []
        self.dream_sessions = {}
        self.active_dreams = 0
        self.dream_mode_enabled_at = None
        self.dream_mode_expires_at = None
        self.rate_limiter = {}

        # Components
        self.knowledge_agent = None  # Will be set by orchestrator
        self.event_bus = None
        self.safety_policy_engine = None
        self.memory_agent = None
        self.planner_agent = None
        self.performance_monitor = None

        # Backends
        self.backends = {}
        self.default_backend = "rule_based"
        self._register_default_backends()

        # Persistence
        if self.enable_persistence:
            self.rollback_manager = RollbackManager(self.config.get('rollback_config', {}))

        # Session tracking
        self.current_session_id = f"session_{int(time.time())}"

    def _register_default_backends(self):
        """Register default creative backends"""
        self.backends["rule_based"] = RuleBasedCreativeBackend()

    async def register_backend(self, name: str, backend: CreativeBackend) -> bool:
        """Register a custom creative backend"""
        try:
            self.backends[name] = backend
            logger.info(f"Registered creative backend: {name}")
            return True
        except Exception as e:
            logger.error(f"Failed to register backend {name}: {e}")
            return False

    def set_knowledge_agent(self, knowledge_agent):
        """Set the knowledge agent for persistence"""
        self.knowledge_agent = knowledge_agent

    def set_event_bus(self, event_bus: EventBus):
        """Set the event bus for broadcasting"""
        self.event_bus = event_bus

    def set_safety_policy_engine(self, safety_engine):
        """Set the safety policy engine"""
        self.safety_policy_engine = safety_engine

    def set_memory_agent(self, memory_agent: MemoryAgent):
        """Set the memory agent for episodic/semantic storage"""
        self.memory_agent = memory_agent

    def set_planner_agent(self, planner_agent: PlannerAgent):
        """Set the planner agent for idea-to-plan conversion"""
        self.planner_agent = planner_agent

    def set_performance_monitor(self, performance_monitor: PerformanceMonitorAgent):
        """Set the performance monitor for metrics tracking"""
        self.performance_monitor = performance_monitor

    async def initialize(self) -> bool:
        """Initialize the creative agent"""
        try:
            logger.info("CreativeAgent initializing")

            # Initialize rollback manager if persistence enabled
            if self.enable_persistence:
                initialized = await self.rollback_manager.initialize()
                if not initialized:
                    logger.warning("Failed to initialize rollback manager, disabling persistence")
                    self.enable_persistence = False

            # Auto-enable dream mode if configured
            if self.enable_dream_mode:
                await self.enable_dream_mode_with_ttl(self.dream_ttl_seconds)

            logger.info("CreativeAgent initialized successfully")
            return True

        except Exception as e:
            logger.exception(f"CreativeAgent initialization failed: {e}")
            return False

    async def enable_dream_mode_with_ttl(self, ttl_seconds: int) -> bool:
        """Enable dream mode with automatic expiration"""
        try:
            self.enable_dream_mode = True
            self.dream_mode_enabled_at = datetime.now()
            self.dream_mode_expires_at = datetime.now() + timedelta(seconds=ttl_seconds)

            # Schedule auto-disable
            asyncio.create_task(self._auto_disable_dream_mode(ttl_seconds))

            # Emit event
            if self.enable_events and self.event_bus:
                await self.event_bus.publish("dream_mode.enabled", {
                    "agent": "CreativeAgent",
                    "ttl_seconds": ttl_seconds,
                    "expires_at": self.dream_mode_expires_at.isoformat()
                })

            logger.info(f"Dream mode enabled for {ttl_seconds} seconds")
            return True

        except Exception as e:
            logger.exception(f"Failed to enable dream mode: {e}")
            return False

    async def _auto_disable_dream_mode(self, delay_seconds: int):
        """Automatically disable dream mode after TTL"""
        await asyncio.sleep(delay_seconds)
        await self.disable_dream_mode()

    async def disable_dream_mode(self) -> bool:
        """Disable dream mode"""
        try:
            self.enable_dream_mode = False
            self.dream_mode_enabled_at = None
            self.dream_mode_expires_at = None

            # Emit event
            if self.enable_events and self.event_bus:
                await self.event_bus.publish("dream_mode.disabled", {
                    "agent": "CreativeAgent",
                    "reason": "ttl_expired"
                })

            logger.info("Dream mode disabled")
            return True

        except Exception as e:
            logger.exception(f"Failed to disable dream mode: {e}")
            return False

    def _check_dream_mode_status(self) -> bool:
        """Check if dream mode is active and within limits"""
        if not self.enable_dream_mode:
            return False

        # Check TTL
        if self.dream_mode_expires_at and datetime.now() > self.dream_mode_expires_at:
            self.enable_dream_mode = False
            return False

        return True

    def _check_rate_limits(self, operation: str) -> bool:
        """Check rate limits for operations"""
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Clean old entries
        self.rate_limiter = {
            op: timestamps for op, timestamps in self.rate_limiter.items()
            if timestamps
        }
        for op in self.rate_limiter:
            self.rate_limiter[op] = [ts for ts in self.rate_limiter[op] if ts > window_start]

        # Check limit
        if operation == "dream":
            current_count = len(self.rate_limiter.get(operation, []))
            if current_count >= self.dream_rate_limit:
                return False

        return True

    def _record_operation(self, operation: str):
        """Record operation for rate limiting"""
        now = time.time()
        if operation not in self.rate_limiter:
            self.rate_limiter[operation] = []
        self.rate_limiter[operation].append(now)

    async def generate_idea(self, domain: str, constraints: Optional[Dict[str, Any]] = None,
                           seed: Optional[int] = None, backend_name: Optional[str] = None) -> Dict[str, Any]:
        """Generate a creative idea with persistence and evaluation"""
        start_time = time.time()
        try:
            # Safety check
            if self.enable_safety and self.safety_policy_engine:
                safety_result = await self.safety_policy_engine.assess_creative_content(domain, constraints)
                if not safety_result["approved"]:
                    return {
                        "status": "blocked",
                        "error": f"Content blocked by safety policy: {safety_result['reason']}",
                        "safety_violation": True
                    }

            # Use specified backend or default
            backend = self.backends.get(backend_name or self.default_backend)
            if not backend:
                return {
                    "status": "error",
                    "error": f"Backend '{backend_name or self.default_backend}' not available"
                }

            # Generate content
            generation_result = await backend.generate(domain, constraints, seed)

            # Create idea ID
            idea_id = f"idea_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

            # Create idea structure
            idea = {
                "idea_id": idea_id,
                "domain": domain,
                "constraints": constraints or {},
                "content": generation_result["content"],
                "backend": backend_name or self.default_backend,
                "method": generation_result.get("method", "unknown"),
                "confidence": generation_result.get("confidence", 0.5),
                "seed": seed,
                "session_id": self.current_session_id,
                "source_agent": "CreativeAgent",
                "version": "1.0",
                "created_at": datetime.now().isoformat()
            }

            # Evaluate the idea
            idea["evaluation"] = await self._evaluate_idea(idea)

            # Persist if enabled
            if self.enable_persistence and self.knowledge_agent:
                meta = {
                    "source_agent": "CreativeAgent",
                    "session_id": self.current_session_id,
                    "seed": seed,
                    "domain": domain,
                    "backend": backend_name or self.default_backend
                }
                version_id = await self.knowledge_agent.create_version(
                    knowledge_id=idea_id,
                    content=idea,
                    metadata=meta
                )
                idea["version_id"] = version_id

            # Store locally
            self.ideas.append(idea)

            # Store in memory agent
            if self.memory_agent:
                # Episodic memory for the creation event
                await self.memory_agent.store_episodic({
                    "event_type": "idea_created",
                    "idea_id": idea_id,
                    "domain": domain,
                    "confidence": idea["confidence"],
                    "backend": backend_name or self.default_backend,
                    "session_id": self.current_session_id,
                    "timestamp": idea["created_at"]
                })

                # Semantic memory for the idea content
                await self.memory_agent.store_semantic(
                    concept=f"creative_idea_{domain}",
                    knowledge={
                        "idea_id": idea_id,
                        "content": idea["content"],
                        "domain": domain,
                        "confidence": idea["confidence"],
                        "evaluation": idea["evaluation"],
                        "created_at": idea["created_at"]
                    }
                )

            # Emit event
            if self.enable_events and self.event_bus:
                await self.event_bus.publish("idea.created", {
                    "agent": "CreativeAgent",
                    "idea_id": idea_id,
                    "domain": domain,
                    "confidence": idea["confidence"]
                })

            logger.info(f"Generated idea {idea_id} in domain {domain}")
            
            # Record performance metrics
            if self.performance_monitor:
                duration = time.time() - start_time
                self.performance_monitor.record_metric(
                    "idea_generation_duration",
                    duration,
                    {
                        "domain": domain,
                        "backend": backend_name or self.default_backend,
                        "confidence": idea["confidence"],
                        "session_id": self.current_session_id
                    }
                )
                self.performance_monitor.record_metric(
                    "idea_generation_success_rate",
                    1.0,
                    {"domain": domain}
                )
            
            return {
                "status": "success",
                "idea": idea
            }

        except Exception as e:
            logger.exception(f"Failed to generate idea: {e}")
            
            # Record failure metrics
            if self.performance_monitor:
                duration = time.time() - start_time
                self.performance_monitor.record_metric(
                    "idea_generation_duration",
                    duration,
                    {
                        "domain": domain,
                        "backend": backend_name or self.default_backend,
                        "error": str(e),
                        "session_id": self.current_session_id
                    }
                )
                self.performance_monitor.record_metric(
                    "idea_generation_success_rate",
                    0.0,
                    {"domain": domain}
                )
            
            return {
                "status": "error",
                "error": str(e)
            }

    async def convert_to_plan(self, idea_id: str) -> Dict[str, Any]:
        """Convert a creative idea into an actionable implementation plan"""
        try:
            # Find the idea
            idea = None
            for i in self.ideas:
                if i["idea_id"] == idea_id:
                    idea = i
                    break

            if not idea:
                return {
                    "status": "error",
                    "error": f"Idea {idea_id} not found"
                }

            if not self.planner_agent:
                return {
                    "status": "error",
                    "error": "PlannerAgent not available"
                }

            # Create plan from idea
            plan_result = await self.planner_agent.execute({
                "action": "plan",
                "params": {
                    "goal": f"Implement creative idea: {idea['content']}",
                    "constraints": {
                        "domain": idea["domain"],
                        "feasibility_score": idea["evaluation"]["feasibility_score"],
                        "resources_required": self._estimate_resources(idea)
                    },
                    "max_steps": 10,
                    "reasoning_context": f"Creative idea with novelty score {idea['evaluation']['novelty_score']:.2f} and feasibility score {idea['evaluation']['feasibility_score']:.2f}"
                }
            })

            if plan_result["status"] == "success":
                plan = plan_result["plan"]
                plan["source_idea_id"] = idea_id
                plan["created_from"] = "CreativeAgent"

                # Store plan in memory
                if self.memory_agent:
                    await self.memory_agent.store_episodic({
                        "event_type": "plan_created_from_idea",
                        "idea_id": idea_id,
                        "plan_id": plan.get("plan_id", f"plan_{idea_id}"),
                        "domain": idea["domain"],
                        "steps_count": len(plan.get("steps", [])),
                        "timestamp": datetime.now().isoformat()
                    })

                # Emit event
                if self.enable_events and self.event_bus:
                    await self.event_bus.publish("idea.converted_to_plan", {
                        "agent": "CreativeAgent",
                        "idea_id": idea_id,
                        "plan_id": plan.get("plan_id"),
                        "domain": idea["domain"],
                        "steps_count": len(plan.get("steps", []))
                    })

                return {
                    "status": "success",
                    "plan": plan,
                    "idea": idea
                }
            else:
                return plan_result

        except Exception as e:
            logger.exception(f"Failed to convert idea to plan: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    def _estimate_resources(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate resources needed to implement the idea"""
        domain = idea["domain"]
        feasibility = idea["evaluation"]["feasibility_score"]

        # Simplified resource estimation
        if domain in ["technology", "science"]:
            return {
                "technical_expertise": "high" if feasibility < 0.6 else "medium",
                "development_time": "6-12 months",
                "team_size": 3 if feasibility > 0.7 else 5,
                "budget_range": "$100K-$500K"
            }
        elif domain == "business":
            return {
                "market_research": "required",
                "development_time": "3-6 months",
                "team_size": 2,
                "budget_range": "$50K-$200K"
            }
        else:
            return {
                "specialized_skills": f"{domain} expertise",
                "development_time": "variable",
                "team_size": "variable",
                "budget_range": "variable"
            }

    async def _evaluate_idea(self, idea: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate idea for novelty, feasibility, risks, and next steps"""
        try:
            # Calculate scores (simplified - can be enhanced with ML models)
            novelty_score = random.uniform(0.4, 0.95)  # Based on uniqueness
            feasibility_score = random.uniform(0.3, 0.85)  # Based on technical/practical constraints

            # Identify risks
            risks = []
            if feasibility_score < 0.5:
                risks.append("High technical complexity")
            if novelty_score > 0.8:
                risks.append("Market acceptance uncertainty")
            if idea.get("domain") in ["healthcare", "finance"]:
                risks.append("Regulatory compliance required")

            # Suggest next steps
            next_steps = [
                "Conduct feasibility analysis",
                "Validate with domain experts",
                "Create proof of concept"
            ]

            if novelty_score > 0.7:
                next_steps.append("File provisional patent")
            if feasibility_score > 0.6:
                next_steps.append("Develop business case")

            return {
                "novelty_score": novelty_score,
                "feasibility_score": feasibility_score,
                "risks": risks,
                "next_steps": next_steps,
                "overall_score": (novelty_score + feasibility_score) / 2
            }

        except Exception as e:
            logger.exception(f"Evaluation failed: {e}")
            return {
                "novelty_score": 0.5,
                "feasibility_score": 0.5,
                "risks": ["Evaluation failed"],
                "next_steps": ["Manual review required"],
                "overall_score": 0.5
            }

    async def dream(self, theme: Optional[str] = None, count: Optional[int] = None) -> Dict[str, Any]:
        """Generate multiple ideas in dream mode with resource controls"""
        try:
            # Check dream mode status
            if not self._check_dream_mode_status():
                return {
                    "status": "error",
                    "error": "Dream mode not enabled or expired"
                }

            # Check rate limits
            if not self._check_rate_limits("dream"):
                return {
                    "status": "rate_limited",
                    "error": f"Dream rate limit exceeded ({self.dream_rate_limit}/minute)"
                }

            # Determine count
            if count is None:
                count = random.randint(3, min(7, self.max_dreams))
            count = min(count, self.max_dreams)

            # Check active dreams limit
            if self.active_dreams + count > self.max_dreams:
                return {
                    "status": "limit_exceeded",
                    "error": f"Would exceed max dreams limit ({self.max_dreams})"
                }

            # Record operation
            self._record_operation("dream")
            self.active_dreams += count

            # Generate dreams in parallel
            domains = ["technology", "art", "science", "business", "healthcare", "education"]
            dream_tasks = []

            for i in range(count):
                domain = random.choice(domains)
                seed = random.randint(0, 2**32 - 1) if random.random() < 0.5 else None
                task = self.generate_idea(domain, {"dream_theme": theme}, seed)
                dream_tasks.append(task)

            # Execute in parallel
            results = await asyncio.gather(*dream_tasks, return_exceptions=True)

            # Process results
            dreams = []
            errors = []

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    errors.append(f"Dream {i}: {str(result)}")
                elif result.get("status") == "success":
                    dream = result["idea"]
                    dream["dream_index"] = i
                    dream["dream_theme"] = theme
                    dreams.append(dream)
                else:
                    errors.append(f"Dream {i}: {result.get('error', 'Unknown error')}")

            # Update active dreams count
            self.active_dreams -= count

            # Emit event
            if self.enable_events and self.event_bus:
                await self.event_bus.publish("dream.completed", {
                    "agent": "CreativeAgent",
                    "dream_count": len(dreams),
                    "error_count": len(errors),
                    "theme": theme
                })

            logger.info(f"Completed dream session: {len(dreams)} ideas, {len(errors)} errors")

            return {
                "status": "success",
                "dreams": dreams,
                "errors": errors,
                "theme": theme,
                "total_requested": count
            }

        except Exception as e:
            logger.exception(f"Dream generation failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute creative tasks"""
        action = task.get("action")

        try:
            if action == "generate_idea":
                return await self.generate_idea(
                    task["domain"],
                    task.get("constraints"),
                    task.get("seed"),
                    task.get("backend")
                )
            elif action == "dream":
                return await self.dream(
                    task.get("theme"),
                    task.get("count")
                )
            elif action == "enable_dream_mode":
                success = await self.enable_dream_mode_with_ttl(task.get("ttl_seconds", 3600))
                return {"status": "success" if success else "error"}
            elif action == "disable_dream_mode":
                success = await self.disable_dream_mode()
                return {"status": "success" if success else "error"}
            elif action == "list_ideas":
                return {"status": "success", "ideas": self.ideas}
            elif action == "get_status":
                return {
                    "status": "success",
                    "dream_mode_enabled": self._check_dream_mode_status(),
                    "active_dreams": self.active_dreams,
                    "total_ideas": len(self.ideas),
                    "available_backends": list(self.backends.keys())
                }
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception(f"Task execution failed: {e}")
            return {"status": "error", "message": str(e)}

    async def shutdown(self) -> bool:
        """Shutdown the creative agent"""
        try:
            logger.info("CreativeAgent shutting down")

            # Disable dream mode
            await self.disable_dream_mode()

            # Clear state
            self.ideas.clear()
            self.dream_sessions.clear()

            logger.info("CreativeAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"CreativeAgent shutdown failed: {e}")
            return False