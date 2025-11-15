# ============================================================
# Kalki — orchestrator.py
# ------------------------------------------------------------
# Orchestrator - The Central Nervous System
# - Coordinates all 20 phases of cognitive evolution
# - Manages multi-agent orchestration
# - Handles complex task decomposition and execution
# - Provides sci-fi grade AI capabilities
# ============================================================

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime
from pathlib import Path
import logging
import uuid

from modules.utils.config import get_config, CONFIG
from modules.utils.logging_config import get_logger
from modules.utils.eventbus import EventBus
from modules.llm import get_llm_engine, LLMEngine
from modules.learning.vectordb import BGEEmbedder, VectorDBManager
from modules.agents.agent_manager import AgentManager
from modules.generative_design_engine import GenerativeDesignEngine
from modules.consciousness_engine import ConsciousnessEngine
from modules.supreme_synthesis_engine import SupremeSynthesisEngine
from modules.self_evolution_manager import SelfEvolutionManager
from modules.metrics.collector import MetricsCollector
from modules.robustness import RobustnessManager
from modules.professional_workflow import (
    ProfessionalWorkflow,
    ProfessionalWorkflowExecutor,
    WorkflowStep,
    ProfessionalRole
)

logger = get_logger("Kalki.Orchestrator")

class TaskComplexity:
    """Task complexity levels for intelligent routing"""
    SIMPLE = "simple"          # Basic queries, calculations
    COMPLEX = "complex"        # Multi-step reasoning, analysis
    CREATIVE = "creative"      # Design, synthesis, innovation
    SCIENTIFIC = "scientific"  # Research, experimentation
    STRATEGIC = "strategic"    # Planning, optimization, evolution

class KalkiOrchestrator:
    """
    The Orchestrator - Central coordination system for Kalki

    Capabilities:
    - 20-phase cognitive evolution orchestration
    - Multi-modal task processing (text, vision, audio, design)
    - Self-evolving agent coordination
    - Real-time system optimization
    - Quantum-inspired decision making
    - Consciousness-driven task execution
    """

    def __init__(self):
        self.config = CONFIG
        self.logger = logger

        # Core systems
        self.eventbus = EventBus()
        self.llm_engine = None
        self.vector_db = None
        self.agent_manager = None
        self.metrics = None
        self.robustness = None

        # Advanced engines
        self.design_engine = None
        self.consciousness_engine = None
        self.synthesis_engine = None
        self.evolution_manager = None
        
        # Professional workflow support
        self.workflow_executor = None

        # Task management
        self.active_tasks = {}
        self.task_history = []
        self.performance_metrics = {}

        # System state
        self.system_health = "initializing"
        self.capabilities = self._load_capabilities()

        # Initialize all systems
        self._initialize_systems()

    def _load_capabilities(self) -> Dict[str, Any]:
        """Load system capabilities manifest"""
        return {
            "cognitive_phases": 20,
            "agent_types": ["planning", "execution", "monitoring", "evolution"],
            "modalities": ["text", "vision", "audio", "design", "simulation"],
            "specializations": ["scientific", "creative", "strategic", "technical"],
            "quantum_capable": True,
            "consciousness_driven": True,
            "self_evolving": True
        }

    def _initialize_systems(self):
        """Initialize all Kalki subsystems"""
        try:
            # Core infrastructure
            self.llm_engine = get_llm_engine()
            self.vector_db = VectorDBManager()
            self.agent_manager = AgentManager()
            self.metrics = MetricsCollector()
            self.robustness = RobustnessManager(self.eventbus)

            # Advanced engines
            self.design_engine = GenerativeDesignEngine()
            self.consciousness_engine = ConsciousnessEngine()
            self.synthesis_engine = SupremeSynthesisEngine()
            self.evolution_manager = SelfEvolutionManager()
            
            # Professional workflow executor (requires team orchestrator)
            # Will be initialized when team orchestrator is available
            self.workflow_executor = None

            self.system_health = "operational"
            self.logger.info("🧠 Kalki Orchestrator fully initialized - All systems operational")

        except Exception as e:
            self.system_health = "degraded"
            self.logger.error(f"❌ System initialization failed: {e}")
            raise

    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process any complex task using the full Kalki system

        Args:
            task: Task specification with query, context, requirements

        Returns:
            Complete task result with all phases executed
        """
        task_id = str(uuid.uuid4())
        start_time = time.time()

        try:
            # Phase 1: Task Analysis & Decomposition
            analysis = await self._analyze_task(task)
            complexity = self._determine_complexity(analysis)

            # Phase 2: Agent Assembly & Coordination
            agents = await self._assemble_agent_team(complexity, analysis)

            # Phase 3: Multi-Modal Processing
            context = await self._gather_context(task, analysis)

            # Phase 4: Consciousness-Driven Execution
            result = await self._execute_with_consciousness(task, agents, context, complexity)

            # Phase 5: Synthesis & Optimization
            agent_results = result.get("agent_results", [])
            
            # Use SupremeSynthesisEngine for complex synthesis
            if complexity in [TaskComplexity.SCIENTIFIC, TaskComplexity.STRATEGIC, TaskComplexity.CREATIVE]:
                final_result = await self._synthesize_with_supreme_engine(task, agent_results, context, complexity)
            else:
                final_result = await self._synthesize_result(task, agent_results, context, complexity)

            # Phase 6: Self-Evolution Learning
            await self._learn_from_execution(task, result, final_result)
            
            # Phase 7: Real-Time Learning & Memory Storage
            await self._store_in_memory(task, final_result, context)
            await self._learn_from_feedback(task, final_result)

            # Record metrics
            execution_time = time.time() - start_time
            await self._record_task_metrics(task_id, task, result, execution_time)

            return {
                "task_id": task_id,
                "status": "completed",
                "result": final_result,
                "complexity": complexity,
                "execution_time": execution_time,
                "agents_used": len(agents),
                "phases_executed": 7  # Now includes real-time learning & memory
            }

        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return {
                "task_id": task_id,
                "status": "failed",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "execution_time": time.time() - start_time
            }

    async def _analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task requirements and determine execution strategy with intelligent routing"""
        query = task.get("query", "")
        context = task.get("context", {})
        
        # Infer domain from query using DomainRegistry
        inferred_domain = None
        try:
            from modules.domains.domain_registry import DomainRegistry
            domain_registry = DomainRegistry()
            if domain_registry:
                # Simple domain inference from query keywords
                query_lower = query.lower()
                if any(word in query_lower for word in ["construction", "building", "house", "adu", "remodel"]):
                    inferred_domain = "construction"
                elif any(word in query_lower for word in ["game", "unity", "unreal", "gameplay"]):
                    inferred_domain = "game_dev"
                elif any(word in query_lower for word in ["robot", "robotics", "sensor", "actuator"]):
                    inferred_domain = "robotics"
                elif any(word in query_lower for word in ["aircraft", "drone", "uav", "aerospace", "flight"]):
                    inferred_domain = "aerospace"
                elif any(word in query_lower for word in ["battery", "solar", "power", "energy", "grid"]):
                    inferred_domain = "power_systems"
        except Exception as e:
            self.logger.debug(f"Domain inference failed: {e}")

        # Use LLM for deep task analysis with advanced reasoning for complex tasks
        analysis_prompt = f"""
        Analyze this task comprehensively:

        Query: {query}
        Context: {json.dumps(context, indent=2)}
        Inferred Domain: {inferred_domain or "unknown"}

        Provide analysis covering:
        1. Task type and domain
        2. Required capabilities and expertise
        3. Complexity level and reasoning requirements
        4. Multi-modal aspects (text, vision, design, etc.)
        5. Expected output format and depth
        6. Potential challenges and edge cases
        7. Recommended routing strategy (which agents/modules to use)
        """

        # Enable advanced reasoning for complex queries
        use_advanced_reasoning = any(word in query.lower() for word in [
            "analyze", "design", "plan", "complex", "optimize", "strategy", "research"
        ])
        
        analysis_result = await self.llm_engine.generate(
            analysis_prompt, 
            max_tokens=1000,
            use_advanced_reasoning=use_advanced_reasoning,
            reasoning_method="cot" if use_advanced_reasoning else None
        )
        
        # Parse analysis result - handle both JSON and plain text responses
        try:
            if analysis_result.strip().startswith('{'):
                return json.loads(analysis_result)
            else:
                # Fallback: create structured analysis from plain text
                return {
                    "task_type": "general",
                    "domain": inferred_domain or "mixed",
                    "capabilities": ["reasoning", "analysis"],
                    "complexity": "medium",
                    "modalities": ["text"],
                    "output_format": "comprehensive",
                    "challenges": [],
                    "inferred_domain": inferred_domain,
                    "raw_analysis": analysis_result
                }
        except json.JSONDecodeError:
            # Fallback for malformed JSON
            return {
                "task_type": "general",
                "domain": inferred_domain or "unknown",
                "capabilities": ["basic_processing"],
                "complexity": "simple",
                "modalities": ["text"],
                "output_format": "text",
                "challenges": ["json_parsing_failed"],
                "inferred_domain": inferred_domain,
                "raw_analysis": analysis_result
            }

    def _determine_complexity(self, analysis: Dict[str, Any]) -> str:
        """Determine task complexity for routing"""
        complexity_indicators = analysis.get("complexity_indicators", [])

        if any(indicator in ["research", "discovery", "breakthrough"] for indicator in complexity_indicators):
            return TaskComplexity.SCIENTIFIC
        elif any(indicator in ["design", "create", "innovate"] for indicator in complexity_indicators):
            return TaskComplexity.CREATIVE
        elif any(indicator in ["plan", "optimize", "strategy"] for indicator in complexity_indicators):
            return TaskComplexity.STRATEGIC
        elif any(indicator in ["analyze", "reason", "multiple_steps"] for indicator in complexity_indicators):
            return TaskComplexity.COMPLEX
        else:
            return TaskComplexity.SIMPLE

    async def _assemble_agent_team(self, complexity: str, analysis: Dict[str, Any]) -> List[Any]:
        """Assemble optimal agent team for task execution with intelligent routing"""
        from modules.agents.base_agent import AgentCapability
        
        agents = []
        
        # Route to domain-specific professional teams if domain is identified
        # Prefer copilots for enhanced processing
        inferred_domain = analysis.get("inferred_domain") or analysis.get("domain")
        if inferred_domain and inferred_domain != "unknown" and inferred_domain != "mixed":
            try:
                from modules.domains.domain_registry import DomainRegistry
                domain_registry = DomainRegistry()
                
                # Try to get copilot first (prefer_copilot=True)
                domain_or_copilot = domain_registry.get_domain(inferred_domain, prefer_copilot=True)
                
                if domain_or_copilot:
                    # Check if it's a copilot (has copilot methods)
                    if hasattr(domain_or_copilot, 'get_team_orchestrator'):
                        # It's a domain with professional systems
                        team_orch = await domain_or_copilot.get_team_orchestrator()
                        if team_orch:
                            self.logger.info(f"Routing to {inferred_domain} domain professional team")
                            agents.append(team_orch)
                    elif hasattr(domain_or_copilot, 'start_new_project') or hasattr(domain_or_copilot, 'start_new_game_project'):
                        # It's a copilot - use it directly
                        self.logger.info(f"Routing to {inferred_domain} copilot for enhanced processing")
                        agents.append(domain_or_copilot)
            except Exception as e:
                self.logger.debug(f"Domain routing failed: {e}")

        # Always include consciousness agent for high-level coordination
        if self.consciousness_engine:
            agents.append(self.consciousness_engine)

        # Add specialized agents based on complexity
        if complexity == TaskComplexity.SCIENTIFIC:
            # Add research and experimentation agents
            research_agents = self.agent_manager.find_agents_by_capability(AgentCapability.ANALYTICS)
            experimentation_agents = self.agent_manager.find_agents_by_capability(AgentCapability.EXPERIMENTATION)
            agents.extend(research_agents)
            agents.extend(experimentation_agents)

        elif complexity == TaskComplexity.CREATIVE:
            # Add design and synthesis agents
            if self.design_engine:
                agents.append(self.design_engine)
            if self.synthesis_engine:
                agents.append(self.synthesis_engine)
            design_agents = self.agent_manager.find_agents_by_capability(AgentCapability.DESIGN)
            agents.extend(design_agents)

        elif complexity == TaskComplexity.STRATEGIC:
            # Add planning and optimization agents
            planning_agents = self.agent_manager.find_agents_by_capability(AgentCapability.PLANNING)
            reasoning_agents = self.agent_manager.find_agents_by_capability(AgentCapability.REASONING)
            agents.extend(planning_agents)
            agents.extend(reasoning_agents)

        # Limit to reasonable number of agents
        return agents[:5]  # Max 5 agents per task

    async def _gather_context(self, task: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Gather comprehensive context from all modalities, including advanced memory"""
        context = {"text": [], "visual": [], "design": [], "knowledge": [], "structured": [], "memory": {}}

        # Text context from vector DB
        query = task.get("query", "")
        if query and self.vector_db:
            text_results = self.vector_db.search_similar(query, top_k=10)
            context["text"] = text_results

        # Design context if applicable - simplified for now
        if "design" in analysis.get("modalities", []):
            context["design"] = []  # TODO: Implement design context gathering

        # Knowledge base context
        knowledge_results = self.vector_db.search_similar(f"knowledge about {query}", top_k=5)
        context["knowledge"] = knowledge_results
        
        # Structured knowledge from HybridLearningSystem
        try:
            from modules.hybrid_learning_system import get_hybrid_system
            hybrid_system = get_hybrid_system()
            if hybrid_system:
                # Query structured knowledge
                formulas = hybrid_system.query_formulas() if hasattr(hybrid_system, 'query_formulas') else []
                materials = hybrid_system.query_materials() if hasattr(hybrid_system, 'query_materials') else []
                design_rules = hybrid_system.query_design_rules() if hasattr(hybrid_system, 'query_design_rules') else []
                context["structured"] = {
                    "formulas": formulas[:5],  # Limit results
                    "materials": materials[:5],
                    "design_rules": design_rules[:5]
                }
        except Exception as e:
            self.logger.debug(f"HybridLearningSystem context gathering failed: {e}")
        
        # Advanced memory retrieval
        try:
            if self.advanced_memory is None:
                from modules.advanced_memory import AdvancedMemorySystem
                if self.llm_engine:
                    self.advanced_memory = AdvancedMemorySystem(self.llm_engine)
            
            if self.advanced_memory:
                domain = analysis.get("domain") or analysis.get("inferred_domain", "general")
                memories = await self.advanced_memory.retrieve_relevant_memories(
                    query=query,
                    context={"domain": domain, "task": task},
                    limit=5
                )
                context["memory"] = memories
        except Exception as e:
            self.logger.debug(f"Memory retrieval failed: {e}")

        return context

    async def _execute_with_consciousness(self, task: Dict[str, Any], agents: List[Any],
                                        context: Dict[str, Any], complexity: str) -> Dict[str, Any]:
        """Execute task with consciousness-driven coordination"""
        try:
            # Consciousness-driven execution
            consciousness_state = await self.consciousness_engine.achieve_consciousness({
                "orchestrator": {"task": task, "complexity": complexity, "agents": len(agents)}
            })

            # Execute with basic coordination but enhanced by consciousness
            result = await self._execute_basic_coordination(task, agents, context)
            
            # Enhance result with consciousness insights
            result["consciousness_state"] = consciousness_state.__dict__ if hasattr(consciousness_state, '__dict__') else str(consciousness_state)
            result["consciousness_level"] = getattr(consciousness_state, 'awareness_level', 0.0)
            
            return result
        except Exception as e:
            self.logger.warning(f"Consciousness execution failed, falling back to basic: {e}")
            return await self._execute_basic_coordination(task, agents, context)

    async def _execute_basic_coordination(self, task: Dict[str, Any], agents: List[Any],
                                        context: Dict[str, Any]) -> Dict[str, Any]:
        """Basic agent coordination without consciousness engine"""
        results = {}

        # Execute agents in parallel where possible
        tasks = []
        for agent in agents:
            if hasattr(agent, 'process'):
                task_coro = agent.process(task, context)
                tasks.append(task_coro)

        if tasks:
            agent_results = await asyncio.gather(*tasks, return_exceptions=True)
            results["agent_results"] = agent_results

        # Synthesize results
        try:
            query_text = ""
            if isinstance(task, dict):
                query_text = str(task.get("query", ""))

            synthesis_context = {
                "raw_results": results,
                "task": task,
                "execution_context": context
            }

            synthesis_result = await self.synthesis_engine.synthesize(
                query_text,
                synthesis_context
            )
            results["synthesis"] = synthesis_result
        except Exception as synthesis_error:
            self.logger.warning(f"Synthesis engine error: {synthesis_error}")

        return results

    async def _synthesize_with_supreme_engine(self, task: Dict[str, Any], agent_results: List[Dict[str, Any]],
                               context: Dict[str, Any], complexity: str) -> Dict[str, Any]:
        """Synthesize results using SupremeSynthesisEngine for complex tasks"""
        try:
            if not self.synthesis_engine:
                return await self._synthesize_basic(task, agent_results, context)
            
            # Supreme synthesis for complex tasks
            synthesis_input = {
                "task": task,
                "agent_results": agent_results,
                "context": context,
                "complexity": complexity,
                "orchestrator_state": {
                    "phase": "synthesis",
                    "agent_count": len(agent_results),
                    "task_type": task.get("type", "unknown")
                }
            }
            
            synthesis_result = await self.synthesis_engine.synthesize(synthesis_input)
            
            # Return enhanced synthesis result
            return {
                "synthesis": synthesis_result,
                "synthesis_method": "supreme_synthesis",
                "agent_contributions": len(agent_results),
                "complexity": complexity
            }
        except Exception as e:
            self.logger.warning(f"Supreme synthesis failed, falling back to basic: {e}")
            return await self._synthesize_basic(task, agent_results, context)
    
    async def _synthesize_result(self, task: Dict[str, Any], agent_results: List[Dict[str, Any]],
                               context: Dict[str, Any], complexity: str) -> Dict[str, Any]:
        """Synthesize results using standard synthesis"""
        return await self._synthesize_basic(task, agent_results, context)

    async def _synthesize_basic(self, task: Dict[str, Any], agent_results: List[Dict[str, Any]],
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Basic synthesis fallback when supreme synthesis fails"""
        try:
            # Simple synthesis: combine agent results
            valid_results = [r for r in agent_results if r is not None and not isinstance(r, Exception)]
            
            if not valid_results:
                return {"synthesis": "No valid agent results to synthesize", "method": "basic_fallback"}
            
            # Combine results based on task type
            task_type = task.get("type", "general")
            
            if task_type == "analysis":
                # For analysis tasks, summarize key insights
                combined_insights = []
                for result in valid_results:
                    if isinstance(result, dict) and "insights" in result:
                        combined_insights.extend(result["insights"])
                
                synthesis = {
                    "method": "basic_synthesis",
                    "task_type": task_type,
                    "combined_insights": combined_insights[:10],  # Limit to top 10
                    "total_agents": len(valid_results),
                    "confidence": min(0.8, len(valid_results) * 0.1)
                }
            
            elif task_type == "design":
                # For design tasks, combine design elements
                combined_designs = []
                for result in valid_results:
                    if isinstance(result, dict) and "design" in result:
                        combined_designs.append(result["design"])
                
                synthesis = {
                    "method": "basic_synthesis",
                    "task_type": task_type,
                    "combined_designs": combined_designs,
                    "total_agents": len(valid_results),
                    "recommendation": "Review combined designs for optimal solution"
                }
            
            else:
                # Generic synthesis
                synthesis = {
                    "method": "basic_synthesis",
                    "task_type": task_type,
                    "agent_results_summary": f"Combined {len(valid_results)} agent results",
                    "total_agents": len(valid_results),
                    "status": "completed"
                }
            
            return synthesis
            
        except Exception as e:
            self.logger.error(f"Basic synthesis failed: {e}")
            return {
                "method": "failed_synthesis",
                "error": str(e),
                "fallback_message": "Synthesis failed, returning raw agent results",
                "raw_results": agent_results
            }

    async def _learn_from_execution(self, task: Dict[str, Any], execution_result: Dict[str, Any],
                                  final_result: Dict[str, Any]):
        """Learn from task execution for self-improvement"""
        if not self.evolution_manager:
            return

        # Record execution for learning
        task_id = task.get("task_id", "unknown")
        learning_data = {
            "task_id": task_id,
            "task": task,
            "execution": execution_result,
            "result": final_result,
            "timestamp": datetime.now().isoformat(),
            "performance_metrics": self.performance_metrics
        }

        await self.evolution_manager.record_execution(task_id, learning_data)

        should_evolve = await self._should_evolve()
        if should_evolve:
            await self.evolution_manager.evolve_system("task_completion", learning_data)
    
    async def _store_in_memory(self, task: Dict[str, Any], result: Dict[str, Any], context: Dict[str, Any]):
        """Store task and result in advanced memory system"""
        try:
            if self.advanced_memory is None:
                # Lazy load advanced memory
                from modules.advanced_memory import AdvancedMemorySystem
                if self.llm_engine:
                    self.advanced_memory = AdvancedMemorySystem(self.llm_engine)
            
            if self.advanced_memory:
                # Store episodic memory
                await self.advanced_memory.store_episode(
                    episode_type="task",
                    content={
                        "task": task,
                        "result": result,
                        "context": context
                    },
                    domain=context.get("domain", "general"),
                    importance=0.7,
                    tags=context.get("tags", [])
                )
                
                # Store semantic memory if concepts extracted
                if "concepts" in result:
                    for concept, knowledge in result.get("concepts", {}).items():
                        await self.advanced_memory.store_semantic(
                            concept=concept,
                            knowledge=knowledge,
                            domain=context.get("domain", "general")
                        )
        except Exception as e:
            self.logger.debug(f"Memory storage failed: {e}")
    
    async def _learn_from_feedback(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Learn from task execution using real-time learning"""
        try:
            if self.realtime_learning is None:
                # Lazy load real-time learning
                from modules.realtime_learning import RealTimeLearningSystem
                if self.llm_engine:
                    self.realtime_learning = RealTimeLearningSystem(self.llm_engine)
            
            if self.realtime_learning and result.get("status") == "completed":
                # Store as learning example
                domain = task.get("context", {}).get("domain", "general")
                await self.realtime_learning.online_update(
                    feedback={
                        "type": "task_completion",
                        "input": task.get("query", ""),
                        "output": str(result.get("result", "")),
                        "quality_score": result.get("quality_score", 0.8),
                        "context": task.get("context", {})
                    },
                    domain=domain
                )
        except Exception as e:
            self.logger.debug(f"Real-time learning failed: {e}")

    async def _record_task_metrics(self, task_id: str, task: Dict[str, Any],
                                 result: Dict[str, Any], execution_time: float):
        """Record comprehensive task execution metrics"""
        metrics = {
            "task_id": task_id,
            "task_type": task.get("type", "unknown"),
            "execution_time": execution_time,
            "result_quality": self._assess_result_quality(result),
            "agent_utilization": len(result.get("agents_used", [])),
            "resource_usage": self._get_resource_usage(),
            "timestamp": datetime.now().isoformat()
        }

        await self.metrics.record_task_metrics(
            task_id=task_id,
            agent_id="orchestrator",
            task_type=task.get("type", "unknown"),
            success=result.get("status") == "completed",
            latency=execution_time,
            tokens_in=0,  # Would need to be tracked from LLM calls
            tokens_out=0,  # Would need to be tracked from LLM calls
            context_switches=0,  # Would need to be tracked
            attention_weight=0.0,  # Would need to be tracked
            metadata=metrics
        )

    def _assess_result_quality(self, result: Dict[str, Any]) -> float:
        """Assess result quality on 0-1 scale"""
        # Simple quality assessment based on result completeness
        if not result:
            return 0.0

        quality_score = 0.5  # Base score

        if "synthesis" in result:
            quality_score += 0.2
        if "agent_results" in result and len(result["agent_results"]) > 1:
            quality_score += 0.2
        if "optimization" in result:
            quality_score += 0.1

        return min(1.0, quality_score)

    def _get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage statistics"""
        # Placeholder for resource monitoring
        return {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "gpu_memory": 0.0
        }

    async def _should_evolve(self) -> bool:
        """Determine if system should trigger evolution"""
        if not self.evolution_manager:
            return False

        # Check performance metrics for evolution triggers
        recent_performance = await self.metrics.get_recent_performance()

        # Evolution triggers
        if recent_performance.get("avg_quality", 0) > 0.9:
            return True  # High performance - evolve to maintain edge
        if recent_performance.get("task_failure_rate", 0) > 0.1:
            return True  # High failure rate - evolve to improve

        return False

    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "health": self.system_health,
            "capabilities": self.capabilities,
            "active_tasks": len(self.active_tasks),
            "total_tasks_processed": len(self.task_history),
            "performance_metrics": self.performance_metrics,
            "subsystem_status": {
                "llm": self.llm_engine is not None,
                "vector_db": self.vector_db is not None,
                "agents": self.agent_manager is not None,
                "design": self.design_engine is not None,
                "consciousness": self.consciousness_engine is not None,
                "synthesis": self.synthesis_engine is not None,
                "evolution": self.evolution_manager is not None
            }
        }

    async def shutdown(self):
        """Gracefully shutdown all systems"""
        self.logger.info("🧠 Initiating Kalki Orchestrator shutdown...")

        # Shutdown subsystems in reverse order
        shutdown_tasks = []

        if self.evolution_manager:
            shutdown_tasks.append(self.evolution_manager.shutdown())
        if self.synthesis_engine:
            shutdown_tasks.append(self.synthesis_engine.shutdown())
        if self.consciousness_engine:
            shutdown_tasks.append(self.consciousness_engine.shutdown())
        if self.design_engine:
            shutdown_tasks.append(self.design_engine.shutdown())
        if self.agent_manager:
            shutdown_tasks.append(self.agent_manager.shutdown())
        if self.vector_db:
            shutdown_tasks.append(self.vector_db.close())
        if self.llm_engine:
            shutdown_tasks.append(self.llm_engine.cleanup())

        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)

        self.system_health = "shutdown"
        self.logger.info("✅ Kalki Orchestrator shutdown complete")
    
    async def execute_workflow(
        self,
        workflow: ProfessionalWorkflow,
        context: Dict[str, Any],
        team_orchestrator: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Execute a professional workflow.
        
        Args:
            workflow: Professional workflow definition
            context: Context for workflow execution
            team_orchestrator: Optional team orchestrator (will create if not provided)
        
        Returns:
            Workflow execution results
        """
        # Initialize workflow executor if needed
        if self.workflow_executor is None:
            if team_orchestrator is None:
                # Create team orchestrator
                from modules.professional_team_orchestrator import ProfessionalTeamOrchestrator
                team_orchestrator = ProfessionalTeamOrchestrator(self.agent_manager, self.llm_engine)
            
            from modules.professional_workflow import ProfessionalWorkflowExecutor
            self.workflow_executor = ProfessionalWorkflowExecutor(team_orchestrator, self.llm_engine)
        
        # Execute workflow
        return await self.workflow_executor.execute_workflow(workflow, context)
    
    async def generate_workflow(
        self,
        requirements: str,
        domain: str,
        context: Dict[str, Any],
        team_orchestrator: Optional[Any] = None
    ) -> ProfessionalWorkflow:
        """
        Generate a professional workflow from requirements using Llama 3.1 8B.
        
        Args:
            requirements: Natural language requirements
            domain: Domain context
            context: Additional context
            team_orchestrator: Optional team orchestrator
        
        Returns:
            Generated workflow
        """
        # Initialize workflow executor if needed
        if self.workflow_executor is None:
            if team_orchestrator is None:
                from modules.professional_team_orchestrator import ProfessionalTeamOrchestrator
                team_orchestrator = ProfessionalTeamOrchestrator(self.agent_manager, self.llm_engine)
            
            from modules.professional_workflow import ProfessionalWorkflowExecutor
            self.workflow_executor = ProfessionalWorkflowExecutor(team_orchestrator, self.llm_engine)
        
        # Generate workflow
        return await self.workflow_executor.generate_workflow_from_requirements(
            requirements, domain, context
        )


# Global orchestrator instance
_orchestrator_instance = None

def get_orchestrator() -> KalkiOrchestrator:
    """Get the global Kalki orchestrator instance"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = KalkiOrchestrator()
    return _orchestrator_instance

async def process_complex_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for processing complex tasks

    Args:
        task: Task specification

    Returns:
        Task result
    """
    orchestrator = get_orchestrator()
    return await orchestrator.process_task(task)
