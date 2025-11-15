# ============================================================
# Kalki Control Hub
# ------------------------------------------------------------
# Unified intelligence orchestrator connecting all subsystems:
# - Consciousness Engine
# - Meta-Core System
# - Supreme Synthesis Engine
# - Design Brain
# - Hybrid Learning System
# - Self-Evolution Manager
# ============================================================

import asyncio
import json
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

from modules.utils.logging_config import get_logger
from modules.meta_core import get_meta_core, ReasoningDepth
from modules.supreme_synthesis_engine import get_supreme_synthesis_engine, SynthesisMode
from modules.hybrid_learning_system import get_hybrid_system
from modules.utils.config import CONFIG
from modules.domains.domain_registry import DomainRegistry

logger = get_logger("Kalki.SupremeControlHub")

@dataclass
class SupremeTaskResult:
    """Result from supreme task processing"""
    task: str
    mode: str
    consciousness_level: float
    reasoning_depth: str
    synthesis_result: Dict[str, Any]
    design_artifacts: Optional[Dict[str, Any]]
    knowledge_used: Dict[str, int]
    quality_score: float
    execution_time: float
    timestamp: str

class SupremeControlHub:
    """
    Supreme Control Hub: Unified Intelligence Orchestrator
    
    Connects all major KALKI subsystems into a coherent intelligence:
    1. Consciousness assessment → emotional + intention coherence
    2. Meta-Core reasoning depth selection
    3. Hybrid Learning knowledge retrieval (formulas, materials, codes)
    4. Supreme Synthesis multi-dimensional analysis
    5. Design Brain generative solution (if applicable)
    6. Self-Evolution feedback loop
    """
    
    def __init__(self):
        self.meta_core = get_meta_core()
        self.supreme_synthesis = get_supreme_synthesis_engine()
        self.hybrid_learning = get_hybrid_system()
        self.domain_registry = DomainRegistry()
        
        # Lazy-load heavy components
        self.consciousness = None
        self.design_brain = None
        self.self_evolution = None
        
        # Statistics
        self.execution_history = []
        
        logger.info(f"Supreme Control Hub initialized with {len(self.domain_registry.list_domains())} domains")
    
    async def _ensure_components_loaded(self):
        """Lazy-load heavy components on first use"""
        if self.consciousness is None:
            try:
                from modules.consciousness_engine import ConsciousnessEngine
                self.consciousness = ConsciousnessEngine()
                logger.info("Consciousness Engine loaded")
            except Exception as e:
                logger.warning(f"Consciousness Engine unavailable: {e}")
        
        if self.design_brain is None:
            try:
                from modules.design_brain import DesignBrain
                self.design_brain = DesignBrain()
                await self.design_brain.initialize()
                logger.info("Design Brain loaded")
            except Exception as e:
                logger.warning(f"Design Brain unavailable: {e}")
        
        if self.self_evolution is None:
            try:
                from modules.self_evolution_manager import SelfEvolutionManager
                self.self_evolution = SelfEvolutionManager()
                logger.info("Self-Evolution Manager loaded")
            except Exception as e:
                logger.warning(f"Self-Evolution Manager unavailable: {e}")
    
    async def process_supreme_task(
        self, 
        task: str, 
        mode: str = "supreme",
        context: Optional[Dict[str, Any]] = None
    ) -> SupremeTaskResult:
        """
        Process any task through the complete intelligence stack
        
        Args:
            task: The task description
            mode: Processing mode ("standard", "advanced", "supreme")
            context: Optional additional context
            
        Returns:
            SupremeTaskResult with complete processing details
        """
        start_time = datetime.now()
        
        # Ensure all components loaded
        await self._ensure_components_loaded()
        
        logger.info(f"Processing supreme task in {mode} mode: {task[:100]}...")
        
        # Initialize result tracking
        consciousness_level = 0.5  # Default if consciousness unavailable
        reasoning_depth = ReasoningDepth.STANDARD
        knowledge_used = {
            "formulas": 0,
            "materials": 0,
            "design_rules": 0,
            "code_requirements": 0,
            "semantic_chunks": 0
        }
        
        # Step 1: Consciousness-informed context (if available)
        if self.consciousness:
            try:
                consciousness_state = await self.consciousness.achieve_consciousness({
                    "task_processor": {
                        "query": task,
                        "context": context or {}
                    }
                })
                consciousness_level = consciousness_state.awareness_level
                logger.info(f"🧠 Consciousness level: {consciousness_level:.3f}")
            except Exception as e:
                logger.warning(f"Consciousness assessment failed: {e}")
        
        # Step 2: Meta-cognitive depth assessment
        reasoning_depth = self.meta_core.assess_task_complexity(task)
        self.meta_core.set_reasoning_depth(reasoning_depth)
        logger.info(f"📊 Reasoning depth: {reasoning_depth.value}")
        
        # Step 3: Knowledge retrieval with structured + semantic
        logger.info("📚 Retrieving knowledge from hybrid learning system...")
        
        # Query structured knowledge
        formulas = self.hybrid_learning.query_formulas()
        materials = self.hybrid_learning.query_materials()
        design_rules = self.hybrid_learning.query_design_rules()
        code_requirements = self.hybrid_learning.query_code_requirements()
        
        # Semantic search
        vector_context = []
        try:
            from modules.learning.vectordb import VectorDBManager
            vector_db = VectorDBManager()
            vector_context = vector_db.search_similar(task, top_k=10)
        except Exception as e:
            logger.warning(f"Vector search unavailable: {e}")
        
        knowledge_used = {
            "formulas": len(formulas),
            "materials": len(materials),
            "design_rules": len(design_rules),
            "code_requirements": len(code_requirements),
            "semantic_chunks": len(vector_context)
        }
        
        logger.info(f"Retrieved: {knowledge_used['formulas']} formulas, "
                   f"{knowledge_used['materials']} materials, "
                   f"{knowledge_used['design_rules']} design rules, "
                   f"{knowledge_used['code_requirements']} code requirements, "
                   f"{knowledge_used['semantic_chunks']} semantic chunks")
        
        # Step 4: Supreme synthesis with full context
        logger.info("⚡ Running supreme synthesis...")
        
        synthesis_mode = SynthesisMode.SUPREME if mode == "supreme" else \
                        SynthesisMode.ADVANCED if mode == "advanced" else \
                        SynthesisMode.STANDARD
        
        synthesis_result = await self.supreme_synthesis.synthesize(
            query=task,
            context={
                "consciousness_level": consciousness_level,
                "formulas": formulas[:20],  # Top 20 most relevant
                "materials": materials[:10],
                "design_rules": design_rules[:15],
                "codes": code_requirements[:10],
                "semantic_context": vector_context[:5]
            },
            mode=synthesis_mode
        )
        
        logger.info(f"✨ Synthesis quality score: {synthesis_result.quality_score:.3f}")
        
        # Step 5: Design generation (if applicable)
        design_artifacts = None
        if self._is_design_task(task) and self.design_brain:
            logger.info("🎨 Generating design with enhanced knowledge context...")
            try:
                design_blueprint = await self.design_brain.process_design_request(task)
                design_artifacts = {
                    "blueprint_id": design_blueprint.id,
                    "components": len(design_blueprint.components),
                    "system_requirements": design_blueprint.system_requirements,
                    "design_parameters": design_blueprint.design_parameters
                }
                logger.info(f"🎯 Design generated: {design_artifacts['components']} components")
            except Exception as e:
                logger.error(f"Design generation failed: {e}")
        
        # Step 6: Self-evolution feedback
        execution_time = (datetime.now() - start_time).total_seconds()
        
        if self.self_evolution:
            try:
                await self.self_evolution.record_execution({
                    "task": task,
                    "consciousness_level": consciousness_level,
                    "quality_score": synthesis_result.quality_score,
                    "reasoning_depth": reasoning_depth.value,
                    "execution_time": execution_time,
                    "knowledge_used": knowledge_used,
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                logger.warning(f"Self-evolution recording failed: {e}")
        
        # Create result
        result = SupremeTaskResult(
            task=task,
            mode=mode,
            consciousness_level=consciousness_level,
            reasoning_depth=reasoning_depth.value,
            synthesis_result=asdict(synthesis_result),
            design_artifacts=design_artifacts,
            knowledge_used=knowledge_used,
            quality_score=synthesis_result.quality_score,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat()
        )
        
        # Track history
        self.execution_history.append(result)
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-50:]
        
        logger.info(f"✅ Supreme task completed in {execution_time:.2f}s")
        
        return result
    
    def _is_design_task(self, task: str) -> bool:
        """Determine if task is a design request"""
        design_keywords = [
            "design", "create", "build", "make", "generate", "develop",
            "robot", "machine", "structure", "building", "vehicle", "system",
            "architecture", "mechanical", "engineering"
        ]
        task_lower = task.lower()
        return any(keyword in task_lower for keyword in design_keywords)
    
    async def process_domain_aware_query(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        project_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process query with automatic domain detection and routing
        
        Args:
            query: User query/task
            context: Optional context (project phase, requirements, etc.)
            project_id: Optional project ID for project-specific queries
            
        Returns:
            Dict with answer, domain info, and confidence
        """
        start_time = datetime.now()
        
        # Auto-detect relevant domains
        inferred_domain_names = await self.domain_registry.infer_domain(query)
        
        if not inferred_domain_names:
            logger.warning(f"No domains matched query: {query}")
            return {
                "success": False,
                "error": "Could not determine relevant domain for query",
                "query": query
            }
        
        logger.info(f"Inferred domains: {inferred_domain_names}")
        
        # Use first matching domain (or combine if multiple - TODO)
        domain_name = inferred_domain_names[0]
        
        # GET DOMAIN/COPILOT - Prefer copilot if available (enhanced processing)
        domain_or_copilot = self.domain_registry.get_domain(domain_name, prefer_copilot=True)
        
        # Check if we got a copilot
        copilot = None
        is_copilot = False
        if domain_or_copilot:
            # Check if it's a copilot (has copilot-specific methods)
            if hasattr(domain_or_copilot, 'start_new_project') or hasattr(domain_or_copilot, 'start_new_game_project'):
                copilot = domain_or_copilot
                is_copilot = True
                logger.info(f"🎯 Using {domain_name} copilot for enhanced processing")
        
        # Use copilot if available
        if copilot and is_copilot:
            try:
                # Game Dev Copilot has special methods
                if domain_name == "game_development":
                    # Check if this is a project creation request
                    query_lower = query.lower()
                    if any(keyword in query_lower for keyword in ["make", "create", "build", "develop", "game"]):
                        result = await copilot.start_new_game_project(query)
                        return {
                            "success": True,
                            "answer": result.get("message", "Game project processing started"),
                            "domain": {
                                "name": domain_name,
                                "description": "Game Development",
                                "copilot_used": True
                            },
                            "project_id": result.get("project_id"),
                            "session_id": result.get("session_id"),
                            "next_question": result.get("next_question"),
                            "confidence": 0.9,
                            "execution_time": (datetime.now() - start_time).total_seconds(),
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        # Check if this is answering a question (has session_id in context)
                        session_id = context.get("session_id") if context else None
                        if session_id:
                            # User is answering a question
                            result = await copilot.answer_question(session_id, query)
                        else:
                            # New query - start new project or answer question
                            if any(keyword in query.lower() for keyword in ["make", "create", "build", "develop"]):
                                result = await copilot.start_new_game_project(query)
                            else:
                                # General game dev query - use domain instead (prefer_copilot=False to get domain)
                                domain = self.domain_registry.get_domain(domain_name, prefer_copilot=False)
                                if domain:
                                    # Use domain's standard methods
                                    result = {"message": "Game development query processed", "status": "processed"}
                                else:
                                    result = {"message": "Query processed", "status": "processed"}
                        
                        return {
                            "success": True,
                            "answer": result.get("message", result.get("response", "Query processed")),
                            "domain": {
                                "name": domain_name,
                                "description": "Game Development",
                                "copilot_used": True
                            },
                            "next_question": result.get("next_question"),
                            "session_id": result.get("session_id"),
                            "project_id": result.get("project_id"),
                            "confidence": 0.85,
                            "execution_time": (datetime.now() - start_time).total_seconds(),
                            "timestamp": datetime.now().isoformat()
                        }
                
                # Construction Copilot
                elif domain_name == "construction":
                    # Check if this is a new project request
                    query_lower = query.lower()
                    if any(keyword in query_lower for keyword in ["new project", "start project", "create project", "build house", "build home"]):
                        result = await copilot.start_new_project(query)
                        return {
                            "success": True,
                            "answer": result.get("response", "Construction project started"),
                            "domain": {
                                "name": domain_name,
                                "description": "Construction",
                                "copilot_used": True
                            },
                            "project_id": result.get("project_id"),
                            "confidence": 0.9,
                            "execution_time": (datetime.now() - start_time).total_seconds(),
                            "timestamp": datetime.now().isoformat()
                        }
                    else:
                        # For general queries, use answer_with_automatic_diagrams for best experience
                        result = await copilot.answer_with_automatic_diagrams(
                            query=query,
                            context=context or {}
                        )
                        return {
                            "success": True,
                            "answer": result.get("answer", "Query processed"),
                            "domain": {
                                "name": domain_name,
                                "description": "Construction",
                                "copilot_used": True
                            },
                            "diagrams": result.get("diagrams", []),
                            "confidence": result.get("confidence", 0.8),
                            "execution_time": (datetime.now() - start_time).total_seconds(),
                            "timestamp": datetime.now().isoformat()
                        }
            except Exception as e:
                logger.warning(f"Copilot processing failed, falling back to domain: {e}")
                # Fall through to domain processing
        
        # Fallback to domain if no copilot or copilot failed
        # Use prefer_copilot=False to get domain, not copilot
        domain = self.domain_registry.get_domain(domain_name, prefer_copilot=False)
        if not domain:
            return {
                "success": False,
                "error": f"Domain {domain_name} not found",
                "query": query
            }
        
        # Get domain-specific knowledge statistics
        domain_knowledge = domain.get_knowledge_stats()
        
        # If project_id provided, load project and get phase-specific context
        project_context = None
        if project_id:
            try:
                from modules.domains.project_persistence import ProjectPersistence
                persistence = ProjectPersistence()
                project_data = persistence.load_project(project_id)
                if project_data:
                    project_context = {
                        "project_id": project_id,
                        "phase": project_data.get("current_phase"),
                        "description": project_data.get("description"),
                        "requirements": project_data.get("requirements", {})
                    }
                    logger.info(f"Loaded project context: phase={project_context['phase']}")
            except Exception as e:
                logger.warning(f"Could not load project context: {e}")
        
        # Query domain knowledge with enhanced context
        enhanced_context = {
            "original_query": query,
            "domain": domain.name,
            "domain_knowledge_stats": domain_knowledge,
            "project_context": project_context,
            "user_context": context or {}
        }
        
        # Use supreme synthesis with domain-specific knowledge
        synthesis_result = await self.supreme_synthesis.synthesize(
            query=query,
            context=enhanced_context,
            synthesis_mode=SynthesisMode.SUPREME
        )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Format synthesis result into readable answer
        answer_parts = []
        if synthesis_result.implementation_code:
            answer_parts.append(synthesis_result.implementation_code)
        if synthesis_result.conceptual_blueprint:
            answer_parts.append(f"\nConceptual Blueprint: {json.dumps(synthesis_result.conceptual_blueprint, indent=2)}")
        if synthesis_result.fabrication_specs:
            answer_parts.append(f"\nFabrication Specs: {json.dumps(synthesis_result.fabrication_specs, indent=2)}")
        
        answer = "\n".join(answer_parts) if answer_parts else "Synthesis completed successfully"
        
        return {
            "success": True,
            "query": query,
            "answer": answer,
            "domain": {
                "name": domain.name,
                "description": domain.description,
                "knowledge_stats": domain_knowledge
            },
            "project_context": project_context,
            "confidence": synthesis_result.quality_score,
            "synthesis_mode": synthesis_result.synthesis_mode.value,
            "execution_time": execution_time,
            "timestamp": datetime.now().isoformat()
        }
    
    async def generate_project_deliverable(
        self,
        project_id: str,
        deliverable_type: str,
        output_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Generate deliverable for a project
        
        Args:
            project_id: Project ID
            deliverable_type: Type of deliverable (drawings, BOM, schedule, etc.)
            output_dir: Output directory (default: output/deliverables)
            
        Returns:
            Dict with deliverable info and file paths
        """
        try:
            from modules.domains.project_persistence import ProjectPersistence
            
            # Load project
            persistence = ProjectPersistence()
            project_data = persistence.load_project(project_id)
            
            if not project_data:
                return {
                    "success": False,
                    "error": f"Project {project_id} not found"
                }
            
            # Get domain
            domain_name = project_data.get("domain")
            domain = self.domain_registry.get_domain(domain_name, prefer_copilot=False)
            
            if not domain:
                return {
                    "success": False,
                    "error": f"Domain {domain_name} not found"
                }
            
            # Reconstruct project state machine
            # This is domain-specific - for now use construction
            if domain_name == "construction":
                from modules.domains.construction_domain.construction_domain import ConstructionProjectStateMachine
                project = ConstructionProjectStateMachine.from_dict(project_data)
            else:
                return {
                    "success": False,
                    "error": f"Deliverable generation not yet implemented for domain: {domain_name}"
                }
            
            # Set output directory
            if output_dir is None:
                output_dir = Path("output/deliverables") / project_id
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate deliverable
            logger.info(f"Generating {deliverable_type} for project {project_id}")
            
            generated = await domain.generate_deliverables(
                project,
                [deliverable_type],
                output_dir
            )
            
            return {
                "success": True,
                "project_id": project_id,
                "deliverable_type": deliverable_type,
                "files": {k: str(v) for k, v in generated.items()},
                "domain": domain_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating deliverable: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_project_contextual_help(
        self,
        project_id: str,
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get phase-specific contextual help for a project
        
        Args:
            project_id: Project ID
            user_query: Optional specific question
            
        Returns:
            Dict with contextual help and recommendations
        """
        try:
            from modules.domains.project_persistence import ProjectPersistence
            
            # Load project
            persistence = ProjectPersistence()
            project_data = persistence.load_project(project_id)
            
            if not project_data:
                return {
                    "success": False,
                    "error": f"Project {project_id} not found"
                }
            
            # Get domain
            domain_name = project_data.get("domain")
            domain = self.domain_registry.get_domain(domain_name, prefer_copilot=False)
            
            if not domain:
                return {
                    "success": False,
                    "error": f"Domain {domain_name} not found"
                }
            
            # Reconstruct project state machine
            if domain_name == "construction":
                from modules.domains.construction_domain.construction_domain import ConstructionProjectStateMachine
                project = ConstructionProjectStateMachine.from_dict(project_data)
            else:
                return {
                    "success": False,
                    "error": f"Contextual help not yet implemented for domain: {domain_name}"
                }
            
            # Get contextual help
            help_text = await project.get_contextual_help(user_query or "what should I do next?")
            
            # Get phase progress
            progress = project.get_phase_progress()
            
            # Get budget status
            budget_status = project.get_budget_status()
            
            return {
                "success": True,
                "project_id": project_id,
                "domain": domain_name,
                "current_phase": project.current_phase.value if hasattr(project.current_phase, 'value') else str(project.current_phase),
                "progress": progress,
                "budget_status": budget_status,
                "help_text": help_text,
                "recommendations": self._generate_recommendations(project, progress, budget_status),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting contextual help: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_recommendations(
        self,
        project,
        progress: Dict[str, Any],
        budget_status: Dict[str, Any]
    ) -> List[str]:
        """Generate smart recommendations based on project state"""
        recommendations = []
        
        # Progress-based recommendations
        if progress['percent_complete'] < 50:
            recommendations.append("Focus on completing critical milestones before phase advancement")
        
        # Budget-based recommendations
        if budget_status['percent_spent'] > 80:
            recommendations.append("⚠️ Budget alert: Over 80% spent - review remaining work carefully")
        elif budget_status['percent_spent'] > 90:
            recommendations.append("🚨 Budget critical: Over 90% spent - immediate cost review needed")
        
        # Phase-specific recommendations
        phase_value = project.current_phase.value if hasattr(project.current_phase, 'value') else str(project.current_phase)
        
        if "foundation" in phase_value.lower():
            recommendations.append("Ensure footing inspection is scheduled before concrete pour")
        elif "framing" in phase_value.lower():
            recommendations.append("Order windows and doors now to avoid framing delays")
        elif "rough" in phase_value.lower() and "mep" in phase_value.lower():
            recommendations.append("Schedule all three MEP inspections (electrical, plumbing, mechanical) together")
        elif "insulation" in phase_value.lower():
            recommendations.append("Have insulation inspector verify R-values before covering")
        elif "final" in phase_value.lower():
            recommendations.append("Create punch list now to ensure timely final inspection")
        
        return recommendations
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {
                "total_executions": 0,
                "average_quality": 0.0,
                "average_consciousness": 0.0,
                "average_execution_time": 0.0
            }
        
        # Add domain statistics
        domain_stats = self.domain_registry.get_statistics()
        
        return {
            "total_executions": len(self.execution_history),
            "average_quality": sum(r.quality_score for r in self.execution_history) / len(self.execution_history),
            "average_consciousness": sum(r.consciousness_level for r in self.execution_history) / len(self.execution_history),
            "average_execution_time": sum(r.execution_time for r in self.execution_history) / len(self.execution_history),
            "total_knowledge_used": {
                "formulas": sum(r.knowledge_used["formulas"] for r in self.execution_history),
                "materials": sum(r.knowledge_used["materials"] for r in self.execution_history),
                "design_rules": sum(r.knowledge_used["design_rules"] for r in self.execution_history),
                "code_requirements": sum(r.knowledge_used["code_requirements"] for r in self.execution_history)
            },
            "reasoning_depth_distribution": self._get_depth_distribution(),
            "domain_statistics": domain_stats
        }
    
    def _get_depth_distribution(self) -> Dict[str, int]:
        """Get distribution of reasoning depths used"""
        distribution = {}
        for result in self.execution_history:
            depth = result.reasoning_depth
            distribution[depth] = distribution.get(depth, 0) + 1
        return distribution


# Global singleton instance
_supreme_hub = None

def get_supreme_control_hub() -> SupremeControlHub:
    """Get or create the global Supreme Control Hub instance"""
    global _supreme_hub
    if _supreme_hub is None:
        _supreme_hub = SupremeControlHub()
    return _supreme_hub
