"""
Domain Professional Integration Utility

Provides professional team orchestration, deliverable generation, 
cross-domain learning, workflow execution, and quality assurance
for all domains.

All domains can use this to integrate with Kalki's professional systems.
"""

import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class DomainProfessionalIntegration:
    """
    Integration helper for domains to use professional systems.
    
    Provides:
    - Professional Team Orchestration
    - Professional Deliverable Generation
    - Cross-Domain Learning
    - Professional Workflow Execution
    - Quality Assurance Framework
    """
    
    def __init__(
        self,
        domain_name: str,
        llm_engine=None,
        agent_manager=None,
        knowledge_graph=None,
        domain_registry=None,
        meta_learning=None
    ):
        """
        Initialize professional integration for a domain.
        
        Args:
            domain_name: Name of the domain (e.g., "game_dev", "robotics")
            llm_engine: LLMEngine instance (optional, will create if not provided)
            agent_manager: AgentManager instance (optional)
            knowledge_graph: VisualKnowledgeGraph instance (optional)
            domain_registry: DomainRegistry instance (optional)
            meta_learning: MetaLearningSystem instance (optional)
        """
        self.domain_name = domain_name
        
        # Lazy initialization - only import when needed
        self._llm_engine = llm_engine
        self._agent_manager = agent_manager
        self._knowledge_graph = knowledge_graph
        self._domain_registry = domain_registry
        self._meta_learning = meta_learning
        
        # Professional systems (initialized lazily)
        self._team_orchestrator = None
        self._deliverable_generator = None
        self._cross_learning = None
        self._workflow_executor = None
        self._quality_framework = None
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize all professional systems"""
        if self._initialized:
            return
        
        try:
            # Import required modules
            from modules.llm import LLMEngine
            from modules.agents.agent_manager import AgentManager
            from modules.agents.event_bus import EventBus
            from modules.visual_knowledge_graph import VisualKnowledgeGraph
            from modules.domains.domain_registry import DomainRegistry
            from modules.meta_learning_system import MetaLearningSystem
            from modules.professional_team_orchestrator import ProfessionalTeamOrchestrator
            from modules.professional_deliverable_generator import ProfessionalDeliverableGenerator
            from modules.cross_domain_learning import CrossDomainLearning
            from modules.professional_workflow import ProfessionalWorkflowExecutor
            from modules.quality_assurance_framework import QualityAssuranceFramework
            
            # Initialize dependencies if not provided
            if self._llm_engine is None:
                self._llm_engine = LLMEngine()
            
            if self._agent_manager is None:
                event_bus = EventBus()
                self._agent_manager = AgentManager(event_bus)
            
            if self._knowledge_graph is None:
                self._knowledge_graph = VisualKnowledgeGraph()
            
            if self._domain_registry is None:
                self._domain_registry = DomainRegistry()
            
            if self._meta_learning is None:
                self._meta_learning = MetaLearningSystem()
            
            # Initialize professional systems
            self._team_orchestrator = ProfessionalTeamOrchestrator(
                self._agent_manager,
                self._llm_engine
            )
            
            self._deliverable_generator = ProfessionalDeliverableGenerator(
                self._llm_engine,
                self._knowledge_graph
            )
            
            self._cross_learning = CrossDomainLearning(
                self._domain_registry,
                self._meta_learning,
                self._llm_engine
            )
            
            self._workflow_executor = ProfessionalWorkflowExecutor(
                self._team_orchestrator,
                self._llm_engine
            )
            
            self._quality_framework = QualityAssuranceFramework(
                self._llm_engine
            )
            
            self._initialized = True
            logger.info(f"✅ Professional integration initialized for {self.domain_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize professional integration for {self.domain_name}: {e}")
            raise
    
    @property
    def team_orchestrator(self):
        """Get professional team orchestrator"""
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        return self._team_orchestrator
    
    @property
    def deliverable_generator(self):
        """Get professional deliverable generator"""
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        return self._deliverable_generator
    
    @property
    def cross_learning(self):
        """Get cross-domain learning system"""
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        return self._cross_learning
    
    @property
    def workflow_executor(self):
        """Get professional workflow executor"""
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        return self._workflow_executor
    
    @property
    def quality_framework(self):
        """Get quality assurance framework"""
        if not self._initialized:
            raise RuntimeError("Must call initialize() first")
        return self._quality_framework
    
    async def initialize_roles(self, roles: list):
        """
        Initialize professional roles for this domain.
        
        Args:
            roles: List of (role, capability) tuples
        """
        if not self._initialized:
            await self.initialize()
        
        from modules.professional_team_orchestrator import ProfessionalRole
        from modules.agents.base_agent import AgentCapability
        
        for role_name, capability_name in roles:
            try:
                role = ProfessionalRole[role_name.upper()]
                capability = AgentCapability[capability_name.upper()]
                
                await self._team_orchestrator.assign_role(
                    role=role,
                    agent_capability=capability
                )
                logger.info(f"✅ Assigned {role_name} role for {self.domain_name}")
            except (KeyError, AttributeError) as e:
                logger.warning(f"⚠️ Could not assign role {role_name}: {e}")


