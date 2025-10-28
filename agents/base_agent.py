"""
Base Agent class for Kalki v2.3 Agent Framework
All agents inherit from this base class
"""
import uuid
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from datetime import datetime


class AgentCapability(Enum):
    """Defines agent capabilities for automatic discovery and routing"""
    # Phase 1-2: Foundation
    DOCUMENT_INGESTION = "document_ingestion"
    SEARCH = "search"
    VECTORIZATION = "vectorization"
    
    # Phase 3-5: Core Cognition
    PLANNING = "planning"
    REASONING = "reasoning"
    ORCHESTRATION = "orchestration"
    MEMORY = "memory"
    
    # Phase 6: Meta-Cognition
    META_REASONING = "meta_reasoning"
    FEEDBACK = "feedback"
    QUALITY_ASSESSMENT = "quality_assessment"
    
    # Phase 7: Knowledge Management
    CONFLICT_DETECTION = "conflict_detection"
    VALIDATION = "validation"
    LIFECYCLE_MANAGEMENT = "lifecycle_management"
    
    # Phase 8: Distributed Computing
    COMPUTE_SCALING = "compute_scaling"
    LOAD_BALANCING = "load_balancing"
    SELF_HEALING = "self_healing"
    
    # Phase 9: Simulation
    SIMULATION = "simulation"
    EXPERIMENTATION = "experimentation"
    SANDBOX = "sandbox"
    
    # Phase 10: Creativity
    CREATIVE_SYNTHESIS = "creative_synthesis"
    PATTERN_RECOGNITION = "pattern_recognition"
    IDEA_FUSION = "idea_fusion"
    
    # Phase 11: Evolution
    SELF_IMPROVEMENT = "self_improvement"
    OPTIMIZATION = "optimization"
    CURRICULUM_DESIGN = "curriculum_design"
    
    # Phase 12: Safety & Ethics
    ETHICS = "ethics"
    RISK_ASSESSMENT = "risk_assessment"
    SAFETY_VERIFICATION = "safety_verification"
    
    # Phase 13: Multi-Modal
    VISION = "vision"
    AUDIO = "audio"
    SENSOR_FUSION = "sensor_fusion"
    NEUROMORPHIC = "neuromorphic"
    
    # Phase 14: Quantum & Predictive
    QUANTUM_REASONING = "quantum_reasoning"
    PREDICTIVE_DISCOVERY = "predictive_discovery"
    TEMPORAL_ANALYSIS = "temporal_analysis"
    
    # Phase 15: Emotional Intelligence
    PERSONA = "persona"
    EMOTIONAL_STATE = "emotional_state"
    NEUROFEEDBACK = "neurofeedback"
    
    # Phase 16: Human-AI Interaction
    VOICE_ASSISTANT = "voice_assistant"
    INTUITION_PROBE = "intuition_probe"
    FLOW_STATE = "flow_state"
    
    # Phase 17: AR/VR/3D
    AR_INSIGHTS = "ar_insights"
    VR_SIMULATION = "vr_simulation"
    ASTROPHYSICAL_SIM = "astrophysical_simulation"
    
    # Phase 18: Cognitive Twin
    COGNITIVE_TWIN = "cognitive_twin"
    PREDICTION = "prediction"
    WISDOM_COMPRESSION = "wisdom_compression"
    
    # Phase 19: Autonomy
    AUTONOMOUS_INVENTION = "autonomous_invention"
    ROBOTICS = "robotics"
    IOT_INTEGRATION = "iot_integration"
    
    # Phase 20: Self-Evolution
    SELF_ARCHITECTING = "self_architecting"
    METAMORPHOSIS = "metamorphosis"
    GLOBAL_SAFETY = "global_safety"


class AgentStatus(Enum):
    """Agent lifecycle status"""
    INITIALIZING = "initializing"
    READY = "ready"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    TERMINATED = "terminated"


class BaseAgent(ABC):
    """
    Base class for all Kalki agents
    Provides standard interface for agent lifecycle, communication, and monitoring
    """
    
    def __init__(self, 
                 name: str,
                 capabilities: List[AgentCapability],
                 description: str = "",
                 dependencies: Optional[List[str]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize base agent
        
        Args:
            name: Unique agent name
            capabilities: List of capabilities this agent provides
            description: Human-readable description
            dependencies: List of agent names this agent depends on
            config: Agent-specific configuration
        """
        self.id = str(uuid.uuid4())
        self.name = name
        self.capabilities = capabilities
        self.description = description
        self.dependencies = dependencies or []
        self.config = config or {}
        self.status = AgentStatus.INITIALIZING
        self.created_at = datetime.utcnow()
        self.last_active = datetime.utcnow()
        self.task_count = 0
        self.error_count = 0
        self.logger = logging.getLogger(f"kalki.agent.{name}")
        
        # Event bus will be set by AgentManager
        self.event_bus = None
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the agent
        Returns True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task
        
        Args:
            task: Task dictionary with 'action', 'params', etc.
            
        Returns:
            Result dictionary with 'status', 'result', 'error', etc.
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the agent
        Returns True if successful, False otherwise
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check agent health
        Returns health status dictionary
        """
        return {
            "agent_id": self.id,
            "agent_name": self.name,
            "status": self.status.value,
            "task_count": self.task_count,
            "error_count": self.error_count,
            "uptime_seconds": (datetime.utcnow() - self.created_at).total_seconds(),
            "last_active": self.last_active.isoformat()
        }
    
    def set_event_bus(self, event_bus):
        """Set the event bus for inter-agent communication"""
        self.event_bus = event_bus
    
    async def emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to the event bus"""
        if self.event_bus:
            await self.event_bus.publish(event_type, {
                "source_agent": self.name,
                "timestamp": datetime.utcnow().isoformat(),
                **data
            })
    
    async def subscribe_to_event(self, event_type: str, callback):
        """Subscribe to events from the event bus"""
        if self.event_bus:
            await self.event_bus.subscribe(event_type, callback)
    
    def update_status(self, status: AgentStatus):
        """Update agent status"""
        self.status = status
        self.last_active = datetime.utcnow()
        self.logger.debug(f"Status changed to {status.value}")
    
    def increment_task_count(self):
        """Increment task counter"""
        self.task_count += 1
        self.last_active = datetime.utcnow()
    
    def increment_error_count(self):
        """Increment error counter"""
        self.error_count += 1
        self.last_active = datetime.utcnow()
    
    def get_info(self) -> Dict[str, Any]:
        """Get agent information"""
        return {
            "id": self.id,
            "name": self.name,
            "capabilities": [c.value for c in self.capabilities],
            "description": self.description,
            "dependencies": self.dependencies,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "task_count": self.task_count,
            "error_count": self.error_count
        }
