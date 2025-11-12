"""
Base Domain Interface

All KALKI domain modules must implement this interface.
This ensures consistent behavior across construction, game dev, robotics, etc.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from enum import Enum
from pathlib import Path


class ProjectPhase(Enum):
    """Generic project phases (domains customize these)"""
    REQUIREMENTS = "requirements"
    DESIGN = "design"
    VALIDATION = "validation"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"


@dataclass
class ValidationResult:
    """Result of requirement validation"""
    valid: bool
    errors: List[str]
    warnings: List[str]
    suggestions: List[str]


@dataclass
class ComplexityScore:
    """Project complexity assessment"""
    overall_score: float  # 0-100
    time_estimate_days: int
    cost_estimate_usd: float
    risk_level: str  # "low", "medium", "high"
    factors: Dict[str, float]  # Breakdown of complexity factors


@dataclass
class KnowledgeExtractor:
    """Domain-specific knowledge extraction definition"""
    name: str
    description: str
    patterns: List[str]  # Regex patterns or keywords
    extractor_func: Any  # Function that performs extraction
    storage_db: str  # Database name for storing extracted knowledge


@dataclass
class DeliverableSpec:
    """Specification for a deliverable"""
    name: str
    description: str
    file_types: List[str]  # ["pdf", "dwg", "json", etc.]
    generator_func: Any
    required_knowledge: List[str]  # Knowledge types needed


class ProjectStateMachine(ABC):
    """Base class for domain-specific project workflows"""
    
    def __init__(self, project_id: str, description: str, domain: str):
        self.project_id = project_id
        self.description = description
        self.domain = domain
        self.current_phase = ProjectPhase.REQUIREMENTS
        self.phase_history = []
        self.metadata = {}
        self.issues = []
        self.milestones = {}
    
    @abstractmethod
    async def advance_phase(self, next_phase: ProjectPhase) -> bool:
        """Advance to next project phase"""
        pass
    
    @abstractmethod
    async def validate_phase_complete(self, phase: ProjectPhase) -> ValidationResult:
        """Check if current phase requirements are met"""
        pass
    
    @abstractmethod
    def get_available_phases(self) -> List[ProjectPhase]:
        """Get phases applicable to this domain"""
        pass
    
    @abstractmethod
    async def get_contextual_help(self, user_query: str) -> str:
        """Provide help relevant to current phase"""
        pass


class BaseDomain(ABC):
    """
    Base class for all KALKI domain modules.
    
    Each domain (construction, game dev, robotics, etc.) inherits from this
    and implements domain-specific functionality.
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.knowledge_base_path = Path(f"data/knowledge/{name}")
        self.knowledge_base_path.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def get_knowledge_extractors(self) -> List[KnowledgeExtractor]:
        """
        Return domain-specific knowledge extractors.
        
        Example for construction:
        - Span tables
        - Construction procedures
        - Inspection criteria
        
        Example for game dev:
        - Game mechanics
        - Design patterns
        - Engine APIs
        """
        pass
    
    @abstractmethod
    async def create_project(
        self,
        description: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> ProjectStateMachine:
        """
        Initialize a new project in this domain.
        
        Args:
            description: Natural language project description
            requirements: Optional structured requirements
        
        Returns:
            Domain-specific ProjectStateMachine instance
        """
        pass
    
    @abstractmethod
    def get_deliverable_types(self) -> List[DeliverableSpec]:
        """
        List all deliverables this domain can generate.
        
        Example for construction:
        - Construction drawings (DWG, PDF)
        - Bill of materials (XLSX, JSON)
        - Construction schedule (PDF, JSON)
        
        Example for game dev:
        - Game design document (PDF)
        - Unity project (ZIP)
        - Source code (files)
        - Assets (PNG, MP3, etc.)
        """
        pass
    
    @abstractmethod
    async def generate_deliverables(
        self,
        project: ProjectStateMachine,
        deliverable_types: List[str],
        output_dir: Path
    ) -> Dict[str, Path]:
        """
        Generate domain-specific deliverables.
        
        Args:
            project: Project state machine
            deliverable_types: List of deliverable names to generate
            output_dir: Where to save generated files
        
        Returns:
            Dict mapping deliverable name to output file path
        """
        pass
    
    @abstractmethod
    async def validate_requirements(
        self,
        requirements: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate project requirements for this domain.
        
        Example for construction:
        - Check building code compliance
        - Verify lot dimensions are valid
        - Ensure material specifications are complete
        
        Example for aerospace:
        - Check physics constraints (thrust > weight)
        - Verify regulatory compliance (FAA Part 103)
        - Validate power budget
        """
        pass
    
    @abstractmethod
    async def estimate_complexity(
        self,
        project: ProjectStateMachine
    ) -> ComplexityScore:
        """
        Estimate project complexity, time, and cost.
        
        Factors vary by domain:
        - Construction: sq footage, stories, custom features
        - Game dev: mechanics complexity, art assets, platform targets
        - Aerospace: aerodynamics, materials, regulatory compliance
        """
        pass
    
    async def load_knowledge(self, pdf_paths: List[Path]) -> Dict[str, int]:
        """
        Load domain-specific knowledge from PDFs.
        
        Uses domain-specific extractors to parse PDFs and populate
        domain knowledge databases.
        
        Args:
            pdf_paths: List of PDF files to ingest
        
        Returns:
            Dict of knowledge_type -> items_extracted
        """
        extractors = self.get_knowledge_extractors()
        results = {}
        
        for pdf_path in pdf_paths:
            for extractor in extractors:
                items = await extractor.extractor_func(pdf_path)
                results[extractor.name] = results.get(extractor.name, 0) + len(items)
        
        return results
    
    def get_knowledge_stats(self) -> Dict[str, int]:
        """Get count of knowledge items by type"""
        # Subclasses should override with actual database queries
        return {}


@dataclass
class DomainModule:
    """Container for loaded domain module"""
    domain: BaseDomain
    knowledge_stats: Dict[str, int]
    is_loaded: bool = True
    
    def __repr__(self):
        items = sum(self.knowledge_stats.values())
        return f"DomainModule({self.domain.name}, {items} knowledge items)"
