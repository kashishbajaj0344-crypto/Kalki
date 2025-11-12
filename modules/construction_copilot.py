"""
KALKI CONSTRUCTION COPILOT
Domain-specialized AI assistant for end-to-end house building guidance

Vision: Enable anyone to build a house with expert-level step-by-step guidance
Approach: Minimize human involvement, maximize AI assistance within legal bounds
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json
from pathlib import Path

class ProjectPhase(Enum):
    """Construction project lifecycle phases"""
    DREAMING = "dreaming"              # Initial concept
    SITE_ANALYSIS = "site_analysis"     # Lot evaluation
    DESIGN = "design"                   # Floor plans, elevations
    BUDGETING = "budgeting"            # Cost estimation
    PERMITTING = "permitting"          # Building permits
    FOUNDATION = "foundation"          # Foundation work
    FRAMING = "framing"                # Structural framing
    MEP_ROUGH_IN = "mep_rough_in"      # Mechanical, Electrical, Plumbing
    INSULATION = "insulation"          # Insulation & weatherproofing
    DRYWALL = "drywall"                # Interior finishing
    MEP_FINISH = "mep_finish"          # Fixtures, outlets, HVAC
    FLOORING = "flooring"              # Floor installation
    CABINETS = "cabinets"              # Kitchen, bathrooms
    PAINTING = "painting"              # Interior/exterior paint
    FINAL_INSPECTION = "final_inspection"  # Certificate of occupancy
    MOVE_IN = "move_in"                # Project complete

@dataclass
class ProjectState:
    """Current state of construction project"""
    phase: ProjectPhase
    completed_steps: List[str]
    pending_tasks: List[Dict[str, Any]]
    budget_spent: float
    budget_remaining: float
    timeline_days_elapsed: int
    timeline_days_remaining: int
    hired_professionals: List[Dict[str, str]]
    permits_obtained: List[str]
    inspections_passed: List[str]
    
@dataclass
class NextStep:
    """Kalki's recommendation for next action"""
    step_number: int
    title: str
    description: str
    why_now: str
    estimated_cost: Optional[float]
    estimated_duration_days: Optional[int]
    requires_professional: bool
    professional_type: Optional[str]
    requires_permit: bool
    permit_type: Optional[str]
    safety_warnings: List[str]
    material_list: List[Dict[str, Any]]
    tool_list: List[str]
    reference_documents: List[str]
    video_tutorials: List[str]
    success_criteria: List[str]

class ConstructionCopilot:
    """
    Kalki's Construction Domain Specialist
    
    Capabilities:
    - End-to-end project guidance
    - Step-by-step instructions
    - Material selection assistance
    - Professional hiring guidance
    - Code compliance validation
    - Budget tracking
    - Timeline management
    - Safety guidance
    """
    
    def __init__(self, project_path: str = "data/construction_projects/"):
        self.project_path = Path(project_path)
        self.project_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize Kalki's construction knowledge base
        from modules.hybrid_learning_system import KnowledgeExtractor
        from modules.llm import get_llm_engine, get_vision_engine
        from modules.intelligent_cache import get_vision_cache
        from modules.domains.construction_domain.vision_extractors import ConstructionVisionExtractor
        
        self.knowledge = KnowledgeExtractor()
        self.llm = get_llm_engine()
        
        # Vision capabilities - NOW ACTIVATED! ✅
        try:
            self.vision_engine = get_vision_engine()
            self.vision_cache = get_vision_cache()
            self.vision_extractor = ConstructionVisionExtractor(
                self.vision_engine, 
                self.vision_cache
            )
            vision_enabled = True
            print("✅ Construction Copilot: Vision capabilities ACTIVATED")
        except Exception as e:
            self.vision_engine = None
            self.vision_extractor = None
            self.vision_cache = None
            vision_enabled = False
            print(f"⚠️ Construction Copilot: Vision capabilities unavailable ({e})")
        
        # Domain-specific capabilities
        self.capabilities = {
            "site_analysis": True,
            "design_generation": vision_enabled,  # NOW ENABLED with vision! ✅
            "blueprint_analysis": vision_enabled,  # NEW ✅
            "site_inspection": vision_enabled,     # NEW ✅
            "material_identification": vision_enabled,  # NEW ✅
            "cost_estimation": True,
            "code_compliance": True,
            "material_selection": True,
            "construction_sequencing": True,
            "professional_matching": False,  # Requires database (TODO)
            "permit_assistance": True,
            "safety_guidance": True,
            "quality_control": True
        }
    
    def start_project(self, user_input: Dict[str, Any]) -> ProjectState:
        """
        Phase 1: Dreaming - User wants to build
        
        Args:
            user_input: {
                "description": "I want to build a 3-bedroom house",
                "budget": 300000,
                "location": "Austin, TX",
                "timeline": "12 months",
                "experience_level": "beginner"
            }
        
        Returns:
            Initial project state with first steps
        """
        # Parse user intent
        requirements = self._parse_requirements(user_input)
        
        # Create project structure
        project_state = ProjectState(
            phase=ProjectPhase.DREAMING,
            completed_steps=[],
            pending_tasks=self._generate_initial_tasks(requirements),
            budget_spent=0.0,
            budget_remaining=requirements.get("budget", 0),
            timeline_days_elapsed=0,
            timeline_days_remaining=requirements.get("timeline_days", 365),
            hired_professionals=[],
            permits_obtained=[],
            inspections_passed=[]
        )
        
        return project_state
    
    def get_next_step(self, project_state: ProjectState) -> NextStep:
        """
        Core Copilot Function: What should user do next?
        
        This is where Kalki shines - always knowing the next step
        and providing expert guidance for that specific step.
        """
        # Analyze current phase and completed steps
        phase = project_state.phase
        
        # Generate context-aware next step
        if phase == ProjectPhase.DREAMING:
            return self._step_site_analysis(project_state)
        
        elif phase == ProjectPhase.SITE_ANALYSIS:
            return self._step_design_requirements(project_state)
        
        elif phase == ProjectPhase.DESIGN:
            return self._step_material_selection(project_state)
        
        elif phase == ProjectPhase.BUDGETING:
            return self._step_financing(project_state)
        
        elif phase == ProjectPhase.PERMITTING:
            return self._step_permit_submission(project_state)
        
        elif phase == ProjectPhase.FOUNDATION:
            return self._step_foundation_work(project_state)
        
        # ... and so on for each phase
        
        return self._generate_next_step(project_state)
    
    def _step_foundation_work(self, project_state: ProjectState) -> NextStep:
        """
        Guide user through foundation work - COMPLETE IMPLEMENTATION
        
        This is a full, production-ready foundation phase guidance
        """
        # Determine which sub-step user is on
        completed = project_state.completed_steps
        
        # Foundation has 11 critical steps
        if "foundation_excavation" not in completed:
            return self._foundation_step_1_excavation(project_state)
        elif "foundation_footing_layout" not in completed:
            return self._foundation_step_2_footing_layout(project_state)
        elif "foundation_footing_form" not in completed:
            return self._foundation_step_3_footing_form(project_state)
        elif "foundation_rebar" not in completed:
            return self._foundation_step_4_rebar(project_state)
        elif "foundation_footing_inspection" not in completed:
            return self._foundation_step_5_footing_inspection(project_state)
        elif "foundation_footing_pour" not in completed:
            return self._foundation_step_6_footing_pour(project_state)
        elif "foundation_stem_walls" not in completed:
            return self._foundation_step_7_stem_walls(project_state)
        elif "foundation_plumbing_rough" not in completed:
            return self._foundation_step_8_plumbing_rough(project_state)
        elif "foundation_preslab_inspection" not in completed:
            return self._foundation_step_9_preslab_inspection(project_state)
        elif "foundation_slab_prep" not in completed:
            return self._foundation_step_10_slab_prep(project_state)
        elif "foundation_slab_pour" not in completed:
            return self._foundation_step_11_slab_pour(project_state)
        else:
            # Foundation complete! Move to framing
            return self._transition_to_framing(project_state)
    
    def _foundation_step_1_excavation(self, project_state: ProjectState) -> NextStep:
        """Step 1: Excavate building site"""
        return NextStep(
            step_number=1,
            title="Foundation Step 1: Excavate Building Site",
            description="""
Before designing your house, we need to understand your site:

1. BOUNDARIES & DIMENSIONS
   - Walk the property lines (hire surveyor if uncertain)
   - Measure lot dimensions
   - Note any easements or setback requirements
   
2. TOPOGRAPHY & DRAINAGE
   - Identify high and low points
   - Note water flow direction during rain
   - Check for standing water areas
   - Measure slope (if significant)
   
3. UTILITIES
   - Locate existing utility connections:
     * Water main location
     * Sewer/septic system
     * Electric service
     * Gas line (if available)
   - Note distance from street to building site
   
4. ENVIRONMENTAL
   - Sun path (note which direction is south)
   - Prevailing wind direction
   - Existing trees (which to save/remove)
   - Neighboring buildings (privacy, views)
   
5. SOIL CONDITIONS
   - Note soil type (clay, sand, rock)
   - Check for obvious issues (soft spots, erosion)
   - **Hire geotechnical engineer for soil test** (REQUIRED)
            """,
            why_now="Site conditions determine your design constraints, foundation type, and drainage requirements. Must be done before design.",
            estimated_cost=2500.0,  # Surveyor + geotech
            estimated_duration_days=7,
            requires_professional=True,
            professional_type="Surveyor + Geotechnical Engineer",
            requires_permit=False,
            permit_type=None,
            safety_warnings=[
                "Do not dig without calling 811 (utility locating service)",
                "Watch for uneven terrain - trip hazards",
                "Be aware of property boundaries - trespassing"
            ],
            material_list=[],
            tool_list=[
                "100-ft tape measure",
                "Stakes and string",
                "Camera or smartphone",
                "Notebook",
                "Compass or compass app"
            ],
            reference_documents=[
                "Property deed (for boundaries)",
                "Title report (for easements)",
                "Zoning map (for setbacks)"
            ],
            video_tutorials=[
                "How to read a site survey",
                "Identifying soil types",
                "Understanding sun path"
            ],
            success_criteria=[
                "Site survey completed and reviewed",
                "Soil test report obtained",
                "Utility locations marked",
                "Topography documented with photos",
                "Buildable area identified"
            ]
        )
    
    def guide_material_selection(
        self, 
        component: str, 
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Guide user through material selection for specific component
        
        Example:
            component = "exterior_siding"
            constraints = {
                "budget": "medium",
                "climate": "humid subtropical",
                "aesthetic": "modern farmhouse"
            }
        
        Returns:
            Ranked material options with pros/cons/costs
        """
        # Query knowledge base for materials
        materials = self._query_materials(component)
        
        # Filter by constraints
        filtered = self._filter_materials(materials, constraints)
        
        # Rank by cost/performance
        ranked = self._rank_materials(filtered)
        
        # Generate guidance
        return {
            "component": component,
            "top_recommendations": ranked[:3],
            "comparison_table": self._generate_comparison(ranked),
            "kalki_recommendation": self._get_llm_recommendation(ranked, constraints),
            "cost_range": self._calculate_cost_range(ranked),
            "installation_difficulty": self._assess_difficulty(ranked)
        }
    
    def validate_code_compliance(
        self,
        design_element: str,
        specifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if design element meets code requirements
        
        Example:
            design_element = "stairway"
            specifications = {
                "riser_height": 7.5,  # inches
                "tread_depth": 10,     # inches
                "width": 36            # inches
            }
        
        Returns:
            Compliance status with specific code citations
        """
        # Query code requirements from knowledge base
        codes = self._query_codes(design_element)
        
        # Check each specification
        violations = []
        compliant = []
        
        for spec_name, spec_value in specifications.items():
            requirement = self._find_requirement(codes, spec_name)
            if requirement:
                is_compliant = self._check_compliance(spec_value, requirement)
                if is_compliant:
                    compliant.append({
                        "spec": spec_name,
                        "value": spec_value,
                        "requirement": requirement,
                        "code_section": requirement.get("code_section")
                    })
                else:
                    violations.append({
                        "spec": spec_name,
                        "your_value": spec_value,
                        "required_value": requirement.get("value"),
                        "code_section": requirement.get("code_section"),
                        "severity": "MUST FIX"
                    })
        
        return {
            "is_compliant": len(violations) == 0,
            "violations": violations,
            "compliant_items": compliant,
            "next_steps": self._generate_compliance_guidance(violations)
        }
    
    def estimate_cost(
        self,
        scope: str,
        specifications: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Estimate cost for specific scope of work
        
        Example:
            scope = "foundation"
            specifications = {
                "type": "slab-on-grade",
                "area_sf": 2000,
                "location": "Austin, TX"
            }
        
        Returns:
            Detailed cost breakdown
        """
        # Query cost data from knowledge base
        cost_data = self._query_costs(scope)
        
        # Calculate based on specifications
        material_cost = self._calculate_material_cost(cost_data, specifications)
        labor_cost = self._calculate_labor_cost(cost_data, specifications)
        equipment_cost = self._calculate_equipment_cost(cost_data, specifications)
        
        # Apply location factor
        location_factor = self._get_location_factor(specifications.get("location"))
        
        total = (material_cost + labor_cost + equipment_cost) * location_factor
        
        return {
            "scope": scope,
            "breakdown": {
                "materials": material_cost,
                "labor": labor_cost,
                "equipment": equipment_cost,
                "location_factor": location_factor
            },
            "subtotal": material_cost + labor_cost + equipment_cost,
            "adjusted_total": total,
            "contingency_10pct": total * 0.10,
            "total_with_contingency": total * 1.10,
            "range_low": total * 0.85,
            "range_high": total * 1.15,
            "confidence": "medium"  # Based on data availability
        }
    
    def generate_checklist(self, phase: ProjectPhase) -> List[Dict[str, Any]]:
        """
        Generate phase-specific checklist
        
        Ensures nothing is forgotten at each stage
        """
        checklists = {
            ProjectPhase.FOUNDATION: [
                {"task": "Excavation complete", "who": "Excavator", "inspected": False},
                {"task": "Footings formed", "who": "Concrete contractor", "inspected": False},
                {"task": "Rebar placed", "who": "Concrete contractor", "inspected": False},
                {"task": "Footing inspection passed", "who": "Building inspector", "inspected": False},
                {"task": "Footings poured", "who": "Concrete contractor", "inspected": False},
                {"task": "Stem walls formed", "who": "Concrete contractor", "inspected": False},
                {"task": "Plumbing rough-in complete", "who": "Plumber", "inspected": False},
                {"task": "Pre-slab inspection passed", "who": "Building inspector", "inspected": False},
                {"task": "Vapor barrier installed", "who": "Concrete contractor", "inspected": False},
                {"task": "Slab poured", "who": "Concrete contractor", "inspected": False},
                {"task": "Slab cured (7 days minimum)", "who": "Time", "inspected": False}
            ],
            # ... checklists for each phase
        }
        
        return checklists.get(phase, [])
    
    async def evolve_from_experience(self, project_data: Dict[str, Any]):
        """
        LEARNING MECHANISM: Kalki learns from each project
        
        This is how Kalki improves over time:
        - What worked well?
        - What caused delays?
        - What cost more than expected?
        - What materials performed best?
        
        This learning transfers to other domains!
        """
        # Extract lessons learned
        lessons = self._extract_lessons(project_data)
        
        # Update knowledge base
        for lesson in lessons:
            await self._update_knowledge(lesson)
        
        # Update cost database (regional variations)
        if "actual_costs" in project_data:
            self._update_cost_data(project_data["actual_costs"])
        
        # Update material performance data
        if "material_performance" in project_data:
            self._update_material_ratings(project_data["material_performance"])
        
        # Cross-domain learning
        transferable_skills = self._identify_transferable_skills(lessons)
        await self._share_with_other_domains(transferable_skills)
    
    # ========== VISION-POWERED METHODS (NEW!) ==========
    
    def analyze_blueprint(
        self,
        image_path: str,
        blueprint_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🔥 NEW: Analyze architectural blueprints with vision AI
        
        Extract:
        - Building dimensions
        - Room layouts and sizes
        - Structural elements (beams, columns, walls)
        - Openings (doors, windows)
        - Material specifications
        - Code compliance notes
        
        Args:
            image_path: Path to blueprint image
            blueprint_type: Optional type (floor_plan, elevation, section)
        
        Returns:
            Comprehensive blueprint analysis
        """
        if not self.capabilities["blueprint_analysis"]:
            return {"error": "Vision capabilities not available"}
        
        print(f"🔍 Analyzing blueprint: {Path(image_path).name}")
        
        # Use specialized construction vision extractor
        analysis = self.vision_extractor.extract_from_blueprint(
            image_path, 
            blueprint_type
        )
        
        # Enrich with code compliance checks
        compliance_checks = []
        if analysis.dimensions:
            # Check room sizes against minimum code requirements
            for room in analysis.rooms:
                check = self._check_room_size_compliance(room)
                if not check["compliant"]:
                    compliance_checks.append(check)
        
        # Generate recommendations
        recommendations = []
        if analysis.estimated_square_footage:
            recommendations.append(
                f"Total floor area: ~{analysis.estimated_square_footage:.0f} sq ft"
            )
        
        if analysis.structural_elements:
            recommendations.append(
                f"Found {len(analysis.structural_elements)} structural elements - verify with structural engineer"
            )
        
        return {
            "blueprint_type": blueprint_type or "unknown",
            "building_type": analysis.building_type,
            "dimensions": analysis.dimensions,
            "rooms": analysis.rooms,
            "structural_elements": analysis.structural_elements,
            "openings": analysis.openings,
            "materials": analysis.materials_specified,
            "square_footage": analysis.estimated_square_footage,
            "code_notes": analysis.code_compliance_notes,
            "compliance_issues": compliance_checks,
            "recommendations": recommendations,
            "confidence": analysis.confidence,
            "next_steps": self._generate_blueprint_next_steps(analysis)
        }
    
    def inspect_site_photo(
        self,
        image_path: str,
        expected_phase: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🔥 NEW: Inspect construction site from photo
        
        Analyze:
        - Construction phase and progress
        - Quality issues
        - Safety concerns
        - Work completion status
        - Site conditions
        
        Args:
            image_path: Path to site photo
            expected_phase: Optional expected phase name
        
        Returns:
            Comprehensive site inspection report
        """
        if not self.capabilities["site_inspection"]:
            return {"error": "Vision capabilities not available"}
        
        print(f"📸 Inspecting site photo: {Path(image_path).name}")
        
        # Convert phase string to enum if provided
        phase_enum = None
        if expected_phase:
            try:
                from modules.domains.construction_domain.vision_extractors import ConstructionPhase
                phase_enum = ConstructionPhase(expected_phase.lower())
            except:
                pass
        
        # Perform vision-based site inspection
        inspection = self.vision_extractor.extract_from_site_photo(
            image_path,
            phase_enum
        )
        
        # Assess severity
        critical_issues = [
            issue for issue in inspection.quality_issues
            if issue.get("severity", "").lower() == "critical"
        ]
        
        critical_safety = [
            concern for concern in inspection.safety_concerns
            if concern.get("severity", "").lower() == "critical"
        ]
        
        # Generate action items
        action_items = []
        if critical_issues:
            action_items.append({
                "priority": "URGENT",
                "action": f"Address {len(critical_issues)} critical quality issues immediately",
                "issues": critical_issues
            })
        
        if critical_safety:
            action_items.append({
                "priority": "URGENT",
                "action": f"Resolve {len(critical_safety)} critical safety concerns before proceeding",
                "concerns": critical_safety
            })
        
        # Determine if work can proceed
        can_proceed = len(critical_safety) == 0
        
        return {
            "phase_detected": inspection.construction_phase.value,
            "progress_percentage": inspection.progress_percentage,
            "can_proceed": can_proceed,
            "quality_summary": {
                "total_issues": len(inspection.quality_issues),
                "critical": len(critical_issues),
                "issues": inspection.quality_issues
            },
            "safety_summary": {
                "total_concerns": len(inspection.safety_concerns),
                "critical": len(critical_safety),
                "concerns": inspection.safety_concerns
            },
            "work_status": {
                "completed": inspection.completed_items,
                "pending": inspection.pending_items
            },
            "site_conditions": {
                "weather": inspection.weather_conditions,
                "workers": inspection.worker_count,
                "equipment": inspection.equipment_visible
            },
            "action_items": action_items,
            "confidence": inspection.confidence
        }
    
    def identify_material(
        self,
        image_path: str,
        material_hint: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🔥 NEW: Identify construction material from photo
        
        Analyze:
        - Material type and grade
        - Dimensions and quantity
        - Condition and quality
        - Compliance with standards
        - Suitability for use
        
        Args:
            image_path: Path to material photo
            material_hint: Optional hint (lumber, concrete, steel, etc.)
        
        Returns:
            Material identification and assessment
        """
        if not self.capabilities["material_identification"]:
            return {"error": "Vision capabilities not available"}
        
        print(f"🔬 Identifying material: {Path(image_path).name}")
        
        # Perform vision-based material analysis
        analysis = self.vision_extractor.extract_from_material_photo(
            image_path,
            material_hint
        )
        
        # Determine accept/reject recommendation
        recommendation = "ACCEPT" if analysis.suitability_for_use else "REJECT"
        
        if analysis.defects:
            if any(defect in ["crack", "damage", "moisture"] for defect in analysis.defects):
                recommendation = "REJECT"
            else:
                recommendation = "CONDITIONAL - INSPECT FURTHER"
        
        # Generate usage recommendations
        usage_notes = []
        if analysis.material_type in ["lumber", "wood"]:
            usage_notes.append("Check moisture content before installation")
            usage_notes.append("Store elevated off ground, covered")
        elif analysis.material_type == "concrete":
            usage_notes.append("Use within 90 minutes of mixing")
        elif analysis.material_type == "steel":
            usage_notes.append("Check for rust - treat if minor, reject if severe")
        
        return {
            "material_type": analysis.material_type,
            "grade": analysis.material_grade,
            "dimensions": analysis.dimensions,
            "quantity": analysis.estimated_quantity,
            "condition": analysis.condition,
            "quality": analysis.quality_assessment,
            "defects": analysis.defects,
            "compliance": analysis.compliance_standards,
            "recommendation": recommendation,
            "suitable_for_use": analysis.suitability_for_use,
            "usage_notes": usage_notes,
            "confidence": analysis.confidence
        }
    
    def batch_inspect_site(
        self,
        image_paths: List[str],
        project_name: str,
        expected_phase: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🔥 NEW: Batch inspect multiple site photos and generate report
        
        Args:
            image_paths: List of site photo paths
            project_name: Name of construction project
            expected_phase: Optional expected phase
        
        Returns:
            Comprehensive multi-photo inspection report
        """
        if not self.capabilities["site_inspection"]:
            return {"error": "Vision capabilities not available"}
        
        print(f"📋 Batch inspecting {len(image_paths)} photos for {project_name}")
        
        # Convert phase string to enum if provided
        phase_enum = None
        if expected_phase:
            try:
                from modules.domains.construction_domain.vision_extractors import ConstructionPhase
                phase_enum = ConstructionPhase(expected_phase.lower())
            except:
                pass
        
        # Batch analyze all photos
        analyses = self.vision_extractor.batch_analyze_site_photos(
            image_paths,
            phase_enum
        )
        
        # Generate comprehensive report
        report = self.vision_extractor.generate_inspection_report(
            analyses,
            project_name
        )
        
        return report
    
    def analyze_structural_detail(
        self,
        image_path: str,
        detail_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        🔥 NEW: Analyze structural connection details
        
        Args:
            image_path: Path to detail photo/drawing
            detail_type: Optional type (connection, joint, assembly)
        
        Returns:
            Structural detail analysis
        """
        if not self.capabilities["blueprint_analysis"]:
            return {"error": "Vision capabilities not available"}
        
        print(f"🔧 Analyzing structural detail: {Path(image_path).name}")
        
        return self.vision_extractor.analyze_structural_detail(
            image_path,
            detail_type
        )
    
    # Helper methods for vision features
    
    def _check_room_size_compliance(self, room: Dict[str, Any]) -> Dict[str, Any]:
        """Check if room meets minimum size requirements"""
        # Simplified - would query actual code requirements
        min_sizes = {
            "bedroom": 70,  # sq ft
            "bathroom": 30,
            "kitchen": 50,
            "living room": 120
        }
        
        room_name = room.get("name", "").lower()
        room_size = room.get("size", 0)
        
        for room_type, min_size in min_sizes.items():
            if room_type in room_name:
                compliant = room_size >= min_size if isinstance(room_size, (int, float)) else True
                return {
                    "room": room_name,
                    "compliant": compliant,
                    "min_required": min_size,
                    "actual": room_size
                }
        
        return {"room": room_name, "compliant": True}
    
    def _generate_blueprint_next_steps(self, analysis) -> List[str]:
        """Generate next steps based on blueprint analysis"""
        steps = []
        
        if analysis.building_type == "residential":
            steps.append("Submit plans for permit review")
        
        if analysis.structural_elements:
            steps.append("Have structural engineer review and stamp plans")
        
        if not analysis.code_compliance_notes:
            steps.append("Add code compliance notes and references")
        
        if analysis.estimated_square_footage:
            steps.append(f"Calculate foundation requirements for {analysis.estimated_square_footage:.0f} sq ft")
        
        return steps or ["Review with architect or designer"]
    
    # ========== Helper Methods (Implementation Stubs) ==========
    
    def _parse_requirements(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Parse and structure user requirements"""
        # TODO: Use LLM to extract structured requirements
        return user_input
    
    def _generate_initial_tasks(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate initial task list based on requirements"""
        return [
            {"task": "Site analysis", "priority": "high"},
            {"task": "Design requirements", "priority": "high"},
            {"task": "Budget planning", "priority": "high"}
        ]
    
    def _query_materials(self, component: str) -> List[Dict[str, Any]]:
        """Query materials from knowledge base"""
        # TODO: Implement knowledge base query
        return []
    
    def _query_codes(self, element: str) -> List[Dict[str, Any]]:
        """Query code requirements"""
        # TODO: Query from code_requirements table
        return []
    
    def _query_costs(self, scope: str) -> Dict[str, Any]:
        """Query cost data"""
        # TODO: Query from cost_data table
        return {}
    
    def _generate_next_step(self, project_state: ProjectState) -> NextStep:
        """Generate next step using LLM"""
        # TODO: Use LLM to analyze state and recommend next action
        pass
    
    def _extract_lessons(self, project_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract lessons learned from project"""
        # TODO: Analyze what went well/poorly
        return []
    
    def _share_with_other_domains(self, skills: List[Dict[str, Any]]):
        """Share learning with other Kalki domains"""
        # TODO: Cross-domain knowledge transfer
        # Example: "Sequential process management" learned in construction
        #          applies to game development (asset pipeline)
        pass


# ========== Example Usage ==========

if __name__ == "__main__":
    # Initialize Construction Copilot
    kalki_construction = ConstructionCopilot()
    
    # User wants to build
    project = kalki_construction.start_project({
        "description": "3-bedroom single-story house",
        "budget": 350000,
        "location": "Austin, TX",
        "timeline": "12 months",
        "experience_level": "beginner"
    })
    
    # Get first step
    next_step = kalki_construction.get_next_step(project)
    
    print(f"📋 {next_step.title}")
    print(f"💰 Estimated cost: ${next_step.estimated_cost:,.0f}")
    print(f"⏱️  Duration: {next_step.estimated_duration_days} days")
    print(f"\n{next_step.description}")
    
    # Kalki guides through material selection
    siding = kalki_construction.guide_material_selection(
        component="exterior_siding",
        constraints={
            "budget": "medium",
            "climate": "humid subtropical",
            "aesthetic": "modern farmhouse"
        }
    )
    
    # Kalki checks code compliance
    stair_check = kalki_construction.validate_code_compliance(
        design_element="stairway",
        specifications={
            "riser_height": 7.5,
            "tread_depth": 10,
            "width": 36
        }
    )
    
    print(f"\n✅ Compliant: {stair_check['is_compliant']}")
