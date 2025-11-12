"""
Construction Domain Vision Extractors
Specialized vision analysis for construction industry use cases.

Extracts knowledge from:
- Blueprints and architectural drawings
- Site inspection photos
- Material identification photos
- Safety compliance images
"""

from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import json
import re


class ConstructionPhase(Enum):
    """Construction project phases"""
    PLANNING = "planning"
    EXCAVATION = "excavation"
    FOUNDATION = "foundation"
    FRAMING = "framing"
    ROUGH_IN = "rough_in"
    INSULATION = "insulation"
    DRYWALL = "drywall"
    FINISH = "finish"
    LANDSCAPING = "landscaping"


@dataclass
class BlueprintAnalysis:
    """Result from blueprint analysis"""
    building_type: str
    dimensions: Dict[str, float]  # width, length, height
    rooms: List[Dict[str, Any]]  # room name, size, purpose
    structural_elements: List[Dict[str, Any]]  # beams, columns, walls
    openings: List[Dict[str, Any]]  # doors, windows
    annotations: List[str]  # text found on blueprint
    materials_specified: List[str]
    code_compliance_notes: List[str]
    estimated_square_footage: Optional[float]
    confidence: float


@dataclass
class SiteInspectionAnalysis:
    """Result from site photo analysis"""
    construction_phase: ConstructionPhase
    progress_percentage: float
    quality_issues: List[Dict[str, Any]]  # issue, severity, location
    safety_concerns: List[Dict[str, Any]]  # concern, severity, recommendation
    completed_items: List[str]
    pending_items: List[str]
    weather_conditions: Optional[str]
    worker_count: Optional[int]
    equipment_visible: List[str]
    confidence: float


@dataclass
class MaterialAnalysis:
    """Result from material photo analysis"""
    material_type: str
    material_grade: Optional[str]
    dimensions: Optional[Dict[str, float]]
    condition: str  # new, used, damaged, etc.
    quality_assessment: str
    estimated_quantity: Optional[float]
    compliance_standards: List[str]
    defects: List[str]
    suitability_for_use: bool
    confidence: float


class ConstructionVisionExtractor:
    """
    Specialized vision extractor for construction domain.
    Uses Llama 3.2 Vision 11B for detailed construction image analysis.
    """
    
    def __init__(self, vision_engine, cache=None):
        """
        Initialize construction vision extractor.
        
        Args:
            vision_engine: LlamaVisionEngine instance from modules/llm.py
            cache: Optional VisionCache for performance
        """
        self.vision_engine = vision_engine
        self.cache = cache
        
        # Standard dimensions for reference
        self.standard_dimensions = {
            "wall_stud_spacing": 16.0,  # inches
            "joist_spacing": 16.0,  # inches
            "door_height_standard": 80.0,  # inches
            "window_sill_height": 36.0,  # inches
            "ceiling_height_standard": 96.0,  # inches (8 ft)
        }
    
    def extract_from_blueprint(
        self,
        image_path: str,
        blueprint_type: Optional[str] = None
    ) -> BlueprintAnalysis:
        """
        Extract detailed information from architectural blueprints.
        
        Args:
            image_path: Path to blueprint image
            blueprint_type: Optional hint (floor_plan, elevation, section, detail)
        
        Returns:
            BlueprintAnalysis with extracted information
        """
        # Construct detailed prompt for blueprint analysis
        prompt = """Analyze this construction blueprint in detail. Extract:

1. BUILDING INFORMATION:
   - Building type (residential, commercial, garage, etc.)
   - Overall dimensions (width, length, height if visible)
   - Number of floors/levels

2. ROOMS AND SPACES:
   - List each room with dimensions
   - Room purpose/function
   - Square footage per room

3. STRUCTURAL ELEMENTS:
   - Beams (location, size, material)
   - Columns and posts (location, dimensions)
   - Load-bearing walls vs partition walls
   - Foundation type

4. OPENINGS:
   - Doors (location, size, type - interior/exterior)
   - Windows (location, size, type)

5. DIMENSIONS AND MEASUREMENTS:
   - Wall lengths
   - Room dimensions
   - Ceiling heights
   - Any critical measurements marked

6. ANNOTATIONS:
   - Text labels and notes
   - Material specifications
   - Code references

7. MATERIALS:
   - Materials specified (wood, concrete, steel, etc.)
   - Grades or quality specifications

8. CODE COMPLIANCE:
   - Building code references
   - Compliance notes or stamps

Provide measurements in feet and inches. Be precise and thorough."""

        # Get cached result or analyze
        result = self._get_cached_or_analyze(image_path, prompt)
        
        # Parse vision model output into structured format
        return self._parse_blueprint_analysis(result, image_path)
    
    def extract_from_site_photo(
        self,
        image_path: str,
        expected_phase: Optional[ConstructionPhase] = None
    ) -> SiteInspectionAnalysis:
        """
        Analyze construction site photos for progress and quality control.
        
        Args:
            image_path: Path to site photo
            expected_phase: Optional expected construction phase
        
        Returns:
            SiteInspectionAnalysis with inspection results
        """
        prompt = """Analyze this construction site photo comprehensively:

1. CONSTRUCTION PHASE:
   - What phase is this project in? (excavation, foundation, framing, rough-in, insulation, drywall, finish, etc.)
   - Estimated progress percentage for this phase

2. QUALITY ASSESSMENT:
   - Quality issues visible (improper installation, gaps, misalignment, etc.)
   - Severity: Critical, Major, Minor
   - Location/description of each issue

3. SAFETY INSPECTION:
   - Safety concerns visible
   - Missing safety equipment
   - Hazardous conditions
   - Recommendations

4. WORK COMPLETION:
   - What work items are completed in this photo?
   - What work items are still pending?
   - Is work proceeding according to standard practices?

5. SITE CONDITIONS:
   - Weather conditions (if determinable)
   - Number of workers visible
   - Equipment present (scaffolding, tools, machinery)
   - Site organization and cleanliness

6. CODE COMPLIANCE:
   - Any visible code violations
   - Required inspections needed

Be specific about locations and provide actionable recommendations."""

        result = self._get_cached_or_analyze(image_path, prompt)
        
        return self._parse_site_inspection(result, image_path, expected_phase)
    
    def extract_from_material_photo(
        self,
        image_path: str,
        material_type_hint: Optional[str] = None
    ) -> MaterialAnalysis:
        """
        Identify and assess construction materials from photos.
        
        Args:
            image_path: Path to material photo
            material_type_hint: Optional hint about material type
        
        Returns:
            MaterialAnalysis with material identification and assessment
        """
        prompt = """Analyze this construction material photo:

1. MATERIAL IDENTIFICATION:
   - What type of material is this? (lumber, concrete, steel, drywall, insulation, etc.)
   - Specific grade or type (e.g., 2x4 SPF, #3 rebar, 5/8" type X drywall)
   - Dimensions (length, width, thickness, diameter)

2. QUANTITY ESTIMATION:
   - How many units/pieces are visible?
   - Estimated total quantity
   - Unit of measurement

3. CONDITION ASSESSMENT:
   - Is the material new, used, or damaged?
   - Quality rating (excellent, good, acceptable, poor)
   - Visible defects (cracks, warping, rust, moisture damage, etc.)

4. COMPLIANCE AND STANDARDS:
   - Does it meet typical building standards?
   - Grade stamps or markings visible?
   - Certifications or approvals

5. SUITABILITY:
   - Is this material suitable for structural use?
   - Any concerns about using this material?
   - Storage conditions (if determinable)

6. RECOMMENDATIONS:
   - Accept, reject, or conditional use?
   - Required testing or inspection?
   - Proper handling or installation notes

Be specific and detailed in your assessment."""

        result = self._get_cached_or_analyze(image_path, prompt)
        
        return self._parse_material_analysis(result, image_path, material_type_hint)
    
    def analyze_structural_detail(
        self,
        image_path: str,
        detail_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze detailed structural connections and assemblies.
        
        Args:
            image_path: Path to detail photo or drawing
            detail_type: Optional type (connection, joint, assembly, etc.)
        
        Returns:
            Dict with detailed structural analysis
        """
        prompt = """Analyze this structural detail:

1. COMPONENT IDENTIFICATION:
   - What structural elements are shown?
   - Material types (wood, steel, concrete, composite)
   - Sizes and dimensions

2. CONNECTION TYPE:
   - How are elements connected?
   - Fasteners used (nails, screws, bolts, welds, adhesive)
   - Connection strength category

3. LOAD PATH:
   - How do loads transfer through this connection?
   - Critical load-bearing elements
   - Potential weak points

4. CODE COMPLIANCE:
   - Does this meet standard construction practices?
   - Building code requirements relevant to this detail
   - Required reinforcement or bracing

5. INSTALLATION:
   - Proper installation sequence
   - Critical tolerances
   - Common installation errors to avoid

6. ASSESSMENT:
   - Is this detail correctly executed?
   - Quality concerns
   - Recommendations for improvement"""

        result = self._get_cached_or_analyze(image_path, prompt)
        
        return {
            "detail_type": detail_type or "unknown",
            "image_path": image_path,
            "analysis": result.get("analysis", ""),
            "components": self._extract_components(result),
            "code_references": self._extract_code_references(result),
            "recommendations": self._extract_recommendations(result),
            "confidence": result.get("confidence", 0.0)
        }
    
    def batch_analyze_site_photos(
        self,
        image_paths: List[str],
        project_phase: Optional[ConstructionPhase] = None
    ) -> List[SiteInspectionAnalysis]:
        """
        Batch analyze multiple site photos efficiently.
        
        Args:
            image_paths: List of image paths
            project_phase: Optional expected project phase
        
        Returns:
            List of inspection analyses
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"📸 Analyzing site photo {i+1}/{len(image_paths)}: {Path(image_path).name}")
            
            try:
                analysis = self.extract_from_site_photo(image_path, project_phase)
                results.append(analysis)
            except Exception as e:
                print(f"⚠️ Error analyzing {image_path}: {e}")
                # Continue with next image
        
        return results
    
    def generate_inspection_report(
        self,
        site_analyses: List[SiteInspectionAnalysis],
        project_name: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive inspection report from multiple site photos.
        
        Args:
            site_analyses: List of site inspection analyses
            project_name: Name of construction project
        
        Returns:
            Comprehensive inspection report
        """
        if not site_analyses:
            return {"error": "No site analyses provided"}
        
        # Aggregate findings
        all_quality_issues = []
        all_safety_concerns = []
        phases_detected = []
        
        for analysis in site_analyses:
            all_quality_issues.extend(analysis.quality_issues)
            all_safety_concerns.extend(analysis.safety_concerns)
            phases_detected.append(analysis.construction_phase.value)
        
        # Categorize by severity
        critical_issues = [
            issue for issue in all_quality_issues 
            if issue.get("severity", "").lower() == "critical"
        ]
        
        critical_safety = [
            concern for concern in all_safety_concerns
            if concern.get("severity", "").lower() == "critical"
        ]
        
        # Calculate overall progress
        avg_progress = sum(a.progress_percentage for a in site_analyses) / len(site_analyses)
        
        return {
            "project_name": project_name,
            "inspection_date": "auto_generated",
            "photos_analyzed": len(site_analyses),
            "overall_progress": round(avg_progress, 1),
            "current_phase": max(set(phases_detected), key=phases_detected.count),
            "summary": {
                "total_quality_issues": len(all_quality_issues),
                "critical_quality_issues": len(critical_issues),
                "total_safety_concerns": len(all_safety_concerns),
                "critical_safety_concerns": len(critical_safety),
            },
            "critical_issues": critical_issues,
            "critical_safety": critical_safety,
            "detailed_analyses": [
                {
                    "phase": a.construction_phase.value,
                    "progress": a.progress_percentage,
                    "quality_issues": len(a.quality_issues),
                    "safety_concerns": len(a.safety_concerns)
                }
                for a in site_analyses
            ],
            "recommendations": self._generate_recommendations(
                critical_issues, 
                critical_safety,
                avg_progress
            )
        }
    
    # Helper methods
    
    def _get_cached_or_analyze(self, image_path: str, query: str) -> Dict[str, Any]:
        """Get cached result or run vision analysis"""
        if self.cache:
            result = self.cache.get(image_path, query)
            if result:
                return result
        
        # Analyze with vision model
        result = self.vision_engine.analyze_image(image_path, query)
        
        # Cache the result
        if self.cache:
            self.cache.put(image_path, result, query)
        
        return result
    
    def _parse_blueprint_analysis(
        self,
        vision_result: Dict[str, Any],
        image_path: str
    ) -> BlueprintAnalysis:
        """Parse vision model output into BlueprintAnalysis"""
        analysis_text = vision_result.get("analysis", "")
        
        # Extract structured information using regex and heuristics
        # This is a simplified parser - production would be more robust
        
        return BlueprintAnalysis(
            building_type=self._extract_building_type(analysis_text),
            dimensions=self._extract_dimensions(analysis_text),
            rooms=self._extract_rooms(analysis_text),
            structural_elements=self._extract_structural_elements(analysis_text),
            openings=self._extract_openings(analysis_text),
            annotations=self._extract_annotations(analysis_text),
            materials_specified=self._extract_materials(analysis_text),
            code_compliance_notes=self._extract_code_references(vision_result),
            estimated_square_footage=self._calculate_square_footage(analysis_text),
            confidence=vision_result.get("confidence", 0.85)
        )
    
    def _parse_site_inspection(
        self,
        vision_result: Dict[str, Any],
        image_path: str,
        expected_phase: Optional[ConstructionPhase]
    ) -> SiteInspectionAnalysis:
        """Parse vision output into SiteInspectionAnalysis"""
        analysis_text = vision_result.get("analysis", "")
        
        return SiteInspectionAnalysis(
            construction_phase=self._detect_phase(analysis_text, expected_phase),
            progress_percentage=self._extract_progress(analysis_text),
            quality_issues=self._extract_quality_issues(analysis_text),
            safety_concerns=self._extract_safety_concerns(analysis_text),
            completed_items=self._extract_list(analysis_text, "completed"),
            pending_items=self._extract_list(analysis_text, "pending"),
            weather_conditions=self._extract_weather(analysis_text),
            worker_count=self._extract_worker_count(analysis_text),
            equipment_visible=self._extract_equipment(analysis_text),
            confidence=vision_result.get("confidence", 0.85)
        )
    
    def _parse_material_analysis(
        self,
        vision_result: Dict[str, Any],
        image_path: str,
        material_hint: Optional[str]
    ) -> MaterialAnalysis:
        """Parse vision output into MaterialAnalysis"""
        analysis_text = vision_result.get("analysis", "")
        
        return MaterialAnalysis(
            material_type=self._extract_material_type(analysis_text, material_hint),
            material_grade=self._extract_grade(analysis_text),
            dimensions=self._extract_dimensions(analysis_text),
            condition=self._extract_condition(analysis_text),
            quality_assessment=self._extract_quality(analysis_text),
            estimated_quantity=self._extract_quantity(analysis_text),
            compliance_standards=self._extract_standards(analysis_text),
            defects=self._extract_defects(analysis_text),
            suitability_for_use=self._assess_suitability(analysis_text),
            confidence=vision_result.get("confidence", 0.85)
        )
    
    # Text extraction helpers (simplified implementations)
    
    def _extract_building_type(self, text: str) -> str:
        """Extract building type from analysis"""
        text_lower = text.lower()
        if "residential" in text_lower or "house" in text_lower or "home" in text_lower:
            return "residential"
        elif "commercial" in text_lower:
            return "commercial"
        elif "garage" in text_lower:
            return "garage"
        else:
            return "unknown"
    
    def _extract_dimensions(self, text: str) -> Dict[str, float]:
        """Extract dimensions from text"""
        dimensions = {}
        
        # Look for patterns like "20 feet", "30'", "15 ft"
        patterns = [
            r"(\d+(?:\.\d+)?)\s*(?:feet|ft|')",
            r"(\d+(?:\.\d+)?)\s*x\s*(\d+(?:\.\d+)?)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                if isinstance(matches[0], tuple):
                    dimensions["width"] = float(matches[0][0])
                    dimensions["length"] = float(matches[0][1])
                else:
                    dimensions["size"] = float(matches[0])
        
        return dimensions
    
    def _extract_rooms(self, text: str) -> List[Dict[str, Any]]:
        """Extract room information"""
        # Simplified - would use NLP in production
        rooms = []
        room_keywords = ["bedroom", "bathroom", "kitchen", "living room", "dining room", "garage"]
        
        for keyword in room_keywords:
            if keyword in text.lower():
                rooms.append({"name": keyword, "size": "unknown"})
        
        return rooms
    
    def _extract_structural_elements(self, text: str) -> List[Dict[str, Any]]:
        """Extract structural elements"""
        elements = []
        element_keywords = ["beam", "column", "wall", "foundation", "joist", "rafter"]
        
        for keyword in element_keywords:
            if keyword in text.lower():
                elements.append({"type": keyword, "details": "see blueprint"})
        
        return elements
    
    def _extract_openings(self, text: str) -> List[Dict[str, Any]]:
        """Extract doors and windows"""
        openings = []
        
        if "door" in text.lower():
            openings.append({"type": "door", "count": text.lower().count("door")})
        if "window" in text.lower():
            openings.append({"type": "window", "count": text.lower().count("window")})
        
        return openings
    
    def _extract_annotations(self, text: str) -> List[str]:
        """Extract text annotations"""
        # In production, this would extract quoted text or labels
        return [line.strip() for line in text.split('\n') if len(line.strip()) > 10][:5]
    
    def _extract_materials(self, text: str) -> List[str]:
        """Extract material specifications"""
        materials = []
        material_keywords = ["wood", "concrete", "steel", "lumber", "drywall", "insulation"]
        
        for keyword in material_keywords:
            if keyword in text.lower():
                materials.append(keyword)
        
        return materials
    
    def _extract_code_references(self, result: Dict) -> List[str]:
        """Extract building code references"""
        text = result.get("analysis", "")
        codes = []
        
        # Look for code patterns
        if "building code" in text.lower():
            codes.append("Building code compliance noted")
        if "code" in text.lower() and ("section" in text.lower() or "part" in text.lower()):
            codes.append("Specific code sections referenced")
        
        return codes
    
    def _calculate_square_footage(self, text: str) -> Optional[float]:
        """Calculate total square footage"""
        # Look for explicit square footage mentions
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:square feet|sq ft|sf)", text.lower())
        if match:
            return float(match.group(1))
        return None
    
    def _detect_phase(self, text: str, expected: Optional[ConstructionPhase]) -> ConstructionPhase:
        """Detect construction phase"""
        text_lower = text.lower()
        
        if "foundation" in text_lower:
            return ConstructionPhase.FOUNDATION
        elif "framing" in text_lower or "frame" in text_lower:
            return ConstructionPhase.FRAMING
        elif "drywall" in text_lower:
            return ConstructionPhase.DRYWALL
        elif "excavation" in text_lower:
            return ConstructionPhase.EXCAVATION
        else:
            return expected or ConstructionPhase.PLANNING
    
    def _extract_progress(self, text: str) -> float:
        """Extract progress percentage"""
        match = re.search(r"(\d+)%", text)
        if match:
            return float(match.group(1))
        return 50.0  # Default
    
    def _extract_quality_issues(self, text: str) -> List[Dict[str, Any]]:
        """Extract quality issues"""
        issues = []
        issue_keywords = ["issue", "problem", "defect", "improper", "incorrect", "missing"]
        
        for keyword in issue_keywords:
            if keyword in text.lower():
                issues.append({
                    "issue": f"{keyword} detected",
                    "severity": "minor",
                    "description": "See full analysis"
                })
        
        return issues
    
    def _extract_safety_concerns(self, text: str) -> List[Dict[str, Any]]:
        """Extract safety concerns"""
        concerns = []
        safety_keywords = ["safety", "hazard", "danger", "unsafe", "risk"]
        
        for keyword in safety_keywords:
            if keyword in text.lower():
                concerns.append({
                    "concern": f"{keyword} identified",
                    "severity": "major",
                    "recommendation": "Immediate review required"
                })
        
        return concerns
    
    def _extract_list(self, text: str, list_type: str) -> List[str]:
        """Extract lists from text"""
        items = []
        lines = text.split('\n')
        
        for line in lines:
            if list_type.lower() in line.lower():
                # Extract items after the list type marker
                parts = line.split(':')
                if len(parts) > 1:
                    items.extend([item.strip() for item in parts[1].split(',')])
        
        return items[:5]  # Limit to 5 items
    
    def _extract_weather(self, text: str) -> Optional[str]:
        """Extract weather conditions"""
        weather_keywords = ["sunny", "cloudy", "rain", "snow", "clear", "overcast"]
        
        for keyword in weather_keywords:
            if keyword in text.lower():
                return keyword
        
        return None
    
    def _extract_worker_count(self, text: str) -> Optional[int]:
        """Extract number of workers"""
        match = re.search(r"(\d+)\s*workers?", text.lower())
        if match:
            return int(match.group(1))
        return None
    
    def _extract_equipment(self, text: str) -> List[str]:
        """Extract equipment mentioned"""
        equipment = []
        equipment_keywords = ["scaffolding", "ladder", "crane", "excavator", "forklift", "saw"]
        
        for keyword in equipment_keywords:
            if keyword in text.lower():
                equipment.append(keyword)
        
        return equipment
    
    def _extract_components(self, result: Dict) -> List[str]:
        """Extract structural components"""
        # Simplified extraction
        return []
    
    def _extract_recommendations(self, result: Dict) -> List[str]:
        """Extract recommendations from analysis"""
        text = result.get("analysis", "")
        recommendations = []
        
        lines = text.split('\n')
        for line in lines:
            if "recommend" in line.lower() or "should" in line.lower():
                recommendations.append(line.strip())
        
        return recommendations[:5]
    
    def _extract_material_type(self, text: str, hint: Optional[str]) -> str:
        """Extract material type"""
        if hint:
            return hint
        
        material_types = ["lumber", "concrete", "steel", "drywall", "insulation", "brick"]
        for mat in material_types:
            if mat in text.lower():
                return mat
        
        return "unknown"
    
    def _extract_grade(self, text: str) -> Optional[str]:
        """Extract material grade"""
        # Look for grade patterns like "Grade A", "#2", "SPF"
        match = re.search(r"(?:grade|#)\s*([A-Z0-9]+)", text, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def _extract_condition(self, text: str) -> str:
        """Extract material condition"""
        conditions = ["new", "used", "damaged", "excellent", "good", "poor"]
        
        for cond in conditions:
            if cond in text.lower():
                return cond
        
        return "unknown"
    
    def _extract_quality(self, text: str) -> str:
        """Extract quality assessment"""
        if "excellent" in text.lower():
            return "excellent"
        elif "good" in text.lower():
            return "good"
        elif "acceptable" in text.lower() or "adequate" in text.lower():
            return "acceptable"
        elif "poor" in text.lower():
            return "poor"
        else:
            return "needs assessment"
    
    def _extract_quantity(self, text: str) -> Optional[float]:
        """Extract quantity estimation"""
        match = re.search(r"(\d+(?:\.\d+)?)\s*(?:units?|pieces?|items?)", text.lower())
        if match:
            return float(match.group(1))
        return None
    
    def _extract_standards(self, text: str) -> List[str]:
        """Extract compliance standards"""
        standards = []
        standard_keywords = ["ASTM", "CSA", "ICC", "building code", "standard"]
        
        for keyword in standard_keywords:
            if keyword in text:
                standards.append(keyword)
        
        return standards
    
    def _extract_defects(self, text: str) -> List[str]:
        """Extract visible defects"""
        defects = []
        defect_keywords = ["crack", "warp", "rust", "moisture", "damage", "defect"]
        
        for keyword in defect_keywords:
            if keyword in text.lower():
                defects.append(keyword)
        
        return defects
    
    def _assess_suitability(self, text: str) -> bool:
        """Assess if material is suitable for use"""
        unsuitable_keywords = ["reject", "unsuitable", "not recommended", "fail", "poor quality"]
        
        for keyword in unsuitable_keywords:
            if keyword in text.lower():
                return False
        
        return True
    
    def _generate_recommendations(
        self,
        critical_issues: List[Dict],
        critical_safety: List[Dict],
        progress: float
    ) -> List[str]:
        """Generate recommendations based on findings"""
        recommendations = []
        
        if critical_issues:
            recommendations.append(f"Address {len(critical_issues)} critical quality issues immediately")
        
        if critical_safety:
            recommendations.append(f"URGENT: Resolve {len(critical_safety)} critical safety concerns before proceeding")
        
        if progress < 25:
            recommendations.append("Project in early stages - ensure quality foundation work")
        elif progress > 75:
            recommendations.append("Project nearing completion - focus on finish quality")
        
        return recommendations


# Convenience function for quick blueprint analysis
def analyze_blueprint(image_path: str, vision_engine) -> BlueprintAnalysis:
    """Quick blueprint analysis function"""
    extractor = ConstructionVisionExtractor(vision_engine)
    return extractor.extract_from_blueprint(image_path)


# Convenience function for quick site inspection
def inspect_site(image_path: str, vision_engine) -> SiteInspectionAnalysis:
    """Quick site inspection function"""
    extractor = ConstructionVisionExtractor(vision_engine)
    return extractor.extract_from_site_photo(image_path)


if __name__ == "__main__":
    print("🏗️ Construction Vision Extractors Ready")
    print("=" * 60)
    print("\nCapabilities:")
    print("  ✅ Blueprint analysis (dimensions, rooms, structure)")
    print("  ✅ Site inspection (progress, quality, safety)")
    print("  ✅ Material identification and assessment")
    print("  ✅ Structural detail analysis")
    print("  ✅ Batch processing for multiple photos")
    print("  ✅ Comprehensive inspection report generation")
    print("\nIntegrates with:")
    print("  - LlamaVisionEngine (modules/llm.py)")
    print("  - VisionCache (modules/intelligent_cache.py)")
    print("  - Construction Domain (modules/domains/construction_domain/)")
