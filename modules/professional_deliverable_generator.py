"""
Professional Deliverable Generator
Unified framework for generating professional deliverables across all domains.

Supports:
- CAD drawings, blueprints
- Technical documents, BOMs, schedules
- Source code, assets
- Simulation models
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from modules.llm import LLMEngine
from modules.visual_knowledge_graph import VisualKnowledgeGraph

logger = logging.getLogger(__name__)


class DeliverableType(Enum):
    """Types of professional deliverables"""
    CAD_DRAWING = "cad_drawing"
    BLUEPRINT = "blueprint"
    TECHNICAL_DOCUMENT = "technical_document"
    SOURCE_CODE = "source_code"
    BILL_OF_MATERIALS = "bill_of_materials"
    SCHEDULE = "schedule"
    COST_ESTIMATE = "cost_estimate"
    TEST_PLAN = "test_plan"
    SIMULATION_MODEL = "simulation_model"
    ASSET = "asset"


@dataclass
class DeliverableSpec:
    """Specification for generating a deliverable"""
    deliverable_type: DeliverableType
    project: Any  # ProjectStateMachine
    specifications: Dict[str, Any]
    output_format: str = "pdf"
    quality_standard: Optional[str] = None


class ProfessionalDeliverableGenerator:
    """
    Generates professional-grade deliverables.
    
    Domains use this to create:
    - Construction: CAD drawings, blueprints, BOMs, schedules
    - Game Dev: Source code, assets, design docs
    - Robotics: CAD models, control code, simulation files
    """
    
    def __init__(self, llm_engine: LLMEngine, knowledge_graph: VisualKnowledgeGraph):
        self.llm_engine = llm_engine
        self.knowledge_graph = knowledge_graph
        self.generators: Dict[DeliverableType, callable] = {}
        self._register_generators()
    
    def _register_generators(self):
        """Register deliverable generators"""
        # Lazy import to avoid circular dependencies
        try:
            from modules.cad_drawings import CADDrawingGenerator
            self.generators[DeliverableType.CAD_DRAWING] = self._generate_cad
        except ImportError:
            logger.warning("CADDrawings module not available")
        
        try:
            from modules.architectural_drawings import ArchitecturalDrawingGenerator
            self.generators[DeliverableType.BLUEPRINT] = self._generate_blueprint
        except ImportError:
            logger.warning("ArchitecturalDrawings module not available")
        
        # Document generation using Llama 3.1 8B
        self.generators[DeliverableType.TECHNICAL_DOCUMENT] = self._generate_technical_document
        self.generators[DeliverableType.BILL_OF_MATERIALS] = self._generate_bom
        self.generators[DeliverableType.SCHEDULE] = self._generate_schedule
        self.generators[DeliverableType.COST_ESTIMATE] = self._generate_cost_estimate
        self.generators[DeliverableType.SOURCE_CODE] = self._generate_source_code
        self.generators[DeliverableType.TEST_PLAN] = self._generate_test_plan
    
    async def generate_deliverable(
        self,
        deliverable_type: DeliverableType,
        project: Any,
        specifications: Dict[str, Any],
        output_format: str = "pdf",
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Generate a professional deliverable.
        
        Args:
            deliverable_type: Type of deliverable to generate
            project: Project state machine
            specifications: Domain-specific specifications
            output_format: Output file format (pdf, dwg, json, etc.)
            output_dir: Output directory (defaults to project output dir)
        
        Returns:
            Path to generated file
        """
        if deliverable_type not in self.generators:
            raise ValueError(f"Generator not available for {deliverable_type}")
        
        generator = self.generators[deliverable_type]
        
        # Get relevant knowledge
        try:
            knowledge_result = self.knowledge_graph.query_with_text(
                query=specifications.get('query', ''),
                top_k=10
            )
            # Extract knowledge from result
            knowledge = []
            if isinstance(knowledge_result, dict):
                knowledge = knowledge_result.get('text_results', [])[:10]
            elif isinstance(knowledge_result, list):
                knowledge = knowledge_result[:10]
            else:
                knowledge = []
        except Exception as e:
            logger.warning(f"Knowledge graph query failed: {e}")
            knowledge = []
        
        # Generate deliverable
        output_path = await generator(
            project=project,
            specifications=specifications,
            knowledge=knowledge,
            output_format=output_format,
            output_dir=output_dir
        )
        
        return output_path
    
    async def _generate_cad(
        self,
        project: Any,
        specifications: Dict[str, Any],
        knowledge: List[Dict],
        output_format: str,
        output_dir: Optional[Path]
    ) -> Path:
        """Generate CAD drawing using Llama 3.2 Vision for design analysis"""
        try:
            # Use Llama 3.2 Vision to analyze design requirements
            if self.llm_engine.vision_engine:
                # Analyze any provided images or design references
                design_analysis = await self._analyze_design_with_vision(
                    specifications, knowledge
                )
                specifications['vision_analysis'] = design_analysis
            
            from modules.cad_drawings import CADDrawingGenerator
            cad_gen = CADDrawingGenerator()
            
            # Generate CAD from specifications
            if output_dir is None:
                output_dir = Path("output/deliverables")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a temporary SCAD specification if needed
            scad_spec = await self._create_cad_specification(specifications, knowledge)
            
            # Generate 2D projection
            result = await cad_gen.generate_2d_projection(
                scad_file=scad_spec,
                view='front',
                output_format=output_format,
                dimensions=True
            )
            
            if result.get("status") == "success":
                return Path(result.get("output_path", output_dir / f"cad_{getattr(project, 'project_id', 'default')}.{output_format}"))
            else:
                # Fallback: Generate CAD specification document
                return await self._generate_cad_spec_document(project, specifications, output_dir, output_format)
                
        except Exception as e:
            logger.error(f"CAD generation failed: {e}, generating specification document instead")
            # Fallback to specification document
            return await self._generate_cad_spec_document(project, specifications, output_dir, output_format)
    
    async def _analyze_design_with_vision(
        self,
        specifications: Dict[str, Any],
        knowledge: List[Dict]
    ) -> Dict[str, Any]:
        """Use Llama 3.2 Vision to analyze design requirements"""
        if not self.llm_engine.vision_engine:
            return {}
        
        # Check for image references in specifications
        image_paths = specifications.get('image_references', [])
        if not image_paths:
            return {}
        
        analyses = []
        for image_path in image_paths[:3]:  # Limit to 3 images
            try:
                analysis = await self.llm_engine.analyze_image(
                    image_path=image_path,
                    prompt="Analyze this design image and extract: dimensions, materials, structural elements, and design intent."
                )
                if isinstance(analysis, dict):
                    analyses.append(analysis.get('text', str(analysis)))
                else:
                    analyses.append(str(analysis))
            except Exception as e:
                logger.warning(f"Vision analysis failed for {image_path}: {e}")
        
        return {
            "image_analyses": analyses,
            "design_insights": " ".join(analyses)
        }
    
    async def _create_cad_specification(
        self,
        specifications: Dict[str, Any],
        knowledge: List[Dict]
    ) -> str:
        """Create CAD specification using Llama 3.1 8B"""
        spec_prompt = f"""Generate an OpenSCAD specification for a CAD model based on these requirements:

Specifications:
{specifications}

Knowledge Base:
{knowledge[:3]}

Generate OpenSCAD code that creates the 3D model."""

        spec_code = await self.llm_engine.generate(
            prompt=spec_prompt,
            max_tokens=1000,
            temperature=0.5
        )
        
        if isinstance(spec_code, dict):
            spec_text = spec_code.get("text", str(spec_code))
        else:
            spec_text = str(spec_code)
        
        # Save to temporary file
        temp_dir = Path("output/temp")
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / "cad_spec.scad"
        with open(temp_file, 'w') as f:
            f.write(spec_text)
        
        return str(temp_file)
    
    async def _generate_cad_spec_document(
        self,
        project: Any,
        specifications: Dict[str, Any],
        output_dir: Path,
        output_format: str
    ) -> Path:
        """Generate CAD specification document as fallback"""
        doc_prompt = f"""Generate a detailed CAD specification document for a {getattr(project, 'domain', 'general')} project.

Specifications:
{specifications}

Include:
- Dimensions and measurements
- Material specifications
- Structural requirements
- Design constraints
- Technical drawings description"""

        doc_content = await self.llm_engine.generate(
            prompt=doc_prompt,
            max_tokens=1500,
            temperature=0.5
        )
        
        if isinstance(doc_content, dict):
            doc_text = doc_content.get("text", str(doc_content))
        else:
            doc_text = str(doc_content)
        
        output_path = output_dir / f"cad_specification_{getattr(project, 'project_id', 'default')}.{output_format}"
        with open(output_path, 'w') as f:
            f.write(doc_text)
        
        return output_path
    
    async def _generate_blueprint(
        self,
        project: Any,
        specifications: Dict[str, Any],
        knowledge: List[Dict],
        output_format: str,
        output_dir: Optional[Path]
    ) -> Path:
        """Generate blueprint using Llama 3.2 Vision for design validation"""
        try:
            # Use Llama 3.2 Vision to validate design if images provided
            if self.llm_engine.vision_engine and specifications.get('image_references'):
                validation = await self._validate_design_with_vision(
                    specifications.get('image_references', [])
                )
                specifications['vision_validation'] = validation
            
            from modules.architectural_drawings import ArchitecturalDrawingGenerator
            blueprint_gen = ArchitecturalDrawingGenerator()
            
            if output_dir is None:
                output_dir = Path("output/deliverables")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate building data from specifications
            building_data = await self._extract_building_data(specifications, knowledge)
            
            # Generate complete drawing set
            drawings = blueprint_gen.generate_complete_set(building_data, output_dir)
            
            if drawings:
                return Path(drawings[0])  # Return first drawing
            else:
                # Fallback: Generate blueprint specification
                return await self._generate_blueprint_spec(project, specifications, output_dir, output_format)
                
        except Exception as e:
            logger.error(f"Blueprint generation failed: {e}, generating specification instead")
            return await self._generate_blueprint_spec(project, specifications, output_dir, output_format)
    
    async def _validate_design_with_vision(
        self,
        image_paths: List[str]
    ) -> Dict[str, Any]:
        """Use Llama 3.2 Vision to validate design"""
        if not self.llm_engine.vision_engine:
            return {}
        
        validations = []
        for image_path in image_paths[:2]:
            try:
                validation = await self.llm_engine.analyze_image(
                    image_path=image_path,
                    prompt="Validate this architectural design for: code compliance, structural feasibility, and design quality. Identify any issues."
                )
                if isinstance(validation, dict):
                    validations.append(validation.get('text', str(validation)))
                else:
                    validations.append(str(validation))
            except Exception as e:
                logger.warning(f"Design validation failed for {image_path}: {e}")
        
        return {
            "validations": validations,
            "overall_assessment": " ".join(validations)
        }
    
    async def _extract_building_data(
        self,
        specifications: Dict[str, Any],
        knowledge: List[Dict]
    ) -> Dict[str, Any]:
        """Extract building data from specifications using Llama 3.1 8B"""
        extract_prompt = f"""Extract building parameters from these specifications:

{specifications}

Extract and return as JSON:
- width_ft (building width in feet)
- depth_ft (building depth in feet)
- levels (number of floors)
- building_type (residential, commercial, etc.)
- any other relevant parameters"""

        extracted = await self.llm_engine.generate(
            prompt=extract_prompt,
            max_tokens=300,
            temperature=0.3
        )
        
        if isinstance(extracted, dict):
            extracted_text = extracted.get("text", str(extracted))
        else:
            extracted_text = str(extracted)
        
        # Try to parse JSON
        import json
        import re
        json_match = re.search(r'\{.*\}', extracted_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except:
                pass
        
        # Fallback to defaults
        return {
            "width_ft": specifications.get("width_ft", 30),
            "depth_ft": specifications.get("depth_ft", 50),
            "levels": specifications.get("levels", 1),
            "building_type": specifications.get("building_type", "residential")
        }
    
    async def _generate_blueprint_spec(
        self,
        project: Any,
        specifications: Dict[str, Any],
        output_dir: Path,
        output_format: str
    ) -> Path:
        """Generate blueprint specification document as fallback"""
        spec_prompt = f"""Generate a detailed architectural blueprint specification for a {getattr(project, 'domain', 'general')} project.

Specifications:
{specifications}

Include:
- Floor plans description
- Elevations description
- Building sections
- Dimensions and measurements
- Material specifications"""

        spec_content = await self.llm_engine.generate(
            prompt=spec_prompt,
            max_tokens=1500,
            temperature=0.5
        )
        
        if isinstance(spec_content, dict):
            spec_text = spec_content.get("text", str(spec_content))
        else:
            spec_text = str(spec_content)
        
        output_path = output_dir / f"blueprint_specification_{getattr(project, 'project_id', 'default')}.{output_format}"
        with open(output_path, 'w') as f:
            f.write(spec_text)
        
        return output_path
    
    async def _generate_technical_document(
        self,
        project: Any,
        specifications: Dict[str, Any],
        knowledge: List[Dict],
        output_format: str,
        output_dir: Optional[Path]
    ) -> Path:
        """Generate technical document using Llama 3.1 8B"""
        # Use LLM to generate document
        doc_prompt = f"""Generate a professional technical document for a {getattr(project, 'domain', 'general')} project.

Project Details:
{specifications}

Knowledge Base:
{knowledge[:5]}

Generate a comprehensive technical document including:
1. Executive summary
2. Technical specifications
3. Implementation details
4. Testing and validation
5. Appendices

Format as professional technical document."""

        doc_content = await self.llm_engine.generate(
            prompt=doc_prompt,
            max_tokens=2000,
            temperature=0.7
        )
        
        # Extract text if dict response
        if isinstance(doc_content, dict):
            doc_text = doc_content.get("text", str(doc_content))
        else:
            doc_text = str(doc_content)
        
        # Save to file
        if output_dir is None:
            output_dir = Path("output/deliverables")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"technical_document_{project.project_id if hasattr(project, 'project_id') else 'default'}.{output_format}"
        
        with open(output_path, 'w') as f:
            f.write(doc_text)
        
        return output_path
    
    async def _generate_bom(
        self,
        project: Any,
        specifications: Dict[str, Any],
        knowledge: List[Dict],
        output_format: str,
        output_dir: Optional[Path]
    ) -> Path:
        """Generate Bill of Materials using Llama 3.1 8B"""
        bom_prompt = f"""Generate a detailed Bill of Materials (BOM) for a {getattr(project, 'domain', 'general')} project.

Project Specifications:
{specifications}

Generate a professional BOM with:
- Item descriptions
- Quantities
- Unit costs
- Total costs
- Suppliers (if applicable)
- Part numbers

Format as structured data (JSON or table)."""

        bom_content = await self.llm_engine.generate(
            prompt=bom_prompt,
            max_tokens=1500,
            temperature=0.5
        )
        
        if isinstance(bom_content, dict):
            bom_text = bom_content.get("text", str(bom_content))
        else:
            bom_text = str(bom_content)
        
        if output_dir is None:
            output_dir = Path("output/deliverables")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"bom_{project.project_id if hasattr(project, 'project_id') else 'default'}.{output_format}"
        
        with open(output_path, 'w') as f:
            f.write(bom_text)
        
        return output_path
    
    async def _generate_schedule(
        self,
        project: Any,
        specifications: Dict[str, Any],
        knowledge: List[Dict],
        output_format: str,
        output_dir: Optional[Path]
    ) -> Path:
        """Generate project schedule using Llama 3.1 8B"""
        schedule_prompt = f"""Generate a detailed project schedule for a {getattr(project, 'domain', 'general')} project.

Project Details:
{specifications}

Generate a professional schedule with:
- Task breakdown
- Dependencies
- Timeline estimates
- Milestones
- Resource allocation

Format as Gantt chart data or structured schedule."""

        schedule_content = await self.llm_engine.generate(
            prompt=schedule_prompt,
            max_tokens=1500,
            temperature=0.5
        )
        
        if isinstance(schedule_content, dict):
            schedule_text = schedule_content.get("text", str(schedule_content))
        else:
            schedule_text = str(schedule_content)
        
        if output_dir is None:
            output_dir = Path("output/deliverables")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"schedule_{project.project_id if hasattr(project, 'project_id') else 'default'}.{output_format}"
        
        with open(output_path, 'w') as f:
            f.write(schedule_text)
        
        return output_path
    
    async def _generate_cost_estimate(
        self,
        project: Any,
        specifications: Dict[str, Any],
        knowledge: List[Dict],
        output_format: str,
        output_dir: Optional[Path]
    ) -> Path:
        """Generate cost estimate using Llama 3.1 8B"""
        cost_prompt = f"""Generate a detailed cost estimate for a {getattr(project, 'domain', 'general')} project.

Project Specifications:
{specifications}

Generate a professional cost estimate with:
- Material costs
- Labor costs
- Equipment costs
- Overhead
- Contingency
- Total cost breakdown

Format as structured cost estimate."""

        cost_content = await self.llm_engine.generate(
            prompt=cost_prompt,
            max_tokens=1000,
            temperature=0.3
        )
        
        if isinstance(cost_content, dict):
            cost_text = cost_content.get("text", str(cost_content))
        else:
            cost_text = str(cost_content)
        
        if output_dir is None:
            output_dir = Path("output/deliverables")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"cost_estimate_{project.project_id if hasattr(project, 'project_id') else 'default'}.{output_format}"
        
        with open(output_path, 'w') as f:
            f.write(cost_text)
        
        return output_path
    
    async def _generate_source_code(
        self,
        project: Any,
        specifications: Dict[str, Any],
        knowledge: List[Dict],
        output_format: str,
        output_dir: Optional[Path]
    ) -> Path:
        """Generate source code using Llama 3.1 8B"""
        code_prompt = f"""Generate professional source code for a {getattr(project, 'domain', 'general')} project.

Requirements:
{specifications}

Generate clean, well-documented, production-ready code following best practices."""

        code_content = await self.llm_engine.generate(
            prompt=code_prompt,
            max_tokens=2000,
            temperature=0.7
        )
        
        if isinstance(code_content, dict):
            code_text = code_content.get("text", str(code_content))
        else:
            code_text = str(code_content)
        
        if output_dir is None:
            output_dir = Path("output/deliverables")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine file extension from output_format or project type
        ext = output_format if output_format in ['py', 'js', 'cpp', 'java'] else 'txt'
        output_path = output_dir / f"source_code_{project.project_id if hasattr(project, 'project_id') else 'default'}.{ext}"
        
        with open(output_path, 'w') as f:
            f.write(code_text)
        
        return output_path
    
    async def _generate_test_plan(
        self,
        project: Any,
        specifications: Dict[str, Any],
        knowledge: List[Dict],
        output_format: str,
        output_dir: Optional[Path]
    ) -> Path:
        """Generate test plan using Llama 3.1 8B"""
        test_prompt = f"""Generate a comprehensive test plan for a {getattr(project, 'domain', 'general')} project.

Project Specifications:
{specifications}

Generate a professional test plan with:
- Test objectives
- Test cases
- Test procedures
- Expected results
- Acceptance criteria"""

        test_content = await self.llm_engine.generate(
            prompt=test_prompt,
            max_tokens=1500,
            temperature=0.5
        )
        
        if isinstance(test_content, dict):
            test_text = test_content.get("text", str(test_content))
        else:
            test_text = str(test_content)
        
        if output_dir is None:
            output_dir = Path("output/deliverables")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"test_plan_{project.project_id if hasattr(project, 'project_id') else 'default'}.{output_format}"
        
        with open(output_path, 'w') as f:
            f.write(test_text)
        
        return output_path
    
    async def generate_deliverable_suite(
        self,
        project: Any,
        deliverable_types: List[DeliverableType],
        specifications: Dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> Dict[DeliverableType, Path]:
        """Generate multiple deliverables for a project"""
        results = {}
        
        for deliverable_type in deliverable_types:
            try:
                output_path = await self.generate_deliverable(
                    deliverable_type=deliverable_type,
                    project=project,
                    specifications=specifications,
                    output_dir=output_dir
                )
                results[deliverable_type] = output_path
                logger.info(f"✅ Generated {deliverable_type.value}: {output_path}")
            except Exception as e:
                logger.error(f"❌ Failed to generate {deliverable_type.value}: {e}")
                results[deliverable_type] = None
        
        return results

