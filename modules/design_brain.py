# ============================================================
# Kalki v2.4 — design_brain.py
# ------------------------------------------------------------
# Reasoning Layer: Multi-modal Generative Design Engine
# - LLaMA 3.1 8B for analytical reasoning and design logic
# - Intent understanding and component decomposition
# - Engineering knowledge integration
# - Parametric design memory
# ============================================================

import os
import asyncio
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from modules.utils.config import CONFIG
from modules.utils.logging_config import get_logger
from modules.llm import LLMEngine
from modules.learning.vectordb import VectorDBManager

logger = get_logger("Kalki.DesignBrain")

@dataclass
class DesignIntent:
    """Parsed design intent from user input"""
    category: str  # "architecture", "robotics", "vehicle", "machine", etc.
    complexity: str  # "simple", "moderate", "complex", "advanced"
    components: List[str]  # Key design components
    constraints: List[str]  # Design constraints and requirements
    materials: List[str]  # Suggested materials
    scale: str  # "small", "medium", "large", "massive"
    power_source: Optional[str] = None
    mobility_type: Optional[str] = None
    environment: Optional[str] = None

@dataclass
class DesignComponent:
    """Individual design component specification"""
    name: str
    function: str
    requirements: List[str]
    interfaces: List[str]  # How it connects to other components
    materials: List[str]
    dimensions: Dict[str, float]  # Rough dimensional estimates
    complexity: str

@dataclass
class DesignBlueprint:
    """Complete design blueprint"""
    id: str
    timestamp: str
    intent: DesignIntent
    components: List[DesignComponent]
    system_requirements: Dict[str, Any]
    design_parameters: Dict[str, Any]
    validation_checks: List[str]

class DesignBrain:
    """Multi-modal generative design reasoning engine"""

    def __init__(self):
        self.llm_engine = None
        self.vector_db = None
        self.design_memory = self._load_design_memory()
        self.engineering_knowledge = self._load_engineering_knowledge()
        self.fine_tuned_model_path = None  # Path to fine-tuned model if available

        # Design categories and their characteristics
        self.design_categories = {
            "architecture": {
                "keywords": ["house", "building", "structure", "home", "office", "bridge", "tower"],
                "components": ["foundation", "structure", "roof", "utilities", "interior"],
                "constraints": ["load-bearing", "environmental", "code compliance"]
            },
            "robotics": {
                "keywords": ["robot", "automation", "mechanism", "arm", "gripper", "sensor"],
                "components": ["chassis", "actuators", "sensors", "controller", "power"],
                "constraints": ["degrees of freedom", "payload", "precision", "safety"]
            },
            "vehicle": {
                "keywords": ["car", "truck", "aircraft", "boat", "drone", "flying"],
                "components": ["frame", "propulsion", "control", "payload", "safety"],
                "constraints": ["aerodynamics", "weight", "range", "stability"]
            },
            "machine": {
                "keywords": ["engine", "pump", "generator", "tool", "equipment"],
                "components": ["power_unit", "transmission", "control", "output"],
                "constraints": ["efficiency", "durability", "maintenance", "safety"]
            }
        }

    async def initialize(self) -> bool:
        """Initialize the design brain"""
        try:
            # Check for fine-tuned model
            config = CONFIG
            fine_tuned_path = config.get('design_brain', {}).get('fine_tuned_model_path')
            if fine_tuned_path and os.path.exists(fine_tuned_path):
                self.fine_tuned_model_path = fine_tuned_path
                self.llm_engine = LLMEngine(model_path=fine_tuned_path)
                logger.info(f"Loading fine-tuned model from: {fine_tuned_path}")
            else:
                self.llm_engine = LLMEngine()
            
            await self.llm_engine.initialize()

            # VectorDBManager is initialized synchronously
            self.vector_db = VectorDBManager()

            logger.info("Design Brain initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Design Brain: {e}")
            return False

    async def process_design_request(self, request: str) -> DesignBlueprint:
        """Process a design request and generate a complete blueprint"""

        # Step 1: Parse intent from natural language
        intent = await self._parse_design_intent(request)

        # Step 2: Query engineering knowledge base
        relevant_knowledge = await self._query_engineering_knowledge(intent)

        # Step 3: Generate design components
        components = await self._generate_design_components(intent, relevant_knowledge)

        # Step 4: Calculate system requirements
        system_requirements = await self._calculate_system_requirements(intent, components)

        # Step 5: Generate design parameters
        design_parameters = await self._generate_design_parameters(intent, components)

        # Step 6: Create validation checks
        validation_checks = await self._generate_validation_checks(intent, components)

        # Step 7: Create blueprint
        blueprint = DesignBlueprint(
            id=f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            intent=intent,
            components=components,
            system_requirements=system_requirements,
            design_parameters=design_parameters,
            validation_checks=validation_checks
        )

        # Step 8: Save to design memory
        await self._save_design_to_memory(blueprint)

        return blueprint

    async def _parse_design_intent(self, request: str) -> DesignIntent:
        """Parse design intent from natural language using LLM with enhanced creativity prompts"""

        prompt = f"""
        You are an expert design engineer with decades of experience in innovative design solutions. Analyze this design request with creative insight and extract comprehensive design parameters:

        Request: "{request}"

        Think step-by-step like a master designer:
        1. Understand the core problem and user needs
        2. Consider innovative approaches beyond conventional solutions
        3. Identify interdisciplinary opportunities
        4. Anticipate future scalability and adaptability

        Please identify with creativity and precision:
        1. Design category (architecture, robotics, vehicle, machine, product, system, etc.)
        2. Complexity level (simple, moderate, complex, advanced, cutting-edge)
        3. Key components needed (include innovative or unconventional elements)
        4. Design constraints and requirements (technical, environmental, user-experience)
        5. Suggested materials (consider sustainable, advanced, or novel materials)
        6. Scale (micro, small, medium, large, massive, distributed)
        7. Power source (if applicable - consider renewable, efficient, or emerging technologies)
        8. Mobility type (if applicable - wheeled, legged, flying, aquatic, hybrid)
        9. Operating environment (indoor, outdoor, extreme, space, underwater, etc.)

        Example for "Design a modern office building":
        {{
            "category": "architecture",
            "complexity": "complex",
            "components": ["sustainable foundation", "modular structure", "green roof", "smart utilities", "collaborative spaces"],
            "constraints": ["LEED certification", "energy efficiency", "occupant comfort", "future adaptability"],
            "materials": ["recycled steel", "cross-laminated timber", "smart glass", "living walls"],
            "scale": "large",
            "power_source": "renewable hybrid",
            "environment": "urban"
        }}

        Format your response as a valid JSON object with these fields. Be innovative yet practical.
        """

        response = await self.llm_engine.generate(prompt, max_new_tokens=500)

        try:
            # Try to parse JSON response
            intent_data = json.loads(response)
            return DesignIntent(**intent_data)
        except:
            # Fallback: extract information manually
            return await self._fallback_intent_parsing(request)

    async def _fallback_intent_parsing(self, request: str) -> DesignIntent:
        """Fallback intent parsing when LLM JSON parsing fails"""

        request_lower = request.lower()

        # Determine category
        category = "machine"  # default
        for cat, data in self.design_categories.items():
            if any(keyword in request_lower for keyword in data["keywords"]):
                category = cat
                break

        # Determine complexity
        complexity = "moderate"
        if any(word in request_lower for word in ["simple", "basic", "straightforward"]):
            complexity = "simple"
        elif any(word in request_lower for word in ["complex", "advanced", "sophisticated"]):
            complexity = "complex"
        elif any(word in request_lower for word in ["cutting-edge", "revolutionary", "futuristic"]):
            complexity = "advanced"

        # Extract components based on category
        components = self.design_categories[category]["components"]

        # Basic constraints
        constraints = self.design_categories[category]["constraints"]

        # Basic materials
        materials = ["steel", "aluminum", "plastic"]  # default

        # Determine scale
        scale = "medium"
        if any(word in request_lower for word in ["small", "mini", "tiny"]):
            scale = "small"
        elif any(word in request_lower for word in ["large", "big", "massive", "huge"]):
            scale = "large"

        return DesignIntent(
            category=category,
            complexity=complexity,
            components=components,
            constraints=constraints,
            materials=materials,
            scale=scale
        )

    async def _query_engineering_knowledge(self, intent: DesignIntent) -> Dict[str, Any]:
        """
        ENHANCED: Query all knowledge sources systematically
        Integrates vector DB + structured databases for comprehensive knowledge
        """
        
        knowledge = {
            "formulas": [],
            "materials": [],
            "design_rules": [],
            "code_requirements": [],
            "semantic_context": []
        }

        # Create search query based on intent
        query = f"{intent.category} design {intent.complexity} {' '.join(intent.components)}"

        # Step 1: Semantic search via vector database
        try:
            vector_results = self.vector_db.search_similar(query, top_k=10)
            knowledge["semantic_context"] = vector_results
            logger.info(f"📚 Vector DB: Found {len(vector_results)} semantic chunks")
        except Exception as e:
            logger.warning(f"Vector DB search failed: {e}")
        
        # Step 2: Query HybridLearningSystem for structured knowledge
        try:
            from modules.hybrid_learning_system import get_hybrid_system
            hybrid_system = get_hybrid_system()
            
            # Query formulas by domain/category
            try:
                all_formulas = hybrid_system.query_formulas()
                # Filter by category relevance
                relevant_formulas = [f for f in all_formulas if 
                                    intent.category in f.get('domain', '').lower() or
                                    any(comp in f.get('name', '').lower() for comp in intent.components)]
                knowledge["formulas"] = relevant_formulas if relevant_formulas else all_formulas[:20]
                logger.info(f"🔢 Formulas: Found {len(knowledge['formulas'])} relevant formulas")
            except Exception as e:
                logger.warning(f"Formula query failed: {e}")
            
            # Query materials matching design intent
            try:
                all_materials = hybrid_system.query_materials()
                # Filter by materials mentioned in intent
                relevant_materials = [m for m in all_materials if 
                                     any(mat.lower() in m.get('material_name', '').lower() 
                                         for mat in intent.materials)]
                knowledge["materials"] = relevant_materials if relevant_materials else all_materials[:10]
                logger.info(f"🏗️ Materials: Found {len(knowledge['materials'])} relevant materials")
            except Exception as e:
                logger.warning(f"Material query failed: {e}")
            
            # Query design rules by category
            try:
                design_rules = hybrid_system.query_design_rules(category=intent.category)
                knowledge["design_rules"] = design_rules[:15]  # Top 15 rules
                logger.info(f"📏 Design Rules: Found {len(knowledge['design_rules'])} relevant rules")
            except Exception as e:
                logger.warning(f"Design rules query failed: {e}")
            
            # Query code requirements
            try:
                if intent.category in ["architecture", "building"]:
                    code_reqs = hybrid_system.query_code_requirements(code_type="building")
                elif intent.category in ["vehicle", "aircraft"]:
                    code_reqs = hybrid_system.query_code_requirements(code_type="safety")
                else:
                    code_reqs = hybrid_system.query_code_requirements()
                knowledge["code_requirements"] = code_reqs[:10]  # Top 10 codes
                logger.info(f"⚖️ Code Requirements: Found {len(knowledge['code_requirements'])} relevant codes")
            except Exception as e:
                logger.warning(f"Code requirements query failed: {e}")
            
            # Summary log
            total_knowledge = (len(knowledge.get("formulas", [])) + 
                             len(knowledge.get("materials", [])) + 
                             len(knowledge.get("design_rules", [])) + 
                             len(knowledge.get("code_requirements", [])) +
                             len(knowledge.get("semantic_context", [])))
            logger.info(f"✅ Total knowledge retrieved: {total_knowledge} items")
            
        except Exception as e:
            logger.warning(f"HybridLearningSystem integration failed: {e}")
        
        # Ensure all keys exist
        for key in ["formulas", "materials", "design_rules", "code_requirements", "semantic_context"]:
            if key not in knowledge:
                knowledge[key] = []
        
        return knowledge

    async def _generate_design_components(self, intent: DesignIntent, knowledge: Dict[str, Any]) -> List[DesignComponent]:
        """Generate detailed design components"""

        components = []

        for component_name in intent.components:
            # Use LLM to generate detailed component specifications with creative engineering
            prompt = f"""
            As a creative engineering specialist, design an innovative {component_name} component for a {intent.complexity} {intent.category} system.

            Context:
            - Category: {intent.category}
            - Scale: {intent.scale}
            - Materials: {', '.join(intent.materials)}
            - Constraints: {', '.join(intent.constraints)}
            - Environment: {getattr(intent, 'environment', 'general')}
            - Power: {getattr(intent, 'power_source', 'standard')}

            Think innovatively:
            1. Consider emerging technologies and materials
            2. Optimize for multi-functionality and efficiency
            3. Ensure seamless integration with other components
            4. Include smart features and adaptability
            5. Balance cost, performance, and sustainability

            Provide detailed specifications including:
            1. Primary function (be specific and innovative)
            2. Key requirements (performance metrics, standards)
            3. Interface points with other components (data, power, mechanical)
            4. Material recommendations (with reasoning for choices)
            5. Rough dimensional estimates (with tolerances)
            6. Complexity assessment (and why this level is appropriate)

            Example for "chassis" in robotics:
            {{
                "function": "Modular, adaptive structural framework with integrated sensing and self-repair capabilities",
                "requirements": ["Load capacity: 50kg", "Modular expansion ports", "Environmental sealing IP67", "Self-diagnostic sensors"],
                "interfaces": ["Power distribution bus", "CAN bus for control", "Mechanical mounting points", "Thermal management"],
                "materials": ["Carbon fiber composite", "Titanium alloy joints", "Smart polymers for flexibility"],
                "dimensions": {{"length": 0.8, "width": 0.6, "height": 0.4, "tolerance": 0.01}},
                "complexity": "advanced"
            }}

            Format as JSON with fields: function, requirements, interfaces, materials, dimensions, complexity
            """

            response = await self.llm_engine.generate(prompt, max_new_tokens=400)

            try:
                comp_data = json.loads(response)
                component = DesignComponent(
                    name=component_name,
                    function=comp_data.get("function", f"Provides {component_name} functionality"),
                    requirements=comp_data.get("requirements", []),
                    interfaces=comp_data.get("interfaces", []),
                    materials=comp_data.get("materials", intent.materials),
                    dimensions=comp_data.get("dimensions", {}),
                    complexity=comp_data.get("complexity", "moderate")
                )
                components.append(component)
            except:
                # Fallback component
                component = DesignComponent(
                    name=component_name,
                    function=f"Provides {component_name} functionality",
                    requirements=["To be determined"],
                    interfaces=["Standard interfaces"],
                    materials=intent.materials,
                    dimensions={"length": 1.0, "width": 1.0, "height": 1.0},
                    complexity="moderate"
                )
                components.append(component)

        return components

    async def _calculate_system_requirements(self, intent: DesignIntent, components: List[DesignComponent]) -> Dict[str, Any]:
        """Calculate overall system requirements"""

        # Aggregate requirements from all components
        total_weight = sum(comp.dimensions.get("weight", 1.0) for comp in components)
        total_power = sum(comp.dimensions.get("power_consumption", 10.0) for comp in components)

        requirements = {
            "total_weight_kg": total_weight,
            "total_power_watts": total_power,
            "material_list": list(set(material for comp in components for material in comp.materials)),
            "complexity_level": intent.complexity,
            "estimated_cost": self._estimate_cost(intent, components),
            "build_time_days": self._estimate_build_time(intent, components)
        }

        return requirements

    async def _generate_design_parameters(self, intent: DesignIntent, components: List[DesignComponent]) -> Dict[str, Any]:
        """Generate design parameters and specifications"""

        parameters = {
            "overall_dimensions": {
                "length": max(comp.dimensions.get("length", 1.0) for comp in components),
                "width": max(comp.dimensions.get("width", 1.0) for comp in components),
                "height": sum(comp.dimensions.get("height", 1.0) for comp in components)
            },
            "performance_specs": {
                "efficiency": 0.85,  # 85% efficiency
                "reliability": 0.95,  # 95% reliability
                "maintainability": 0.8  # 80% maintainability
            },
            "operational_limits": {
                "temperature_range": "-20°C to 50°C",
                "humidity_range": "10% to 90%",
                "operational_lifetime": "5 years"
            }
        }

        return parameters

    async def _generate_validation_checks(self, intent: DesignIntent, components: List[DesignComponent]) -> List[str]:
        """Generate validation checks for the design"""

        checks = [
            f"Verify all {intent.category} safety standards are met",
            f"Check {intent.complexity} complexity constraints",
            f"Validate material compatibility: {', '.join(intent.materials)}",
            f"Ensure component interfaces are compatible",
            f"Verify dimensional constraints fit within {intent.scale} scale",
            "Perform basic structural integrity analysis",
            "Check for manufacturing feasibility",
            "Validate cost estimates against budget"
        ]

        return checks

    def _estimate_cost(self, intent: DesignIntent, components: List[DesignComponent]) -> float:
        """Estimate total design and build cost"""
        base_costs = {
            "simple": 1000,
            "moderate": 5000,
            "complex": 25000,
            "advanced": 100000
        }

        scale_multipliers = {
            "small": 0.5,
            "medium": 1.0,
            "large": 2.0,
            "massive": 5.0
        }

        base_cost = base_costs.get(intent.complexity, 5000)
        scale_multiplier = scale_multipliers.get(intent.scale, 1.0)
        component_multiplier = len(components) * 0.1 + 1.0

        return base_cost * scale_multiplier * component_multiplier

    def _estimate_build_time(self, intent: DesignIntent, components: List[DesignComponent]) -> int:
        """Estimate build time in days"""
        base_times = {
            "simple": 7,
            "moderate": 30,
            "complex": 90,
            "advanced": 180
        }

        complexity_multiplier = {
            "small": 0.5,
            "medium": 1.0,
            "large": 1.5,
            "massive": 2.0
        }

        base_time = base_times.get(intent.complexity, 30)
        scale_multiplier = complexity_multiplier.get(intent.scale, 1.0)

        return int(base_time * scale_multiplier)

    async def _save_design_to_memory(self, blueprint: DesignBlueprint):
        """Save design to persistent memory"""

        memory_entry = {
            "id": blueprint.id,
            "timestamp": blueprint.timestamp,
            "intent": asdict(blueprint.intent),
            "components": [asdict(comp) for comp in blueprint.components],
            "system_requirements": blueprint.system_requirements,
            "design_parameters": blueprint.design_parameters,
            "validation_checks": blueprint.validation_checks
        }

        # Save to design memory file
        memory_file = Path("data/design_memory.json")
        memory_file.parent.mkdir(exist_ok=True)

        # Load existing memory
        if memory_file.exists():
            with open(memory_file, 'r') as f:
                existing_memory = json.load(f)
        else:
            existing_memory = {"designs": []}

        # Add new design
        existing_memory["designs"].append(memory_entry)

        # Keep only last 100 designs
        existing_memory["designs"] = existing_memory["designs"][-100:]

        # Save back to file
        with open(memory_file, 'w') as f:
            json.dump(existing_memory, f, indent=2)

    def _load_design_memory(self) -> Dict[str, Any]:
        """Load existing design memory"""
        memory_file = Path("data/design_memory.json")
        if memory_file.exists():
            try:
                with open(memory_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {"designs": []}

    def _load_engineering_knowledge(self) -> Dict[str, Any]:
        """Load engineering knowledge base"""
        return {
            "architecture": {
                "materials": ["concrete", "steel", "wood", "glass"],
                "standards": ["building codes", "structural integrity"],
                "considerations": ["load bearing", "environmental factors"]
            },
            "robotics": {
                "materials": ["aluminum", "carbon fiber", "plastic"],
                "standards": ["safety standards", "precision requirements"],
                "considerations": ["degrees of freedom", "power efficiency"]
            },
            "vehicle": {
                "materials": ["steel", "aluminum", "carbon fiber"],
                "standards": ["safety standards", "emissions"],
                "considerations": ["aerodynamics", "weight distribution"]
            },
            "machine": {
                "materials": ["steel", "cast iron", "aluminum"],
                "standards": ["safety standards", "efficiency"],
                "considerations": ["durability", "maintenance"]
            }
        }

    async def get_similar_designs(self, intent: DesignIntent, limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar designs from memory"""
        memory = self._load_design_memory()
        designs = memory.get("designs", [])

        # Simple similarity scoring based on category and components
        similar_designs = []
        for design in designs:
            if design["intent"]["category"] == intent.category:
                score = len(set(design["intent"]["components"]) & set(intent.components))
                similar_designs.append((design, score))

        # Sort by similarity score
        similar_designs.sort(key=lambda x: x[1], reverse=True)

        return [design for design, score in similar_designs[:limit]]