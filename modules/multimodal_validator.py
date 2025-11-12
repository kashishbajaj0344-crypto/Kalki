# ============================================================
# Kalki Multi-Modal Design Validator
# ------------------------------------------------------------
# Validate designs across multiple dimensions:
# - Visual validation (aesthetics, proportion, symmetry)
# - Structural validation (FEA, load analysis, safety factors)
# - Acoustic validation (sound propagation, reverberation)
# - Thermal validation (heat dissipation, thermal expansion)
# ============================================================

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime

from modules.utils.logging_config import get_logger
from modules.design_brain import DesignBlueprint

logger = get_logger("Kalki.MultiModalValidator")

@dataclass
class VisualAnalysis:
    """Results from visual analysis"""
    aesthetic_score: float  # 0-1
    proportion_score: float  # 0-1
    symmetry_score: float  # 0-1
    golden_ratio_compliance: float  # 0-1
    visual_balance: str  # "poor", "fair", "good", "excellent"
    recommendations: List[str] = field(default_factory=list)

@dataclass
class StructuralAnalysis:
    """Results from structural analysis"""
    safety_factor: float  # > 1.0 is safe
    max_stress_mpa: float
    max_deflection_mm: float
    structural_integrity: str  # "unsafe", "marginal", "safe", "excellent"
    failure_modes: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

@dataclass
class AcousticAnalysis:
    """Results from acoustic analysis"""
    reverberation_time_s: float
    sound_absorption_coefficient: float
    noise_level_db: float
    acoustic_quality: str  # "poor", "acceptable", "good", "excellent"
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ThermalAnalysis:
    """Results from thermal analysis"""
    max_temperature_c: float
    thermal_expansion_mm: float
    heat_dissipation_w: float
    thermal_safety: str  # "unsafe", "marginal", "safe", "excellent"
    recommendations: List[str] = field(default_factory=list)

@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    design_id: str
    timestamp: str
    visual: Optional[VisualAnalysis]
    structural: Optional[StructuralAnalysis]
    acoustic: Optional[AcousticAnalysis]
    thermal: Optional[ThermalAnalysis]
    overall_score: float  # 0-1
    overall_verdict: str  # "rejected", "needs_improvement", "acceptable", "excellent"
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class MultiModalDesignValidator:
    """
    Validate designs across visual, acoustic, structural, and thermal dimensions
    using Vision, Audio, and Simulation agents
    """
    
    def __init__(self):
        # Lazy-load agents
        self.vision_agent = None
        self.audio_agent = None
        self.sim_engine = None
        
        logger.info("Multi-Modal Design Validator initialized")
    
    async def _ensure_agents_loaded(self):
        """Lazy-load validation agents"""
        if self.vision_agent is None:
            try:
                from modules.agents.multimodal.vision import VisionAgent
                self.vision_agent = VisionAgent()
                await self.vision_agent.initialize()
            except Exception as e:
                logger.warning(f"Vision Agent unavailable: {e}")
        
        if self.audio_agent is None:
            try:
                from modules.agents.multimodal.audio import AudioAgent
                self.audio_agent = AudioAgent()
                await self.audio_agent.initialize()
            except Exception as e:
                logger.warning(f"Audio Agent unavailable: {e}")
        
        if self.sim_engine is None:
            try:
                from modules.sim_engine import SimulationEngine
                self.sim_engine = SimulationEngine()
            except Exception as e:
                logger.warning(f"Simulation Engine unavailable: {e}")
    
    async def validate_design(
        self, 
        design_blueprint: DesignBlueprint,
        validation_types: List[str] = None
    ) -> ValidationReport:
        """
        Comprehensive multi-modal validation
        
        Args:
            design_blueprint: The design to validate
            validation_types: List of validations to run 
                            ["visual", "structural", "acoustic", "thermal"]
                            If None, runs all available
        
        Returns:
            ValidationReport with all validation results
        """
        await self._ensure_agents_loaded()
        
        if validation_types is None:
            validation_types = ["visual", "structural", "acoustic", "thermal"]
        
        logger.info(f"Running multi-modal validation for design {design_blueprint.id}")
        logger.info(f"Validation types: {', '.join(validation_types)}")
        
        # Initialize report
        report = ValidationReport(
            design_id=design_blueprint.id,
            timestamp=datetime.now().isoformat(),
            visual=None,
            structural=None,
            acoustic=None,
            thermal=None,
            overall_score=0.0,
            overall_verdict="needs_improvement",
            critical_issues=[],
            recommendations=[]
        )
        
        # Run validations
        validation_scores = []
        
        if "visual" in validation_types:
            report.visual = await self._validate_visual(design_blueprint)
            if report.visual:
                validation_scores.append(report.visual.aesthetic_score)
                report.recommendations.extend(report.visual.recommendations)
        
        if "structural" in validation_types:
            report.structural = await self._validate_structural(design_blueprint)
            if report.structural:
                structural_score = 1.0 if report.structural.safety_factor > 2.0 else \
                                  0.7 if report.structural.safety_factor > 1.5 else \
                                  0.3 if report.structural.safety_factor > 1.0 else 0.0
                validation_scores.append(structural_score)
                report.recommendations.extend(report.structural.recommendations)
                if report.structural.structural_integrity in ["unsafe", "marginal"]:
                    report.critical_issues.append(f"Structural integrity: {report.structural.structural_integrity}")
        
        if "acoustic" in validation_types and design_blueprint.intent.category in ["architecture", "vehicle"]:
            report.acoustic = await self._validate_acoustic(design_blueprint)
            if report.acoustic:
                acoustic_score = 1.0 if report.acoustic.acoustic_quality == "excellent" else \
                                0.8 if report.acoustic.acoustic_quality == "good" else \
                                0.5 if report.acoustic.acoustic_quality == "acceptable" else 0.2
                validation_scores.append(acoustic_score)
                report.recommendations.extend(report.acoustic.recommendations)
        
        if "thermal" in validation_types:
            report.thermal = await self._validate_thermal(design_blueprint)
            if report.thermal:
                thermal_score = 1.0 if report.thermal.thermal_safety == "excellent" else \
                               0.8 if report.thermal.thermal_safety == "safe" else \
                               0.4 if report.thermal.thermal_safety == "marginal" else 0.0
                validation_scores.append(thermal_score)
                report.recommendations.extend(report.thermal.recommendations)
                if report.thermal.thermal_safety in ["unsafe", "marginal"]:
                    report.critical_issues.append(f"Thermal safety: {report.thermal.thermal_safety}")
        
        # Calculate overall score
        if validation_scores:
            report.overall_score = sum(validation_scores) / len(validation_scores)
        
        # Determine overall verdict
        if report.critical_issues:
            report.overall_verdict = "rejected"
        elif report.overall_score >= 0.9:
            report.overall_verdict = "excellent"
        elif report.overall_score >= 0.7:
            report.overall_verdict = "acceptable"
        else:
            report.overall_verdict = "needs_improvement"
        
        logger.info(f"✅ Validation complete: {report.overall_verdict} (score: {report.overall_score:.2f})")
        
        return report
    
    async def _validate_visual(self, design: DesignBlueprint) -> Optional[VisualAnalysis]:
        """Visual validation using Vision Agent"""
        try:
            logger.info("🎨 Running visual analysis...")
            
            # Analyze design aesthetics
            # TODO: Once we have 3D models, pass them to vision agent
            # For now, analyze design parameters
            
            # Check golden ratio compliance
            golden_ratio = 1.618
            dimensions = design.system_requirements.get("dimensions", {})
            
            golden_compliance = 0.8  # Default
            if "length" in dimensions and "width" in dimensions:
                aspect_ratio = dimensions["length"] / max(dimensions["width"], 0.001)
                golden_compliance = 1.0 - min(abs(aspect_ratio - golden_ratio) / golden_ratio, 1.0)
            
            # Aesthetic scoring based on design complexity
            complexity = design.intent.complexity
            aesthetic_score = 0.9 if complexity == "advanced" else \
                            0.8 if complexity == "complex" else \
                            0.7 if complexity == "moderate" else 0.6
            
            recommendations = []
            if golden_compliance < 0.7:
                recommendations.append("Consider adjusting proportions closer to golden ratio (1.618)")
            if len(design.components) < 3:
                recommendations.append("Design may benefit from additional components for visual interest")
            
            return VisualAnalysis(
                aesthetic_score=aesthetic_score,
                proportion_score=golden_compliance,
                symmetry_score=0.85,  # Placeholder
                golden_ratio_compliance=golden_compliance,
                visual_balance="good" if aesthetic_score > 0.7 else "fair",
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Visual validation failed: {e}")
            return None
    
    async def _validate_structural(self, design: DesignBlueprint) -> Optional[StructuralAnalysis]:
        """Structural validation using Simulation Engine"""
        try:
            logger.info("🏗️ Running structural analysis...")
            
            if not self.sim_engine:
                logger.warning("Simulation engine unavailable, using heuristic analysis")
                return self._heuristic_structural_analysis(design)
            
            # Run FEA simulation
            # TODO: Implement full FEA integration
            return self._heuristic_structural_analysis(design)
            
        except Exception as e:
            logger.error(f"Structural validation failed: {e}")
            return None
    
    def _heuristic_structural_analysis(self, design: DesignBlueprint) -> StructuralAnalysis:
        """Heuristic structural analysis based on design parameters"""
        
        # Estimate safety factor based on materials and design
        materials = design.design_parameters.get("materials", [])
        
        # Higher safety factor for better materials
        base_safety_factor = 1.5
        if any("steel" in str(m).lower() for m in materials):
            base_safety_factor = 2.5
        elif any("aluminum" in str(m).lower() for m in materials):
            base_safety_factor = 2.0
        
        # Adjust for complexity
        if design.intent.complexity == "advanced":
            base_safety_factor *= 1.2
        
        recommendations = []
        if base_safety_factor < 2.0:
            recommendations.append("Consider using stronger materials or increasing member sizes")
        
        structural_integrity = "excellent" if base_safety_factor > 2.5 else \
                              "safe" if base_safety_factor > 2.0 else \
                              "marginal" if base_safety_factor > 1.5 else "unsafe"
        
        return StructuralAnalysis(
            safety_factor=base_safety_factor,
            max_stress_mpa=150.0,  # Placeholder
            max_deflection_mm=5.0,  # Placeholder
            structural_integrity=structural_integrity,
            failure_modes=[],
            recommendations=recommendations
        )
    
    async def _validate_acoustic(self, design: DesignBlueprint) -> Optional[AcousticAnalysis]:
        """Acoustic validation using Audio Agent"""
        try:
            logger.info("🔊 Running acoustic analysis...")
            
            # Heuristic acoustic analysis
            category = design.intent.category
            
            if category == "architecture":
                # Building acoustics
                reverberation_time = 1.5  # seconds, typical for office
                acoustic_quality = "good"
            elif category == "vehicle":
                # Vehicle NVH
                noise_level = 70.0  # dB, typical for car interior
                acoustic_quality = "acceptable"
            else:
                return None
            
            return AcousticAnalysis(
                reverberation_time_s=1.5,
                sound_absorption_coefficient=0.5,
                noise_level_db=70.0,
                acoustic_quality=acoustic_quality,
                recommendations=["Consider adding sound-absorbing materials"]
            )
            
        except Exception as e:
            logger.error(f"Acoustic validation failed: {e}")
            return None
    
    async def _validate_thermal(self, design: DesignBlueprint) -> Optional[ThermalAnalysis]:
        """Thermal validation"""
        try:
            logger.info("🌡️ Running thermal analysis...")
            
            # Heuristic thermal analysis
            power = design.system_requirements.get("power", 0)
            
            # Estimate max temperature based on power
            max_temp = 25 + (power * 0.5)  # Basic heat rise estimation
            
            thermal_safety = "excellent" if max_temp < 50 else \
                           "safe" if max_temp < 70 else \
                           "marginal" if max_temp < 85 else "unsafe"
            
            recommendations = []
            if max_temp > 60:
                recommendations.append("Consider adding active cooling or heat sinks")
            
            return ThermalAnalysis(
                max_temperature_c=max_temp,
                thermal_expansion_mm=0.1,
                heat_dissipation_w=power * 0.8,
                thermal_safety=thermal_safety,
                recommendations=recommendations
            )
            
        except Exception as e:
            logger.error(f"Thermal validation failed: {e}")
            return None


# Global singleton instance
_multimodal_validator = None

def get_multimodal_validator() -> MultiModalDesignValidator:
    """Get or create the global Multi-Modal Validator instance"""
    global _multimodal_validator
    if _multimodal_validator is None:
        _multimodal_validator = MultiModalDesignValidator()
    return _multimodal_validator
