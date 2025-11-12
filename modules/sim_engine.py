# ============================================================
# Kalki v2.4 — sim_engine.py
# ------------------------------------------------------------
# Simulation Engine: Physics & Engineering Analysis
# - Structural analysis (FEA)
# - Fluid dynamics simulation
# - Thermal analysis
# - Motion simulation
# - Performance validation
# ============================================================

import os
import json
import asyncio
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import math

from modules.utils.config import get_config
from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.SimEngine")

@dataclass
class SimulationResult:
    """Results from a simulation run"""
    simulation_id: str
    simulation_type: str
    status: str  # "success", "failed", "running"
    results: Dict[str, Any]
    metrics: Dict[str, float]
    timestamp: str

@dataclass
class PhysicsBody:
    """Physical body for simulation"""
    id: str
    mass: float
    dimensions: Dict[str, float]
    material_properties: Dict[str, Any]
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float] = (0, 0, 0)

@dataclass
class SimulationScenario:
    """Simulation scenario specification"""
    name: str
    description: str
    physics_bodies: List[PhysicsBody]
    environmental_conditions: Dict[str, Any]
    time_duration: float
    time_step: float

class SimulationEngine:
    """Physics and engineering simulation engine"""

    def __init__(self):
        self.simulations = {}
        self.templates = self._load_simulation_templates()

    def _load_simulation_templates(self) -> Dict[str, Any]:
        """Load simulation templates for different analysis types"""
        return {
            "structural": {
                "description": "Finite Element Analysis for structural integrity",
                "parameters": {
                    "material_youngs_modulus": 200e9,  # Pa
                    "material_poisson_ratio": 0.3,
                    "load_cases": ["static", "dynamic", "fatigue"],
                    "safety_factors": [1.5, 2.0, 3.0]
                }
            },
            "thermal": {
                "description": "Heat transfer and thermal analysis",
                "parameters": {
                    "thermal_conductivity": 50.0,  # W/m·K
                    "specific_heat": 500.0,  # J/kg·K
                    "ambient_temperature": 293.15,  # K
                    "heat_transfer_coefficient": 10.0  # W/m²·K
                }
            },
            "fluid": {
                "description": "Computational Fluid Dynamics",
                "parameters": {
                    "fluid_density": 1.225,  # kg/m³ (air)
                    "fluid_viscosity": 1.81e-5,  # Pa·s
                    "flow_velocity": 10.0,  # m/s
                    "turbulence_model": "k-epsilon"
                }
            },
            "motion": {
                "description": "Kinematics and dynamics simulation",
                "parameters": {
                    "gravity": 9.81,  # m/s²
                    "friction_coefficient": 0.3,
                    "time_step": 0.01,  # s
                    "simulation_time": 10.0  # s
                }
            }
        }

    async def run_structural_analysis(self, design_blueprint: Dict[str, Any]) -> SimulationResult:
        """Run structural finite element analysis"""

        sim_id = f"structural_{design_blueprint['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting structural analysis: {sim_id}")

        # Extract design parameters
        components = design_blueprint["components"]
        system_reqs = design_blueprint["system_requirements"]

        # Simplified FEA simulation
        results = await self._simulate_structural_fea(components, system_reqs)

        # Calculate safety factors
        safety_factors = self._calculate_safety_factors(results)

        # Generate recommendations
        recommendations = self._generate_structural_recommendations(results, safety_factors)

        simulation_result = SimulationResult(
            simulation_id=sim_id,
            simulation_type="structural",
            status="success",
            results={
                "stress_analysis": results,
                "safety_factors": safety_factors,
                "recommendations": recommendations,
                "critical_components": self._identify_critical_components(results)
            },
            metrics={
                "max_stress": results["max_stress"],
                "min_safety_factor": min(safety_factors.values()),
                "total_deformation": results["total_deformation"],
                "analysis_time": 2.5  # seconds
            },
            timestamp=datetime.now().isoformat()
        )

        self.simulations[sim_id] = simulation_result
        return simulation_result

    async def _simulate_structural_fea(self, components: List[Dict[str, Any]], system_reqs: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified structural FEA simulation"""

        # Calculate total mass and load
        total_mass = system_reqs["total_weight_kg"]
        gravity_load = total_mass * 9.81  # N

        # Material properties (simplified)
        youngs_modulus = 200e9  # Pa (steel)
        yield_strength = 250e6  # Pa

        # Calculate stresses for each component
        component_stresses = {}
        max_stress = 0

        for component in components:
            dims = component.get("dimensions", {})
            volume = dims.get("length", 1) * dims.get("width", 1) * dims.get("height", 1) * 1e-9  # m³
            cross_section = min(dims.get("width", 1), dims.get("height", 1)) * 1e-3  # m²

            # Simplified stress calculation
            stress = gravity_load / cross_section if cross_section > 0 else 0
            component_stresses[component["name"]] = stress
            max_stress = max(max_stress, stress)

        # Calculate deformation
        total_deformation = (gravity_load * 1.0) / (youngs_modulus * 0.01)  # Simplified

        return {
            "component_stresses": component_stresses,
            "max_stress": max_stress,
            "yield_strength": yield_strength,
            "total_deformation": total_deformation,
            "material_utilization": max_stress / yield_strength if yield_strength > 0 else 0
        }

    def _calculate_safety_factors(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate safety factors for different failure modes"""

        yield_strength = results["yield_strength"]
        max_stress = results["max_stress"]

        return {
            "yield_safety": yield_strength / max_stress if max_stress > 0 else float('inf'),
            "ultimate_safety": (yield_strength * 1.5) / max_stress if max_stress > 0 else float('inf'),
            "fatigue_safety": (yield_strength * 0.5) / max_stress if max_stress > 0 else float('inf')
        }

    def _generate_structural_recommendations(self, results: Dict[str, Any], safety_factors: Dict[str, float]) -> List[str]:
        """Generate structural improvement recommendations"""

        recommendations = []

        min_safety = min(safety_factors.values())

        if min_safety < 1.5:
            recommendations.append("Increase material cross-sections to improve safety factors")
        if results["total_deformation"] > 0.01:
            recommendations.append("Add structural reinforcements to reduce deformation")
        if results["material_utilization"] > 0.8:
            recommendations.append("Consider using higher strength materials")

        return recommendations

    def _identify_critical_components(self, results: Dict[str, Any]) -> List[str]:
        """Identify components with highest stress concentrations"""

        component_stresses = results["component_stresses"]
        sorted_components = sorted(component_stresses.items(), key=lambda x: x[1], reverse=True)

        return [comp[0] for comp in sorted_components[:3]]  # Top 3 critical components

    async def run_thermal_analysis(self, design_blueprint: Dict[str, Any]) -> SimulationResult:
        """Run thermal analysis simulation"""

        sim_id = f"thermal_{design_blueprint['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting thermal analysis: {sim_id}")

        components = design_blueprint["components"]
        system_reqs = design_blueprint["system_requirements"]

        # Simplified thermal simulation
        results = await self._simulate_thermal_analysis(components, system_reqs)

        simulation_result = SimulationResult(
            simulation_id=sim_id,
            simulation_type="thermal",
            status="success",
            results={
                "temperature_distribution": results["temperatures"],
                "heat_flux": results["heat_flux"],
                "thermal_stress": results["thermal_stress"]
            },
            metrics={
                "max_temperature": results["max_temp"],
                "min_temperature": results["min_temp"],
                "thermal_gradient": results["thermal_gradient"],
                "analysis_time": 1.8
            },
            timestamp=datetime.now().isoformat()
        )

        self.simulations[sim_id] = simulation_result
        return simulation_result

    async def _simulate_thermal_analysis(self, components: List[Dict[str, Any]], system_reqs: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified thermal analysis"""

        power_consumption = system_reqs["total_power_watts"]
        ambient_temp = 293.15  # K (20°C)

        # Calculate temperatures for each component
        temperatures = {}
        max_temp = ambient_temp
        min_temp = ambient_temp

        for component in components:
            # Simplified heat generation based on power
            heat_generation = power_consumption / len(components)  # W

            # Surface area for convection
            dims = component.get("dimensions", {})
            surface_area = 2 * (dims.get("length", 1) * dims.get("width", 1) +
                              dims.get("length", 1) * dims.get("height", 1) +
                              dims.get("width", 1) * dims.get("height", 1)) * 1e-6  # m²

            # Simplified temperature rise
            temp_rise = heat_generation / (10 * surface_area) if surface_area > 0 else 0
            temp = ambient_temp + temp_rise

            temperatures[component["name"]] = temp
            max_temp = max(max_temp, temp)
            min_temp = min(min_temp, temp)

        thermal_gradient = max_temp - min_temp

        return {
            "temperatures": temperatures,
            "max_temp": max_temp,
            "min_temp": min_temp,
            "thermal_gradient": thermal_gradient,
            "heat_flux": power_consumption / 0.01,  # Simplified
            "thermal_stress": thermal_gradient * 1e7  # Simplified thermal stress
        }

    async def run_motion_simulation(self, design_blueprint: Dict[str, Any], scenario: str = "basic_motion") -> SimulationResult:
        """Run motion and dynamics simulation"""

        sim_id = f"motion_{design_blueprint['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting motion simulation: {sim_id}")

        components = design_blueprint["components"]

        # Create physics bodies
        physics_bodies = []
        for component in components:
            dims = component.get("dimensions", {})
            volume = dims.get("length", 1) * dims.get("width", 1) * dims.get("height", 1) * 1e-9  # m³
            mass = volume * 7800  # kg (steel density)

            physics_bodies.append(PhysicsBody(
                id=component["name"],
                mass=mass,
                dimensions=dims,
                material_properties={"density": 7800, "friction": 0.3},
                position=(0, 0, 0)
            ))

        # Run simulation
        results = await self._simulate_motion(physics_bodies, scenario)

        simulation_result = SimulationResult(
            simulation_id=sim_id,
            simulation_type="motion",
            status="success",
            results={
                "trajectories": results["trajectories"],
                "forces": results["forces"],
                "collisions": results["collisions"]
            },
            metrics={
                "simulation_time": results["sim_time"],
                "total_energy": results["total_energy"],
                "max_velocity": results["max_velocity"],
                "analysis_time": 3.2
            },
            timestamp=datetime.now().isoformat()
        )

        self.simulations[sim_id] = simulation_result
        return simulation_result

    async def _simulate_motion(self, bodies: List[PhysicsBody], scenario: str) -> Dict[str, Any]:
        """Simplified motion simulation"""

        # Time parameters
        dt = 0.01
        total_time = 5.0
        steps = int(total_time / dt)

        trajectories = {}
        forces = {}
        total_energy = 0
        max_velocity = 0

        # Initialize trajectories
        for body in bodies:
            trajectories[body.id] = [(body.position, body.velocity)]

        # Simple physics simulation
        for step in range(steps):
            for body in bodies:
                # Apply gravity
                gravity_force = (0, 0, -body.mass * 9.81)

                # Update velocity and position (simplified Euler integration)
                ax = gravity_force[0] / body.mass
                ay = gravity_force[1] / body.mass
                az = gravity_force[2] / body.mass

                vx, vy, vz = body.velocity
                vx += ax * dt
                vy += ay * dt
                vz += az * dt

                x, y, z = body.position
                x += vx * dt
                y += vy * dt
                z += vz * dt

                body.position = (x, y, z)
                body.velocity = (vx, vy, vz)

                # Track max velocity
                velocity_magnitude = math.sqrt(vx**2 + vy**2 + vz**2)
                max_velocity = max(max_velocity, velocity_magnitude)

                # Calculate energy
                kinetic_energy = 0.5 * body.mass * velocity_magnitude**2
                potential_energy = body.mass * 9.81 * z
                total_energy = max(total_energy, kinetic_energy + potential_energy)

                # Store trajectory point
                trajectories[body.id].append((body.position, body.velocity))

        return {
            "trajectories": trajectories,
            "forces": forces,
            "collisions": [],  # No collisions in this simple simulation
            "sim_time": total_time,
            "total_energy": total_energy,
            "max_velocity": max_velocity
        }

    async def run_fluid_dynamics(self, design_blueprint: Dict[str, Any]) -> SimulationResult:
        """Run fluid dynamics simulation"""

        sim_id = f"fluid_{design_blueprint['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"Starting fluid dynamics simulation: {sim_id}")

        # Simplified CFD simulation
        results = await self._simulate_cfd(design_blueprint)

        simulation_result = SimulationResult(
            simulation_id=sim_id,
            simulation_type="fluid",
            status="success",
            results={
                "velocity_field": results["velocity_field"],
                "pressure_distribution": results["pressure"],
                "drag_force": results["drag_force"]
            },
            metrics={
                "max_velocity": results["max_velocity"],
                "average_pressure": results["avg_pressure"],
                "drag_coefficient": results["drag_coeff"],
                "analysis_time": 4.1
            },
            timestamp=datetime.now().isoformat()
        )

        self.simulations[sim_id] = simulation_result
        return simulation_result

    async def _simulate_cfd(self, design_blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """Simplified CFD simulation"""

        # Simplified aerodynamic analysis
        dims = design_blueprint["design_parameters"]["overall_dimensions"]
        frontal_area = dims["width"] * dims["height"] * 1e-6  # m²

        # Air properties
        density = 1.225  # kg/m³
        velocity = 10.0  # m/s

        # Simplified drag calculation
        drag_coeff = 0.5  # Approximate for bluff body
        drag_force = 0.5 * density * velocity**2 * frontal_area * drag_coeff

        # Pressure distribution (simplified)
        dynamic_pressure = 0.5 * density * velocity**2
        pressure = dynamic_pressure * drag_coeff

        return {
            "velocity_field": {"inlet": velocity, "outlet": velocity * 0.8},
            "pressure": {"front": pressure, "back": pressure * 0.1},
            "drag_force": drag_force,
            "max_velocity": velocity,
            "avg_pressure": pressure * 0.5,
            "drag_coeff": drag_coeff
        }

    def get_simulation_history(self) -> List[SimulationResult]:
        """Get history of all simulations"""
        return list(self.simulations.values())

    def get_simulation_status(self, sim_id: str) -> Optional[SimulationResult]:
        """Get status of a specific simulation"""
        return self.simulations.get(sim_id)

    def export_simulation_results(self, sim_id: str, output_dir: str = "output/simulations") -> str:
        """Export simulation results to file"""

        if sim_id not in self.simulations:
            raise ValueError(f"Simulation {sim_id} not found")

        Path(output_dir).mkdir(parents=True, exist_ok=True)

        result = self.simulations[sim_id]
        output_file = f"{output_dir}/{sim_id}.json"

        with open(output_file, 'w') as f:
            json.dump({
                "simulation_id": result.simulation_id,
                "simulation_type": result.simulation_type,
                "status": result.status,
                "results": result.results,
                "metrics": result.metrics,
                "timestamp": result.timestamp
            }, f, indent=2)

        return output_file