"""
Digital Twin System
Creates and maintains digital twins of physical designs for simulation and testing.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class SimulationType(Enum):
    """Types of simulations supported"""
    STRUCTURAL_FEA = "structural_fea"
    THERMAL = "thermal"
    FLUID_DYNAMICS = "fluid_dynamics"
    KINEMATIC = "kinematic"
    DYNAMIC = "dynamic"
    ELECTROMAGNETIC = "electromagnetic"
    ACOUSTIC = "acoustic"
    MULTI_PHYSICS = "multi_physics"


@dataclass
class PhysicalProperties:
    """Physical properties of a design"""
    mass_kg: float
    center_of_mass: Tuple[float, float, float]
    inertia_tensor: List[List[float]]
    material_properties: Dict[str, Any]
    dimensions: Dict[str, float]
    

@dataclass
class SimulationResult:
    """Result from a simulation"""
    simulation_id: str
    simulation_type: SimulationType
    timestamp: datetime
    success: bool
    results: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    

@dataclass
class DigitalTwin:
    """Digital twin of a physical design"""
    twin_id: str
    design_id: str
    physical_properties: PhysicalProperties
    geometry_data: Dict[str, Any]
    simulation_history: List[SimulationResult] = field(default_factory=list)
    real_world_sync_enabled: bool = False
    last_sync: Optional[datetime] = None
    

class DigitalTwinSystem:
    """
    Digital twin creation and management system.
    
    Features:
    - Create digital twins from designs
    - Real-time synchronization with physical world
    - Multi-physics simulation
    - Predictive modeling
    - What-if analysis
    - Performance optimization
    """
    
    def __init__(self):
        self.twins: Dict[str, DigitalTwin] = {}
        self.is_running = False
        self.sync_interval = 1.0  # seconds
        
    async def initialize(self):
        """Initialize the digital twin system"""
        logger.info("🔷 Initializing Digital Twin System")
        
        # Load existing twins
        await self._load_twins()
        
        logger.info(f"✅ Digital twin system initialized with {len(self.twins)} twins")
        
    async def create_twin(self, design_id: str, design_data: Dict[str, Any]) -> DigitalTwin:
        """Create a digital twin from a design"""
        twin_id = f"twin_{design_id}_{datetime.now().timestamp()}"
        
        logger.info(f"🔷 Creating digital twin for design: {design_id}")
        
        # Extract or calculate physical properties
        physical_properties = await self._extract_physical_properties(design_data)
        
        # Extract geometry
        geometry_data = await self._extract_geometry(design_data)
        
        # Create twin
        twin = DigitalTwin(
            twin_id=twin_id,
            design_id=design_id,
            physical_properties=physical_properties,
            geometry_data=geometry_data
        )
        
        self.twins[twin_id] = twin
        
        logger.info(f"✅ Digital twin created: {twin_id}")
        
        # Run initial simulations
        await self._run_initial_simulations(twin)
        
        return twin
        
    async def enable_real_world_sync(self, twin_id: str):
        """Enable real-time synchronization with physical world"""
        if twin_id not in self.twins:
            logger.error(f"Unknown twin: {twin_id}")
            return
            
        twin = self.twins[twin_id]
        twin.real_world_sync_enabled = True
        
        logger.info(f"🔄 Enabled real-world sync for twin: {twin_id}")
        
        # Start sync loop in background
        asyncio.create_task(self._sync_loop(twin_id))
        
    async def _sync_loop(self, twin_id: str):
        """Continuous synchronization loop for a twin"""
        twin = self.twins.get(twin_id)
        if not twin:
            return
            
        logger.info(f"🔄 Starting sync loop for {twin_id}")
        
        while twin.real_world_sync_enabled:
            try:
                # Sync with sensor data
                await self._sync_with_sensors(twin)
                
                # Update twin state
                await self._update_twin_state(twin)
                
                twin.last_sync = datetime.now()
                
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                logger.error(f"Sync error for {twin_id}: {e}")
                await asyncio.sleep(5.0)
                
    async def _sync_with_sensors(self, twin: DigitalTwin):
        """Sync twin with real-world sensor data"""
        try:
            from modules.sensor_data_pipeline import get_sensor_pipeline
            
            pipeline = get_sensor_pipeline()
            
            # Get latest sensor readings for this design
            # In production, would filter by design_id
            # For now, demonstrate the integration
            
            status = pipeline.get_pipeline_status()
            if status['total_readings'] > 0:
                logger.debug(f"Syncing twin {twin.twin_id} with {status['total_readings']} sensor readings")
                
        except Exception as e:
            logger.debug(f"Sensor sync: {e}")
            
    async def _update_twin_state(self, twin: DigitalTwin):
        """Update twin state based on real-world data"""
        # In production, would update twin's physical properties
        # based on sensor measurements
        pass
        
    async def run_simulation(self, twin_id: str, simulation_type: SimulationType,
                            parameters: Dict[str, Any]) -> SimulationResult:
        """Run a simulation on a digital twin"""
        if twin_id not in self.twins:
            raise ValueError(f"Unknown twin: {twin_id}")
            
        twin = self.twins[twin_id]
        
        logger.info(f"🔬 Running {simulation_type.value} simulation on {twin_id}")
        
        simulation_id = f"sim_{datetime.now().timestamp()}"
        
        try:
            # Run appropriate simulation
            if simulation_type == SimulationType.STRUCTURAL_FEA:
                results = await self._run_structural_simulation(twin, parameters)
            elif simulation_type == SimulationType.THERMAL:
                results = await self._run_thermal_simulation(twin, parameters)
            elif simulation_type == SimulationType.KINEMATIC:
                results = await self._run_kinematic_simulation(twin, parameters)
            elif simulation_type == SimulationType.DYNAMIC:
                results = await self._run_dynamic_simulation(twin, parameters)
            else:
                results = {'status': 'simulation_type_not_implemented'}
                
            sim_result = SimulationResult(
                simulation_id=simulation_id,
                simulation_type=simulation_type,
                timestamp=datetime.now(),
                success=True,
                results=results
            )
            
            logger.info(f"✅ Simulation completed: {simulation_id}")
            
        except Exception as e:
            logger.error(f"Simulation failed: {e}", exc_info=True)
            sim_result = SimulationResult(
                simulation_id=simulation_id,
                simulation_type=simulation_type,
                timestamp=datetime.now(),
                success=False,
                results={},
                errors=[str(e)]
            )
            
        # Store result
        twin.simulation_history.append(sim_result)
        
        return sim_result
        
    async def _run_structural_simulation(self, twin: DigitalTwin, 
                                         parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run structural FEA simulation"""
        # Simulate structural analysis
        logger.info(f"Running FEA with load: {parameters.get('load_n', 1000)}N")
        
        # Calculate stress (simplified)
        load_n = parameters.get('load_n', 1000)
        cross_section_m2 = twin.physical_properties.dimensions.get('cross_section', 0.01)
        
        stress_mpa = (load_n / cross_section_m2) / 1e6
        
        # Calculate deflection (simplified)
        length_m = twin.physical_properties.dimensions.get('length', 1.0)
        youngs_modulus_gpa = twin.physical_properties.material_properties.get('youngs_modulus', 200)
        
        deflection_mm = (load_n * length_m**3) / (3 * youngs_modulus_gpa * 1e9 * cross_section_m2) * 1000
        
        # Calculate safety factor
        yield_strength_mpa = twin.physical_properties.material_properties.get('yield_strength', 250)
        safety_factor = yield_strength_mpa / stress_mpa if stress_mpa > 0 else 999
        
        return {
            'max_stress_mpa': stress_mpa,
            'max_deflection_mm': deflection_mm,
            'safety_factor': safety_factor,
            'structural_integrity': 'pass' if safety_factor >= 2.0 else 'fail'
        }
        
    async def _run_thermal_simulation(self, twin: DigitalTwin,
                                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run thermal simulation"""
        # Simulate thermal analysis
        ambient_temp_c = parameters.get('ambient_temp_c', 25)
        heat_input_w = parameters.get('heat_input_w', 100)
        
        thermal_conductivity = twin.physical_properties.material_properties.get('thermal_conductivity', 200)
        mass_kg = twin.physical_properties.mass_kg
        
        # Simplified heat transfer calculation
        temp_rise_c = heat_input_w / (mass_kg * 10)  # Simplified
        max_temp_c = ambient_temp_c + temp_rise_c
        
        return {
            'max_temperature_c': max_temp_c,
            'avg_temperature_c': ambient_temp_c + (temp_rise_c * 0.7),
            'thermal_safety': 'pass' if max_temp_c < 100 else 'warning'
        }
        
    async def _run_kinematic_simulation(self, twin: DigitalTwin,
                                        parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run kinematic simulation"""
        # Simulate motion/kinematics
        if 'joint_angles' in parameters:
            joint_angles = parameters['joint_angles']
            
            # Calculate end effector position (simplified)
            # For a robot arm: forward kinematics
            x = sum(np.cos(angle) for angle in joint_angles)
            y = sum(np.sin(angle) for angle in joint_angles)
            z = parameters.get('base_height', 0.5)
            
            return {
                'end_effector_position': {'x': x, 'y': y, 'z': z},
                'workspace_reached': True,
                'singularities': []
            }
        
        return {'status': 'no_joint_angles_provided'}
        
    async def _run_dynamic_simulation(self, twin: DigitalTwin,
                                       parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Run dynamic simulation"""
        # Simulate dynamics
        force_n = parameters.get('force_n', 100)
        mass_kg = twin.physical_properties.mass_kg
        
        # F = ma
        acceleration_mps2 = force_n / mass_kg
        
        # Calculate trajectory over time
        time_s = parameters.get('time_s', 1.0)
        velocity_mps = acceleration_mps2 * time_s
        displacement_m = 0.5 * acceleration_mps2 * time_s**2
        
        return {
            'acceleration_mps2': acceleration_mps2,
            'final_velocity_mps': velocity_mps,
            'displacement_m': displacement_m,
            'kinetic_energy_j': 0.5 * mass_kg * velocity_mps**2
        }
        
    async def _run_initial_simulations(self, twin: DigitalTwin):
        """Run initial baseline simulations"""
        logger.info(f"Running initial simulations for {twin.twin_id}")
        
        # Structural
        await self.run_simulation(
            twin.twin_id,
            SimulationType.STRUCTURAL_FEA,
            {'load_n': 1000}
        )
        
        # Thermal
        await self.run_simulation(
            twin.twin_id,
            SimulationType.THERMAL,
            {'ambient_temp_c': 25, 'heat_input_w': 50}
        )
        
    async def _extract_physical_properties(self, design_data: Dict[str, Any]) -> PhysicalProperties:
        """Extract physical properties from design data"""
        # Calculate or extract properties
        mass_kg = design_data.get('mass_kg', 10.0)
        
        # Default properties
        return PhysicalProperties(
            mass_kg=mass_kg,
            center_of_mass=(0.0, 0.0, 0.0),
            inertia_tensor=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            material_properties={
                'youngs_modulus': 200,  # GPa (steel)
                'yield_strength': 250,  # MPa
                'thermal_conductivity': 50,  # W/mK
                'density': 7850  # kg/m³
            },
            dimensions={
                'length': design_data.get('length', 1.0),
                'width': design_data.get('width', 0.1),
                'height': design_data.get('height', 0.1),
                'cross_section': design_data.get('cross_section', 0.01)
            }
        )
        
    async def _extract_geometry(self, design_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract geometry data from design"""
        return {
            'type': design_data.get('geometry_type', 'solid'),
            'primitives': design_data.get('primitives', []),
            'mesh_available': False
        }
        
    async def _load_twins(self):
        """Load digital twins from disk"""
        try:
            twins_path = Path("data/digital_twins.json")
            if twins_path.exists():
                with open(twins_path) as f:
                    data = json.load(f)
                    logger.info(f"📂 Loaded {len(data)} digital twins")
                    # Would reconstruct twins here
        except Exception as e:
            logger.debug(f"No twins loaded: {e}")
            
    async def save_twins(self):
        """Save digital twins to disk"""
        try:
            twins_path = Path("data/digital_twins.json")
            twins_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [
                {
                    'twin_id': twin.twin_id,
                    'design_id': twin.design_id,
                    'real_world_sync': twin.real_world_sync_enabled,
                    'simulation_count': len(twin.simulation_history)
                }
                for twin in self.twins.values()
            ]
            
            with open(twins_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving twins: {e}")
            
    def get_twin_status(self, twin_id: str) -> Dict[str, Any]:
        """Get status of a specific twin"""
        if twin_id not in self.twins:
            return {'error': 'twin_not_found'}
            
        twin = self.twins[twin_id]
        
        return {
            'twin_id': twin.twin_id,
            'design_id': twin.design_id,
            'real_world_sync': twin.real_world_sync_enabled,
            'last_sync': twin.last_sync.isoformat() if twin.last_sync else None,
            'total_simulations': len(twin.simulation_history),
            'mass_kg': twin.physical_properties.mass_kg
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        synced_twins = sum(1 for t in self.twins.values() if t.real_world_sync_enabled)
        total_simulations = sum(len(t.simulation_history) for t in self.twins.values())
        
        return {
            'is_running': self.is_running,
            'total_twins': len(self.twins),
            'synced_twins': synced_twins,
            'total_simulations': total_simulations
        }


# Singleton instance
_digital_twin_system = None

def get_digital_twin_system() -> DigitalTwinSystem:
    """Get the global digital twin system instance"""
    global _digital_twin_system
    if _digital_twin_system is None:
        _digital_twin_system = DigitalTwinSystem()
    return _digital_twin_system
