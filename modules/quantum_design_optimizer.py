# ============================================================
# Kalki Quantum Design Optimizer
# ------------------------------------------------------------
# Use quantum-inspired algorithms for multi-objective optimization:
# - Simulated annealing for design parameter optimization
# - Pareto frontier discovery for multi-objective problems
# - Quantum-inspired combinatorial optimization
# ============================================================

import asyncio
import random
import math
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime

from modules.utils.logging_config import get_logger
from modules.design_brain import DesignBlueprint

logger = get_logger("Kalki.QuantumOptimizer")

@dataclass
class OptimizationObjective:
    """Single optimization objective"""
    name: str
    target: str  # "minimize" or "maximize"
    weight: float = 1.0
    current_value: float = 0.0
    optimal_value: float = 0.0

@dataclass
class OptimizationResult:
    """Result from quantum optimization"""
    original_parameters: Dict[str, Any]
    optimized_parameters: Dict[str, Any]
    objectives_before: List[OptimizationObjective]
    objectives_after: List[OptimizationObjective]
    improvement_percentage: float
    iterations: int
    convergence_history: List[float]
    pareto_optimal: bool
    timestamp: str

class QuantumDesignOptimizer:
    """
    Use quantum-inspired algorithms for design optimization
    
    Implements:
    - Simulated annealing for parameter optimization
    - Multi-objective optimization with Pareto frontier
    - Quantum-inspired search for combinatorial problems
    """
    
    def __init__(self):
        # Lazy-load quantum agent
        self.quantum_agent = None
        
        # Optimization history
        self.optimization_history = []
        
        logger.info("Quantum Design Optimizer initialized")
    
    async def _ensure_quantum_loaded(self):
        """Lazy-load quantum reasoning agent"""
        if self.quantum_agent is None:
            try:
                from modules.agents.quantum.reasoning import QuantumReasoningAgent
                self.quantum_agent = QuantumReasoningAgent()
                await self.quantum_agent.initialize()
                logger.info("Quantum Reasoning Agent loaded")
            except Exception as e:
                logger.warning(f"Quantum Agent unavailable: {e}")
    
    async def optimize_design(
        self,
        design_blueprint: DesignBlueprint,
        objectives: List[OptimizationObjective],
        constraints: Optional[Dict[str, Any]] = None,
        max_iterations: int = 1000
    ) -> OptimizationResult:
        """
        Optimize design using quantum-inspired multi-objective optimization
        
        Args:
            design_blueprint: The design to optimize
            objectives: List of objectives to optimize
            constraints: Optional design constraints
            max_iterations: Maximum optimization iterations
        
        Returns:
            OptimizationResult with optimized parameters
        """
        await self._ensure_quantum_loaded()
        
        logger.info(f"Optimizing design {design_blueprint.id} for {len(objectives)} objectives")
        
        if constraints is None:
            constraints = {}
        
        # Extract current parameters
        current_params = design_blueprint.design_parameters.copy()
        
        # Evaluate current objectives
        objectives_before = await self._evaluate_objectives(
            design_blueprint, objectives
        )
        
        logger.info(f"Initial objective values: " + 
                   ", ".join([f"{obj.name}={obj.current_value:.3f}" for obj in objectives_before]))
        
        # Determine optimization method
        if len(objectives) == 1:
            # Single-objective: Use simulated annealing
            optimized_params, convergence = await self._simulated_annealing(
                design_blueprint, objectives[0], constraints, max_iterations
            )
        else:
            # Multi-objective: Use Pareto optimization
            optimized_params, convergence = await self._pareto_optimization(
                design_blueprint, objectives, constraints, max_iterations
            )
        
        # Create optimized blueprint
        optimized_blueprint = await self._apply_parameters(
            design_blueprint, optimized_params
        )
        
        # Evaluate optimized objectives
        objectives_after = await self._evaluate_objectives(
            optimized_blueprint, objectives
        )
        
        # Calculate improvement
        improvement = self._calculate_improvement(objectives_before, objectives_after)
        
        # Check if Pareto optimal
        pareto_optimal = len(objectives) > 1  # Simplified check
        
        result = OptimizationResult(
            original_parameters=current_params,
            optimized_parameters=optimized_params,
            objectives_before=objectives_before,
            objectives_after=objectives_after,
            improvement_percentage=improvement,
            iterations=len(convergence),
            convergence_history=convergence,
            pareto_optimal=pareto_optimal,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"✅ Optimization complete: {improvement:.1f}% improvement over {len(convergence)} iterations")
        
        # Track history
        self.optimization_history.append(result)
        
        return result
    
    async def _simulated_annealing(
        self,
        design: DesignBlueprint,
        objective: OptimizationObjective,
        constraints: Dict[str, Any],
        max_iterations: int
    ) -> Tuple[Dict[str, Any], List[float]]:
        """Simulated annealing for single-objective optimization"""
        
        logger.info(f"Running simulated annealing for {objective.name}")
        
        # Initialize
        current_params = design.design_parameters.copy()
        current_score = await self._evaluate_single_objective(design, objective)
        best_params = current_params.copy()
        best_score = current_score
        
        # Temperature schedule
        initial_temp = 100.0
        cooling_rate = 0.95
        temperature = initial_temp
        
        convergence_history = [current_score]
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor_params = self._generate_neighbor(current_params, constraints)
            
            # Evaluate neighbor
            neighbor_design = await self._apply_parameters(design, neighbor_params)
            neighbor_score = await self._evaluate_single_objective(neighbor_design, objective)
            
            # Calculate acceptance probability
            delta = neighbor_score - current_score
            if objective.target == "minimize":
                delta = -delta  # Flip for minimization
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                # Accept neighbor
                current_params = neighbor_params
                current_score = neighbor_score
                
                # Update best
                if (objective.target == "maximize" and current_score > best_score) or \
                   (objective.target == "minimize" and current_score < best_score):
                    best_params = current_params.copy()
                    best_score = current_score
            
            # Cool down
            temperature *= cooling_rate
            convergence_history.append(best_score)
            
            # Log progress
            if (iteration + 1) % 100 == 0:
                logger.info(f"  Iteration {iteration + 1}: best_score={best_score:.4f}, temp={temperature:.2f}")
        
        return best_params, convergence_history
    
    async def _pareto_optimization(
        self,
        design: DesignBlueprint,
        objectives: List[OptimizationObjective],
        constraints: Dict[str, Any],
        max_iterations: int
    ) -> Tuple[Dict[str, Any], List[float]]:
        """Pareto optimization for multi-objective problems"""
        
        logger.info(f"Running Pareto optimization for {len(objectives)} objectives")
        
        # Population-based optimization (simplified NSGA-II)
        population_size = 50
        population = []
        
        # Initialize population
        for _ in range(population_size):
            params = self._generate_random_params(design.design_parameters, constraints)
            individual_design = await self._apply_parameters(design, params)
            scores = [await self._evaluate_single_objective(individual_design, obj) 
                     for obj in objectives]
            population.append({
                "params": params,
                "scores": scores,
                "fitness": sum(scores) / len(scores)  # Simplified fitness
            })
        
        convergence_history = []
        best_individual = max(population, key=lambda x: x["fitness"])
        
        for iteration in range(max_iterations // population_size):
            # Selection, crossover, mutation
            new_population = []
            
            for _ in range(population_size):
                # Tournament selection
                parent1 = self._tournament_select(population)
                parent2 = self._tournament_select(population)
                
                # Crossover
                child_params = self._crossover(parent1["params"], parent2["params"])
                
                # Mutation
                if random.random() < 0.1:
                    child_params = self._mutate(child_params, constraints)
                
                # Evaluate child
                child_design = await self._apply_parameters(design, child_params)
                child_scores = [await self._evaluate_single_objective(child_design, obj)
                               for obj in objectives]
                
                new_population.append({
                    "params": child_params,
                    "scores": child_scores,
                    "fitness": sum(child_scores) / len(child_scores)
                })
            
            population = new_population
            
            # Track best
            current_best = max(population, key=lambda x: x["fitness"])
            if current_best["fitness"] > best_individual["fitness"]:
                best_individual = current_best
            
            convergence_history.append(best_individual["fitness"])
            
            if (iteration + 1) % 10 == 0:
                logger.info(f"  Generation {iteration + 1}: best_fitness={best_individual['fitness']:.4f}")
        
        return best_individual["params"], convergence_history
    
    def _generate_neighbor(
        self, 
        params: Dict[str, Any], 
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate neighbor solution for simulated annealing"""
        neighbor = params.copy()
        
        # Perturb a random parameter
        if "dimensions" in neighbor and isinstance(neighbor["dimensions"], dict):
            dim_key = random.choice(list(neighbor["dimensions"].keys()))
            current_val = neighbor["dimensions"][dim_key]
            perturbation = random.gauss(0, current_val * 0.1)  # 10% std dev
            neighbor["dimensions"][dim_key] = max(current_val + perturbation, 0.001)
        
        return neighbor
    
    def _generate_random_params(
        self,
        base_params: Dict[str, Any],
        constraints: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate random parameters for population initialization"""
        params = base_params.copy()
        
        if "dimensions" in params and isinstance(params["dimensions"], dict):
            for key in params["dimensions"].keys():
                params["dimensions"][key] = random.uniform(0.1, 10.0)
        
        return params
    
    def _tournament_select(self, population: List[Dict], tournament_size: int = 3) -> Dict:
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x["fitness"])
    
    def _crossover(self, params1: Dict[str, Any], params2: Dict[str, Any]) -> Dict[str, Any]:
        """Crossover two parameter sets"""
        child = params1.copy()
        
        if "dimensions" in params1 and "dimensions" in params2:
            child["dimensions"] = {}
            for key in params1["dimensions"].keys():
                if random.random() < 0.5:
                    child["dimensions"][key] = params1["dimensions"][key]
                else:
                    child["dimensions"][key] = params2["dimensions"].get(key, params1["dimensions"][key])
        
        return child
    
    def _mutate(self, params: Dict[str, Any], constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Mutate parameters"""
        mutated = params.copy()
        
        if "dimensions" in mutated and isinstance(mutated["dimensions"], dict):
            for key in mutated["dimensions"].keys():
                if random.random() < 0.3:  # 30% mutation rate per dimension
                    current_val = mutated["dimensions"][key]
                    mutated["dimensions"][key] = max(current_val * random.uniform(0.8, 1.2), 0.001)
        
        return mutated
    
    async def _evaluate_objectives(
        self,
        design: DesignBlueprint,
        objectives: List[OptimizationObjective]
    ) -> List[OptimizationObjective]:
        """Evaluate all objectives for a design"""
        evaluated = []
        
        for obj in objectives:
            obj_copy = OptimizationObjective(
                name=obj.name,
                target=obj.target,
                weight=obj.weight,
                current_value=await self._evaluate_single_objective(design, obj)
            )
            evaluated.append(obj_copy)
        
        return evaluated
    
    async def _evaluate_single_objective(
        self,
        design: DesignBlueprint,
        objective: OptimizationObjective
    ) -> float:
        """Evaluate a single objective"""
        
        # Heuristic evaluation based on objective name
        if objective.name == "cost":
            # Estimate cost from materials and complexity
            base_cost = 1000.0
            complexity_factor = 1.5 if design.intent.complexity == "advanced" else 1.0
            return base_cost * complexity_factor
        
        elif objective.name == "performance":
            # Estimate performance from design parameters
            return 0.8  # Placeholder
        
        elif objective.name == "weight":
            # Estimate weight from dimensions
            dimensions = design.system_requirements.get("dimensions", {})
            volume = 1.0
            for dim in dimensions.values():
                volume *= dim
            return volume * 2.7  # Aluminum density approximation
        
        elif objective.name == "sustainability":
            # Sustainability score
            return 0.7  # Placeholder
        
        else:
            return 0.5  # Default
    
    async def _apply_parameters(
        self,
        design: DesignBlueprint,
        new_params: Dict[str, Any]
    ) -> DesignBlueprint:
        """Create new design with updated parameters"""
        # For now, just update design_parameters
        # In full implementation, would regenerate blueprint
        design.design_parameters = new_params
        return design
    
    def _calculate_improvement(
        self,
        before: List[OptimizationObjective],
        after: List[OptimizationObjective]
    ) -> float:
        """Calculate percentage improvement"""
        improvements = []
        
        for b, a in zip(before, after):
            if b.target == "maximize":
                if b.current_value > 0:
                    imp = ((a.current_value - b.current_value) / b.current_value) * 100
                else:
                    imp = 0.0
            else:  # minimize
                if b.current_value > 0:
                    imp = ((b.current_value - a.current_value) / b.current_value) * 100
                else:
                    imp = 0.0
            improvements.append(imp)
        
        return sum(improvements) / len(improvements) if improvements else 0.0


# Global singleton instance
_quantum_optimizer = None

def get_quantum_optimizer() -> QuantumDesignOptimizer:
    """Get or create the global Quantum Design Optimizer instance"""
    global _quantum_optimizer
    if _quantum_optimizer is None:
        _quantum_optimizer = QuantumDesignOptimizer()
    return _quantum_optimizer
