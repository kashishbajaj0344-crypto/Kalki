"""
Simulation Agent (Phase 9)
==========================

Implements system simulation, scenario modeling, and predictive simulation.
Uses Monte Carlo methods, discrete event simulation, and agent-based modeling.
"""

import asyncio
import time
import random
import numpy as np
import logging
import psutil
import tempfile
import os
import shutil
from typing import Dict, Any, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
from scipy import stats
import math
import subprocess
import json

# Optional Docker import
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

# Optional Pydantic import
try:
    from pydantic import BaseModel, Field, validator, ValidationError
    from pydantic.types import PositiveInt, PositiveFloat, NonNegativeFloat
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = object
    Field = lambda **kwargs: None
    validator = lambda func: func
    ValidationError = Exception
    # Define dummy types for fallback
    PositiveInt = int
    PositiveFloat = float
    NonNegativeFloat = float

from modules.logging_config import get_logger
from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from ..knowledge.rollback_manager import RollbackManager

logger = get_logger("Kalki.Simulation")


@dataclass
class SafetyPolicy:
    """Safety policy definition"""
    policy_id: str
    name: str
    description: str
    risk_level: str  # "low", "medium", "high", "critical"
    requires_approval: bool
    approval_authority: str
    prohibited_keywords: List[str]
    warning_keywords: List[str]
    max_execution_time: int
    monitoring_required: bool


@dataclass
class SafetyAssessment:
    """Safety assessment result"""
    assessment_id: str
    scenario_id: str
    risk_level: str
    requires_approval: bool
    approval_granted: bool = False
    approved_by: Optional[str] = None
    assessment_time: float = field(default_factory=lambda: time.time())
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class SimulationMetrics:
    """Comprehensive simulation performance metrics"""
    simulation_id: str
    scenario_id: str
    simulation_type: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    # Performance metrics
    cpu_usage_percent: List[float] = field(default_factory=list)
    memory_usage_mb: List[float] = field(default_factory=list)
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    
    # Simulation-specific metrics
    entities_processed: int = 0
    events_processed: int = 0
    steps_completed: int = 0
    convergence_achieved: bool = False
    convergence_iterations: int = 0
    
    # Statistical metrics
    mean_outcome: Optional[float] = None
    std_outcome: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    statistical_power: Optional[float] = None
    
    # Quality metrics
    numerical_stability: float = 1.0  # 0-1 scale
    result_consistency: float = 1.0   # 0-1 scale
    validation_score: float = 1.0     # 0-1 scale
    
    # Resource efficiency
    resource_utilization: float = 0.0  # 0-1 scale
    parallel_efficiency: float = 1.0   # 0-1 scale
    
    # Error tracking
    warnings_count: int = 0
    errors_count: int = 0
    retries_count: int = 0
    
    # Sandbox metrics
    sandbox_isolation_level: str = "none"  # "none", "process", "container"
    sandbox_resource_limits: Dict[str, Any] = field(default_factory=dict)
    sandbox_security_events: List[str] = field(default_factory=list)
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def record_system_metrics(self):
        """Record current system resource usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / (1024 * 1024)
            
            self.cpu_usage_percent.append(cpu_percent)
            self.memory_usage_mb.append(memory_mb)
            
        except Exception as e:
            logger.warning(f"Failed to record system metrics: {e}")
    
    def calculate_aggregates(self):
        """Calculate aggregate metrics"""
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time
        
        if self.cpu_usage_percent:
            self.resource_utilization = sum(self.cpu_usage_percent) / len(self.cpu_usage_percent) / 100.0
        
        # Calculate quality scores
        self._calculate_quality_metrics()
    
    def _calculate_quality_metrics(self):
        """Calculate quality assurance metrics"""
        # Numerical stability based on variance in results
        if len(self.cpu_usage_percent) > 1:
            cpu_variance = np.var(self.cpu_usage_percent)
            self.numerical_stability = max(0.1, 1.0 - (cpu_variance / 100.0))  # Normalize
        
        # Result consistency (placeholder - would be based on repeated runs)
        self.result_consistency = 0.95  # High default, adjust based on actual validation
        
        # Validation score based on error rates
        total_operations = self.entities_processed + self.events_processed + self.steps_completed
        if total_operations > 0:
            error_rate = (self.errors_count + self.warnings_count) / total_operations
            self.validation_score = max(0.1, 1.0 - error_rate)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization"""
        return {
            "simulation_id": self.simulation_id,
            "scenario_id": self.scenario_id,
            "simulation_type": self.simulation_type,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "performance": {
                "cpu_usage_percent": self.cpu_usage_percent,
                "memory_usage_mb": self.memory_usage_mb,
                "disk_io_read_mb": self.disk_io_read_mb,
                "disk_io_write_mb": self.disk_io_write_mb,
                "resource_utilization": self.resource_utilization,
                "parallel_efficiency": self.parallel_efficiency
            },
            "simulation": {
                "entities_processed": self.entities_processed,
                "events_processed": self.events_processed,
                "steps_completed": self.steps_completed,
                "convergence_achieved": self.convergence_achieved,
                "convergence_iterations": self.convergence_iterations
            },
            "statistics": {
                "mean_outcome": self.mean_outcome,
                "std_outcome": self.std_outcome,
                "confidence_interval": self.confidence_interval,
                "statistical_power": self.statistical_power
            },
            "quality": {
                "numerical_stability": self.numerical_stability,
                "result_consistency": self.result_consistency,
                "validation_score": self.validation_score
            },
            "errors": {
                "warnings_count": self.warnings_count,
                "errors_count": self.errors_count,
                "retries_count": self.retries_count
            },
            "custom_metrics": self.custom_metrics
        }


class MetricsCollector:
    """Centralized metrics collection and aggregation"""
    
    def __init__(self):
        self.metrics: Dict[str, SimulationMetrics] = {}
        self.aggregated_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.collection_interval = 1.0  # seconds
        
    def start_collection(self, simulation_id: str, scenario_id: str, simulation_type: str) -> SimulationMetrics:
        """Start collecting metrics for a simulation"""
        metrics = SimulationMetrics(
            simulation_id=simulation_id,
            scenario_id=scenario_id,
            simulation_type=simulation_type,
            start_time=time.time()
        )
        
        self.metrics[simulation_id] = metrics
        
        # Start background collection
        asyncio.create_task(self._collect_metrics_async(simulation_id))
        
        return metrics
    
    def stop_collection(self, simulation_id: str) -> Optional[SimulationMetrics]:
        """Stop collecting metrics and return final metrics"""
        if simulation_id in self.metrics:
            metrics = self.metrics[simulation_id]
            metrics.end_time = time.time()
            metrics.calculate_aggregates()
            
            # Update aggregated statistics
            self._update_aggregated_stats(metrics)
            
            return metrics
        return None
    
    async def _collect_metrics_async(self, simulation_id: str):
        """Asynchronously collect metrics during simulation"""
        while simulation_id in self.metrics:
            try:
                metrics = self.metrics[simulation_id]
                metrics.record_system_metrics()
                await asyncio.sleep(self.collection_interval)
            except Exception as e:
                logger.warning(f"Metrics collection error for {simulation_id}: {e}")
                break
    
    def _update_aggregated_stats(self, metrics: SimulationMetrics):
        """Update aggregated statistics across all simulations"""
        sim_type = metrics.simulation_type
        
        if sim_type not in self.aggregated_stats:
            self.aggregated_stats[sim_type] = {
                "total_runs": 0,
                "total_duration": 0,
                "avg_duration": 0,
                "avg_cpu_usage": 0,
                "avg_memory_usage": 0,
                "success_rate": 0,
                "avg_quality_score": 0,
                "performance_trend": []
            }
        
        stats = self.aggregated_stats[sim_type]
        stats["total_runs"] += 1
        
        if metrics.duration:
            stats["total_duration"] += metrics.duration
            stats["avg_duration"] = stats["total_duration"] / stats["total_runs"]
        
        if metrics.cpu_usage_percent:
            avg_cpu = sum(metrics.cpu_usage_percent) / len(metrics.cpu_usage_percent)
            stats["avg_cpu_usage"] = (stats["avg_cpu_usage"] * (stats["total_runs"] - 1) + avg_cpu) / stats["total_runs"]
        
        if metrics.memory_usage_mb:
            avg_mem = sum(metrics.memory_usage_mb) / len(metrics.memory_usage_mb)
            stats["avg_memory_usage"] = (stats["avg_memory_usage"] * (stats["total_runs"] - 1) + avg_mem) / stats["total_runs"]
        
        # Quality score
        quality_score = (metrics.numerical_stability + metrics.result_consistency + metrics.validation_score) / 3.0
        stats["avg_quality_score"] = (stats["avg_quality_score"] * (stats["total_runs"] - 1) + quality_score) / stats["total_runs"]
        
        # Performance trend (last 10 runs)
        stats["performance_trend"].append({
            "timestamp": metrics.end_time,
            "duration": metrics.duration,
            "cpu_usage": stats["avg_cpu_usage"],
            "quality_score": quality_score
        })
        
        # Keep only last 10 entries
        if len(stats["performance_trend"]) > 10:
            stats["performance_trend"].pop(0)
    
    def get_aggregated_stats(self, simulation_type: str = None) -> Dict[str, Any]:
        """Get aggregated statistics"""
        if simulation_type:
            return self.aggregated_stats.get(simulation_type, {})
        return dict(self.aggregated_stats)
    
    def get_simulation_metrics(self, simulation_id: str) -> Optional[SimulationMetrics]:
        """Get metrics for a specific simulation"""
        return self.metrics.get(simulation_id)


# Pydantic validation models for input validation
if PYDANTIC_AVAILABLE:

    class ResourceLimitsModel(BaseModel):
        """Validated resource limits"""
        cpu_percent: "NonNegativeFloat" = Field(..., ge=0.0, le=100.0, description="CPU usage percentage limit")
        memory_mb: "PositiveInt" = Field(..., gt=0, description="Memory limit in MB")
        disk_mb: "PositiveInt" = Field(..., gt=0, description="Disk space limit in MB")

        @validator('cpu_percent')
        def validate_cpu_percent(cls, v):
            if v > 100.0:
                raise ValueError('CPU percent cannot exceed 100%')
            return v

    class SandboxConfigModel(BaseModel):
        """Validated sandbox configuration"""
        isolation_level: str = Field(..., pattern="^(none|process|container)$", description="Isolation level")
        network_access: bool = Field(False, description="Whether network access is allowed")
        filesystem_access: str = Field(..., pattern="^(none|readonly|readwrite)$", description="Filesystem access level")
        resource_limits: ResourceLimitsModel = Field(..., description="Resource limits for sandbox")
        security_profile: str = Field(..., pattern="^(minimal|standard|strict)$", description="Security profile")
        allowed_commands: List[str] = Field(..., min_items=1, description="List of allowed commands")
        timeout_seconds: "PositiveInt" = Field(..., le=3600, description="Timeout in seconds (max 1 hour)")
        cleanup_on_exit: bool = Field(True, description="Whether to cleanup resources on exit")

    class InputDistributionModel(BaseModel):
        """Validated input distribution specification"""
        type: str = Field(..., pattern="^(uniform|normal|exponential|custom)$", description="Distribution type")
        low: Optional[float] = Field(None, description="Lower bound for uniform distribution")
        high: Optional[float] = Field(None, description="Upper bound for uniform distribution")
        mean: Optional[float] = Field(None, description="Mean for normal distribution")
        std: Optional[float] = Field(None, description="Standard deviation for normal distribution")
        scale: Optional[float] = Field(None, description="Scale parameter for exponential distribution")
        default: Optional[float] = Field(None, description="Default value if distribution fails")

        @validator('low', 'high')
        def validate_uniform_bounds(cls, v, values):
            if values.get('type') == 'uniform':
                if v is None:
                    raise ValueError('low and high are required for uniform distribution')
            return v

        @validator('mean', 'std')
        def validate_normal_params(cls, v, values):
            if values.get('type') == 'normal':
                if v is None:
                    raise ValueError('mean and std are required for normal distribution')
            return v

        @validator('scale')
        def validate_exponential_params(cls, v, values):
            if values.get('type') == 'exponential':
                if v is None:
                    raise ValueError('scale is required for exponential distribution')
            return v

    class SimulationScenarioModel(BaseModel):
        """Validated simulation scenario definition"""
        scenario_id: str = Field(..., min_length=1, max_length=100, pattern="^[a-zA-Z0-9_-]+$", description="Unique scenario identifier")
        name: str = Field(..., min_length=1, max_length=200, description="Human-readable scenario name")
        description: str = Field("", max_length=1000, description="Scenario description")
        type: str = Field(..., pattern="^(monte_carlo|discrete_event|agent_based)$", description="Simulation methodology type")
        parameters: Dict[str, Any] = Field(default_factory=dict, description="Scenario-specific parameters")
        input_distributions: Dict[str, InputDistributionModel] = Field(default_factory=dict, description="Input parameter distributions")
        duration: "PositiveInt" = Field(..., le=86400, description="Simulation duration (max 24 hours)")
        time_step: "PositiveFloat" = Field(..., le=3600.0, description="Simulation time step (max 1 hour)")
        entities: List[Dict[str, Any]] = Field(default_factory=list, description="Simulation entities")
        constraints: List[Dict[str, Any]] = Field(default_factory=list, description="Simulation constraints")
        termination_conditions: List[Dict[str, Any]] = Field(default_factory=list, description="Termination conditions")

        @validator('entities')
        def validate_entities(cls, v):
            if len(v) > 1000:
                raise ValueError('Maximum 1000 entities allowed')
            return v

        @validator('constraints')
        def validate_constraints(cls, v):
            if len(v) > 100:
                raise ValueError('Maximum 100 constraints allowed')
            return v

    class SimulationParametersModel(BaseModel):
        """Validated simulation execution parameters"""
        num_runs: "PositiveInt" = Field(..., le=100000, description="Number of simulation runs (max 100k)")
        confidence_level: "NonNegativeFloat" = Field(0.95, ge=0.0, le=1.0, description="Statistical confidence level")
        convergence_threshold: "NonNegativeFloat" = Field(0.01, ge=0.0, le=1.0, description="Convergence threshold")
        check_convergence_every: "PositiveInt" = Field(100, le=10000, description="Check convergence every N runs")
        random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
        max_parallel_runs: "PositiveInt" = Field(10, le=100, description="Maximum parallel simulation runs")
        timeout_seconds: "PositiveInt" = Field(3600, le=86400, description="Execution timeout (max 24 hours)")

    class SafetyPolicyModel(BaseModel):
        """Validated safety policy"""
        policy_id: str = Field(..., min_length=1, max_length=50, pattern="^[a-zA-Z0-9_-]+$", description="Unique policy identifier")
        name: str = Field(..., min_length=1, max_length=100, description="Policy name")
        description: str = Field("", max_length=500, description="Policy description")
        risk_level: str = Field(..., pattern="^(low|medium|high|critical)$", description="Risk level")
        requires_approval: bool = Field(False, description="Whether approval is required")
        approval_authority: str = Field("", description="Approval authority")
        prohibited_keywords: List[str] = Field(default_factory=list, description="Prohibited keywords")
        warning_keywords: List[str] = Field(default_factory=list, description="Warning keywords")
        max_execution_time: "PositiveInt" = Field(3600, le=86400, description="Maximum execution time")
        monitoring_required: bool = Field(False, description="Whether monitoring is required")

        @validator('prohibited_keywords', 'warning_keywords')
        def validate_keywords(cls, v):
            if len(v) > 50:
                raise ValueError('Maximum 50 keywords allowed')
            for keyword in v:
                if len(keyword) > 100:
                    raise ValueError('Keyword length cannot exceed 100 characters')
            return v

    class MetricsConfigModel(BaseModel):
        """Validated metrics configuration"""
        enable_metrics: bool = Field(True, description="Whether to enable metrics collection")
        collection_interval: "PositiveFloat" = Field(1.0, le=60.0, description="Metrics collection interval (max 60s)")
        retention_days: "PositiveInt" = Field(30, le=365, description="Metrics retention period in days")
        enable_system_metrics: bool = Field(True, description="Whether to collect system metrics")
        enable_simulation_metrics: bool = Field(True, description="Whether to collect simulation metrics")
        enable_sandbox_metrics: bool = Field(True, description="Whether to collect sandbox metrics")
        alert_thresholds: Dict[str, float] = Field(default_factory=dict, description="Alert thresholds")

    class SimulationRequestModel(BaseModel):
        """Validated simulation execution request"""
        scenario: SimulationScenarioModel = Field(..., description="Simulation scenario")
        parameters: SimulationParametersModel = Field(..., description="Execution parameters")
        sandbox_config: Optional[SandboxConfigModel] = Field(None, description="Sandbox configuration")
        metrics_config: Optional[MetricsConfigModel] = Field(None, description="Metrics configuration")
        safety_policies: List[SafetyPolicyModel] = Field(default_factory=list, description="Applicable safety policies")

        @validator('safety_policies')
        def validate_safety_policies(cls, v):
            if len(v) > 10:
                raise ValueError('Maximum 10 safety policies allowed')
            return v

else:
    # Fallback classes when pydantic is not available
    class ValidationModel:
        """Fallback validation model"""
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    ResourceLimitsModel = ValidationModel
    SandboxConfigModel = ValidationModel
    InputDistributionModel = ValidationModel
    SimulationScenarioModel = ValidationModel
    SimulationParametersModel = ValidationModel
    SafetyPolicyModel = ValidationModel
    MetricsConfigModel = ValidationModel
    SimulationRequestModel = ValidationModel


class RetryManager:
    """Manages retry logic with exponential backoff and circuit breaker patterns"""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, max_delay: float = 60.0,
                 backoff_factor: float = 2.0, circuit_breaker_threshold: int = 5,
                 circuit_breaker_timeout: float = 300.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # Circuit breaker state
        self.failure_count = 0
        self.last_failure_time = 0
        self.circuit_open = False
        
        # Retry statistics
        self.retry_stats = {
            "total_attempts": 0,
            "successful_retries": 0,
            "failed_retries": 0,
            "circuit_breaker_trips": 0
        }
        
        self.logger = logging.getLogger("kalki.retry")
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic
        
        Args:
            func: The async function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
            
        Returns:
            The result of the function call
            
        Raises:
            The last exception if all retries fail
        """
        self.retry_stats["total_attempts"] += 1
        
        # Check circuit breaker
        if self._is_circuit_open():
            self.logger.warning("Circuit breaker is open, skipping execution")
            raise CircuitBreakerOpenError("Circuit breaker is open")
        
        last_exception = None
        
        for attempt in range(self.max_retries + 1):  # +1 for initial attempt
            try:
                result = await func(*args, **kwargs)
                
                # Success - reset circuit breaker
                if attempt > 0:
                    self.retry_stats["successful_retries"] += 1
                    self.logger.info(f"Retry successful on attempt {attempt + 1}")
                
                self._reset_circuit_breaker()
                return result
                
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                # Don't retry on the last attempt
                if attempt == self.max_retries:
                    break
                
                # Record failure for circuit breaker
                self._record_failure()
                
                # Calculate delay with exponential backoff
                delay = self._calculate_delay(attempt)
                self.logger.info(f"Retrying in {delay:.2f} seconds...")
                await asyncio.sleep(delay)
        
        # All retries failed
        self.retry_stats["failed_retries"] += 1
        self.logger.error(f"All {self.max_retries + 1} attempts failed")
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter"""
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter (±25%)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        delay += jitter
        
        return max(0.1, delay)  # Minimum 100ms delay
    
    def _record_failure(self):
        """Record a failure for circuit breaker logic"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.circuit_breaker_threshold:
            self.circuit_open = True
            self.retry_stats["circuit_breaker_trips"] += 1
            self.logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def _reset_circuit_breaker(self):
        """Reset circuit breaker state on success"""
        self.failure_count = 0
        self.circuit_open = False
    
    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is open"""
        if not self.circuit_open:
            return False
        
        # Check if timeout has expired
        if time.time() - self.last_failure_time > self.circuit_breaker_timeout:
            self.logger.info("Circuit breaker timeout expired, resetting")
            self._reset_circuit_breaker()
            return False
        
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retry statistics"""
        return {
            "max_retries": self.max_retries,
            "base_delay": self.base_delay,
            "backoff_factor": self.backoff_factor,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "circuit_breaker_timeout": self.circuit_breaker_timeout,
            "current_failure_count": self.failure_count,
            "circuit_open": self.circuit_open,
            "last_failure_time": self.last_failure_time,
            "retry_stats": self.retry_stats.copy()
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class SimulationBackend(ABC):
    """Abstract base class for simulation backends"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"kalki.backend.{self.__class__.__name__}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the backend"""
        pass
    
    @abstractmethod
    async def execute_simulation(self, scenario: "SimulationScenario", parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simulation using this backend"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of supported simulation types"""
        pass
    
    @abstractmethod
    def get_requirements(self) -> List[str]:
        """Get list of required packages/libraries"""
        pass
    
    async def validate_scenario(self, scenario: "SimulationScenario") -> Tuple[bool, List[str]]:
        """Validate that scenario is compatible with this backend"""
        return True, []
    
    async def cleanup(self):
        """Clean up backend resources"""
        pass


class MonteCarloBackend(SimulationBackend):
    """Backend for Monte Carlo simulations using numpy/scipy"""
    
    def get_capabilities(self) -> List[str]:
        return ["monte_carlo"]
    
    def get_requirements(self) -> List[str]:
        return ["numpy", "scipy"]
    
    async def initialize(self) -> bool:
        try:
            import numpy as np
            import scipy.stats
            self.np = np
            self.stats = scipy.stats
            return True
        except ImportError as e:
            self.logger.error(f"Failed to import required packages: {e}")
            return False
    
    async def execute_simulation(self, scenario: "SimulationScenario", parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Monte Carlo simulation"""
        num_runs = parameters.get("num_runs", 1000)
        confidence_level = parameters.get("confidence_level", 0.95)
        random_seed = parameters.get("random_seed")
        
        if random_seed is not None:
            self.np.random.seed(random_seed)
        
        results = []
        
        # Generate samples based on scenario input distributions
        if hasattr(scenario, 'input_distributions') and scenario.input_distributions:
            # Use defined input distributions
            for _ in range(num_runs):
                sample_result = {}
                for param_name, dist_config in scenario.input_distributions.items():
                    if dist_config.type == "uniform":
                        sample = self.np.random.uniform(dist_config.low, dist_config.high)
                    elif dist_config.type == "normal":
                        sample = self.np.random.normal(dist_config.mean, dist_config.std)
                    elif dist_config.type == "exponential":
                        sample = self.np.random.exponential(dist_config.scale)
                    else:
                        # Default to normal distribution
                        sample = self.np.random.normal(0, 1)
                    sample_result[param_name] = sample
                
                # For single parameter case, use the first parameter's value
                if len(sample_result) == 1:
                    results.append(list(sample_result.values())[0])
                else:
                    # For multi-parameter, store as dict or compute aggregate
                    results.append(sample_result)
        else:
            # Fallback to default normal distribution
            for _ in range(num_runs):
                sample = self.np.random.normal(0, 1)
                results.append(sample)
        
        # Calculate statistics
        results_array = self.np.array(results)
        mean_result = float(self.np.mean(results_array))
        std_result = float(self.np.std(results_array))
        
        # Confidence interval
        confidence_interval = self.stats.norm.interval(confidence_level, loc=mean_result, scale=std_result/self.np.sqrt(num_runs))
        confidence_interval = [float(ci) for ci in confidence_interval]
        
        return {
            "method": "monte_carlo",
            "num_runs": num_runs,
            "results": results,
            "statistics": {
                "mean": mean_result,
                "std": std_result,
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level
            }
        }


class DiscreteEventBackend(SimulationBackend):
    """Backend for discrete event simulations using SimPy"""
    
    def get_capabilities(self) -> List[str]:
        return ["discrete_event"]
    
    def get_requirements(self) -> List[str]:
        return ["simpy"]
    
    async def initialize(self) -> bool:
        try:
            import simpy
            self.simpy = simpy
            return True
        except ImportError:
            self.logger.warning("SimPy not available, discrete event backend disabled")
            return False
    
    async def execute_simulation(self, scenario: "SimulationScenario", parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute discrete event simulation"""
        max_events = parameters.get("max_events", 10000)
        time_step = parameters.get("time_step", 1.0)
        
        env = self.simpy.Environment()
        event_log = []
        entities = {}
        
        # Initialize entities
        for entity_data in scenario.entities:
            entity_id = entity_data["id"]
            entities[entity_id] = {
                "state": entity_data.get("initial_state", {}),
                "type": entity_data.get("type", "generic")
            }
        
        # Define event processing
        def process_event(env, event_data):
            nonlocal event_log
            event_log.append({
                "time": env.now,
                "event": event_data
            })
            
            # Schedule next event
            if len(event_log) < max_events:
                next_time = env.now + self.np.random.exponential(1.0)  # Random interarrival
                env.process(process_event(env, {"type": "next_event"}))
        
        # Start initial events
        for entity_data in scenario.entities:
            if entity_data.get("initial_event"):
                env.process(process_event(env, entity_data["initial_event"]))
        
        # Run simulation
        env.run(until=max_events * time_step)
        
        return {
            "method": "discrete_event",
            "events_processed": len(event_log),
            "simulation_time": env.now,
            "event_log": event_log[-1000:],  # Last 1000 events
            "final_entities": entities
        }


class AgentBasedBackend(SimulationBackend):
    """Backend for agent-based simulations using Mesa"""
    
    def get_capabilities(self) -> List[str]:
        return ["agent_based"]
    
    def get_requirements(self) -> List[str]:
        return ["mesa"]
    
    async def initialize(self) -> bool:
        try:
            from mesa import Model, Agent
            self.mesa = mesa
            self.Model = Model
            self.Agent = Agent
            return True
        except ImportError:
            self.logger.warning("Mesa not available, agent-based backend disabled")
            return False
    
    async def execute_simulation(self, scenario: "SimulationScenario", parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent-based simulation"""
        num_steps = parameters.get("num_steps", 100)
        
        # Create Mesa model
        model = self.Model()
        
        # Initialize agents
        agents = []
        for entity_data in scenario.entities:
            agent = self.SimpleAgent(entity_data["id"], model, entity_data)
            agents.append(agent)
            model.schedule.add(agent)
        
        # Run simulation
        step_results = []
        for step in range(num_steps):
            model.step()
            
            # Record step data
            step_result = {
                "step": step,
                "agent_states": [agent.get_state() for agent in agents],
                "model_metrics": model.datacollector.get_model_vars_dataframe().to_dict() if hasattr(model, 'datacollector') else {}
            }
            step_results.append(step_result)
        
        return {
            "method": "agent_based",
            "steps_completed": num_steps,
            "final_agent_states": [agent.get_state() for agent in agents],
            "step_results": step_results[-100:],  # Last 100 steps
            "model_metrics": model.datacollector.get_model_vars_dataframe().to_dict() if hasattr(model, 'datacollector') else {}
        }
    
    class SimpleAgent:
        """Simple agent for Mesa-based simulations"""
        def __init__(self, unique_id, model, entity_data):
            # Mesa Agent-like interface
            self.unique_id = unique_id
            self.model = model
            self.entity_data = entity_data
            self.state = entity_data.get("initial_state", {})
        
        def step(self):
            # Simple random walk behavior
            import random
            self.state["x"] = self.state.get("x", 0) + random.randint(-1, 2)
            self.state["y"] = self.state.get("y", 0) + random.randint(-1, 2)
        
        def get_state(self):
            return {
                "id": self.unique_id,
                "state": self.state.copy()
            }


class OptimizationBackend(SimulationBackend):
    """Backend for optimization problems using PySCIPOpt or similar"""
    
    def get_capabilities(self) -> List[str]:
        return ["optimization", "linear_programming", "mixed_integer"]
    
    def get_requirements(self) -> List[str]:
        return ["pyscipopt"]  # Alternative: ["ortools", "gurobipy", "cplex"]
    
    async def initialize(self) -> bool:
        try:
            import pyscipopt
            self.scip = pyscipopt
            return True
        except ImportError:
            self.logger.warning("PySCIPOpt not available, optimization backend disabled")
            return False
    
    async def execute_simulation(self, scenario: "SimulationScenario", parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute optimization"""
        # Placeholder for optimization logic
        # Would parse scenario constraints and objective function
        
        model = self.scip.Model("OptimizationProblem")
        
        # Add variables (placeholder)
        x = model.addVar("x", vtype="C")
        y = model.addVar("y", vtype="C")
        
        # Add constraints (placeholder)
        model.addCons(x + y <= 10)
        model.addCons(x >= 0)
        model.addCons(y >= 0)
        
        # Set objective
        model.setObjective(x + 2*y, sense="maximize")
        
        # Solve
        model.optimize()
        
        if model.getStatus() == "optimal":
            return {
                "method": "optimization",
                "status": "optimal",
                "objective_value": model.getObjVal(),
                "solution": {
                    "x": model.getVal(x),
                    "y": model.getVal(y)
                }
            }
        else:
            return {
                "method": "optimization",
                "status": model.getStatus(),
                "error": "Optimization failed to find optimal solution"
            }


class BackendManager:
    """Manages pluggable simulation backends"""
    
    def __init__(self):
        self.backends: Dict[str, SimulationBackend] = {}
        self.logger = logging.getLogger("kalki.backend_manager")
        
        # Register built-in backends
        self._register_builtin_backends()
    
    def _register_builtin_backends(self):
        """Register built-in simulation backends"""
        backend_classes = [
            MonteCarloBackend,
            DiscreteEventBackend,
            AgentBasedBackend,
            OptimizationBackend
        ]
        
        for backend_class in backend_classes:
            backend_name = backend_class.__name__.replace("Backend", "").lower()
            try:
                backend = backend_class()
                # Try to initialize synchronously first
                try:
                    # Check if initialize is async
                    if asyncio.iscoroutinefunction(backend.initialize):
                        # For now, skip async initialization in sync context
                        self.logger.info(f"Deferring async initialization for backend: {backend_name}")
                        self.backends[backend_name] = backend
                    else:
                        if backend.initialize():
                            self.backends[backend_name] = backend
                            self.logger.info(f"Registered backend: {backend_name} ({backend.get_capabilities()})")
                        else:
                            self.logger.warning(f"Failed to initialize backend: {backend_name}")
                except Exception as init_e:
                    self.logger.warning(f"Failed to initialize backend {backend_name}: {init_e}")
                    # Still register the backend for later async initialization
                    self.backends[backend_name] = backend
                    
            except Exception as e:
                self.logger.warning(f"Error creating backend {backend_name}: {e}")
    
    async def register_backend(self, name: str, backend: SimulationBackend) -> bool:
        """Register a custom backend"""
        try:
            if await backend.initialize():
                self.backends[name] = backend
                self.logger.info(f"Registered custom backend: {name}")
                return True
            else:
                self.logger.error(f"Failed to initialize custom backend: {name}")
                return False
        except Exception as e:
            self.logger.error(f"Error registering custom backend {name}: {e}")
            return False
    
    def get_backend(self, name: str) -> Optional[SimulationBackend]:
        """Get a backend by name"""
        return self.backends.get(name)
    
    def get_backends_for_capability(self, capability: str) -> List[str]:
        """Get list of backend names that support a capability"""
        return [name for name, backend in self.backends.items() 
                if capability in backend.get_capabilities()]
    
    def get_available_backends(self) -> Dict[str, List[str]]:
        """Get all available backends and their capabilities"""
        return {name: backend.get_capabilities() for name, backend in self.backends.items()}
    
    async def execute_with_backend(self, backend_name: str, scenario: "SimulationScenario", 
                                 parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simulation using specified backend"""
        backend = self.get_backend(backend_name)
        if not backend:
            return {
                "status": "error",
                "error": f"Backend '{backend_name}' not found",
                "available_backends": list(self.backends.keys())
            }
        
        # Validate scenario compatibility
        is_valid, errors = await backend.validate_scenario(scenario)
        if not is_valid:
            return {
                "status": "error",
                "error": f"Scenario not compatible with backend {backend_name}",
                "validation_errors": errors
            }
        
        try:
            result = await backend.execute_simulation(scenario, parameters)
            result["backend"] = backend_name
            result["status"] = "success"
            return result
        except Exception as e:
            self.logger.exception(f"Error executing simulation with backend {backend_name}")
            return {
                "status": "error",
                "error": f"Backend execution failed: {str(e)}",
                "backend": backend_name
            }
    
    async def initialize_backends(self):
        """Asynchronously initialize all registered backends"""
        initialized_backends = {}
        
        for name, backend in self.backends.items():
            try:
                if await backend.initialize():
                    initialized_backends[name] = backend
                    self.logger.info(f"Initialized backend: {name} ({backend.get_capabilities()})")
                else:
                    self.logger.warning(f"Failed to initialize backend: {name}")
            except Exception as e:
                self.logger.warning(f"Error initializing backend {name}: {e}")
        
        self.backends = initialized_backends


class InputValidator:
    """Comprehensive input validation for simulation requests"""

    def __init__(self):
        self.logger = logging.getLogger("kalki.validation")

    def validate_simulation_request(self, request_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """
        Validate a complete simulation request

        Returns:
            Tuple of (is_valid, error_messages, validated_data)
        """
        if not PYDANTIC_AVAILABLE:
            self.logger.warning("Pydantic not available, skipping input validation")
            return True, [], request_data

        errors = []
        validated_data = {}

        try:
            # Validate the complete request
            request = SimulationRequestModel(**request_data)
            validated_data = request.dict()

            return True, [], validated_data

        except ValidationError as e:
            for error in e.errors():
                field_path = ".".join(str(loc) for loc in error['loc'])
                error_msg = f"{field_path}: {error['msg']}"
                errors.append(error_msg)

            self.logger.warning(f"Validation failed: {errors}")
            return False, errors, {}

        except Exception as e:
            error_msg = f"Unexpected validation error: {str(e)}"
            errors.append(error_msg)
            self.logger.error(error_msg)
            return False, errors, {}

    def validate_scenario(self, scenario_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate simulation scenario"""
        if not PYDANTIC_AVAILABLE:
            return True, [], scenario_data

        try:
            scenario = SimulationScenarioModel(**scenario_data)
            return True, [], scenario.dict()
        except ValidationError as e:
            errors = [f"scenario.{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in e.errors()]
            return False, errors, {}
        except Exception as e:
            return False, [f"Scenario validation error: {str(e)}"], {}

    def validate_parameters(self, params_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate simulation parameters"""
        if not PYDANTIC_AVAILABLE:
            return True, [], params_data

        try:
            params = SimulationParametersModel(**params_data)
            return True, [], params.dict()
        except ValidationError as e:
            errors = [f"parameters.{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in e.errors()]
            return False, errors, {}
        except Exception as e:
            return False, [f"Parameters validation error: {str(e)}"], {}

    def validate_sandbox_config(self, config_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate sandbox configuration"""
        if not PYDANTIC_AVAILABLE:
            return True, [], config_data

        try:
            config = SandboxConfigModel(**config_data)
            return True, [], config.dict()
        except ValidationError as e:
            errors = [f"sandbox.{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in e.errors()]
            return False, errors, {}
        except Exception as e:
            return False, [f"Sandbox config validation error: {str(e)}"], {}

    def validate_safety_policy(self, policy_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate safety policy"""
        if not PYDANTIC_AVAILABLE:
            return True, [], policy_data

        try:
            policy = SafetyPolicyModel(**policy_data)
            return True, [], policy.dict()
        except ValidationError as e:
            errors = [f"policy.{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in e.errors()]
            return False, errors, {}
        except Exception as e:
            return False, [f"Safety policy validation error: {str(e)}"], {}

    def sanitize_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize and normalize input data"""
        sanitized = {}

        # Deep copy to avoid modifying original
        import copy
        sanitized = copy.deepcopy(input_data)

        # Remove potentially dangerous keys
        dangerous_keys = ['__class__', '__module__', '__subclasshook__', '__dict__', '__weakref__']
        def remove_dangerous(obj):
            if isinstance(obj, dict):
                for key in list(obj.keys()):
                    if key in dangerous_keys:
                        del obj[key]
                    else:
                        remove_dangerous(obj[key])
            elif isinstance(obj, list):
                for item in obj:
                    remove_dangerous(item)

        remove_dangerous(sanitized)

        # Normalize string values
        def normalize_strings(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str):
                        # Remove null bytes and normalize whitespace
                        obj[key] = value.replace('\x00', '').strip()
                    else:
                        normalize_strings(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, str):
                        obj[i] = item.replace('\x00', '').strip()
                    else:
                        normalize_strings(item)

        normalize_strings(sanitized)

        return sanitized

    def validate_resource_limits(self, limits_data: Dict[str, Any]) -> Tuple[bool, List[str], Dict[str, Any]]:
        """Validate resource limits"""
        if not PYDANTIC_AVAILABLE:
            return True, [], limits_data

        try:
            limits = ResourceLimitsModel(**limits_data)
            return True, [], limits.dict()
        except ValidationError as e:
            errors = [f"limits.{'.'.join(str(loc) for loc in error['loc'])}: {error['msg']}" for error in e.errors()]
            return False, errors, {}
        except Exception as e:
            return False, [f"Resource limits validation error: {str(e)}"], {}


@dataclass
class SandboxConfig:
    """Configuration for sandbox execution"""
    isolation_level: str = "container"  # "none", "process", "container"
    network_access: bool = False
    filesystem_access: str = "readonly"  # "none", "readonly", "readwrite"
    resource_limits: "ResourceLimits" = field(default_factory=lambda: ResourceLimits())
    security_profile: str = "standard"  # "minimal", "standard", "strict"
    allowed_commands: List[str] = field(default_factory=lambda: ["python3", "python", "numpy", "scipy", "matplotlib"])
    timeout_seconds: int = 3600
    cleanup_on_exit: bool = True


@dataclass
class DockerSandboxConfig:
    """Docker-specific sandbox configuration"""
    image: str = "python:3.11-slim"
    volumes: Dict[str, Dict[str, str]] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    network_mode: str = "none"
    cap_drop: List[str] = field(default_factory=lambda: ["ALL"])
    cap_add: List[str] = field(default_factory=list)
    security_opt: List[str] = field(default_factory=lambda: ["no-new-privileges:true"])
    tmpfs: Dict[str, str] = field(default_factory=lambda: {"/tmp": "rw,noexec,nosuid,size=100m"})
    read_only: bool = True
    working_dir: str = "/app"


class DockerSandbox:
    """Docker-based sandbox for secure simulation execution"""
    
    def __init__(self, config: DockerSandboxConfig = None):
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker is not available. Install docker package to use DockerSandbox.")
        
        self.config = config or DockerSandboxConfig()
        
        try:
            # Try different Docker client initialization methods
            try:
                self.client = docker.from_env()
            except AttributeError:
                try:
                    self.client = docker.APIClient()
                except:
                    self.client = docker.Client()
        except Exception as e:
            logger.warning(f"Failed to initialize Docker client: {e}")
            raise RuntimeError("Docker daemon not available or docker package not properly installed")
        self.containers: Dict[str, Any] = {}
        self.temp_dirs: Dict[str, str] = {}
        
    async def create_sandbox(self, sandbox_id: str, simulation_code: str, 
                           input_data: Dict[str, Any] = None) -> str:
        """Create a sandboxed container for simulation execution"""
        try:
            # Create temporary directory for simulation files
            temp_dir = tempfile.mkdtemp(prefix=f"sandbox_{sandbox_id}_")
            self.temp_dirs[sandbox_id] = temp_dir
            
            # Write simulation code to file
            code_file = os.path.join(temp_dir, "simulation.py")
            with open(code_file, "w") as f:
                f.write(simulation_code)
            
            # Write input data if provided
            if input_data:
                input_file = os.path.join(temp_dir, "input.json")
                with open(input_file, "w") as f:
                    json.dump(input_data, f)
            
            # Create requirements.txt for dependencies
            requirements_file = os.path.join(temp_dir, "requirements.txt")
            with open(requirements_file, "w") as f:
                f.write("numpy\nscipy\npsutil\n")
            
            # Prepare container configuration
            container_config = {
                "image": self.config.image,
                "command": ["python3", "/app/simulation.py"],
                "working_dir": self.config.working_dir,
                "network_mode": self.config.network_mode,
                "cap_drop": self.config.cap_drop,
                "cap_add": self.config.cap_add,
                "security_opt": self.config.security_opt,
                "tmpfs": self.config.tmpfs,
                "read_only": self.config.read_only,
                "environment": self.config.environment.copy(),
                "volumes": {
                    temp_dir: {"bind": "/app", "mode": "rw"}
                },
                "cpu_quota": int(self.config.resource_limits.cpu_percent * 1000),
                "mem_limit": f"{self.config.resource_limits.memory_mb}m",
                "detach": True
            }
            
            # Merge with custom volumes
            container_config["volumes"].update(self.config.volumes)
            
            # Create and start container
            container = self.client.containers.create(**container_config)
            container.start()
            
            self.containers[sandbox_id] = container
            
            logger.info(f"Created sandbox container {container.id} for simulation {sandbox_id}")
            return container.id
            
        except Exception as e:
            logger.error(f"Failed to create sandbox for {sandbox_id}: {e}")
            await self.cleanup_sandbox(sandbox_id)
            raise
    
    async def execute_simulation(self, sandbox_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Execute simulation in sandbox and collect results"""
        if sandbox_id not in self.containers:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        container = self.containers[sandbox_id]
        
        try:
            # Wait for completion with timeout
            result = container.wait(timeout=timeout)
            
            # Get logs
            logs = container.logs().decode('utf-8')
            
            # Get exit code
            exit_code = result["StatusCode"]
            
            # Try to read output file
            output_file = os.path.join(self.temp_dirs[sandbox_id], "output.json")
            output_data = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        output_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read output file: {e}")
            
            execution_result = {
                "exit_code": exit_code,
                "logs": logs,
                "output": output_data,
                "success": exit_code == 0,
                "container_id": container.id
            }
            
            logger.info(f"Sandbox execution completed for {sandbox_id}: exit_code={exit_code}")
            return execution_result
            
        except Exception as e:
            logger.error(f"Sandbox execution failed for {sandbox_id}: {e}")
            raise
        finally:
            await self.cleanup_sandbox(sandbox_id)
    
    async def cleanup_sandbox(self, sandbox_id: str):
        """Clean up sandbox resources"""
        try:
            # Stop and remove container
            if sandbox_id in self.containers:
                container = self.containers[sandbox_id]
                try:
                    container.stop(timeout=10)
                    container.remove(force=True)
                except Exception as e:
                    logger.warning(f"Failed to remove container {container.id}: {e}")
                del self.containers[sandbox_id]
            
            # Clean up temporary directory
            if sandbox_id in self.temp_dirs:
                temp_dir = self.temp_dirs[sandbox_id]
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to remove temp dir {temp_dir}: {e}")
                del self.temp_dirs[sandbox_id]
            
            logger.info(f"Cleaned up sandbox {sandbox_id}")
            
        except Exception as e:
            logger.error(f"Error during sandbox cleanup for {sandbox_id}: {e}")
    
    def get_container_stats(self, sandbox_id: str) -> Dict[str, Any]:
        """Get container resource statistics"""
        if sandbox_id not in self.containers:
            return {}
        
        try:
            container = self.containers[sandbox_id]
            stats = container.stats(stream=False)
            
            return {
                "cpu_usage": stats.get("cpu_stats", {}).get("cpu_usage", {}),
                "memory_usage": stats.get("memory_stats", {}).get("usage", 0),
                "network_stats": stats.get("networks", {}),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.warning(f"Failed to get container stats for {sandbox_id}: {e}")
            return {}
    
    async def cleanup_all(self):
        """Clean up all sandbox resources"""
        sandbox_ids = list(self.containers.keys()) + list(self.temp_dirs.keys())
        for sandbox_id in sandbox_ids:
            await self.cleanup_sandbox(sandbox_id)


class ProcessSandbox:
    """Process-based sandbox using subprocess with resource limits"""
    
    def __init__(self, config: SandboxConfig = None):
        self.config = config or SandboxConfig()
        self.processes: Dict[str, subprocess.Popen] = {}
        self.temp_dirs: Dict[str, str] = {}
        
    async def create_sandbox(self, sandbox_id: str, simulation_code: str,
                           input_data: Dict[str, Any] = None) -> int:
        """Create a sandboxed process for simulation execution"""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix=f"process_sandbox_{sandbox_id}_")
            self.temp_dirs[sandbox_id] = temp_dir
            
            # Write simulation code
            code_file = os.path.join(temp_dir, "simulation.py")
            with open(code_file, "w") as f:
                f.write(simulation_code)
            
            # Write input data
            if input_data:
                input_file = os.path.join(temp_dir, "input.json")
                with open(input_file, "w") as f:
                    json.dump(input_data, f)
            
            # Prepare environment
            env = os.environ.copy()
            env.update({
                "PYTHONPATH": temp_dir,
                "SANDBOX_MODE": "true",
                "MAX_MEMORY_MB": str(self.config.resource_limits.memory_mb),
                "MAX_CPU_PERCENT": str(self.config.resource_limits.cpu_percent)
            })
            
            # Start process with resource limits (simplified for cross-platform)
            if os.name == 'posix':  # Unix-like systems
                try:
                    # Try prlimit if available
                    cmd = [
                        "prlimit",
                        f"--cpu={int(self.config.resource_limits.cpu_percent)}",
                        f"--as={self.config.resource_limits.memory_mb * 1024 * 1024}",
                        "python3", code_file
                    ]
                    process = subprocess.Popen(
                        cmd,
                        cwd=temp_dir,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid if os.name == 'posix' else None
                    )
                except FileNotFoundError:
                    # Fallback without prlimit
                    logger.warning("prlimit not available, using basic resource limits")
                    cmd = ["python3", code_file]
                    process = subprocess.Popen(
                        cmd,
                        cwd=temp_dir,
                        env=env,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        preexec_fn=os.setsid if os.name == 'posix' else None
                    )
            else:  # Windows or other systems
                cmd = ["python3", code_file]
                process = subprocess.Popen(
                    cmd,
                    cwd=temp_dir,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
            
            self.processes[sandbox_id] = process
            
            logger.info(f"Created process sandbox {process.pid} for simulation {sandbox_id}")
            return process.pid
            
        except Exception as e:
            logger.error(f"Failed to create process sandbox for {sandbox_id}: {e}")
            await self.cleanup_sandbox(sandbox_id)
            raise
    
    async def execute_simulation(self, sandbox_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Execute simulation in process sandbox"""
        if sandbox_id not in self.processes:
            raise ValueError(f"Process sandbox {sandbox_id} not found")
        
        process = self.processes[sandbox_id]
        
        try:
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                raise TimeoutError(f"Process sandbox {sandbox_id} timed out")
            
            # Decode output
            stdout_str = stdout.decode('utf-8') if stdout else ""
            stderr_str = stderr.decode('utf-8') if stderr else ""
            
            # Try to read output file
            output_file = os.path.join(self.temp_dirs[sandbox_id], "output.json")
            output_data = {}
            if os.path.exists(output_file):
                try:
                    with open(output_file, "r") as f:
                        output_data = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to read output file: {e}")
            
            execution_result = {
                "exit_code": exit_code,
                "stdout": stdout_str,
                "stderr": stderr_str,
                "output": output_data,
                "success": exit_code == 0,
                "process_id": process.pid
            }
            
            logger.info(f"Process sandbox execution completed for {sandbox_id}: exit_code={exit_code}")
            return execution_result
            
        except Exception as e:
            logger.error(f"Process sandbox execution failed for {sandbox_id}: {e}")
            raise
        finally:
            await self.cleanup_sandbox(sandbox_id)
    
    async def cleanup_sandbox(self, sandbox_id: str):
        """Clean up process sandbox resources"""
        try:
            # Terminate process
            if sandbox_id in self.processes:
                process = self.processes[sandbox_id]
                try:
                    if process.poll() is None:  # Still running
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                except Exception as e:
                    logger.warning(f"Failed to terminate process {process.pid}: {e}")
                del self.processes[sandbox_id]
            
            # Clean up temporary directory
            if sandbox_id in self.temp_dirs:
                temp_dir = self.temp_dirs[sandbox_id]
                try:
                    shutil.rmtree(temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to remove temp dir {temp_dir}: {e}")
                del self.temp_dirs[sandbox_id]
            
            logger.info(f"Cleaned up process sandbox {sandbox_id}")
            
        except Exception as e:
            logger.error(f"Error during process sandbox cleanup for {sandbox_id}: {e}")
    
    async def cleanup_all(self):
        """Clean up all process sandbox resources"""
        sandbox_ids = list(self.processes.keys()) + list(self.temp_dirs.keys())
        for sandbox_id in sandbox_ids:
            await self.cleanup_sandbox(sandbox_id)


class SandboxManager:
    """Unified sandbox manager supporting multiple isolation levels"""
    
    def __init__(self, default_isolation: str = "container"):
        self.default_isolation = default_isolation
        self.docker_sandbox = None
        self.process_sandbox = ProcessSandbox()
        self.active_sandboxes: Dict[str, Dict[str, Any]] = {}
        
        # Initialize Docker sandbox only if needed
        if default_isolation == "container":
            try:
                self.docker_sandbox = DockerSandbox()
            except Exception as e:
                logger.warning(f"Docker not available, falling back to process isolation: {e}")
                self.default_isolation = "process"
        
    async def create_simulation_sandbox(self, simulation_id: str, simulation_code: str,
                                      config: SandboxConfig = None) -> str:
        """Create appropriate sandbox for simulation"""
        config = config or SandboxConfig(isolation_level=self.default_isolation)
        
        sandbox_id = f"sandbox_{simulation_id}_{int(time.time())}"
        
        try:
            if config.isolation_level == "container" and self.docker_sandbox:
                container_id = await self.docker_sandbox.create_sandbox(
                    sandbox_id, simulation_code
                )
                sandbox_type = "docker"
                sandbox_handle = container_id
                
            elif config.isolation_level == "process" or (config.isolation_level == "container" and not self.docker_sandbox):
                process_id = await self.process_sandbox.create_sandbox(
                    sandbox_id, simulation_code
                )
                sandbox_type = "process"
                sandbox_handle = process_id
                
            else:  # "none" or unsupported
                logger.warning(f"Unsupported isolation level {config.isolation_level}, falling back to process isolation")
                process_id = await self.process_sandbox.create_sandbox(
                    sandbox_id, simulation_code
                )
                sandbox_type = "process"
                sandbox_handle = process_id
            
            self.active_sandboxes[sandbox_id] = {
                "simulation_id": simulation_id,
                "type": sandbox_type,
                "handle": sandbox_handle,
                "config": config,
                "created_at": time.time()
            }
            
            logger.info(f"Created {sandbox_type} sandbox {sandbox_id} for simulation {simulation_id}")
            return sandbox_id
            
        except Exception as e:
            logger.error(f"Failed to create sandbox for simulation {simulation_id}: {e}")
            raise
    
    async def execute_in_sandbox(self, sandbox_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Execute simulation in sandbox"""
        if sandbox_id not in self.active_sandboxes:
            raise ValueError(f"Sandbox {sandbox_id} not found")
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        sandbox_type = sandbox_info["type"]
        
        try:
            if sandbox_type == "docker":
                result = await self.docker_sandbox.execute_simulation(sandbox_id, timeout)
            elif sandbox_type == "process":
                result = await self.process_sandbox.execute_simulation(sandbox_id, timeout)
            else:
                raise ValueError(f"Unsupported sandbox type: {sandbox_type}")
            
            # Add sandbox metadata to result
            result["sandbox_type"] = sandbox_type
            result["sandbox_config"] = sandbox_info["config"]
            result["isolation_level"] = sandbox_info["config"].isolation_level
            
            return result
            
        except Exception as e:
            logger.error(f"Sandbox execution failed for {sandbox_id}: {e}")
            raise
        finally:
            # Clean up sandbox info
            if sandbox_id in self.active_sandboxes:
                del self.active_sandboxes[sandbox_id]
    
    def get_sandbox_stats(self, sandbox_id: str) -> Dict[str, Any]:
        """Get sandbox resource statistics"""
        if sandbox_id not in self.active_sandboxes:
            return {}
        
        sandbox_info = self.active_sandboxes[sandbox_id]
        sandbox_type = sandbox_info["type"]
        
        if sandbox_type == "docker" and self.docker_sandbox:
            return self.docker_sandbox.get_container_stats(sandbox_id)
        elif sandbox_type == "process":
            # Get actual process statistics
            try:
                import psutil
                process = psutil.Process(sandbox_info["process"].pid)
                return {
                    "type": "process",
                    "pid": process.pid,
                    "cpu_percent": process.cpu_percent(),
                    "memory_percent": process.memory_percent(),
                    "memory_info": process.memory_info()._asdict(),
                    "status": process.status(),
                    "create_time": process.create_time(),
                    "num_threads": process.num_threads()
                }
            except ImportError:
                return {"type": "process", "note": "psutil not available for process stats"}
            except Exception as e:
                return {"type": "process", "error": f"Failed to get process stats: {e}"}
        
        return {}
    
    async def cleanup_sandbox(self, sandbox_id: str):
        """Clean up specific sandbox"""
        if sandbox_id in self.active_sandboxes:
            sandbox_info = self.active_sandboxes[sandbox_id]
            sandbox_type = sandbox_info["type"]
            
            if sandbox_type == "docker" and self.docker_sandbox:
                await self.docker_sandbox.cleanup_sandbox(sandbox_id)
            elif sandbox_type == "process":
                await self.process_sandbox.cleanup_sandbox(sandbox_id)
            
            del self.active_sandboxes[sandbox_id]
    
    async def cleanup_all(self):
        """Clean up all sandboxes"""
        sandbox_ids = list(self.active_sandboxes.keys())
        for sandbox_id in sandbox_ids:
            await self.cleanup_sandbox(sandbox_id)
        
        if self.docker_sandbox:
            await self.docker_sandbox.cleanup_all()
        await self.process_sandbox.cleanup_all()


@dataclass
class ResourceLimits:
    """Resource limits for simulation execution"""
    cpu_percent: float = 80.0
    memory_mb: int = 1024
    disk_mb: int = 5120


@dataclass
class AsyncTask:
    """Async task with timeout and resource management"""
    task_id: str
    coroutine: Callable
    timeout: float
    resource_limits: ResourceLimits
    start_time: float = None
    task_handle: asyncio.Task = None
    cancelled: bool = False


class AsyncTaskManager:
    """
    Manages async execution with timeouts and resource quotas
    """
    
    def __init__(self, max_concurrent_tasks: int = 5, default_timeout: float = 3600, 
                 resource_limits: Dict[str, Any] = None):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.default_timeout = default_timeout
        self.resource_limits = ResourceLimits(**resource_limits) if resource_limits else ResourceLimits()
        
        self.active_tasks: Dict[str, AsyncTask] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)
        self.task_counter = 0
        
        # Resource monitoring
        self.resource_monitor = ResourceMonitor(self.resource_limits)
        
    async def submit_task(self, coroutine: Callable, timeout: float = None, 
                         resource_limits: ResourceLimits = None) -> str:
        """Submit a task for async execution with timeout and resource management"""
        task_id = f"task_{self.task_counter}"
        self.task_counter += 1
        
        timeout = timeout or self.default_timeout
        resource_limits = resource_limits or self.resource_limits
        
        task = AsyncTask(
            task_id=task_id,
            coroutine=coroutine,
            timeout=timeout,
            resource_limits=resource_limits
        )
        
        self.active_tasks[task_id] = task
        
        # Submit to queue for processing
        await self.task_queue.put(task)
        
        # Start processing if not already running
        if not hasattr(self, '_processor_task') or self._processor_task.done():
            self._processor_task = asyncio.create_task(self._process_queue())
        
        return task_id
    
    async def _process_queue(self):
        """Process tasks from the queue with resource and concurrency management"""
        while True:
            try:
                # Get next task
                task = await self.task_queue.get()
                
                # Check resource availability
                if not await self.resource_monitor.check_resources(task.resource_limits):
                    logger.warning(f"Insufficient resources for task {task.task_id}, re-queuing")
                    await asyncio.sleep(1)  # Wait before retry
                    await self.task_queue.put(task)
                    continue
                
                # Execute task with semaphore and timeout
                asyncio.create_task(self._execute_task(task))
                
            except Exception as e:
                logger.exception(f"Error processing task queue: {e}")
    
    async def _execute_task(self, task: AsyncTask):
        """Execute a single task with timeout"""
        async with self.semaphore:
            try:
                task.start_time = time.time()
                
                # Create task with timeout
                task.task_handle = asyncio.create_task(task.coroutine())
                
                # Wait for completion or timeout
                try:
                    result = await asyncio.wait_for(task.task_handle, timeout=task.timeout)
                    logger.info(f"Task {task.task_id} completed successfully")
                    return result
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Task {task.task_id} timed out after {task.timeout}s")
                    task.cancelled = True
                    if not task.task_handle.done():
                        task.task_handle.cancel()
                    raise
                    
            except Exception as e:
                logger.exception(f"Task {task.task_id} failed: {e}")
                raise
                
            finally:
                # Cleanup
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.cancelled = True
            if task.task_handle and not task.task_handle.done():
                task.task_handle.cancel()
                return True
        return False
    
    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get status of a task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "task_id": task_id,
                "running": not task.task_handle.done() if task.task_handle else False,
                "cancelled": task.cancelled,
                "start_time": task.start_time,
                "elapsed": time.time() - task.start_time if task.start_time else 0
            }
        return {"task_id": task_id, "status": "not_found"}
    
    def get_active_tasks(self) -> List[str]:
        """Get list of active task IDs"""
        return list(self.active_tasks.keys())


class ResourceMonitor:
    """
    Monitors system resources to prevent over-utilization
    """
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        
    async def check_resources(self, required: ResourceLimits) -> bool:
        """Check if required resources are available"""
        try:
            # Check CPU
            if required.cpu_percent > self.limits.cpu_percent:
                return False
                
            # Check memory (simplified - in real implementation would check actual usage)
            if required.memory_mb > self.limits.memory_mb:
                return False
                
            # Check disk space (simplified)
            if required.disk_mb > self.limits.disk_mb:
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Resource check failed: {e}")
            return False


@dataclass
class SimulationScenario:
    """Simulation scenario definition"""
    scenario_id: str
    name: str
    description: str
    parameters: Dict[str, Any]
    duration: int  # simulation time units
    time_step: float
    entities: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]


class SafetyPolicyEngine:
    """
    Safety policy engine for simulation governance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.policies: Dict[str, SafetyPolicy] = {}
        self.assessments: Dict[str, SafetyAssessment] = {}
        self.logger = logging.getLogger("kalki.safety")
        
        # Initialize default safety policies
        self._initialize_default_policies()
        
    def _initialize_default_policies(self):
        """Initialize default safety policies for biology/chemistry simulations"""
        
        # High-risk biology policy
        self.policies["biology_high_risk"] = SafetyPolicy(
            policy_id="biology_high_risk",
            name="High-Risk Biology Simulation",
            description="Policy for high-risk biological simulations",
            risk_level="critical",
            requires_approval=True,
            approval_authority="biosafety_officer",
            prohibited_keywords=[
                "virus", "pathogen", "toxin", "bioweapon", "pandemic",
                "genetic_modification", "crispr", "synthetic_biology",
                "bioterrorism", "contagious_disease"
            ],
            warning_keywords=[
                "dna", "rna", "protein", "enzyme", "mutation",
                "evolution", "epidemic", "infection"
            ],
            max_execution_time=1800,  # 30 minutes
            monitoring_required=True
        )
        
        # Chemistry hazard policy
        self.policies["chemistry_hazard"] = SafetyPolicy(
            policy_id="chemistry_hazard",
            name="Chemical Hazard Simulation",
            description="Policy for hazardous chemical simulations",
            risk_level="high",
            requires_approval=True,
            approval_authority="chemistry_safety_officer",
            prohibited_keywords=[
                "explosive", "toxic", "carcinogen", "radioactive",
                "chemical_weapon", "nerve_agent", "poison",
                "environmental_toxin", "heavy_metal"
            ],
            warning_keywords=[
                "chemical_reaction", "synthesis", "compound",
                "catalyst", "solvent", "acid", "base", "organic"
            ],
            max_execution_time=3600,  # 1 hour
            monitoring_required=True
        )
        
        # Medium-risk general policy
        self.policies["general_medium_risk"] = SafetyPolicy(
            policy_id="general_medium_risk",
            name="Medium-Risk General Simulation",
            description="Policy for medium-risk general simulations",
            risk_level="medium",
            requires_approval=False,
            approval_authority="supervisor",
            prohibited_keywords=[],
            warning_keywords=[
                "risk", "hazard", "danger", "safety", "accident"
            ],
            max_execution_time=7200,  # 2 hours
            monitoring_required=False
        )
        
        # Low-risk policy
        self.policies["low_risk"] = SafetyPolicy(
            policy_id="low_risk",
            name="Low-Risk Simulation",
            description="Policy for low-risk simulations",
            risk_level="low",
            requires_approval=False,
            approval_authority="none",
            prohibited_keywords=[],
            warning_keywords=[],
            max_execution_time=14400,  # 4 hours
            monitoring_required=False
        )
    
    async def assess_scenario_safety(self, scenario: SimulationScenario) -> SafetyAssessment:
        """Assess safety of a simulation scenario"""
        assessment_id = f"assessment_{int(time.time())}_{scenario.scenario_id}"
        
        violations = []
        warnings = []
        recommendations = []
        max_risk_level = "low"
        requires_approval = False
        
        # Analyze scenario description and parameters
        scenario_text = f"{scenario.name} {scenario.description}".lower()
        
        for policy in self.policies.values():
            # Check prohibited keywords
            for keyword in policy.prohibited_keywords:
                if keyword.lower() in scenario_text:
                    violations.append(f"Prohibited keyword '{keyword}' found in {policy.name}")
                    max_risk_level = max(max_risk_level, policy.risk_level, key=self._risk_level_value)
                    if policy.requires_approval:
                        requires_approval = True
            
            # Check warning keywords
            for keyword in policy.warning_keywords:
                if keyword.lower() in scenario_text:
                    warnings.append(f"Warning keyword '{keyword}' found - {policy.name} applies")
                    max_risk_level = max(max_risk_level, policy.risk_level, key=self._risk_level_value)
        
        # Additional checks for entities and constraints
        for entity in scenario.entities:
            entity_text = f"{entity.get('entity_type', '')} {entity.get('description', '')}".lower()
            for policy in self.policies.values():
                for keyword in policy.prohibited_keywords:
                    if keyword.lower() in entity_text:
                        violations.append(f"Prohibited keyword '{keyword}' in entity: {entity.get('entity_id', 'unknown')}")
        
        # Generate recommendations
        if max_risk_level in ["high", "critical"]:
            recommendations.append("Human oversight required for high-risk simulation")
            recommendations.append("Consider using sandbox environment")
        if requires_approval:
            recommendations.append("Approval required before execution")
        
        assessment = SafetyAssessment(
            assessment_id=assessment_id,
            scenario_id=scenario.scenario_id,
            risk_level=max_risk_level,
            requires_approval=requires_approval,
            assessment_time=time.time(),
            violations=violations,
            warnings=warnings,
            recommendations=recommendations
        )
        
        self.assessments[assessment_id] = assessment
        
        # Log safety assessment
        self.logger.info(f"Safety assessment {assessment_id}: risk={max_risk_level}, approval={requires_approval}")
        if violations:
            self.logger.warning(f"Safety violations in {scenario.scenario_id}: {violations}")
        
        return assessment
    
    def _risk_level_value(self, level: str) -> int:
        """Convert risk level to numeric value for comparison"""
        levels = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return levels.get(level, 0)
    
    async def approve_assessment(self, assessment_id: str, approved_by: str) -> bool:
        """Approve a safety assessment"""
        if assessment_id in self.assessments:
            assessment = self.assessments[assessment_id]
            assessment.approval_granted = True
            assessment.approved_by = approved_by
            
            self.logger.info(f"Safety assessment {assessment_id} approved by {approved_by}")
            return True
        return False
    
    def get_assessment(self, assessment_id: str) -> Optional[SafetyAssessment]:
        """Get a safety assessment"""
        return self.assessments.get(assessment_id)


@dataclass
class SimulationResult:
    """Simulation run result"""
    scenario_id: str
    run_id: str
    duration: float
    metrics: Dict[str, List[float]]
    events: List[Dict[str, Any]]
    final_state: Dict[str, Any]
    performance_stats: Dict[str, Any]


@dataclass
class SimulationEntity:
    """Simulated entity with state and behavior"""
    entity_id: str
    entity_type: str
    state: Dict[str, Any]
    behavior_rules: List[Dict[str, Any]]
    interactions: List[Dict[str, Any]]


class SimulationAgent(BaseAgent):
    """
    Advanced simulation agent using multiple simulation methodologies.
    Implements Monte Carlo, discrete event, and agent-based simulations.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="SimulationAgent",
            capabilities=[
                AgentCapability.SIMULATION,
                AgentCapability.EXPERIMENTATION,
                AgentCapability.PREDICTIVE_DISCOVERY
            ],
            description="Multi-methodology system simulation and scenario modeling",
            config=config or {}
        )

        # Simulation parameters
        self.max_simulation_time = self.config.get('max_simulation_time', 3600)  # 1 hour
        self.default_time_step = self.config.get('default_time_step', 1.0)
        self.max_parallel_simulations = self.config.get('max_parallel_simulations', 5)

        # Simulation state
        self.active_simulations = {}
        self.simulation_history = defaultdict(list)
        self.scenarios = {}

        # Random number generators for reproducibility
        self.rng = np.random.RandomState(42)

        # Performance tracking
        self.simulation_stats = {
            "total_runs": 0,
            "successful_runs": 0,
            "failed_runs": 0,
            "average_duration": 0,
            "total_computation_time": 0
        }

        # Persistence and checkpointing
        self.rollback_manager = RollbackManager(self.config.get('rollback_config', {}))
        self.checkpoint_interval = self.config.get('checkpoint_interval', 300)  # 5 minutes

        # Async execution and resource management
        self.task_manager = AsyncTaskManager(
            max_concurrent_tasks=self.config.get('max_concurrent_simulations', 5),
            default_timeout=self.config.get('default_simulation_timeout', 3600),
            resource_limits=self.config.get('resource_limits', {
                'cpu_percent': 80.0,
                'memory_mb': 1024,
                'disk_mb': 5120
            })
        )

        # Safety and compliance
        self.safety_engine = SafetyPolicyEngine(self.config.get('safety_config', {}))
        
        # Metrics and telemetry
        self.metrics_collector = MetricsCollector()
        self.enable_metrics = self.config.get('enable_metrics', True)
        
        # Sandbox and security
        self.sandbox_manager = SandboxManager(
            default_isolation=self.config.get('default_sandbox_isolation', 'container')
        )
        self.enable_sandbox = self.config.get('enable_sandbox', True)
        
        # Input validation
        self.input_validator = InputValidator()
        self.enable_validation = self.config.get('enable_validation', True)
        
        # Retry mechanisms
        self.retry_manager = RetryManager(
            max_retries=self.config.get('max_retries', 3),
            base_delay=self.config.get('retry_base_delay', 1.0),
            max_delay=self.config.get('retry_max_delay', 60.0),
            backoff_factor=self.config.get('retry_backoff_factor', 2.0),
            circuit_breaker_threshold=self.config.get('circuit_breaker_threshold', 5),
            circuit_breaker_timeout=self.config.get('circuit_breaker_timeout', 300.0)
        )
        self.enable_retry = self.config.get('enable_retry', True)
        
        # Pluggable backends
        self.backend_manager = BackendManager()
        self.default_backend = self.config.get('default_backend', 'montecarlo')

    async def initialize(self) -> bool:
        """Initialize simulation environment"""
        try:
            logger.info("SimulationAgent initializing multi-methodology simulation engine")

            # Initialize rollback manager
            rollback_initialized = await self.rollback_manager.initialize()
            if not rollback_initialized:
                logger.warning("Failed to initialize rollback manager, proceeding without persistence")
            
            # Initialize task manager
            await self.task_manager.resource_monitor.check_resources(self.task_manager.resource_limits)
            
            # Initialize default scenarios
            await self._initialize_default_scenarios()

            # Initialize backends
            await self.backend_manager.initialize_backends()
            
            # Warm up simulation engine
            await self._warm_up_engine()

            logger.info("SimulationAgent initialized with Monte Carlo and discrete event capabilities")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize SimulationAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute simulation operations"""
        action = task.get("action")
        params = task.get("params", {})
        
        # Check if backend is specified
        backend_name = params.get("backend", self.default_backend)
        
        if action == "run_monte_carlo":
            return await self._run_monte_carlo_simulation(params, backend_name)
        elif action == "run_discrete_event":
            return await self._run_discrete_event_simulation(params, backend_name)
        elif action == "run_agent_based":
            return await self._run_agent_based_simulation(params, backend_name)
        elif action == "create_scenario":
            return await self._create_simulation_scenario(params)
        elif action == "analyze_results":
            return await self._analyze_simulation_results(params)
        elif action == "optimize_scenario":
            return await self._optimize_simulation_scenario(params)
        elif action == "list_backends":
            return await self._list_available_backends()
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _run_monte_carlo_simulation(self, params: Dict[str, Any], backend_name: str = None) -> Dict[str, Any]:
        """Run Monte Carlo simulation with probabilistic modeling"""
        try:
            # Input validation
            if self.enable_validation:
                is_valid, errors, validated_params = self.input_validator.validate_simulation_request({
                    "scenario": params,
                    "parameters": params,
                    "sandbox_config": params.get("sandbox_config"),
                    "metrics_config": params.get("metrics_config"),
                    "safety_policies": params.get("safety_policies", [])
                })
                
                if not is_valid:
                    await self.emit_event("simulation.validation_failed", {
                        "simulation_type": "monte_carlo",
                        "errors": errors,
                        "params": params
                    })
                    return {
                        "status": "error",
                        "error": "Input validation failed",
                        "validation_errors": errors
                    }
                
                # Use validated parameters
                params = validated_params.get("parameters", params)

            scenario_id = params.get("scenario_id", "default")
            num_runs = params.get("num_runs", 1000)
            confidence_level = params.get("confidence_level", 0.95)

            if scenario_id not in self.scenarios:
                return {"status": "error", "error": f"Scenario {scenario_id} not found"}

            scenario = self.scenarios[scenario_id]

            # Perform safety assessment
            safety_assessment = await self.safety_engine.assess_scenario_safety(scenario)
            
            # Check if approval is required
            if safety_assessment.requires_approval and not safety_assessment.approval_granted:
                await self.emit_event("simulation.safety_blocked", {
                    "scenario_id": scenario_id,
                    "assessment_id": safety_assessment.assessment_id,
                    "risk_level": safety_assessment.risk_level,
                    "violations": safety_assessment.violations,
                    "warnings": safety_assessment.warnings
                })
                return {
                    "status": "blocked",
                    "error": "Simulation requires safety approval",
                    "assessment": safety_assessment
                }

            # Emit safety assessment event
            await self.emit_event("simulation.safety_assessed", {
                "scenario_id": scenario_id,
                "assessment_id": safety_assessment.assessment_id,
                "risk_level": safety_assessment.risk_level,
                "approved": safety_assessment.approval_granted,
                "violations_count": len(safety_assessment.violations),
                "warnings_count": len(safety_assessment.warnings)
            })

            # Start metrics collection
            simulation_id = f"mc_{scenario_id}_{int(time.time())}"
            if self.enable_metrics:
                metrics = self.metrics_collector.start_collection(
                    simulation_id=simulation_id,
                    scenario_id=scenario_id,
                    simulation_type="monte_carlo"
                )

            # Execute simulation using specified backend
            backend_name = backend_name or self.default_backend
            
            # Define backend execution function for retry
            async def execute_with_backend():
                return await self.backend_manager.execute_with_backend(
                    backend_name, scenario, {
                        "num_runs": num_runs,
                        "confidence_level": confidence_level,
                        "random_seed": params.get("random_seed"),
                        "max_parallel_runs": params.get("max_parallel_runs", 10),
                        "timeout_seconds": params.get("timeout_seconds", 3600)
                    }
                )
            
            # Execute with retry
            if self.enable_retry:
                try:
                    backend_result = await self.retry_manager.execute_with_retry(
                        execute_with_backend
                    )
                except CircuitBreakerOpenError:
                    await self.emit_event("simulation.circuit_breaker_open", {
                        "simulation_type": "monte_carlo",
                        "scenario_id": scenario_id,
                        "backend": backend_name,
                        "retry_stats": self.retry_manager.get_stats()
                    })
                    return {
                        "status": "error",
                        "error": "Circuit breaker is open - too many recent failures",
                        "retry_stats": self.retry_manager.get_stats()
                    }
            else:
                backend_result = await execute_with_backend()
            
            # Check backend result
            if backend_result["status"] != "success":
                raise RuntimeError(f"Backend execution failed: {backend_result.get('error', 'Unknown error')}")
            
            # Extract results from backend
            results = backend_result.get("results", [])
            analysis = backend_result.get("statistics", {})
            
            # Update metrics
            if self.enable_metrics and metrics:
                metrics.backend_used = backend_name
                metrics.simulation_method = backend_result.get("method", "unknown")
                if "statistics" in backend_result:
                    stats = backend_result["statistics"]
                    metrics.mean_outcome = stats.get("mean")
                    metrics.std_outcome = stats.get("std")
                    metrics.confidence_interval = stats.get("confidence_interval")
            simulation_state = {
                "scenario_id": scenario_id,
                "simulation_type": "monte_carlo",
                "num_runs": num_runs,
                "start_time": time.time(),
                "active_simulations": dict(self.active_simulations),
                "simulation_stats": dict(self.simulation_stats),
                "safety_assessment": {
                    "assessment_id": safety_assessment.assessment_id,
                    "risk_level": safety_assessment.risk_level,
                    "approved": safety_assessment.approval_granted
                }
            }
            
            checkpoint_id = await self.rollback_manager.create_checkpoint(
                name=f"pre_monte_carlo_{scenario_id}",
                state=simulation_state,
                tags=["simulation", "monte_carlo", "checkpoint"],
                metadata={"scenario_name": scenario.name, "num_runs": num_runs},
                source_agent=self.name,
                description=f"Pre-simulation checkpoint for Monte Carlo scenario {scenario_id}"
            )

            # Emit simulation start event
            await self.emit_event("simulation.started", {
                "simulation_type": "monte_carlo",
                "scenario_id": scenario_id,
                "num_runs": num_runs,
                "confidence_level": confidence_level,
                "scenario_name": scenario.name,
                "checkpoint_id": checkpoint_id
            })

            # Run Monte Carlo simulations
            results = []
            start_time = time.time()
            progress_interval = max(1, num_runs // 10)  # Report progress every 10%

            for run_id in range(num_runs):
                if time.time() - start_time > self.max_simulation_time:
                    # Emit timeout event
                    await self.emit_event("simulation.timeout", {
                        "simulation_type": "monte_carlo",
                        "scenario_id": scenario_id,
                        "runs_completed": len(results),
                        "total_runs": num_runs,
                        "elapsed_time": time.time() - start_time
                    })
                    break

                result = await self._execute_monte_carlo_run(scenario, run_id)
                results.append(result)

                # Update metrics
                if self.enable_metrics and run_id % 10 == 0:  # Record every 10 runs
                    metrics = self.metrics_collector.get_simulation_metrics(simulation_id)
                    if metrics:
                        metrics.entities_processed += 10
                        metrics.record_system_metrics()

                # Periodic checkpointing
                current_time = time.time()
                if current_time - start_time > self.checkpoint_interval and (run_id + 1) % progress_interval == 0:
                    progress_state = {
                        "scenario_id": scenario_id,
                        "simulation_type": "monte_carlo",
                        "runs_completed": len(results),
                        "total_runs": num_runs,
                        "start_time": start_time,
                        "current_time": current_time,
                        "partial_results": results[-100:],  # Last 100 results
                        "active_simulations": dict(self.active_simulations)
                    }
                    
                    await self.rollback_manager.create_checkpoint(
                        name=f"progress_monte_carlo_{scenario_id}_{len(results)}",
                        state=progress_state,
                        tags=["simulation", "monte_carlo", "progress"],
                        metadata={"scenario_name": scenario.name, "progress": len(results)/num_runs},
                        source_agent=self.name,
                        description=f"Progress checkpoint for Monte Carlo simulation {scenario_id} at {len(results)} runs"
                    )

                # Emit progress event
                if (run_id + 1) % progress_interval == 0:
                    progress = (run_id + 1) / num_runs
                    await self.emit_event("simulation.progress", {
                        "simulation_type": "monte_carlo",
                        "scenario_id": scenario_id,
                        "progress": progress,
                        "runs_completed": run_id + 1,
                        "total_runs": num_runs,
                        "elapsed_time": time.time() - start_time
                    })

            # Analyze results
            analysis = self._analyze_monte_carlo_results(results, confidence_level)

            # Stop metrics collection
            final_metrics = None
            if self.enable_metrics:
                final_metrics = self.metrics_collector.stop_collection(simulation_id)
                if final_metrics:
                    # Update final metrics with results
                    final_metrics.mean_outcome = analysis.get('mean', 0)
                    final_metrics.std_outcome = analysis.get('std', 0)
                    final_metrics.confidence_interval = analysis.get('confidence_interval')
                    final_metrics.steps_completed = len(results)

            # Store results
            simulation_result = {
                "scenario_id": scenario_id,
                "method": "monte_carlo",
                "num_runs": len(results),
                "total_time": time.time() - start_time,
                "results": results,
                "analysis": analysis,
                "metrics": final_metrics.to_dict() if final_metrics else None
            }

            self.simulation_history[scenario_id].append(simulation_result)
            self._update_simulation_stats(len(results), time.time() - start_time)

            # Create completion checkpoint
            completion_state = {
                "scenario_id": scenario_id,
                "simulation_type": "monte_carlo",
                "completed_at": time.time(),
                "simulation_result": simulation_result,
                "simulation_stats": dict(self.simulation_stats),
                "analysis": analysis
            }
            
            completion_checkpoint_id = await self.rollback_manager.create_checkpoint(
                name=f"completed_monte_carlo_{scenario_id}",
                state=completion_state,
                tags=["simulation", "monte_carlo", "completed"],
                metadata={"scenario_name": scenario.name, "total_runs": len(results), "success": True},
                source_agent=self.name,
                description=f"Completion checkpoint for successful Monte Carlo simulation {scenario_id}"
            )

            # Emit simulation completion event
            await self.emit_event("simulation.completed", {
                "simulation_type": "monte_carlo",
                "scenario_id": scenario_id,
                "runs_completed": len(results),
                "total_time": time.time() - start_time,
                "analysis_summary": {
                    "mean_outcome": analysis.get("mean", 0),
                    "confidence_interval": analysis.get("confidence_interval", [0, 0]),
                    "standard_error": analysis.get("standard_error", 0)
                },
                "checkpoint_id": completion_checkpoint_id
            })

            return {
                "status": "success",
                "simulation_result": simulation_result
            }

        except Exception as e:
            logger.exception(f"Monte Carlo simulation error: {e}")
            
            # Create error checkpoint for recovery
            error_state = {
                "scenario_id": params.get("scenario_id", "unknown"),
                "simulation_type": "monte_carlo",
                "error_time": time.time(),
                "error": str(e),
                "error_type": type(e).__name__,
                "active_simulations": dict(self.active_simulations),
                "simulation_stats": dict(self.simulation_stats),
                "partial_results": locals().get("results", []) if "results" in locals() else []
            }
            
            await self.rollback_manager.create_checkpoint(
                name=f"error_monte_carlo_{params.get('scenario_id', 'unknown')}",
                state=error_state,
                tags=["simulation", "monte_carlo", "error"],
                metadata={"error_type": type(e).__name__, "scenario_id": params.get("scenario_id", "unknown")},
                source_agent=self.name,
                description=f"Error checkpoint for failed Monte Carlo simulation"
            )
            
            # Emit simulation error event
            await self.emit_event("simulation.error", {
                "simulation_type": "monte_carlo",
                "scenario_id": params.get("scenario_id", "unknown"),
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {"status": "error", "error": str(e)}

    async def _run_discrete_event_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run discrete event simulation"""
        try:
            # Input validation
            if self.enable_validation:
                is_valid, errors, validated_params = self.input_validator.validate_simulation_request({
                    "scenario": params,
                    "parameters": params,
                    "sandbox_config": params.get("sandbox_config"),
                    "metrics_config": params.get("metrics_config"),
                    "safety_policies": params.get("safety_policies", [])
                })
                
                if not is_valid:
                    await self.emit_event("simulation.validation_failed", {
                        "simulation_type": "discrete_event",
                        "errors": errors,
                        "params": params
                    })
                    return {
                        "status": "error",
                        "error": "Input validation failed",
                        "validation_errors": errors
                    }
                
                # Use validated parameters
                params = validated_params.get("parameters", params)

            scenario_id = params.get("scenario_id", "default")
            max_events = params.get("max_events", 10000)

            if scenario_id not in self.scenarios:
                return {"status": "error", "error": f"Scenario {scenario_id} not found"}

            scenario = self.scenarios[scenario_id]

            # Emit simulation start event
            await self.emit_event("simulation.started", {
                "simulation_type": "discrete_event",
                "scenario_id": scenario_id,
                "max_events": max_events,
                "scenario_name": scenario.name
            })

            # Initialize event queue
            event_queue = self._initialize_event_queue(scenario)
            current_time = 0
            events_processed = 0

            # Simulation state
            sim_state = self._initialize_simulation_state(scenario)
            event_log = []

            start_time = time.time()
            progress_interval = max(1, max_events // 10)  # Report progress every 10%

            # Run simulation
            while event_queue and events_processed < max_events:
                if time.time() - start_time > self.max_simulation_time:
                    # Emit timeout event
                    await self.emit_event("simulation.timeout", {
                        "simulation_type": "discrete_event",
                        "scenario_id": scenario_id,
                        "events_processed": events_processed,
                        "max_events": max_events,
                        "simulation_time": current_time,
                        "elapsed_time": time.time() - start_time
                    })
                    break

                # Get next event
                event_time, event = event_queue.pop(0)
                current_time = event_time

                # Process event
                new_events, state_updates = await self._process_simulation_event(event, sim_state, current_time)

                # Update state
                sim_state.update(state_updates)

                # Add new events to queue
                for new_event in new_events:
                    self._schedule_event(event_queue, new_event)

                # Log event
                event_log.append({
                    "time": current_time,
                    "event": event,
                    "state_updates": state_updates
                })

                events_processed += 1

                # Emit progress event
                if events_processed % progress_interval == 0:
                    progress = events_processed / max_events
                    await self.emit_event("simulation.progress", {
                        "simulation_type": "discrete_event",
                        "scenario_id": scenario_id,
                        "progress": progress,
                        "events_processed": events_processed,
                        "max_events": max_events,
                        "simulation_time": current_time,
                        "elapsed_time": time.time() - start_time
                    })

                # Sort event queue by time
                event_queue.sort(key=lambda x: x[0])

            # Create result
            simulation_result = {
                "scenario_id": scenario_id,
                "method": "discrete_event",
                "events_processed": events_processed,
                "simulation_time": current_time,
                "real_time": time.time() - start_time,
                "final_state": sim_state,
                "event_log": event_log[-1000:]  # Last 1000 events
            }

            self.simulation_history[scenario_id].append(simulation_result)
            self._update_simulation_stats(1, time.time() - start_time)

            # Emit simulation completion event
            await self.emit_event("simulation.completed", {
                "simulation_type": "discrete_event",
                "scenario_id": scenario_id,
                "events_processed": events_processed,
                "simulation_time": current_time,
                "real_time": time.time() - start_time,
                "final_state_summary": {
                    "num_entities": len(sim_state.get("entities", {})),
                    "total_events": len(event_log)
                }
            })

            return {
                "status": "success",
                "simulation_result": simulation_result
            }

        except Exception as e:
            logger.exception(f"Discrete event simulation error: {e}")
            # Emit simulation error event
            await self.emit_event("simulation.error", {
                "simulation_type": "discrete_event",
                "scenario_id": params.get("scenario_id", "unknown"),
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {"status": "error", "error": str(e)}

    async def _run_agent_based_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Run agent-based simulation"""
        try:
            # Input validation
            if self.enable_validation:
                is_valid, errors, validated_params = self.input_validator.validate_simulation_request({
                    "scenario": params,
                    "parameters": params,
                    "sandbox_config": params.get("sandbox_config"),
                    "metrics_config": params.get("metrics_config"),
                    "safety_policies": params.get("safety_policies", [])
                })
                
                if not is_valid:
                    await self.emit_event("simulation.validation_failed", {
                        "simulation_type": "agent_based",
                        "errors": errors,
                        "params": params
                    })
                    return {
                        "status": "error",
                        "error": "Input validation failed",
                        "validation_errors": errors
                    }
                
                # Use validated parameters
                params = validated_params.get("parameters", params)

            scenario_id = params.get("scenario_id", "default")
            num_steps = params.get("num_steps", 100)

            if scenario_id not in self.scenarios:
                return {"status": "error", "error": f"Scenario {scenario_id} not found"}

            scenario = self.scenarios[scenario_id]

            # Emit simulation start event
            await self.emit_event("simulation.started", {
                "simulation_type": "agent_based",
                "scenario_id": scenario_id,
                "num_steps": num_steps,
                "scenario_name": scenario.name
            })

            # Initialize agents
            agents = self._initialize_simulation_agents(scenario)
            environment = self._initialize_environment(scenario)

            # Simulation loop
            step_results = []
            start_time = time.time()
            progress_interval = max(1, num_steps // 10)  # Report progress every 10%

            for step in range(num_steps):
                if time.time() - start_time > self.max_simulation_time:
                    # Emit timeout event
                    await self.emit_event("simulation.timeout", {
                        "simulation_type": "agent_based",
                        "scenario_id": scenario_id,
                        "steps_completed": len(step_results),
                        "total_steps": num_steps,
                        "elapsed_time": time.time() - start_time
                    })
                    break

                # Execute agent actions
                actions = []
                for agent in agents:
                    action = await self._execute_agent_action(agent, environment, step)
                    actions.append(action)

                # Update environment
                environment_updates = await self._update_environment(environment, actions, step)

                # Update agents
                for i, agent in enumerate(agents):
                    agent_updates = await self._update_agent_state(agent, actions[i], environment_updates, step)
                    agent.update(agent_updates)

                # Record step results
                step_result = {
                    "step": step,
                    "agent_states": [dict(agent) for agent in agents],
                    "environment_state": dict(environment),
                    "actions_taken": actions
                }
                step_results.append(step_result)

                # Emit progress event
                if (step + 1) % progress_interval == 0:
                    progress = (step + 1) / num_steps
                    await self.emit_event("simulation.progress", {
                        "simulation_type": "agent_based",
                        "scenario_id": scenario_id,
                        "progress": progress,
                        "steps_completed": step + 1,
                        "total_steps": num_steps,
                        "elapsed_time": time.time() - start_time
                    })

            # Create result
            simulation_result = {
                "scenario_id": scenario_id,
                "method": "agent_based",
                "steps_completed": len(step_results),
                "real_time": time.time() - start_time,
                "final_agent_states": [dict(agent) for agent in agents],
                "final_environment": dict(environment),
                "step_results": step_results[-100:]  # Last 100 steps
            }

            self.simulation_history[scenario_id].append(simulation_result)
            self._update_simulation_stats(1, time.time() - start_time)

            # Emit simulation completion event
            await self.emit_event("simulation.completed", {
                "simulation_type": "agent_based",
                "scenario_id": scenario_id,
                "steps_completed": len(step_results),
                "real_time": time.time() - start_time,
                "final_state_summary": {
                    "num_agents": len(agents),
                    "num_steps": len(step_results)
                }
            })

            return {
                "status": "success",
                "simulation_result": simulation_result
            }

        except Exception as e:
            logger.exception(f"Agent-based simulation error: {e}")
            # Emit simulation error event
            await self.emit_event("simulation.error", {
                "simulation_type": "agent_based",
                "scenario_id": params.get("scenario_id", "unknown"),
                "error": str(e),
                "error_type": type(e).__name__
            })
            return {"status": "error", "error": str(e)}

    async def _execute_monte_carlo_run(self, scenario: SimulationScenario, run_id: int) -> Dict[str, Any]:
        """Execute a single Monte Carlo run"""
        try:
            # Initialize random variables
            variables = {}
            for param_name, param_config in scenario.parameters.items():
                if param_config.get("distribution") == "normal":
                    variables[param_name] = self.rng.normal(
                        param_config.get("mean", 0),
                        param_config.get("std", 1)
                    )
                elif param_config.get("distribution") == "uniform":
                    variables[param_name] = self.rng.uniform(
                        param_config.get("min", 0),
                        param_config.get("max", 1)
                    )
                elif param_config.get("distribution") == "exponential":
                    variables[param_name] = self.rng.exponential(param_config.get("rate", 1))
                else:
                    variables[param_name] = param_config.get("default", 0)

            # Simulate scenario with these variables
            result = await self._simulate_scenario_with_variables(scenario, variables)

            return {
                "run_id": run_id,
                "variables": variables,
                "result": result
            }

        except Exception as e:
            logger.exception(f"Monte Carlo run error: {e}")
            return {"run_id": run_id, "error": str(e)}

    async def _simulate_scenario_with_variables(self, scenario: SimulationScenario,
                                              variables: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate scenario with given variables"""
        # Simplified scenario simulation
        # In a real implementation, this would be much more sophisticated

        # Simulate some outcome based on variables
        base_outcome = 100
        modifiers = []

        for var_name, var_value in variables.items():
            if "risk" in var_name.lower():
                modifiers.append(var_value * 0.1)
            elif "cost" in var_name.lower():
                modifiers.append(-var_value * 0.05)
            elif "benefit" in var_name.lower():
                modifiers.append(var_value * 0.08)

        total_modifier = sum(modifiers)
        final_outcome = base_outcome + total_modifier

        # Add some randomness
        final_outcome += self.rng.normal(0, 5)

        return {
            "outcome": final_outcome,
            "modifiers": modifiers,
            "variables_used": list(variables.keys())
        }

    def _initialize_event_queue(self, scenario: SimulationScenario) -> List[Tuple[float, Dict[str, Any]]]:
        """Initialize event queue for discrete event simulation"""
        event_queue = []

        # Add initial events
        for entity in scenario.entities:
            if entity.get("initial_event"):
                event = {
                    "type": "initial",
                    "entity_id": entity["id"],
                    "data": entity.get("initial_event")
                }
                event_queue.append((0, event))

        return event_queue

    def _initialize_simulation_state(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Initialize simulation state"""
        state = {
            "time": 0,
            "entities": {},
            "global_metrics": {}
        }

        for entity in scenario.entities:
            state["entities"][entity["id"]] = {
                "state": entity.get("initial_state", {}),
                "type": entity.get("type", "generic")
            }

        return state

    async def _process_simulation_event(self, event: Dict[str, Any], sim_state: Dict[str, Any],
                                      current_time: float) -> Tuple[List[Tuple[float, Dict[str, Any]]], Dict[str, Any]]:
        """Process a simulation event"""
        new_events = []
        state_updates = {}

        event_type = event.get("type")

        if event_type == "initial":
            # Initialize entity
            entity_id = event["entity_id"]
            state_updates[f"entities.{entity_id}.initialized"] = True

            # Schedule next event
            next_time = current_time + random.expovariate(0.1)
            next_event = {
                "type": "activity",
                "entity_id": entity_id,
                "data": {"activity": "random_action"}
            }
            new_events.append((next_time, next_event))

        elif event_type == "activity":
            # Process activity
            entity_id = event["entity_id"]
            activity = event["data"].get("activity")

            # Update metrics
            if "global_metrics" not in state_updates:
                state_updates["global_metrics"] = {}
            state_updates["global_metrics"]["activities_completed"] = \
                sim_state.get("global_metrics", {}).get("activities_completed", 0) + 1

            # Schedule next activity
            next_time = current_time + random.expovariate(0.2)
            next_event = {
                "type": "activity",
                "entity_id": entity_id,
                "data": {"activity": "another_action"}
            }
            new_events.append((next_time, next_event))

        return new_events, state_updates

    def _schedule_event(self, event_queue: List[Tuple[float, Dict[str, Any]]],
                       event_tuple: Tuple[float, Dict[str, Any]]):
        """Schedule an event in the queue"""
        event_queue.append(event_tuple)

    def _initialize_simulation_agents(self, scenario: SimulationScenario) -> List[Dict[str, Any]]:
        """Initialize agents for agent-based simulation"""
        agents = []

        for entity in scenario.entities:
            agent = {
                "id": entity["id"],
                "type": entity.get("type", "generic"),
                "state": entity.get("initial_state", {}),
                "goals": entity.get("goals", []),
                "capabilities": entity.get("capabilities", [])
            }
            agents.append(agent)

        return agents

    def _initialize_environment(self, scenario: SimulationScenario) -> Dict[str, Any]:
        """Initialize simulation environment"""
        return {
            "time": 0,
            "resources": scenario.parameters.get("resources", {}),
            "constraints": scenario.constraints,
            "global_state": {}
        }

    async def _execute_agent_action(self, agent: Dict[str, Any], environment: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Execute an agent action"""
        # Simple agent behavior
        action = {
            "agent_id": agent["id"],
            "type": "interact",
            "target": "environment",
            "parameters": {"step": step}
        }

        return action

    async def _update_environment(self, environment: Dict[str, Any], actions: List[Dict[str, Any]],
                                step: int) -> Dict[str, Any]:
        """Update environment based on actions"""
        updates = {
            "time": step + 1,
            "actions_processed": len(actions)
        }

        return updates

    async def _update_agent_state(self, agent: Dict[str, Any], action: Dict[str, Any], environment_updates: Dict[str, Any], step: int) -> Dict[str, Any]:
        """Update agent state"""
        updates = {
            "last_action": action["type"],
            "steps_taken": agent.get("steps_taken", 0) + 1
        }

        return updates

    def _analyze_monte_carlo_results(self, results: List[Dict[str, Any]],
                                   confidence_level: float) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        if not results:
            return {}

        # Extract outcomes
        outcomes = []
        for result in results:
            if "result" in result and "outcome" in result["result"]:
                outcomes.append(result["result"]["outcome"])

        if not outcomes:
            return {"error": "No valid outcomes found"}

        # Calculate statistics
        outcomes_array = np.array(outcomes)
        mean = np.mean(outcomes_array)
        std = np.std(outcomes_array)

        # Confidence interval
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_error = z_score * std / np.sqrt(len(outcomes))
        confidence_interval = (mean - margin_error, mean + margin_error)

        # Percentiles
        percentiles = {
            "10th": np.percentile(outcomes_array, 10),
            "25th": np.percentile(outcomes_array, 25),
            "50th": np.percentile(outcomes_array, 50),
            "75th": np.percentile(outcomes_array, 75),
            "90th": np.percentile(outcomes_array, 90)
        }

        return {
            "sample_size": len(outcomes),
            "mean": mean,
            "std_dev": std,
            "confidence_interval": confidence_interval,
            "confidence_level": confidence_level,
            "percentiles": percentiles,
            "min": np.min(outcomes_array),
            "max": np.max(outcomes_array)
        }

    async def _create_simulation_scenario(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new simulation scenario"""
        try:
            # Input validation
            if self.enable_validation:
                is_valid, errors, validated_params = self.input_validator.validate_simulation_request({
                    "scenario": params,
                    "parameters": params,
                    "sandbox_config": params.get("sandbox_config"),
                    "metrics_config": params.get("metrics_config"),
                    "safety_policies": params.get("safety_policies", [])
                })
                
                if not is_valid:
                    await self.emit_event("simulation.validation_failed", {
                        "action": "create_scenario",
                        "errors": errors,
                        "params": params
                    })
                    return {
                        "status": "error",
                        "error": "Input validation failed",
                        "validation_errors": errors
                    }
                
                # Use validated parameters
                params = validated_params.get("scenario", params)

            scenario = SimulationScenario(
                scenario_id=params.get("scenario_id", f"scenario_{len(self.scenarios)}"),
                name=params.get("name", "Unnamed Scenario"),
                description=params.get("description", ""),
                parameters=params.get("parameters", {}),
                duration=params.get("duration", 100),
                time_step=params.get("time_step", self.default_time_step),
                entities=params.get("entities", []),
                constraints=params.get("constraints", [])
            )

            self.scenarios[scenario.scenario_id] = scenario

            return {
                "status": "success",
                "scenario_id": scenario.scenario_id,
                "message": f"Scenario {scenario.scenario_id} created"
            }

        except Exception as e:
            logger.exception(f"Scenario creation error: {e}")
            return {"status": "error", "error": str(e)}

    async def _analyze_simulation_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze simulation results"""
        try:
            scenario_id = params.get("scenario_id")

            if scenario_id not in self.simulation_history:
                return {"status": "error", "error": f"No results found for scenario {scenario_id}"}

            results = self.simulation_history[scenario_id]

            # Aggregate analysis
            analysis = {
                "total_runs": len(results),
                "methods_used": list(set(r.get("method", "unknown") for r in results)),
                "performance_summary": self._summarize_performance(results)
            }

            return {
                "status": "success",
                "scenario_id": scenario_id,
                "analysis": analysis
            }

        except Exception as e:
            logger.exception(f"Results analysis error: {e}")
            return {"status": "error", "error": str(e)}

    async def _optimize_simulation_scenario(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize simulation scenario parameters"""
        try:
            scenario_id = params.get("scenario_id")
            optimization_target = params.get("target", "performance")

            if scenario_id not in self.scenarios:
                return {"status": "error", "error": f"Scenario {scenario_id} not found"}

            scenario = self.scenarios[scenario_id]

            # Simple parameter optimization (could use more sophisticated methods)
            optimized_params = {}
            for param_name, param_config in scenario.parameters.items():
                if param_config.get("optimizable", False):
                    # Try different values and find best
                    best_value = param_config.get("default", 0)
                    best_score = -float('inf')

                    for test_value in self._generate_test_values(param_config):
                        score = await self._evaluate_parameter_value(scenario, param_name, test_value, optimization_target)
                        if score > best_score:
                            best_score = score
                            best_value = test_value

                    optimized_params[param_name] = best_value

            return {
                "status": "success",
                "scenario_id": scenario_id,
                "optimized_parameters": optimized_params,
                "optimization_target": optimization_target
            }

        except Exception as e:
            logger.exception(f"Scenario optimization error: {e}")
            return {"status": "error", "error": str(e)}

    async def _execute_monte_carlo_with_timeout(self, scenario: SimulationScenario, num_runs: int, 
                                               confidence_level: float, scenario_id: str) -> Dict[str, Any]:
        """Execute Monte Carlo simulation with built-in timeout and resource management"""
        results = []
        start_time = time.time()
        progress_interval = max(1, num_runs // 10)  # Report progress every 10%
        last_checkpoint_time = start_time

        for run_id in range(num_runs):
            # Check for cancellation
            if hasattr(self.task_manager, 'active_tasks') and any(
                task.cancelled for task in self.task_manager.active_tasks.values()
            ):
                raise asyncio.CancelledError("Simulation was cancelled")

            result = await self._execute_monte_carlo_run(scenario, run_id)
            results.append(result)

            # Periodic checkpointing
            current_time = time.time()
            if current_time - last_checkpoint_time > self.checkpoint_interval:
                progress_state = {
                    "scenario_id": scenario_id,
                    "simulation_type": "monte_carlo",
                    "runs_completed": len(results),
                    "total_runs": num_runs,
                    "start_time": start_time,
                    "current_time": current_time,
                    "partial_results": results[-100:],  # Last 100 results
                    "active_simulations": dict(self.active_simulations)
                }
                
                await self.rollback_manager.create_checkpoint(
                    name=f"progress_monte_carlo_{scenario_id}_{len(results)}",
                    state=progress_state,
                    tags=["simulation", "monte_carlo", "progress"],
                    metadata={"scenario_name": scenario.name, "progress": len(results)/num_runs},
                    source_agent=self.name,
                    description=f"Progress checkpoint for Monte Carlo simulation {scenario_id} at {len(results)} runs"
                )
                last_checkpoint_time = current_time

            # Emit progress event
            if (run_id + 1) % progress_interval == 0:
                progress = (run_id + 1) / num_runs
                await self.emit_event("simulation.progress", {
                    "simulation_type": "monte_carlo",
                    "scenario_id": scenario_id,
                    "progress": progress,
                    "runs_completed": run_id + 1,
                    "total_runs": num_runs,
                    "elapsed_time": time.time() - start_time
                })

        # Analyze results
        analysis = self._analyze_monte_carlo_results(results, confidence_level)

        # Store results
        simulation_result = {
            "scenario_id": scenario_id,
            "method": "monte_carlo",
            "num_runs": len(results),
            "total_time": time.time() - start_time,
            "results": results,
            "analysis": analysis
        }

        self.simulation_history[scenario_id].append(simulation_result)
        self._update_simulation_stats(len(results), time.time() - start_time)

        # Create completion checkpoint
        completion_state = {
            "scenario_id": scenario_id,
            "simulation_type": "monte_carlo",
            "completed_at": time.time(),
            "simulation_result": simulation_result,
            "simulation_stats": dict(self.simulation_stats),
            "analysis": analysis
        }
        
        completion_checkpoint_id = await self.rollback_manager.create_checkpoint(
            name=f"completed_monte_carlo_{scenario_id}",
            state=completion_state,
            tags=["simulation", "monte_carlo", "completed"],
            metadata={"scenario_name": scenario.name, "total_runs": len(results), "success": True},
            source_agent=self.name,
            description=f"Completion checkpoint for successful Monte Carlo simulation {scenario_id}"
        )

        # Emit simulation completion event
        await self.emit_event("simulation.completed", {
            "simulation_type": "monte_carlo",
            "scenario_id": scenario_id,
            "runs_completed": len(results),
            "total_time": time.time() - start_time,
            "analysis_summary": {
                "mean_outcome": analysis.get("mean", 0),
                "confidence_interval": analysis.get("confidence_interval", [0, 0]),
                "standard_error": analysis.get("standard_error", 0)
            },
            "checkpoint_id": completion_checkpoint_id
        })

        return {
            "status": "success",
            "simulation_result": simulation_result
        }

    async def _wait_for_task_completion(self, task_id: str) -> Dict[str, Any]:
        """Wait for an async task to complete"""
        while True:
            status = await self.task_manager.get_task_status(task_id)
            if not status.get("running", False):
                # Task completed or failed
                if task_id in self.task_manager.active_tasks:
                    task = self.task_manager.active_tasks[task_id]
                    if task.task_handle and task.task_handle.done():
                        try:
                            return task.task_handle.result()
                        except Exception as e:
                            raise e
                break
            await asyncio.sleep(0.1)  # Poll every 100ms

    def _generate_test_values(self, param_config: Dict[str, Any]) -> List[float]:
        """Generate test values for parameter optimization"""
        param_type = param_config.get("type", "float")

        if param_type == "int":
            min_val = param_config.get("min", 0)
            max_val = param_config.get("max", 100)
            return list(range(min_val, max_val + 1, max(1, (max_val - min_val) // 10)))
        else:
            min_val = param_config.get("min", 0.0)
            max_val = param_config.get("max", 1.0)
            return [min_val + i * (max_val - min_val) / 10 for i in range(11)]

    async def _evaluate_parameter_value(self, scenario: SimulationScenario, param_name: str,
                                      value: float, target: str) -> float:
        """Evaluate a parameter value"""
        # Simplified evaluation - run a quick simulation
        test_scenario = scenario.parameters.copy()
        test_scenario[param_name] = value

        # Run quick Monte Carlo
        quick_results = []
        for _ in range(10):  # Quick test with 10 runs
            result = await self._simulate_scenario_with_variables(scenario, test_scenario)
            quick_results.append(result.get("outcome", 0))

        if target == "performance":
            return np.mean(quick_results)
        elif target == "stability":
            return -np.std(quick_results)  # Negative std dev for stability
        else:
            return np.mean(quick_results)

    async def _initialize_default_scenarios(self):
        """Initialize default simulation scenarios"""
        # Risk analysis scenario
        risk_scenario = SimulationScenario(
            scenario_id="risk_analysis",
            name="Financial Risk Analysis",
            description="Monte Carlo simulation for financial risk assessment",
            parameters={
                "market_volatility": {
                    "distribution": "normal",
                    "mean": 0.15,
                    "std": 0.05,
                    "optimizable": True
                },
                "interest_rate": {
                    "distribution": "uniform",
                    "min": 0.02,
                    "max": 0.08,
                    "optimizable": True
                }
            },
            duration=252,  # Trading days in a year
            time_step=1.0,
            entities=[],
            constraints=[]
        )

        # Queueing system scenario
        queue_scenario = SimulationScenario(
            scenario_id="queue_system",
            name="Service Queue Simulation",
            description="Discrete event simulation of service queues",
            parameters={
                "arrival_rate": {
                    "distribution": "exponential",
                    "rate": 0.5,
                    "default": 0.5
                },
                "service_rate": {
                    "distribution": "exponential",
                    "rate": 0.8,
                    "default": 0.8
                }
            },
            duration=1000,
            time_step=1.0,
            entities=[
                {"id": "server_1", "type": "server", "initial_state": {"busy": False}},
                {"id": "queue", "type": "queue", "initial_state": {"length": 0}}
            ],
            constraints=[]
        )

        # Agent interaction scenario
        agent_scenario = SimulationScenario(
            scenario_id="agent_interaction",
            name="Multi-Agent Interaction",
            description="Agent-based simulation of interacting agents",
            parameters={},
            duration=200,
            time_step=1.0,
            entities=[
                {"id": "agent_1", "type": "cooperative", "goals": ["maximize_utility"]},
                {"id": "agent_2", "type": "competitive", "goals": ["maximize_own_utility"]},
                {"id": "resource_pool", "type": "resource", "initial_state": {"available": 100}}
            ],
            constraints=[]
        )

        self.scenarios.update({
            "risk_analysis": risk_scenario,
            "queue_system": queue_scenario,
            "agent_interaction": agent_scenario
        })

    async def _warm_up_engine(self):
        """Warm up the simulation engine"""
        # Run a quick test simulation
        try:
            quick_scenario = SimulationScenario(
                scenario_id="warmup",
                name="Warmup",
                description="Engine warmup",
                parameters={"test": {"default": 1}},
                duration=10,
                time_step=1.0,
                entities=[],
                constraints=[]
            )

            await self._execute_monte_carlo_run(quick_scenario, 0)
            logger.info("Simulation engine warmed up")

        except Exception as e:
            logger.exception(f"Engine warmup failed: {e}")

    def _summarize_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize simulation performance"""
        if not results:
            return {}

        methods = {}
        for result in results:
            method = result.get("method", "unknown")
            if method not in methods:
                methods[method] = []
            methods[method].append(result.get("real_time", 0))

        summary = {}
        for method, times in methods.items():
            summary[method] = {
                "runs": len(times),
                "avg_time": np.mean(times),
                "total_time": sum(times)
            }

        return summary

    def _update_simulation_stats(self, runs_completed: int, computation_time: float):
        """Update simulation statistics"""
        self.simulation_stats["total_runs"] += runs_completed
        self.simulation_stats["successful_runs"] += runs_completed
        self.simulation_stats["total_computation_time"] += computation_time

        if self.simulation_stats["total_runs"] > 0:
            self.simulation_stats["average_duration"] = \
                self.simulation_stats["total_computation_time"] / self.simulation_stats["total_runs"]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "active_simulations": len(self.active_simulations),
            "total_simulations_run": self.simulation_stats["total_runs"],
            "simulation_stats": dict(self.simulation_stats),
            "aggregated_metrics": self.metrics_collector.get_aggregated_stats(),
            "system_health": self._get_system_health_status()
        }
    
    def _get_system_health_status(self) -> Dict[str, Any]:
        """Get current system health status"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "cpu_usage_percent": cpu_percent,
                "memory_usage_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_usage_percent": disk.percent,
                "disk_available_gb": disk.free / (1024 * 1024 * 1024),
                "timestamp": time.time()
            }
        except Exception as e:
            logger.warning(f"Failed to get system health: {e}")
            return {"error": str(e)}

    def _generate_monte_carlo_code(self, scenario: SimulationScenario, num_runs: int, 
                                 confidence_level: float) -> str:
        """Generate sandboxed Monte Carlo simulation code"""
        code_template = '''
import json
import numpy as np
import time
import random
from scipy import stats

def run_monte_carlo_simulation():
    """Sandboxed Monte Carlo simulation"""
    # Load input data
    try:
        with open("input.json", "r") as f:
            input_data = json.load(f)
    except:
        input_data = {}
    
    # Simulation parameters
    num_runs = {num_runs}
    confidence_level = {confidence_level}
    
    # Model function (would be provided by scenario)
    def model_function(**params):
        # Placeholder model - replace with actual scenario model
        return sum(params.values()) + np.random.normal(0, 1)
    
    results = []
    start_time = time.time()
    
    for i in range(num_runs):
        # Generate random inputs based on scenario
        inputs = {{}}
        # Add input generation logic here based on scenario.input_distributions
        
        # Run model
        try:
            output = model_function(**inputs)
            results.append(output)
        except Exception as e:
            print(f"Run {{i}} failed: {{e}}")
            continue
    
    # Calculate statistics
    if results:
        mean_result = np.mean(results)
        std_result = np.std(results)
        
        # Confidence interval
        if len(results) > 1:
            se = std_result / np.sqrt(len(results))
            t_value = 1.96  # Approximate for large n
            ci_lower = mean_result - t_value * se
            ci_upper = mean_result + t_value * se
            confidence_interval = [ci_lower, ci_upper]
        else:
            confidence_interval = [mean_result, mean_result]
        
        analysis = {{
            "mean": mean_result,
            "std": std_result,
            "confidence_interval": confidence_interval,
            "num_samples": len(results)
        }}
    else:
        analysis = {{"error": "No valid results"}}
    
    # Save results
    output = {{
        "results": results,
        "analysis": analysis,
        "execution_time": time.time() - start_time,
        "num_runs_completed": len(results)
    }}
    
    with open("output.json", "w") as f:
        json.dump(output, f)
    
    print(f"Simulation completed: {{len(results)}} runs in {{time.time() - start_time:.2f}}s")

if __name__ == "__main__":
    run_monte_carlo_simulation()
'''
        
        return code_template.format(num_runs=num_runs, confidence_level=confidence_level)

    async def _execute_monte_carlo_direct(self, scenario: SimulationScenario, num_runs: int,
                                        confidence_level: float, simulation_id: str) -> Tuple[List[float], Dict[str, Any]]:
        """Execute Monte Carlo simulation directly (fallback when sandbox disabled)"""
        results = []
        
        # Get metrics for this simulation
        metrics = self.metrics_collector.get_simulation_metrics(simulation_id)
        
        for i in range(num_runs):
            # Generate random inputs
            inputs = {}  # Would be based on scenario.input_distributions
            
            # Run model (placeholder)
            try:
                output = sum(inputs.values()) + np.random.normal(0, 1) if inputs else np.random.normal(0, 1)
                results.append(output)
                
                # Update metrics
                if metrics and i % 10 == 0:  # Record every 10 runs
                    metrics.entities_processed += 10
                    metrics.record_system_metrics()
                    
            except Exception as e:
                if metrics:
                    metrics.errors_count += 1
                logger.warning(f"Monte Carlo iteration {i} failed: {e}")
                continue
        
        # Calculate statistics
        if results:
            mean_result = np.mean(results)
            std_result = np.std(results)
            confidence_interval = self._calculate_confidence_interval(results, confidence_level)
            
            if metrics:
                metrics.mean_outcome = mean_result
                metrics.std_outcome = std_result
                metrics.confidence_interval = confidence_interval
                metrics.steps_completed = len(results)
        
        analysis = {
            "mean": mean_result if results else None,
            "std": std_result if results else None,
            "confidence_interval": confidence_interval if results else None,
            "num_samples": len(results)
        }
        
        return results, analysis

    async def run_parallel_simulations(self, scenario: "SimulationScenario", 
                                      parameters: Dict[str, Any], 
                                      num_runs: int = 10,
                                      max_parallel: int = 5,
                                      backend_name: Optional[str] = None) -> Dict[str, Any]:
        """Run multiple simulations in parallel using AsyncTaskManager"""
        try:
            # Validate inputs
            validation_result = self.input_validator.validate_simulation_request(
                scenario, parameters, "parallel"
            )
            if not validation_result["valid"]:
                return {
                    "status": "error",
                    "error": f"Input validation failed: {validation_result['errors']}",
                    "method": "parallel_simulations"
                }
            
            # Check safety
            safety_result = await self.safety_engine.assess_scenario_safety(scenario)
            if not safety_result.approval_granted:
                return {
                    "status": "error", 
                    "error": f"Safety check failed: {safety_result.violations}",
                    "method": "parallel_simulations"
                }
            
            # Determine backend
            if backend_name is None:
                backend_name = self.default_backend
            
            if not self.backend_manager.is_backend_available(backend_name):
                return {
                    "status": "error",
                    "error": f"Backend '{backend_name}' is not available",
                    "available_backends": self.backend_manager.get_available_backends(),
                    "method": "parallel_simulations"
                }
            
            # Emit start event
            await self.emit_event("simulation.parallel_start", {
                "backend": backend_name,
                "num_runs": num_runs,
                "max_parallel": max_parallel,
                "scenario": scenario.scenario_id
            })
            
            # Create checkpoint
            checkpoint_id = await self.rollback_manager.create_checkpoint(
                f"parallel_simulation_{backend_name}_{num_runs}runs"
            )
            
            # Execute parallel simulations using AsyncTaskManager
            result = await self._execute_parallel_with_task_manager(
                scenario, parameters, backend_name, num_runs, max_parallel
            )
            
            # Emit completion event
            await self.emit_event("simulation.parallel_complete", {
                "backend": backend_name,
                "num_runs": num_runs,
                "successful_runs": result.get("successful_runs", 0),
                "failed_runs": result.get("failed_runs", 0),
                "execution_time": result.get("execution_time", 0)
            })
            
            # Collect metrics
            await self.metrics_collector.collect_metric(
                "parallel_simulation_runs", num_runs, {"backend": backend_name}
            )
            await self.metrics_collector.collect_metric(
                "parallel_simulation_success_rate", 
                result.get("successful_runs", 0) / num_runs, 
                {"backend": backend_name}
            )
            
            result["status"] = "success"
            result["checkpoint_id"] = checkpoint_id
            return result
            
        except Exception as e:
            error_msg = f"Parallel simulation execution failed: {str(e)}"
            await self.emit_event("simulation.parallel_error", {
                "error": error_msg,
                "backend": backend_name or "unknown",
                "num_runs": num_runs
            })
            return {
                "status": "error",
                "error": error_msg,
                "method": "parallel_simulations"
            }

    async def _execute_parallel_with_task_manager(self, scenario: "SimulationScenario", 
                                                 parameters: Dict[str, Any], backend_name: str,
                                                 num_runs: int, max_parallel: int) -> Dict[str, Any]:
        """Execute multiple simulations in parallel using AsyncTaskManager"""
        start_time = time.time()
        results = []
        errors = []
        
        # Create tasks for each simulation run
        async def run_single_simulation(run_id: int):
            try:
                # Create parameters for this specific run
                run_params = parameters.copy()
                run_params["run_id"] = run_id
                
                # Execute using backend
                result = await self.backend_manager.execute_with_backend(
                    backend_name, scenario, run_params
                )
                
                if result["status"] == "success":
                    return result
                else:
                    error_msg = f"Run {run_id} failed: {result.get('error', 'Unknown error')}"
                    errors.append(error_msg)
                    return {"run_id": run_id, "error": error_msg}
                    
            except Exception as e:
                error_msg = f"Run {run_id} exception: {str(e)}"
                errors.append(error_msg)
                return {"run_id": run_id, "error": error_msg}
        
        # Submit all tasks to AsyncTaskManager
        task_ids = []
        for run_id in range(num_runs):
            task_id = await self.task_manager.submit_task(
                run_single_simulation(run_id),
                timeout=self.task_manager.default_timeout,
                resource_limits=self.task_manager.resource_limits
            )
            task_ids.append(task_id)
        
        # Wait for all tasks to complete with progress tracking
        completed_count = 0
        progress_interval = max(1, num_runs // 10)
        
        while completed_count < num_runs:
            await asyncio.sleep(1)  # Check every second
            
            # Count completed tasks
            current_completed = 0
            for task_id in task_ids:
                task_status = self.task_manager.get_task_status(task_id)
                if task_status.get("completed", False):
                    current_completed += 1
            
            # Emit progress event if significant progress made
            if current_completed > completed_count and current_completed % progress_interval == 0:
                progress = current_completed / num_runs
                await self.emit_event("simulation.parallel_progress", {
                    "backend": backend_name,
                    "completed": current_completed,
                    "total": num_runs,
                    "progress": progress,
                    "errors": len(errors)
                })
            
            completed_count = current_completed
        
        # Collect results from completed tasks
        successful_results = []
        for task_id in task_ids:
            task_result = self.task_manager.get_task_result(task_id)
            if task_result and "error" not in task_result:
                successful_results.append(task_result)
            elif task_result and "error" in task_result:
                errors.append(task_result["error"])
        
        # Aggregate results
        execution_time = time.time() - start_time
        
        if successful_results:
            # Extract outcome values for statistics
            outcomes = []
            for result in successful_results:
                if "results" in result and result["results"]:
                    # For Monte Carlo, results is a list of values
                    outcomes.extend(result["results"])
                elif "statistics" in result and "mean" in result["statistics"]:
                    # For other backends, use mean as representative value
                    outcomes.append(result["statistics"]["mean"])
            
            if outcomes:
                import numpy as np
                outcomes_array = np.array(outcomes)
                aggregated_stats = {
                    "count": len(outcomes),
                    "mean": float(np.mean(outcomes_array)),
                    "std": float(np.std(outcomes_array)),
                    "min": float(np.min(outcomes_array)),
                    "max": float(np.max(outcomes_array)),
                    "median": float(np.median(outcomes_array))
                }
            else:
                aggregated_stats = {}
        else:
            aggregated_stats = {}
        
        return {
            "method": "parallel_execution",
            "backend": backend_name,
            "total_runs": num_runs,
            "successful_runs": len(successful_results),
            "failed_runs": len(errors),
            "errors": errors,
            "aggregated_statistics": aggregated_stats,
            "individual_results": successful_results[:100],  # Limit to first 100 results
            "execution_time": execution_time
        }

    async def shutdown(self) -> bool:
        """Shutdown the simulation agent"""
        try:
            logger.info("SimulationAgent shutting down")

            # Clear active simulations
            self.active_simulations.clear()

            # Clear simulation history
            self.simulation_history.clear()
            self.scenarios.clear()
            
            # Clean up sandbox resources
            await self.sandbox_manager.cleanup_all()

            logger.info("SimulationAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"Error during SimulationAgent shutdown: {e}")
            return False