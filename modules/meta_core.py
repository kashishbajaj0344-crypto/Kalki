"""
Kalki v2.4 — Meta-Core Foundation
Implements self-modeling and meta-cognition capabilities for autonomous reasoning.

This addresses the critical limitation: "Still lacks true self-modeling (meta_core, self_model_manager not yet active)"
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import statistics

from modules.config import CONFIG, register_module_version
from modules.logger import get_logger

__version__ = "Kalki v2.4 — meta_core.py v1.0"
register_module_version("meta_core.py", __version__)

logger = get_logger("meta_core")

class ReasoningDepth(Enum):
    """Levels of autonomous reasoning depth."""
    REACTIVE = "reactive"           # Simple stimulus-response
    REFLECTIVE = "reflective"       # Basic self-analysis
    ADAPTIVE = "adaptive"           # Learning from experience
    AUTONOMOUS = "autonomous"       # Full self-modeling and adaptation

@dataclass
class TaskResult:
    """Result of a task execution with metadata."""
    task_id: str
    task_type: str
    success: bool
    execution_time: float
    output_quality: float  # 0.0 to 1.0
    error_message: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

@dataclass
class CapabilityAssessment:
    """Assessment of current capability for a task type."""
    confidence_score: float  # 0.0 to 1.0
    known_limitations: List[str]
    recommended_approach: str
    reasoning_depth: ReasoningDepth

@dataclass
class AdaptationPlan:
    """Plan for adapting behavior based on performance analysis."""
    task_type: str
    current_performance: float
    target_performance: float
    adaptation_actions: List[str]
    expected_improvement: float
    implementation_priority: str

class SelfModel:
    """Internal model of Kalki's own capabilities and limitations."""

    def __init__(self):
        self.capabilities: Dict[str, Dict[str, Any]] = {}
        self.limitations: List[str] = []
        self.performance_history: Dict[str, List[float]] = {}
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7

    async def integrate_insights(self, insights: Dict[str, Any]):
        """Integrate new insights into the self-model."""
        for capability, data in insights.get('capabilities', {}).items():
            if capability not in self.capabilities:
                self.capabilities[capability] = {
                    'confidence': 0.5,
                    'last_updated': datetime.utcnow(),
                    'performance_samples': []
                }

            # Update capability confidence based on new data
            current_confidence = self.capabilities[capability]['confidence']
            new_performance = data.get('performance', 0.5)

            # Exponential moving average update
            updated_confidence = current_confidence + self.learning_rate * (new_performance - current_confidence)
            self.capabilities[capability]['confidence'] = updated_confidence
            self.capabilities[capability]['last_updated'] = datetime.utcnow()
            self.capabilities[capability]['performance_samples'].append(new_performance)

            # Keep only recent samples
            samples = self.capabilities[capability]['performance_samples']
            if len(samples) > 100:
                samples.pop(0)

        # Update limitations
        new_limitations = insights.get('limitations', [])
        for limitation in new_limitations:
            if limitation not in self.limitations:
                self.limitations.append(limitation)

        logger.info(f"Integrated insights: {len(insights.get('capabilities', {}))} capabilities, {len(new_limitations)} new limitations")

    async def assess_capability(self, task_type: str) -> CapabilityAssessment:
        """Assess current capability for a task type."""
        if task_type not in self.capabilities:
            return CapabilityAssessment(
                confidence_score=0.5,
                known_limitations=self.limitations.copy(),
                recommended_approach="exploratory",
                reasoning_depth=ReasoningDepth.REACTIVE
            )

        capability_data = self.capabilities[task_type]
        confidence = capability_data['confidence']
        samples = capability_data['performance_samples']

        # Determine reasoning depth based on confidence and experience
        if len(samples) < 5:
            depth = ReasoningDepth.REACTIVE
        elif confidence < 0.6:
            depth = ReasoningDepth.REFLECTIVE
        elif confidence < 0.8:
            depth = ReasoningDepth.ADAPTIVE
        else:
            depth = ReasoningDepth.AUTONOMOUS

        # Generate recommended approach
        if confidence > self.confidence_threshold:
            approach = "confident_execution"
        elif len(samples) > 10:
            approach = "adaptive_with_fallback"
        else:
            approach = "cautious_exploration"

        return CapabilityAssessment(
            confidence_score=confidence,
            known_limitations=self.limitations.copy(),
            recommended_approach=approach,
            reasoning_depth=depth
        )

    async def get_performance_trend(self, task_type: str, window_days: int = 7) -> Optional[float]:
        """Get performance trend for a task type over recent window."""
        if task_type not in self.capabilities:
            return None

        samples = self.capabilities[task_type]['performance_samples']
        if len(samples) < 2:
            return None

        # Simple linear trend calculation
        recent_samples = samples[-min(len(samples), window_days):]
        if len(recent_samples) < 2:
            return None

        # Calculate slope of recent performance
        x = list(range(len(recent_samples)))
        slope = statistics.linear_regression(x, recent_samples).slope

        return slope

class MetaReasoningEngine:
    """Engine for meta-level reasoning about task performance and adaptation."""

    def __init__(self):
        self.analysis_patterns = self._load_analysis_patterns()
        self.performance_baselines: Dict[str, float] = {}

    def _load_analysis_patterns(self) -> Dict[str, Callable]:
        """Load patterns for analyzing different types of tasks."""
        return {
            'classification': self._analyze_classification_task,
            'generation': self._analyze_generation_task,
            'reasoning': self._analyze_reasoning_task,
            'search': self._analyze_search_task,
            'default': self._analyze_generic_task
        }

    async def analyze_task(self, task_result: TaskResult) -> Dict[str, Any]:
        """Analyze a task result and extract insights."""
        # Select appropriate analysis pattern
        analysis_func = self.analysis_patterns.get(
            task_result.task_type,
            self.analysis_patterns['default']
        )

        # Perform analysis
        analysis = await analysis_func(task_result)

        # Update performance baseline
        if task_result.task_type not in self.performance_baselines:
            self.performance_baselines[task_result.task_type] = task_result.output_quality
        else:
            # Exponential moving average
            current_baseline = self.performance_baselines[task_result.task_type]
            self.performance_baselines[task_result.task_type] = (
                0.9 * current_baseline + 0.1 * task_result.output_quality
            )

        # Generate insights
        insights = {
            'task_type': task_result.task_type,
            'performance_delta': task_result.output_quality - self.performance_baselines[task_result.task_type],
            'execution_efficiency': self._calculate_efficiency(task_result),
            'error_patterns': self._identify_error_patterns(task_result),
            'improvement_opportunities': self._identify_improvements(task_result),
            'capability_assessment': analysis.get('capability_level', 'unknown')
        }

        return insights

    async def _analyze_classification_task(self, task_result: TaskResult) -> Dict[str, Any]:
        """Analyze classification task performance."""
        analysis = {
            'capability_level': 'high' if task_result.output_quality > 0.8 else 'medium',
            'accuracy_focus': True,
            'precision_recall_balance': task_result.context.get('precision', 0.5) > 0.7
        }

        if task_result.output_quality < 0.7:
            analysis['issues'] = ['low_accuracy', 'potential_overfitting']

        return analysis

    async def _analyze_generation_task(self, task_result: TaskResult) -> Dict[str, Any]:
        """Analyze generation task performance."""
        analysis = {
            'capability_level': 'high' if task_result.output_quality > 0.75 else 'medium',
            'creativity_coherence_balance': True
        }

        if task_result.execution_time > 30:  # seconds
            analysis['issues'] = ['slow_generation', 'optimization_needed']

        return analysis

    async def _analyze_reasoning_task(self, task_result: TaskResult) -> Dict[str, Any]:
        """Analyze reasoning task performance."""
        analysis = {
            'capability_level': 'medium',  # Reasoning is complex
            'logical_consistency': task_result.context.get('logical_errors', 0) == 0
        }

        if task_result.output_quality < 0.6:
            analysis['issues'] = ['weak_reasoning', 'logic_gaps']

        return analysis

    async def _analyze_search_task(self, task_result: TaskResult) -> Dict[str, Any]:
        """Analyze search task performance."""
        analysis = {
            'capability_level': 'high' if task_result.output_quality > 0.8 else 'medium',
            'retrieval_effectiveness': True
        }

        if task_result.execution_time > 5:  # seconds
            analysis['issues'] = ['slow_search', 'index_optimization_needed']

        return analysis

    async def _analyze_generic_task(self, task_result: TaskResult) -> Dict[str, Any]:
        """Generic task analysis fallback."""
        return {
            'capability_level': 'medium' if task_result.output_quality > 0.5 else 'low',
            'generic_analysis': True
        }

    def _calculate_efficiency(self, task_result: TaskResult) -> float:
        """Calculate execution efficiency score."""
        # Efficiency = quality / (time + 1) - normalize to 0-1 range
        raw_efficiency = task_result.output_quality / (task_result.execution_time + 1)
        return min(raw_efficiency * 10, 1.0)  # Scale and cap

    def _identify_error_patterns(self, task_result: TaskResult) -> List[str]:
        """Identify patterns in errors."""
        patterns = []

        if task_result.error_message:
            error_lower = task_result.error_message.lower()

            if 'timeout' in error_lower:
                patterns.append('timeout_errors')
            if 'memory' in error_lower:
                patterns.append('memory_issues')
            if 'connection' in error_lower:
                patterns.append('connectivity_problems')
            if 'permission' in error_lower:
                patterns.append('access_denied')

        return patterns

    def _identify_improvements(self, task_result: TaskResult) -> List[str]:
        """Identify potential improvements."""
        improvements = []

        if task_result.execution_time > 10:
            improvements.append('optimize_execution_time')

        if task_result.output_quality < 0.7:
            improvements.append('improve_output_quality')

        if task_result.context.get('retries', 0) > 2:
            improvements.append('reduce_error_rate')

        return improvements

class AdaptationManager:
    """Manages adaptation strategies based on performance analysis."""

    def __init__(self):
        self.adaptation_strategies = self._load_strategies()
        self.active_adaptations: Dict[str, AdaptationPlan] = {}

    def _load_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Load adaptation strategies for different scenarios."""
        return {
            'performance_decline': {
                'actions': ['increase_learning_rate', 'adjust_parameters', 'add_fallbacks'],
                'expected_improvement': 0.15
            },
            'slow_execution': {
                'actions': ['optimize_algorithms', 'cache_results', 'parallelize_tasks'],
                'expected_improvement': 0.25
            },
            'high_error_rate': {
                'actions': ['add_validation', 'implement_retries', 'error_recovery'],
                'expected_improvement': 0.20
            },
            'resource_constraints': {
                'actions': ['reduce_batch_size', 'optimize_memory', 'prioritize_tasks'],
                'expected_improvement': 0.10
            }
        }

    async def create_adaptation_plan(self, analysis: Dict[str, Any]) -> Optional[AdaptationPlan]:
        """Create an adaptation plan based on performance analysis."""
        task_type = analysis['task_type']
        performance_delta = analysis['performance_delta']
        improvements_needed = analysis['improvement_opportunities']

        # Determine if adaptation is needed
        if performance_delta > -0.1 and not improvements_needed:  # Performance is stable/good
            return None

        # Select adaptation strategy
        strategy_key = self._select_strategy(analysis)
        strategy = self.adaptation_strategies[strategy_key]

        # Calculate expected improvement
        current_performance = self._estimate_current_performance(task_type)
        expected_improvement = strategy['expected_improvement']

        plan = AdaptationPlan(
            task_type=task_type,
            current_performance=current_performance,
            target_performance=current_performance + expected_improvement,
            adaptation_actions=strategy['actions'],
            expected_improvement=expected_improvement,
            implementation_priority=self._calculate_priority(analysis)
        )

        # Store active adaptation
        self.active_adaptations[task_type] = plan

        return plan

    def _select_strategy(self, analysis: Dict[str, Any]) -> str:
        """Select the most appropriate adaptation strategy."""
        improvements = analysis['improvement_opportunities']
        error_patterns = analysis['error_patterns']

        if 'optimize_execution_time' in improvements:
            return 'slow_execution'
        elif 'reduce_error_rate' in improvements or error_patterns:
            return 'high_error_rate'
        elif analysis['performance_delta'] < -0.2:
            return 'performance_decline'
        else:
            return 'resource_constraints'

    def _estimate_current_performance(self, task_type: str) -> float:
        """Estimate current performance for a task type."""
        # This would integrate with the SelfModel's performance tracking
        # For now, return a reasonable default
        return 0.7

    def _calculate_priority(self, analysis: Dict[str, Any]) -> str:
        """Calculate implementation priority for the adaptation."""
        performance_delta = analysis['performance_delta']
        error_patterns = analysis['error_patterns']

        if performance_delta < -0.3 or error_patterns:
            return 'critical'
        elif performance_delta < -0.2:
            return 'high'
        else:
            return 'medium'

class MetaCore:
    """Main meta-cognition and self-modeling engine."""

    def __init__(self):
        self.self_model = SelfModel()
        self.reasoning_engine = MetaReasoningEngine()
        self.adaptation_manager = AdaptationManager()
        self.reasoning_depth = ReasoningDepth.REFLECTIVE

    async def initialize(self):
        """Initialize the meta-core system."""
        logger.info("Initializing Meta-Core v1.0 - Self-modeling engine")

        # Load existing self-model if available
        await self._load_self_model()

        # Set initial reasoning depth
        await self._assess_reasoning_capability()

        logger.info(f"Meta-Core initialized with reasoning depth: {self.reasoning_depth.value}")

    async def process_task_result(self, task_result: TaskResult) -> Optional[AdaptationPlan]:
        """Process a task result and potentially generate adaptation."""
        # Analyze the task result
        analysis = await self.reasoning_engine.analyze_task(task_result)

        # Update self-model with insights
        insights = {
            'capabilities': {
                task_result.task_type: {
                    'performance': task_result.output_quality,
                    'execution_time': task_result.execution_time,
                    'timestamp': task_result.timestamp
                }
            },
            'limitations': analysis.get('error_patterns', [])
        }

        await self.self_model.integrate_insights(insights)

        # Check if adaptation is needed
        adaptation_plan = await self.adaptation_manager.create_adaptation_plan(analysis)

        if adaptation_plan:
            logger.info(f"Generated adaptation plan for {task_result.task_type}: {adaptation_plan.adaptation_actions}")

        # Update reasoning depth based on performance
        await self._update_reasoning_depth()

        return adaptation_plan

    async def assess_task_capability(self, task_type: str) -> CapabilityAssessment:
        """Assess capability for a given task type."""
        return await self.self_model.assess_capability(task_type)

    async def get_system_insights(self) -> Dict[str, Any]:
        """Get current system insights and self-assessment."""
        capabilities_summary = {}
        for task_type in self.self_model.capabilities.keys():
            assessment = await self.assess_task_capability(task_type)
            capabilities_summary[task_type] = {
                'confidence': assessment.confidence_score,
                'reasoning_depth': assessment.reasoning_depth.value,
                'limitations': len(assessment.known_limitations)
            }

        return {
            'reasoning_depth': self.reasoning_depth.value,
            'total_capabilities': len(capabilities_summary),
            'capabilities_summary': capabilities_summary,
            'active_limitations': self.self_model.limitations,
            'active_adaptations': len(self.adaptation_manager.active_adaptations)
        }

    async def _load_self_model(self):
        """Load existing self-model from storage."""
        # Implementation would load from persistent storage
        # For now, start with empty model
        pass

    async def _assess_reasoning_capability(self):
        """Assess current reasoning capability."""
        # Start with reflective reasoning
        # This would be upgraded based on performance over time
        self.reasoning_depth = ReasoningDepth.REFLECTIVE

    async def _update_reasoning_depth(self):
        """Update reasoning depth based on recent performance."""
        # Calculate average confidence across all capabilities
        if not self.self_model.capabilities:
            return

        total_confidence = 0
        count = 0

        for capability_data in self.self_model.capabilities.values():
            total_confidence += capability_data['confidence']
            count += 1

        if count == 0:
            return

        avg_confidence = total_confidence / count

        # Update reasoning depth based on average confidence
        if avg_confidence > 0.85:
            self.reasoning_depth = ReasoningDepth.AUTONOMOUS
        elif avg_confidence > 0.7:
            self.reasoning_depth = ReasoningDepth.ADAPTIVE
        elif avg_confidence > 0.5:
            self.reasoning_depth = ReasoningDepth.REFLECTIVE
        else:
            self.reasoning_depth = ReasoningDepth.REACTIVE

# Global meta-core instance
_meta_core_instance: Optional[MetaCore] = None

async def get_meta_core() -> MetaCore:
    """Get or create the global meta-core instance."""
    global _meta_core_instance
    if _meta_core_instance is None:
        _meta_core_instance = MetaCore()
        await _meta_core_instance.initialize()
    return _meta_core_instance

# Synchronous wrapper for backward compatibility
def get_meta_core_sync() -> MetaCore:
    """Synchronous wrapper for meta-core access."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, we need to handle this differently
            # For now, return a basic instance
            return MetaCore()
        else:
            return loop.run_until_complete(get_meta_core())
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(get_meta_core())</content>
<parameter name="filePath">/Users/kashish/Desktop/Kalki/modules/meta_core.py