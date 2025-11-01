"""
Kalki v2.4 — Self-Model Manager
Manages Kalki's internal model of itself, capabilities, and limitations.

This addresses the critical limitation: "Still lacks true self-modeling (meta_core, self_model_manager not yet active)"
"""

import asyncio
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from pathlib import Path
import statistics

from modules.config import CONFIG, register_module_version, DIRS
from modules.logger import get_logger
from modules.meta_core import TaskResult, CapabilityAssessment, ReasoningDepth

__version__ = "Kalki v2.4 — self_model_manager.py v1.0"
register_module_version("self_model_manager.py", __version__)

logger = get_logger("self_model_manager")

@dataclass
class CapabilityProfile:
    """Profile of a specific capability."""
    name: str
    description: str
    confidence_score: float
    experience_count: int
    last_used: datetime
    performance_history: List[float] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

@dataclass
class LimitationRecord:
    """Record of a known limitation."""
    description: str
    impact_level: str  # 'low', 'medium', 'high', 'critical'
    discovered_date: datetime
    affected_capabilities: List[str]
    mitigation_strategies: List[str] = field(default_factory=list)
    status: str = "active"  # 'active', 'mitigated', 'resolved'

@dataclass
class SelfAssessment:
    """Complete self-assessment of Kalki's current state."""
    overall_confidence: float
    reasoning_depth: ReasoningDepth
    total_capabilities: int
    active_limitations: int
    recent_performance_trend: float
    adaptation_readiness: str
    capability_breakdown: Dict[str, float]

class CapabilitiesModel:
    """Model of Kalki's capabilities."""

    def __init__(self):
        self.capabilities: Dict[str, CapabilityProfile] = {}
        self.capability_categories = self._initialize_categories()

    def _initialize_categories(self) -> Dict[str, List[str]]:
        """Initialize capability categories."""
        return {
            'reasoning': ['logical_reasoning', 'causal_analysis', 'problem_solving'],
            'learning': ['pattern_recognition', 'experience_integration', 'adaptation'],
            'communication': ['natural_language', 'technical_explanation', 'user_interaction'],
            'execution': ['task_planning', 'resource_management', 'error_handling'],
            'memory': ['episodic_recall', 'semantic_storage', 'context_retention'],
            'perception': ['data_analysis', 'pattern_detection', 'anomaly_identification']
        }

    async def update_capability(self, task_result: TaskResult):
        """Update capability based on task result."""
        capability_name = task_result.task_type

        if capability_name not in self.capabilities:
            self.capabilities[capability_name] = CapabilityProfile(
                name=capability_name,
                description=f"Capability for {capability_name.replace('_', ' ')} tasks",
                confidence_score=0.5,
                experience_count=0,
                last_used=task_result.timestamp
            )

        capability = self.capabilities[capability_name]

        # Update experience count
        capability.experience_count += 1
        capability.last_used = task_result.timestamp

        # Update performance history
        capability.performance_history.append(task_result.output_quality)

        # Keep only recent history (last 50 experiences)
        if len(capability.performance_history) > 50:
            capability.performance_history.pop(0)

        # Update confidence score using exponential moving average
        if len(capability.performance_history) > 1:
            recent_performance = statistics.mean(capability.performance_history[-5:])  # Last 5 experiences
            learning_rate = 0.1
            capability.confidence_score = (
                capability.confidence_score * (1 - learning_rate) +
                recent_performance * learning_rate
            )

        # Identify limitations based on performance patterns
        await self._identify_limitations(capability, task_result)

        # Generate improvement suggestions
        await self._generate_improvements(capability)

        logger.debug(f"Updated capability {capability_name}: confidence={capability.confidence_score:.2f}")

    async def _identify_limitations(self, capability: CapabilityProfile, task_result: TaskResult):
        """Identify limitations based on task performance."""
        limitations = []

        # Performance-based limitations
        if capability.confidence_score < 0.6:
            limitations.append("Low confidence in task execution")

        if len(capability.performance_history) > 10:
            recent_avg = statistics.mean(capability.performance_history[-10:])
            older_avg = statistics.mean(capability.performance_history[:-10]) if len(capability.performance_history) > 20 else recent_avg

            if recent_avg < older_avg * 0.9:  # 10% decline
                limitations.append("Performance degradation over time")

        # Time-based limitations
        if task_result.execution_time > 30:  # seconds
            limitations.append("Slow execution times")

        # Error-based limitations
        if task_result.error_message:
            if "timeout" in task_result.error_message.lower():
                limitations.append("Timeout issues under load")
            elif "memory" in task_result.error_message.lower():
                limitations.append("Memory constraints")
            elif "connection" in task_result.error_message.lower():
                limitations.append("Connectivity issues")

        # Update capability limitations
        for limitation in limitations:
            if limitation not in capability.known_limitations:
                capability.known_limitations.append(limitation)

    async def _generate_improvements(self, capability: CapabilityProfile):
        """Generate improvement suggestions for a capability."""
        suggestions = []

        # Confidence-based suggestions
        if capability.confidence_score < 0.7:
            suggestions.append("Increase training data or experience")
            suggestions.append("Refine algorithms or approaches")

        # Performance-based suggestions
        if len(capability.performance_history) > 5:
            std_dev = statistics.stdev(capability.performance_history)
            if std_dev > 0.2:  # High variance
                suggestions.append("Improve consistency and stability")

        # Experience-based suggestions
        if capability.experience_count < 10:
            suggestions.append("Gain more experience with similar tasks")

        # Time-based suggestions
        if hasattr(capability, 'avg_execution_time') and capability.avg_execution_time > 10:
            suggestions.append("Optimize for better performance")

        capability.improvement_suggestions = suggestions[:3]  # Keep top 3

    async def get_capability_assessment(self, capability_name: str) -> Optional[CapabilityAssessment]:
        """Get detailed assessment of a capability."""
        if capability_name not in self.capabilities:
            return None

        capability = self.capabilities[capability_name]

        # Determine reasoning depth based on confidence and experience
        if capability.experience_count < 5:
            depth = ReasoningDepth.REACTIVE
        elif capability.confidence_score < 0.6:
            depth = ReasoningDepth.REFLECTIVE
        elif capability.confidence_score < 0.8:
            depth = ReasoningDepth.ADAPTIVE
        else:
            depth = ReasoningDepth.AUTONOMOUS

        # Generate recommended approach
        if capability.confidence_score > 0.8:
            approach = "confident_execution"
        elif capability.experience_count > 20:
            approach = "experienced_adaptive"
        elif len(capability.known_limitations) > 0:
            approach = "cautious_with_mitigations"
        else:
            approach = "exploratory_learning"

        return CapabilityAssessment(
            confidence_score=capability.confidence_score,
            known_limitations=capability.known_limitations.copy(),
            recommended_approach=approach,
            reasoning_depth=depth
        )

    async def get_capability_summary(self) -> Dict[str, Any]:
        """Get summary of all capabilities."""
        summary = {
            'total_capabilities': len(self.capabilities),
            'capability_breakdown': {},
            'overall_confidence': 0.0,
            'most_confident': None,
            'least_confident': None
        }

        if not self.capabilities:
            return summary

        confidences = []

        for name, capability in self.capabilities.items():
            summary['capability_breakdown'][name] = {
                'confidence': capability.confidence_score,
                'experience': capability.experience_count,
                'limitations': len(capability.known_limitations)
            }
            confidences.append(capability.confidence_score)

        summary['overall_confidence'] = statistics.mean(confidences) if confidences else 0.0

        if confidences:
            max_conf = max(confidences)
            min_conf = min(confidences)

            summary['most_confident'] = next(
                (name for name, cap in self.capabilities.items() if cap.confidence_score == max_conf),
                None
            )
            summary['least_confident'] = next(
                (name for name, cap in self.capabilities.items() if cap.confidence_score == min_conf),
                None
            )

        return summary

class LimitationsModel:
    """Model of Kalki's limitations and constraints."""

    def __init__(self):
        self.limitations: Dict[str, LimitationRecord] = {}
        self.mitigation_strategies: Dict[str, List[str]] = self._initialize_mitigations()

    def _initialize_mitigations(self) -> Dict[str, List[str]]:
        """Initialize known mitigation strategies."""
        return {
            'memory_constraints': [
                'Implement memory-efficient algorithms',
                'Add memory monitoring and cleanup',
                'Use streaming processing for large datasets'
            ],
            'slow_execution': [
                'Optimize algorithms and data structures',
                'Implement caching mechanisms',
                'Use parallel processing where applicable'
            ],
            'low_accuracy': [
                'Increase training data quality and quantity',
                'Fine-tune model parameters',
                'Implement ensemble methods'
            ],
            'connectivity_issues': [
                'Add retry mechanisms with exponential backoff',
                'Implement offline capabilities',
                'Use multiple connection strategies'
            ]
        }

    async def record_limitation(self, description: str, impact_level: str, affected_capabilities: List[str]):
        """Record a new limitation."""
        limitation_id = f"lim_{len(self.limitations)}"

        record = LimitationRecord(
            description=description,
            impact_level=impact_level,
            discovered_date=datetime.utcnow(),
            affected_capabilities=affected_capabilities,
            mitigation_strategies=self.mitigation_strategies.get(description, [])
        )

        self.limitations[limitation_id] = record
        logger.info(f"Recorded new limitation: {description} (impact: {impact_level})")

    async def get_active_limitations(self) -> List[str]:
        """Get list of active limitations."""
        return [
            record.description
            for record in self.limitations.values()
            if record.status == "active"
        ]

    async def get_limitations_by_impact(self, impact_level: str) -> List[LimitationRecord]:
        """Get limitations filtered by impact level."""
        return [
            record for record in self.limitations.values()
            if record.impact_level == impact_level and record.status == "active"
        ]

    async def mitigate_limitation(self, limitation_description: str, strategy_used: str):
        """Record successful mitigation of a limitation."""
        for record in self.limitations.values():
            if record.description == limitation_description:
                if strategy_used not in record.mitigation_strategies:
                    record.mitigation_strategies.append(strategy_used)

                # If this is a significant mitigation, consider marking as mitigated
                if len(record.mitigation_strategies) >= 2:
                    record.status = "mitigated"

                logger.info(f"Mitigated limitation: {limitation_description} using {strategy_used}")
                break

class PerformanceHistory:
    """Tracks performance history across capabilities."""

    def __init__(self):
        self.performance_data: Dict[str, List[Dict[str, Any]]] = {}
        self.retention_days = 30

    async def record_performance(self, task_result: TaskResult):
        """Record a performance data point."""
        if task_result.task_type not in self.performance_data:
            self.performance_data[task_result.task_type] = []

        data_point = {
            'timestamp': task_result.timestamp,
            'quality': task_result.output_quality,
            'execution_time': task_result.execution_time,
            'success': task_result.success,
            'context': task_result.context
        }

        self.performance_data[task_result.task_type].append(data_point)

        # Clean old data
        await self._cleanup_old_data(task_result.task_type)

    async def get_performance_trend(self, task_type: str, days: int = 7) -> Optional[float]:
        """Get performance trend over recent days."""
        if task_type not in self.performance_data:
            return None

        # Get data from last N days
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        recent_data = [
            point for point in self.performance_data[task_type]
            if point['timestamp'] > cutoff_date
        ]

        if len(recent_data) < 2:
            return None

        # Calculate trend using linear regression on quality scores
        qualities = [point['quality'] for point in recent_data]
        x_values = list(range(len(qualities)))

        if len(set(qualities)) == 1:  # No variation
            return 0.0

        slope = statistics.linear_regression(x_values, qualities).slope
        return slope

    async def get_performance_stats(self, task_type: str) -> Optional[Dict[str, Any]]:
        """Get performance statistics for a task type."""
        if task_type not in self.performance_data:
            return None

        data_points = self.performance_data[task_type]
        if not data_points:
            return None

        qualities = [point['quality'] for point in data_points]
        execution_times = [point['execution_time'] for point in data_points]

        return {
            'sample_count': len(data_points),
            'avg_quality': statistics.mean(qualities),
            'quality_stddev': statistics.stdev(qualities) if len(qualities) > 1 else 0,
            'avg_execution_time': statistics.mean(execution_times),
            'execution_time_stddev': statistics.stdev(execution_times) if len(execution_times) > 1 else 0,
            'success_rate': sum(1 for point in data_points if point['success']) / len(data_points)
        }

    async def _cleanup_old_data(self, task_type: str):
        """Clean up old performance data."""
        if task_type not in self.performance_data:
            return

        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
        self.performance_data[task_type] = [
            point for point in self.performance_data[task_type]
            if point['timestamp'] > cutoff_date
        ]

class SelfModelManager:
    """Main manager for Kalki's self-model."""

    def __init__(self):
        self.capabilities_model = CapabilitiesModel()
        self.limitations_model = LimitationsModel()
        self.performance_history = PerformanceHistory()
        self.last_self_assessment = None
        self.assessment_interval = timedelta(hours=1)  # Reassess every hour

    async def initialize(self):
        """Initialize the self-model manager."""
        logger.info("Initializing Self-Model Manager v1.0")

        # Load existing model data if available
        await self._load_persistent_data()

        # Perform initial self-assessment
        await self._perform_self_assessment()

        logger.info("Self-Model Manager initialized successfully")

    async def process_task_result(self, task_result: TaskResult):
        """Process a task result and update the self-model."""
        # Update capabilities model
        await self.capabilities_model.update_capability(task_result)

        # Record in performance history
        await self.performance_history.record_performance(task_result)

        # Check for new limitations
        await self._identify_new_limitations(task_result)

        # Update self-assessment if needed
        if (self.last_self_assessment is None or
            datetime.utcnow() - self.last_self_assessment > self.assessment_interval):
            await self._perform_self_assessment()

    async def assess_capability(self, task_type: str) -> Optional[CapabilityAssessment]:
        """Assess capability for a given task type."""
        return await self.capabilities_model.get_capability_assessment(task_type)

    async def get_self_assessment(self) -> SelfAssessment:
        """Get current self-assessment."""
        if (self.last_self_assessment is None or
            datetime.utcnow() - self.last_self_assessment > self.assessment_interval):
            await self._perform_self_assessment()

        # Return cached assessment or perform new one
        return await self._perform_self_assessment()

    async def get_capability_insights(self, task_type: str) -> Dict[str, Any]:
        """Get detailed insights about a capability."""
        assessment = await self.assess_capability(task_type)
        performance_stats = await self.performance_history.get_performance_stats(task_type)

        if not assessment:
            return {'capability': task_type, 'status': 'unknown'}

        return {
            'capability': task_type,
            'confidence': assessment.confidence_score,
            'reasoning_depth': assessment.reasoning_depth.value,
            'limitations': assessment.known_limitations,
            'recommended_approach': assessment.recommended_approach,
            'performance_stats': performance_stats,
            'improvement_suggestions': []  # Would be populated from capabilities model
        }

    async def _identify_new_limitations(self, task_result: TaskResult):
        """Identify new limitations from task results."""
        # Check for error-based limitations
        if task_result.error_message:
            error_lower = task_result.error_message.lower()

            if "memory" in error_lower and "out of memory" in error_lower:
                await self.limitations_model.record_limitation(
                    "memory_constraints",
                    "high",
                    [task_result.task_type]
                )
            elif "timeout" in error_lower:
                await self.limitations_model.record_limitation(
                    "slow_execution",
                    "medium",
                    [task_result.task_type]
                )

        # Check for performance-based limitations
        if task_result.output_quality < 0.5:
            await self.limitations_model.record_limitation(
                "low_accuracy",
                "medium",
                [task_result.task_type]
            )

    async def _perform_self_assessment(self) -> SelfAssessment:
        """Perform a comprehensive self-assessment."""
        capabilities_summary = await self.capabilities_model.get_capability_summary()
        active_limitations = await self.limitations_model.get_active_limitations()

        # Calculate overall reasoning depth
        overall_confidence = capabilities_summary['overall_confidence']
        total_capabilities = capabilities_summary['total_capabilities']

        if overall_confidence > 0.85 and total_capabilities > 10:
            reasoning_depth = ReasoningDepth.AUTONOMOUS
        elif overall_confidence > 0.7 and total_capabilities > 5:
            reasoning_depth = ReasoningDepth.ADAPTIVE
        elif overall_confidence > 0.5:
            reasoning_depth = ReasoningDepth.REFLECTIVE
        else:
            reasoning_depth = ReasoningDepth.REACTIVE

        # Calculate recent performance trend (average across all capabilities)
        recent_trend = 0.0
        trend_count = 0

        for capability_name in capabilities_summary['capability_breakdown'].keys():
            trend = await self.performance_history.get_performance_trend(capability_name, 7)
            if trend is not None:
                recent_trend += trend
                trend_count += 1

        avg_trend = recent_trend / trend_count if trend_count > 0 else 0.0

        # Determine adaptation readiness
        if reasoning_depth == ReasoningDepth.AUTONOMOUS:
            adaptation_readiness = "full_autonomy"
        elif reasoning_depth == ReasoningDepth.ADAPTIVE:
            adaptation_readiness = "supervised_adaptation"
        else:
            adaptation_readiness = "guided_operation"

        assessment = SelfAssessment(
            overall_confidence=overall_confidence,
            reasoning_depth=reasoning_depth,
            total_capabilities=total_capabilities,
            active_limitations=len(active_limitations),
            recent_performance_trend=avg_trend,
            adaptation_readiness=adaptation_readiness,
            capability_breakdown={
                name: data['confidence']
                for name, data in capabilities_summary['capability_breakdown'].items()
            }
        )

        self.last_self_assessment = datetime.utcnow()
        return assessment

    async def _load_persistent_data(self):
        """Load persistent self-model data."""
        # Implementation would load from disk/database
        # For now, start with fresh model
        pass

    async def save_model_state(self):
        """Save current model state to persistent storage."""
        # Implementation would save to disk/database
        pass

# Global self-model manager instance
_self_model_manager_instance: Optional[SelfModelManager] = None

async def get_self_model_manager() -> SelfModelManager:
    """Get or create the global self-model manager instance."""
    global _self_model_manager_instance
    if _self_model_manager_instance is None:
        _self_model_manager_instance = SelfModelManager()
        await _self_model_manager_instance.initialize()
    return _self_model_manager_instance

# Synchronous wrapper for backward compatibility
def get_self_model_manager_sync() -> SelfModelManager:
    """Synchronous wrapper for self-model manager access."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            return SelfModelManager()  # Return basic instance if loop is running
        else:
            return loop.run_until_complete(get_self_model_manager())
    except RuntimeError:
        return asyncio.run(get_self_model_manager())</content>
<parameter name="filePath">/Users/kashish/Desktop/Kalki/modules/self_model_manager.py