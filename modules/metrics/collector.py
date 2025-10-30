"""
Phase 20 - Evaluation, Metrics & CI Integration
Metrics collection and evaluation for the multi-agent system.
"""

import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


@dataclass
class TaskMetrics:
    """Metrics for a single task execution."""
    task_id: str
    agent_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    latency: float = 0.0  # seconds
    success: bool = False
    retries: int = 0
    memory_lookups: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data


@dataclass
class SystemMetrics:
    """Aggregate system metrics."""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_retries: int = 0
    total_memory_lookups: int = 0
    average_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    agent_utilization: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_tasks': self.total_tasks,
            'successful_tasks': self.successful_tasks,
            'failed_tasks': self.failed_tasks,
            'success_rate': self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0,
            'total_retries': self.total_retries,
            'total_memory_lookups': self.total_memory_lookups,
            'average_latency': self.average_latency,
            'min_latency': self.min_latency if self.min_latency != float('inf') else 0.0,
            'max_latency': self.max_latency,
            'agent_utilization': dict(self.agent_utilization)
        }


class MetricsCollector:
    """Collects and aggregates metrics for the multi-agent system."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.active_tasks: Dict[str, datetime] = {}
        self.memory_lookup_count: Dict[str, int] = defaultdict(int)
    
    def start_task(self, task_id: str, agent_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start tracking a task.
        
        Args:
            task_id: Task identifier
            agent_id: Agent executing the task
            metadata: Optional metadata
        """
        start_time = datetime.now()
        self.active_tasks[task_id] = start_time
        
        metrics = TaskMetrics(
            task_id=task_id,
            agent_id=agent_id,
            start_time=start_time,
            metadata=metadata or {}
        )
        
        self.task_metrics[task_id] = metrics
    
    def end_task(self, task_id: str, success: bool, error: Optional[str] = None) -> None:
        """
        End tracking a task.
        
        Args:
            task_id: Task identifier
            success: Whether task succeeded
            error: Optional error message
        """
        if task_id not in self.task_metrics:
            return
        
        end_time = datetime.now()
        metrics = self.task_metrics[task_id]
        metrics.end_time = end_time
        metrics.success = success
        
        if task_id in self.active_tasks:
            start_time = self.active_tasks[task_id]
            metrics.latency = (end_time - start_time).total_seconds()
            del self.active_tasks[task_id]
        
        if error:
            metrics.errors.append(error)
        
        # Add memory lookup count
        if task_id in self.memory_lookup_count:
            metrics.memory_lookups = self.memory_lookup_count[task_id]
    
    def record_retry(self, task_id: str) -> None:
        """Record a task retry."""
        if task_id in self.task_metrics:
            self.task_metrics[task_id].retries += 1
    
    def record_memory_lookup(self, task_id: str) -> None:
        """Record a memory lookup for a task."""
        self.memory_lookup_count[task_id] += 1
    
    def get_task_metrics(self, task_id: str) -> Optional[TaskMetrics]:
        """Get metrics for a specific task."""
        return self.task_metrics.get(task_id)
    
    def get_system_metrics(self) -> SystemMetrics:
        """
        Get aggregate system metrics.
        
        Returns:
            SystemMetrics with aggregated data
        """
        metrics = SystemMetrics()
        
        completed_tasks = [m for m in self.task_metrics.values() if m.end_time is not None]
        
        metrics.total_tasks = len(completed_tasks)
        metrics.successful_tasks = sum(1 for m in completed_tasks if m.success)
        metrics.failed_tasks = metrics.total_tasks - metrics.successful_tasks
        metrics.total_retries = sum(m.retries for m in completed_tasks)
        metrics.total_memory_lookups = sum(m.memory_lookups for m in completed_tasks)
        
        # Calculate latency statistics
        latencies = [m.latency for m in completed_tasks if m.latency > 0]
        
        if latencies:
            metrics.average_latency = sum(latencies) / len(latencies)
            metrics.min_latency = min(latencies)
            metrics.max_latency = max(latencies)
        
        # Calculate agent utilization
        for m in completed_tasks:
            metrics.agent_utilization[m.agent_id] += 1
        
        return metrics
    
    def get_all_task_metrics(self) -> List[TaskMetrics]:
        """Get all task metrics."""
        return list(self.task_metrics.values())
    
    def clear(self) -> None:
        """Clear all collected metrics."""
        self.task_metrics.clear()
        self.active_tasks.clear()
        self.memory_lookup_count.clear()


class PerformanceMonitor:
    """Monitor and report on system performance."""
    
    def __init__(self, collector: MetricsCollector):
        """
        Initialize performance monitor.
        
        Args:
            collector: MetricsCollector instance
        """
        self.collector = collector
    
    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a performance report.
        
        Returns:
            Dictionary with performance metrics and analysis
        """
        system_metrics = self.collector.get_system_metrics()
        
        report = {
            'summary': system_metrics.to_dict(),
            'analysis': self._analyze_performance(system_metrics),
            'recommendations': self._generate_recommendations(system_metrics)
        }
        
        return report
    
    def _analyze_performance(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Analyze performance metrics."""
        analysis = {}
        
        # Success rate analysis
        success_rate = metrics.successful_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 0
        
        if success_rate >= 0.95:
            analysis['success_rate_status'] = 'excellent'
        elif success_rate >= 0.80:
            analysis['success_rate_status'] = 'good'
        elif success_rate >= 0.60:
            analysis['success_rate_status'] = 'fair'
        else:
            analysis['success_rate_status'] = 'poor'
        
        # Latency analysis
        if metrics.average_latency < 0.1:
            analysis['latency_status'] = 'excellent'
        elif metrics.average_latency < 0.5:
            analysis['latency_status'] = 'good'
        elif metrics.average_latency < 2.0:
            analysis['latency_status'] = 'fair'
        else:
            analysis['latency_status'] = 'poor'
        
        # Retry analysis
        avg_retries = metrics.total_retries / metrics.total_tasks if metrics.total_tasks > 0 else 0
        
        if avg_retries < 0.1:
            analysis['retry_status'] = 'excellent'
        elif avg_retries < 0.5:
            analysis['retry_status'] = 'good'
        else:
            analysis['retry_status'] = 'needs_improvement'
        
        return analysis
    
    def _generate_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        success_rate = metrics.successful_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 0
        
        if success_rate < 0.80:
            recommendations.append("Success rate is below 80%. Review failed tasks and improve error handling.")
        
        if metrics.average_latency > 1.0:
            recommendations.append("Average latency is high. Consider optimizing task execution or adding more agents.")
        
        avg_retries = metrics.total_retries / metrics.total_tasks if metrics.total_tasks > 0 else 0
        
        if avg_retries > 0.5:
            recommendations.append("High retry rate detected. Investigate root causes of task failures.")
        
        # Check for agent load balancing
        if metrics.agent_utilization:
            max_util = max(metrics.agent_utilization.values())
            min_util = min(metrics.agent_utilization.values())
            
            if max_util > 3 * min_util:
                recommendations.append("Uneven agent utilization detected. Consider load balancing improvements.")
        
        if not recommendations:
            recommendations.append("System is performing well. No major issues detected.")
        
        return recommendations
