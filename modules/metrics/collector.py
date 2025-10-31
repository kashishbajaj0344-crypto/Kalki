"""
Kalki Metrics Module - Phase 20 Enhanced
Performance metrics, evaluation, and CI integration with advanced features.
"""

import math
import time
import psutil
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from modules.testing.harness import AdversarialTestHarness
from datetime import datetime, timedelta
from collections import defaultdict
from statistics import mean, stdev
from pathlib import Path

# Memory system integration
try:
    from modules.agents.memory.sqlite_store import SQLiteMemoryStore
    from modules.agents.memory.base import MemoryEntry
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    SQLiteMemoryStore = None
    MemoryEntry = None

# Testing integration
try:
    from modules.testing.harness import AdversarialTestHarness
    TESTING_AVAILABLE = True
except ImportError:
    TESTING_AVAILABLE = False
    AdversarialTestHarness = None

from typing import Dict, List, Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    if MEMORY_AVAILABLE:
        from modules.agents.memory.sqlite_store import SQLiteMemoryStore
    if TESTING_AVAILABLE:
        from modules.testing.harness import AdversarialTestHarness
    TestReport = None

logger = logging.getLogger('Kalki.Metrics')


@dataclass
class TaskMetrics:
    """Enhanced metrics for a single task execution."""
    task_id: str
    agent_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    task_type: str = "general"  # New: task categorization
    latency: float = 0.0  # seconds (using monotonic time)
    success: bool = False
    retries: int = 0
    memory_lookups: int = 0
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Enhanced cognitive metrics
    tokens_in: int = 0
    tokens_out: int = 0
    context_switches: int = 0
    attention_weight: float = 0.0

    # Real-time instrumentation
    cpu_time_start: float = 0.0
    cpu_time_end: float = 0.0
    memory_start: int = 0
    memory_end: int = 0
    memory_peak: int = 0

    def start_instrumentation(self) -> None:
        """Start real-time instrumentation."""
        self.cpu_time_start = time.perf_counter()
        process = psutil.Process()
        self.memory_start = process.memory_info().rss

    def end_instrumentation(self) -> None:
        """End real-time instrumentation."""
        self.cpu_time_end = time.perf_counter()
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_end = memory_info.rss
        self.memory_peak = max(self.memory_peak, memory_info.rss)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data['start_time'] = self.start_time.isoformat()
        data['end_time'] = self.end_time.isoformat() if self.end_time else None
        return data


@dataclass
class SystemMetrics:
    """Enhanced aggregate system metrics."""
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    total_retries: int = 0
    total_memory_lookups: int = 0
    average_latency: float = 0.0
    min_latency: float = math.inf
    max_latency: float = 0.0
    agent_utilization: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Enhanced metrics
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_context_switches: int = 0
    average_attention_weight: float = 0.0
    system_cpu_percent: float = 0.0
    system_memory_percent: float = 0.0
    disk_usage_percent: float = 0.0

    # Predictive risk score (Phase 20 integration)
    predictive_risk_score: float = 0.0

    def collect_system_metrics(self) -> None:
        """Collect current system resource metrics."""
        try:
            self.system_cpu_percent = psutil.cpu_percent(interval=0.1)
            self.system_memory_percent = psutil.virtual_memory().percent
            self.disk_usage_percent = psutil.disk_usage('/').percent
        except Exception as e:
            logger.warning(f"Failed to collect system metrics: {e}")

    def calculate_predictive_risk(self, historical_data: Optional[List['SystemMetrics']] = None) -> float:
        """Calculate predictive risk score based on current and historical data."""
        if not historical_data:
            # Base risk calculation
            success_rate = self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 1.0
            risk_factors = [
                (1.0 - success_rate) * 0.4,  # Success rate impact
                min(self.average_latency / 5.0, 1.0) * 0.3,  # Latency impact
                (self.total_retries / max(self.total_tasks, 1)) * 0.3  # Retry impact
            ]
            self.predictive_risk_score = min(1.0, sum(risk_factors))
        else:
            # Trend-based risk calculation
            recent_trends = self._analyze_trends(historical_data[-10:])  # Last 10 data points
            self.predictive_risk_score = min(1.0, recent_trends)

        return self.predictive_risk_score

    def _analyze_trends(self, historical: List['SystemMetrics']) -> float:
        """Analyze trends in historical data for risk assessment."""
        if len(historical) < 2:
            return 0.0

        # Calculate trend slopes (simplified)
        success_rates = [h.successful_tasks / h.total_tasks if h.total_tasks > 0 else 0 for h in historical]
        latencies = [h.average_latency for h in historical]

        # Simple linear trend (slope)
        def calculate_slope(values: List[float]) -> float:
            if len(values) < 2:
                return 0.0
            n = len(values)
            x = list(range(n))
            slope = sum((x[i] - mean(x)) * (values[i] - mean(values)) for i in range(n))
            slope /= sum((x[i] - mean(x)) ** 2 for i in range(n))
            return slope

        success_trend = calculate_slope(success_rates)
        latency_trend = calculate_slope(latencies)

        # Risk increases with declining success or increasing latency
        risk = 0.0
        if success_trend < -0.01:  # Success rate declining
            risk += 0.4
        if latency_trend > 0.1:  # Latency increasing
            risk += 0.4
        if self.system_cpu_percent > 80 or self.system_memory_percent > 85:
            risk += 0.2

        return risk

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
            'min_latency': self.min_latency if self.min_latency != math.inf else 0.0,
            'max_latency': self.max_latency,
            'agent_utilization': dict(self.agent_utilization),
            'total_tokens_in': self.total_tokens_in,
            'total_tokens_out': self.total_tokens_out,
            'total_context_switches': self.total_context_switches,
            'average_attention_weight': self.average_attention_weight,
            'system_cpu_percent': self.system_cpu_percent,
            'system_memory_percent': self.system_memory_percent,
            'disk_usage_percent': self.disk_usage_percent,
            'predictive_risk_score': self.predictive_risk_score
        }


class MetricsCollector:
    """Enhanced metrics collector with advanced features."""

    def __init__(self, memory_store: Optional[Any] = None):
        """Initialize metrics collector."""
        self.task_metrics: Dict[str, TaskMetrics] = {}
        self.active_tasks: Dict[str, datetime] = {}
        self.memory_lookup_count: Dict[str, int] = defaultdict(int)
        self.memory_store = memory_store
        self.historical_reports: List[SystemMetrics] = []

        # CI/Regression thresholds
        self.ci_thresholds = {
            'min_success_rate': 0.90,
            'max_average_latency': 2.0,
            'max_retry_rate': 0.5,
            'max_predictive_risk': 0.7
        }

    def start_task(self, task_id: str, agent_id: str, task_type: str = "general",
                   metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Start tracking a task with enhanced instrumentation.

        Args:
            task_id: Task identifier
            agent_id: Agent executing the task
            task_type: Task category (ingestion, planning, dialogue, etc.)
            metadata: Optional metadata
        """
        start_time = datetime.now()
        self.active_tasks[task_id] = start_time

        metrics = TaskMetrics(
            task_id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            start_time=start_time,
            metadata=metadata or {}
        )

        metrics.start_instrumentation()
        self.task_metrics[task_id] = metrics

    def end_task(self, task_id: str, success: bool, error: Optional[str] = None,
                 tokens_in: int = 0, tokens_out: int = 0, context_switches: int = 0,
                 attention_weight: float = 0.0) -> None:
        """
        End tracking a task with cognitive metrics.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            error: Optional error message
            tokens_in: Input tokens used
            tokens_out: Output tokens generated
            context_switches: Number of context switches
            attention_weight: Attention weight metric
        """
        if task_id not in self.task_metrics:
            return

        end_time = datetime.now()
        metrics = self.task_metrics[task_id]
        metrics.end_time = end_time
        metrics.success = success

        # End instrumentation
        metrics.end_instrumentation()

        # Calculate monotonic latency
        if task_id in self.active_tasks:
            start_time = self.active_tasks[task_id]
            metrics.latency = time.perf_counter() - metrics.cpu_time_start
            del self.active_tasks[task_id]

        # Add cognitive metrics
        metrics.tokens_in = tokens_in
        metrics.tokens_out = tokens_out
        metrics.context_switches = context_switches
        metrics.attention_weight = attention_weight

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
        Get aggregate system metrics with enhanced calculations.

        Returns:
            SystemMetrics with aggregated data
        """
        metrics = SystemMetrics()
        metrics.collect_system_metrics()

        completed_tasks = [m for m in self.task_metrics.values() if m.end_time is not None]

        metrics.total_tasks = len(completed_tasks)
        metrics.successful_tasks = sum(1 for m in completed_tasks if m.success)
        metrics.failed_tasks = metrics.total_tasks - metrics.successful_tasks
        metrics.total_retries = sum(m.retries for m in completed_tasks)
        metrics.total_memory_lookups = sum(m.memory_lookups for m in completed_tasks)

        # Calculate latency statistics
        latencies = [m.latency for m in completed_tasks if m.latency > 0]

        if latencies:
            metrics.average_latency = mean(latencies)
            metrics.min_latency = min(latencies)
            metrics.max_latency = max(latencies)

        # Calculate cognitive metrics
        if completed_tasks:
            metrics.total_tokens_in = sum(m.tokens_in for m in completed_tasks)
            metrics.total_tokens_out = sum(m.tokens_out for m in completed_tasks)
            metrics.total_context_switches = sum(m.context_switches for m in completed_tasks)
            attention_weights = [m.attention_weight for m in completed_tasks if m.attention_weight > 0]
            if attention_weights:
                metrics.average_attention_weight = mean(attention_weights)

        # Calculate agent utilization
        for m in completed_tasks:
            metrics.agent_utilization[m.agent_id] += 1

        # Calculate predictive risk
        metrics.calculate_predictive_risk(self.historical_reports)

        # Store for historical analysis
        self.historical_reports.append(metrics)

        return metrics

    def get_all_task_metrics(self) -> List[TaskMetrics]:
        """Get all task metrics."""
        return list(self.task_metrics.values())

    def clear(self) -> None:
        """Clear all collected metrics."""
        self.task_metrics.clear()
        self.active_tasks.clear()
        self.memory_lookup_count.clear()

    def save_to_memory(self) -> bool:
        """Save current metrics to memory store."""
        if not self.memory_store or not MEMORY_AVAILABLE:
            return False

        try:
            system_metrics = self.get_system_metrics()
            report_id = f"metrics_report_{int(time.time())}"

            self.memory_store.put(report_id, system_metrics.to_dict(), {
                "type": "metrics_report",
                "phase": 20,
                "timestamp": datetime.now().isoformat(),
                "total_tasks": system_metrics.total_tasks,
                "success_rate": system_metrics.successful_tasks / system_metrics.total_tasks if system_metrics.total_tasks > 0 else 0.0
            })

            # Save individual task metrics
            for task_metrics in self.task_metrics.values():
                if task_metrics.end_time:  # Only save completed tasks
                    task_id = f"task_metrics_{task_metrics.task_id}"
                    self.memory_store.put(task_id, task_metrics.to_dict(), {
                        "type": "task_metrics",
                        "phase": 20,
                        "agent_id": task_metrics.agent_id,
                        "task_type": task_metrics.task_type,
                        "success": task_metrics.success
                    })

            return True
        except Exception as e:
            logger.error(f"Failed to save metrics to memory: {e}")
            return False

    def run_ci_checks(self) -> Dict[str, Any]:
        """
        Run CI/regression checks against thresholds.

        Returns:
            Dictionary with check results and pass/fail status
        """
        system_metrics = self.get_system_metrics()

        checks = {
            'success_rate_check': {
                'value': system_metrics.successful_tasks / system_metrics.total_tasks if system_metrics.total_tasks > 0 else 0.0,
                'threshold': self.ci_thresholds['min_success_rate'],
                'passed': (system_metrics.successful_tasks / system_metrics.total_tasks if system_metrics.total_tasks > 0 else 0.0) >= self.ci_thresholds['min_success_rate']
            },
            'latency_check': {
                'value': system_metrics.average_latency,
                'threshold': self.ci_thresholds['max_average_latency'],
                'passed': system_metrics.average_latency <= self.ci_thresholds['max_average_latency']
            },
            'retry_rate_check': {
                'value': system_metrics.total_retries / system_metrics.total_tasks if system_metrics.total_tasks > 0 else 0.0,
                'threshold': self.ci_thresholds['max_retry_rate'],
                'passed': (system_metrics.total_retries / system_metrics.total_tasks if system_metrics.total_tasks > 0 else 0.0) <= self.ci_thresholds['max_retry_rate']
            },
            'predictive_risk_check': {
                'value': system_metrics.predictive_risk_score,
                'threshold': self.ci_thresholds['max_predictive_risk'],
                'passed': system_metrics.predictive_risk_score <= self.ci_thresholds['max_predictive_risk']
            }
        }

        overall_pass = all(check['passed'] for check in checks.values())

        return {
            'overall_pass': overall_pass,
            'checks': checks,
            'summary': {
                'total_checks': len(checks),
                'passed_checks': sum(1 for check in checks.values() if check['passed']),
                'failed_checks': sum(1 for check in checks.values() if not check['passed'])
            }
        }


class PerformanceMonitor:
    """Enhanced performance monitor with advanced analytics."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize performance monitor.

        Args:
            collector: MetricsCollector instance
        """
        self.collector = collector

    def get_performance_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive performance report.

        Returns:
            Dictionary with performance metrics and analysis
        """
        system_metrics = self.collector.get_system_metrics()

        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': system_metrics.to_dict(),
            'analysis': self._analyze_performance(system_metrics),
            'recommendations': self._generate_recommendations(system_metrics),
            'trends': self._analyze_trends(),
            'ci_status': self.collector.run_ci_checks()
        }

        return report

    def _analyze_performance(self, metrics: SystemMetrics) -> Dict[str, Any]:
        """Analyze performance metrics with enhanced criteria."""
        analysis = {}

        # Success rate analysis
        success_rate = metrics.successful_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 0

        if success_rate >= 0.95:
            analysis['success_rate_status'] = 'excellent'
        elif success_rate >= 0.85:
            analysis['success_rate_status'] = 'good'
        elif success_rate >= 0.70:
            analysis['success_rate_status'] = 'fair'
        else:
            analysis['success_rate_status'] = 'poor'

        # Latency analysis
        if metrics.average_latency < 0.1:
            analysis['latency_status'] = 'excellent'
        elif metrics.average_latency < 0.5:
            analysis['latency_status'] = 'good'
        elif metrics.average_latency < 1.0:
            analysis['latency_status'] = 'fair'
        else:
            analysis['latency_status'] = 'poor'

        # Retry analysis
        avg_retries = metrics.total_retries / metrics.total_tasks if metrics.total_tasks > 0 else 0

        if avg_retries < 0.1:
            analysis['retry_status'] = 'excellent'
        elif avg_retries < 0.3:
            analysis['retry_status'] = 'good'
        else:
            analysis['retry_status'] = 'needs_improvement'

        # Cognitive load analysis
        if metrics.average_attention_weight > 0:
            if metrics.average_attention_weight > 0.8:
                analysis['cognitive_load_status'] = 'high'
            elif metrics.average_attention_weight > 0.5:
                analysis['cognitive_load_status'] = 'moderate'
            else:
                analysis['cognitive_load_status'] = 'low'

        # System resource analysis
        if metrics.system_cpu_percent > 80:
            analysis['cpu_status'] = 'high_utilization'
        elif metrics.system_cpu_percent > 50:
            analysis['cpu_status'] = 'moderate_utilization'
        else:
            analysis['cpu_status'] = 'normal'

        if metrics.system_memory_percent > 85:
            analysis['memory_status'] = 'high_utilization'
        elif metrics.system_memory_percent > 70:
            analysis['memory_status'] = 'moderate_utilization'
        else:
            analysis['memory_status'] = 'normal'

        return analysis

    def _generate_recommendations(self, metrics: SystemMetrics) -> List[str]:
        """Generate performance recommendations with adaptive thresholds."""
        recommendations = []

        success_rate = metrics.successful_tasks / metrics.total_tasks if metrics.total_tasks > 0 else 0

        if success_rate < 0.80:
            recommendations.append("Success rate is below 80%. Review failed tasks and improve error handling.")
        elif success_rate < 0.90:
            recommendations.append("Success rate could be improved. Consider adding retry logic for transient failures.")

        if metrics.average_latency > 1.0:
            recommendations.append("Average latency is high. Consider optimizing task execution or adding more agents.")
        elif metrics.average_latency > 2.0:
            recommendations.append("CRITICAL: Latency exceeds 2 seconds. Immediate optimization required.")

        avg_retries = metrics.total_retries / metrics.total_tasks if metrics.total_tasks > 0 else 0

        if avg_retries > 0.5:
            recommendations.append("High retry rate detected. Investigate root causes of task failures.")
        elif avg_retries > 0.3:
            recommendations.append("Moderate retry rate. Consider improving task reliability.")

        # Check for agent load balancing
        if metrics.agent_utilization:
            max_util = max(metrics.agent_utilization.values())
            min_util = min(metrics.agent_utilization.values())

            if max_util > 3 * min_util and min_util > 0:
                recommendations.append("Uneven agent utilization detected. Consider load balancing improvements.")

        # System resource recommendations
        if metrics.system_cpu_percent > 80:
            recommendations.append("High CPU utilization detected. Consider scaling or optimization.")
        if metrics.system_memory_percent > 85:
            recommendations.append("High memory utilization detected. Monitor for memory leaks.")

        # Cognitive load recommendations
        if hasattr(metrics, 'average_attention_weight') and metrics.average_attention_weight > 0.8:
            recommendations.append("High cognitive load detected. Consider task complexity reduction.")

        if not recommendations:
            recommendations.append("System is performing well. No major issues detected.")

        return recommendations

    def _analyze_trends(self) -> Dict[str, Any]:
        """Analyze historical trends."""
        if len(self.collector.historical_reports) < 2:
            return {'available': False, 'message': 'Insufficient historical data'}

        recent_reports = self.collector.historical_reports[-10:]  # Last 10 reports

        trends = {
            'available': True,
            'data_points': len(recent_reports),
            'success_rate_trend': self._calculate_trend_static([r.successful_tasks / r.total_tasks if r.total_tasks > 0 else 0 for r in recent_reports]),
            'latency_trend': self._calculate_trend_static([r.average_latency for r in recent_reports]),
            'retry_trend': self._calculate_trend_static([r.total_retries / r.total_tasks if r.total_tasks > 0 else 0 for r in recent_reports])
        }

        return trends

    @staticmethod
    def _calculate_trend_static(values: List[float]) -> Dict[str, Any]:
        """Calculate trend statistics for a metric."""
        if len(values) < 2:
            return {'slope': 0.0, 'direction': 'stable', 'volatility': 0.0}

        try:
            # Simple linear regression slope
            n = len(values)
            x = list(range(n))
            slope = sum((x[i] - mean(x)) * (values[i] - mean(values)) for i in range(n))
            slope /= sum((x[i] - mean(x)) ** 2 for i in range(n))

            # Direction
            if slope > 0.01:
                direction = 'increasing'
            elif slope < -0.01:
                direction = 'decreasing'
            else:
                direction = 'stable'

            # Volatility (standard deviation)
            volatility = stdev(values) if len(values) > 1 else 0.0

            return {
                'slope': slope,
                'direction': direction,
                'volatility': volatility,
                'latest_value': values[-1]
            }
        except Exception:
            return {'slope': 0.0, 'direction': 'unknown', 'volatility': 0.0}


class MetricsAnalyzer:
    """Historical trend analyzer for advanced metrics analysis."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize metrics analyzer.

        Args:
            collector: MetricsCollector instance
        """
        self.collector = collector

    def trend_analysis(self, days_back: int = 7) -> Dict[str, Any]:
        """
        Perform comprehensive trend analysis.

        Args:
            days_back: Number of days to analyze

        Returns:
            Dictionary with trend analysis results
        """
        # This would typically query historical data from memory store
        # For now, using in-memory historical reports
        historical_data = self.collector.historical_reports

        if len(historical_data) < 2:
            return {'available': False, 'message': 'Insufficient historical data for trend analysis'}

        analysis = {
            'available': True,
            'time_range': f"{len(historical_data)} data points",
            'metrics': {}
        }

        # Analyze each key metric
        metrics_to_analyze = [
            ('success_rate', lambda r: r.successful_tasks / r.total_tasks if r.total_tasks > 0 else 0),
            ('latency', lambda r: r.average_latency),
            ('cpu_utilization', lambda r: r.system_cpu_percent),
            ('memory_utilization', lambda r: r.system_memory_percent),
            ('retry_rate', lambda r: r.total_retries / r.total_tasks if r.total_tasks > 0 else 0)
        ]

        for metric_name, extractor in metrics_to_analyze:
            values = [extractor(r) for r in historical_data]
            analysis['metrics'][metric_name] = self._analyze_metric_trend(values, metric_name)

        # Overall system health trend
        analysis['system_health_trend'] = self._calculate_system_health_trend(historical_data)

        return analysis

    def _analyze_metric_trend(self, values: List[float], metric_name: str) -> Dict[str, Any]:
        """Analyze trend for a specific metric."""
        if len(values) < 2:
            return {'trend': 'insufficient_data'}

        # Create a temporary performance monitor to access trend calculation
        temp_monitor = PerformanceMonitor(self.collector)
        trend_info = temp_monitor._calculate_trend_static(values)

        # Add metric-specific insights
        insights = []
        latest_value = values[-1]

        if metric_name == 'success_rate':
            if trend_info['direction'] == 'decreasing' and latest_value < 0.85:
                insights.append("Declining success rate may indicate system degradation")
            elif trend_info['direction'] == 'increasing':
                insights.append("Improving success rate indicates positive system evolution")
        elif metric_name == 'latency':
            if trend_info['direction'] == 'increasing' and latest_value > 1.0:
                insights.append("Increasing latency may require performance optimization")
        elif metric_name == 'cpu_utilization':
            if latest_value > 80:
                insights.append("High CPU utilization may impact system responsiveness")

        return {
            'trend': trend_info,
            'insights': insights,
            'anomaly_detected': self._detect_anomalies(values)
        }

    def _calculate_system_health_trend(self, historical_data: List[SystemMetrics]) -> Dict[str, Any]:
        """Calculate overall system health trend."""
        if not historical_data:
            return {'score': 0.5, 'trend': 'unknown'}

        # Simple health score based on multiple factors
        health_scores = []
        for report in historical_data:
            success_rate = report.successful_tasks / report.total_tasks if report.total_tasks > 0 else 0
            latency_score = max(0, 1 - (report.average_latency / 2.0))  # Normalize latency
            cpu_score = max(0, 1 - (report.system_cpu_percent / 100))
            memory_score = max(0, 1 - (report.system_memory_percent / 100))

            health_score = (success_rate * 0.4 + latency_score * 0.3 +
                          cpu_score * 0.15 + memory_score * 0.15)
            health_scores.append(health_score)

        temp_monitor = PerformanceMonitor(self.collector)
        trend_info = temp_monitor._calculate_trend_static(health_scores)

        return {
            'current_score': health_scores[-1] if health_scores else 0.5,
            'trend': trend_info['direction'],
            'slope': trend_info['slope']
        }

    def _detect_anomalies(self, values: List[float], threshold: float = 2.0) -> bool:
        """Detect anomalies in metric values using simple statistical method."""
        if len(values) < 3:
            return False

        try:
            mean_val = mean(values[:-1])  # Use all but latest
            std_val = stdev(values[:-1]) if len(values) > 2 else 0

            if std_val == 0:
                return False

            latest = values[-1]
            z_score = abs(latest - mean_val) / std_val

            return z_score > threshold
        except Exception:
            return False


class AdaptiveController:
    """Proto-self-regulation controller using metrics for adaptive thresholds."""

    def __init__(self, collector: MetricsCollector):
        """
        Initialize adaptive controller.

        Args:
            collector: MetricsCollector instance
        """
        self.collector = collector
        self.adaptation_rules = {
            'success_rate_low': {
                'condition': lambda m: (m.successful_tasks / m.total_tasks if m.total_tasks > 0 else 0) < 0.8,
                'action': 'increase_safety_strictness',
                'cooldown': 300  # 5 minutes
            },
            'latency_high': {
                'condition': lambda m: m.average_latency > 2.0,
                'action': 'scale_agents_up',
                'cooldown': 600  # 10 minutes
            },
            'cpu_high': {
                'condition': lambda m: m.system_cpu_percent > 80,
                'action': 'throttle_operations',
                'cooldown': 120  # 2 minutes
            }
        }
        self.last_actions = {}

    def evaluate_and_adapt(self) -> List[str]:
        """
        Evaluate current metrics and trigger adaptive actions.

        Returns:
            List of actions taken
        """
        actions_taken = []
        current_time = time.time()
        system_metrics = self.collector.get_system_metrics()

        for rule_name, rule in self.adaptation_rules.items():
            # Check cooldown
            if rule_name in self.last_actions:
                time_since_last = current_time - self.last_actions[rule_name]
                if time_since_last < rule['cooldown']:
                    continue

            # Check condition
            if rule['condition'](system_metrics):
                actions_taken.append(rule['action'])
                self.last_actions[rule_name] = current_time

                # Here you would integrate with actual system components
                # For now, just log the intended action
                logger.info(f"Adaptive action triggered: {rule['action']} (rule: {rule_name})")

        return actions_taken


# Integration functions
def integrate_with_testing(collector: MetricsCollector, test_harness: Any) -> None:
    """Integrate metrics collection with adversarial testing."""
    if not TESTING_AVAILABLE:
        logger.warning("Testing module not available for integration")
        return

    # Monkey patch the harness to collect metrics
    original_run_scenario = test_harness._run_scenario

    def instrumented_run_scenario(scenario):
        scenario_id = f"scenario_{scenario.name}_{int(time.time())}"

        collector.start_task(
            task_id=scenario_id,
            agent_id='AdversarialTestAgent',
            task_type='adversarial_testing',
            metadata={'scenario_type': scenario.scenario_type.value}
        )

        try:
            result = original_run_scenario(scenario)
            collector.end_task(
                task_id=scenario_id,
                success=result.result.name == 'PASS',
                error=result.message if result.result.name == 'ERROR' else None
            )
            return result
        except Exception as e:
            collector.end_task(task_id=scenario_id, success=False, error=str(e))
            raise

    test_harness._run_scenario = instrumented_run_scenario
    logger.info("✅ Metrics collection integrated with adversarial testing")


def create_visualization_data(performance_report: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create visualization-ready data from performance report.

    This is a foundation for future dashboard integration.
    """
    viz_data = {
        'charts': {
            'success_rate_trend': {
                'type': 'line',
                'data': [],  # Would be populated from historical data
                'title': 'Success Rate Over Time'
            },
            'latency_distribution': {
                'type': 'histogram',
                'data': [],  # Would be populated from task metrics
                'title': 'Task Latency Distribution'
            },
            'agent_utilization': {
                'type': 'bar',
                'data': performance_report['summary']['agent_utilization'],
                'title': 'Agent Utilization'
            }
        },
        'alerts': [],
        'summary_cards': {
            'total_tasks': performance_report['summary']['total_tasks'],
            'success_rate': f"{performance_report['summary']['success_rate']:.1%}",
            'average_latency': f"{performance_report['summary']['average_latency']:.2f}s",
            'predictive_risk': f"{performance_report['summary']['predictive_risk_score']:.2f}"
        }
    }

    # Add alerts based on CI checks
    ci_status = performance_report.get('ci_status', {})
    if not ci_status.get('overall_pass', True):
        for check_name, check_data in ci_status.get('checks', {}).items():
            if not check_data.get('passed', True):
                viz_data['alerts'].append({
                    'type': 'warning',
                    'message': f"{check_name.replace('_', ' ').title()} failed: {check_data['value']:.2f} > {check_data['threshold']}"
                })

    return viz_data