#!/usr/bin/env python3
"""
PerformanceMonitorAgent — Meta-metrics and health tracking
Aggregation logic with timestamped entries and statistical analysis
"""
import json
import psutil
import time
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from ..base_agent import BaseAgent


class PerformanceMonitorAgent(BaseAgent):
    """
    Meta-metrics and health tracking for system performance monitoring.
    Aggregates metrics with statistical analysis and anomaly detection.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="PerformanceMonitorAgent", config=config)
        self.metrics_history = defaultdict(list)
        self.anomaly_threshold = config.get("anomaly_threshold", 2.0) if config else 2.0  # Standard deviations
        self.metrics_file = Path.home() / "Desktop" / "Kalki" / "vector_db" / "performance_metrics.json"
        self._load_metrics_history()

    def _load_metrics_history(self):
        """Load persisted metrics history from disk"""
        try:
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    for metric_name, entries in data.get('metrics_history', {}).items():
                        self.metrics_history[metric_name] = entries
                self.logger.info(f"Loaded metrics history for {len(self.metrics_history)} metrics")
        except Exception as e:
            self.logger.warning(f"Failed to load metrics history: {e}")

    def _save_metrics_history(self):
        """Persist metrics history to disk"""
        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

            # Keep only recent entries (last 7 days)
            cutoff_time = (datetime.now() - timedelta(days=7)).isoformat()

            filtered_history = {}
            for metric_name, entries in self.metrics_history.items():
                recent_entries = [e for e in entries if e.get('timestamp', '') > cutoff_time]
                if recent_entries:
                    filtered_history[metric_name] = recent_entries[-1000:]  # Keep last 1000 entries

            data = {
                'metrics_history': filtered_history,
                'last_updated': datetime.now().isoformat()
            }

            with open(self.metrics_file, 'w') as f:
                json.dump(data, f, indent=2)

        except Exception as e:
            self.logger.error(f"Failed to save metrics history: {e}")

    def record_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Record a performance metric
        """
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "value": value,
                "metadata": metadata or {}
            }

            self.metrics_history[metric_name].append(entry)

            # Keep only recent entries in memory
            if len(self.metrics_history[metric_name]) > 1000:
                self.metrics_history[metric_name] = self.metrics_history[metric_name][-1000:]

            self._save_metrics_history()

            # Check for anomalies
            anomaly = self._detect_anomaly(metric_name, value)
            if anomaly:
                self.logger.warning(f"Anomaly detected in {metric_name}: {value} (deviation: {anomaly['deviation']:.2f}σ)")
                entry["anomaly"] = anomaly

            return entry

        except Exception as e:
            self.logger.exception(f"Failed to record metric {metric_name}: {e}")
            raise

    def get_system_metrics(self) -> Dict[str, Any]:
        """
        Collect current system performance metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            # Memory metrics
            memory = psutil.virtual_memory()

            # Disk metrics
            disk = psutil.disk_usage('/')

            # Network metrics (basic)
            network = psutil.net_io_counters()

            metrics = {
                "cpu_percent": cpu_percent,
                "cpu_count": cpu_count,
                "cpu_freq_current": cpu_freq.current if cpu_freq else None,
                "cpu_freq_max": cpu_freq.max if cpu_freq else None,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "memory_used_gb": memory.used / (1024**3),
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_percent": disk.percent,
                "network_bytes_sent": network.bytes_sent,
                "network_bytes_recv": network.bytes_recv,
                "timestamp": datetime.now().isoformat()
            }

            # Record key metrics automatically
            self.record_metric("cpu_percent", cpu_percent, {"system": True})
            self.record_metric("memory_percent", memory.percent, {"system": True})
            self.record_metric("disk_percent", disk.percent, {"system": True})

            return metrics

        except Exception as e:
            self.logger.exception(f"Failed to collect system metrics: {e}")
            return {"error": str(e)}

    def get_metric_stats(self, metric_name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get statistical summary for a metric over the specified time period
        """
        try:
            cutoff_time = (datetime.now() - timedelta(hours=hours)).isoformat()

            entries = [
                e for e in self.metrics_history.get(metric_name, [])
                if e.get('timestamp', '') > cutoff_time
            ]

            if not entries:
                return {
                    "metric_name": metric_name,
                    "count": 0,
                    "message": f"No data for {metric_name} in the last {hours} hours"
                }

            values = [e['value'] for e in entries]

            # Calculate statistics
            avg_value = sum(values) / len(values)
            min_value = min(values)
            max_value = max(values)

            # Calculate standard deviation
            variance = sum((x - avg_value) ** 2 for x in values) / len(values)
            std_dev = variance ** 0.5

            # Calculate trend (simple linear regression slope)
            if len(values) > 1:
                x_values = list(range(len(values)))
                slope = self._calculate_trend(x_values, values)
            else:
                slope = 0

            stats = {
                "metric_name": metric_name,
                "count": len(entries),
                "average": avg_value,
                "minimum": min_value,
                "maximum": max_value,
                "std_deviation": std_dev,
                "trend_slope": slope,
                "period_hours": hours,
                "latest_value": values[-1] if values else None,
                "latest_timestamp": entries[-1]['timestamp'] if entries else None
            }

            return stats

        except Exception as e:
            self.logger.exception(f"Failed to calculate stats for {metric_name}: {e}")
            return {"error": str(e)}

    def _calculate_trend(self, x_values: List[float], y_values: List[float]) -> float:
        """
        Calculate linear regression slope for trend analysis
        """
        try:
            n = len(x_values)
            if n < 2:
                return 0

            sum_x = sum(x_values)
            sum_y = sum(y_values)
            sum_xy = sum(x * y for x, y in zip(x_values, y_values))
            sum_x2 = sum(x * x for x in x_values)

            denominator = n * sum_x2 - sum_x * sum_x
            if denominator == 0:
                return 0

            slope = (n * sum_xy - sum_x * sum_y) / denominator
            return slope

        except Exception:
            return 0

    def _detect_anomaly(self, metric_name: str, current_value: float) -> Optional[Dict[str, Any]]:
        """
        Detect anomalies based on statistical deviation
        """
        try:
            # Get recent values (last hour)
            recent_entries = self.get_metric_stats(metric_name, hours=1)
            if recent_entries.get("count", 0) < 10:  # Need minimum data points
                return None

            avg_value = recent_entries["average"]
            std_dev = recent_entries["std_deviation"]

            if std_dev == 0:
                return None

            deviation = abs(current_value - avg_value) / std_dev

            if deviation > self.anomaly_threshold:
                return {
                    "deviation": deviation,
                    "threshold": self.anomaly_threshold,
                    "expected_range": {
                        "min": avg_value - (self.anomaly_threshold * std_dev),
                        "max": avg_value + (self.anomaly_threshold * std_dev)
                    },
                    "actual_value": current_value,
                    "timestamp": datetime.now().isoformat()
                }

            return None

        except Exception as e:
            self.logger.exception(f"Failed to detect anomaly for {metric_name}: {e}")
            return None

    def get_health_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive system health report
        """
        try:
            system_metrics = self.get_system_metrics()

            # Get stats for key metrics
            cpu_stats = self.get_metric_stats("cpu_percent", hours=1)
            memory_stats = self.get_metric_stats("memory_percent", hours=1)
            disk_stats = self.get_metric_stats("disk_percent", hours=1)

            # Determine health status
            health_score = 100

            # CPU health (lower is better)
            if cpu_stats.get("average", 0) > 80:
                health_score -= 20
            elif cpu_stats.get("average", 0) > 60:
                health_score -= 10

            # Memory health (lower is better)
            if memory_stats.get("average", 0) > 90:
                health_score -= 30
            elif memory_stats.get("average", 0) > 75:
                health_score -= 15

            # Disk health (lower is better)
            if disk_stats.get("average", 0) > 95:
                health_score -= 25
            elif disk_stats.get("average", 0) > 85:
                health_score -= 10

            health_status = "healthy" if health_score >= 80 else "warning" if health_score >= 60 else "critical"

            report = {
                "health_score": max(0, health_score),
                "health_status": health_status,
                "timestamp": datetime.now().isoformat(),
                "system_metrics": system_metrics,
                "performance_stats": {
                    "cpu": cpu_stats,
                    "memory": memory_stats,
                    "disk": disk_stats
                },
                "recommendations": self._generate_recommendations(health_score, system_metrics)
            }

            return report

        except Exception as e:
            self.logger.exception(f"Failed to generate health report: {e}")
            return {"error": str(e)}

    def _generate_recommendations(self, health_score: int, metrics: Dict[str, Any]) -> List[str]:
        """
        Generate health recommendations based on metrics
        """
        recommendations = []

        if metrics.get("cpu_percent", 0) > 80:
            recommendations.append("High CPU usage detected. Consider optimizing compute-intensive operations.")

        if metrics.get("memory_percent", 0) > 85:
            recommendations.append("High memory usage. Monitor for potential memory leaks.")

        if metrics.get("disk_percent", 0) > 90:
            recommendations.append("Low disk space. Consider cleaning up old data or expanding storage.")

        if health_score < 60:
            recommendations.append("Critical health issues detected. Immediate attention required.")

        if not recommendations:
            recommendations.append("System health is good. Continue monitoring.")

        return recommendations

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance monitoring tasks"""
        action = task.get("action")

        if action == "record":
            entry = self.record_metric(task["metric_name"], task["value"], task.get("metadata"))
            return {"status": "success", "entry": entry}
        elif action == "system_metrics":
            metrics = self.get_system_metrics()
            return {"status": "success", "metrics": metrics}
        elif action == "stats":
            stats = self.get_metric_stats(task["metric_name"], task.get("hours", 24))
            return {"status": "success", "stats": stats}
        elif action == "health_report":
            report = self.get_health_report()
            return {"status": "success", "report": report}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}