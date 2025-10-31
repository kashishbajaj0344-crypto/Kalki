"""
Performance Monitor Agent (Phase 6)
===================================

Tracks and reports performance metrics with statistical analysis.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from collections import defaultdict
import statistics
from modules.logging_config import get_logger

from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.PerformanceMonitor")


class PerformanceMonitorAgent(BaseAgent):
    """
    Tracks and reports performance metrics
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="PerformanceMonitorAgent",
            capabilities=[AgentCapability.QUALITY_ASSESSMENT],
            description="Performance metrics tracking and statistical analysis",
            config=config or {}
        )
        self.metrics = defaultdict(list)
        self.alerts = []

    async def initialize(self) -> bool:
        """Initialize performance monitoring system"""
        try:
            logger.info("PerformanceMonitorAgent initialized with statistical tracking")
            return True
        except Exception as e:
            logger.exception(f"Failed to initialize PerformanceMonitorAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance monitoring tasks"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "record":
            self.record_metric(
                params["metric_name"],
                params["value"],
                params.get("metadata")
            )
            return {"status": "success"}
        elif action == "stats":
            stats = self.get_metric_stats(params["metric_name"])
            return {"status": "success", "stats": stats}
        elif action == "list_metrics":
            return {"status": "success", "metrics": list(self.metrics.keys())}
        elif action == "alerts":
            return {"status": "success", "alerts": self.alerts}
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    def record_metric(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        try:
            entry = {
                "value": value,
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": metadata or {}
            }

            self.metrics[metric_name].append(entry)

            # Keep only last 1000 entries per metric
            if len(self.metrics[metric_name]) > 1000:
                self.metrics[metric_name] = self.metrics[metric_name][-1000:]

            # Check for alerts
            self._check_alerts(metric_name, value, metadata)

            logger.debug(f"Recorded metric {metric_name}: {value}")

        except Exception as e:
            logger.exception(f"Failed to record metric: {e}")

    def get_metric_stats(self, metric_name: str) -> Dict[str, Any]:
        """Get statistics for a metric"""
        try:
            if metric_name not in self.metrics:
                return {}

            values = [m["value"] for m in self.metrics[metric_name]]

            if not values:
                return {}

            stats = {
                "metric_name": metric_name,
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "latest": values[-1],
                "std_dev": statistics.stdev(values) if len(values) > 1 else 0,
                "time_range": {
                    "first": self.metrics[metric_name][0]["timestamp"],
                    "last": self.metrics[metric_name][-1]["timestamp"]
                }
            }

            # Calculate trend (last 10 vs previous 10)
            if len(values) >= 20:
                recent = values[-10:]
                previous = values[-20:-10]
                stats["trend"] = {
                    "recent_mean": statistics.mean(recent),
                    "previous_mean": statistics.mean(previous),
                    "change": statistics.mean(recent) - statistics.mean(previous),
                    "change_percent": ((statistics.mean(recent) - statistics.mean(previous)) / statistics.mean(previous)) * 100 if statistics.mean(previous) != 0 else 0
                }

            return stats

        except Exception as e:
            logger.exception(f"Failed to get metric stats: {e}")
            return {}

    def _check_alerts(self, metric_name: str, value: float, metadata: Optional[Dict[str, Any]]):
        """Check for performance alerts"""
        try:
            values = [m["value"] for m in self.metrics[metric_name]]

            if len(values) < 10:
                return  # Need minimum data for alerts

            # Calculate rolling statistics
            recent_values = values[-10:]
            mean = statistics.mean(recent_values)
            std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0

            # Check for anomalies (3 standard deviations from mean)
            if std_dev > 0 and abs(value - mean) > 3 * std_dev:
                alert = {
                    "alert_id": f"alert_{len(self.alerts)}",
                    "metric_name": metric_name,
                    "type": "anomaly",
                    "value": value,
                    "expected_range": {
                        "mean": mean,
                        "std_dev": std_dev,
                        "lower_bound": mean - 3 * std_dev,
                        "upper_bound": mean + 3 * std_dev
                    },
                    "severity": "high" if abs(value - mean) > 5 * std_dev else "medium",
                    "timestamp": datetime.utcnow().isoformat(),
                    "metadata": metadata
                }
                self.alerts.append(alert)
                logger.warning(f"Performance alert: {metric_name} anomaly detected (value: {value})")

            # Check for performance degradation (significant downward trend)
            if len(values) >= 20:
                recent_mean = statistics.mean(values[-10:])
                previous_mean = statistics.mean(values[-20:-10])

                if previous_mean > 0 and (previous_mean - recent_mean) / previous_mean > 0.2:  # 20% degradation
                    alert = {
                        "alert_id": f"alert_{len(self.alerts)}",
                        "metric_name": metric_name,
                        "type": "degradation",
                        "recent_mean": recent_mean,
                        "previous_mean": previous_mean,
                        "degradation_percent": ((previous_mean - recent_mean) / previous_mean) * 100,
                        "severity": "high",
                        "timestamp": datetime.utcnow().isoformat(),
                        "metadata": metadata
                    }
                    self.alerts.append(alert)
                    logger.warning(f"Performance alert: {metric_name} degradation detected ({alert['degradation_percent']:.1f}%)")

        except Exception as e:
            logger.exception(f"Alert checking error: {e}")

    def get_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent alerts"""
        return self.alerts[-limit:] if self.alerts else []

    async def shutdown(self) -> bool:
        """Shutdown the performance monitor agent"""
        try:
            logger.info("PerformanceMonitorAgent shutting down")
            return True
        except Exception as e:
            logger.exception(f"Shutdown error: {e}")
            return False