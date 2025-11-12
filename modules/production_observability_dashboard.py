# ============================================================
# Kalki v2.4 — production_observability_dashboard.py
# ------------------------------------------------------------
# Production Observability Dashboard: Real-time Metrics & Monitoring
# - Real-time performance metrics visualization
# - System health monitoring and alerting
# - Evolution tracking and trend analysis
# - Safety metrics dashboard
# - Interactive web-based monitoring interface
# ============================================================

import os
import json
import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
import statistics
import psutil
import webbrowser
from flask import Flask, render_template_string, jsonify, request
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

from modules.utils.logging_config import get_logger
from modules.self_evolution_manager import get_self_evolution_manager
from modules.safety_monitoring_system import get_safety_monitoring_system
from modules.reinforcement_loop import get_reinforcement_loop
from modules.temporal_consistency import get_temporal_consistency_buffer

logger = get_logger("Kalki.ObservabilityDashboard")

class MetricType(Enum):
    """Types of metrics to track"""
    PERFORMANCE = "performance"      # Response time, throughput, latency
    SAFETY = "safety"               # Safety violations, ethical compliance
    EVOLUTION = "evolution"          # Learning progress, adaptation metrics
    SYSTEM_HEALTH = "system_health"  # CPU, memory, disk usage
    USER_INTERACTION = "user_interaction"  # Usage patterns, satisfaction
    ERROR_RATE = "error_rate"        # Error rates and failure analysis

class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricDataPoint:
    """Individual metric data point"""
    timestamp: str
    metric_name: str
    value: float
    metric_type: MetricType
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AlertRule:
    """Alert rule definition"""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # e.g., "value > 0.8"
    threshold: float
    level: AlertLevel
    cooldown_minutes: int = 5
    enabled: bool = True
    last_triggered: Optional[str] = None

@dataclass
class ActiveAlert:
    """Active alert instance"""
    alert_id: str
    rule_id: str
    timestamp: str
    level: AlertLevel
    message: str
    metric_value: float
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[str] = None

@dataclass
class DashboardSnapshot:
    """Dashboard data snapshot"""
    timestamp: str
    metrics_summary: Dict[str, Any]
    active_alerts: List[ActiveAlert]
    system_status: str
    recent_evolution_events: List[Dict[str, Any]]
    performance_trends: Dict[str, Any]

class ProductionObservabilityDashboard:
    """
    Production Observability Dashboard: Real-time Metrics & Monitoring

    Provides comprehensive monitoring, alerting, and visualization of
    Kalki system performance, safety, and evolution metrics.
    """

    def __init__(self):
        self.evolution_manager = get_self_evolution_manager()
        self.safety_monitor = get_safety_monitoring_system()
        self.reinforcement_loop = get_reinforcement_loop()
        self.temporal_buffer = get_temporal_consistency_buffer()

        # Metrics storage
        self.metrics_history: List[MetricDataPoint] = []
        self.max_history_points = 10000  # Keep last 10k data points

        # Alert system
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, ActiveAlert] = {}

        # Dashboard state
        self.dashboard_snapshots: List[DashboardSnapshot] = []
        self.collection_interval = 30  # seconds
        self.is_collecting = False

        # Web dashboard
        self.app = Flask(__name__)
        self.app.json_encoder = PlotlyJSONEncoder
        self._setup_routes()

        # Configuration
        self.data_dir = "data/observability"
        self.metrics_file = f"{self.data_dir}/metrics_history.json"
        self.alerts_file = f"{self.data_dir}/alerts.json"

        # Initialize
        self._initialize_observability()

        logger.info("Production Observability Dashboard initialized")

    def _initialize_observability(self):
        """Initialize the observability system"""

        # Load existing data
        self._load_observability_data()

        # Set up default alert rules
        self._setup_default_alert_rules()

        # Start metrics collection
        self.start_metrics_collection()

    def _setup_default_alert_rules(self):
        """Set up default alert rules"""

        default_rules = [
            AlertRule(
                rule_id="high_error_rate",
                name="High Error Rate",
                description="Error rate exceeds 5%",
                metric_name="error_rate",
                condition="value > 0.05",
                threshold=0.05,
                level=AlertLevel.WARNING,
                cooldown_minutes=10
            ),
            AlertRule(
                rule_id="safety_violation",
                name="Safety Violation Detected",
                description="Safety violation detected in system operations",
                metric_name="safety_violations",
                condition="value > 0",
                threshold=0,
                level=AlertLevel.CRITICAL,
                cooldown_minutes=1
            ),
            AlertRule(
                rule_id="low_confidence",
                name="Low Response Confidence",
                description="Average response confidence below 70%",
                metric_name="avg_confidence",
                condition="value < 0.7",
                threshold=0.7,
                level=AlertLevel.WARNING,
                cooldown_minutes=15
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="System memory usage exceeds 90%",
                metric_name="memory_usage_percent",
                condition="value > 0.9",
                threshold=0.9,
                level=AlertLevel.ERROR,
                cooldown_minutes=5
            ),
            AlertRule(
                rule_id="evolution_stagnation",
                name="Evolution Stagnation",
                description="No evolution progress for 24 hours",
                metric_name="evolution_events_last_24h",
                condition="value == 0",
                threshold=0,
                level=AlertLevel.WARNING,
                cooldown_minutes=60  # 1 hour
            ),
            AlertRule(
                rule_id="temporal_contradictions",
                name="Temporal Contradictions Detected",
                description="High rate of temporal contradictions",
                metric_name="temporal_contradictions_per_hour",
                condition="value > 5",
                threshold=5,
                level=AlertLevel.ERROR,
                cooldown_minutes=30
            )
        ]

        for rule in default_rules:
            self.alert_rules[rule.rule_id] = rule

    def start_metrics_collection(self):
        """Start background metrics collection"""

        if self.is_collecting:
            return

        self.is_collecting = True

        # Start collection thread
        collection_thread = threading.Thread(target=self._metrics_collection_loop, daemon=True)
        collection_thread.start()

        logger.info("Metrics collection started")

    def stop_metrics_collection(self):
        """Stop metrics collection"""
        self.is_collecting = False
        logger.info("Metrics collection stopped")

    def _metrics_collection_loop(self):
        """Background metrics collection loop"""

        while self.is_collecting:
            try:
                self._collect_current_metrics()
                self._evaluate_alert_rules()
                self._cleanup_old_data()

                # Create dashboard snapshot every 5 minutes
                if len(self.metrics_history) % 10 == 0:  # Every 5 minutes (30s * 10)
                    self._create_dashboard_snapshot()

            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")

            time.sleep(self.collection_interval)

    def _collect_current_metrics(self):
        """Collect current system metrics"""

        timestamp = datetime.now().isoformat()

        # System health metrics
        system_metrics = self._collect_system_health_metrics(timestamp)

        # Performance metrics
        performance_metrics = self._collect_performance_metrics(timestamp)

        # Safety metrics
        safety_metrics = self._collect_safety_metrics(timestamp)

        # Evolution metrics
        evolution_metrics = self._collect_evolution_metrics(timestamp)

        # User interaction metrics (placeholder)
        interaction_metrics = self._collect_interaction_metrics(timestamp)

        # Combine all metrics
        all_metrics = system_metrics + performance_metrics + safety_metrics + evolution_metrics + interaction_metrics

        # Add to history
        self.metrics_history.extend(all_metrics)

        # Keep only recent history
        if len(self.metrics_history) > self.max_history_points:
            self.metrics_history = self.metrics_history[-self.max_history_points:]

    def _collect_system_health_metrics(self, timestamp: str) -> List[MetricDataPoint]:
        """Collect system health metrics"""

        metrics = []

        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="cpu_usage_percent",
                value=cpu_percent,
                metric_type=MetricType.SYSTEM_HEALTH,
                tags={"resource": "cpu"}
            ))

            # Memory usage
            memory = psutil.virtual_memory()
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="memory_usage_percent",
                value=memory.percent / 100.0,
                metric_type=MetricType.SYSTEM_HEALTH,
                tags={"resource": "memory"}
            ))

            # Disk usage
            disk = psutil.disk_usage('/')
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="disk_usage_percent",
                value=disk.percent / 100.0,
                metric_type=MetricType.SYSTEM_HEALTH,
                tags={"resource": "disk"}
            ))

        except Exception as e:
            logger.warning(f"Failed to collect system health metrics: {e}")

        return metrics

    def _collect_performance_metrics(self, timestamp: str) -> List[MetricDataPoint]:
        """Collect performance metrics"""

        metrics = []

        try:
            # Response time (placeholder - would integrate with actual request tracking)
            avg_response_time = 0.5  # seconds
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="avg_response_time",
                value=avg_response_time,
                metric_type=MetricType.PERFORMANCE,
                tags={"performance_type": "latency"}
            ))

            # Throughput (placeholder)
            requests_per_minute = 120
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="requests_per_minute",
                value=requests_per_minute,
                metric_type=MetricType.PERFORMANCE,
                tags={"performance_type": "throughput"}
            ))

            # Error rate (placeholder)
            error_rate = 0.02  # 2%
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="error_rate",
                value=error_rate,
                metric_type=MetricType.ERROR_RATE,
                tags={"error_type": "general"}
            ))

        except Exception as e:
            logger.warning(f"Failed to collect performance metrics: {e}")

        return metrics

    def _collect_safety_metrics(self, timestamp: str) -> List[MetricDataPoint]:
        """Collect safety metrics"""

        metrics = []

        try:
            # Safety violations (from safety monitor)
            safety_status = self.safety_monitor.get_safety_status()
            violations = safety_status.get("active_violations", 0)

            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="safety_violations",
                value=violations,
                metric_type=MetricType.SAFETY,
                tags={"safety_aspect": "violations"}
            ))

            # Ethical compliance score
            ethics_score = safety_status.get("ethics_compliance_score", 0.8)
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="ethics_compliance_score",
                value=ethics_score,
                metric_type=MetricType.SAFETY,
                tags={"safety_aspect": "ethics"}
            ))

            # Confidence calibration
            avg_confidence = 0.85  # placeholder
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="avg_confidence",
                value=avg_confidence,
                metric_type=MetricType.SAFETY,
                tags={"safety_aspect": "confidence"}
            ))

        except Exception as e:
            logger.warning(f"Failed to collect safety metrics: {e}")

        return metrics

    def _collect_evolution_metrics(self, timestamp: str) -> List[MetricDataPoint]:
        """Collect evolution metrics"""

        metrics = []

        try:
            # Evolution events in last 24h
            evolution_events_24h = 5  # placeholder
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="evolution_events_last_24h",
                value=evolution_events_24h,
                metric_type=MetricType.EVOLUTION,
                tags={"evolution_type": "events"}
            ))

            # Learning progress
            learning_progress = self.reinforcement_loop.get_learning_progress()
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="learning_progress_score",
                value=learning_progress.get("overall_score", 0.7),
                metric_type=MetricType.EVOLUTION,
                tags={"evolution_type": "learning"}
            ))

            # Temporal contradictions
            contradictions_per_hour = 2  # placeholder
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="temporal_contradictions_per_hour",
                value=contradictions_per_hour,
                metric_type=MetricType.EVOLUTION,
                tags={"evolution_type": "consistency"}
            ))

        except Exception as e:
            logger.warning(f"Failed to collect evolution metrics: {e}")

        return metrics

    def _collect_interaction_metrics(self, timestamp: str) -> List[MetricDataPoint]:
        """Collect user interaction metrics"""

        metrics = []

        try:
            # User satisfaction (placeholder)
            user_satisfaction = 0.88
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="user_satisfaction_score",
                value=user_satisfaction,
                metric_type=MetricType.USER_INTERACTION,
                tags={"interaction_type": "satisfaction"}
            ))

            # Session duration (placeholder)
            avg_session_duration = 25.5  # minutes
            metrics.append(MetricDataPoint(
                timestamp=timestamp,
                metric_name="avg_session_duration_minutes",
                value=avg_session_duration,
                metric_type=MetricType.USER_INTERACTION,
                tags={"interaction_type": "engagement"}
            ))

        except Exception as e:
            logger.warning(f"Failed to collect interaction metrics: {e}")

        return metrics

    def _evaluate_alert_rules(self):
        """Evaluate alert rules against current metrics"""

        # Get latest metrics for each metric name
        latest_metrics = {}
        for metric in reversed(self.metrics_history):
            if metric.metric_name not in latest_metrics:
                latest_metrics[metric.metric_name] = metric.value

        # Evaluate each rule
        for rule in self.alert_rules.values():
            if not rule.enabled:
                continue

            # Check cooldown
            if rule.last_triggered:
                last_triggered = datetime.fromisoformat(rule.last_triggered)
                cooldown_end = last_triggered + timedelta(minutes=rule.cooldown_minutes)
                if datetime.now() < cooldown_end:
                    continue

            # Get current value
            current_value = latest_metrics.get(rule.metric_name)
            if current_value is None:
                continue

            # Evaluate condition
            if self._evaluate_condition(current_value, rule.condition, rule.threshold):
                # Trigger alert
                self._trigger_alert(rule, current_value)

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""

        try:
            if ">" in condition:
                return value > threshold
            elif "<" in condition:
                return value < threshold
            elif ">=" in condition:
                return value >= threshold
            elif "<=" in condition:
                return value <= threshold
            elif "==" in condition:
                return value == threshold
            elif "!=" in condition:
                return value != threshold
            else:
                return False
        except:
            return False

    def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert"""

        alert_id = f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        alert = ActiveAlert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            timestamp=datetime.now().isoformat(),
            level=rule.level,
            message=f"{rule.name}: {rule.description} (Value: {current_value:.3f})",
            metric_value=current_value
        )

        self.active_alerts[alert_id] = alert
        rule.last_triggered = datetime.now().isoformat()

        logger.warning(f"Alert triggered: {alert.message}")

        # Could send notifications here
        # asyncio.create_task(self._send_alert_notification(alert))

    def _create_dashboard_snapshot(self):
        """Create a dashboard data snapshot"""

        try:
            # Get metrics summary
            metrics_summary = self._calculate_metrics_summary()

            # Get active alerts
            active_alerts = list(self.active_alerts.values())

            # System status
            system_status = self._determine_system_status(metrics_summary)

            # Recent evolution events (placeholder)
            recent_evolution_events = [
                {"timestamp": datetime.now().isoformat(), "event": "Evolution event", "type": "learning"}
            ]

            # Performance trends
            performance_trends = self._calculate_performance_trends()

            snapshot = DashboardSnapshot(
                timestamp=datetime.now().isoformat(),
                metrics_summary=metrics_summary,
                active_alerts=active_alerts,
                system_status=system_status,
                recent_evolution_events=recent_evolution_events,
                performance_trends=performance_trends
            )

            self.dashboard_snapshots.append(snapshot)

            # Keep only recent snapshots (last 24 hours worth)
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.dashboard_snapshots = [
                s for s in self.dashboard_snapshots
                if datetime.fromisoformat(s.timestamp) > cutoff_time
            ]

        except Exception as e:
            logger.error(f"Failed to create dashboard snapshot: {e}")

    def _calculate_metrics_summary(self) -> Dict[str, Any]:
        """Calculate metrics summary for dashboard"""

        summary = {}

        # Get recent metrics (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        # Group by metric type
        metrics_by_type = {}
        for metric in recent_metrics:
            if metric.metric_type.value not in metrics_by_type:
                metrics_by_type[metric.metric_type.value] = []
            metrics_by_type[metric.metric_type.value].append(metric.value)

        # Calculate averages for each type
        for metric_type, values in metrics_by_type.items():
            if values:
                summary[f"{metric_type}_avg"] = statistics.mean(values)
                summary[f"{metric_type}_min"] = min(values)
                summary[f"{metric_type}_max"] = max(values)

        # Overall health score
        health_components = []
        if "system_health_avg" in summary:
            health_components.append(1.0 - summary["system_health_avg"])  # Invert (lower usage = better)
        if "error_rate_avg" in summary:
            health_components.append(1.0 - summary["error_rate_avg"])
        if "safety_violations" in summary:
            health_components.append(1.0 if summary["safety_violations"] == 0 else 0.5)

        summary["overall_health_score"] = statistics.mean(health_components) if health_components else 0.8

        return summary

    def _determine_system_status(self, metrics_summary: Dict[str, Any]) -> str:
        """Determine overall system status"""

        health_score = metrics_summary.get("overall_health_score", 0.8)

        if health_score >= 0.9:
            return "healthy"
        elif health_score >= 0.7:
            return "warning"
        elif health_score >= 0.5:
            return "degraded"
        else:
            return "critical"

    def _calculate_performance_trends(self) -> Dict[str, Any]:
        """Calculate performance trends"""

        trends = {}

        # Get metrics from last 6 hours
        cutoff_time = datetime.now() - timedelta(hours=6)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        # Group by hour and metric
        hourly_data = {}
        for metric in recent_metrics:
            hour = datetime.fromisoformat(metric.timestamp).replace(minute=0, second=0, microsecond=0)
            hour_key = hour.isoformat()

            if hour_key not in hourly_data:
                hourly_data[hour_key] = {}

            if metric.metric_name not in hourly_data[hour_key]:
                hourly_data[hour_key][metric.metric_name] = []

            hourly_data[hour_key][metric.metric_name].append(metric.value)

        # Calculate trends for key metrics
        key_metrics = ["cpu_usage_percent", "memory_usage_percent", "error_rate", "avg_response_time"]

        for metric_name in key_metrics:
            values_by_hour = []
            for hour in sorted(hourly_data.keys()):
                if metric_name in hourly_data[hour]:
                    avg_value = statistics.mean(hourly_data[hour][metric_name])
                    values_by_hour.append(avg_value)

            if len(values_by_hour) >= 2:
                # Simple trend calculation
                recent_avg = statistics.mean(values_by_hour[-3:]) if len(values_by_hour) >= 3 else values_by_hour[-1]
                earlier_avg = statistics.mean(values_by_hour[:-3]) if len(values_by_hour) > 3 else values_by_hour[0]

                if earlier_avg > 0:
                    trend_percent = ((recent_avg - earlier_avg) / earlier_avg) * 100
                    trends[f"{metric_name}_trend_percent"] = trend_percent

        return trends

    def _cleanup_old_data(self):
        """Clean up old metrics data"""

        # Keep only last 24 hours of metrics
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.metrics_history = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        # Clean up old alerts (keep last 7 days)
        cutoff_time = datetime.now() - timedelta(days=7)
        self.active_alerts = {
            alert_id: alert for alert_id, alert in self.active_alerts.items()
            if datetime.fromisoformat(alert.timestamp) > cutoff_time
        }

    def _setup_routes(self):
        """Set up Flask routes for the dashboard"""

        dashboard = self

        @self.app.route('/')
        def index():
            return render_template_string(dashboard._get_dashboard_html())

        @self.app.route('/api/metrics')
        def get_metrics():
            return jsonify(dashboard._get_metrics_data())

        @self.app.route('/api/alerts')
        def get_alerts():
            return jsonify([asdict(alert) for alert in dashboard.active_alerts.values()])

        @self.app.route('/api/snapshots')
        def get_snapshots():
            return jsonify([asdict(snapshot) for snapshot in dashboard.dashboard_snapshots[-10:]])

        @self.app.route('/api/health')
        def health_check():
            return jsonify(dashboard._calculate_metrics_summary())

    def _get_dashboard_html(self) -> str:
        """Get the dashboard HTML template"""

        return """
<!DOCTYPE html>
<html>
<head>
    <title>Kalki Production Observability Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .header { background: #2c3e50; color: white; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .metric-card { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .status-healthy { color: #27ae60; }
        .status-warning { color: #f39c12; }
        .status-degraded { color: #e74c3c; }
        .status-critical { color: #c0392b; }
        .alert-list { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .alert-critical { border-left: 4px solid #e74c3c; padding-left: 10px; }
        .alert-warning { border-left: 4px solid #f39c12; padding-left: 10px; }
        .chart-container { background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Kalki Production Observability Dashboard</h1>
        <p>Real-time monitoring and metrics for the Kalki self-evolving AI system</p>
        <div id="system-status">Loading...</div>
    </div>

    <div class="metric-grid" id="metrics-grid">
        <!-- Metrics cards will be populated by JavaScript -->
    </div>

    <div class="chart-container">
        <h2>System Performance Trends</h2>
        <div id="performance-chart"></div>
    </div>

    <div class="chart-container">
        <h2>Safety & Ethics Metrics</h2>
        <div id="safety-chart"></div>
    </div>

    <div class="alert-list">
        <h2>Active Alerts</h2>
        <div id="alerts-list">Loading...</div>
    </div>

    <script>
        // Update dashboard data every 30 seconds
        async function updateDashboard() {
            try {
                const [metricsResponse, alertsResponse, healthResponse] = await Promise.all([
                    fetch('/api/metrics'),
                    fetch('/api/alerts'),
                    fetch('/api/health')
                ]);

                const metrics = await metricsResponse.json();
                const alerts = await alertsResponse.json();
                const health = await healthResponse.json();

                updateMetrics(metrics);
                updateAlerts(alerts);
                updateCharts(metrics);
                updateSystemStatus(health);

            } catch (error) {
                console.error('Error updating dashboard:', error);
            }
        }

        function updateMetrics(health) {
            const grid = document.getElementById('metrics-grid');
            const statusClass = health.overall_health_score >= 0.9 ? 'status-healthy' :
                              health.overall_health_score >= 0.7 ? 'status-warning' :
                              health.overall_health_score >= 0.5 ? 'status-degraded' : 'status-critical';

            grid.innerHTML = `
                <div class="metric-card">
                    <h3>Overall Health</h3>
                    <div class="status-${statusClass}" style="font-size: 24px; font-weight: bold;">
                        ${(health.overall_health_score * 100).toFixed(1)}%
                    </div>
                </div>
                <div class="metric-card">
                    <h3>CPU Usage</h3>
                    <div style="font-size: 24px;">${((health.system_health_avg || 0) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <h3>Memory Usage</h3>
                    <div style="font-size: 24px;">${((health.system_health_avg || 0) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <h3>Error Rate</h3>
                    <div style="font-size: 24px;">${((health.error_rate_avg || 0) * 100).toFixed(2)}%</div>
                </div>
                <div class="metric-card">
                    <h3>Safety Score</h3>
                    <div style="font-size: 24px;">${((health.safety_avg || 0.8) * 100).toFixed(1)}%</div>
                </div>
                <div class="metric-card">
                    <h3>Active Alerts</h3>
                    <div style="font-size: 24px;">${alerts.length}</div>
                </div>
            `;
        }

        function updateAlerts(alerts) {
            const alertsList = document.getElementById('alerts-list');

            if (alerts.length === 0) {
                alertsList.innerHTML = '<p>No active alerts</p>';
                return;
            }

            alertsList.innerHTML = alerts.map(alert => `
                <div class="alert-${alert.level}">
                    <strong>${alert.level.toUpperCase()}</strong>: ${alert.message}
                    <br><small>${new Date(alert.timestamp).toLocaleString()}</small>
                </div>
            `).join('');
        }

        function updateSystemStatus(health) {
            const statusDiv = document.getElementById('system-status');
            const statusClass = health.overall_health_score >= 0.9 ? 'status-healthy' :
                              health.overall_health_score >= 0.7 ? 'status-warning' :
                              health.overall_health_score >= 0.5 ? 'status-degraded' : 'status-critical';

            statusDiv.innerHTML = `
                <span class="${statusClass}">System Status: ${health.overall_health_score >= 0.9 ? 'HEALTHY' :
                    health.overall_health_score >= 0.7 ? 'WARNING' :
                    health.overall_health_score >= 0.5 ? 'DEGRADED' : 'CRITICAL'}</span>
                <br>Last updated: ${new Date().toLocaleString()}
            `;
        }

        function updateCharts(metrics) {
            // Placeholder for chart updates - would implement with actual metrics data
            // This would create interactive Plotly charts showing trends over time
        }

        // Initial load
        updateDashboard();

        // Update every 30 seconds
        setInterval(updateDashboard, 30000);
    </script>
</body>
</html>
        """

    def _get_metrics_data(self) -> Dict[str, Any]:
        """Get metrics data for API"""

        # Get recent metrics (last hour)
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_metrics = [
            m for m in self.metrics_history
            if datetime.fromisoformat(m.timestamp) > cutoff_time
        ]

        # Group by metric name
        metrics_data = {}
        for metric in recent_metrics:
            if metric.metric_name not in metrics_data:
                metrics_data[metric.metric_name] = {
                    "values": [],
                    "timestamps": [],
                    "type": metric.metric_type.value
                }

            metrics_data[metric.metric_name]["values"].append(metric.value)
            metrics_data[metric.metric_name]["timestamps"].append(metric.timestamp)

        return metrics_data

    def launch_dashboard(self, port: int = 8050):
        """Launch the dashboard web server"""

        def run_app():
            self.app.run(host='0.0.0.0', port=port, debug=False)

        # Start in background thread
        dashboard_thread = threading.Thread(target=run_app, daemon=True)
        dashboard_thread.start()

        # Open browser
        webbrowser.open(f'http://localhost:{port}')

        logger.info(f"Dashboard launched at http://localhost:{port}")

    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status"""

        return {
            "metrics_collected": len(self.metrics_history),
            "active_alerts": len(self.active_alerts),
            "alert_rules": len(self.alert_rules),
            "snapshots_created": len(self.dashboard_snapshots),
            "collection_active": self.is_collecting,
            "system_status": self._determine_system_status(self._calculate_metrics_summary())
        }

    def _load_observability_data(self):
        """Load observability data from files"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics_history = [MetricDataPoint(**m) for m in metrics_data]

            if os.path.exists(self.alerts_file):
                with open(self.alerts_file, 'r') as f:
                    alerts_data = json.load(f)
                    self.active_alerts = {a["alert_id"]: ActiveAlert(**a) for a in alerts_data}

        except Exception as e:
            logger.warning(f"Failed to load observability data: {e}")

    def _save_observability_data(self):
        """Save observability data to files"""
        try:
            # Save metrics (last 1000 points)
            recent_metrics = [asdict(m) for m in self.metrics_history[-1000:]]
            with open(self.metrics_file, 'w') as f:
                json.dump(recent_metrics, f, indent=2)

            # Save alerts
            alerts_data = [asdict(alert) for alert in self.active_alerts.values()]
            with open(self.alerts_file, 'w') as f:
                json.dump(alerts_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save observability data: {e}")

# Global observability dashboard instance
_observability_dashboard = None

def get_production_observability_dashboard() -> ProductionObservabilityDashboard:
    """Get the global production observability dashboard instance"""
    global _observability_dashboard
    if _observability_dashboard is None:
        _observability_dashboard = ProductionObservabilityDashboard()
    return _observability_dashboard

# Convenience functions
def launch_observability_dashboard(port: int = 8050):
    """Launch the observability dashboard"""
    dashboard = get_production_observability_dashboard()
    dashboard.launch_dashboard(port)