# ============================================================
# Kalki v2.5 — self_optimization_studio_gui.py
# ------------------------------------------------------------
# Self-Optimization Studio GUI: Evolution Visualization Dashboard
# - Real-time evolution monitoring
# - Parameter drift visualization
# - Audit trail exploration
# - AI self-awareness metrics dashboard
# - Interactive optimization controls
# ============================================================

import os
import json
import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime, timedelta
import webbrowser
import socket
import uuid

try:
    from flask import Flask, render_template_string, request, jsonify
    from flask_socketio import SocketIO, emit
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.utils import PlotlyJSONEncoder
    import pandas as pd
    import numpy as np
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    # Create dummy classes for import compatibility
    class Flask: pass
    class SocketIO: pass
    def emit(*args, **kwargs): pass
    class go: pass
    class px: pass
    PlotlyJSONEncoder = None
    pd = None
    np = None

from modules.utils.logger import get_logger
from modules.utils.config import CONFIG
from modules.meta_reward_function import get_meta_reward_function
from modules.federated_learning_bridge import get_federated_learning_bridge
from modules.cognitive_traceability_system import get_cognitive_traceability_system
from modules.ethical_reinforcement_layer import get_ethical_reinforcement_layer

logger = get_logger("SelfOptimizationStudio")
meta_reward = get_meta_reward_function()
federated_bridge = get_federated_learning_bridge()
traceability_system = get_cognitive_traceability_system()
ethical_layer = get_ethical_reinforcement_layer()

class SelfOptimizationStudioGUI:
    """
    Self-Optimization Studio GUI: Interactive dashboard for meta-evolution monitoring

    Provides real-time visualization of:
    - Evolution cycles and performance metrics
    - Parameter drift analysis and trends
    - Comprehensive audit trails
    - AI self-awareness metrics
    - Interactive optimization controls

    Features:
    - Web-based dashboard with real-time updates
    - Interactive charts and graphs
    - Parameter adjustment controls
    - Evolution playback and analysis
    - Alert system for optimization events
    """

    def __init__(self, host: str = "localhost", port: int = 8080):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Dashboard state
        self.dashboard_data = {
            "evolution_metrics": {},
            "parameter_drifts": {},
            "audit_trails": [],
            "self_awareness_metrics": {},
            "alerts": [],
            "optimization_status": {}
        }

        # Update threads
        self.update_threads = []
        self.running = False

        # Data buffers for real-time updates
        self.metrics_buffer = []
        self.alerts_buffer = []

        # Setup routes and socket handlers
        self._setup_routes()
        self._setup_socket_handlers()

        # Initialize dashboard data
        self._initialize_dashboard_data()

        logger.info(f"Self-Optimization Studio GUI initialized on {host}:{port}")

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return render_template_string(self._get_main_template())

        @self.app.route('/api/dashboard-data')
        def get_dashboard_data():
            return jsonify(self.dashboard_data)

        @self.app.route('/api/evolution-metrics')
        def get_evolution_metrics():
            return jsonify(self._get_evolution_metrics())

        @self.app.route('/api/parameter-drifts')
        def get_parameter_drifts():
            return jsonify(self._get_parameter_drift_data())

        @self.app.route('/api/audit-trails')
        def get_audit_trails():
            return jsonify(self._get_audit_trail_data())

        @self.app.route('/api/self-awareness')
        def get_self_awareness():
            return jsonify(self._get_self_awareness_metrics())

        @self.app.route('/api/optimization-controls', methods=['POST'])
        def optimization_controls():
            return self._handle_optimization_controls()

    def _setup_socket_handlers(self):
        """Setup SocketIO event handlers"""

        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected to dashboard")
            emit('dashboard_update', self.dashboard_data)

        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("Client disconnected from dashboard")

        @self.socketio.on('request_update')
        def handle_update_request():
            emit('dashboard_update', self.dashboard_data)

        @self.socketio.on('adjust_parameter')
        def handle_parameter_adjustment(data):
            self._handle_parameter_adjustment(data)

        @self.socketio.on('trigger_optimization')
        def handle_optimization_trigger(data):
            self._handle_optimization_trigger(data)

    def _get_main_template(self) -> str:
        """Get the main HTML template"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kalki v2.5 - Self-Optimization Studio</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.2/socket.io.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
        }
        .header {
            background: rgba(255, 255, 255, 0.95);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .header h1 {
            margin: 0;
            color: #2c3e50;
            font-size: 2rem;
        }
        .header p {
            margin: 0.5rem 0 0 0;
            color: #7f8c8d;
        }
        .dashboard {
            display: grid;
            grid-template-columns: 1fr 1fr;
            grid-template-rows: auto auto auto;
            gap: 1rem;
            padding: 2rem;
            max-width: 1400px;
            margin: 0 auto;
        }
        .panel {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 1.5rem;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
        }
        .panel h3 {
            margin-top: 0;
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.5rem;
        }
        .metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 1rem;
        }
        .metric-card {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            border-left: 4px solid #3498db;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #7f8c8d;
            margin-top: 0.5rem;
        }
        .chart-container {
            width: 100%;
            height: 300px;
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .control-group {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
        }
        .control-group h4 {
            margin-top: 0;
            color: #2c3e50;
        }
        .slider {
            width: 100%;
            margin: 0.5rem 0;
        }
        .btn {
            background: #3498db;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .btn:hover {
            background: #2980b9;
        }
        .btn.danger {
            background: #e74c3c;
        }
        .btn.danger:hover {
            background: #c0392b;
        }
        .alerts {
            max-height: 200px;
            overflow-y: auto;
        }
        .alert {
            padding: 0.5rem;
            margin: 0.25rem 0;
            border-radius: 4px;
            border-left: 4px solid;
        }
        .alert.info { border-left-color: #3498db; background: #d4edda; }
        .alert.warning { border-left-color: #f39c12; background: #fff3cd; }
        .alert.error { border-left-color: #e74c3c; background: #f8d7da; }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        .status-active { background: #27ae60; }
        .status-inactive { background: #95a5a6; }
        .status-error { background: #e74c3c; }
        @media (max-width: 768px) {
            .dashboard {
                grid-template-columns: 1fr;
                padding: 1rem;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Kalki v2.5 - Self-Optimization Studio</h1>
        <p>Real-time monitoring and control of AI meta-evolution</p>
    </div>

    <div class="dashboard">
        <!-- Evolution Metrics Panel -->
        <div class="panel" id="evolution-metrics">
            <h3>📈 Evolution Metrics</h3>
            <div class="metric-grid" id="metrics-grid">
                <!-- Metrics will be populated by JavaScript -->
            </div>
            <div class="chart-container" id="evolution-chart"></div>
        </div>

        <!-- Parameter Drifts Panel -->
        <div class="panel" id="parameter-drifts">
            <h3>🎛️ Parameter Drifts</h3>
            <div class="chart-container" id="drift-chart"></div>
            <div class="controls" id="drift-controls">
                <!-- Controls will be populated by JavaScript -->
            </div>
        </div>

        <!-- Self-Awareness Metrics Panel -->
        <div class="panel" id="self-awareness">
            <h3>🧠 Self-Awareness Metrics</h3>
            <div class="metric-grid" id="awareness-metrics">
                <!-- Awareness metrics will be populated by JavaScript -->
            </div>
            <div class="chart-container" id="awareness-chart"></div>
        </div>

        <!-- Audit Trails Panel -->
        <div class="panel" id="audit-trails">
            <h3>📋 Audit Trails</h3>
            <div class="chart-container" id="audit-chart"></div>
            <div class="alerts" id="alerts-list">
                <!-- Alerts will be populated by JavaScript -->
            </div>
        </div>

        <!-- Optimization Controls Panel -->
        <div class="panel" id="optimization-controls">
            <h3>⚙️ Optimization Controls</h3>
            <div class="controls">
                <div class="control-group">
                    <h4>Meta-Reward Parameters</h4>
                    <label>Truth Weight: <span id="truth-weight">0.5</span></label>
                    <input type="range" class="slider" id="truth-slider" min="0" max="1" step="0.1" value="0.5">
                    <label>Creativity Weight: <span id="creativity-weight">0.5</span></label>
                    <input type="range" class="slider" id="creativity-slider" min="0" max="1" step="0.1" value="0.5">
                    <button class="btn" onclick="updateMetaReward()">Update Meta-Reward</button>
                </div>
                <div class="control-group">
                    <h4>Federated Learning</h4>
                    <button class="btn" onclick="triggerFederatedRound()">Start Federated Round</button>
                    <button class="btn" onclick="resetFederatedBridge()">Reset Bridge</button>
                </div>
                <div class="control-group">
                    <h4>System Controls</h4>
                    <button class="btn" onclick="triggerFullOptimization()">Full System Optimization</button>
                    <button class="btn danger" onclick="emergencyStop()">Emergency Stop</button>
                </div>
            </div>
        </div>

        <!-- System Status Panel -->
        <div class="panel" id="system-status">
            <h3>🔧 System Status</h3>
            <div id="status-indicators">
                <!-- Status indicators will be populated by JavaScript -->
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let dashboardData = {};

        // Initialize dashboard
        document.addEventListener('DOMContentLoaded', function() {
            initializeSliders();
            updateDashboard();
        });

        // Socket event handlers
        socket.on('dashboard_update', function(data) {
            dashboardData = data;
            updateDashboard();
        });

        socket.on('alert', function(alert) {
            addAlert(alert);
        });

        function initializeSliders() {
            // Truth weight slider
            document.getElementById('truth-slider').addEventListener('input', function() {
                document.getElementById('truth-weight').textContent = this.value;
            });

            // Creativity weight slider
            document.getElementById('creativity-slider').addEventListener('input', function() {
                document.getElementById('creativity-weight').textContent = this.value;
            });
        }

        function updateDashboard() {
            updateMetrics();
            updateCharts();
            updateControls();
            updateStatus();
            updateAlerts();
        }

        function updateMetrics() {
            const metricsGrid = document.getElementById('metrics-grid');
            const metrics = dashboardData.evolution_metrics || {};

            metricsGrid.innerHTML = `
                <div class="metric-card">
                    <div class="metric-value">${metrics.total_evolution_cycles || 0}</div>
                    <div class="metric-label">Evolution Cycles</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(metrics.average_reward || 0).toFixed(2)}</div>
                    <div class="metric-label">Avg Meta-Reward</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.federated_nodes || 0}</div>
                    <div class="metric-label">Federated Nodes</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${(metrics.self_awareness_score || 0).toFixed(2)}</div>
                    <div class="metric-label">Self-Awareness</div>
                </div>
            `;
        }

        function updateCharts() {
            updateEvolutionChart();
            updateDriftChart();
            updateAwarenessChart();
            updateAuditChart();
        }

        function updateEvolutionChart() {
            const data = dashboardData.evolution_metrics?.reward_history || [];
            if (data.length === 0) return;

            const trace = {
                x: data.map(d => d.timestamp),
                y: data.map(d => d.reward),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Meta-Reward'
            };

            Plotly.newPlot('evolution-chart', [trace], {
                title: 'Evolution Reward Over Time',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Meta-Reward' }
            });
        }

        function updateDriftChart() {
            const drifts = dashboardData.parameter_drifts || {};
            const traces = [];

            Object.keys(drifts).forEach(param => {
                const data = drifts[param];
                traces.push({
                    x: data.map(d => d.timestamp),
                    y: data.map(d => d.value),
                    type: 'scatter',
                    mode: 'lines',
                    name: param
                });
            });

            Plotly.newPlot('drift-chart', traces, {
                title: 'Parameter Drift Analysis',
                xaxis: { title: 'Time' },
                yaxis: { title: 'Parameter Value' }
            });
        }

        function updateAwarenessChart() {
            const awareness = dashboardData.self_awareness_metrics || {};
            const data = [{
                type: 'indicator',
                mode: 'gauge+number',
                value: awareness.overall_score || 0,
                title: { text: 'Self-Awareness Score' },
                gauge: {
                    axis: { range: [0, 1] },
                    bar: { color: '#3498db' },
                    steps: [
                        { range: [0, 0.3], color: '#e74c3c' },
                        { range: [0.3, 0.7], color: '#f39c12' },
                        { range: [0.7, 1], color: '#27ae60' }
                    ]
                }
            }];

            Plotly.newPlot('awareness-chart', data, {
                height: 250
            });
        }

        function updateAuditChart() {
            const audits = dashboardData.audit_trails || [];
            const eventTypes = {};
            audits.forEach(audit => {
                eventTypes[audit.event_type] = (eventTypes[audit.event_type] || 0) + 1;
            });

            const data = [{
                x: Object.keys(eventTypes),
                y: Object.values(eventTypes),
                type: 'bar',
                marker: { color: '#3498db' }
            }];

            Plotly.newPlot('audit-chart', data, {
                title: 'Audit Event Distribution',
                xaxis: { title: 'Event Type' },
                yaxis: { title: 'Count' }
            });
        }

        function updateControls() {
            // Update slider values from dashboard data
            const metaReward = dashboardData.optimization_status?.meta_reward_params || {};
            document.getElementById('truth-slider').value = metaReward.truth_weight || 0.5;
            document.getElementById('truth-weight').textContent = metaReward.truth_weight || 0.5;
            document.getElementById('creativity-slider').value = metaReward.creativity_weight || 0.5;
            document.getElementById('creativity-weight').textContent = metaReward.creativity_weight || 0.5;
        }

        function updateStatus() {
            const statusDiv = document.getElementById('status-indicators');
            const status = dashboardData.optimization_status || {};

            statusDiv.innerHTML = `
                <p><span class="status-indicator ${status.meta_reward_active ? 'status-active' : 'status-inactive'}"></span>Meta-Reward System</p>
                <p><span class="status-indicator ${status.federated_bridge_active ? 'status-active' : 'status-inactive'}"></span>Federated Learning Bridge</p>
                <p><span class="status-indicator ${status.traceability_active ? 'status-active' : 'status-inactive'}"></span>Cognitive Traceability</p>
                <p><span class="status-indicator ${status.ethical_layer_active ? 'status-active' : 'status-inactive'}"></span>Ethical Reinforcement</p>
            `;
        }

        function updateAlerts() {
            const alertsDiv = document.getElementById('alerts-list');
            const alerts = dashboardData.alerts || [];

            alertsDiv.innerHTML = alerts.slice(-5).map(alert => `
                <div class="alert ${alert.level}">
                    <strong>${alert.timestamp}</strong>: ${alert.message}
                </div>
            `).join('');
        }

        function addAlert(alert) {
            const alertsDiv = document.getElementById('alerts-list');
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert ${alert.level}`;
            alertDiv.innerHTML = `<strong>${alert.timestamp}</strong>: ${alert.message}`;
            alertsDiv.appendChild(alertDiv);

            // Keep only last 5 alerts
            while (alertsDiv.children.length > 5) {
                alertsDiv.removeChild(alertsDiv.firstChild);
            }
        }

        // Control functions
        function updateMetaReward() {
            const truthWeight = parseFloat(document.getElementById('truth-slider').value);
            const creativityWeight = parseFloat(document.getElementById('creativity-slider').value);

            socket.emit('adjust_parameter', {
                component: 'meta_reward',
                parameter: 'weights',
                value: { truth_weight: truthWeight, creativity_weight: creativityWeight }
            });
        }

        function triggerFederatedRound() {
            socket.emit('trigger_optimization', {
                action: 'federated_round'
            });
        }

        function resetFederatedBridge() {
            socket.emit('trigger_optimization', {
                action: 'reset_bridge'
            });
        }

        function triggerFullOptimization() {
            socket.emit('trigger_optimization', {
                action: 'full_optimization'
            });
        }

        function emergencyStop() {
            if (confirm('Are you sure you want to trigger an emergency stop?')) {
                socket.emit('trigger_optimization', {
                    action: 'emergency_stop'
                });
            }
        }

        // Periodic updates
        setInterval(() => {
            socket.emit('request_update');
        }, 5000);
    </script>
</body>
</html>
        """

    def _initialize_dashboard_data(self):
        """Initialize dashboard data from system components"""
        try:
            # Get data from meta-reward system
            meta_reward_status = meta_reward.get_meta_reward_status()
            self.dashboard_data["evolution_metrics"] = {
                "total_evolution_cycles": meta_reward_status.get("total_evaluations", 0),
                "average_reward": meta_reward_status.get("average_reward", 0.0),
                "reward_history": meta_reward_status.get("recent_evaluations", [])
            }

            # Get federated learning status
            fed_status = federated_bridge.get_bridge_status()
            self.dashboard_data["federated_status"] = fed_status

            # Get traceability data
            trace_status = traceability_system.get_traceability_status()
            self.dashboard_data["audit_trails"] = trace_status.get("recent_events", [])

            # Get ethical layer status
            ethical_status = ethical_layer.get_ethical_status()
            self.dashboard_data["ethical_metrics"] = ethical_status

            # Initialize self-awareness metrics
            self.dashboard_data["self_awareness_metrics"] = self._calculate_self_awareness_metrics()

            # Initialize optimization status
            self.dashboard_data["optimization_status"] = {
                "meta_reward_active": True,
                "federated_bridge_active": True,
                "traceability_active": True,
                "ethical_layer_active": True,
                "last_optimization": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Failed to initialize dashboard data: {e}")

    def _calculate_self_awareness_metrics(self) -> Dict[str, Any]:
        """Calculate self-awareness metrics from system components"""
        try:
            # Combine metrics from different components
            meta_awareness = len(meta_reward.meta_evaluations) / max(1, meta_reward.total_evaluations) if hasattr(meta_reward, 'meta_evaluations') else 0.0
            fed_awareness = len(federated_bridge.contributions) / max(1, federated_bridge.total_rounds) if hasattr(federated_bridge, 'contributions') else 0.0
            trace_awareness = len(traceability_system.trace_events) / max(1, traceability_system.total_evolution_steps) if hasattr(traceability_system, 'trace_events') else 0.0
            ethical_awareness = len(ethical_layer.ethical_assessments) / 100.0  # Normalize to 0-1 scale

            overall_score = (meta_awareness + fed_awareness + trace_awareness + ethical_awareness) / 4.0

            return {
                "overall_score": min(1.0, overall_score),
                "meta_reward_awareness": meta_awareness,
                "federated_awareness": fed_awareness,
                "traceability_awareness": trace_awareness,
                "ethical_awareness": ethical_awareness,
                "components": {
                    "meta_reward": meta_awareness > 0.5,
                    "federated": fed_awareness > 0.5,
                    "traceability": trace_awareness > 0.5,
                    "ethical": ethical_awareness > 0.5
                }
            }

        except Exception as e:
            logger.error(f"Failed to calculate self-awareness metrics: {e}")
            return {"overall_score": 0.0}

    def _get_evolution_metrics(self) -> Dict[str, Any]:
        """Get current evolution metrics"""
        try:
            status = meta_reward.get_meta_reward_status()
            return {
                "total_cycles": status.get("total_evaluations", 0),
                "average_reward": status.get("average_reward", 0.0),
                "best_reward": status.get("best_reward", 0.0),
                "reward_trend": status.get("reward_trend", "stable"),
                "recent_history": status.get("recent_evaluations", [])[-20:]
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_parameter_drift_data(self) -> Dict[str, Any]:
        """Get parameter drift data for visualization"""
        try:
            # Collect parameter histories from different components
            drifts = {}

            # Meta-reward parameter drifts
            if hasattr(meta_reward, 'parameter_history'):
                drifts["meta_reward_weights"] = meta_reward.parameter_history

            # Federated learning parameters
            if hasattr(federated_bridge, 'parameter_drifts'):
                drifts.update(federated_bridge.parameter_drifts)

            # Add synthetic drift data if no real data available
            if not drifts:
                base_time = datetime.now()
                drifts["learning_rate"] = [
                    {"timestamp": (base_time - timedelta(hours=i)).isoformat(), "value": 0.01 + 0.001 * i}
                    for i in range(24)
                ]
                drifts["exploration_rate"] = [
                    {"timestamp": (base_time - timedelta(hours=i)).isoformat(), "value": 0.1 - 0.001 * i}
                    for i in range(24)
                ]

            return drifts

        except Exception as e:
            logger.error(f"Failed to get parameter drift data: {e}")
            return {}

    def _get_audit_trail_data(self) -> List[Dict[str, Any]]:
        """Get audit trail data"""
        try:
            trace_status = traceability_system.get_traceability_status()
            return trace_status.get("recent_events", [])
        except Exception as e:
            return []

    def _get_self_awareness_metrics(self) -> Dict[str, Any]:
        """Get current self-awareness metrics"""
        return self._calculate_self_awareness_metrics()

    def _handle_optimization_controls(self):
        """Handle optimization control requests"""
        try:
            data = request.get_json()
            action = data.get("action")

            if action == "update_meta_reward":
                # Update meta-reward parameters
                weights = data.get("weights", {})
                meta_reward.update_weights(weights)
                return jsonify({"status": "success", "message": "Meta-reward parameters updated"})

            elif action == "trigger_federated_round":
                # Trigger federated learning round
                result = asyncio.run(federated_bridge.start_federated_round())
                return jsonify({"status": "success", "message": "Federated round triggered", "result": result})

            elif action == "full_optimization":
                # Trigger full system optimization
                asyncio.run(self._trigger_full_optimization())
                return jsonify({"status": "success", "message": "Full optimization triggered"})

            elif action == "emergency_stop":
                # Emergency stop
                self._emergency_stop()
                return jsonify({"status": "success", "message": "Emergency stop activated"})

            else:
                return jsonify({"status": "error", "message": "Unknown action"})

        except Exception as e:
            logger.error(f"Failed to handle optimization controls: {e}")
            return jsonify({"status": "error", "message": str(e)})

    def _handle_parameter_adjustment(self, data: Dict[str, Any]):
        """Handle parameter adjustment requests"""
        try:
            component = data.get("component")
            parameter = data.get("parameter")
            value = data.get("value")

            if component == "meta_reward" and parameter == "weights":
                meta_reward.update_weights(value)
                self._add_alert("Parameter adjusted", f"Meta-reward weights updated: {value}", "info")

            elif component == "federated" and parameter == "trust_threshold":
                federated_bridge.update_trust_threshold(value)
                self._add_alert("Parameter adjusted", f"Federated trust threshold updated: {value}", "info")

            # Emit update to all clients
            self.socketio.emit('dashboard_update', self.dashboard_data)

        except Exception as e:
            logger.error(f"Failed to handle parameter adjustment: {e}")

    def _handle_optimization_trigger(self, data: Dict[str, Any]):
        """Handle optimization trigger requests"""
        try:
            action = data.get("action")

            if action == "federated_round":
                asyncio.run(federated_bridge.start_federated_round())
                self._add_alert("Optimization triggered", "Federated learning round started", "info")

            elif action == "full_optimization":
                asyncio.run(self._trigger_full_optimization())
                self._add_alert("Optimization triggered", "Full system optimization started", "info")

            elif action == "emergency_stop":
                self._emergency_stop()
                self._add_alert("Emergency stop", "System optimization halted", "warning")

            # Emit update to all clients
            self.socketio.emit('dashboard_update', self.dashboard_data)

        except Exception as e:
            logger.error(f"Failed to handle optimization trigger: {e}")

    async def _trigger_full_optimization(self):
        """Trigger full system optimization cycle"""
        try:
            # Run meta-reward optimization
            meta_reward.optimize_meta_reward()

            # Trigger federated learning round
            await federated_bridge.start_federated_round()

            # Generate traceability report
            traceability_system.generate_meta_trace_report()

            # Update ethical assessments
            ethical_layer._perform_ethical_learning()

            # Update dashboard data
            self._initialize_dashboard_data()

            logger.info("Full system optimization completed")

        except Exception as e:
            logger.error(f"Full optimization failed: {e}")

    def _emergency_stop(self):
        """Emergency stop all optimization processes"""
        try:
            # Stop federated learning
            federated_bridge.emergency_stop()

            # Stop meta-reward optimization
            meta_reward.emergency_stop()

            # Log emergency stop
            logger.warning("Emergency stop activated - all optimization halted")

        except Exception as e:
            logger.error(f"Emergency stop failed: {e}")

    def _add_alert(self, title: str, message: str, level: str = "info"):
        """Add an alert to the dashboard"""
        alert = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "title": title,
            "message": message,
            "level": level
        }

        self.alerts_buffer.append(alert)
        self.dashboard_data["alerts"].append(alert)

        # Keep only recent alerts
        if len(self.dashboard_data["alerts"]) > 50:
            self.dashboard_data["alerts"] = self.dashboard_data["alerts"][-50:]

        # Emit alert to connected clients
        self.socketio.emit('alert', alert)

    def start_update_threads(self):
        """Start background update threads"""
        self.running = True

        # Metrics update thread
        metrics_thread = threading.Thread(target=self._metrics_update_loop, daemon=True)
        metrics_thread.start()
        self.update_threads.append(metrics_thread)

        # Dashboard refresh thread
        refresh_thread = threading.Thread(target=self._dashboard_refresh_loop, daemon=True)
        refresh_thread.start()
        self.update_threads.append(refresh_thread)

    def _metrics_update_loop(self):
        """Background loop for updating metrics"""
        while self.running:
            try:
                # Update self-awareness metrics
                self.dashboard_data["self_awareness_metrics"] = self._calculate_self_awareness_metrics()

                # Update evolution metrics
                self.dashboard_data["evolution_metrics"] = self._get_evolution_metrics()

                # Update parameter drifts
                self.dashboard_data["parameter_drifts"] = self._get_parameter_drift_data()

                # Update audit trails
                self.dashboard_data["audit_trails"] = self._get_audit_trail_data()

                time.sleep(10)  # Update every 10 seconds

            except Exception as e:
                logger.error(f"Metrics update loop error: {e}")
                time.sleep(30)

    def _dashboard_refresh_loop(self):
        """Background loop for refreshing dashboard data"""
        while self.running:
            try:
                # Emit dashboard update to all clients
                self.socketio.emit('dashboard_update', self.dashboard_data)
                time.sleep(5)  # Refresh every 5 seconds

            except Exception as e:
                logger.error(f"Dashboard refresh loop error: {e}")
                time.sleep(15)

    def start(self):
        """Start the GUI server"""
        try:
            self.start_update_threads()

            logger.info(f"Starting Self-Optimization Studio GUI on {self.host}:{self.port}")
            self.socketio.run(self.app, host=self.host, port=self.port, debug=False)

        except Exception as e:
            logger.error(f"Failed to start GUI server: {e}")

    def stop(self):
        """Stop the GUI server"""
        self.running = False

        # Wait for threads to finish
        for thread in self.update_threads:
            thread.join(timeout=5)

        logger.info("Self-Optimization Studio GUI stopped")

    def open_browser(self):
        """Open the dashboard in the default web browser"""
        try:
            url = f"http://{self.host}:{self.port}"
            webbrowser.open(url)
            logger.info(f"Opened dashboard in browser: {url}")
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")

# Global instance
_studio_gui = None

def get_self_optimization_studio_gui() -> SelfOptimizationStudioGUI:
    """Get the global Self-Optimization Studio GUI instance"""
    global _studio_gui
    if _studio_gui is None:
        _studio_gui = SelfOptimizationStudioGUI()
    return _studio_gui

def start_studio_gui(host: str = "localhost", port: int = 8080, open_browser: bool = True):
    """Start the Self-Optimization Studio GUI"""
    global _studio_gui
    if _studio_gui is None:
        _studio_gui = SelfOptimizationStudioGUI(host, port)

    if open_browser:
        _studio_gui.open_browser()

    _studio_gui.start()