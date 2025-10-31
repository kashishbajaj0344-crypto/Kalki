"""
Observability Agent (Phase 8)
============================

Implements comprehensive observability with Prometheus metrics export,
distributed tracing, performance analytics, and real-time monitoring dashboards.
"""

import asyncio
import time
import json
import psutil
import threading
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import statistics

from modules.logging_config import get_logger
from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = get_logger("Kalki.Observability")


@dataclass
class MetricPoint:
    """A single metric measurement"""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class TraceSpan:
    """A distributed tracing span"""
    span_id: str
    trace_id: str
    parent_span_id: Optional[str]
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    logs: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PerformanceProfile:
    """Performance profiling data"""
    operation: str
    start_time: float
    end_time: float
    duration: float
    memory_usage: float
    cpu_usage: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObservabilityAgent(BaseAgent):
    """
    Comprehensive observability agent with metrics, tracing, and analytics.
    Provides Prometheus-compatible metrics export and performance monitoring.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="ObservabilityAgent",
            capabilities=[
                AgentCapability.OBSERVABILITY,
                AgentCapability.MONITORING,
                AgentCapability.ANALYTICS
            ],
            description="Comprehensive observability with metrics and tracing",
            config=config or {}
        )

        # Metrics configuration
        self.metrics_port = self.config.get('metrics_port', 9090)
        self.metrics_enabled = self.config.get('metrics_enabled', True)
        self.metrics_interval = self.config.get('metrics_interval', 15.0)  # seconds

        # Tracing configuration
        self.tracing_enabled = self.config.get('tracing_enabled', True)
        self.trace_sampling_rate = self.config.get('trace_sampling_rate', 0.1)  # 10%

        # Analytics configuration
        self.analytics_enabled = self.config.get('analytics_enabled', True)
        self.performance_profiling = self.config.get('performance_profiling', True)

        # Metrics storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.gauges: Dict[str, float] = {}
        self.counters: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)

        # Tracing storage
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_traces: deque = deque(maxlen=5000)
        self.trace_buffer: deque = deque(maxlen=1000)

        # Performance profiling
        self.performance_profiles: deque = deque(maxlen=10000)
        self.active_profiles: Dict[str, PerformanceProfile] = {}

        # System monitoring
        self.system_metrics_enabled = True
        self.custom_metrics: Dict[str, Callable] = {}

        # HTTP server for metrics endpoint
        self.http_server = None
        self.server_thread = None

    async def initialize(self) -> bool:
        """Initialize observability systems"""
        try:
            logger.info("ObservabilityAgent initializing comprehensive monitoring")

            # Start metrics collection
            if self.metrics_enabled:
                asyncio.create_task(self._metrics_collection_loop())

            # Start HTTP server for Prometheus metrics
            if self.metrics_enabled:
                self._start_metrics_server()

            # Start trace processing
            if self.tracing_enabled:
                asyncio.create_task(self._trace_processing_loop())

            # Start performance profiling
            if self.performance_profiling:
                asyncio.create_task(self._performance_monitoring_loop())

            logger.info("ObservabilityAgent initialized with full monitoring capabilities")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize ObservabilityAgent: {e}")
            return False

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute observability operations"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "record_metric":
            return await self._record_metric(params)
        elif action == "start_trace":
            return await self._start_trace(params)
        elif action == "end_trace":
            return await self._end_trace(params)
        elif action == "get_metrics":
            return await self._get_metrics(params)
        elif action == "get_traces":
            return await self._get_traces(params)
        elif action == "performance_report":
            return await self._performance_report(params)
        elif action == "register_custom_metric":
            return await self._register_custom_metric(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _record_metric(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Record a metric measurement"""
        try:
            name = params.get("name")
            value = params.get("value")
            metric_type = params.get("type", "gauge")  # gauge, counter, histogram
            labels = params.get("labels", {})

            if not name or value is None:
                return {"status": "error", "error": "name and value required"}

            timestamp = time.time()

            # Store metric based on type
            if metric_type == "gauge":
                self.gauges[name] = value
            elif metric_type == "counter":
                self.counters[name] += value
            elif metric_type == "histogram":
                self.histograms[name].append(value)

            # Store in time series
            metric_point = MetricPoint(
                name=name,
                value=value,
                timestamp=timestamp,
                labels=labels
            )
            self.metrics[name].append(metric_point)

            return {"status": "success", "metric_name": name, "timestamp": timestamp}

        except Exception as e:
            logger.exception(f"Failed to record metric: {e}")
            return {"status": "error", "error": str(e)}

    async def _start_trace(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start a trace span"""
        try:
            operation_name = params.get("operation")
            trace_id = params.get("trace_id", f"trace_{int(time.time() * 1000000)}")
            parent_span_id = params.get("parent_span_id")
            tags = params.get("tags", {})

            if not operation_name:
                return {"status": "error", "error": "operation name required"}

            # Check sampling rate
            if random.random() > self.trace_sampling_rate:
                return {"status": "success", "sampled": False, "span_id": None}

            span_id = f"span_{int(time.time() * 1000000)}_{random.randint(1000, 9999)}"

            span = TraceSpan(
                span_id=span_id,
                trace_id=trace_id,
                parent_span_id=parent_span_id,
                operation_name=operation_name,
                start_time=time.time(),
                tags=tags
            )

            self.active_spans[span_id] = span

            return {
                "status": "success",
                "sampled": True,
                "span_id": span_id,
                "trace_id": trace_id
            }

        except Exception as e:
            logger.exception(f"Failed to start trace: {e}")
            return {"status": "error", "error": str(e)}

    async def _end_trace(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """End a trace span"""
        try:
            span_id = params.get("span_id")
            tags = params.get("tags", {})

            if not span_id or span_id not in self.active_spans:
                return {"status": "error", "error": "Invalid or unknown span_id"}

            span = self.active_spans[span_id]
            span.end_time = time.time()
            span.duration = span.end_time - span.start_time

            # Add final tags
            span.tags.update(tags)

            # Move to completed traces
            self.completed_traces.append(span)
            del self.active_spans[span_id]

            return {
                "status": "success",
                "span_id": span_id,
                "duration": span.duration,
                "trace_id": span.trace_id
            }

        except Exception as e:
            logger.exception(f"Failed to end trace: {e}")
            return {"status": "error", "error": str(e)}

    async def _register_custom_metric(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Register a custom metric collection function"""
        try:
            name = params.get("name")
            collector_func = params.get("collector")

            if not name or not callable(collector_func):
                return {"status": "error", "error": "name and callable collector required"}

            self.custom_metrics[name] = collector_func

            return {"status": "success", "metric_name": name}

        except Exception as e:
            logger.exception(f"Failed to register custom metric: {e}")
            return {"status": "error", "error": str(e)}

    async def _metrics_collection_loop(self):
        """Continuous metrics collection"""
        while True:
            try:
                await self._collect_system_metrics()
                await self._collect_custom_metrics()
                await asyncio.sleep(self.metrics_interval)
            except Exception as e:
                logger.exception(f"Metrics collection error: {e}")
                await asyncio.sleep(self.metrics_interval)

    async def _collect_system_metrics(self):
        """Collect system-level metrics"""
        try:
            timestamp = time.time()

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric({
                "name": "system_cpu_percent",
                "value": cpu_percent,
                "type": "gauge"
            })

            # Memory metrics
            memory = psutil.virtual_memory()
            await self._record_metric({
                "name": "system_memory_percent",
                "value": memory.percent,
                "type": "gauge"
            })
            await self._record_metric({
                "name": "system_memory_used_bytes",
                "value": memory.used,
                "type": "gauge"
            })

            # Disk metrics
            disk = psutil.disk_usage('/')
            await self._record_metric({
                "name": "system_disk_percent",
                "value": disk.percent,
                "type": "gauge"
            })

            # Network metrics
            net = psutil.net_io_counters()
            if net:
                await self._record_metric({
                    "name": "system_network_bytes_sent",
                    "value": net.bytes_sent,
                    "type": "counter"
                })
                await self._record_metric({
                    "name": "system_network_bytes_recv",
                    "value": net.bytes_recv,
                    "type": "counter"
                })

            # Process metrics
            process_count = len(psutil.pids())
            await self._record_metric({
                "name": "system_process_count",
                "value": process_count,
                "type": "gauge"
            })

        except Exception as e:
            logger.exception(f"System metrics collection failed: {e}")

    async def _collect_custom_metrics(self):
        """Collect custom metrics"""
        try:
            for name, collector in self.custom_metrics.items():
                try:
                    value = await collector()
                    if value is not None:
                        await self._record_metric({
                            "name": name,
                            "value": value,
                            "type": "gauge"
                        })
                except Exception as e:
                    logger.warning(f"Custom metric collection failed for {name}: {e}")

        except Exception as e:
            logger.exception(f"Custom metrics collection failed: {e}")

    async def _trace_processing_loop(self):
        """Process and buffer traces"""
        while True:
            try:
                # Process completed traces for analysis
                await self._analyze_traces()
                await asyncio.sleep(10.0)  # Process every 10 seconds
            except Exception as e:
                logger.exception(f"Trace processing error: {e}")
                await asyncio.sleep(10.0)

    async def _analyze_traces(self):
        """Analyze completed traces for performance insights"""
        try:
            if not self.completed_traces:
                return

            # Analyze recent traces
            recent_traces = list(self.completed_traces)[-100:]  # Last 100 traces

            # Group by operation
            operation_stats = defaultdict(list)

            for trace in recent_traces:
                if trace.duration:
                    operation_stats[trace.operation_name].append(trace.duration)

            # Calculate statistics
            for operation, durations in operation_stats.items():
                if len(durations) >= 5:  # Need minimum samples
                    avg_duration = statistics.mean(durations)
                    p95_duration = statistics.quantiles(durations, n=20)[18]  # 95th percentile

                    await self._record_metric({
                        "name": f"trace_duration_avg_{operation}",
                        "value": avg_duration,
                        "type": "gauge",
                        "labels": {"operation": operation}
                    })

                    await self._record_metric({
                        "name": f"trace_duration_p95_{operation}",
                        "value": p95_duration,
                        "type": "gauge",
                        "labels": {"operation": operation}
                    })

        except Exception as e:
            logger.exception(f"Trace analysis failed: {e}")

    async def _performance_monitoring_loop(self):
        """Continuous performance monitoring"""
        while True:
            try:
                await self._collect_performance_metrics()
                await asyncio.sleep(30.0)  # Every 30 seconds
            except Exception as e:
                logger.exception(f"Performance monitoring error: {e}")
                await asyncio.sleep(30.0)

    async def _collect_performance_metrics(self):
        """Collect detailed performance metrics"""
        try:
            # Memory usage trend
            memory_percent = psutil.virtual_memory().percent
            await self._record_metric({
                "name": "performance_memory_trend",
                "value": memory_percent,
                "type": "gauge"
            })

            # CPU usage trend
            cpu_percent = psutil.cpu_percent(interval=1)
            await self._record_metric({
                "name": "performance_cpu_trend",
                "value": cpu_percent,
                "type": "gauge"
            })

            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                await self._record_metric({
                    "name": "performance_disk_read_bytes",
                    "value": disk_io.read_bytes,
                    "type": "counter"
                })
                await self._record_metric({
                    "name": "performance_disk_write_bytes",
                    "value": disk_io.write_bytes,
                    "type": "counter"
                })

        except Exception as e:
            logger.exception(f"Performance metrics collection failed: {e}")

    def _start_metrics_server(self):
        """Start HTTP server for Prometheus metrics"""
        try:
            import http.server
            import socketserver
            import urllib.parse

            class MetricsHandler(http.server.BaseHTTPRequestHandler):
                def __init__(self, *args, observability_agent=None, **kwargs):
                    self.observability_agent = observability_agent
                    super().__init__(*args, **kwargs)

                def do_GET(self):
                    if self.path == '/metrics':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/plain; charset=utf-8')
                        self.end_headers()

                        metrics_output = self.observability_agent._generate_prometheus_metrics()
                        self.wfile.write(metrics_output.encode('utf-8'))
                    else:
                        self.send_response(404)
                        self.end_headers()

                def log_message(self, format, *args):
                    # Suppress default HTTP server logs
                    pass

            def run_server():
                with socketserver.TCPServer(("", self.metrics_port), lambda *args, **kwargs: MetricsHandler(*args, observability_agent=self, **kwargs)) as httpd:
                    logger.info(f"Metrics server started on port {self.metrics_port}")
                    httpd.serve_forever()

            self.server_thread = threading.Thread(target=run_server, daemon=True)
            self.server_thread.start()

        except Exception as e:
            logger.exception(f"Failed to start metrics server: {e}")

    def _generate_prometheus_metrics(self) -> str:
        """Generate Prometheus-compatible metrics output"""
        try:
            lines = []

            # Gauges
            for name, value in self.gauges.items():
                lines.append(f"# HELP {name} {name}")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")

            # Counters
            for name, value in self.counters.items():
                lines.append(f"# HELP {name} {name}")
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name} {value}")

            # Histograms (simplified)
            for name, values in self.histograms.items():
                if values:
                    lines.append(f"# HELP {name} {name}")
                    lines.append(f"# TYPE {name} histogram")
                    count = len(values)
                    sum_val = sum(values)
                    lines.append(f"{name}_count {count}")
                    lines.append(f"{name}_sum {sum_val}")

            return "\n".join(lines) + "\n"

        except Exception as e:
            logger.exception(f"Failed to generate Prometheus metrics: {e}")
            return "# Error generating metrics\n"

    async def _get_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics data"""
        try:
            metric_name = params.get("name")
            time_range = params.get("time_range", 3600)  # 1 hour default

            if metric_name:
                # Specific metric
                metric_data = list(self.metrics.get(metric_name, []))
                # Filter by time range
                cutoff_time = time.time() - time_range
                metric_data = [m for m in metric_data if m.timestamp > cutoff_time]

                return {
                    "status": "success",
                    "metric": metric_name,
                    "data_points": len(metric_data),
                    "data": [
                        {
                            "timestamp": m.timestamp,
                            "value": m.value,
                            "labels": m.labels
                        } for m in metric_data[-100:]  # Last 100 points
                    ]
                }
            else:
                # All metrics summary
                summary = {}
                for name, data in self.metrics.items():
                    if data:
                        values = [m.value for m in data]
                        summary[name] = {
                            "count": len(values),
                            "latest": values[-1] if values else None,
                            "avg": statistics.mean(values) if values else None,
                            "min": min(values) if values else None,
                            "max": max(values) if values else None
                        }

                return {
                    "status": "success",
                    "metrics_count": len(summary),
                    "summary": summary
                }

        except Exception as e:
            logger.exception(f"Failed to get metrics: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_traces(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get trace data"""
        try:
            trace_id = params.get("trace_id")
            operation = params.get("operation")
            time_range = params.get("time_range", 3600)

            traces = list(self.completed_traces)
            cutoff_time = time.time() - time_range
            traces = [t for t in traces if t.start_time > cutoff_time]

            if trace_id:
                traces = [t for t in traces if t.trace_id == trace_id]
            if operation:
                traces = [t for t in traces if t.operation_name == operation]

            return {
                "status": "success",
                "trace_count": len(traces),
                "traces": [
                    {
                        "span_id": t.span_id,
                        "trace_id": t.trace_id,
                        "operation": t.operation_name,
                        "start_time": t.start_time,
                        "duration": t.duration,
                        "tags": t.tags
                    } for t in traces[-50:]  # Last 50 traces
                ]
            }

        except Exception as e:
            logger.exception(f"Failed to get traces: {e}")
            return {"status": "error", "error": str(e)}

    async def _performance_report(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance report"""
        try:
            time_range = params.get("time_range", 3600)

            # Collect performance data
            report = {
                "time_range_seconds": time_range,
                "generated_at": time.time(),
                "system_performance": {},
                "operation_performance": {},
                "anomalies": []
            }

            # System performance summary
            cutoff_time = time.time() - time_range
            cpu_metrics = [m for m in self.metrics.get("system_cpu_percent", []) if m.timestamp > cutoff_time]
            memory_metrics = [m for m in self.metrics.get("system_memory_percent", []) if m.timestamp > cutoff_time]

            if cpu_metrics:
                cpu_values = [m.value for m in cpu_metrics]
                report["system_performance"]["cpu"] = {
                    "avg": statistics.mean(cpu_values),
                    "max": max(cpu_values),
                    "min": min(cpu_values),
                    "p95": statistics.quantiles(cpu_values, n=20)[18] if len(cpu_values) >= 20 else max(cpu_values)
                }

            if memory_metrics:
                memory_values = [m.value for m in memory_metrics]
                report["system_performance"]["memory"] = {
                    "avg": statistics.mean(memory_values),
                    "max": max(memory_values),
                    "min": min(memory_values),
                    "p95": statistics.quantiles(memory_values, n=20)[18] if len(memory_values) >= 20 else max(memory_values)
                }

            # Operation performance from traces
            operation_durations = defaultdict(list)
            for trace in self.completed_traces:
                if trace.start_time > cutoff_time and trace.duration:
                    operation_durations[trace.operation_name].append(trace.duration)

            for operation, durations in operation_durations.items():
                if len(durations) >= 5:
                    report["operation_performance"][operation] = {
                        "count": len(durations),
                        "avg_duration": statistics.mean(durations),
                        "p95_duration": statistics.quantiles(durations, n=20)[18],
                        "max_duration": max(durations),
                        "min_duration": min(durations)
                    }

            return {
                "status": "success",
                "report": report
            }

        except Exception as e:
            logger.exception(f"Failed to generate performance report: {e}")
            return {"status": "error", "error": str(e)}

    async def shutdown(self) -> bool:
        """Shutdown the observability agent"""
        try:
            logger.info("ObservabilityAgent shutting down")

            # Stop HTTP server
            if self.http_server:
                self.http_server.shutdown()

            # Clear all data
            self.metrics.clear()
            self.gauges.clear()
            self.counters.clear()
            self.histograms.clear()
            self.active_spans.clear()
            self.completed_traces.clear()
            self.performance_profiles.clear()
            self.custom_metrics.clear()

            logger.info("ObservabilityAgent shutdown complete")
            return True

        except Exception as e:
            logger.exception(f"Error during ObservabilityAgent shutdown: {e}")
            return False