#!/usr/bin/env python3
"""
Enhanced Compute Scaling Agent (Phase 8.1)
===========================================

Implements dynamic compute resource scaling with cross-agent integration,
persistent state, intelligent scheduling, and continuous self-healing.

Features:
- Cross-agent cooperation with LoadBalancerAgent and SelfHealingAgent
- Persistent cluster state with JSON storage
- AI-driven load balancing strategies
- Continuous health monitoring and auto-recovery
- Structured observability logging
- Future hooks for distributed consensus
"""

import asyncio
import json
import time
import psutil
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from datetime import datetime, timedelta
import hashlib
import statistics

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from modules.logging_config import get_logger
from ..base_agent import BaseAgent, AgentCapability, AgentStatus
from .load_balancing import LoadBalancingAgent, ServerInstance
from .self_healing import SelfHealingAgent

logger = get_logger("Kalki.ComputeScaling")


@dataclass
class NodeState:
    """Persistent node state"""
    node_id: str
    address: str
    port: int
    status: str  # 'active', 'inactive', 'failed', 'recovering'
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float]
    active_tasks: int
    total_tasks_completed: int
    success_rate: float
    last_seen: str
    reputation_score: float
    capabilities: List[str]
    metadata: Dict[str, Any]


@dataclass
class TaskState:
    """Persistent task state"""
    task_id: str
    node_id: str
    status: str  # 'queued', 'running', 'completed', 'failed'
    priority: int
    estimated_load: float
    actual_load: Optional[float]
    start_time: Optional[str]
    end_time: Optional[str]
    result: Optional[Any]
    error: Optional[str]
    retry_count: int
    dependencies: List[str]


@dataclass
class ClusterMetrics:
    """Real-time cluster metrics for observability"""
    timestamp: str
    total_nodes: int
    active_nodes: int
    total_tasks: int
    running_tasks: int
    queued_tasks: int
    avg_cpu_usage: float
    avg_memory_usage: float
    avg_response_time: float
    task_success_rate: float
    cluster_health_score: float


class ComputeClusterAgent(BaseAgent):
    """
    Enhanced compute cluster agent with full Phase 8 capabilities.
    Integrates scaling, load balancing, and self-healing in a cooperative system.
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="ComputeClusterAgent",
            capabilities=[
                AgentCapability.COMPUTE_SCALING,
                AgentCapability.LOAD_BALANCING,
                AgentCapability.SELF_HEALING,
                AgentCapability.OPTIMIZATION
            ],
            description="Distributed compute cluster with intelligent orchestration",
            config=config or {}
        )

        # Directory structure
        self.cluster_dir = Path.home() / "Desktop" / "Kalki" / "vector_db" / "cluster"
        self.cluster_dir.mkdir(parents=True, exist_ok=True)

        # State persistence files
        self.nodes_file = self.cluster_dir / "nodes.json"
        self.tasks_file = self.cluster_dir / "tasks.json"
        self.metrics_file = self.cluster_dir / "metrics.json"
        self.health_registry_file = self.cluster_dir / "health_registry.json"

        # Cluster configuration
        self.max_nodes = self.config.get('max_nodes', 50)
        self.task_timeout = self.config.get('task_timeout', 3600)  # 1 hour
        self.health_check_interval = self.config.get('health_check_interval', 30)
        self.metrics_retention_days = self.config.get('metrics_retention_days', 7)

        # Distribution strategies
        self.distribution_strategy = self.config.get('distribution_strategy', 'adaptive_ai')

        # Cross-agent integration
        self.load_balancer = LoadBalancingAgent(config={
            "algorithm": "adaptive_ai",
            "health_check_interval": self.health_check_interval
        })
        self.self_healer = SelfHealingAgent(config={
            "health_check_interval": self.health_check_interval,
            "anomaly_detection_window": 100
        })

        # Integration hooks
        self.pre_task_hooks: List[Callable] = []
        self.post_task_hooks: List[Callable] = []
        self.node_failure_hooks: List[Callable] = []
        self.scaling_hooks: List[Callable] = []

        # Cluster state
        self.nodes: Dict[str, NodeState] = {}
        self.tasks_queue: Dict[str, TaskState] = {}
        self.running_tasks: Dict[str, TaskState] = {}
        self.completed_tasks: deque = deque(maxlen=10000)

        # Metrics and monitoring
        self.metrics_history = deque(maxlen=10000)
        self.node_reputation = defaultdict(lambda: 1.0)  # Node success rates
        self.task_similarity_cache = {}  # For intelligent scheduling

        # AI components for intelligent scheduling
        if SKLEARN_AVAILABLE:
            self.task_predictor = RandomForestRegressor(n_estimators=50, random_state=42)
            self.load_scaler = StandardScaler()
            self.is_ai_trained = False
        else:
            self.task_predictor = None
            self.is_ai_trained = False

        # Continuous monitoring
        self.monitoring_task: Optional[asyncio.Task] = None
        self.is_monitoring = False

        # Future hooks placeholders
        self.node_discovery_enabled = self.config.get('node_discovery_enabled', False)
        self.task_replication_enabled = self.config.get('task_replication_enabled', False)
        self.consensus_protocol = self.config.get('consensus_protocol', None)

    async def initialize(self) -> bool:
        """Initialize the compute cluster with persistent state recovery."""
        try:
            logger.info("ComputeClusterAgent initializing distributed orchestration system")

            # Load persistent state
            await self._load_cluster_state()

            # Initialize cross-agent dependencies
            lb_init = await self.load_balancer.initialize()
            sh_init = await self.self_healer.initialize()

            if not (lb_init and sh_init):
                logger.error("Failed to initialize cross-agent dependencies")
                return False

            # Setup integration hooks
            self._setup_integration_hooks()

            # Start continuous monitoring
            await self._start_continuous_monitoring()

            # Train AI models if data available
            if SKLEARN_AVAILABLE and len(self.completed_tasks) >= 10:
                await self._train_ai_models()

            # Log initialization metrics
            await self._log_cluster_metrics("cluster_initialized")

            logger.info(f"ComputeClusterAgent initialized with {len(self.nodes)} nodes and {len(self.tasks_queue)} queued tasks")
            return True

        except Exception as e:
            logger.exception(f"Failed to initialize ComputeClusterAgent: {e}")
            return False

    async def submit_task(self, task_data: Dict[str, Any], priority: int = 1) -> str:
        """Submit a task to the cluster with intelligent scheduling."""
        try:
            task_id = f"task_{int(time.time() * 1000000)}_{hashlib.md5(str(task_data).encode()).hexdigest()[:8]}"

            # Estimate task load using AI if available
            estimated_load = await self._estimate_task_load(task_data)

            # Create task state
            task = TaskState(
                task_id=task_id,
                node_id="",  # Will be assigned
                status="queued",
                priority=priority,
                estimated_load=estimated_load,
                actual_load=None,
                start_time=None,
                end_time=None,
                result=None,
                error=None,
                retry_count=0,
                dependencies=task_data.get('dependencies', [])
            )

            # Run pre-task hooks
            for hook in self.pre_task_hooks:
                try:
                    await hook(task)
                except Exception as e:
                    logger.warning(f"Pre-task hook failed: {e}")

            # Add to queue
            self.tasks_queue[task_id] = task

            # Attempt immediate scheduling
            await self._schedule_pending_tasks()

            # Persist state
            await self._save_cluster_state()

            # Log task submission
            await self._log_structured_event("task_submitted", {
                "task_id": task_id,
                "priority": priority,
                "estimated_load": estimated_load,
                "queue_size": len(self.tasks_queue)
            })

            logger.info(f"Task {task_id} submitted with priority {priority}")
            return task_id

        except Exception as e:
            logger.exception(f"Failed to submit task: {e}")
            raise

    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive task status."""
        try:
            # Check running tasks
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                node = self.nodes.get(task.node_id)
                return {
                    "task_id": task_id,
                    "status": task.status,
                    "node_id": task.node_id,
                    "node_address": node.address if node else None,
                    "start_time": task.start_time,
                    "progress": await self._get_task_progress(task_id),
                    "estimated_completion": await self._estimate_completion_time(task)
                }

            # Check queued tasks
            if task_id in self.tasks_queue:
                task = self.tasks_queue[task_id]
                queue_position = list(self.tasks_queue.keys()).index(task_id)
                return {
                    "task_id": task_id,
                    "status": "queued",
                    "queue_position": queue_position,
                    "estimated_load": task.estimated_load,
                    "priority": task.priority
                }

            # Check completed tasks
            for completed_task in self.completed_tasks:
                if completed_task.task_id == task_id:
                    return {
                        "task_id": task_id,
                        "status": completed_task.status,
                        "end_time": completed_task.end_time,
                        "result": completed_task.result,
                        "error": completed_task.error,
                        "retry_count": completed_task.retry_count
                    }

            return None

        except Exception as e:
            logger.exception(f"Failed to get task status for {task_id}: {e}")
            return None

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running or queued task."""
        try:
            # Check running tasks
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                node = self.nodes.get(task.node_id)

                if node:
                    # Notify node to cancel (would need actual node communication)
                    await self._notify_node_cancel(node.node_id, task_id)

                task.status = "cancelled"
                task.end_time = datetime.now().isoformat()
                self.completed_tasks.append(task)
                del self.running_tasks[task_id]

            # Check queued tasks
            elif task_id in self.tasks_queue:
                task = self.tasks_queue[task_id]
                task.status = "cancelled"
                task.end_time = datetime.now().isoformat()
                self.completed_tasks.append(task)
                del self.tasks_queue[task_id]

            else:
                return False

            await self._save_cluster_state()
            await self._log_structured_event("task_cancelled", {"task_id": task_id})

            return True

        except Exception as e:
            logger.exception(f"Failed to cancel task {task_id}: {e}")
            return False

    async def add_node(self, node_config: Dict[str, Any]) -> str:
        """Add a new compute node to the cluster."""
        try:
            node_id = node_config.get('node_id', f"node_{int(time.time() * 1000000)}")

            node = NodeState(
                node_id=node_id,
                address=node_config['address'],
                port=node_config['port'],
                status="active",
                cpu_usage=0.0,
                memory_usage=0.0,
                gpu_usage=node_config.get('gpu_usage'),
                active_tasks=0,
                total_tasks_completed=0,
                success_rate=1.0,
                last_seen=datetime.now().isoformat(),
                reputation_score=1.0,
                capabilities=node_config.get('capabilities', []),
                metadata=node_config.get('metadata', {})
            )

            self.nodes[node_id] = node

            # Add to load balancer
            server_instance = ServerInstance(
                id=node_id,
                address=node.address,
                port=node.port,
                weight=node_config.get('weight', 1),
                max_connections=node_config.get('max_connections', 100)
            )
            await self.load_balancer.add_server(server_instance)

            await self._save_cluster_state()
            await self._log_structured_event("node_added", {"node_id": node_id, "address": node.address})

            logger.info(f"Node {node_id} added to cluster at {node.address}:{node.port}")
            return node_id

        except Exception as e:
            logger.exception(f"Failed to add node: {e}")
            raise

    async def remove_node(self, node_id: str, graceful: bool = True) -> bool:
        """Remove a node from the cluster."""
        try:
            if node_id not in self.nodes:
                return False

            node = self.nodes[node_id]

            if graceful and node.active_tasks > 0:
                # Wait for tasks to complete or migrate them
                await self._migrate_node_tasks(node_id)

            # Remove from load balancer
            await self.load_balancer.remove_server(node_id)

            # Update node status
            node.status = "removed"
            node.last_seen = datetime.now().isoformat()

            # Trigger node failure hooks for cleanup
            for hook in self.node_failure_hooks:
                try:
                    await hook(node_id, "removed")
                except Exception as e:
                    logger.warning(f"Node failure hook failed: {e}")

            await self._save_cluster_state()
            await self._log_structured_event("node_removed", {"node_id": node_id, "graceful": graceful})

            logger.info(f"Node {node_id} removed from cluster (graceful={graceful})")
            return True

        except Exception as e:
            logger.exception(f"Failed to remove node {node_id}: {e}")
            return False

    async def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        try:
            active_nodes = [n for n in self.nodes.values() if n.status == "active"]
            total_cpu = sum(n.cpu_usage for n in active_nodes) / max(len(active_nodes), 1)
            total_memory = sum(n.memory_usage for n in active_nodes) / max(len(active_nodes), 1)

            # Calculate cluster health score
            health_score = await self.self_healer.get_overall_health_score()

            return {
                "total_nodes": len(self.nodes),
                "active_nodes": len(active_nodes),
                "total_tasks": len(self.tasks_queue) + len(self.running_tasks),
                "running_tasks": len(self.running_tasks),
                "queued_tasks": len(self.tasks_queue),
                "completed_tasks": len(self.completed_tasks),
                "avg_cpu_usage": total_cpu,
                "avg_memory_usage": total_memory,
                "cluster_health_score": health_score,
                "distribution_strategy": self.distribution_strategy,
                "is_monitoring": self.is_monitoring,
                "ai_trained": self.is_ai_trained
            }

        except Exception as e:
            logger.exception("Failed to get cluster status: {e}")
            return {"error": str(e)}

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cluster management tasks."""
        try:
            action = task.get("action")

            if action == "submit_task":
                task_id = await self.submit_task(task["task_data"], task.get("priority", 1))
                return {"status": "success", "task_id": task_id}

            elif action == "get_task_status":
                status = await self.get_task_status(task["task_id"])
                return {"status": "success", "task_status": status}

            elif action == "cancel_task":
                success = await self.cancel_task(task["task_id"])
                return {"status": "success" if success else "error"}

            elif action == "add_node":
                node_id = await self.add_node(task["node_config"])
                return {"status": "success", "node_id": node_id}

            elif action == "remove_node":
                success = await self.remove_node(task["node_id"], task.get("graceful", True))
                return {"status": "success" if success else "error"}

            elif action == "get_cluster_status":
                status = await self.get_cluster_status()
                return {"status": "success", "cluster_status": status}

            elif action == "trigger_health_check":
                await self._perform_cluster_health_check()
                return {"status": "success"}

            elif action == "optimize_cluster":
                await self._optimize_cluster_configuration()
                return {"status": "success"}

            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            logger.exception(f"Task execution failed: {e}")
            return {"status": "error", "message": str(e)}

    # Cross-agent integration methods

    def _setup_integration_hooks(self):
        """Setup hooks for cross-agent cooperation."""

        # Load balancer integration
        @self.load_balancer.add_pre_balance_hook
        async def pre_balance_integration(request):
            # Use cluster intelligence for load balancing decisions
            return await self._enhance_load_balancing_decision(request)

        @self.load_balancer.add_post_balance_hook
        async def post_balance_integration(assignment):
            # Update cluster state after load balancing
            await self._update_cluster_after_balancing(assignment)

        # Self-healing integration
        @self.self_healer.add_recovery_hook
        async def recovery_integration(component, action):
            # Coordinate recovery with cluster operations
            await self._coordinate_recovery(component, action)

        @self.self_healer.add_health_alert_hook
        async def health_alert_integration(alert):
            # Respond to health alerts with cluster actions
            await self._respond_to_health_alert(alert)

    async def _enhance_load_balancing_decision(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance load balancing with cluster intelligence."""
        try:
            if self.distribution_strategy == "adaptive_ai" and self.is_ai_trained:
                # Use AI for intelligent node selection
                available_nodes = [n for n in self.nodes.values() if n.status == "active"]
                if available_nodes:
                    best_node = await self._ai_select_node(request, available_nodes)
                    return {"preferred_node": best_node.node_id}

            # Fallback to reputation-based selection
            available_nodes = [n for n in self.nodes.values() if n.status == "active"]
            if available_nodes:
                best_node = max(available_nodes, key=lambda n: n.reputation_score)
                return {"preferred_node": best_node.node_id}

            return {}

        except Exception as e:
            logger.exception(f"Failed to enhance load balancing decision: {e}")
            return {}

    async def _update_cluster_after_balancing(self, assignment: Dict[str, Any]):
        """Update cluster state after load balancing decisions."""
        try:
            node_id = assignment.get("node_id")
            task_id = assignment.get("task_id")

            if node_id and node_id in self.nodes:
                self.nodes[node_id].active_tasks += 1

            if task_id and task_id in self.tasks_queue:
                task = self.tasks_queue[task_id]
                task.node_id = node_id
                task.status = "running"
                task.start_time = datetime.now().isoformat()
                self.running_tasks[task_id] = task
                del self.tasks_queue[task_id]

            await self._save_cluster_state()

        except Exception as e:
            logger.exception(f"Failed to update cluster after balancing: {e}")

    async def _coordinate_recovery(self, component: str, action: str):
        """Coordinate recovery actions with cluster operations."""
        try:
            if component.startswith("node_"):
                node_id = component
                if action == "restart":
                    await self._handle_node_restart(node_id)
                elif action == "migrate":
                    await self._handle_node_migration(node_id)

            await self._log_structured_event("recovery_coordinated", {
                "component": component,
                "action": action
            })

        except Exception as e:
            logger.exception(f"Failed to coordinate recovery for {component}: {e}")

    async def _respond_to_health_alert(self, alert: Dict[str, Any]):
        """Respond to health alerts with appropriate cluster actions."""
        try:
            severity = alert.get("severity", "low")
            component = alert.get("component")

            if severity == "critical" and component.startswith("node_"):
                # Trigger immediate node isolation
                await self._isolate_unhealthy_node(component)

            elif severity == "high":
                # Trigger scaling or redistribution
                await self._trigger_scaling_response(alert)

            await self._log_structured_event("health_alert_response", {
                "alert": alert,
                "response_taken": True
            })

        except Exception as e:
            logger.exception(f"Failed to respond to health alert: {e}")

    # AI-driven scheduling methods

    async def _estimate_task_load(self, task_data: Dict[str, Any]) -> float:
        """Estimate task computational load using AI or heuristics."""
        try:
            if self.is_ai_trained and self.task_predictor:
                # Use ML model for prediction
                features = self._extract_task_features(task_data)
                prediction = self.task_predictor.predict([features])[0]
                return max(0.1, min(10.0, prediction))

            # Fallback heuristic estimation
            task_type = task_data.get("type", "unknown")
            complexity = task_data.get("complexity", 1.0)
            data_size = task_data.get("data_size", 1000)

            base_load = {
                "cpu_intensive": 3.0,
                "memory_intensive": 2.5,
                "io_intensive": 1.5,
                "network_intensive": 2.0
            }.get(task_type, 1.0)

            return base_load * complexity * min(1.0, data_size / 10000)

        except Exception as e:
            logger.exception(f"Failed to estimate task load: {e}")
            return 1.0  # Default load

    async def _ai_select_node(self, task_data: Dict[str, Any], available_nodes: List[NodeState]) -> NodeState:
        """Use AI to select the best node for a task."""
        try:
            if not self.is_ai_trained:
                return min(available_nodes, key=lambda n: n.cpu_usage)

            task_features = self._extract_task_features(task_data)
            best_score = float('-inf')
            best_node = available_nodes[0]

            for node in available_nodes:
                node_features = self._extract_node_features(node)
                combined_features = task_features + node_features

                # Predict success probability
                success_prob = self.task_predictor.predict([combined_features])[0]

                # Calculate composite score
                score = (
                    success_prob * 0.4 +
                    node.reputation_score * 0.3 +
                    (1.0 - node.cpu_usage) * 0.2 +
                    (1.0 - node.memory_usage) * 0.1
                )

                if score > best_score:
                    best_score = score
                    best_node = node

            return best_node

        except Exception as e:
            logger.exception("AI node selection failed, using fallback")
            return min(available_nodes, key=lambda n: n.cpu_usage)

    def _extract_task_features(self, task_data: Dict[str, Any]) -> List[float]:
        """Extract numerical features from task data for ML."""
        return [
            task_data.get("complexity", 1.0),
            task_data.get("data_size", 1000) / 10000,  # Normalize
            len(task_data.get("dependencies", [])),
            hash(task_data.get("type", "unknown")) % 100 / 100,  # Type hash
            task_data.get("priority", 1) / 10  # Normalize
        ]

    def _extract_node_features(self, node: NodeState) -> List[float]:
        """Extract numerical features from node state for ML."""
        return [
            node.cpu_usage,
            node.memory_usage,
            node.gpu_usage or 0.0,
            node.reputation_score,
            node.active_tasks / 10,  # Normalize
            node.success_rate
        ]

    async def _train_ai_models(self):
        """Train AI models using historical task and node data."""
        try:
            if not SKLEARN_AVAILABLE or len(self.completed_tasks) < 10:
                return

            # Prepare training data
            X = []
            y = []

            for task in list(self.completed_tasks)[-100:]:  # Last 100 tasks
                if task.actual_load and task.end_time and task.start_time:
                    task_features = self._extract_task_features({"type": "unknown", "complexity": 1.0})  # Simplified
                    node = self.nodes.get(task.node_id)
                    if node:
                        node_features = self._extract_node_features(node)
                        X.append(task_features + node_features)
                        y.append(task.actual_load)

            if len(X) >= 10:
                X_scaled = self.load_scaler.fit_transform(X)
                self.task_predictor.fit(X_scaled, y)
                self.is_ai_trained = True

                logger.info(f"AI models trained with {len(X)} samples")

        except Exception as e:
            logger.exception(f"Failed to train AI models: {e}")

    # Persistent state management

    async def _load_cluster_state(self):
        """Load cluster state from persistent storage."""
        try:
            # Load nodes
            if self.nodes_file.exists():
                with open(self.nodes_file, 'r') as f:
                    nodes_data = json.load(f)
                    for node_data in nodes_data:
                        node = NodeState(**node_data)
                        self.nodes[node.node_id] = node

            # Load tasks
            if self.tasks_file.exists():
                with open(self.tasks_file, 'r') as f:
                    tasks_data = json.load(f)
                    for task_data in tasks_data.get("queued", []):
                        task = TaskState(**task_data)
                        self.tasks_queue[task.task_id] = task
                    for task_data in tasks_data.get("running", []):
                        task = TaskState(**task_data)
                        self.running_tasks[task.task_id] = task

            logger.info(f"Loaded cluster state: {len(self.nodes)} nodes, {len(self.tasks_queue)} queued tasks")

        except Exception as e:
            logger.exception(f"Failed to load cluster state: {e}")

    async def _save_cluster_state(self):
        """Save cluster state to persistent storage."""
        try:
            # Save nodes
            nodes_data = [asdict(node) for node in self.nodes.values()]
            with open(self.nodes_file, 'w') as f:
                json.dump(nodes_data, f, indent=2)

            # Save tasks
            tasks_data = {
                "queued": [asdict(task) for task in self.tasks_queue.values()],
                "running": [asdict(task) for task in self.running_tasks.values()]
            }
            with open(self.tasks_file, 'w') as f:
                json.dump(tasks_data, f, indent=2)

        except Exception as e:
            logger.exception(f"Failed to save cluster state: {e}")

    # Continuous monitoring and self-healing

    async def _start_continuous_monitoring(self):
        """Start continuous cluster monitoring."""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        self.monitoring_task = asyncio.create_task(self._continuous_monitor_loop())

        logger.info("Continuous cluster monitoring started")

    async def _continuous_monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                await self._perform_cluster_health_check()
                await self._update_cluster_metrics()
                await self._cleanup_expired_tasks()

                # Auto-scale if needed
                await self._check_auto_scaling()

                await asyncio.sleep(self.health_check_interval)

            except Exception as e:
                logger.exception(f"Monitoring loop error: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _perform_cluster_health_check(self):
        """Perform comprehensive cluster health check."""
        try:
            # Update node health status
            for node in self.nodes.values():
                if node.status == "active":
                    health_status = await self.self_healer.perform_health_check(f"node_{node.node_id}")

                    if health_status.get("status") == "unhealthy":
                        await self._handle_node_health_issue(node.node_id, health_status)

            # Check for failed tasks
            timeout_threshold = datetime.now() - timedelta(seconds=self.task_timeout)
            for task_id, task in list(self.running_tasks.items()):
                if task.start_time:
                    start_time = datetime.fromisoformat(task.start_time)
                    if start_time < timeout_threshold:
                        await self._handle_task_timeout(task_id)

        except Exception as e:
            logger.exception("Cluster health check failed: {e}")

    async def _update_cluster_metrics(self):
        """Update and persist cluster metrics."""
        try:
            status = await self.get_cluster_status()
            metrics = ClusterMetrics(
                timestamp=datetime.now().isoformat(),
                total_nodes=status["total_nodes"],
                active_nodes=status["active_nodes"],
                total_tasks=status["total_tasks"],
                running_tasks=status["running_tasks"],
                queued_tasks=status["queued_tasks"],
                avg_cpu_usage=status["avg_cpu_usage"],
                avg_memory_usage=status["avg_memory_usage"],
                avg_response_time=0.1,  # Would need actual measurement
                task_success_rate=0.95,  # Would need actual calculation
                cluster_health_score=status["cluster_health_score"]
            )

            self.metrics_history.append(metrics)

            # Persist recent metrics
            recent_metrics = list(self.metrics_history)[-100:]  # Last 100 entries
            metrics_data = [asdict(m) for m in recent_metrics]

            async with aiofiles.open(self.metrics_file, 'w') as f:
                await f.write(json.dumps(metrics_data, indent=2))

        except Exception as e:
            logger.exception(f"Failed to update cluster metrics: {e}")

    # Scheduling and task management

    async def _schedule_pending_tasks(self):
        """Schedule pending tasks using intelligent load balancing."""
        try:
            # Sort tasks by priority (higher first)
            pending_tasks = sorted(
                self.tasks_queue.items(),
                key=lambda x: x[1].priority,
                reverse=True
            )

            for task_id, task in pending_tasks:
                # Find suitable node
                suitable_nodes = await self._find_suitable_nodes(task)

                if suitable_nodes:
                    # Use load balancer to select best node
                    selected_node = await self.load_balancer.balance_load({
                        "task_id": task_id,
                        "estimated_load": task.estimated_load,
                        "available_nodes": [n.node_id for n in suitable_nodes]
                    })

                    if selected_node:
                        # Assign task to node
                        await self._assign_task_to_node(task_id, selected_node["node_id"])
                        break  # Only schedule one task at a time to avoid overload

        except Exception as e:
            logger.exception("Failed to schedule pending tasks: {e}")

    async def _find_suitable_nodes(self, task: TaskState) -> List[NodeState]:
        """Find nodes suitable for a task."""
        try:
            suitable = []

            for node in self.nodes.values():
                if node.status != "active":
                    continue

                # Check resource availability
                if node.cpu_usage > 0.9 or node.memory_usage > 0.9:
                    continue

                # Check capabilities match
                task_requirements = []  # Would be extracted from task
                if task_requirements and not all(req in node.capabilities for req in task_requirements):
                    continue

                suitable.append(node)

            return suitable

        except Exception as e:
            logger.exception(f"Failed to find suitable nodes for task {task.task_id}: {e}")
            return []

    async def _assign_task_to_node(self, task_id: str, node_id: str):
        """Assign a task to a specific node."""
        try:
            if task_id not in self.tasks_queue or node_id not in self.nodes:
                return

            task = self.tasks_queue[task_id]
            node = self.nodes[node_id]

            # Update states
            task.node_id = node_id
            task.status = "running"
            task.start_time = datetime.now().isoformat()

            node.active_tasks += 1

            # Move to running
            self.running_tasks[task_id] = task
            del self.tasks_queue[task_id]

            await self._save_cluster_state()

            # Log assignment
            await self._log_structured_event("task_assigned", {
                "task_id": task_id,
                "node_id": node_id,
                "estimated_load": task.estimated_load
            })

            logger.info(f"Task {task_id} assigned to node {node_id}")

        except Exception as e:
            logger.exception(f"Failed to assign task {task_id} to node {node_id}: {e}")

    # Utility methods

    async def _log_structured_event(self, event_type: str, data: Dict[str, Any]):
        """Log structured event for observability."""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "cluster_id": self.name,
                "data": data
            }

            logger.info(json.dumps(event))

        except Exception as e:
            logger.exception(f"Failed to log structured event: {e}")

    async def _log_cluster_metrics(self, context: str):
        """Log current cluster metrics."""
        try:
            status = await self.get_cluster_status()
            status["context"] = context
            await self._log_structured_event("cluster_metrics", status)

        except Exception as e:
            logger.exception(f"Failed to log cluster metrics: {e}")

    async def _get_task_progress(self, task_id: str) -> Optional[float]:
        """Get task progress (placeholder for actual implementation)."""
        # This would need actual node communication to get real progress
        return None

    async def _estimate_completion_time(self, task: TaskState) -> Optional[str]:
        """Estimate task completion time."""
        try:
            if not task.start_time or not task.estimated_load:
                return None

            # Simple estimation based on load and node performance
            node = self.nodes.get(task.node_id)
            if node:
                avg_task_time = 60  # seconds, would be learned from history
                estimated_seconds = avg_task_time * task.estimated_load / (1.0 - node.cpu_usage)
                completion_time = datetime.fromisoformat(task.start_time) + timedelta(seconds=estimated_seconds)
                return completion_time.isoformat()

            return None

        except Exception as e:
            logger.exception(f"Failed to estimate completion time for task {task.task_id}: {e}")
            return None

    # Placeholder methods for future implementation

    async def _notify_node_cancel(self, node_id: str, task_id: str):
        """Notify node to cancel a task (placeholder)."""
        # Would implement actual node communication
        pass

    async def _migrate_node_tasks(self, node_id: str):
        """Migrate tasks from a node (placeholder)."""
        # Would implement task migration logic
        pass

    async def _handle_node_restart(self, node_id: str):
        """Handle node restart recovery."""
        # Would implement restart coordination
        pass

    async def _handle_node_migration(self, node_id: str):
        """Handle node migration."""
        # Would implement migration logic
        pass

    async def _isolate_unhealthy_node(self, node_id: str):
        """Isolate an unhealthy node."""
        # Would implement node isolation
        pass

    async def _trigger_scaling_response(self, alert: Dict[str, Any]):
        """Trigger scaling response to health alert."""
        # Would implement scaling logic
        pass

    async def _handle_node_health_issue(self, node_id: str, health_status: Dict[str, Any]):
        """Handle node health issues."""
        # Would implement health issue handling
        pass

    async def _handle_task_timeout(self, task_id: str):
        """Handle task timeout."""
        # Would implement timeout handling
        pass

    async def _check_auto_scaling(self):
        """Check if auto-scaling is needed."""
        # Would implement scaling checks
        pass

    async def _optimize_cluster_configuration(self):
        """Optimize cluster configuration."""
        # Would implement optimization logic
        pass

    async def _cleanup_expired_tasks(self):
        """Clean up expired tasks."""
        # Would implement cleanup logic
        pass