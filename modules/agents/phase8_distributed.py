#!/usr/bin/env python3
"""
Phase 8: Distributed Sentience & Compute Scaling
- ComputeClusterAgent: Distributed processing coordination
- LoadBalancerAgent: Workload distribution
- SelfHealingAgent: Fault tolerance and recovery
"""
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from .base_agent import BaseAgent
from ..utils import now_ts

logger = logging.getLogger("kalki.agents.phase8")


class ComputeClusterAgent(BaseAgent):
    """
    Manages distributed compute cluster for multi-device/grid compute
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="ComputeClusterAgent", config=config)
        self.nodes = {}
        self.tasks_queue = []
    
    def register_node(self, node_id: str, capabilities: Dict[str, Any]) -> Dict[str, Any]:
        """Register a compute node in the cluster"""
        try:
            node = {
                "node_id": node_id,
                "capabilities": capabilities,
                "status": "available",
                "registered_at": now_ts(),
                "tasks_completed": 0
            }
            
            self.nodes[node_id] = node
            self.logger.info(f"Registered compute node {node_id}")
            return node
        except Exception as e:
            self.logger.exception(f"Failed to register node: {e}")
            raise
    
    def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit a task to the cluster"""
        try:
            task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            task["task_id"] = task_id
            task["status"] = "queued"
            task["submitted_at"] = now_ts()
            
            self.tasks_queue.append(task)
            self.logger.info(f"Submitted task {task_id}")
            return task_id
        except Exception as e:
            self.logger.exception(f"Failed to submit task: {e}")
            raise
    
    def assign_task_to_node(self, task_id: str, node_id: str):
        """Assign a task to a specific node"""
        try:
            task = next((t for t in self.tasks_queue if t["task_id"] == task_id), None)
            if not task:
                raise ValueError(f"Task {task_id} not found")
            
            if node_id not in self.nodes:
                raise ValueError(f"Node {node_id} not found")
            
            task["status"] = "assigned"
            task["assigned_node"] = node_id
            task["assigned_at"] = now_ts()
            
            self.nodes[node_id]["status"] = "busy"
            self.logger.info(f"Assigned task {task_id} to node {node_id}")
        except Exception as e:
            self.logger.exception(f"Failed to assign task: {e}")
            raise
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute cluster management tasks"""
        action = task.get("action")
        
        if action == "register_node":
            node = self.register_node(task["node_id"], task["capabilities"])
            return {"status": "success", "node": node}
        elif action == "submit_task":
            task_id = self.submit_task(task["task_data"])
            return {"status": "success", "task_id": task_id}
        elif action == "assign":
            self.assign_task_to_node(task["task_id"], task["node_id"])
            return {"status": "success"}
        elif action == "list_nodes":
            return {"status": "success", "nodes": list(self.nodes.values())}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class LoadBalancerAgent(BaseAgent):
    """
    Distributes workload across compute resources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="LoadBalancerAgent", config=config)
        self.distribution_strategy = config.get("strategy", "round_robin") if config else "round_robin"
        self.node_loads = {}
    
    def distribute_task(self, task: Dict[str, Any], available_nodes: List[str]) -> str:
        """Distribute task to the best available node"""
        try:
            if not available_nodes:
                raise ValueError("No available nodes")
            
            if self.distribution_strategy == "round_robin":
                selected_node = self._round_robin_select(available_nodes)
            elif self.distribution_strategy == "least_loaded":
                selected_node = self._least_loaded_select(available_nodes)
            else:
                selected_node = available_nodes[0]
            
            # Update load tracking
            if selected_node not in self.node_loads:
                self.node_loads[selected_node] = 0
            self.node_loads[selected_node] += 1
            
            self.logger.info(f"Distributed task to node {selected_node}")
            return selected_node
        except Exception as e:
            self.logger.exception(f"Failed to distribute task: {e}")
            raise
    
    def _round_robin_select(self, nodes: List[str]) -> str:
        """Round-robin selection"""
        # Simple round-robin based on current loads
        min_load = min(self.node_loads.get(n, 0) for n in nodes)
        return next(n for n in nodes if self.node_loads.get(n, 0) == min_load)
    
    def _least_loaded_select(self, nodes: List[str]) -> str:
        """Least loaded node selection"""
        return min(nodes, key=lambda n: self.node_loads.get(n, 0))
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute load balancing tasks"""
        action = task.get("action")
        
        if action == "distribute":
            node = self.distribute_task(task["task_data"], task["available_nodes"])
            return {"status": "success", "selected_node": node}
        elif action == "get_loads":
            return {"status": "success", "loads": self.node_loads}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}


class SelfHealingAgent(BaseAgent):
    """
    Monitors system health and performs automatic recovery
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="SelfHealingAgent", config=config)
        self.health_checks = []
        self.recovery_actions = []
    
    def perform_health_check(self, component: str) -> Dict[str, Any]:
        """Perform health check on a component"""
        try:
            health_status = {
                "component": component,
                "status": "healthy",
                "checked_at": now_ts(),
                "issues": []
            }
            
            # Simplified health check (can be enhanced with actual checks)
            # In a real implementation, this would check component-specific metrics
            
            self.health_checks.append(health_status)
            self.logger.debug(f"Health check for {component}: {health_status['status']}")
            return health_status
        except Exception as e:
            self.logger.exception(f"Health check failed: {e}")
            return {
                "component": component,
                "status": "error",
                "checked_at": now_ts(),
                "issues": [str(e)]
            }
    
    def trigger_recovery(self, component: str, issue: str) -> Dict[str, Any]:
        """Trigger recovery action for a component"""
        try:
            recovery_action = {
                "component": component,
                "issue": issue,
                "action": self._determine_recovery_action(component, issue),
                "triggered_at": now_ts(),
                "status": "initiated"
            }
            
            self.recovery_actions.append(recovery_action)
            self.logger.info(f"Triggered recovery for {component}: {recovery_action['action']}")
            return recovery_action
        except Exception as e:
            self.logger.exception(f"Failed to trigger recovery: {e}")
            raise
    
    def _determine_recovery_action(self, component: str, issue: str) -> str:
        """Determine appropriate recovery action"""
        # Simplified recovery determination
        if "memory" in issue.lower():
            return "restart_component"
        elif "timeout" in issue.lower():
            return "increase_timeout"
        else:
            return "log_and_monitor"
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute self-healing tasks"""
        action = task.get("action")
        
        if action == "health_check":
            health = self.perform_health_check(task["component"])
            return {"status": "success", "health": health}
        elif action == "recover":
            recovery = self.trigger_recovery(task["component"], task["issue"])
            return {"status": "success", "recovery": recovery}
        elif action == "list_checks":
            return {"status": "success", "health_checks": self.health_checks}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}
