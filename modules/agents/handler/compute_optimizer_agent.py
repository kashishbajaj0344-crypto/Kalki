"""
Compute Optimizer Agent - Dynamically allocates and manages computational resources
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import psutil

from modules.agents.base_agent import BaseAgent, AgentCapability
from modules.config import get_config, CONFIG


class ComputeOptimizerAgent(BaseAgent):
    """
    Dynamically allocates CPU/GPU/memory resources with monitoring and optimization
    Enhanced with async execution, persistent resource tracking, and intelligent allocation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="ComputeOptimizerAgent",
            capabilities=[AgentCapability.COMPUTE_SCALING, AgentCapability.LOAD_BALANCING],
            description="Manages computational resources with monitoring and optimization",
            config=config
        )
        self.resource_allocations = {}
        self.resource_history = []
        self.monitoring_active = False

        # Resource thresholds
        self.cpu_threshold = config.get("cpu_threshold", 80.0) if config else 80.0
        self.memory_threshold = config.get("memory_threshold", 85.0) if config else 85.0
        self.disk_threshold = config.get("disk_threshold", 90.0) if config else 90.0

        # Persistence setup
        self.data_dir = Path(CONFIG.get("data_dir", "data")) / "resources"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing allocations
        self._load_allocations()

        # Start background monitoring
        self.monitoring_task = None

    def _load_allocations(self):
        """Load persisted resource allocations from disk"""
        try:
            alloc_file = self.data_dir / "allocations.json"
            if alloc_file.exists():
                with open(alloc_file, 'r') as f:
                    data = json.load(f)
                    self.resource_allocations = data.get("allocations", {})
                    self.resource_history = data.get("history", [])
                self.logger.info(f"Loaded {len(self.resource_allocations)} resource allocations from disk")
        except Exception as e:
            self.logger.exception(f"Failed to load allocations: {e}")

    def _save_allocations(self):
        """Persist resource allocations to disk"""
        try:
            alloc_file = self.data_dir / "allocations.json"
            data = {
                "allocations": self.resource_allocations,
                "history": self.resource_history[-1000:],  # Keep last 1000 entries
                "last_updated": datetime.now().isoformat()
            }
            with open(alloc_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug("Saved resource allocations to disk")
        except Exception as e:
            self.logger.exception(f"Failed to save allocations: {e}")

    async def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource utilization asynchronously"""
        try:
            # Run CPU monitoring in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            cpu_percent = await loop.run_in_executor(None, lambda: psutil.cpu_percent(interval=0.1))

            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Get per-CPU usage
            cpu_percents = await loop.run_in_executor(None, lambda: psutil.cpu_percent(percpu=True, interval=0.1))

            resources = {
                "cpu_percent": cpu_percent,
                "cpu_percents": cpu_percents,
                "cpu_count": psutil.cpu_count(),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "memory_percent": memory.percent,
                "disk_total_gb": disk.total / (1024**3),
                "disk_free_gb": disk.free / (1024**3),
                "disk_used_gb": disk.used / (1024**3),
                "disk_percent": disk.percent,
                "timestamp": datetime.now().isoformat()
            }

            # Store in history
            self.resource_history.append(resources)
            if len(self.resource_history) > 1000:
                self.resource_history = self.resource_history[-1000:]

            return resources

        except Exception as e:
            self.logger.exception(f"Failed to get system resources: {e}")
            return {}

    async def allocate_resources(self, task_id: str, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        Allocate resources for a task with intelligent optimization
        """
        try:
            current_resources = await self.get_system_resources()

            # Check if allocation is feasible
            if current_resources.get("cpu_percent", 0) > self.cpu_threshold:
                raise ResourceWarning(f"CPU usage too high: {current_resources['cpu_percent']}%")

            if current_resources.get("memory_percent", 0) > self.memory_threshold:
                raise ResourceWarning(f"Memory usage too high: {current_resources['memory_percent']}%")

            # Calculate optimal allocation
            allocation = await self._calculate_optimal_allocation(task_id, requirements, current_resources)

            self.resource_allocations[task_id] = allocation
            self._save_allocations()

            self.logger.info(f"Allocated resources for task {task_id}: {allocation}")
            return allocation

        except Exception as e:
            self.logger.exception(f"Failed to allocate resources: {e}")
            raise

    async def _calculate_optimal_allocation(self, task_id: str, requirements: Dict[str, Any], current_resources: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal resource allocation based on requirements and availability"""
        # Requested resources
        requested_cpu = requirements.get("cpu_cores", 1)
        requested_memory = requirements.get("memory_gb", 1.0)
        priority = requirements.get("priority", "normal")

        # Available resources
        available_cpu = current_resources.get("cpu_count", 1)
        available_memory = current_resources.get("memory_available_gb", 1.0)

        # Adjust based on priority
        priority_multipliers = {
            "low": 0.5,
            "normal": 1.0,
            "high": 1.5,
            "critical": 2.0
        }
        multiplier = priority_multipliers.get(priority, 1.0)

        # Allocate resources (don't exceed available)
        allocated_cpu = min(int(requested_cpu * multiplier), available_cpu)
        allocated_memory = min(requested_memory * multiplier, available_memory)

        # Ensure minimum allocations
        allocated_cpu = max(allocated_cpu, 1)
        allocated_memory = max(allocated_memory, 0.5)

        allocation = {
            "task_id": task_id,
            "cpu_cores": allocated_cpu,
            "memory_gb": allocated_memory,
            "priority": priority,
            "allocated": True,
            "allocated_at": datetime.now().isoformat(),
            "current_system_state": {
                "cpu_percent": current_resources.get("cpu_percent"),
                "memory_percent": current_resources.get("memory_percent")
            }
        }

        return allocation

    async def release_resources(self, task_id: str):
        """Release resources allocated to a task"""
        if task_id in self.resource_allocations:
            allocation = self.resource_allocations[task_id]
            allocation["released_at"] = datetime.now().isoformat()
            allocation["status"] = "released"

            # Move to history (keep in allocations for audit)
            del self.resource_allocations[task_id]

            self._save_allocations()
            self.logger.info(f"Released resources for task {task_id}")

    async def monitor_resources(self, interval_seconds: float = 5.0):
        """Continuously monitor system resources"""
        self.monitoring_active = True
        self.logger.info("Starting resource monitoring")

        try:
            while self.monitoring_active:
                resources = await self.get_system_resources()

                # Check thresholds and log warnings
                if resources.get("cpu_percent", 0) > self.cpu_threshold:
                    self.logger.warning(f"High CPU usage: {resources['cpu_percent']}%")

                if resources.get("memory_percent", 0) > self.memory_threshold:
                    self.logger.warning(f"High memory usage: {resources['memory_percent']}%")

                if resources.get("disk_percent", 0) > self.disk_threshold:
                    self.logger.warning(f"High disk usage: {resources['disk_percent']}%")

                await asyncio.sleep(interval_seconds)

        except asyncio.CancelledError:
            self.logger.info("Resource monitoring cancelled")
        except Exception as e:
            self.logger.exception(f"Resource monitoring failed: {e}")
        finally:
            self.monitoring_active = False

    def start_monitoring(self):
        """Start background resource monitoring"""
        if not self.monitoring_task or self.monitoring_task.done():
            self.monitoring_task = asyncio.create_task(self.monitor_resources())

    def stop_monitoring(self):
        """Stop background resource monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()

    async def optimize_allocation(self, task_id: str) -> Dict[str, Any]:
        """Optimize resource allocation for a running task"""
        try:
            if task_id not in self.resource_allocations:
                raise ValueError(f"No allocation found for task {task_id}")

            current_resources = await self.get_system_resources()
            current_allocation = self.resource_allocations[task_id]

            # Simple optimization: reduce allocation if system is under pressure
            optimization = {"task_id": task_id, "changes": {}}

            if current_resources.get("cpu_percent", 0) > self.cpu_threshold:
                # Reduce CPU allocation
                old_cpu = current_allocation["cpu_cores"]
                new_cpu = max(1, old_cpu - 1)
                current_allocation["cpu_cores"] = new_cpu
                optimization["changes"]["cpu_cores"] = {"old": old_cpu, "new": new_cpu}

            if current_resources.get("memory_percent", 0) > self.memory_threshold:
                # Reduce memory allocation
                old_memory = current_allocation["memory_gb"]
                new_memory = max(0.5, old_memory * 0.8)
                current_allocation["memory_gb"] = new_memory
                optimization["changes"]["memory_gb"] = {"old": old_memory, "new": new_memory}

            if optimization["changes"]:
                current_allocation["optimized_at"] = datetime.now().isoformat()
                self._save_allocations()
                self.logger.info(f"Optimized allocation for task {task_id}: {optimization}")

            return optimization

        except Exception as e:
            self.logger.exception(f"Failed to optimize allocation: {e}")
            raise

    def get_allocation(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get current resource allocation for a task"""
        return self.resource_allocations.get(task_id)

    def list_allocations(self) -> List[Dict[str, Any]]:
        """List all current resource allocations"""
        return list(self.resource_allocations.values())

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Async execute compute optimization tasks"""
        try:
            action = task.get("action")

            if action == "get_resources":
                resources = await self.get_system_resources()
                return {"status": "success", "resources": resources}
            elif action == "allocate":
                allocation = await self.allocate_resources(task["task_id"], task["requirements"])
                return {"status": "success", "allocation": allocation}
            elif action == "release":
                await self.release_resources(task["task_id"])
                return {"status": "success"}
            elif action == "optimize":
                optimization = await self.optimize_allocation(task["task_id"])
                return {"status": "success", "optimization": optimization}
            elif action == "get_allocation":
                allocation = self.get_allocation(task["task_id"])
                if allocation:
                    return {"status": "success", "allocation": allocation}
                else:
                    return {"status": "error", "message": f"No allocation found for task {task['task_id']}"}
            elif action == "list_allocations":
                allocations = self.list_allocations()
                return {"status": "success", "allocations": allocations}
            elif action == "start_monitoring":
                self.start_monitoring()
                return {"status": "success", "message": "Resource monitoring started"}
            elif action == "stop_monitoring":
                self.stop_monitoring()
                return {"status": "success", "message": "Resource monitoring stopped"}
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

        except Exception as e:
            self.logger.exception(f"Failed to execute compute optimization task: {e}")
            return {"status": "error", "message": str(e)}

    async def initialize(self) -> bool:
        """
        Initialize the compute optimizer agent
        """
        try:
            # Start resource monitoring
            await self.start_monitoring()
            self.logger.info("ComputeOptimizerAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize ComputeOptimizerAgent: {e}")
            return False

    async def shutdown(self) -> bool:
        """
        Gracefully shutdown the compute optimizer agent
        """
        try:
            # Stop resource monitoring
            self.stop_monitoring()
            self.logger.info("ComputeOptimizerAgent shutdown successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown ComputeOptimizerAgent: {e}")
            return False