#!/usr/bin/env python3
"""
AutoFineTuneAgent: Automatically fine-tunes models for optimal performance
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time
from ..memory.memory_agent import now_ts
from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = logging.getLogger("kalki.agents.phase11")


class AutoFineTuneAgent(BaseAgent):
    """
    Automatically fine-tunes models for optimal performance with governance and safety controls
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="AutoFineTuneAgent",
            capabilities=[AgentCapability.SELF_IMPROVEMENT, AgentCapability.OPTIMIZATION],
            description="Automatically fine-tunes models for optimal performance with governance controls",
            config=config
        )
        self.tuning_history = []
        self.pending_actions = {}
        self.tuning_quotas = {}  # Track tuning quotas per model
        self.audit_trail = []

        # Rate limiting: max 1 tune per model per 24h unless approved
        self.tuning_cooldown = timedelta(hours=24)

    async def tune_model(self, model_id: str, performance_metrics: Dict[str, float],
                        approval_required: bool = True) -> Dict[str, Any]:
        """Fine-tune a model based on performance metrics with optional approval gate"""
        try:
            # Check rate limits
            if not self._check_tuning_quota(model_id):
                return {
                    "status": "rate_limited",
                    "message": f"Tuning quota exceeded for model {model_id}"
                }

            # Check resource availability
            if not await self._check_resources_available():
                return {
                    "status": "resource_unavailable",
                    "message": "Insufficient resources for tuning operation"
                }

            if approval_required:
                # Create pending action
                tuning_id = self._create_pending("tune", {
                    "model_id": model_id,
                    "performance_metrics": performance_metrics
                })

                # Emit event for approval workflow
                await self._emit_event("action.pending", {
                    "id": tuning_id,
                    "type": "model_tuning",
                    "model_id": model_id,
                    "requires_approval": True
                })

                return {"status": "pending", "id": tuning_id}

            # Execute tuning directly
            return await self._execute_tuning(model_id, performance_metrics)

        except Exception as e:
            self.logger.exception(f"Model tuning request failed: {e}")
            await self._record_audit("tune_failed", {"model_id": model_id, "error": str(e)})
            raise

    async def _execute_tuning(self, model_id: str, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Execute the actual tuning operation with timeout and monitoring"""
        tuning_id = f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        start_time = time.time()

        try:
            # Use asyncio.wait_for with timeout
            result = await asyncio.wait_for(
                self._perform_tuning(model_id, performance_metrics),
                timeout=3600  # 1 hour timeout
            )

            # Record successful tuning
            tuning_record = {
                "tuning_id": tuning_id,
                "model_id": model_id,
                "metrics_before": performance_metrics,
                "metrics_after": result.get("metrics_after", {}),
                "strategy": result.get("strategy"),
                "status": "completed",
                "improvement": result.get("improvement", 0.0),
                "tuned_at": now_ts(),
                "duration": time.time() - start_time,
                "source_agent": self.name,
                "version": "1.0"
            }

            # Persist to KnowledgeLifecycleAgent
            await self._persist_tuning_record(tuning_record)

            self.tuning_history.append(tuning_record)
            self._update_tuning_quota(model_id)

            # Emit telemetry
            await self._record_audit("tune_completed", tuning_record)

            self.logger.info(f"Tuned model {model_id} with strategy {result.get('strategy')}")
            return tuning_record

        except asyncio.TimeoutError:
            await self._record_audit("tune_timeout", {"tuning_id": tuning_id, "model_id": model_id})
            return {"status": "timeout", "tuning_id": tuning_id}
        except Exception as e:
            await self._record_audit("tune_error", {"tuning_id": tuning_id, "model_id": model_id, "error": str(e)})
            raise

    async def _perform_tuning(self, model_id: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Perform the actual tuning logic"""
        # Determine tuning strategy based on metrics
        strategy = self._determine_tuning_strategy(metrics)

        # Simulate tuning process (in real implementation, this would call actual ML tuning)
        await asyncio.sleep(0.1)  # Simulate processing time

        improvement = self._simulate_improvement(strategy)

        return {
            "strategy": strategy,
            "improvement": improvement,
            "metrics_after": {
                "accuracy": metrics.get("accuracy", 0.5) + improvement,
                "speed": metrics.get("speed", 0.5) + (improvement * 0.5)
            }
        }

    def _determine_tuning_strategy(self, metrics: Dict[str, float]) -> str:
        """Determine optimal tuning strategy"""
        accuracy = metrics.get("accuracy", 0.5)
        speed = metrics.get("speed", 0.5)

        if accuracy < 0.7:
            return "improve_accuracy"
        elif speed < 0.5:
            return "optimize_speed"
        else:
            return "balanced_tuning"

    def _simulate_improvement(self, strategy: str) -> float:
        """Simulate improvement from tuning"""
        improvements = {
            "improve_accuracy": 0.15,
            "optimize_speed": 0.25,
            "balanced_tuning": 0.10
        }
        return improvements.get(strategy, 0.05)

    def _check_tuning_quota(self, model_id: str) -> bool:
        """Check if tuning quota allows this operation"""
        now = datetime.now()
        if model_id not in self.tuning_quotas:
            self.tuning_quotas[model_id] = []

        # Remove expired entries
        self.tuning_quotas[model_id] = [
            ts for ts in self.tuning_quotas[model_id]
            if now - ts < self.tuning_cooldown
        ]

        # Allow max 1 tuning per cooldown period
        return len(self.tuning_quotas[model_id]) == 0

    def _update_tuning_quota(self, model_id: str):
        """Update tuning quota after successful tuning"""
        self.tuning_quotas[model_id].append(datetime.now())

    async def _check_resources_available(self) -> bool:
        """Check if resources are available for tuning"""
        # In a real implementation, this would check with ComputeOptimizerAgent
        # For now, always return True
        return True

    def _create_pending(self, action_type: str, data: Dict[str, Any]) -> str:
        """Create a pending action requiring approval"""
        action_id = f"pending_{action_type}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        self.pending_actions[action_id] = {
            "id": action_id,
            "type": action_type,
            "data": data,
            "created_at": now_ts(),
            "status": "pending"
        }
        return action_id

    async def approve_action(self, action_id: str) -> Dict[str, Any]:
        """Approve and execute a pending action"""
        if action_id not in self.pending_actions:
            return {"status": "error", "message": "Action not found"}

        action = self.pending_actions[action_id]
        if action["status"] != "pending":
            return {"status": "error", "message": "Action already processed"}

        try:
            # Execute the approved action
            if action["type"] == "tune":
                result = await self._execute_tuning(
                    action["data"]["model_id"],
                    action["data"]["performance_metrics"]
                )
                action["status"] = "approved"
                await self._emit_event("action.approved", {"id": action_id, "result": result})
                return {"status": "success", "result": result}
            else:
                return {"status": "error", "message": f"Unknown action type: {action['type']}"}
        except Exception as e:
            action["status"] = "failed"
            await self._record_audit("approval_failed", {"action_id": action_id, "error": str(e)})
            return {"status": "error", "message": str(e)}

    async def _persist_tuning_record(self, record: Dict[str, Any]):
        """Persist tuning record using KnowledgeLifecycleAgent"""
        # In a real implementation, this would call KnowledgeLifecycleAgent.create_version()
        # For now, just log
        self.logger.info(f"Persisted tuning record: {record['tuning_id']}")

    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit event to EventBus"""
        # In a real implementation, this would publish to EventBus
        self.logger.info(f"Emitted event {event_type}: {data}")

    async def _record_audit(self, action: str, data: Dict[str, Any]):
        """Record audit trail and telemetry"""
        audit_entry = {
            "timestamp": now_ts(),
            "agent": self.name,
            "action": action,
            "data": data
        }
        self.audit_trail.append(audit_entry)

        # In a real implementation, this would also send to PerformanceMonitorAgent
        self.logger.info(f"Audit: {action} - {data}")

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute tuning tasks"""
        action = task.get("action")

        if action == "tune":
            result = await self.tune_model(
                task["model_id"],
                task["performance_metrics"],
                task.get("approval_required", True)
            )
            return result
        elif action == "approve":
            return await self.approve_action(task["action_id"])
        elif action == "history":
            return {"status": "success", "history": self.tuning_history}
        elif action == "pending":
            return {"status": "success", "pending": list(self.pending_actions.values())}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail for compliance"""
        return self.audit_trail.copy()

    async def initialize(self) -> bool:
        """Initialize the AutoFineTuneAgent"""
        try:
            self.logger.info("Initializing AutoFineTuneAgent")
            # Load existing tuning history if available
            # In a real implementation, this would load from persistent storage
            self.status = AgentStatus.READY
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize AutoFineTuneAgent: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the AutoFineTuneAgent"""
        try:
            self.logger.info("Shutting down AutoFineTuneAgent")
            # Save any pending state
            self.status = AgentStatus.TERMINATED
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown AutoFineTuneAgent: {e}")
            return False