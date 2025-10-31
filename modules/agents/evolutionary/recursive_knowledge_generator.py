#!/usr/bin/env python3
"""
RecursiveKnowledgeGenerator: Spawns micro-agents to generate new knowledge recursively
"""
import logging
import asyncio
import hashlib
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import time
from ..memory.memory_agent import now_ts
from ..base_agent import BaseAgent, AgentCapability, AgentStatus

logger = logging.getLogger("kalki.agents.phase11")


class RecursiveKnowledgeGenerator(BaseAgent):
    """
    Spawns micro-agents to generate new knowledge recursively with governance and safety controls
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="RecursiveKnowledgeGenerator",
            capabilities=[AgentCapability.SELF_IMPROVEMENT, AgentCapability.LIFECYCLE_MANAGEMENT],
            description="Spawns micro-agents to generate new knowledge recursively with governance controls",
            config=config
        )
        self.knowledge_tree = {}
        self.micro_agents = []
        self.pending_actions = {}
        self.spawn_quotas = []  # Track micro-agent spawning
        self.audit_trail = []

        # Rate limiting: max 3 micro-agents per hour
        self.spawn_limit = 3
        self.spawn_window = timedelta(hours=1)

    async def spawn_micro_agent(self, topic: str, depth: int = 0,
                               approval_required: bool = True) -> Dict[str, Any]:
        """Spawn a micro-agent to explore a topic with optional approval gate"""
        try:
            # Check rate limits
            if not self._check_spawn_quota():
                return {
                    "status": "rate_limited",
                    "message": "Micro-agent spawn quota exceeded"
                }

            # Check resource availability
            if not await self._check_resources_available():
                return {
                    "status": "resource_unavailable",
                    "message": "Insufficient resources for micro-agent spawning"
                }

            if approval_required:
                # Create pending action
                spawn_id = self._create_pending("spawn", {
                    "topic": topic,
                    "depth": depth
                })

                # Emit event for approval workflow
                await self._emit_event("action.pending", {
                    "id": spawn_id,
                    "type": "micro_agent_spawn",
                    "topic": topic,
                    "requires_approval": True
                })

                return {"status": "pending", "id": spawn_id}

            # Execute spawning directly
            return await self._execute_spawn(topic, depth)

        except Exception as e:
            self.logger.exception(f"Micro-agent spawn request failed: {e}")
            await self._record_audit("spawn_failed", {"topic": topic, "error": str(e)})
            raise

    async def _execute_spawn(self, topic: str, depth: int) -> Dict[str, Any]:
        """Execute the actual micro-agent spawning"""
        agent_id = f"micro_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        micro_agent = {
            "agent_id": agent_id,
            "topic": topic,
            "depth": depth,
            "status": "active",
            "knowledge_generated": [],
            "spawned_at": now_ts(),
            "source_agent": self.name,
            "version": "1.0"
        }

        self.micro_agents.append(micro_agent)
        self._update_spawn_quota()

        # Persist micro-agent record
        await self._persist_micro_agent(micro_agent)

        # Emit telemetry
        await self._record_audit("spawn_completed", micro_agent)

        self.logger.info(f"Spawned micro-agent {agent_id} for topic '{topic}'")
        return micro_agent

    async def generate_knowledge(self, topic: str, max_depth: int = 3,
                                approval_required: bool = True) -> Dict[str, Any]:
        """Recursively generate knowledge on a topic with optional approval gate"""
        try:
            # Check resource availability for heavy operation
            if not await self._check_resources_available():
                return {
                    "status": "resource_unavailable",
                    "message": "Insufficient resources for knowledge generation"
                }

            if approval_required:
                # Create pending action
                gen_id = self._create_pending("generate", {
                    "topic": topic,
                    "max_depth": max_depth
                })

                # Emit event for approval workflow
                await self._emit_event("action.pending", {
                    "id": gen_id,
                    "type": "knowledge_generation",
                    "topic": topic,
                    "requires_approval": True
                })

                return {"status": "pending", "id": gen_id}

            # Execute generation directly
            return await self._execute_generation(topic, max_depth)

        except Exception as e:
            self.logger.exception(f"Knowledge generation request failed: {e}")
            await self._record_audit("generate_failed", {"topic": topic, "error": str(e)})
            raise

    async def _execute_generation(self, topic: str, max_depth: int) -> Dict[str, Any]:
        """Execute the actual knowledge generation with timeout"""
        start_time = time.time()

        try:
            # Use asyncio.wait_for with timeout
            result = await asyncio.wait_for(
                self._perform_generation(topic, max_depth),
                timeout=1800  # 30 minutes timeout
            )

            # Record successful generation
            generation_record = {
                "topic": topic,
                "knowledge": result["knowledge"],
                "depth": max_depth,
                "generated_at": now_ts(),
                "duration": time.time() - start_time,
                "source_agent": self.name,
                "version": "1.0",
                "checksum": self._calculate_checksum(result["knowledge"])
            }

            # Persist knowledge tree
            await self._persist_knowledge_tree(topic, generation_record)

            # Emit telemetry
            await self._record_audit("generate_completed", generation_record)

            self.logger.info(f"Generated knowledge tree for '{topic}' with depth {max_depth}")
            return {"status": "success", **generation_record}

        except asyncio.TimeoutError:
            await self._record_audit("generate_timeout", {"topic": topic, "max_depth": max_depth})
            return {"status": "timeout", "topic": topic}
        except Exception as e:
            await self._record_audit("generate_error", {"topic": topic, "max_depth": max_depth, "error": str(e)})
            raise

    async def _perform_generation(self, topic: str, max_depth: int) -> Dict[str, Any]:
        """Perform the actual knowledge generation logic"""
        if topic not in self.knowledge_tree:
            self.knowledge_tree[topic] = {"subtopics": [], "knowledge": []}

        # Spawn micro-agent for this topic
        agent = await self._execute_spawn(topic, depth=0)

        # Generate knowledge recursively
        knowledge = await self._generate_topic_knowledge(topic, 0, max_depth)

        # Update agent status
        agent["status"] = "completed"
        agent["knowledge_generated"] = knowledge

        return {
            "topic": topic,
            "knowledge": knowledge,
            "depth": max_depth
        }

    async def _generate_topic_knowledge(self, topic: str, current_depth: int, max_depth: int) -> List[Dict[str, Any]]:
        """Generate knowledge recursively with deduplication"""
        if current_depth >= max_depth:
            return []

        knowledge = []

        # Generate base knowledge
        base_concept = {
            "concept": f"Core concept of {topic}",
            "depth": current_depth,
            "type": "foundational",
            "checksum": self._calculate_checksum([f"Core concept of {topic}"])
        }

        # Check for duplicates
        if not self._is_duplicate_concept(base_concept):
            knowledge.append(base_concept)

        # Generate subtopics (simplified)
        if current_depth < max_depth - 1:
            subtopics = [f"{topic}_advanced", f"{topic}_applications"]
            for subtopic in subtopics:
                sub_knowledge = await self._generate_topic_knowledge(subtopic, current_depth + 1, max_depth)
                knowledge.extend(sub_knowledge)

        return knowledge

    def _is_duplicate_concept(self, concept: Dict[str, Any]) -> bool:
        """Check if concept is duplicate based on checksum"""
        checksum = concept.get("checksum")
        if not checksum:
            return False

        # Check existing knowledge for duplicates
        for topic_data in self.knowledge_tree.values():
            for existing in topic_data.get("knowledge", []):
                if existing.get("checksum") == checksum:
                    return True
        return False

    def _calculate_checksum(self, data) -> str:
        """Calculate checksum for data deduplication"""
        if isinstance(data, (list, dict)):
            # Convert to string representation for hashing
            if isinstance(data, dict):
                data_str = str(sorted(data.items()))
            else:
                # For lists, convert each item to string and sort
                data_str = str(sorted([str(item) for item in data]))
        else:
            data_str = str(data)
        return hashlib.md5(data_str.encode()).hexdigest()

    def _check_spawn_quota(self) -> bool:
        """Check if micro-agent spawn quota allows this operation"""
        now = datetime.now()

        # Remove expired entries
        self.spawn_quotas = [
            ts for ts in self.spawn_quotas
            if now - ts < self.spawn_window
        ]

        # Allow max spawn_limit spawns per window
        return len(self.spawn_quotas) < self.spawn_limit

    def _update_spawn_quota(self):
        """Update spawn quota after successful spawn"""
        self.spawn_quotas.append(datetime.now())

    async def _check_resources_available(self) -> bool:
        """Check if resources are available for operations"""
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
            if action["type"] == "spawn":
                result = await self._execute_spawn(
                    action["data"]["topic"],
                    action["data"]["depth"]
                )
                action["status"] = "approved"
                await self._emit_event("action.approved", {"id": action_id, "result": result})
                return {"status": "success", "result": result}
            elif action["type"] == "generate":
                result = await self._execute_generation(
                    action["data"]["topic"],
                    action["data"]["max_depth"]
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

    async def _persist_micro_agent(self, agent: Dict[str, Any]):
        """Persist micro-agent record using KnowledgeLifecycleAgent"""
        # In a real implementation, this would call KnowledgeLifecycleAgent.create_version()
        # For now, just log
        self.logger.info(f"Persisted micro-agent: {agent['agent_id']}")

    async def _persist_knowledge_tree(self, topic: str, record: Dict[str, Any]):
        """Persist knowledge tree using KnowledgeLifecycleAgent"""
        # In a real implementation, this would call KnowledgeLifecycleAgent.create_version()
        # For now, just log
        self.logger.info(f"Persisted knowledge tree for topic: {topic}")

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
        """Execute knowledge generation tasks"""
        action = task.get("action")

        if action == "spawn":
            result = await self.spawn_micro_agent(
                task["topic"],
                task.get("depth", 0),
                task.get("approval_required", True)
            )
            return result
        elif action == "generate":
            result = await self.generate_knowledge(
                task["topic"],
                task.get("max_depth", 3),
                task.get("approval_required", True)
            )
            return result
        elif action == "approve":
            return await self.approve_action(task["action_id"])
        elif action == "list_agents":
            return {"status": "success", "micro_agents": self.micro_agents}
        elif action == "pending":
            return {"status": "success", "pending": list(self.pending_actions.values())}
        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get audit trail for compliance"""
        return self.audit_trail.copy()

    async def initialize(self) -> bool:
        """Initialize the RecursiveKnowledgeGenerator"""
        try:
            self.logger.info("Initializing RecursiveKnowledgeGenerator")
            # Load existing knowledge tree and micro agents if available
            # In a real implementation, this would load from persistent storage
            self.status = AgentStatus.READY
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize RecursiveKnowledgeGenerator: {e}")
            return False

    async def shutdown(self) -> bool:
        """Shutdown the RecursiveKnowledgeGenerator"""
        try:
            self.logger.info("Shutting down RecursiveKnowledgeGenerator")
            # Save any pending state
            self.status = AgentStatus.TERMINATED
            return True
        except Exception as e:
            self.logger.exception(f"Failed to shutdown RecursiveKnowledgeGenerator: {e}")
            return False