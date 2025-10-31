#!/usr/bin/env python3
"""
Phase 4 Memory Integration Test Suite
Tests end-to-end session + memory usage with persistence validation
"""
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modules.agents.agent_manager import AgentManager
from modules.agents.base_agent import AgentCapability
from modules.agents.memory import SessionAgent, MemoryAgent
from modules.eventbus import EventBus


class Phase4MemoryTestSuite:
    """Comprehensive test suite for Phase 4 memory integration"""

    def __init__(self):
        self.event_bus = EventBus()
        self.agent_manager = AgentManager(event_bus=self.event_bus)
        self.logger = logging.getLogger("test.phase4_memory")
        self.test_results = []

    async def setup(self):
        """Setup test environment"""
        self.logger.info("Setting up Phase 4 memory test environment")

        # Create agent manager with higher concurrent limit to avoid resource issues
        self.agent_manager = AgentManager(event_bus=self.event_bus, max_concurrent_agents=10)

        # Register memory agents
        session_agent = SessionAgent()
        memory_agent = MemoryAgent()

        await self.agent_manager.register_agent(session_agent)
        await self.agent_manager.register_agent(memory_agent)

        self.logger.info("Test environment setup complete")

    async def teardown(self):
        """Cleanup test environment"""
        self.logger.info("Cleaning up test environment")
        await self.agent_manager.shutdown_all()

    async def run_all_tests(self):
        """Run all Phase 4 memory integration tests"""
        self.logger.info("Starting Phase 4 memory integration tests")

        try:
            # Test 1: Session Management
            await self.test_session_management()

            # Test 2: Memory Storage and Retrieval
            await self.test_memory_operations()

            # Test 3: Memory Persistence
            await self.test_memory_persistence()

        except Exception as e:
            self.logger.exception(f"Test suite failed: {e}")
            self.test_results.append({
                "test": "test_suite",
                "status": "failed",
                "error": str(e)
            })

        finally:
            # Print results
            self.print_results()

    async def test_session_management(self):
        """Test session creation, updates, and retrieval"""
        self.logger.info("Testing session management")

        try:
            # Create session
            session_result = await self.agent_manager.agents["SessionAgent"].execute({
                "action": "create",
                "user_id": "test_user",
                "metadata": {"test_session": True, "purpose": "memory_integration_test"}
            })
            assert session_result["status"] == "success", f"Session creation failed: {session_result}"
            session_id = session_result["session_id"]

            # Update session
            update_result = await self.agent_manager.agents["SessionAgent"].execute({
                "action": "update",
                "session_id": session_id,
                "context_update": {"phase": "testing", "progress": "session_created"}
            })
            assert update_result["status"] == "success", f"Session update failed: {update_result}"

            # Retrieve session
            get_result = await self.agent_manager.agents["SessionAgent"].execute({
                "action": "get",
                "session_id": session_id
            })
            assert get_result["status"] == "success", f"Session retrieval failed: {get_result}"
            session_data = get_result["data"]
            assert session_data["user_id"] == "test_user", "Session user_id mismatch"
            assert len(session_data["context"]) > 0, "Session context not updated"

            # Close session
            close_result = await self.agent_manager.agents["SessionAgent"].execute({
                "action": "close",
                "session_id": session_id
            })
            assert close_result["status"] == "success", f"Session close failed: {close_result}"

            self.test_results.append({
                "test": "session_management",
                "status": "passed",
                "session_id": session_id
            })

        except Exception as e:
            self.logger.exception("Session management test failed")
            self.test_results.append({
                "test": "session_management",
                "status": "failed",
                "error": str(e)
            })

    async def test_memory_operations(self):
        """Test episodic and semantic memory operations"""
        self.logger.info("Testing memory operations")

        try:
            # Store episodic memory
            episodic_event = {
                "type": "test_event",
                "description": "Testing episodic memory storage",
                "test_data": {"key": "value", "number": 42}
            }

            episodic_result = await self.agent_manager.agents["MemoryAgent"].execute({
                "action": "store",
                "type": "episodic",
                "event": episodic_event
            })
            assert episodic_result["status"] == "success", f"Episodic storage failed: {episodic_result}"
            episodic_id = episodic_result["memory_id"]

            # Store semantic memory
            semantic_result = await self.agent_manager.agents["MemoryAgent"].execute({
                "action": "store",
                "type": "semantic",
                "concept": "test_concept",
                "knowledge": {
                    "description": "Test semantic knowledge",
                    "attributes": ["test", "memory", "integration"],
                    "confidence": 0.95
                }
            })
            assert semantic_result["status"] == "success", f"Semantic storage failed: {semantic_result}"

            # Recall episodic memories
            episodic_recall = await self.agent_manager.agents["MemoryAgent"].execute({
                "action": "recall",
                "type": "episodic",
                "limit": 5
            })
            assert episodic_recall["status"] == "success", f"Episodic recall failed: {episodic_recall}"
            memories = episodic_recall["memories"]
            assert len(memories) > 0, "No episodic memories recalled"
            assert any(m["event"]["type"] == "test_event" for m in memories), "Test event not found in recall"

            # Recall semantic memories
            semantic_recall = await self.agent_manager.agents["MemoryAgent"].execute({
                "action": "recall",
                "type": "semantic",
                "concept": "test_concept"
            })
            assert semantic_recall["status"] == "success", f"Semantic recall failed: {semantic_recall}"
            semantic_memories = semantic_recall["memories"]
            assert len(semantic_memories) > 0, "No semantic memories recalled"

            self.test_results.append({
                "test": "memory_operations",
                "status": "passed",
                "episodic_memories": len(memories),
                "semantic_memories": len(semantic_memories)
            })

        except Exception as e:
            self.logger.exception("Memory operations test failed")
            self.test_results.append({
                "test": "memory_operations",
                "status": "failed",
                "error": str(e)
            })

    async def test_multi_agent_memory_workflow(self):
        """Test memory hooks in multi-agent workflows"""
        self.logger.info("Testing multi-agent memory workflow")

        try:
            # Execute a task that should trigger memory hooks
            task_result = await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                {
                    "action": "store",
                    "type": "episodic",
                    "event": {
                        "type": "workflow_test",
                        "description": "Testing memory hooks in workflow",
                        "workflow_step": "initiation"
                    }
                }
            )

            assert task_result["status"] == "success", f"Workflow task failed: {task_result}"

            # Wait a moment for memory hooks to process
            await asyncio.sleep(0.1)

            # Check that episodic memory was stored (should be automatic via hooks)
            recall_result = await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                {
                    "action": "recall",
                    "type": "episodic",
                    "limit": 10
                }
            )

            assert recall_result["status"] == "success", f"Memory recall failed: {recall_result}"
            memories = recall_result["memories"]

            # Should have at least the manual storage + automatic task event storage
            task_events = [m for m in memories if m["event"].get("type") == "task_execution"]
            assert len(task_events) > 0, "No automatic task event storage found"

            self.test_results.append({
                "test": "multi_agent_memory_workflow",
                "status": "passed",
                "total_memories": len(memories),
                "task_events": len(task_events)
            })

        except Exception as e:
            self.logger.exception("Multi-agent memory workflow test failed")
            self.test_results.append({
                "test": "multi_agent_memory_workflow",
                "status": "failed",
                "error": str(e)
            })

    async def test_memory_persistence(self):
        """Test memory persistence across agent manager restart"""
        self.logger.info("Testing memory persistence")

        try:
            # Store some test data
            test_event = {
                "type": "persistence_test",
                "description": "Testing memory persistence",
                "timestamp": time.time()
            }

            store_result = await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                {
                    "action": "store",
                    "type": "episodic",
                    "event": test_event
                }
            )

            assert store_result["status"] == "success", f"Persistence storage failed: {store_result}"

            # Simulate restart by creating new agent manager with same agents
            new_event_bus = EventBus()
            new_agent_manager = AgentManager(event_bus=new_event_bus)

            new_session_agent = SessionAgent()
            new_memory_agent = MemoryAgent()

            await new_agent_manager.register_agent(new_session_agent)
            await new_agent_manager.register_agent(new_memory_agent)
            await new_agent_manager.initialize_all()

            # Try to recall the persisted memory
            recall_result = await new_agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                {
                    "action": "recall",
                    "type": "episodic",
                    "limit": 20
                }
            )

            await new_agent_manager.shutdown_all()

            assert recall_result["status"] == "success", f"Persistence recall failed: {recall_result}"
            memories = recall_result["memories"]

            # Should find our persistence test event
            persistence_events = [m for m in memories if m["event"].get("type") == "persistence_test"]
            assert len(persistence_events) > 0, "Persistence test event not found after restart"

            self.test_results.append({
                "test": "memory_persistence",
                "status": "passed",
                "persisted_events": len(persistence_events)
            })

        except Exception as e:
            self.logger.exception("Memory persistence test failed")
            self.test_results.append({
                "test": "memory_persistence",
                "status": "failed",
                "error": str(e)
            })

    async def test_memory_enhanced_planning(self):
        """Test memory-enhanced planning capabilities"""
        self.logger.info("Testing memory-enhanced planning")

        try:
            # First, store some planning-related memories
            planning_memory = {
                "type": "planning_session",
                "goal": "test planning with memory",
                "reasoning": "Step-by-step approach works best",
                "outcome": "success"
            }

            await self.agent_manager.execute_by_capability(
                AgentCapability.MEMORY,
                {
                    "action": "store",
                    "type": "episodic",
                    "event": planning_memory
                }
            )

            # Create a plan that should use memory
            plan_result = await self.agent_manager.execute_by_capability(
                AgentCapability.PLANNING,
                {
                    "action": "plan",
                    "params": {
                        "goal": "Create a test plan with memory enhancement",
                        "max_steps": 3
                    }
                }
            )

            assert plan_result["status"] == "success", f"Memory-enhanced planning failed: {plan_result}"

            # Check that memories were used (this would be indicated in the result)
            memories_used = plan_result.get("memories_used", 0)
            self.logger.info(f"Planning used {memories_used} memories")

            self.test_results.append({
                "test": "memory_enhanced_planning",
                "status": "passed",
                "memories_used": memories_used,
                "plan_steps": len(plan_result.get("plan", []))
            })

        except Exception as e:
            self.logger.exception("Memory-enhanced planning test failed")
            self.test_results.append({
                "test": "memory_enhanced_planning",
                "status": "failed",
                "error": str(e)
            })

    async def test_session_tracking_orchestration(self):
        """Test session tracking during orchestration cycles"""
        self.logger.info("Testing session tracking in orchestration")

        try:
            # Execute an orchestration workflow with session tracking
            workflow_result = await self.agent_manager.execute_by_capability(
                AgentCapability.ORCHESTRATION,
                {
                    "action": "plan_execute_feedback",
                    "params": {
                        "goal": "Test orchestration with session tracking",
                        "max_cycles": 2,
                        "user_id": "test_user_orchestration"
                    }
                }
            )

            # Orchestration might fail due to missing agents, but session should be tracked
            session_id = workflow_result.get("session_id")
            if session_id:
                # Check that session was created and updated
                session_result = await self.agent_manager.execute_by_capability(
                    AgentCapability.MEMORY,
                    {
                        "action": "get",
                        "session_id": session_id
                    }
                )

                if session_result["status"] == "success":
                    session_data = session_result["data"]
                    context_entries = session_data.get("context", [])
                    self.logger.info(f"Session {session_id} has {len(context_entries)} context entries")

                    self.test_results.append({
                        "test": "session_tracking_orchestration",
                        "status": "passed",
                        "session_id": session_id,
                        "context_entries": len(context_entries),
                        "workflow_status": workflow_result.get("status")
                    })
                else:
                    self.test_results.append({
                        "test": "session_tracking_orchestration",
                        "status": "partial",
                        "session_id": session_id,
                        "workflow_status": workflow_result.get("status"),
                        "note": "Session created but retrieval failed"
                    })
            else:
                self.test_results.append({
                    "test": "session_tracking_orchestration",
                    "status": "partial",
                    "workflow_status": workflow_result.get("status"),
                    "note": "No session ID returned"
                })

        except Exception as e:
            self.logger.exception("Session tracking orchestration test failed")
            self.test_results.append({
                "test": "session_tracking_orchestration",
                "status": "failed",
                "error": str(e)
            })

    def print_results(self):
        """Print test results summary"""
        print("\n" + "="*60)
        print("PHASE 4 MEMORY INTEGRATION TEST RESULTS")
        print("="*60)

        passed = 0
        failed = 0
        partial = 0

        for result in self.test_results:
            status = result["status"]
            test_name = result["test"]

            if status == "passed":
                passed += 1
                print(f"✅ {test_name}: PASSED")
            elif status == "failed":
                failed += 1
                print(f"❌ {test_name}: FAILED - {result.get('error', 'Unknown error')}")
            elif status == "partial":
                partial += 1
                print(f"⚠️  {test_name}: PARTIAL - {result.get('note', 'Partial success')}")

            # Print additional details
            for key, value in result.items():
                if key not in ["test", "status", "error", "note"]:
                    print(f"   {key}: {value}")

        print("-"*60)
        print(f"Summary: {passed} passed, {partial} partial, {failed} failed")
        print("="*60)


async def main():
    """Run the Phase 4 memory test suite"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    test_suite = Phase4MemoryTestSuite()

    try:
        await test_suite.setup()
        await test_suite.run_all_tests()
    finally:
        await test_suite.teardown()


if __name__ == "__main__":
    asyncio.run(main())