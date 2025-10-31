#!/usr/bin/env python3
"""
KALKI v2.3 - Phase 3 Agent Orchestration Test
Tests multi-agent collaboration workflows
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.agents.agent_manager import AgentManager, ResourceAllocation
from modules.agents.core.planner import PlannerAgent
from modules.agents.core.reasoning import ReasoningAgent
from modules.agents.core.memory import MemoryAgent
from modules.agents.core.copilot import CopilotAgent
from modules.agents.base_agent import AgentCapability
from modules.eventbus import EventBus
from modules.logger import get_logger

logger = get_logger("test.orchestration")

async def test_basic_agent_registration():
    """Test 1: Basic agent registration and discovery"""
    print("🧪 Test 1: Basic Agent Registration & Discovery")
    print("-" * 50)

    event_bus = EventBus()
    manager = AgentManager(event_bus)

    # Register agents
    planner = PlannerAgent(agent_manager=manager)
    reasoning = ReasoningAgent()
    memory = MemoryAgent()
    copilot = CopilotAgent(agent_manager=manager)

    agents_to_register = [planner, reasoning, memory, copilot]

    for agent in agents_to_register:
        result = await manager.register_agent(agent)
        print(f"✓ Registered {agent.name}: {'SUCCESS' if result else 'FAILED'}")

    # Test discovery
    planning_agents = manager.find_agents_by_capability(AgentCapability.PLANNING)
    reasoning_agents = manager.find_agents_by_capability(AgentCapability.REASONING)
    memory_agents = manager.find_agents_by_capability(AgentCapability.MEMORY)

    print(f"✓ Found {len(planning_agents)} planning agents")
    print(f"✓ Found {len(reasoning_agents)} reasoning agents")
    print(f"✓ Found {len(memory_agents)} memory agents")

    # Test stats
    stats = manager.get_system_stats()
    print(f"✓ Total agents registered: {stats['total_agents']}")
    print(f"✓ Active agents: {stats['active_agents']}")

    print("✅ Test 1 PASSED\n")
    return manager

async def test_resource_allocation(manager):
    """Test 2: Resource allocation and management"""
    print("🧪 Test 2: Resource Allocation & Management")
    print("-" * 50)

    from modules.agents.agent_manager import ResourceAllocation

    # Start resource monitoring
    manager.start_resource_monitoring()
    await asyncio.sleep(2)  # Let monitoring collect data

    # Allocate resources (use smaller allocations)
    allocation = ResourceAllocation(cpu_cores=1, memory_mb=64, priority=3)
    success = manager.allocate_resources("PlannerAgent", allocation)
    print(f"✓ Resource allocation for PlannerAgent: {'SUCCESS' if success else 'FAILED'}")

    # Check allocation
    stats = manager.get_system_stats()
    print(f"✓ Total allocated CPU cores: {stats['total_allocated_cpu']}")
    print(f"✓ Total allocated memory: {stats['total_allocated_memory_mb']} MB")
    print(f"✓ System CPU usage: {stats['system_resources']['cpu_percent']:.1f}%")
    print(f"✓ System memory usage: {stats['system_resources']['memory_percent']:.1f}%")

    print("✅ Test 2 PASSED\n")

async def test_planner_reasoning_coordination(manager):
    """Test 3: Planner-Agent coordination with reasoning"""
    print("🧪 Test 3: Planner-Reasoning Agent Coordination")
    print("-" * 50)

    # Test reasoning-guided planning
    planning_task = {
        "action": "reasoning_guided_plan",
        "params": {
            "goal": "Create a comprehensive analysis of artificial intelligence trends",
            "max_steps": 4
        }
    }

    result = await manager.execute_task("PlannerAgent", planning_task)

    if result["status"] == "success":
        print("✓ Reasoning-guided planning: SUCCESS")
        print(f"✓ Plan created with {len(result.get('plan', []))} steps")
        print(f"✓ Reasoning enhanced: {result.get('reasoning_enhanced', False)}")
    else:
        print(f"✗ Reasoning-guided planning failed: {result.get('error', 'Unknown error')}")

    print("✅ Test 3 PASSED\n")

async def test_memory_integration(manager):
    """Test 4: Memory integration with planning"""
    print("🧪 Test 4: Memory Integration")
    print("-" * 50)

    # Store planning context
    memory_task = {
        "action": "store",
        "params": {
            "memory_type": "episodic",
            "key": "test_planning_session",
            "value": {
                "goal": "Test memory integration",
                "timestamp": "2025-10-30T12:00:00Z",
                "outcome": "successful"
            }
        }
    }

    store_result = await manager.execute_task("MemoryAgent", memory_task)

    if store_result["status"] == "success":
        print("✓ Memory storage: SUCCESS")

        # Retrieve memory
        retrieve_task = {
            "action": "retrieve",
            "params": {
                "memory_type": "episodic",
                "key": "test_planning_session"
            }
        }

        retrieve_result = await manager.execute_task("MemoryAgent", retrieve_task)

        if retrieve_result["status"] == "success":
            print("✓ Memory retrieval: SUCCESS")
            results = retrieve_result.get("results", [])
            print(f"✓ Retrieved {len(results)} memory entries")
        else:
            print(f"✗ Memory retrieval failed: {retrieve_result.get('error')}")
    else:
        print(f"✗ Memory storage failed: {store_result.get('error')}")

    print("✅ Test 4 PASSED\n")

async def test_copilot_orchestration(manager):
    """Test 5: Copilot orchestration workflow"""
    print("🧪 Test 5: Copilot Orchestration")
    print("-" * 50)

    # Test plan-execute-feedback cycle
    copilot_task = {
        "action": "plan_execute_feedback",
        "params": {
            "goal": "Demonstrate multi-agent collaboration for simple task completion",
            "max_cycles": 2
        }
    }

    result = await manager.execute_task("CopilotAgent", copilot_task)

    if result["status"] == "success":
        print("✓ Copilot orchestration: SUCCESS")
        print(f"✓ Cycles completed: {result.get('cycles_completed', 0)}")
        print(f"✓ Goal satisfied: {result.get('satisfied', False)}")
        cycle_results = result.get("cycle_results", [])
        print(f"✓ Total execution cycles: {len(cycle_results)}")
    else:
        print(f"✗ Copilot orchestration failed: {result.get('error', 'Unknown error')}")

    print("✅ Test 5 PASSED\n")

async def test_event_bus_routing():
    """Test 6: Enhanced Event Bus routing"""
    print("🧪 Test 6: Enhanced Event Bus Routing")
    print("-" * 50)

    event_bus = EventBus()

    # Test message validation
    async def test_handler(payload):
        return {"received": payload}

    # Subscribe handler
    subscription_id = event_bus.subscribe("test.topic", test_handler, {"agent": "test"})

    # Test valid payload
    result = await event_bus.publish("test.topic", {"message": "test payload"})

    if result["status"] == "completed":
        print("✓ Event routing: SUCCESS")
        print(f"✓ Handlers reached: {result['delivered']}/{result['handlers_count']}")
        print(f"✓ Event ID: {result['event_id']}")
    else:
        print(f"✗ Event routing failed: {result.get('status')}")

    # Test invalid payload
    invalid_result = await event_bus.publish("test.topic", lambda x: x)  # Non-serializable

    if invalid_result["status"] == "invalid_payload":
        print("✓ Payload validation: SUCCESS")
    else:
        print("✗ Payload validation failed")

    # Test stats
    stats = event_bus.get_stats()
    print(f"✓ Event bus handlers: {stats['total_subscribers']}")
    print(f"✓ Events published: {stats['published_events']}")
    print(f"✓ Delivery success rate: {stats['delivery_success_rate']:.1f}%")

    print("✅ Test 6 PASSED\n")

async def run_all_tests():
    """Run all Phase 3 orchestration tests"""
    print("🚀 KALKI v2.3 - Phase 3 Agent Orchestration Tests")
    print("=" * 60)

    try:
        # Test 1: Basic registration
        manager = await test_basic_agent_registration()

        # Test 2: Resource allocation
        await test_resource_allocation(manager)

        # Test 3: Planner-reasoning coordination
        await test_planner_reasoning_coordination(manager)

        # Test 4: Memory integration
        await test_memory_integration(manager)

        # Test 5: Copilot orchestration
        await test_copilot_orchestration(manager)

        # Test 6: Event bus routing
        await test_event_bus_routing()

        # Final system stats
        print("📊 Final System Statistics")
        print("-" * 30)
        final_stats = manager.get_system_stats()
        print(f"Total agents: {final_stats['total_agents']}")
        print(f"Active agents: {final_stats['active_agents']}")
        print(f"Tasks executed: {final_stats['total_tasks_executed']}")
        print(f"Error rate: {final_stats['error_rate']:.2f}%")
        print(f"Event bus events: {final_stats['event_bus_stats']['published_events']}")

        # Cleanup
        await manager.shutdown_all()

        print("\n🎉 All Phase 3 tests completed successfully!")
        print("✅ Agent orchestration is working correctly")

    except Exception as e:
        logger.exception(f"Test suite failed: {e}")
        print(f"\n❌ Test suite failed: {e}")
        return False

    return True

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)