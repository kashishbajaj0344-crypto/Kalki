"""
Test script to verify Kalki Agent System functionality
"""
import asyncio
import logging
import sys

# Setup minimal logging for test
logging.basicConfig(level=logging.WARNING)

async def test_agent_system():
    """Test basic agent system functionality"""
    print("="*60)
    print("KALKI v2.3 AGENT SYSTEM - VERIFICATION TEST")
    print("="*60)
    
    try:
        # Test imports
        print("\n[1/7] Testing imports...")
        from agents import AgentManager, EventBus
        from agents.base_agent import AgentCapability
        from agents.core import SearchAgent, PlannerAgent, MemoryAgent
        from agents.cognitive import CreativeAgent, MetaHypothesisAgent
        from agents.safety import EthicsAgent
        from agents.multimodal import VisionAgent
        print("✓ All imports successful")
        
        # Test event bus
        print("\n[2/7] Testing Event Bus...")
        event_bus = EventBus()
        test_events = []
        
        async def test_callback(event):
            test_events.append(event)
        
        await event_bus.subscribe("test.event", test_callback)
        await event_bus.publish("test.event", {"message": "hello"})
        await asyncio.sleep(0.1)  # Give time for async processing
        
        if len(test_events) > 0:
            print("✓ Event bus working")
        else:
            print("✗ Event bus failed")
            return False
        
        # Test agent manager
        print("\n[3/7] Testing Agent Manager...")
        manager = AgentManager(event_bus)
        print("✓ Agent manager created")
        
        # Test agent registration
        print("\n[4/7] Testing agent registration...")
        test_agent = PlannerAgent()
        success = await manager.register_agent(test_agent)
        
        if success:
            print(f"✓ Agent registered: {test_agent.name}")
        else:
            print("✗ Agent registration failed")
            return False
        
        # Test agent execution
        print("\n[5/7] Testing agent execution...")
        result = await manager.execute_task("PlannerAgent", {
            "action": "plan",
            "params": {"goal": "test goal"}
        })
        
        if result.get("status") == "success":
            print(f"✓ Agent executed successfully")
            print(f"  Plan has {len(result.get('plan', []))} steps")
        else:
            print(f"✗ Agent execution failed: {result.get('error')}")
            return False
        
        # Test capability-based execution
        print("\n[6/7] Testing capability-based execution...")
        result = await manager.execute_by_capability(
            AgentCapability.PLANNING,
            {"action": "plan", "params": {"goal": "another test"}}
        )
        
        if result.get("status") == "success":
            print("✓ Capability-based execution working")
        else:
            print("✗ Capability-based execution failed")
            return False
        
        # Test multiple agents
        print("\n[7/7] Testing multiple agent types...")
        agents_to_test = [
            MemoryAgent(),
            CreativeAgent(),
            EthicsAgent(),
            VisionAgent()
        ]
        
        registered_count = 0
        for agent in agents_to_test:
            if await manager.register_agent(agent):
                registered_count += 1
        
        print(f"✓ Registered {registered_count}/{len(agents_to_test)} additional agents")
        
        # Show final stats
        print("\n" + "="*60)
        print("VERIFICATION RESULTS")
        print("="*60)
        stats = manager.get_system_stats()
        print(f"Total Agents: {stats['total_agents']}")
        print(f"Total Tasks: {stats['total_tasks_executed']}")
        print(f"Capabilities: {len(stats['capabilities'])}")
        print("\nRegistered Agents:")
        for name in manager.agents.keys():
            print(f"  - {name}")
        
        # Cleanup
        await manager.shutdown_all()
        print("\n✓ All tests passed! System is functional.")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_agent_system())
    sys.exit(0 if success else 1)
