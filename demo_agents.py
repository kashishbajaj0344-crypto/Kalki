"""
Kalki v2.3 Agent System - Quick Demo
Demonstrates key agent capabilities
"""
import asyncio
import logging

# Suppress excessive logging for demo
logging.basicConfig(level=logging.WARNING)

from agents import AgentManager, EventBus
from agents.base_agent import AgentCapability
from agents.core import SearchAgent, PlannerAgent, ReasoningAgent, MemoryAgent
from agents.cognitive import CreativeAgent, MetaHypothesisAgent, OptimizationAgent
from agents.safety import EthicsAgent, RiskAssessmentAgent
from agents.multimodal import VisionAgent, AudioAgent, SensorFusionAgent


async def demo():
    """Run a comprehensive demo of the agent system"""
    
    print("\n" + "="*70)
    print("🚀 KALKI v2.3 - MULTI-PHASE AI AGENT SYSTEM DEMO")
    print("="*70)
    
    # Initialize system
    print("\n📦 Initializing agent system...")
    event_bus = EventBus()
    manager = AgentManager(event_bus)
    
    # Register agents
    agents = [
        SearchAgent(),
        PlannerAgent(),
        ReasoningAgent(),
        MemoryAgent(),
        CreativeAgent(),
        MetaHypothesisAgent(),
        OptimizationAgent(),
        EthicsAgent(),
        RiskAssessmentAgent(),
        VisionAgent(),
        AudioAgent(),
        SensorFusionAgent()
    ]
    
    print("   Registering agents...")
    for agent in agents:
        await manager.register_agent(agent)
        print(f"   ✓ {agent.name}")
    
    print(f"\n✅ System initialized with {len(agents)} agents")
    
    # Demo 1: Planning
    print("\n" + "-"*70)
    print("📋 DEMO 1: Planning Agent - Task Decomposition")
    print("-"*70)
    
    result = await manager.execute_by_capability(
        AgentCapability.PLANNING,
        {
            "action": "plan",
            "params": {"goal": "Build a sustainable AI system"}
        }
    )
    
    if result.get("status") == "success":
        print(f"\nGoal: {result['goal']}")
        print("\nExecution Plan:")
        for step in result['plan']:
            print(f"  {step['step']}. {step['description']}")
    
    # Demo 2: Creative Ideation
    print("\n" + "-"*70)
    print("💡 DEMO 2: Creative Agent - Idea Generation")
    print("-"*70)
    
    result = await manager.execute_by_capability(
        AgentCapability.CREATIVE_SYNTHESIS,
        {
            "action": "ideate",
            "params": {"topic": "quantum computing applications", "count": 3}
        }
    )
    
    if result.get("status") == "success":
        print(f"\nTopic: {result['topic']}")
        print(f"Novelty Score: {result['novelty_score']:.2f}")
        print("\nGenerated Ideas:")
        for idea in result['ideas']:
            print(f"  • {idea}")
    
    # Demo 3: Idea Fusion
    print("\n" + "-"*70)
    print("🔬 DEMO 3: Creative Agent - Idea Fusion")
    print("-"*70)
    
    result = await manager.execute_by_capability(
        AgentCapability.IDEA_FUSION,
        {
            "action": "fuse",
            "params": {"concepts": ["AI", "blockchain", "renewable energy"]}
        }
    )
    
    if result.get("status") == "success":
        print(f"\nConcepts: {' + '.join(result['concepts'])}")
        print(f"Fusion: {result['fusion']}")
        print(f"Creativity Score: {result['creativity_score']:.2f}")
    
    # Demo 4: Ethics Review
    print("\n" + "-"*70)
    print("⚖️  DEMO 4: Ethics Agent - Safety Verification")
    print("-"*70)
    
    result = await manager.execute_by_capability(
        AgentCapability.ETHICS,
        {
            "action": "review",
            "params": {
                "action_description": "Deploy autonomous decision-making system for healthcare"
            }
        }
    )
    
    if result.get("status") == "success":
        print(f"\nAction: {result['action']}")
        print(f"Ethical: {'✓ Yes' if result['is_ethical'] else '✗ No'}")
        if result.get('concerns'):
            print("\nConcerns:")
            for concern in result['concerns']:
                print(f"  ⚠️  {concern}")
        if result.get('recommendations'):
            print("\nRecommendations:")
            for rec in result['recommendations']:
                print(f"  • {rec}")
    
    # Demo 5: Risk Assessment
    print("\n" + "-"*70)
    print("🛡️  DEMO 5: Risk Assessment Agent - System Evaluation")
    print("-"*70)
    
    result = await manager.execute_by_capability(
        AgentCapability.RISK_ASSESSMENT,
        {
            "action": "assess",
            "params": {
                "scenario": "Deploying AI in production",
                "factors": ["data quality", "model reliability", "user impact"]
            }
        }
    )
    
    if result.get("status") == "success":
        print(f"\nScenario: {result['scenario']}")
        print(f"Risk Score: {result['risk_score']:.2f}")
        print(f"Risk Level: {result['risk_level'].upper()}")
        print(f"Mitigation Required: {'Yes' if result['mitigation_required'] else 'No'}")
    
    # Demo 6: Meta-Reasoning
    print("\n" + "-"*70)
    print("🧠 DEMO 6: Meta-Hypothesis Agent - Hypothesis Generation")
    print("-"*70)
    
    result = await manager.execute_by_capability(
        AgentCapability.META_REASONING,
        {
            "action": "hypothesize",
            "params": {"topic": "emergent AI capabilities"}
        }
    )
    
    if result.get("status") == "success":
        print(f"\nTopic: {result['topic']}")
        print(f"Confidence: {result['confidence']:.2f}")
        print("\nGenerated Hypotheses:")
        for hyp in result['hypotheses']:
            print(f"  • {hyp}")
    
    # Demo 7: Memory System
    print("\n" + "-"*70)
    print("💾 DEMO 7: Memory Agent - Knowledge Storage")
    print("-"*70)
    
    # Store memory
    store_result = await manager.execute_by_capability(
        AgentCapability.MEMORY,
        {
            "action": "store",
            "params": {
                "memory_type": "semantic",
                "key": "demo_session",
                "value": "Kalki v2.3 demo completed successfully"
            }
        }
    )
    print(f"✓ Memory stored: {store_result.get('stored')}")
    
    # Retrieve memory
    retrieve_result = await manager.execute_by_capability(
        AgentCapability.MEMORY,
        {
            "action": "retrieve",
            "params": {
                "memory_type": "semantic",
                "key": "demo_session"
            }
        }
    )
    if retrieve_result.get("status") == "success":
        print(f"✓ Memory retrieved: {retrieve_result.get('value')}")
    
    # Demo 8: Sensor Fusion
    print("\n" + "-"*70)
    print("🎯 DEMO 8: Sensor Fusion Agent - Multi-Modal Integration")
    print("-"*70)
    
    result = await manager.execute_by_capability(
        AgentCapability.SENSOR_FUSION,
        {
            "action": "fuse",
            "params": {
                "sensor_data": {"vision": {}, "audio": {}},
                "modalities": ["vision", "audio"]
            }
        }
    )
    
    if result.get("status") == "success":
        fusion = result['fusion']
        print(f"\nModalities Fused: {', '.join(fusion['modalities_fused'])}")
        print(f"Combined Confidence: {fusion['combined_confidence']:.2f}")
        print("\nInsights:")
        for insight in fusion['insights']:
            print(f"  • {insight}")
    
    # System Statistics
    print("\n" + "-"*70)
    print("📊 SYSTEM STATISTICS")
    print("-"*70)
    
    stats = manager.get_system_stats()
    print(f"\nTotal Agents: {stats['total_agents']}")
    print(f"Tasks Executed: {stats['total_tasks_executed']}")
    print(f"Errors: {stats['total_errors']}")
    print(f"\nCapabilities Available: {len(stats['capabilities'])}")
    
    # Health Check
    print("\n" + "-"*70)
    print("🏥 AGENT HEALTH STATUS")
    print("-"*70)
    
    health = await manager.health_check_all()
    print()
    for agent_name, status in health.items():
        emoji = "✓" if status['status'] == 'ready' else "✗"
        print(f"{emoji} {agent_name:30s} | Status: {status['status']:12s} | Tasks: {status['task_count']:3d}")
    
    # Cleanup
    print("\n" + "-"*70)
    print("🔒 Shutting down system...")
    print("-"*70)
    await manager.shutdown_all()
    
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE - All agent capabilities demonstrated!")
    print("="*70)
    print("\nTo explore more:")
    print("  • Run: python kalki_agents.py (interactive mode)")
    print("  • Read: README_AGENTS.md (full documentation)")
    print("  • Test: python test_agent_system.py (verification)")
    print("="*70 + "\n")


if __name__ == "__main__":
    asyncio.run(demo())
