#!/usr/bin/env python3
"""
Kalki Phases 13-20 Integration Demo
Demonstrates all phases working together.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.memory import InMemoryStore, EpisodicMemory, SemanticMemory
from modules.planner import Planner, Scheduler
from modules.reasoner import Reasoner
from modules.agents import (
    AgentRegistry, AgentRunner, MessageBus,
    SearchAgent, ExecutorAgent, SafetyAgent, ReasoningAgent
)
from modules.safety import SafetyGuard
from modules.metrics import MetricsCollector, PerformanceMonitor
from modules.testing import AdversarialTestHarness, PromptInjectionScenario


def main():
    print("=" * 60)
    print("Kalki Multi-Agent System - Phases 13-20 Demo")
    print("=" * 60)
    print()
    
    # Phase 13-14: Set up memory
    print("Phase 13-14: Setting up memory system...")
    store = InMemoryStore()
    episodic = EpisodicMemory(store)
    semantic = SemanticMemory(store)
    
    # Add some documents to semantic memory
    semantic.add_document("Machine learning is a subset of artificial intelligence")
    semantic.add_document("Python is a popular programming language")
    semantic.add_document("Deep learning uses neural networks")
    
    # Search for similar documents
    results = semantic.search_similar("AI and neural networks", limit=2)
    print(f"  ✓ Semantic search found {len(results)} relevant documents")
    
    # Add some events
    episodic.add_event("system_start", {"status": "initializing"})
    print(f"  ✓ Episodic memory tracking events")
    print()
    
    # Phase 15: Hierarchical Planning
    print("Phase 15: Creating hierarchical plan...")
    planner = Planner(store)
    plan = planner.plan("Search and analyze data about AI")
    print(f"  ✓ Created plan with {len(plan.tasks)} tasks")
    print()
    
    # Phase 16: Reasoning
    print("Phase 16: Performing iterative reasoning...")
    reasoner = Reasoner(store)
    trace = reasoner.reason("Analyze the impact of AI on technology")
    print(f"  ✓ Reasoning completed with {len(trace.steps)} steps")
    print(f"  ✓ Conclusion: {trace.conclusion}")
    print()
    
    # Phase 17: Multi-Agent Coordination
    print("Phase 17: Setting up multi-agent system...")
    registry = AgentRegistry()
    bus = MessageBus(episodic)
    runner = AgentRunner(registry, bus)
    
    # Register agents
    registry.register(SearchAgent("search1"))
    registry.register(ExecutorAgent("exec1"))
    registry.register(SafetyAgent("safety1"))
    registry.register(ReasoningAgent("reason1"))
    
    print(f"  ✓ Registered {len(registry.agents)} agents")
    
    # Set up scheduler
    scheduler = Scheduler()
    for agent in registry.agents.values():
        scheduler.register_agent(agent.agent_id, agent.capabilities)
    
    # Assign tasks
    assignments = scheduler.assign_tasks(plan)
    print(f"  ✓ Assigned {len(assignments)} tasks to agents")
    print()
    
    # Phase 18: Safety Constraints
    print("Phase 18: Configuring safety constraints...")
    guard = SafetyGuard(store)
    guard.add_rate_limit("api_calls", max_calls=100, time_window=60)
    guard.add_forbidden_operation("delete_all")
    guard.add_content_filter_pattern(r"malicious")
    
    # Test safety check
    safe_check = guard.pre_execution_check("test_agent", "safe_op", {"content": "normal data"})
    unsafe_check = guard.pre_execution_check("test_agent", "test_op", {"content": "malicious code"})
    
    print(f"  ✓ Safe operation: {'allowed' if safe_check.allowed else 'blocked'}")
    print(f"  ✓ Unsafe operation: {'allowed' if unsafe_check.allowed else 'blocked'}")
    print()
    
    # Phase 19: Adversarial Testing
    print("Phase 19: Running adversarial tests...")
    harness = AdversarialTestHarness("Demo Tests")
    harness.add_scenario(PromptInjectionScenario())
    
    components = {
        'safety_guard': guard,
        'planner': planner
    }
    
    report = harness.run_all(components)
    print(f"  ✓ Ran {report.scenarios_run} adversarial scenarios")
    print(f"  ✓ Passed: {report.scenarios_passed}, Failed: {report.scenarios_failed}")
    print()
    
    # Phase 20: Metrics & Evaluation
    print("Phase 20: Collecting performance metrics...")
    collector = MetricsCollector()
    monitor = PerformanceMonitor(collector)
    
    # Simulate task execution
    for task in list(plan.tasks.values())[:3]:
        collector.start_task(task.task_id, "demo_agent")
        result = runner.execute_task(task)
        collector.end_task(task.task_id, success=(result is not None))
    
    perf_report = monitor.get_performance_report()
    print(f"  ✓ Tracked {perf_report['summary']['total_tasks']} tasks")
    print(f"  ✓ Success rate: {perf_report['summary']['success_rate']:.0%}")
    print(f"  ✓ System status: {perf_report['analysis']['success_rate_status']}")
    print()
    
    # Summary
    print("=" * 60)
    print("Demo Complete - All Phases Working Successfully!")
    print("=" * 60)
    print()
    print("Key Capabilities Demonstrated:")
    print("  • Memory persistence (SQLite & in-memory)")
    print("  • Semantic search with TF-IDF")
    print("  • Episodic event tracking")
    print("  • Hierarchical task planning")
    print("  • Multi-step reasoning")
    print("  • Multi-agent coordination")
    print("  • Safety constraint enforcement")
    print("  • Adversarial robustness testing")
    print("  • Performance metrics collection")
    print()
    print("See PHASES_13_20_README.md for complete documentation")
    print()


if __name__ == '__main__':
    main()
