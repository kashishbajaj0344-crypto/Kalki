#!/usr/bin/env python3
"""
Generate performance metrics report.
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.metrics import MetricsCollector, PerformanceMonitor
from modules.agents import AgentRegistry, AgentRunner, SearchAgent, ExecutorAgent, MessageBus
from modules.planner import Task
from modules.memory import InMemoryStore, EpisodicMemory


def main():
    parser = argparse.ArgumentParser(description='Generate performance metrics')
    parser.add_argument('--output', default='reports/metrics_report.json',
                       help='Output report path')
    args = parser.parse_args()
    
    # Set up system
    store = InMemoryStore()
    episodic = EpisodicMemory(store)
    registry = AgentRegistry()
    bus = MessageBus(episodic)
    runner = AgentRunner(registry, bus)
    
    # Register agents
    registry.register(SearchAgent("search1"))
    registry.register(ExecutorAgent("exec1"))
    
    # Set up metrics
    collector = MetricsCollector()
    monitor = PerformanceMonitor(collector)
    
    # Simulate some task executions
    print("Simulating task executions...")
    
    tasks = [
        Task(task_id="t1", description="Search for data", required_capabilities={"search"}),
        Task(task_id="t2", description="Execute analysis", required_capabilities={"execution"}),
        Task(task_id="t3", description="Process results", required_capabilities={"execution"}),
    ]
    
    for task in tasks:
        collector.start_task(task.task_id, "test_agent")
        
        # Simulate execution
        result = runner.execute_task(task)
        
        collector.end_task(
            task.task_id,
            success=result is not None and result.success,
            error=None if (result and result.success) else "Task failed"
        )
    
    # Generate report
    report = monitor.get_performance_report()
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\nPerformance Metrics:")
    print(f"  Total Tasks: {report['summary']['total_tasks']}")
    print(f"  Success Rate: {report['summary']['success_rate']:.2%}")
    print(f"  Average Latency: {report['summary']['average_latency']:.4f}s")
    print(f"\nAnalysis:")
    for key, value in report['analysis'].items():
        print(f"  {key}: {value}")
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    print(f"\nReport saved to: {args.output}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
