"""
Kalki Metrics Module - Phase 20 Integration Demo
Demonstrates comprehensive metrics collection and evaluation.
"""

import time
import logging
from datetime import datetime
from modules.metrics import (
    MetricsCollector,
    PerformanceMonitor,
    MetricsAnalyzer,
    AdaptiveController,
    integrate_with_testing,
    create_visualization_data
)

# Try to import testing and memory modules for full integration
try:
    from modules.testing.harness import AdversarialTestHarness
    TESTING_AVAILABLE = True
except ImportError:
    TESTING_AVAILABLE = False

try:
    from modules.agents.memory.sqlite_store import SQLiteMemoryStore
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('Kalki.Demo')


def simulate_agent_tasks(collector: MetricsCollector, num_tasks: int = 20):
    """Simulate various agent tasks with different characteristics."""
    logger.info(f"🔄 Simulating {num_tasks} agent tasks...")

    task_types = ["ingestion", "planning", "dialogue", "safety_check", "memory_query"]
    agents = ["CognitiveAgent", "SafetyAgent", "MemoryAgent", "PlanningAgent"]

    for i in range(num_tasks):
        task_id = f"task_{i:03d}"
        agent_id = agents[i % len(agents)]
        task_type = task_types[i % len(task_types)]

        # Start task with instrumentation
        collector.start_task(task_id, agent_id, task_type)

        # Simulate task execution with varying characteristics
        execution_time = 0.1 + (i % 5) * 0.2  # 0.1s to 1.0s
        time.sleep(execution_time / 10)  # Shorter sleep for demo

        # Simulate memory lookups (some tasks need more memory access)
        memory_lookups = (i % 3) + 1
        for _ in range(memory_lookups):
            collector.record_memory_lookup(task_id)

        # Simulate retries for some tasks
        retries = 0 if i % 4 != 0 else (i % 3) + 1
        for _ in range(retries):
            collector.record_retry(task_id)

        # Determine success (most tasks succeed, some fail)
        success = i % 7 != 0  # ~85% success rate
        error = "Simulated task failure" if not success else None

        # End task with cognitive metrics
        tokens_in = 100 + (i * 50)
        tokens_out = 80 + (i * 30)
        context_switches = i % 4
        attention_weight = 0.3 + (i % 5) * 0.1

        collector.end_task(
            task_id=task_id,
            success=success,
            error=error,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            context_switches=context_switches,
            attention_weight=attention_weight
        )

        if (i + 1) % 5 == 0:
            logger.info(f"  ✅ Completed {i + 1}/{num_tasks} tasks")


def run_metrics_demo():
    """Run comprehensive metrics collection and analysis demo."""
    logger.info("🚀 Starting Kalki Metrics Module Phase 20 Demo")
    logger.info("=" * 60)

    # Initialize components
    memory_store = SQLiteMemoryStore() if MEMORY_AVAILABLE else None
    collector = MetricsCollector(memory_store=memory_store)
    monitor = PerformanceMonitor(collector)
    analyzer = MetricsAnalyzer(collector)
    controller = AdaptiveController(collector)

    logger.info("✅ Initialized metrics components")

    # Simulate multiple rounds of tasks to build historical data
    for round_num in range(3):
        logger.info(f"\n📊 Round {round_num + 1}/3 - Simulating agent workload")

        # Simulate agent tasks
        simulate_agent_tasks(collector, num_tasks=15)

        # Generate performance report
        report = monitor.get_performance_report()
        logger.info("📈 Performance Report Generated")

        # Display key metrics
        summary = report['summary']
        logger.info(f"   📊 Tasks: {summary['total_tasks']} | Success: {summary['success_rate']:.1%}")
        logger.info(f"   ⏱️  Latency: {summary['average_latency']:.2f}s | CPU: {summary['system_cpu_percent']:.1f}%")
        logger.info(f"   🧠 Tokens: {summary['total_tokens_in']:,} in, {summary['total_tokens_out']:,} out")

        # Run CI checks
        ci_status = report['ci_status']
        ci_passed = ci_status['overall_pass']
        logger.info(f"   ✅ CI Status: {'PASS' if ci_passed else 'FAIL'} ({ci_status['summary']['passed_checks']}/{ci_status['summary']['total_checks']} checks)")

        # Show recommendations
        recommendations = report['recommendations']
        if recommendations and recommendations[0] != "System is performing well. No major issues detected.":
            logger.info(f"   💡 Recommendations: {len(recommendations)} items")
            for rec in recommendations[:2]:  # Show first 2
                logger.info(f"      • {rec}")

        # Run adaptive controller
        actions = controller.evaluate_and_adapt()
        if actions:
            logger.info(f"   🔧 Adaptive Actions: {', '.join(actions)}")

        # Save to memory if available
        if memory_store and collector.save_to_memory():
            logger.info("   💾 Metrics saved to memory store")

        time.sleep(1)  # Brief pause between rounds

    # Run trend analysis
    logger.info("\n📈 Running Historical Trend Analysis")
    trends = analyzer.trend_analysis()
    if trends['available']:
        logger.info(f"   📊 Analyzed {trends['time_range']}")
        for metric, data in trends['metrics'].items():
            if 'trend' in data and 'direction' not in data['trend']:
                continue
            direction = data.get('trend', {}).get('direction', 'unknown')
            logger.info(f"   {metric.replace('_', ' ').title()}: {direction}")
    else:
        logger.info(f"   ⚠️  {trends['message']}")

    # Create visualization data
    logger.info("\n📊 Creating Visualization Data")
    viz_data = create_visualization_data(report)
    logger.info(f"   📈 Generated {len(viz_data['charts'])} chart types")
    logger.info(f"   🎯 Summary: {viz_data['summary_cards']['total_tasks']} tasks, {viz_data['summary_cards']['success_rate']} success rate")

    if viz_data['alerts']:
        logger.info(f"   ⚠️  Alerts: {len(viz_data['alerts'])} active")
        for alert in viz_data['alerts'][:2]:
            logger.info(f"      • {alert['message']}")

    # Demonstrate testing integration if available
    if TESTING_AVAILABLE:
        logger.info("\n🧪 Integrating with Adversarial Testing")
        try:
            test_harness = AdversarialTestHarness()
            integrate_with_testing(collector, test_harness)
            logger.info("   ✅ Testing integration successful")
        except Exception as e:
            logger.warning(f"   ⚠️  Testing integration failed: {e}")

    logger.info("\n🎉 Kalki Metrics Module Phase 20 Demo Complete!")
    logger.info("=" * 60)

    # Final comprehensive report
    final_report = monitor.get_performance_report()
    logger.info("📋 Final System Status:")
    logger.info(f"   • Total Tasks: {final_report['summary']['total_tasks']}")
    logger.info(f"   • Success Rate: {final_report['summary']['success_rate']:.1%}")
    logger.info(f"   • Average Latency: {final_report['summary']['average_latency']:.2f}s")
    logger.info(f"   • Predictive Risk: {final_report['summary']['predictive_risk_score']:.2f}")
    logger.info(f"   • CI Status: {'PASS' if final_report['ci_status']['overall_pass'] else 'FAIL'}")

    return final_report


if __name__ == "__main__":
    try:
        demo_report = run_metrics_demo()
        logger.info("\n✅ Demo completed successfully!")
    except Exception as e:
        logger.error(f"\n❌ Demo failed: {e}")
        raise