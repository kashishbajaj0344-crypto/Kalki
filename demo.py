#!/usr/bin/env python3
"""
Kalki Demo — Showcase of the Complete 20-Phase AI Framework
==========================================================

This demo demonstrates the full capabilities of Kalki v3.0,
showcasing real algorithms across all implemented phases.
"""

import asyncio
import time
from kalki_complete import KalkiOrchestrator

async def demo_phase_14():
    """Demonstrate Phase 14: Quantum & Predictive capabilities"""
    print("\n" + "="*60)
    print("🎯 PHASE 14 DEMO: Quantum & Predictive Discovery")
    print("="*60)

    orchestrator = KalkiOrchestrator()
    await orchestrator.initialize_system()

    # 1. Quantum Optimization
    print("\n⚛️ Quantum Optimization Demo")
    quantum_agent = next((a for a in orchestrator.phase_agents.get('quantum_predictive', [])
                         if a.name == "QuantumReasoningAgent"), None)

    if quantum_agent:
        task = {
            "action": "optimize_combination",
            "params": {
                "problem": "portfolio_optimization",
                "variables": ["stocks", "bonds", "crypto"],
                "constraints": {"stocks": 60, "bonds": 30, "crypto": 10},
                "objective": "maximize_returns"
            }
        }
        result = await quantum_agent.execute(task)
        print(f"   Optimization Result: {result.get('status', 'unknown')}")

    # 2. Technology Prediction
    print("\n🔮 Technology Prediction Demo")
    predictive_agent = next((a for a in orchestrator.phase_agents.get('quantum_predictive', [])
                           if a.name == "PredictiveDiscoveryAgent"), None)

    if predictive_agent:
        task = {
            "action": "forecast_technology_trend",
            "params": {
                "technology": "artificial_intelligence",
                "historical_data": [
                    {"year": 2010, "adoption": 0.05},
                    {"year": 2015, "adoption": 0.15},
                    {"year": 2020, "adoption": 0.35},
                    {"year": 2023, "adoption": 0.65}
                ],
                "forecast_years": 5
            }
        }
        result = await predictive_agent.execute(task)
        print(f"   Prediction Result: {result.get('status', 'unknown')}")

    # 3. Intention Impact Analysis
    print("\n🎯 Intention Impact Analysis Demo")
    impact_agent = next((a for a in orchestrator.phase_agents.get('quantum_predictive', [])
                        if a.name == "IntentionImpactAnalyzer"), None)

    if impact_agent:
        task = {
            "action": "analyze_intention",
            "params": {
                "intention": {
                    "description": "Deploy autonomous vehicles city-wide",
                    "actor": "municipal_government",
                    "domains_affected": ["technology", "infrastructure", "social"],
                    "initial_impact": 0.8,
                    "probability": 0.9
                }
            }
        }
        result = await impact_agent.execute(task)
        if result.get("status") == "success":
            analysis = result.get("impact_analysis", {})
            print(f"   Risk Level: {analysis.get('overall_risk', 'unknown')}")
            print(f"   Unintended Consequences: {analysis.get('unintended_consequences', 0)}")

    await orchestrator.shutdown()

async def demo_full_system():
    """Demonstrate the complete Kalki system"""
    print("\n" + "="*60)
    print("🚀 KALKI v3.0 COMPLETE SYSTEM DEMO")
    print("="*60)

    orchestrator = KalkiOrchestrator()
    await orchestrator.initialize_system()

    # Show system status
    status = await orchestrator.get_system_status()
    print("\n🖥️ System Status:")
    print(f"   Status: {status['system_status']}")
    print(f"   Active Phases: {status['phases_active']}")
    print(f"   Total Agents: {status['total_agents']}")

    # Process sample queries
    sample_queries = [
        "What are the latest developments in quantum computing?",
        "Analyze the potential impact of implementing a four-day workweek globally",
        "How might artificial intelligence evolve over the next decade?",
        "What are the ethical considerations of autonomous weapons systems?"
    ]

    print("\n🔍 Processing Sample Queries:")
    for i, query in enumerate(sample_queries, 1):
        print(f"\n   {i}. {query}")
        start_time = time.time()
        result = await orchestrator.process_user_query(query)
        elapsed = time.time() - start_time
        print(f"   ⏱️  Response time: {elapsed:.2f}s")
    await orchestrator.shutdown()

async def demo_interactive():
    """Interactive demo mode"""
    print("\n" + "="*60)
    print("🎮 KALKI INTERACTIVE DEMO")
    print("="*60)
    print("Type your questions or 'exit' to quit")
    print("Try queries like:")
    print("  • What is quantum entanglement?")
    print("  • Predict AI adoption in healthcare")
    print("  • Analyze impact of universal basic income")
    print("-" * 60)

    orchestrator = KalkiOrchestrator()
    await orchestrator.initialize_system()

    try:
        while True:
            query = input("\nkalki> ").strip()
            if query.lower() in ['exit', 'quit', 'q']:
                break
            if query:
                print("Processing...")
                result = await orchestrator.process_user_query(query)
                if result.get("status") == "success":
                    print("✅ Complete")
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
    except KeyboardInterrupt:
        print("\nGoodbye!")
    finally:
        await orchestrator.shutdown()

async def main():
    """Main demo function"""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║        ██╗  ██╗ █████╗ ██╗     ██╗  ██╗██╗    ██████╗  ██████╗              ║
║        ██║ ██╔╝██╔══██╗██║     ██║ ██╔╝██║    ██╔══██╗██╔═══██╗             ║
║        █████╔╝ ███████║██║     █████╔╝ ██║    ██████╔╝██║   ██║             ║
║        ██╔═██╗ ██╔══██║██║     ██╔═██╗ ██║    ██╔══██╗██║   ██║             ║
║        ██║  ██╗██║  ██║███████╗██║  ██╗██║    ██████╔╝╚██████╔╝             ║
║        ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝    ╚═════╝  ╚═════╝              ║
║                                                                              ║
║                      COMPLETE 20-PHASE AI FRAMEWORK                          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    import argparse
    parser = argparse.ArgumentParser(description="Kalki Demo")
    parser.add_argument('--mode', choices=['phase14', 'full', 'interactive'],
                       default='full', help='Demo mode')
    args = parser.parse_args()

    if args.mode == 'phase14':
        await demo_phase_14()
    elif args.mode == 'full':
        await demo_full_system()
    elif args.mode == 'interactive':
        await demo_interactive()

if __name__ == "__main__":
    asyncio.run(main())