#!/usr/bin/env python3
"""
Kalki CLI — Command Line Interface for the Complete 20-Phase AI Framework
========================================================================

Usage:
    kalki query "What is the meaning of life?"
    kalki status
    kalki agents list
    kalki phase 14 status
    kalki quantum optimize --problem resource_allocation
    kalki predict --technology quantum_computing --years 5
    kalki analyze --intention "implement universal basic income"
    kalki shutdown

Features:
- Interactive query processing through all 20 phases
- Agent management and monitoring
- Phase-specific operations
- System health monitoring
- Batch processing capabilities
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.logging_config import setup_logging, get_logger
from kalki_complete import KalkiOrchestrator

logger = get_logger("Kalki.CLI")


class KalkiCLI:
    """Command Line Interface for the complete Kalki system"""

    def __init__(self):
        self.orchestrator: Optional[KalkiOrchestrator] = None

    async def initialize(self):
        """Initialize the Kalki system"""
        self.orchestrator = KalkiOrchestrator()
        success = await self.orchestrator.initialize_system()
        if not success:
            print("❌ Failed to initialize Kalki system")
            sys.exit(1)
        return self.orchestrator

    async def query(self, query: str, **kwargs):
        """Process a natural language query"""
        if not self.orchestrator:
            await self.initialize()

        print(f"🔍 Processing: {query}")
        result = await self.orchestrator.process_user_query(query)

        if result.get("status") == "success":
            response = result.get("response", "Query processed successfully")
            print(f"📝 Response: {response}")

            # Show additional metadata if available
            if "metadata" in result:
                print(f"📊 Metadata: {json.dumps(result['metadata'], indent=2)}")
        else:
            print(f"❌ Error: {result.get('error', 'Unknown error')}")

    async def status(self, **kwargs):
        """Show system status"""
        if not self.orchestrator:
            await self.initialize()

        status = await self.orchestrator.get_system_status()

        print("🖥️  Kalki System Status")
        print("=" * 50)
        print(f"Status: {status['system_status']}")
        print(f"Version: {status['version']}")
        print(f"Active Phases: {status['phases_active']}")
        print(f"Total Agents: {status['total_agents']}")
        print(f"Session ID: {status['session_id']}")
        print(f"Uptime: {status['uptime']}")

        # Show phase breakdown
        print("\n📊 Phase Status:")
        for phase, agents in self.orchestrator.phase_agents.items():
            phase_name = phase.replace('_', ' ').title()
            agent_count = len(agents)
            status_icon = "✅" if agent_count > 0 else "⏳"
            print(f"  {status_icon} {phase_name}: {agent_count} agents")

    async def agents_list(self, **kwargs):
        """List all active agents"""
        if not self.orchestrator:
            await self.initialize()

        print("🤖 Active Agents by Phase")
        print("=" * 50)

        for phase, agents in self.orchestrator.phase_agents.items():
            if agents:
                phase_name = phase.replace('_', ' ').title()
                print(f"\n📁 {phase_name} ({len(agents)} agents):")

                for agent in agents:
                    status = "🟢" if agent.status == "running" else "🟡" if agent.status == "idle" else "🔴"
                    capabilities = [cap.value for cap in agent.capabilities]
                    print(f"  {status} {agent.name}")
                    print(f"    Capabilities: {', '.join(capabilities)}")
                    print(f"    Description: {agent.description}")

    async def phase_status(self, phase_number: int, **kwargs):
        """Show status of a specific phase"""
        if not self.orchestrator:
            await self.initialize()

        phase_map = {
            1: "foundation", 2: "foundation",
            3: "core_cognition", 4: "core_cognition", 5: "core_cognition",
            6: "meta_cognition", 7: "meta_cognition",
            8: "distributed_simulation", 9: "distributed_simulation",
            10: "creativity_evolution", 11: "creativity_evolution",
            12: "safety_multimodal", 13: "safety_multimodal",
            14: "quantum_predictive",
            15: "emotional_intelligence", 16: "emotional_intelligence",
            17: "ar_vr_cognitive", 18: "ar_vr_cognitive",
            19: "autonomy_evolution", 20: "autonomy_evolution"
        }

        phase_key = phase_map.get(phase_number)
        if not phase_key:
            print(f"❌ Invalid phase number: {phase_number}")
            return

        agents = self.orchestrator.phase_agents.get(phase_key, [])
        phase_name = phase_key.replace('_', ' ').title()

        print(f"📊 Phase {phase_number}: {phase_name}")
        print("=" * 50)
        print(f"Agents: {len(agents)}")

        if agents:
            print("\nAgent Details:")
            for agent in agents:
                print(f"  • {agent.name}: {agent.description}")
        else:
            print("  (Phase not yet fully implemented)")

    async def quantum_optimize(self, problem: str, **kwargs):
        """Run quantum optimization"""
        if not self.orchestrator:
            await self.initialize()

        quantum_agent = next((a for a in self.orchestrator.phase_agents.get('quantum_predictive', [])
                            if a.name == "QuantumReasoningAgent"), None)

        if not quantum_agent:
            print("❌ QuantumReasoningAgent not available")
            return

        print(f"⚛️ Running quantum optimization for: {problem}")

        task = {
            "action": "optimize_combination",
            "params": {
                "problem": problem,
                "variables": kwargs.get("variables", ["x", "y", "z"]),
                "constraints": kwargs.get("constraints", {}),
                "objective": kwargs.get("objective", "maximize")
            }
        }

        result = await quantum_agent.execute(task)
        if result.get("status") == "success":
            print("✅ Optimization complete")
            print(f"Result: {result.get('result', 'N/A')}")
        else:
            print(f"❌ Optimization failed: {result.get('error', 'Unknown error')}")

    async def predict(self, technology: str, years: int, **kwargs):
        """Run technology prediction"""
        if not self.orchestrator:
            await self.initialize()

        predictive_agent = next((a for a in self.orchestrator.phase_agents.get('quantum_predictive', [])
                               if a.name == "PredictiveDiscoveryAgent"), None)

        if not predictive_agent:
            print("❌ PredictiveDiscoveryAgent not available")
            return

        print(f"🔮 Predicting {technology} adoption for {years} years")

        task = {
            "action": "forecast_technology_trend",
            "params": {
                "technology": technology,
                "forecast_years": years
            }
        }

        result = await predictive_agent.execute(task)
        if result.get("status") == "success":
            print("✅ Prediction complete")
            forecast = result.get("forecast", {})
            print(f"Trend: {forecast.get('trend', 'N/A')}")
            print(f"Confidence: {forecast.get('confidence', 'N/A')}")
        else:
            print(f"❌ Prediction failed: {result.get('error', 'Unknown error')}")

    async def analyze_intention(self, intention: str, **kwargs):
        """Analyze intention impact"""
        if not self.orchestrator:
            await self.initialize()

        impact_agent = next((a for a in self.orchestrator.phase_agents.get('quantum_predictive', [])
                           if a.name == "IntentionImpactAnalyzer"), None)

        if not impact_agent:
            print("❌ IntentionImpactAnalyzer not available")
            return

        print(f"🎯 Analyzing intention: {intention}")

        task = {
            "action": "analyze_intention",
            "params": {
                "intention": {
                    "description": intention,
                    "actor": "user",
                    "domains_affected": kwargs.get("domains", ["technology"]),
                    "initial_impact": kwargs.get("impact", 0.5),
                    "probability": kwargs.get("probability", 0.8)
                }
            }
        }

        result = await impact_agent.execute(task)
        if result.get("status") == "success":
            analysis = result.get("impact_analysis", {})
            print("✅ Impact analysis complete")
            print(f"Risk Level: {analysis.get('overall_risk', 'N/A')}")
            print(f"Unintended Consequences: {analysis.get('unintended_consequences', 0)}")
            print(f"Mitigation Suggestions: {len(analysis.get('mitigation_suggestions', []))}")
        else:
            print(f"❌ Analysis failed: {result.get('error', 'Unknown error')}")

    async def web_search(self, query: str, num_results: int = 5, provider: str = None, **kwargs):
        """Perform web search"""
        if not self.orchestrator:
            await self.initialize()

        web_search_agent = next((a for a in self.orchestrator.phase_agents.get('foundation', [])
                               if hasattr(a, 'name') and a.name == "WebSearchAgent"), None)

        if not web_search_agent:
            print("❌ WebSearchAgent not available - check API key configuration")
            return

        print(f"🌐 Searching web for: {query}")

        params = {
            "query": query,
            "max_results": num_results
        }

        if provider:
            params["provider"] = provider

        result = await web_search_agent.execute({
            "action": "search",
            "params": params
        })

        if result.get("status") == "success":
            print(f"✅ Found {result.get('total_results', 0)} results using {result.get('provider', 'unknown')}")
            print("\n📋 Results:")

            for i, item in enumerate(result.get("results", []), 1):
                print(f"\n{i}. {item.get('title', 'No title')}")
                print(f"   URL: {item.get('url', 'No URL')}")
                print(f"   {item.get('snippet', 'No description')[:200]}...")

        else:
            print(f"❌ Search failed: {result.get('error', 'Unknown error')}")

    async def web_research(self, topic: str, depth: str = "basic", **kwargs):
        """Perform comprehensive research"""
        if not self.orchestrator:
            await self.initialize()

        web_search_agent = next((a for a in self.orchestrator.phase_agents.get('foundation', [])
                               if hasattr(a, 'name') and a.name == "WebSearchAgent"), None)

        if not web_search_agent:
            print("❌ WebSearchAgent not available - check API key configuration")
            return

        print(f"🔬 Researching topic: {topic} (depth: {depth})")

        result = await web_search_agent.execute({
            "action": "research_topic",
            "params": {
                "topic": topic,
                "depth": depth
            }
        })

        if result.get("status") == "success":
            research = result.get("research", {})
            print(f"✅ Research complete - found {len(research.get('sources', []))} sources")
            print(f"📊 Summary: {research.get('summary', 'No summary available')}")

            if research.get("sources"):
                print("\n📋 Key Sources:")
                for i, source in enumerate(research["sources"][:5], 1):  # Show top 5
                    print(f"{i}. {source.get('title', 'No title')}")

        else:
            print(f"❌ Research failed: {result.get('error', 'Unknown error')}")

    async def web_fetch(self, url: str, max_length: int = 50000, **kwargs):
        """Fetch content from URL"""
        if not self.orchestrator:
            await self.initialize()

        web_search_agent = next((a for a in self.orchestrator.phase_agents.get('foundation', [])
                               if hasattr(a, 'name') and a.name == "WebSearchAgent"), None)

        if not web_search_agent:
            print("❌ WebSearchAgent not available - check API key configuration")
            return

        print(f"📥 Fetching content from: {url}")

        result = await web_search_agent.execute({
            "action": "fetch_url",
            "params": {
                "url": url,
                "max_length": max_length
            }
        })

        if result.get("status") == "success":
            print(f"✅ Content fetched successfully")
            print(f"📄 Title: {result.get('title', 'No title')}")
            print(f"📊 Content length: {result.get('content_length', 0)} characters")
            print(f"\n📝 Content preview:\n{result.get('content', '')[:500]}...")

        else:
            print(f"❌ Fetch failed: {result.get('error', 'Unknown error')}")

    async def shutdown(self, **kwargs):
        """Shutdown the system"""
        if self.orchestrator:
            await self.orchestrator.shutdown()
        print("👋 Kalki shutdown complete")
        sys.exit(0)


def create_parser():
    """Create the argument parser"""
    parser = argparse.ArgumentParser(
        description="Kalki CLI - The Complete 20-Phase AI Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  kalki query "What is quantum computing?"
  kalki status
  kalki agents list
  kalki phase 14 status
  kalki quantum optimize --problem resource_allocation
  kalki predict --technology ai --years 10
  kalki analyze --intention "implement flying cars"
  kalki web search "Call of Duty game mechanics" --results 3
  kalki web research "artificial intelligence trends" --depth comprehensive
  kalki web fetch "https://en.wikipedia.org/wiki/Call_of_Duty"
  kalki shutdown
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Query command
    query_parser = subparsers.add_parser('query', help='Process a natural language query')
    query_parser.add_argument('query', help='The query to process')

    # Status command
    subparsers.add_parser('status', help='Show system status')

    # Agents command
    agents_parser = subparsers.add_parser('agents', help='Agent management')
    agents_subparsers = agents_parser.add_subparsers(dest='agents_command')
    agents_subparsers.add_parser('list', help='List all agents')

    # Phase command
    phase_parser = subparsers.add_parser('phase', help='Phase-specific operations')
    phase_parser.add_argument('number', type=int, help='Phase number (1-20)')
    phase_parser.add_argument('action', choices=['status'], help='Action to perform')

    # Quantum command
    quantum_parser = subparsers.add_parser('quantum', help='Quantum operations')
    quantum_subparsers = quantum_parser.add_subparsers(dest='quantum_command')
    optimize_parser = quantum_subparsers.add_parser('optimize', help='Run quantum optimization')
    optimize_parser.add_argument('--problem', required=True, help='Optimization problem')
    optimize_parser.add_argument('--variables', nargs='+', help='Variables to optimize')
    optimize_parser.add_argument('--constraints', help='Constraints (JSON)')
    optimize_parser.add_argument('--objective', help='Objective function')

    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Technology prediction')
    predict_parser.add_argument('--technology', required=True, help='Technology to predict')
    predict_parser.add_argument('--years', type=int, required=True, help='Years to forecast')

    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Intention impact analysis')
    analyze_parser.add_argument('--intention', required=True, help='Intention to analyze')
    analyze_parser.add_argument('--domains', nargs='+', help='Affected domains')
    analyze_parser.add_argument('--impact', type=float, help='Initial impact (0-1)')
    analyze_parser.add_argument('--probability', type=float, help='Probability (0-1)')

    # Web search command
    web_parser = subparsers.add_parser('web', help='Web search and external data retrieval')
    web_subparsers = web_parser.add_subparsers(dest='web_command')

    # Web search subcommand
    search_parser = web_subparsers.add_parser('search', help='Search the web')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('--results', type=int, default=5, help='Number of results (default: 5)')
    search_parser.add_argument('--provider', choices=['google', 'bing', 'serpapi', 'duckduckgo'],
                              help='Search provider (auto-select if not specified)')

    # Web research subcommand
    research_parser = web_subparsers.add_parser('research', help='Comprehensive research on a topic')
    research_parser.add_argument('topic', help='Topic to research')
    research_parser.add_argument('--depth', choices=['basic', 'intermediate', 'comprehensive'],
                                default='basic', help='Research depth')

    # Web fetch subcommand
    fetch_parser = web_subparsers.add_parser('fetch', help='Fetch content from a URL')
    fetch_parser.add_argument('url', help='URL to fetch')
    fetch_parser.add_argument('--max-length', type=int, default=50000,
                             help='Maximum content length (default: 50000)')

    # Shutdown command
    subparsers.add_parser('shutdown', help='Shutdown the system')

    return parser


async def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Set up logging
    setup_logging(log_level="INFO")

    # Create CLI instance
    cli = KalkiCLI()

    try:
        # Route to appropriate handler
        if args.command == 'query':
            await cli.query(args.query)
        elif args.command == 'status':
            await cli.status()
        elif args.command == 'agents':
            if args.agents_command == 'list':
                await cli.agents_list()
        elif args.command == 'phase':
            if args.action == 'status':
                await cli.phase_status(args.number)
        elif args.command == 'quantum':
            if args.quantum_command == 'optimize':
                await cli.quantum_optimize(
                    args.problem,
                    variables=args.variables,
                    constraints=args.constraints,
                    objective=args.objective
                )
        elif args.command == 'predict':
            await cli.predict(args.technology, args.years)
        elif args.command == 'analyze':
            await cli.analyze_intention(
                args.intention,
                domains=args.domains,
                impact=args.impact,
                probability=args.probability
            )
        elif args.command == 'web':
            if args.web_command == 'search':
                await cli.web_search(
                    args.query,
                    num_results=args.results,
                    provider=args.provider
                )
            elif args.web_command == 'research':
                await cli.web_research(
                    args.topic,
                    depth=args.depth
                )
            elif args.web_command == 'fetch':
                await cli.web_fetch(
                    args.url,
                    max_length=args.max_length
                )
        elif args.command == 'shutdown':
            await cli.shutdown()
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"CLI error: {e}")
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        if cli.orchestrator:
            await cli.orchestrator.shutdown()


if __name__ == "__main__":
    asyncio.run(main())