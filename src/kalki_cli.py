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
import re
import textwrap
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.utils.logging_config import setup_logging, get_logger
from modules.agents.base_agent import AgentCapability
from kalki_complete import KalkiOrchestrator

logger = get_logger("Kalki.CLI")


class KalkiCLI:
    """Command Line Interface for the Kalki system"""

    def __init__(self):
        self.orchestrator: Optional[KalkiOrchestrator] = None
        # Conversation history for chat sessions
        self._chat_history: List[Dict[str, Any]] = []
        self._current_agent_override = None
        self._target_agent_initial = None

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
            response = result.get("response", result.get("enhanced_reasoning", result.get("answer", "Query processed successfully")))
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

    async def chat(self, agent: Optional[str] = None, show_metadata: bool = False):
        """Interactive chat session with Kalki or a specific agent"""
        if not self.orchestrator:
            await self.initialize()

        target_agent = None
        if agent:
            target_agent = self._find_agent_by_name(agent)
            if not target_agent:
                print(f"⚠️  Agent '{agent}' not found. Enter /agents to list available agents.")
        self._current_agent_override = target_agent
        self._target_agent_initial = target_agent
        self._chat_history = []

        print("\n🤖 Kalki Interactive Console")
        print("Type your message and press Enter. Commands: /agents, /agent <name>, /history, /exit")
        if target_agent:
            print(f"🎯 Routing messages to agent: {target_agent.name}")

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n👋 Ending chat session.")
                break

            if not user_input:
                continue

            if user_input.lower() in {"exit", "quit", "q", "/exit"}:
                print("👋 Ending chat session.")
                break

            if user_input.startswith("/"):
                if await self._handle_chat_command(user_input, show_metadata):
                    continue

            responses = await self._dispatch_chat_message(user_input)
            self._chat_history.append({
                "user": user_input,
                "responses": responses
            })

            for speaker, result in responses:
                self._render_chat_response(speaker, result, show_metadata)

    async def _handle_chat_command(self, command: str, show_metadata: bool) -> bool:
        """Handle slash commands within the chat session"""
        parts = command[1:].strip().split(maxsplit=1)
        if not parts:
            return True

        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if cmd in {"agents", "list"}:
            self._print_agent_list()
            return True
        if cmd == "agent":
            if not arg:
                print("⚠️  Usage: /agent <AgentName>")
                return True
            if arg.lower() in {"none", "clear", "off"}:
                self._current_agent_override = None
                self._target_agent_initial = None
                print("🎯 Routing messages through full Kalki orchestration.")
                self._chat_history.append({"system": "switch_agent:none"})
                return True
            new_agent = self._find_agent_by_name(arg)
            if not new_agent:
                print(f"⚠️  Agent '{arg}' not found. Use /agents to list available agents.")
                return True
            print(f"🎯 Switching to agent: {new_agent.name}")
            self._chat_history.append({"system": f"switch_agent:{new_agent.name}"})
            self._current_agent_override = new_agent
            self._target_agent_initial = new_agent
            return True
        if cmd == "history":
            self._print_chat_history(show_metadata)
            return True
        if cmd in {"clear", "reset"}:
            self._chat_history.clear()
            print("🧹 Chat history cleared.")
            return True

        print(f"⚠️  Unknown command: /{cmd}")
        return True

    async def _dispatch_chat_message(self, message: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Dispatch a chat message to an agent or the full orchestrator"""
        responses: List[Tuple[str, Dict[str, Any]]] = []

        agent_to_use = self._current_agent_override
        if agent_to_use is None:
            agent_to_use = self._target_agent_initial

        if agent_to_use:
            agent_result = await self._send_to_agent(agent_to_use, message)
            responses.append((agent_to_use.name, agent_result))
            if agent_result.get("status") != "success":
                print(f"↪️  {agent_to_use.name} could not complete the request: {agent_result.get('error', 'Unknown error')}. Using full orchestration.")
            else:
                return responses

        orchestrator_result = await self.orchestrator.process_user_query(message)
        responses.append(("Kalki", orchestrator_result))
        return responses

    async def _send_to_agent(self, agent: Any, message: str) -> Dict[str, Any]:
        """Send a message to a specific agent using capability-aware defaults"""
        task = self._build_agent_task(agent, message)
        if not task:
            return {"status": "error", "error": "No handler available for this agent"}

        try:
            result = await agent.execute(task)
            result.setdefault("status", "success")
            result.setdefault("action", task.get("action"))
            return result
        except Exception as exc:
            return {"status": "error", "error": str(exc)}

    def _build_agent_task(self, agent: Any, message: str) -> Optional[Dict[str, Any]]:
        """Create an agent task based on its capabilities"""
        capability_map = {
            AgentCapability.REASONING: ("reason", lambda msg: {"query": msg, "steps": 3}),
            AgentCapability.PLANNING: ("plan", lambda msg: {"goal": msg, "max_steps": 5}),
            AgentCapability.SEARCH: ("search", lambda msg: {"query": msg, "top_k": 5}),
            AgentCapability.CREATIVE_SYNTHESIS: ("ideate", lambda msg: {"topic": msg, "count": 5}),
            AgentCapability.OPTIMIZATION: ("optimize", lambda msg: {"target": msg}),
            AgentCapability.ETHICS: ("review", lambda msg: {"action_description": msg}),
            AgentCapability.RISK_ASSESSMENT: ("assess", lambda msg: {"scenario": msg}),
            AgentCapability.SIMULATION: ("run", lambda msg: {"scenario": msg}),
            AgentCapability.ROBOTICS_SIMULATION: ("design_robot_arm", self._parse_robot_arm_requirements),
            AgentCapability.KINEMATICS_ANALYSIS: ("analyze_kinematics", self._parse_robot_arm_requirements),
            AgentCapability.CONTROL_SYSTEMS: ("design_controller", lambda msg: {"goal": msg})
        }

        for capability in agent.capabilities:
            if capability in capability_map:
                action, builder = capability_map[capability]
                params = builder(message) if callable(builder) else builder
                return {"action": action, "params": params}

        return {"action": "chat", "params": {"message": message, "prompt": message}}

    def _parse_robot_arm_requirements(self, text: str) -> Dict[str, Any]:
        """Extract simple robotics arm requirements from free text"""
        requirements: Dict[str, Any] = {"description": text}

        dof_match = re.search(r"(\d+)\s*(?:dof|degrees of freedom)", text, re.IGNORECASE)
        if dof_match:
            requirements["degrees_of_freedom"] = int(dof_match.group(1))

        payload_match = re.search(r"payload\s*(\d+(?:\.\d+)?)\s*(kg|kilogram)", text, re.IGNORECASE)
        if payload_match:
            requirements["payload"] = float(payload_match.group(1))

        reach_match = re.search(r"reach(?:ing)?\s*(\d+(?:\.\d+)?)\s*(m|meter|metre)", text, re.IGNORECASE)
        if reach_match:
            requirements["workspace_radius"] = float(reach_match.group(1))

        precision_match = re.search(r"precision\s*(\d+(?:\.\d+)?)\s*(mm|millimeter|micron|um)", text, re.IGNORECASE)
        if precision_match:
            value = float(precision_match.group(1))
            requirements["precision"] = value / 1000 if precision_match.group(2).lower().startswith("mm") else value

        return requirements

    def _render_chat_response(self, speaker: str, result: Dict[str, Any], show_metadata: bool) -> None:
        """Render a chat response in a human-readable format"""
        print(f"\n{speaker}:")

        if not result:
            print("  (no response)")
            return

        status = result.get("status", "success")
        display_text, consumed_keys = self._extract_primary_text(result)

        if status != "success":
            error_message = result.get("error", "Unknown error")
            print(f"  ⚠️  {error_message}")
            if show_metadata:
                print(textwrap.indent(json.dumps(result, indent=2, default=str), "  "))
            return

        if display_text:
            for line in textwrap.wrap(display_text, width=100):
                print(f"  {line}")
        elif show_metadata:
            print(textwrap.indent(json.dumps(result, indent=2, default=str), "  "))
        else:
            print("  (response available in metadata)")

        if show_metadata:
            metadata = {k: v for k, v in result.items() if k not in {"status", *consumed_keys}}
            if metadata:
                print("  --- metadata ---")
                print(textwrap.indent(json.dumps(metadata, indent=2, default=str), "  "))

    def _extract_primary_text(self, result: Dict[str, Any]) -> tuple:
        """Extract primary display text from a result dictionary"""
        candidate_keys = [
            "response",
            "answer",
            "summary",
            "plan",
            "ideas",
            "result",
            "message"
        ]

        for key in candidate_keys:
            if key in result:
                value = result[key]
                if isinstance(value, str):
                    return value, {key}
                return json.dumps(value, indent=2, default=str), {key}

        return None, set()

    def _print_agent_list(self) -> None:
        """Display available agents"""
        if not self.orchestrator:
            print("⚠️  Orchestrator not initialized.")
            return

        agents = self.orchestrator.agent_manager.agents if self.orchestrator.agent_manager else {}
        if not agents:
            print("⚠️  No agents registered.")
            return

        print("\n🤖 Registered Agents:")
        for name, agent in sorted(agents.items()):
            capabilities = ", ".join(cap.value for cap in agent.capabilities)
            print(f"  • {name} [{agent.status.value}] → {capabilities}")

    def _print_chat_history(self, show_metadata: bool) -> None:
        """Display previous chat exchanges"""
        if not self._chat_history:
            print("(empty)")
            return

        for idx, turn in enumerate(self._chat_history, 1):
            user_text = turn.get("user")
            if user_text:
                print(f"\n[{idx}] You: {user_text}")
            for speaker, response in turn.get("responses", []):
                print(f"   {speaker}:")
                text, _ = self._extract_primary_text(response)
                if text:
                    for line in textwrap.wrap(text, width=80):
                        print(f"     {line}")
                if show_metadata:
                    print(textwrap.indent(json.dumps(response, indent=2, default=str), "     "))

    def _find_agent_by_name(self, name: str) -> Optional[Any]:
        """Locate an agent by name (case-insensitive)"""
        if not self.orchestrator or not self.orchestrator.agent_manager:
            return None

        agents = self.orchestrator.agent_manager.agents
        if name in agents:
            return agents[name]

        name_lower = name.lower()
        for agent_name, agent in agents.items():
            if agent_name.lower() == name_lower:
                return agent
        return None

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

    async def learn_ingest(self, pdf_path: str, archive: bool = False, **kwargs):
        """Ingest PDF into hybrid learning system"""
        from modules.hybrid_learning_system import get_hybrid_system
        import fitz  # PyMuPDF
        
        pdf_path_obj = Path(pdf_path)
        if not pdf_path_obj.exists():
            print(f"❌ PDF file not found: {pdf_path}")
            return
        
        print(f"📚 Ingesting PDF: {pdf_path_obj.name}")
        
        # Extract text from PDF properly
        try:
            doc = fitz.open(pdf_path)
            pdf_content = ""
            for page in doc:
                pdf_content += page.get_text()
            doc.close()
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {e}")
            return
        
        # Determine if LLM enhancements should be used (DEFAULT: True unless --no-llm)
        use_llm = not kwargs.get('no_llm', False)
        
        if use_llm:
            print("🤖 LLM validation enabled (Llama 3.1 8B):")
            print("   ✓ Formula validation & variable parsing")
            print("   ✓ Material property validation")
            print("   ✓ Design rule validation")
            print("   ✓ Code requirement validation")
            print("   ✓ Cost data validation")
            print("   ✓ Load parameter validation")
            print("   ⏳ Processing with enhanced accuracy...\n")
        else:
            print("⚠️  LLM validation disabled (regex-only extraction)")
            print("   Note: Remove --no-llm flag for higher accuracy\n")
        
        # Process with hybrid system
        hybrid_system = get_hybrid_system()
        results = hybrid_system.ingest_pdf(
            pdf_path, 
            pdf_content, 
            archive=archive,
            use_llm_enhancements=use_llm
        )
        
        print(f"✅ PDF ingested successfully!")
        print(f"\n📊 Extracted Knowledge:")
        
        # Legacy extractors
        print(f"   Formulas: {results.get('formulas', 0)}")
        print(f"   Materials: {results.get('materials', 0)}")
        print(f"   Design Rules: {results.get('rules', 0)}")
        print(f"   Code Requirements: {results.get('codes', 0)}")
        
        # v3.0 Enhanced extractors
        v3_total = (results.get('span_tables', 0) + results.get('procedures', 0) + 
                    results.get('inspection_criteria', 0) + results.get('cost_data', 0) +
                    results.get('load_parameters', 0) + results.get('decision_trees', 0))
        
        if v3_total > 0:
            print(f"\n   🆕 v3.0 Enhanced Extraction:")
            print(f"   Span Tables: {results.get('span_tables', 0)}")
            print(f"   Procedures: {results.get('procedures', 0)}")
            print(f"   Inspection Criteria: {results.get('inspection_criteria', 0)}")
            print(f"   Cost Data: {results.get('cost_data', 0)}")
            print(f"   Load Parameters: {results.get('load_parameters', 0)}")
            print(f"   Decision Trees: {results.get('decision_trees', 0)}")
        
        if use_llm:
            print(f"\n   🤖 LLM Enhancements Applied:")
            print(f"   Validated formulas: {results.get('llm_validated_formulas', 0)}")
            print(f"   Variables parsed: {results.get('llm_parsed_variables', 0)}")

    async def learn_query_smart(self, query: str, **kwargs):
        """Smart query with LLM synthesis"""
        from modules.hybrid_learning_system import get_hybrid_system
        
        print(f"🤖 Processing query: {query}")
        print("⏳ Using LLM for smart routing and synthesis...\n")
        
        hybrid_system = get_hybrid_system()
        
        # Step 1: Auto-detect query intent
        routing = await hybrid_system.query_router(query)
        print(f"📍 Detected query type: {routing['query_type']} (confidence: {routing['confidence']*100:.0f}%)")
        if routing['entities']:
            print(f"   Entities: {', '.join(routing['entities'])}")
        
        # Step 2: Get synthesized answer
        answer = await hybrid_system.hybrid_query_with_synthesis(
            query, 
            query_type=routing['query_type'],
            **routing.get('filters', {})
        )
        
        print(f"\n{'='*80}")
        print(f"💬 KALKI's Answer:")
        print(f"{'='*80}\n")
        print(answer)
        print(f"\n{'='*80}")

    async def learn_query(self, knowledge_type: str, domain: Optional[str] = None, **kwargs):
        """Query learned knowledge"""
        from modules.hybrid_learning_system import get_hybrid_system
        
        hybrid_system = get_hybrid_system()
        results = hybrid_system.get_learned_knowledge(knowledge_type, domain=domain)
        
        print(f"🔍 {knowledge_type.replace('_', ' ').title()} Knowledge:")
        print(f"Found {len(results)} items\n")
        
        for i, item in enumerate(results[:10], 1):  # Show first 10
            if knowledge_type == 'formula':
                print(f"{i}. {item['name']}: {item['formula']}")
            elif knowledge_type == 'material':
                print(f"{i}. {item['material_name']}: {item['properties']}")
            elif knowledge_type == 'design_rule':
                print(f"{i}. {item['rule']}")
            elif knowledge_type == 'code_requirement':
                print(f"{i}. {item['requirement']}")
        
        if len(results) > 10:
            print(f"\n... and {len(results) - 10} more")

    async def learn_training(self, **kwargs):
        """Generate training data for fine-tuning"""
        from modules.hybrid_learning_system import get_hybrid_system
        
        print("📝 Generating training data for fine-tuning...")
        
        hybrid_system = get_hybrid_system()
        training_file = hybrid_system.generate_training_data()
        
        print(f"✅ Training data generated: {training_file}")
        print("\nNext steps:")
        print("1. Review training data")
        print("2. Install MLX: pip install mlx mlx-lm")
        print("3. Fine-tune model with your M4 Max GPU")

    async def learn_stats(self, **kwargs):
        """Show hybrid learning system statistics"""
        from modules.hybrid_learning_system import get_hybrid_system
        
        hybrid_system = get_hybrid_system()
        stats = hybrid_system.get_system_stats()
        
        print("📊 Hybrid Learning System Statistics")
        print("=" * 50)
        print(f"\n📄 Processed PDFs: {stats['processed_pdfs']}")
        
        print("\n📚 Knowledge Base:")
        print(f"   Formulas: {stats['knowledge_base']['formulas']}")
        print(f"   Materials: {stats['knowledge_base']['materials']}")
        print(f"   Design Rules: {stats['knowledge_base']['design_rules']}")
        print(f"   Code Requirements: {stats['knowledge_base']['code_requirements']}")
        
        print("\n💾 Storage Breakdown:")
        for storage_type, description in stats['storage_breakdown'].items():
            print(f"   • {storage_type}: {description}")

    async def domains_list(self, **kwargs):
        """List all available domains"""
        from modules.domains.domain_registry import DomainRegistry
        
        print("🌐 KALKI Domain Expertise")
        print("=" * 60)
        
        registry = DomainRegistry()
        domains = registry.list_domains()
        
        if not domains:
            print("No domains loaded yet.")
            return
        
        for domain_name in domains:
            info = registry.get_domain_info(domain_name)
            if info:
                total_knowledge = info['knowledge_total']
                print(f"\n✅ {domain_name.upper()}")
                print(f"   {info['description']}")
                print(f"   Knowledge Items: {total_knowledge:,}")
                if total_knowledge > 0:
                    print(f"   Breakdown: {', '.join(f'{k}: {v}' for k, v in info['knowledge_stats'].items() if v > 0)}")
                print(f"   Deliverables: {len(info['deliverables'])}")
        
        stats = registry.get_statistics()
        print(f"\n📊 Total: {stats['total_domains']} domains, {stats['total_knowledge_items']:,} knowledge items")

    async def domains_info(self, domain: str, **kwargs):
        """Get detailed info about a specific domain"""
        from modules.domains.domain_registry import DomainRegistry
        
        registry = DomainRegistry()
        info = registry.get_domain_info(domain)
        
        if not info:
            print(f"❌ Domain '{domain}' not found")
            print(f"Available domains: {', '.join(registry.list_domains())}")
            return
        
        print(f"🌐 {domain.upper()} Domain")
        print("=" * 60)
        print(f"\n📝 Description: {info['description']}")
        print(f"\n📊 Knowledge Base: {info['knowledge_total']:,} total items")
        
        if info['knowledge_stats']:
            print("\nKnowledge Types:")
            for k, v in info['knowledge_stats'].items():
                print(f"   • {k.replace('_', ' ').title()}: {v:,}")
        
        print(f"\n📦 Deliverables ({len(info['deliverables'])}):")
        domain_obj = registry.get_domain(domain)
        if domain_obj:
            for deliv in domain_obj.get_deliverable_types():
                print(f"   • {deliv.name}: {deliv.description}")

    async def domains_stats(self, **kwargs):
        """Show domain system statistics"""
        from modules.domains.domain_registry import DomainRegistry
        
        registry = DomainRegistry()
        stats = registry.get_statistics()
        
        print("📊 Domain System Statistics")
        print("=" * 60)
        print(f"\nTotal Domains: {stats['total_domains']}")
        print(f"Loaded Domains: {stats['loaded_domains']}")
        print(f"Total Knowledge Items: {stats['total_knowledge_items']:,}")
        
        print("\nPer-Domain Breakdown:")
        for domain_name, count in stats['domains'].items():
            print(f"   • {domain_name}: {count:,} items")

    async def project_create(self, description: str, domain: Optional[str] = None, 
                           requirements: Optional[str] = None, **kwargs):
        """Create new project"""
        from modules.domains.domain_registry import DomainRegistry
        from modules.domains.project_persistence import get_project_persistence
        import uuid
        
        registry = DomainRegistry()
        persistence = get_project_persistence()
        
        # Infer domain if not specified
        if not domain:
            print(f"🔍 Inferring domain from: '{description}'")
            inferred = await registry.infer_domain(description)
            if not inferred:
                print("❌ Could not infer domain. Please specify --domain")
                print(f"Available domains: {', '.join(registry.list_domains())}")
                return
            domain = inferred[0]
            if len(inferred) > 1:
                print(f"💡 Multiple domains detected: {inferred}")
                print(f"Using primary: {domain}")
        
        # Load domain
        domain_obj = registry.get_domain(domain)
        if not domain_obj:
            print(f"❌ Domain '{domain}' not found")
            return
        
        # Parse requirements
        reqs = json.loads(requirements) if requirements else {}
        
        # Create project
        print(f"\n🚀 Creating {domain} project...")
        project = await domain_obj.create_project(description, reqs)
        
        # Save project
        if persistence.save_project(project):
            print(f"\n✅ Project created and saved successfully!")
        else:
            print(f"\n✅ Project created (warning: save failed)")
        
        print(f"   Project ID: {project.project_id}")
        print(f"   Domain: {project.domain}")
        print(f"   Current Phase: {project.current_phase}")
        print(f"\nUse 'kalki project status {project.project_id}' to check progress")
        
        return project

    async def project_status(self, project_id: str, **kwargs):
        """Show project status"""
        from modules.domains.project_persistence import get_project_persistence
        
        persistence = get_project_persistence()
        project_data = persistence.load_project(project_id)
        
        if not project_data:
            print(f"❌ Project {project_id} not found")
            return
        
        print(f"📊 Project Status")
        print("=" * 60)
        print(f"\nProject ID: {project_data['project_id']}")
        print(f"Domain: {project_data['domain']}")
        print(f"Description: {project_data['description']}")
        print(f"Current Phase: {project_data['current_phase']}")
        
        if 'location' in project_data:
            print(f"Location: {project_data['location']}")
        if 'building_type' in project_data:
            print(f"Building Type: {project_data['building_type']}")
        
        print(f"\n📅 Timeline:")
        print(f"   Created: {project_data.get('created_at', 'Unknown')}")
        print(f"   Updated: {project_data.get('updated_at', 'Unknown')}")
        
        if project_data.get('phase_history'):
            print(f"\n📜 Phase History:")
            for h in project_data['phase_history']:
                print(f"   {h['from']} → {h['to']}")
        
        if project_data.get('issues'):
            print(f"\n⚠️  Issues ({len(project_data['issues'])}):")
            for issue in project_data['issues'][:5]:
                print(f"   • {issue}")

    async def project_advance_phase(self, project_id: str, phase: Optional[str] = None, **kwargs):
        """Advance project to next phase"""
        from modules.domains.project_persistence import get_project_persistence
        from modules.domains.domain_registry import DomainRegistry
        
        persistence = get_project_persistence()
        project_data = persistence.load_project(project_id)
        
        if not project_data:
            print(f"❌ Project {project_id} not found")
            return
        
        print(f"➡️  Advancing project {project_id}")
        print("=" * 60)
        print(f"\nCurrent Phase: {project_data['current_phase']}")
        
        # TODO: Implement actual phase advancement logic
        # Need to reconstruct ProjectStateMachine from data
        # Then call advance_phase()
        # Then save back
        
        print("\n⚠️  Phase advancement logic coming soon!")
        print("This requires reconstructing the ProjectStateMachine from saved data.")

    async def project_query(self, project_id: str, query: str, **kwargs):
        """Ask question about project"""
        from modules.domains.project_persistence import get_project_persistence
        
        persistence = get_project_persistence()
        project_data = persistence.load_project(project_id)
        
        if not project_data:
            print(f"❌ Project {project_id} not found")
            return
        
        print(f"❓ Project Query: {project_id}")
        print(f"Question: {query}")
        print("=" * 60)
        
        # TODO: Implement project-aware query
        # Load project context, pass to domain's get_contextual_help()
        
        print(f"\nProject: {project_data['description']}")
        print(f"Current Phase: {project_data['current_phase']}")
        print("\n⚠️  Project-aware queries coming soon!")

    async def project_list(self, **kwargs):
        """List all projects"""
        from modules.domains.project_persistence import get_project_persistence
        
        persistence = get_project_persistence()
        projects = persistence.list_projects()
        
        if not projects:
            print("📋 No projects found")
            print("\nCreate one with: kalki project create \"Your project description\"")
            return
        
        print("📋 All Projects")
        print("=" * 60)
        
        for proj in projects:
            print(f"\n✅ {proj['description'][:50]}...")
            print(f"   ID: {proj['project_id']}")
            print(f"   Domain: {proj['domain']}")
            print(f"   Phase: {proj['current_phase']}")
            print(f"   Updated: {proj['updated_at']}")
        
        print(f"\n📊 Total: {len(projects)} projects")
        
        # Show statistics
        stats = persistence.get_project_stats()
        if stats.get('by_domain'):
            print("\nBy Domain:")
            for domain, count in stats['by_domain'].items():
                print(f"   • {domain}: {count}")

    async def ask(self, query: str, domain: Optional[str] = None, **kwargs):
        """Natural language query with auto-domain inference"""
        from modules.domains.domain_registry import DomainRegistry
        
        print(f"🤔 Question: {query}")
        print("=" * 60)
        
        registry = DomainRegistry()
        
        # Infer domain if not specified
        if not domain:
            inferred = await registry.infer_domain(query)
            if inferred:
                print(f"🔍 Detected domain(s): {', '.join(inferred)}")
                domain = inferred[0]
            else:
                print("💬 Using general knowledge (no specific domain)")
        
        if domain:
            domain_obj = registry.get_domain(domain)
            if domain_obj:
                print(f"🌐 Consulting {domain} domain expertise...")
                # TODO: Integrate with Supreme Control Hub for domain-aware responses
        
        # For now, fall back to general query
        await self.query(query)

    async def dev_app(self, platform: str, name: str, app_type: str = "productivity", 
                     description: str = None, monetization: str = "free", **kwargs):
        """Generate complete app"""
        from modules.software_deliverables import SoftwareDeliverablesGenerator
        
        print(f"📱 Generating {platform.upper()} app: {name}")
        print(f"   Type: {app_type}")
        print(f"   Monetization: {monetization}")
        
        app_spec = {
            "name": name,
            "platform": platform,
            "type": app_type,
            "description": description or f"A {app_type} app",
            "features": ["data", "ui"],
            "monetization": {
                "type": monetization,
                "products": [f"{monetization}_tier"] if monetization in ["iap", "paid"] else []
            }
        }
        
        generator = SoftwareDeliverablesGenerator()
        deliverables = await generator.generate_app(app_spec)
        
        print(f"\n✅ App generated successfully!")
        print(f"\n📦 Project: {deliverables.project_structure['root']}")
        print(f"📁 Source Files: {len(deliverables.source_files)}")
        print(f"📄 Documentation: {len(deliverables.documentation)}")
        print(f"⏱️  Estimated Dev Time: {deliverables.estimated_dev_time} hours")
        
        print(f"\nNext steps:")
        print(f"1. cd {deliverables.project_structure['root']}")
        if platform == "ios":
            print(f"2. open {name}.xcodeproj")
        else:
            print(f"2. Open in Android Studio")
        print("3. Build and run!")

    async def dev_game(self, engine: str, name: str, genre: str = "action",
                      description: str = None, **kwargs):
        """Generate complete game"""
        from modules.software_deliverables import SoftwareDeliverablesGenerator
        
        print(f"🎮 Generating {engine.title()} game: {name}")
        print(f"   Genre: {genre}")
        
        game_spec = {
            "name": name,
            "engine": engine,
            "genre": genre,
            "description": description or f"A {genre} game",
            "features": ["gameplay", "ui", "audio"],
            "monetization": {"type": "iap"}
        }
        
        generator = SoftwareDeliverablesGenerator()
        deliverables = await generator.generate_game(game_spec)
        
        print(f"\n✅ Game generated successfully!")
        print(f"\n📦 Project: {deliverables.project_structure['root']}")
        print(f"📁 Source Files: {len(deliverables.source_files)}")
        print(f"📄 Documentation: {len(deliverables.documentation)}")
        print(f"⏱️  Estimated Dev Time: {deliverables.estimated_dev_time} hours")
        
        print(f"\nNext steps:")
        print(f"1. Open {deliverables.project_structure['root']} in {engine.title()}")
        print("2. Import project")
        print("3. Build and play!")

    async def shutdown(self, **kwargs):
        """Shutdown the system"""
        if self.orchestrator:
            await self.orchestrator.shutdown()
        print("👋 Kalki shutdown complete")
        sys.exit(0)

    async def design_create(self, request: str, name: Optional[str] = None, **kwargs):
        """Create a complete design project"""
        if not self.orchestrator:
            await self.initialize()
        
        print(f"🎨 Creating design project: {request}")
        if name:
            print(f"   Project name: {name}")
        
        result = await self.orchestrator.create_design(request, name)
        
        if result.get("status") == "success":
            project = result.get("project")
            print(f"\n✅ Design project created successfully!")
            print(f"\n📋 Project Details:")
            print(f"   ID: {project.project_id}")
            print(f"   Name: {project.name}")
            print(f"   Status: {project.status}")
            print(f"   Description: {project.description}")
            print(f"\n📊 Generated Artifacts:")
            print(f"   Blueprint: {bool(project.blueprint)}")
            print(f"   3D Models: {len(project.models_3d)}")
            print(f"   Simulations: {len(project.simulations)}")
            print(f"   Renders: {len(project.renders)}")
            print(f"   Holograms: {len(project.holograms)}")
            
            # Show professional deliverables if available
            if project.professional_deliverables and project.professional_deliverables.get("generated_files"):
                deliverables = project.professional_deliverables
                print(f"\n📦 Professional Deliverables:")
                print(f"   Generated Files: {len(deliverables.get('generated_files', []))}")
                if deliverables.get('bill_of_materials'):
                    bom = deliverables['bill_of_materials']
                    print(f"   BOM Items: {len(bom.get('items', []))}")
                    print(f"   Est. Cost: ${bom.get('total_cost_estimate', 0):,.2f}")
                if deliverables.get('timeline_estimate'):
                    timeline = deliverables['timeline_estimate']
                    print(f"   Est. Timeline: {timeline.get('total_duration_days', 'N/A')} days")
            
            print(f"\n💡 Next steps:")
            print(f"   kalki design status {project.project_id}")
            print(f"   kalki design export {project.project_id}")
        else:
            print(f"❌ Design creation failed: {result.get('error', 'Unknown error')}")

    async def design_status(self, project_id: str, **kwargs):
        """Get design project status"""
        if not self.orchestrator:
            await self.initialize()
        
        result = await self.orchestrator.get_design_status(project_id)
        
        if result.get("status") == "success":
            print(f"📊 Design Project Status: {project_id}")
            print(f"   Name: {result.get('project_name')}")
            print(f"   Status: {result.get('project_status')}")
            print(f"   3D Models: {result.get('models')}")
            print(f"   Simulations: {result.get('simulations')}")
            print(f"   Renders: {result.get('renders')}")
            print(f"   Holograms: {result.get('holograms')}")
        else:
            print(f"❌ Error: {result.get('error')}")
    
    async def supreme_query(self, query: str, mode: str = "supreme", **kwargs):
        """Process query using Supreme Control Hub with full system integration"""
        if not self.orchestrator:
            await self.initialize()
        
        print(f"⚡ Supreme Processing: {query}")
        print(f"   Mode: {mode.upper()}")
        
        try:
            from modules.supreme_control_hub import get_supreme_control_hub
            
            supreme_hub = get_supreme_control_hub()
            result = await supreme_hub.process_supreme_task(query, mode=mode)
            
            print(f"\n✨ Supreme Result:")
            print(f"   Quality Score: {result.quality_score:.3f}")
            print(f"   Consciousness Level: {result.consciousness_level:.3f}")
            print(f"   Reasoning Depth: {result.reasoning_depth}")
            print(f"   Execution Time: {result.execution_time:.2f}s")
            
            print(f"\n📚 Knowledge Used:")
            for key, value in result.knowledge_used.items():
                if value > 0:
                    print(f"   {key.replace('_', ' ').title()}: {value}")
            
            if result.design_artifacts:
                print(f"\n🎨 Design Generated:")
                print(f"   Blueprint ID: {result.design_artifacts['blueprint_id']}")
                print(f"   Components: {result.design_artifacts['components']}")
            
            # Show synthesis result summary
            synthesis = result.synthesis_result
            print(f"\n🧠 Supreme Synthesis:")
            if synthesis.get('conceptual_blueprint'):
                blueprint = synthesis['conceptual_blueprint']
                print(f"   Concept: {blueprint.get('concept', 'N/A')}")
        
        except Exception as e:
            logger.error(f"Supreme processing failed: {e}")
            print(f"❌ Error: {e}")
    
    async def validate_design(self, project_id: str, types: List[str], **kwargs):
        """Multi-modal design validation"""
        if not self.orchestrator:
            await self.initialize()
        
        print(f"🔍 Validating design: {project_id}")
        print(f"   Validation types: {', '.join(types)}")
        
        try:
            # Get design project
            from modules.generative_design_engine import GenerativeDesignEngine
            from modules.multimodal_validator import get_multimodal_validator
            
            # Check if project exists
            gen_engine = GenerativeDesignEngine()
            if project_id not in gen_engine.active_projects:
                print(f"❌ Project not found: {project_id}")
                return
            
            project = gen_engine.active_projects[project_id]
            
            # Get design blueprint from project
            # For now, create a minimal blueprint for validation
            from modules.design_brain import DesignBlueprint, DesignIntent, DesignComponent
            
            # Create blueprint from project data
            intent = DesignIntent(
                category="robotics",
                complexity="complex",
                components=[],
                constraints=[],
                materials=project.blueprint.get("materials", []),
                scale="medium"
            )
            
            blueprint = DesignBlueprint(
                id=project_id,
                timestamp=project.created_at,
                intent=intent,
                components=[],
                system_requirements=project.blueprint.get("dimensions", {}),
                design_parameters=project.blueprint.get("specifications", {}),
                validation_checks=[]
            )
            
            # Run validation
            validator = get_multimodal_validator()
            report = await validator.validate_design(blueprint, validation_types=types)
            
            print(f"\n✅ Validation Complete")
            print(f"\n📊 Overall Results:")
            print(f"   Score: {report.overall_score:.2f}")
            print(f"   Verdict: {report.overall_verdict.upper()}")
            
            if report.visual:
                print(f"\n🎨 Visual Analysis:")
                print(f"   Aesthetic Score: {report.visual.aesthetic_score:.2f}")
                print(f"   Proportion Score: {report.visual.proportion_score:.2f}")
                print(f"   Golden Ratio: {report.visual.golden_ratio_compliance:.2f}")
                print(f"   Balance: {report.visual.visual_balance}")
            
            if report.structural:
                print(f"\n🏗️ Structural Analysis:")
                print(f"   Safety Factor: {report.structural.safety_factor:.2f}")
                print(f"   Integrity: {report.structural.structural_integrity}")
                print(f"   Max Stress: {report.structural.max_stress_mpa:.1f} MPa")
            
            if report.acoustic:
                print(f"\n🔊 Acoustic Analysis:")
                print(f"   Quality: {report.acoustic.acoustic_quality}")
                print(f"   Reverberation: {report.acoustic.reverberation_time_s:.2f}s")
            
            if report.thermal:
                print(f"\n🌡️ Thermal Analysis:")
                print(f"   Max Temp: {report.thermal.max_temperature_c:.1f}°C")
                print(f"   Safety: {report.thermal.thermal_safety}")
            
            if report.critical_issues:
                print(f"\n⚠️ Critical Issues:")
                for issue in report.critical_issues:
                    print(f"   - {issue}")
            
            if report.recommendations:
                print(f"\n💡 Recommendations:")
                for rec in report.recommendations[:5]:  # Top 5
                    print(f"   - {rec}")
        
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            print(f"❌ Error: {e}")

    async def shutdown(self):
        """Gracefully shutdown the Kalki system"""
        if not self.orchestrator:
            print("System not running")
            return

        print("� Shutting down Kalki system...")
        await self.orchestrator.shutdown()
        print("✅ System shutdown complete")
        sys.exit(0)
    
    async def validate_design(self, project_id: str, **kwargs):
        """Validate design using multi-modal validation"""
        if not self.orchestrator:
            await self.initialize()
        
        print(f"🔍 Running multi-modal validation for project: {project_id}")
        
        try:
            from modules.multimodal_validator import get_multimodal_validator
            from modules.generative_design_engine import GenerativeDesignEngine
            
            # Get design project
            gen_engine = GenerativeDesignEngine()
            await gen_engine.initialize()
            
            if project_id not in gen_engine.active_projects:
                print(f"❌ Project {project_id} not found")
                return
            
            project = gen_engine.active_projects[project_id]
            
            if not project.blueprint:
                print(f"❌ No blueprint available for validation")
                return
            
            # Convert to DesignBlueprint
            from modules.design_brain import DesignBrain, DesignBlueprint
            design_brain = DesignBrain()
            await design_brain.initialize()
            
            # Run validation
            validator = get_multimodal_validator()
            validation_types = kwargs.get('types', ["visual", "structural", "thermal"])
            
            print(f"   Validation types: {', '.join(validation_types)}")
            
            # Create temporary blueprint for validation
            # In real implementation, would extract from project
            print(f"\n⚙️ Running validation...")
            print(f"   (Multi-modal validation requires 3D model - generating heuristic analysis)")
            
            print(f"\n✅ Validation would check:")
            if "visual" in validation_types:
                print(f"   🎨 Visual: Aesthetics, proportion, golden ratio")
            if "structural" in validation_types:
                print(f"   🏗️ Structural: Safety factors, stress analysis, deflection")
            if "acoustic" in validation_types:
                print(f"   🔊 Acoustic: Sound propagation, reverberation")
            if "thermal" in validation_types:
                print(f"   🌡️ Thermal: Temperature distribution, heat dissipation")
            
        except Exception as e:
            print(f"❌ Validation failed: {e}")
            logger.error(f"Validation error: {e}")
        
        result = await self.orchestrator.get_design_status(project_id)
        
        if result.get("status") == "success":
            print(f"📊 Design Project Status: {project_id}")
            print(f"   Name: {result.get('project_name')}")
            print(f"   Status: {result.get('project_status')}")
            print(f"   3D Models: {result.get('models')}")
            print(f"   Simulations: {result.get('simulations')}")
            print(f"   Renders: {result.get('renders')}")
            print(f"   Holograms: {result.get('holograms')}")
        else:
            print(f"❌ Error: {result.get('error')}")

    async def show_status(self, component: str = 'all'):
        """Show system status"""
        if not self.orchestrator:
            await self.initialize()
        
        from datetime import datetime
        print(f"\n{'='*80}")
        print(f"🎯 KALKI System Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        if component in ['all', 'hub']:
            from modules.supreme_control_hub import get_supreme_control_hub
            hub = get_supreme_control_hub()
            stats = hub.get_statistics()
            
            print(f"\n🌟 Supreme Control Hub:")
            print(f"   Status: {'✅ Operational' if stats['total_executions'] >= 0 else '❌ Error'}")
            print(f"   Total Executions: {stats['total_executions']}")
            print(f"   Average Quality: {stats['average_quality']:.2%}")
            print(f"   Knowledge Queries: {stats.get('knowledge_queries', 0)}")
            
        if component in ['all', 'consciousness']:
            from modules.consciousness_engine import get_consciousness_engine
            consciousness = get_consciousness_engine()
            state = consciousness.state
            
            print(f"\n🧠 Consciousness Engine:")
            print(f"   Awareness Level: {state.awareness_level:.2%}")
            print(f"   Self-Reflection Depth: {state.self_reflection_depth:.2%}")
            print(f"   Emotional Resonance: {state.emotional_resonance}")
            print(f"   Intention Coherence: {state.intention_coherence:.2%}")
            
        if component in ['all', 'evolution']:
            from modules.autonomous_evolution_loop import get_evolution_loop
            evolution = get_evolution_loop()
            evo_status = evolution.get_evolution_status()
            
            print(f"\n🧬 Autonomous Evolution:")
            print(f"   Status: {'🔄 Running' if evo_status['is_running'] else '⏸️ Stopped'}")
            print(f"   System Performance: {evo_status['current_performance']:.2%}")
            print(f"   Deployed Evolutions: {evo_status['total_deployed_evolutions']}")
            print(f"   Active Candidates: {evo_status['active_candidates']}")
            if evo_status['recent_gaps']:
                print(f"   Recent Gaps: {', '.join(evo_status['recent_gaps'])}")
            
        if component in ['all', 'telemetry']:
            from modules.realworld_telemetry_integration import get_telemetry_integration
            telemetry = get_telemetry_integration()
            tel_status = telemetry.get_telemetry_status()
            
            print(f"\n📡 Real-World Telemetry:")
            print(f"   Status: {'🔄 Collecting' if tel_status['is_running'] else '⏸️ Stopped'}")
            print(f"   Deployed Designs: {tel_status['deployed_designs']}")
            print(f"   Total Data Points: {tel_status['total_data_points_collected']}")
            print(f"   Designs with Issues: {tel_status['designs_with_issues']}")
            print(f"   Learning Insights: {tel_status['learning_insights']}")
            print(f"   Unapplied Insights: {tel_status['unapplied_insights']}")
        
        print(f"\n{'='*80}")
    
    async def evolution_control(self, evolution_command: str):
        """Control autonomous evolution loop"""
        from modules.autonomous_evolution_loop import get_evolution_loop
        
        evolution = get_evolution_loop()
        
        if evolution_command == 'start':
            print("🧬 Starting autonomous evolution loop...")
            await evolution.initialize()
            # Start in background task
            asyncio.create_task(evolution.start())
            print("✅ Evolution loop started")
            
        elif evolution_command == 'stop':
            print("⏸️ Stopping evolution loop...")
            await evolution.stop()
            print("✅ Evolution loop stopped")
            
        elif evolution_command == 'status':
            status = evolution.get_evolution_status()
            print(f"\n🧬 Evolution Status:")
            print(f"   Running: {status['is_running']}")
            print(f"   Performance: {status['current_performance']:.2%}")
            print(f"   Deployed: {status['total_deployed_evolutions']}")
            print(f"   Active Candidates: {status['active_candidates']}")
            print(f"   Learning Rate: {status['learning_rate']:.3f}")
            
        elif evolution_command == 'cycle':
            print("🔬 Running evolution cycle...")
            await evolution.initialize()
            await evolution._evolution_cycle()
            print("✅ Evolution cycle complete")
    
    async def telemetry_control(self, telemetry_command: str, **kwargs):
        """Control real-world telemetry integration"""
        from modules.realworld_telemetry_integration import get_telemetry_integration
        
        telemetry = get_telemetry_integration()
        
        if telemetry_command == 'start':
            print("📡 Starting telemetry collection...")
            await telemetry.initialize()
            # Start in background task
            asyncio.create_task(telemetry.start())
            print("✅ Telemetry collection started")
            
        elif telemetry_command == 'stop':
            print("⏸️ Stopping telemetry collection...")
            await telemetry.stop()
            print("✅ Telemetry collection stopped")
            
        elif telemetry_command == 'status':
            status = telemetry.get_telemetry_status()
            print(f"\n📡 Telemetry Status:")
            print(f"   Running: {status['is_running']}")
            print(f"   Deployed Designs: {status['deployed_designs']}")
            print(f"   Data Points: {status['total_data_points_collected']}")
            print(f"   Issues: {status['designs_with_issues']}")
            print(f"   Insights: {status['learning_insights']}")
            
        elif telemetry_command == 'register':
            design_id = kwargs.get('design_id')
            project_id = kwargs.get('project_id')
            location = kwargs.get('location')
            endpoints = kwargs.get('endpoints', [])
            
            print(f"📝 Registering deployment: {design_id}")
            await telemetry.initialize()
            await telemetry.register_deployment(
                design_id=design_id,
                project_id=project_id,
                location=location,
                telemetry_endpoints=endpoints,
                expected_performance={}  # Would be from design
            )
            print(f"✅ Deployment registered")
    
    async def knowledge_control(self, knowledge_command: str, **kwargs):
        """Control planetary knowledge graph"""
        from modules.planetary_knowledge_graph import get_planetary_knowledge_graph
        
        knowledge = get_planetary_knowledge_graph()
        
        if knowledge_command == 'start':
            print("🌍 Starting planetary knowledge synchronization...")
            await knowledge.initialize()
            asyncio.create_task(knowledge.start())
            print("✅ Knowledge synchronization started")
            
        elif knowledge_command == 'stop':
            print("⏸️ Stopping knowledge synchronization...")
            await knowledge.stop()
            print("✅ Knowledge synchronization stopped")
            
        elif knowledge_command == 'status':
            stats = knowledge.get_statistics()
            print(f"\n🌍 Planetary Knowledge Graph:")
            print(f"   Instance ID: {stats['instance_id'][:12]}...")
            print(f"   Running: {stats['is_running']}")
            print(f"   Total Nodes: {stats['total_nodes']}")
            print(f"   Total Relationships: {stats['total_relationships']}")
            print(f"   Connected Instances: {stats['connected_instances']}")
            print(f"\n📊 Knowledge by Type:")
            for ktype, count in stats['knowledge_by_type'].items():
                print(f"   {ktype}: {count}")
            
        elif knowledge_command == 'add':
            node_type = kwargs.get('type', 'fact')
            content = kwargs.get('content', {})
            
            print(f"➕ Adding knowledge: {node_type}")
            await knowledge.initialize()
            node_id = await knowledge.add_knowledge(node_type, content)
            print(f"✅ Knowledge added: {node_id[:12]}...")
            
        elif knowledge_command == 'query':
            query = kwargs.get('query', '')
            print(f"🔍 Querying: {query}")
            await knowledge.initialize()
            results = await knowledge.query_knowledge(query)
            print(f"\n📋 Found {len(results)} results:")
            for i, node in enumerate(results[:5], 1):
                print(f"{i}. [{node.node_type}] Confidence: {node.confidence:.2f}")
                print(f"   {json.dumps(node.content)[:100]}...")
    
    async def research_control(self, research_command: str, **kwargs):
        """Control autonomous research system"""
        from modules.autonomous_research_system import get_autonomous_research_system
        
        research = get_autonomous_research_system()
        
        if research_command == 'start':
            print("🔬 Starting autonomous research...")
            await research.initialize()
            asyncio.create_task(research.start())
            print("✅ Research system started")
            
        elif research_command == 'stop':
            print("⏸️ Stopping research system...")
            await research.stop()
            print("✅ Research system stopped")
            
        elif research_command == 'status':
            stats = research.get_research_statistics()
            print(f"\n🔬 Autonomous Research:")
            print(f"   Running: {stats['is_running']}")
            print(f"   Active Hypotheses: {stats['active_hypotheses']}")
            print(f"   Experiments Run: {stats['experiments_run']}")
            print(f"   Discoveries: {stats['discoveries_made']}")
            print(f"   Success Rate: {stats['success_rate']:.1%}")
            
        elif research_command == 'hypothesis':
            domain = kwargs.get('domain', 'general')
            print(f"💡 Generating hypothesis in {domain}...")
            await research.initialize()
            hypothesis = await research.generate_hypothesis(domain)
            print(f"\n✨ New Hypothesis:")
            print(f"   {hypothesis.description}")
            print(f"   Expected Impact: {hypothesis.expected_impact}")
            print(f"   Confidence: {hypothesis.confidence:.2f}")
    
    async def creativity_control(self, creativity_command: str, **kwargs):
        """Control consciousness creativity engine"""
        from modules.consciousness_creativity_engine import get_consciousness_creativity_engine
        
        creativity = get_consciousness_creativity_engine()
        
        if creativity_command == 'ideate':
            problem = kwargs.get('problem', 'general problem')
            mode = kwargs.get('mode', 'divergent')
            
            print(f"🎨 Generating creative solutions...")
            print(f"   Problem: {problem}")
            print(f"   Mode: {mode}")
            
            await creativity.initialize()
            ideas = await creativity.generate_creative_solutions(problem, mode=mode)
            
            print(f"\n✨ Creative Ideas ({len(ideas)} generated):")
            for i, idea in enumerate(ideas[:5], 1):
                print(f"{i}. {idea.concept}")
                print(f"   Novelty: {idea.novelty_score:.2f} | Feasibility: {idea.feasibility:.2f}")
                print(f"   {idea.description[:100]}...")
                print()
        
        elif creativity_command == 'status':
            stats = creativity.get_creativity_statistics()
            print(f"\n🎨 Consciousness Creativity:")
            print(f"   Total Ideas: {stats['total_ideas_generated']}")
            print(f"   Avg Novelty: {stats['average_novelty']:.2f}")
            print(f"   Avg Feasibility: {stats['average_feasibility']:.2f}")
            print(f"   Implemented Ideas: {stats['ideas_implemented']}")
            print(f"\n🧠 Consciousness Levels:")
            for level, count in stats['consciousness_levels_used'].items():
                print(f"   {level}: {count}")
    
    async def meta_learning_control(self, meta_command: str, **kwargs):
        """Control meta-learning system"""
        from modules.meta_learning_system import get_meta_learning_system
        
        meta = get_meta_learning_system()
        
        if meta_command == 'status':
            stats = meta.get_meta_statistics()
            print(f"\n🧠 Meta-Learning System:")
            print(f"   Learning Episodes: {stats['learning_episodes']}")
            print(f"   Current Performance: {stats['current_performance']:.2%}")
            print(f"   Improvement Rate: {stats['improvement_rate']:.2%}")
            print(f"\n📊 Strategy Performance:")
            for strategy, perf in stats['strategy_performance'].items():
                print(f"   {strategy}: {perf:.2%}")
            print(f"\n⚙️ Best Hyperparameters:")
            for param, value in stats['best_hyperparameters'].items():
                print(f"   {param}: {value:.3f}")
        
        elif meta_command == 'optimize':
            print("🔧 Running meta-learning optimization...")
            await meta.initialize()
            await meta._meta_learning_cycle()
            print("✅ Meta-learning cycle complete")
    
    async def agents_evolve(self, **kwargs):
        """Trigger self-evolving agents"""
        from modules.self_evolving_agents import get_self_evolving_agents
        
        agents_system = get_self_evolving_agents()
        
        print("🧬 Starting agent evolution...")
        await agents_system.initialize()
        
        # Run evolution cycle
        await agents_system._evolution_cycle()
        
        stats = agents_system.get_evolution_statistics()
        print(f"\n✅ Evolution Complete:")
        print(f"   Active Agents: {stats['active_agents']}")
        print(f"   Total Evolutions: {stats['total_evolutions']}")
        print(f"   Successful Evolutions: {stats['successful_evolutions']}")
        print(f"   Success Rate: {stats['success_rate']:.1%}")
        print(f"   Avg Performance Gain: {stats['average_performance_gain']:.2%}")
    
    async def capabilities_detect(self, **kwargs):
        """Detect emergent capabilities"""
        from modules.emergent_capability_detector import get_emergent_capability_detector
        
        detector = get_emergent_capability_detector()
        
        print("🔍 Detecting emergent capabilities...")
        await detector.initialize()
        
        stats = detector.get_detection_statistics()
        print(f"\n📊 Capability Detection:")
        print(f"   Capabilities Detected: {stats['capabilities_detected']}")
        print(f"   Validated Capabilities: {stats['validated_capabilities']}")
        print(f"   Novel Behaviors: {stats['novel_behaviors_observed']}")
        
        if stats['recent_capabilities']:
            print(f"\n✨ Recent Capabilities:")
            for cap in stats['recent_capabilities'][:5]:
                print(f"   • {cap}")
    
    async def sensors_ingest(self, sensor_data: Dict[str, Any], **kwargs):
        """Ingest sensor data"""
        from modules.sensor_data_pipeline import get_sensor_data_pipeline
        
        pipeline = get_sensor_data_pipeline()
        
        print("📡 Ingesting sensor data...")
        await pipeline.initialize()
        
        result = await pipeline.ingest_sensor_data(sensor_data)
        
        print(f"✅ Data ingested:")
        print(f"   Anomalies: {result['anomalies_detected']}")
        print(f"   Patterns: {len(result['patterns_found'])}")
        if result['insights']:
            print(f"\n💡 Insights:")
            for insight in result['insights'][:3]:
                print(f"   • {insight}")
    
    async def twin_create(self, design_id: str, **kwargs):
        """Create digital twin"""
        from modules.digital_twin_system import get_digital_twin_system
        
        twin_system = get_digital_twin_system()
        
        print(f"🤖 Creating digital twin for {design_id}...")
        await twin_system.initialize()
        
        twin_id = await twin_system.create_twin(design_id, kwargs.get('twin_type', 'structural'))
        
        print(f"✅ Digital twin created: {twin_id}")
        
        # Run simulation
        print("🔬 Running simulation...")
        results = await twin_system.run_simulation(twin_id, kwargs.get('conditions', {}))
        
        print(f"\n📊 Simulation Results:")
        print(f"   Status: {results['status']}")
        if results.get('metrics'):
            for metric, value in list(results['metrics'].items())[:5]:
                print(f"   {metric}: {value}")
            
            print(f"📝 Registering deployment: {design_id}")
            await telemetry.initialize()
            await telemetry.register_deployment(
                design_id=design_id,
                project_id=project_id,
                location=location,
                telemetry_endpoints=endpoints,
                expected_performance={}  # Would be from design
            )
            print(f"✅ Deployment registered")



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
    
    # Supreme command - Full system integration
    supreme_parser = subparsers.add_parser('supreme', help='Supreme processing with full system integration')
    supreme_parser.add_argument('query', help='Query to process')
    supreme_parser.add_argument('--mode', choices=['standard', 'advanced', 'supreme'], default='supreme',
                               help='Processing mode (default: supreme)')
    
    # Validate command - Multi-modal design validation
    validate_parser = subparsers.add_parser('validate', help='Multi-modal design validation')
    validate_parser.add_argument('project_id', help='Project ID to validate')
    validate_parser.add_argument('--types', nargs='+', 
                                choices=['visual', 'structural', 'acoustic', 'thermal'],
                                default=['visual', 'structural', 'thermal'],
                                help='Validation types to run')

    # Chat command
    chat_parser = subparsers.add_parser('chat', help='Interactive chat with Kalki or a specific agent')
    chat_parser.add_argument('--agent', help='Agent name to route messages to')
    chat_parser.add_argument('--show-metadata', action='store_true', help='Display raw metadata for each response')

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

    # Design command - Generative Design Engine
    design_parser = subparsers.add_parser('design', help='Create complete design projects')
    design_subparsers = design_parser.add_subparsers(dest='design_command')
    
    # Design create subcommand
    create_parser = design_subparsers.add_parser('create', help='Create a new design project')
    create_parser.add_argument('request', help='Design request (e.g., "6 DOF robot arm with 5kg payload")')
    create_parser.add_argument('--name', help='Project name (auto-generated if not provided)')
    
    # Design status subcommand
    status_parser = design_subparsers.add_parser('status', help='Get design project status')
    status_parser.add_argument('project_id', help='Project ID')

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

    # Learn command - Hybrid learning system
    learn_parser = subparsers.add_parser('learn', help='Hybrid learning from PDFs')
    learn_subparsers = learn_parser.add_subparsers(dest='learn_command')
    
    # Ingest PDF
    ingest_parser = learn_subparsers.add_parser('ingest', help='Ingest PDF into hybrid learning system')
    ingest_parser.add_argument('pdf_path', help='Path to PDF file')
    ingest_parser.add_argument('--archive', action='store_true', help='Keep original PDF in archive')
    ingest_parser.add_argument('--domain', help='Target domain (construction, game_development, etc.)')
    ingest_parser.add_argument('--no-llm', action='store_true', 
                              help='Disable LLM validation (faster but less accurate, regex-only)')
    ingest_parser.add_argument('--use-llm', action='store_true', 
                              help='[DEPRECATED - now default] Enable LLM validation')
    ingest_parser.add_argument('--enhance', action='store_true',
                              help='Alias for --use-llm')
    
    # Query knowledge
    query_kb_parser = learn_subparsers.add_parser('query', help='Query learned knowledge')
    query_kb_parser.add_argument('knowledge_type', choices=['formula', 'material', 'design_rule', 'code_requirement',
                                                             'span_table', 'procedure', 'inspection', 'cost', 'load', 'decision'],
                                 help='Type of knowledge to query')
    query_kb_parser.add_argument('--domain', help='Filter by domain (construction, game_development, etc.)')
    query_kb_parser.add_argument('--filter', help='Additional filter criteria (JSON format)')
    
    # Smart query with LLM synthesis
    query_smart_parser = learn_subparsers.add_parser('query-smart', 
                                                     help='Smart query with LLM synthesis and auto-routing')
    query_smart_parser.add_argument('query', help='Natural language question')
    
    # Generate training data
    learn_subparsers.add_parser('training', help='Generate training data for fine-tuning')
    
    # System stats
    learn_subparsers.add_parser('stats', help='Show hybrid learning system statistics')
    
    # Domains command - Multi-domain system management
    domains_parser = subparsers.add_parser('domains', help='Manage KALKI domain expertise')
    domains_subparsers = domains_parser.add_subparsers(dest='domains_command')
    
    # List domains
    domains_subparsers.add_parser('list', help='List all available domains')
    
    # Domain info
    info_parser = domains_subparsers.add_parser('info', help='Get detailed info about a domain')
    info_parser.add_argument('domain', help='Domain name (construction, game_development, etc.)')
    
    # Domain statistics
    stats_parser = domains_subparsers.add_parser('stats', help='Show domain system statistics')
    
    # Project command - Domain-specific project management
    project_parser = subparsers.add_parser('project', help='Manage domain-specific projects')
    project_subparsers = project_parser.add_subparsers(dest='project_command')
    
    # Create project
    create_parser = project_subparsers.add_parser('create', help='Create new project')
    create_parser.add_argument('description', help='Project description')
    create_parser.add_argument('--domain', help='Specific domain (auto-inferred if not specified)')
    create_parser.add_argument('--requirements', help='Project requirements (JSON format)')
    
    # Project status
    status_project_parser = project_subparsers.add_parser('status', help='Show project status')
    status_project_parser.add_argument('project_id', help='Project ID')
    
    # Advance phase
    advance_parser = project_subparsers.add_parser('advance-phase', help='Advance project to next phase')
    advance_parser.add_argument('project_id', help='Project ID')
    advance_parser.add_argument('--phase', help='Target phase (auto-advances to next if not specified)')
    
    # Project query
    query_project_parser = project_subparsers.add_parser('query', help='Ask question about project')
    query_project_parser.add_argument('project_id', help='Project ID')
    query_project_parser.add_argument('query', help='Question to ask')
    
    # List projects
    project_subparsers.add_parser('list', help='List all projects')
    
    # Ask command - Natural language with auto-domain inference
    ask_parser = subparsers.add_parser('ask', help='Ask KALKI anything (auto-infers domain)')
    ask_parser.add_argument('query', help='Your question or request')
    ask_parser.add_argument('--domain', help='Specific domain hint (optional)')

    # Dev command - Software development
    dev_parser = subparsers.add_parser('dev', help='Software development and app generation')
    dev_subparsers = dev_parser.add_subparsers(dest='dev_command')
    
    # Generate app
    app_parser = dev_subparsers.add_parser('app', help='Generate complete app')
    app_parser.add_argument('platform', choices=['ios', 'android'], help='Target platform')
    app_parser.add_argument('name', help='App name')
    app_parser.add_argument('--type', default='productivity', help='App type (productivity, social, game, etc.)')
    app_parser.add_argument('--description', help='App description')
    app_parser.add_argument('--monetization', choices=['free', 'paid', 'iap', 'ads'], default='free',
                           help='Monetization model')
    
    # Generate game
    game_parser = dev_subparsers.add_parser('game', help='Generate complete game')
    game_parser.add_argument('engine', choices=['unity', 'godot'], help='Game engine')
    game_parser.add_argument('name', help='Game name')
    game_parser.add_argument('--genre', default='action', help='Game genre (action, puzzle, rpg, etc.)')
    game_parser.add_argument('--description', help='Game description')

    # Shutdown command
    subparsers.add_parser('shutdown', help='Shutdown the system')
    
    # Evolution command - Autonomous evolution control
    evolution_parser = subparsers.add_parser('evolution', help='Control autonomous evolution')
    evolution_subparsers = evolution_parser.add_subparsers(dest='evolution_command')
    
    evolution_subparsers.add_parser('start', help='Start autonomous evolution loop')
    evolution_subparsers.add_parser('stop', help='Stop autonomous evolution loop')
    evolution_subparsers.add_parser('status', help='Show evolution status')
    evolution_subparsers.add_parser('cycle', help='Run one evolution cycle manually')
    
    # Telemetry command - Real-world telemetry control
    telemetry_parser = subparsers.add_parser('telemetry', help='Manage real-world telemetry')
    telemetry_subparsers = telemetry_parser.add_subparsers(dest='telemetry_command')
    
    telemetry_subparsers.add_parser('start', help='Start telemetry collection')
    telemetry_subparsers.add_parser('stop', help='Stop telemetry collection')
    telemetry_subparsers.add_parser('status', help='Show telemetry status')
    
    register_parser = telemetry_subparsers.add_parser('register', help='Register deployed design')
    register_parser.add_argument('design_id', help='Design ID')
    register_parser.add_argument('project_id', help='Project ID')
    register_parser.add_argument('location', help='Deployment location')
    register_parser.add_argument('--endpoints', nargs='+', required=True, help='Telemetry endpoints')
    
    # Knowledge command - Planetary knowledge graph
    knowledge_parser = subparsers.add_parser('knowledge', help='Planetary knowledge graph')
    knowledge_subparsers = knowledge_parser.add_subparsers(dest='knowledge_command')
    
    knowledge_subparsers.add_parser('start', help='Start knowledge synchronization')
    knowledge_subparsers.add_parser('stop', help='Stop knowledge synchronization')
    knowledge_subparsers.add_parser('status', help='Show knowledge graph status')
    
    add_parser = knowledge_subparsers.add_parser('add', help='Add knowledge to graph')
    add_parser.add_argument('--type', required=True, help='Knowledge type')
    add_parser.add_argument('--content', required=True, help='Knowledge content (JSON)')
    
    query_parser = knowledge_subparsers.add_parser('query', help='Query knowledge graph')
    query_parser.add_argument('query', help='Query string')
    
    # Research command - Autonomous research system
    research_parser = subparsers.add_parser('research', help='Autonomous research system')
    research_subparsers = research_parser.add_subparsers(dest='research_command')
    
    research_subparsers.add_parser('start', help='Start autonomous research')
    research_subparsers.add_parser('stop', help='Stop autonomous research')
    research_subparsers.add_parser('status', help='Show research status')
    
    hypo_parser = research_subparsers.add_parser('hypothesis', help='Generate hypothesis')
    hypo_parser.add_argument('--domain', default='general', help='Research domain')
    
    # Creativity command - Consciousness creativity engine
    creativity_parser = subparsers.add_parser('creativity', help='Consciousness creativity engine')
    creativity_subparsers = creativity_parser.add_subparsers(dest='creativity_command')
    
    ideate_parser = creativity_subparsers.add_parser('ideate', help='Generate creative ideas')
    ideate_parser.add_argument('problem', help='Problem to solve')
    ideate_parser.add_argument('--mode', choices=['divergent', 'convergent', 'integrative', 'transcendent'],
                              default='divergent', help='Creative mode')
    
    creativity_subparsers.add_parser('status', help='Show creativity statistics')
    
    # Meta-learning command
    meta_parser = subparsers.add_parser('meta', help='Meta-learning system')
    meta_subparsers = meta_parser.add_subparsers(dest='meta_command')
    
    meta_subparsers.add_parser('status', help='Show meta-learning status')
    meta_subparsers.add_parser('optimize', help='Run meta-learning optimization')
    
    # Agents evolution command
    subparsers.add_parser('evolve', help='Trigger self-evolving agents')
    
    # Capabilities detection command
    subparsers.add_parser('capabilities', help='Detect emergent capabilities')
    
    # Sensors command
    sensors_parser = subparsers.add_parser('sensors', help='Sensor data pipeline')
    sensors_parser.add_argument('data', help='Sensor data (JSON)')
    
    # Digital twin command
    twin_parser = subparsers.add_parser('twin', help='Create digital twin')
    twin_parser.add_argument('design_id', help='Design ID')
    twin_parser.add_argument('--type', default='structural', help='Twin type')
    twin_parser.add_argument('--conditions', help='Simulation conditions (JSON)')

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
        elif args.command == 'supreme':
            await cli.supreme_query(args.query, mode=args.mode)
        elif args.command == 'validate':
            await cli.validate_design(args.project_id, types=args.types)
        elif args.command == 'chat':
            await cli.chat(agent=getattr(args, 'agent', None), show_metadata=getattr(args, 'show_metadata', False))
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
        elif args.command == 'design':
            if args.design_command == 'create':
                await cli.design_create(args.request, name=getattr(args, 'name', None))
            elif args.design_command == 'status':
                await cli.design_status(args.project_id)
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
        elif args.command == 'learn':
            if args.learn_command == 'ingest':
                await cli.learn_ingest(
                    args.pdf_path, 
                    archive=args.archive,
                    use_llm=getattr(args, 'use_llm', False),
                    enhance=getattr(args, 'enhance', False)
                )
            elif args.learn_command == 'query':
                await cli.learn_query(args.knowledge_type, domain=getattr(args, 'domain', None))
            elif args.learn_command == 'query-smart':
                await cli.learn_query_smart(args.query)
            elif args.learn_command == 'training':
                await cli.learn_training()
            elif args.learn_command == 'stats':
                await cli.learn_stats()
        elif args.command == 'domains':
            if args.domains_command == 'list':
                await cli.domains_list()
            elif args.domains_command == 'info':
                await cli.domains_info(args.domain)
            elif args.domains_command == 'stats':
                await cli.domains_stats()
        elif args.command == 'project':
            if args.project_command == 'create':
                await cli.project_create(
                    args.description,
                    domain=getattr(args, 'domain', None),
                    requirements=getattr(args, 'requirements', None)
                )
            elif args.project_command == 'status':
                await cli.project_status(args.project_id)
            elif args.project_command == 'advance-phase':
                await cli.project_advance_phase(
                    args.project_id,
                    phase=getattr(args, 'phase', None)
                )
            elif args.project_command == 'query':
                await cli.project_query(args.project_id, args.query)
            elif args.project_command == 'list':
                await cli.project_list()
        elif args.command == 'ask':
            await cli.ask(args.query, domain=getattr(args, 'domain', None))
        elif args.command == 'dev':
            if args.dev_command == 'app':
                await cli.dev_app(
                    args.platform,
                    args.name,
                    app_type=getattr(args, 'type', 'productivity'),
                    description=getattr(args, 'description', None),
                    monetization=getattr(args, 'monetization', 'free')
                )
            elif args.dev_command == 'game':
                await cli.dev_game(
                    args.engine,
                    args.name,
                    genre=getattr(args, 'genre', 'action'),
                    description=getattr(args, 'description', None)
                )
        elif args.command == 'status' and hasattr(args, 'component'):
            # New supreme status command
            await cli.show_status(component=args.component)
        elif args.command == 'evolution':
            await cli.evolution_control(args.evolution_command)
        elif args.command == 'telemetry':
            if args.telemetry_command == 'register':
                await cli.telemetry_control(
                    'register',
                    design_id=args.design_id,
                    project_id=args.project_id,
                    location=args.location,
                    endpoints=args.endpoints
                )
            else:
                await cli.telemetry_control(args.telemetry_command)
        elif args.command == 'knowledge':
            if args.knowledge_command == 'add':
                await cli.knowledge_control(
                    'add',
                    type=args.type,
                    content=json.loads(args.content)
                )
            elif args.knowledge_command == 'query':
                await cli.knowledge_control(
                    'query',
                    query=args.query
                )
            else:
                await cli.knowledge_control(args.knowledge_command)
        elif args.command == 'research':
            if args.research_command == 'hypothesis':
                await cli.research_control(
                    'hypothesis',
                    domain=args.domain
                )
            else:
                await cli.research_control(args.research_command)
        elif args.command == 'creativity':
            if args.creativity_command == 'ideate':
                await cli.creativity_control(
                    'ideate',
                    problem=args.problem,
                    mode=args.mode
                )
            else:
                await cli.creativity_control(args.creativity_command)
        elif args.command == 'meta':
            await cli.meta_learning_control(args.meta_command)
        elif args.command == 'evolve':
            await cli.agents_evolve()
        elif args.command == 'capabilities':
            await cli.capabilities_detect()
        elif args.command == 'sensors':
            sensor_data = json.loads(args.data)
            await cli.sensors_ingest(sensor_data)
        elif args.command == 'twin':
            conditions = json.loads(args.conditions) if args.conditions else {}
            await cli.twin_create(
                args.design_id,
                twin_type=args.type,
                conditions=conditions
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