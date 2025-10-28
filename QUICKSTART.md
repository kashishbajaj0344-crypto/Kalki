# Kalki v2.3 - Quick Start Guide

## Welcome to Kalki v2.3! 🚀

This is a functional, multi-phase AI agent system with 17+ specialized agents covering planning, reasoning, creativity, safety, ethics, and multi-modal processing.

---

## Installation (2 minutes)

### 1. Install Python Dependencies

```bash
# Core dependencies (required)
pip install asyncio

# Optional: For full functionality with existing modules
pip install openai chromadb pdfplumber langchain-community python-dotenv keyring
```

### 2. Set Environment Variables (Optional)

```bash
# Only needed for SearchAgent and ReasoningAgent
export OPENAI_API_KEY="your-api-key-here"
```

---

## Quick Test (30 seconds)

### Run the Verification Test

```bash
python test_agent_system.py
```

Expected output:
```
✓ All imports successful
✓ Event bus working
✓ Agent registered: PlannerAgent
✓ Agent executed successfully
✓ All tests passed! System is functional.
```

---

## See It In Action (2 minutes)

### Run the Demo

```bash
python demo_agents.py
```

This demonstrates:
- 📋 Planning & task decomposition
- 💡 Creative ideation & idea fusion
- ⚖️ Ethics review & safety verification
- 🛡️ Risk assessment
- 🧠 Meta-reasoning & hypothesis generation
- 💾 Memory storage & retrieval
- 🎯 Multi-modal sensor fusion

---

## Interactive Mode (Start Exploring!)

### Launch Interactive CLI

```bash
python kalki_agents.py
```

### Try These Commands

```
kalki> status
  # Shows system status and all agents

kalki> plan Build a recommendation system
  # Creates a multi-step execution plan

kalki> ideate sustainable transportation
  # Generates creative ideas

kalki> ethics Deploy facial recognition in public spaces
  # Reviews ethical implications

kalki> help
  # Shows all available commands
```

---

## Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `status` | System status & health | `status` |
| `list` | List all agents | `list` |
| `plan <goal>` | Create execution plan | `plan launch startup` |
| `ideate <topic>` | Generate ideas | `ideate quantum computing` |
| `ethics <action>` | Ethics review | `ethics collect user data` |
| `help` | Show help | `help` |
| `quit` | Exit | `quit` |

---

## System Architecture

### 17+ Specialized Agents

**Core Agents:**
- PlannerAgent - Task planning & decomposition
- ReasoningAgent - Multi-step reasoning
- MemoryAgent - Knowledge storage
- SearchAgent - Semantic search

**Cognitive Agents:**
- CreativeAgent - Ideation & synthesis
- MetaHypothesisAgent - Meta-reasoning
- OptimizationAgent - Self-improvement
- FeedbackAgent - Performance monitoring

**Safety Agents:**
- EthicsAgent - Ethical oversight
- RiskAssessmentAgent - Risk evaluation
- SimulationVerifierAgent - Safety verification

**Multi-Modal Agents:**
- VisionAgent - Visual processing
- AudioAgent - Audio analysis
- SensorFusionAgent - Multi-sensor integration
- ARInsightAgent - Augmented reality

### 53 Agent Capabilities

Covering all 20 phases:
- ✅ Foundation & Ingestion
- ✅ Search & Vectorization
- ✅ Orchestration & Planning
- ✅ Memory & Reasoning
- ✅ Meta-Cognition
- ✅ Safety & Ethics
- ✅ Multi-Modal Processing
- ✅ Creative Synthesis
- ✅ Self-Optimization
- And more...

---

## Programmatic Usage

### Basic Example

```python
import asyncio
from agents import AgentManager, EventBus
from agents.core import PlannerAgent
from agents.base_agent import AgentCapability

async def main():
    # Initialize
    manager = AgentManager(EventBus())
    
    # Register agent
    planner = PlannerAgent()
    await manager.register_agent(planner)
    
    # Execute task
    result = await manager.execute_by_capability(
        AgentCapability.PLANNING,
        {
            "action": "plan",
            "params": {"goal": "Build AI system"}
        }
    )
    
    print(result)
    
    # Cleanup
    await manager.shutdown_all()

asyncio.run(main())
```

### Creating Custom Agents

```python
from agents.base_agent import BaseAgent, AgentCapability

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MyAgent",
            capabilities=[AgentCapability.CUSTOM],
            description="My custom agent"
        )
    
    async def initialize(self):
        return True
    
    async def execute(self, task):
        return {"status": "success", "result": "done"}
    
    async def shutdown(self):
        return True
```

---

## Features

### ✅ Event-Driven Architecture
- Asynchronous pub/sub communication
- Agent coordination via EventBus
- Event history and auditing

### ✅ Safety Built-In
- Ethics review for all actions
- Risk assessment and mitigation
- Simulation verification

### ✅ Multi-Modal Support
- Vision processing
- Audio analysis
- Sensor fusion
- AR/VR capabilities

### ✅ Self-Improvement
- Meta-reasoning
- Performance feedback
- Automatic optimization
- Continuous learning

### ✅ Extensible Design
- Easy to add new agents
- Capability-based routing
- Dependency management
- Health monitoring

---

## Documentation

📖 **Full Documentation**: `README_AGENTS.md`

Contains:
- Complete architecture overview
- All 53 agent capabilities
- API reference
- Extension guide
- Best practices

---

## Next Steps

1. ✅ **Run the demo** - See all capabilities in action
2. ✅ **Try interactive mode** - Explore with CLI commands
3. ✅ **Read the docs** - Understand the architecture
4. ✅ **Create custom agents** - Extend the system
5. ✅ **Build applications** - Use the programmatic API

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'dotenv'"

Some agents require optional dependencies:

```bash
pip install python-dotenv openai langchain-community
```

Core agents (Planner, Creative, Ethics, etc.) work without these.

### "OPENAI_API_KEY not found"

Only needed for SearchAgent and ReasoningAgent:

```bash
export OPENAI_API_KEY="your-key"
```

---

## System Status

Run `python test_agent_system.py` to verify:

```
✓ All imports successful
✓ Event bus working
✓ Agent manager created
✓ Agent registered: PlannerAgent
✓ Agent executed successfully
✓ Capability-based execution working
✓ Registered 4/4 additional agents
✓ All tests passed! System is functional.
```

---

## Support & Contributing

- 📝 Issues: Report bugs or request features
- 💬 Questions: Ask in discussions
- 🤝 Contribute: Create pull requests
- 📖 Docs: Improve documentation

---

**🎉 You're ready to build with Kalki v2.3!**

Start with: `python demo_agents.py` or `python kalki_agents.py`
