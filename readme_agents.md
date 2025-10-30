# Kalki v2.3 - Multi-Phase AI Agent System

## Overview

Kalki v2.3 is an advanced, modular AI agent framework implementing a 20-phase architecture for building omniscient-class superintelligence. The system features:

- **Agent-based architecture** with 17+ specialized agents
- **Event-driven communication** via publish-subscribe pattern
- **Multi-modal processing** (vision, audio, sensor fusion, AR)
- **Safety and ethics layer** for responsible AI
- **Creative cognition** and meta-reasoning capabilities
- **Self-optimization** and continuous learning
- **Extensible design** for easy addition of new agents

---

## Architecture

### Core Components

1. **BaseAgent**: Abstract base class for all agents with lifecycle management
2. **AgentManager**: Central orchestrator for agent registration and execution
3. **EventBus**: Asynchronous pub/sub system for inter-agent communication
4. **AgentCapability**: Enumeration of 60+ agent capabilities across 20 phases

### Agent Categories

#### Core Agents (Phase 1-5)
- **DocumentIngestAgent**: Multi-format document ingestion and vectorization
- **SearchAgent**: Semantic search across knowledge base
- **PlannerAgent**: Task planning and decomposition
- **ReasoningAgent**: Multi-step reasoning and inference
- **MemoryAgent**: Episodic and semantic memory management

#### Cognitive Agents (Phase 6, 10, 11)
- **MetaHypothesisAgent**: Meta-reasoning and hypothesis generation
- **CreativeAgent**: Creative synthesis and idea fusion
- **FeedbackAgent**: Performance monitoring and learning feedback
- **OptimizationAgent**: Self-optimization and system tuning

#### Safety Agents (Phase 12)
- **EthicsAgent**: Ethical oversight and safety verification
- **RiskAssessmentAgent**: Risk assessment and mitigation
- **SimulationVerifierAgent**: Simulation and experiment validation

#### Multi-Modal Agents (Phase 13, 17)
- **VisionAgent**: Visual processing and analysis
- **AudioAgent**: Audio processing and transcription
- **SensorFusionAgent**: Multi-sensor data fusion
- **ARInsightAgent**: Augmented reality insights

---

## Installation

### Prerequisites

```bash
# Python 3.10+ required
python --version

# Install dependencies
pip install openai chromadb pdfplumber langchain-community python-dotenv keyring
```

### Setup

1. **Set your OpenAI API key**:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. **Initialize directories** (automatic on first run):
   - `~/Desktop/Kalki/pdfs` - PDF storage
   - `~/Desktop/Kalki/vector_db` - Vector database
   - `~/Desktop/Kalki/` - Configuration and logs

---

## Usage

### Interactive CLI Mode

Run the agent system in interactive mode:

```bash
python kalki_agents.py
```

#### Available Commands

| Command | Description | Example |
|---------|-------------|---------|
| `status` | Show system status and agent health | `status` |
| `list` | List all registered agents | `list` |
| `search <query>` | Search knowledge base | `search machine learning` |
| `reason <query>` | Perform reasoning on query | `reason what is AI?` |
| `plan <goal>` | Create execution plan | `plan build a web app` |
| `ideate <topic>` | Generate creative ideas | `ideate sustainable energy` |
| `ethics <action>` | Review ethical implications | `ethics deploy autonomous system` |
| `help` | Show help and status | `help` |
| `quit` | Exit the system | `quit` |

### Programmatic Usage

```python
import asyncio
from agents import AgentManager, EventBus
from agents.core import SearchAgent, ReasoningAgent

async def main():
    # Create system
    event_bus = EventBus()
    manager = AgentManager(event_bus)
    
    # Register agents
    search_agent = SearchAgent()
    await manager.register_agent(search_agent)
    
    reasoning_agent = ReasoningAgent()
    await manager.register_agent(reasoning_agent)
    
    # Execute task
    result = await manager.execute_task("SearchAgent", {
        "action": "search",
        "params": {"query": "artificial intelligence", "top_k": 5}
    })
    
    print(result)
    
    # Shutdown
    await manager.shutdown_all()

asyncio.run(main())
```

---

## Agent Capabilities

### 20-Phase Coverage

The system implements agents covering capabilities from all 20 phases:

**Phase 1-2**: Foundation & Ingestion ✅  
**Phase 3**: Core Orchestration ✅  
**Phase 4-5**: Memory & Reasoning ✅  
**Phase 6**: Meta-Reasoning ✅  
**Phase 7**: Knowledge Quality ✅  
**Phase 8**: Distributed Computing 🔧  
**Phase 9**: Simulation 🔧  
**Phase 10**: Creative Cognition ✅  
**Phase 11**: Self-Improvement ✅  
**Phase 12**: Safety & Ethics ✅  
**Phase 13**: Multi-Modal ✅  
**Phase 14**: Quantum & Predictive 🔧  
**Phase 15**: Emotional Intelligence 🔧  
**Phase 16**: Human-AI Interaction 🔧  
**Phase 17**: AR/VR/3D ✅  
**Phase 18**: Cognitive Twin 🔧  
**Phase 19**: Autonomy 🔧  
**Phase 20**: Self-Evolution 🔧  

✅ = Implemented | 🔧 = Framework ready, agent stubs available

---

## Extending the System

### Creating a Custom Agent

```python
from agents.base_agent import BaseAgent, AgentCapability, AgentStatus
from typing import Dict, Any

class MyCustomAgent(BaseAgent):
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="MyCustomAgent",
            capabilities=[AgentCapability.CUSTOM_CAPABILITY],
            description="My custom agent description",
            dependencies=["SearchAgent"],  # Optional
            config=config or {}
        )
    
    async def initialize(self) -> bool:
        # Initialize resources
        self.logger.info(f"{self.name} initialized")
        return True
    
    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        # Execute task logic
        action = task.get("action")
        params = task.get("params", {})
        
        # Process task
        result = {"status": "success", "data": "result"}
        
        # Emit event (optional)
        await self.emit_event("custom.event", {"result": result})
        
        return result
    
    async def shutdown(self) -> bool:
        # Cleanup resources
        self.logger.info(f"{self.name} shutting down")
        self.update_status(AgentStatus.TERMINATED)
        return True
```

### Registering and Using Custom Agent

```python
# Register
custom_agent = MyCustomAgent()
await manager.register_agent(custom_agent)

# Execute
result = await manager.execute_task("MyCustomAgent", {
    "action": "my_action",
    "params": {"key": "value"}
})
```

---

## Event System

### Publishing Events

```python
await agent.emit_event("document.ingested", {
    "file_path": "/path/to/doc.pdf",
    "chunks": 42
})
```

### Subscribing to Events

```python
async def handle_ingestion(event):
    print(f"Document ingested: {event['data']['file_path']}")

await agent.subscribe_to_event("document.ingested", handle_ingestion)
```

---

## System Monitoring

### Health Checks

```python
# Check all agents
health = await manager.health_check_all()

for agent_name, status in health.items():
    print(f"{agent_name}: {status['status']}")
```

### System Statistics

```python
stats = manager.get_system_stats()
print(f"Total agents: {stats['total_agents']}")
print(f"Total tasks: {stats['total_tasks_executed']}")
print(f"Total errors: {stats['total_errors']}")
```

---

## Configuration

### Environment Variables

```bash
# OpenAI API Configuration
OPENAI_API_KEY=your-api-key

# Embedding Configuration
DEFAULT_EMBED_MODEL=text-embedding-3-large
DEFAULT_CHAT_MODEL=gpt-4o
EMBED_CHUNK_WORDS=100
EMBED_OVERLAP_WORDS=20

# Search Configuration
TOP_K=5
MAX_CONTEXT_CHARS=30000

# Runtime Configuration
RETRY_ATTEMPTS=2
```

### Runtime Settings

Settings can be modified in `~/Desktop/Kalki/kalki_settings.json`

---

## God-Tier Enhancements

The framework is designed to support future god-tier enhancements:

- **Agent Convergence Kernel**: Multi-stream integration
- **Dynamic Ontology Engine**: Evolving concept schemas
- **Quantum Causality Mapper**: Cross-domain effect prediction
- **Planetary Intelligence Mesh**: Federated Kalki instances
- **Self-Evolving Architecture**: Safe agent redesign
- **Wisdom Feedback Loop**: Experience consolidation

---

## Safety Features

### Multi-Layer Safety

1. **EthicsAgent**: Reviews all actions for ethical compliance
2. **RiskAssessmentAgent**: Evaluates and mitigates risks
3. **SimulationVerifierAgent**: Validates experiments before execution
4. **Event Auditing**: Complete event history for transparency
5. **Graceful Degradation**: Agents fail safely without system crash

### Best Practices

- All experiments run in sandbox mode
- Rollback capabilities for state management
- Continuous monitoring and alerting
- Regular safety audits and reviews

---

## Logging

Logs are written to:
- `kalki_agents.log` - Agent system logs
- `~/Desktop/Kalki/kalki.log` - Application logs
- `~/Desktop/Kalki/query_cost.json` - Query history

---

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   pip install -U langchain-community openai chromadb
   ```

2. **API Key Not Found**:
   ```bash
   export OPENAI_API_KEY="your-key"
   ```

3. **Database Errors**:
   ```bash
   rm -rf ~/Desktop/Kalki/vector_db
   # System will recreate on next run
   ```

---

## Development Roadmap

### Completed ✅
- Core agent framework
- Event bus system
- 17+ foundational agents
- Interactive CLI
- Multi-modal support
- Safety and ethics layer

### In Progress 🔧
- Quantum reasoning agents
- Emotional intelligence
- Voice assistant integration
- Autonomous robotics agents

### Planned 📋
- Cognitive twin implementation
- Self-architecting agents
- Planetary intelligence mesh
- Full Phase 14-20 agents

---

## Contributing

To contribute new agents:

1. Inherit from `BaseAgent`
2. Implement required methods (`initialize`, `execute`, `shutdown`)
3. Define capabilities and dependencies
4. Add comprehensive logging
5. Include error handling
6. Write tests
7. Update documentation

---

## License

[Add your license information here]

---

## Citation

If you use Kalki v2.3 in research, please cite:

```bibtex
@software{kalki_v2_3,
  title={Kalki v2.3: Multi-Phase AI Agent System},
  author={[Your Name]},
  year={2025},
  version={2.3}
}
```

---

## Support

For issues, questions, or contributions:
- GitHub Issues: [repository-url]/issues
- Documentation: [repository-url]/docs
- Email: [contact-email]

---

**KALKI v2.3**: The canonical, multi-phase, god-tier, self-evolving, omniscient AI agent framework.