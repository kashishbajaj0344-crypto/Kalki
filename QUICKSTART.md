# Kalki v2.3 - Quick Start Guide

## Installation

```bash
# Clone the repository
git clone https://github.com/kashishbajaj0344-crypto/Kalki.git
cd Kalki

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key (for RAG integration)
export OPENAI_API_KEY="your-key-here"
```

## Run Tests

```bash
# Test all agents
python test_agents.py

# Expected output: All tests passing ✓
```

## Try Examples

```bash
# Run usage examples
python examples.py

# Run complete workflow demo
python demo_workflow.py
```

## Basic Usage

### 1. Session Management
```python
from modules.agents.phase4_memory import SessionAgent

session_agent = SessionAgent()
session_agent.initialize()

# Create a session
result = session_agent.execute({
    "action": "create",
    "user_id": "your_user_id",
    "metadata": {"project": "AI Research"}
})
session_id = result["session_id"]
print(f"Session created: {session_id}")
```

### 2. Store & Recall Memories
```python
from modules.agents.phase4_memory import MemoryAgent

memory_agent = MemoryAgent()
memory_agent.initialize()

# Store episodic memory
memory_agent.execute({
    "action": "store",
    "type": "episodic",
    "event": {
        "summary": "User asked about AI safety",
        "details": "Discussed latest developments in AI alignment"
    }
})

# Recall memories
result = memory_agent.execute({
    "action": "recall",
    "type": "episodic",
    "limit": 5
})
print(f"Found {len(result['memories'])} memories")
```

### 3. Plan and Execute Tasks
```python
from modules.agents.phase5_reasoning import PlannerAgent, OrchestratorAgent

# Create a plan
planner = PlannerAgent()
plan_result = planner.execute({
    "action": "create_plan",
    "task": "Analyze research papers and create summary",
    "context": {"domain": "machine_learning"}
})
plan = plan_result["plan"]

# Execute the plan
orchestrator = OrchestratorAgent()
workflow_id = f"workflow_{plan['plan_id']}"

orchestrator.execute({
    "action": "create_workflow",
    "workflow_id": workflow_id,
    "plan": plan
})

result = orchestrator.execute({
    "action": "execute_workflow",
    "workflow_id": workflow_id
})
print(f"Workflow status: {result['workflow']['status']}")
```

### 4. Ethics & Safety Checks
```python
from modules.agents.phase12_safety import EthicsAgent, RiskAssessmentAgent

# Check ethics
ethics = EthicsAgent()
result = ethics.execute({
    "action": "evaluate",
    "action_to_evaluate": {
        "type": "data_processing",
        "description": "Process user data"
    }
})
print(f"Ethics score: {result['evaluation']['ethical_score']}")
print(f"Recommendation: {result['evaluation']['recommendation']}")

# Assess risk
risk = RiskAssessmentAgent()
result = risk.execute({
    "action": "assess",
    "action_to_assess": {"type": "data_access"}
})
print(f"Risk level: {result['assessment']['severity']}")
```

### 5. Creative Idea Generation
```python
from modules.agents.phase10_creative import CreativeAgent, IdeaFusionAgent

# Generate ideas
creative = CreativeAgent()
idea1 = creative.execute({
    "action": "generate",
    "domain": "technology"
})

idea2 = creative.execute({
    "action": "generate",
    "domain": "biology"
})

# Fuse ideas
fusion = IdeaFusionAgent()
result = fusion.execute({
    "action": "fuse",
    "ideas": [idea1["idea"], idea2["idea"]]
})
print(f"Fused idea: {result['fusion']['description']}")
```

### 6. Run Simulations
```python
from modules.agents.phase9_simulation import SimulationAgent

sim = SimulationAgent()

# Create simulation
result = sim.execute({
    "action": "create",
    "sim_type": "physics",
    "parameters": {"mass": 10, "velocity": 50}
})

# Run simulation
result = sim.execute({
    "action": "run",
    "sim_id": result["sim_id"]
})
print(f"Simulation outcome: {result['simulation']['results']['outcome']}")
```

### 7. Monitor Performance
```python
from modules.agents.phase6_adaptive import PerformanceMonitorAgent

monitor = PerformanceMonitorAgent()

# Record metrics
monitor.execute({
    "action": "record",
    "metric_name": "query_latency",
    "value": 0.5
})

# Get statistics
result = monitor.execute({
    "action": "stats",
    "metric_name": "query_latency"
})
print(f"Average latency: {result['stats']['avg']:.3f}s")
```

## Common Workflows

### Complete Research Workflow
```python
from modules.agents.phase4_memory import SessionAgent, MemoryAgent
from modules.agents.phase5_reasoning import PlannerAgent, OrchestratorAgent
from modules.agents.phase12_safety import EthicsAgent

# 1. Create session
session_agent = SessionAgent()
session_agent.initialize()
session_id = session_agent.execute({
    "action": "create",
    "user_id": "researcher"
})["session_id"]

# 2. Plan research task
planner = PlannerAgent()
plan = planner.execute({
    "action": "create_plan",
    "task": "Analyze AI safety papers"
})["plan"]

# 3. Ethics check
ethics = EthicsAgent()
ethics_ok = ethics.execute({
    "action": "evaluate",
    "action_to_evaluate": {"type": "research_analysis"}
})["evaluation"]["recommendation"] == "approve"

# 4. Execute if ethical
if ethics_ok:
    orchestrator = OrchestratorAgent()
    workflow_id = f"workflow_{plan['plan_id']}"
    orchestrator.execute({
        "action": "create_workflow",
        "workflow_id": workflow_id,
        "plan": plan
    })
    result = orchestrator.execute({
        "action": "execute_workflow",
        "workflow_id": workflow_id
    })
    
    # 5. Store results in memory
    memory = MemoryAgent()
    memory.initialize()
    memory.execute({
        "action": "store",
        "type": "episodic",
        "event": {
            "summary": "Completed AI safety research",
            "status": result["workflow"]["status"]
        }
    })
```

## Available Agents

| Agent | Purpose | Phase |
|-------|---------|-------|
| SessionAgent | Session management | 4 |
| MemoryAgent | Episodic/semantic memory | 4 |
| PlannerAgent | Task planning | 5 |
| OrchestratorAgent | Workflow coordination | 5 |
| ComputeOptimizerAgent | Resource allocation | 5 |
| CopilotAgent | Interactive assistance | 5 |
| MetaHypothesisAgent | Hypothesis generation | 6 |
| FeedbackAgent | Learning loops | 6 |
| PerformanceMonitorAgent | Metrics tracking | 6 |
| ConflictDetectionAgent | Conflict resolution | 6 |
| KnowledgeLifecycleAgent | Version management | 7 |
| RollbackManager | Checkpointing | 7 |
| ComputeClusterAgent | Cluster management | 8 |
| LoadBalancerAgent | Load balancing | 8 |
| SelfHealingAgent | Fault tolerance | 8 |
| SimulationAgent | Simulations | 9 |
| SandboxExperimentAgent | Safe testing | 9 |
| HypotheticalTestingLoop | Scenario testing | 9 |
| CreativeAgent | Idea generation | 10 |
| PatternRecognitionAgent | Pattern discovery | 10 |
| IdeaFusionAgent | Cross-domain synthesis | 10 |
| AutoFineTuneAgent | Model optimization | 11 |
| RecursiveKnowledgeGenerator | Knowledge expansion | 11 |
| AutonomousCurriculumDesigner | Curriculum design | 11 |
| EthicsAgent | Ethical validation | 12 |
| RiskAssessmentAgent | Risk analysis | 12 |
| OmniEthicsEngine | Consequence simulation | 12 |

## Next Steps

1. **Run the tests**: `python test_agents.py`
2. **Try examples**: `python examples.py`
3. **See complete workflow**: `python demo_workflow.py`
4. **Read full docs**: Check `README.md` and `IMPLEMENTATION_SUMMARY.md`
5. **Integrate with your app**: See `agent_integration.py` for RAG integration

## Support

- 📖 Full documentation: `README.md`
- 🧪 Test suite: `test_agents.py`
- 💡 Examples: `examples.py`
- 🔄 Workflow demo: `demo_workflow.py`
- 📊 Implementation details: `IMPLEMENTATION_SUMMARY.md`

## License

See LICENSE file for details.

---

**Ready to build superintelligent AI! 🚀**
