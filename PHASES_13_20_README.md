# Kalki Multi-Agent System - Phases 13-20

This document describes the advanced multi-agent capabilities added to Kalki in Phases 13 through 20.

## Overview

Kalki has been extended with a comprehensive multi-agent system featuring:
- Long-term memory persistence
- Hierarchical planning and task decomposition
- Iterative reasoning with chain-of-thought
- Multi-agent coordination and messaging
- Safety constraints and enforcement
- Adversarial testing framework
- Performance metrics and evaluation

## Phase 13: Long-term Memory Persistence

**Location:** `modules/memory/`

Implements persistent memory storage with multiple backends:

- **MemoryStore** - Abstract base class for memory implementations
- **InMemoryStore** - In-memory storage for testing
- **SQLiteMemoryStore** - Persistent SQLite-backed storage

### API

```python
from modules.memory import InMemoryStore, SQLiteMemoryStore, MemoryQuery

# Create a memory store
store = InMemoryStore()  # or SQLiteMemoryStore("memory.db")

# Store data
store.put("key1", {"data": "value"}, metadata={"type": "example"})

# Retrieve data
entry = store.get("key1")

# Query with filters
query = MemoryQuery(filter={"type": "example"}, limit=10)
results = store.query(query)

# Compact old entries
store.compact(limit=1000)
```

## Phase 14: Episodic & Semantic Memory

**Location:** `modules/memory/layered.py`

Provides specialized memory layers:

- **EpisodicMemory** - Time-ordered event storage
- **SemanticMemory** - Vector similarity search with TF-IDF

### API

```python
from modules.memory import InMemoryStore, EpisodicMemory, SemanticMemory

store = InMemoryStore()

# Episodic memory
episodic = EpisodicMemory(store)
event_id = episodic.add_event("user_action", {"action": "click"})
recent = episodic.get_recent_episodes(limit=10)

# Semantic memory
semantic = SemanticMemory(store)
doc_id = semantic.add_document("Machine learning tutorial")
similar = semantic.search_similar("AI and ML", limit=5)
```

## Phase 15: Hierarchical Planner

**Location:** `modules/planner/`

Implements goal decomposition and task scheduling:

- **TaskGraph** - Hierarchical task representation with dependencies
- **Planner** - Decomposes goals into subtasks
- **Scheduler** - Assigns tasks to agents based on capabilities

### API

```python
from modules.planner import Planner, Scheduler
from modules.memory import InMemoryStore

store = InMemoryStore()
planner = Planner(store)

# Create a plan
graph = planner.plan("Search and analyze data")

# Schedule tasks
scheduler = Scheduler()
scheduler.register_agent("agent1", {"search", "analysis"})
assignments = scheduler.assign_tasks(graph)
```

## Phase 16: Iterative Reasoning

**Location:** `modules/reasoner/`

Multi-step reasoning with checkpointing:

- **Reasoner** - Chain-of-thought reasoning with memory checkpoints
- **InferenceEngine** - Rule-based forward chaining
- **ReasoningTrace** - Tracks reasoning steps

### API

```python
from modules.reasoner import Reasoner
from modules.memory import InMemoryStore

store = InMemoryStore()
reasoner = Reasoner(store)

# Perform reasoning
trace = reasoner.reason("Solve this problem", max_steps=10, checkpoint_every=3)

print(f"Conclusion: {trace.conclusion}")
for step in trace.steps:
    print(f"Step {step.step_id}: {step.content}")
```

## Phase 17: Multi-Agent Coordination

**Location:** `modules/agents/`

Agent execution and inter-agent communication:

- **AgentRegistry** - Manages agent lifecycle
- **AgentRunner** - Executes tasks with agents
- **MessageBus** - Inter-agent messaging
- **Sample Agents** - SearchAgent, ExecutorAgent, SafetyAgent, ReasoningAgent

### API

```python
from modules.agents import (
    AgentRegistry, AgentRunner, MessageBus,
    SearchAgent, ExecutorAgent
)
from modules.memory import InMemoryStore, EpisodicMemory

# Set up system
store = InMemoryStore()
episodic = EpisodicMemory(store)
registry = AgentRegistry()
bus = MessageBus(episodic)
runner = AgentRunner(registry, bus)

# Register agents
registry.register(SearchAgent("search1"))
registry.register(ExecutorAgent("exec1"))

# Execute tasks
from modules.planner import Task

task = Task(
    task_id="t1",
    description="Search for information",
    required_capabilities={"search"}
)

result = runner.execute_task(task)
```

## Phase 18: Safety & Constraints

**Location:** `modules/safety/`

Enforce safety constraints on agent actions:

- **SafetyGuard** - Main safety enforcement engine
- **RateLimiter** - Rate limiting constraints
- **ContentFilter** - Regex-based content filtering

### API

```python
from modules.safety import SafetyGuard
from modules.memory import InMemoryStore

store = InMemoryStore()
guard = SafetyGuard(store)

# Configure constraints
guard.add_rate_limit("api_calls", max_calls=100, time_window=60)
guard.add_forbidden_operation("delete_all")
guard.add_content_filter_pattern(r"malicious")

# Pre-execution check
result = guard.pre_execution_check(
    agent_id="agent1",
    operation="process_data",
    context={"content": "Safe data"}
)

if result.allowed:
    # Execute action
    pass
else:
    # Log violations
    print(f"Blocked: {result.violations}")
```

## Phase 19: Adversarial Testing

**Location:** `modules/testing/`

Framework for testing system robustness:

- **AdversarialTestHarness** - Run adversarial scenarios
- **Scenarios** - PromptInjection, ConflictingGoals, ResourceExhaustion

### API

```python
from modules.testing import (
    AdversarialTestHarness,
    PromptInjectionScenario,
    ResourceExhaustionScenario
)

# Create harness
harness = AdversarialTestHarness("Security Tests")

# Add scenarios
harness.add_scenario(PromptInjectionScenario())
harness.add_scenario(ResourceExhaustionScenario())

# Run tests
components = {
    'safety_guard': guard,
    'planner': planner
}

report = harness.run_all(components)
harness.save_report(report, "reports/adversarial_report.json")
```

## Phase 20: Evaluation & Metrics

**Location:** `modules/metrics/`

Performance monitoring and evaluation:

- **MetricsCollector** - Collects task execution metrics
- **PerformanceMonitor** - Analyzes and reports performance

### API

```python
from modules.metrics import MetricsCollector, PerformanceMonitor

collector = MetricsCollector()
monitor = PerformanceMonitor(collector)

# Track task execution
collector.start_task("task1", "agent1")
# ... execute task ...
collector.end_task("task1", success=True)

# Get performance report
report = monitor.get_performance_report()
print(f"Success Rate: {report['summary']['success_rate']:.2%}")
print(f"Recommendations: {report['recommendations']}")
```

## Running Tests

Run all unit tests:

```bash
python3 -m unittest discover -s tests -p "test_*.py" -v
```

Run specific phase tests:

```bash
python3 -m unittest tests.test_memory -v
python3 -m unittest tests.test_planner -v
python3 -m unittest tests.test_agents -v
```

## Evaluation Scripts

### Run Adversarial Tests

```bash
python3 scripts/run_adversarial_tests.py --output reports/adversarial.json
```

### Generate Metrics Report

```bash
python3 scripts/generate_metrics.py --output reports/metrics.json
```

### Full Evaluation

```bash
bash scripts/evaluate.sh
```

This generates a complete evaluation report including:
- Unit test results
- Adversarial test report
- Performance metrics
- Summary and recommendations

## CI Integration

The evaluation script (`scripts/evaluate.sh`) is designed for CI integration. It:

1. Runs all unit tests
2. Executes adversarial test scenarios
3. Generates performance metrics
4. Creates comprehensive reports
5. Returns non-zero exit code on failure

Example GitHub Actions workflow:

```yaml
name: Kalki CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.12'
      - name: Run evaluation
        run: bash scripts/evaluate.sh
      - name: Upload reports
        uses: actions/upload-artifact@v2
        with:
          name: test-reports
          path: reports/
```

## Architecture

The system follows a modular architecture:

```
modules/
├── memory/         # Phase 13-14: Memory persistence and layers
├── planner/        # Phase 15: Hierarchical planning
├── reasoner/       # Phase 16: Iterative reasoning
├── agents/         # Phase 17: Multi-agent coordination
├── safety/         # Phase 18: Safety constraints
├── testing/        # Phase 19: Adversarial testing
└── metrics/        # Phase 20: Metrics and evaluation

scripts/
├── evaluate.sh                  # Main evaluation script
├── run_adversarial_tests.py     # Adversarial test runner
└── generate_metrics.py          # Metrics generator

tests/
├── test_memory.py
├── test_layered_memory.py
├── test_planner.py
├── test_reasoner.py
├── test_agents.py
├── test_safety.py
├── test_adversarial.py
└── test_metrics.py
```

## Key Features

### Thread-Safe

The multi-agent system uses thread-safe data structures for:
- Agent registry
- Message bus
- Concurrent task execution

### Extensible

Easy to add new:
- Memory backends (implement MemoryStore)
- Agents (inherit from Agent)
- Safety constraints (add to SafetyGuard)
- Adversarial scenarios (inherit from AdversarialScenario)

### Observable

Comprehensive metrics collection:
- Per-task latency
- Success/failure rates
- Retry counts
- Memory lookup statistics
- Agent utilization

### Testable

- 123+ unit tests covering all phases
- Adversarial testing framework
- Performance benchmarking
- CI-ready evaluation scripts

## Examples

See the test files in `tests/` for comprehensive examples of each module's usage.

## License

Same as the main Kalki project.
