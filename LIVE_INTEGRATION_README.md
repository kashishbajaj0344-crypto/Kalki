# Kalki Agent Integration v2.3 - Live Integration Guide

## Overview

The Kalki Agent Integration provides enterprise-grade safety orchestration for all RAG operations. This integration wraps the core `ask_kalki()` and `ingest_pdf_file()` functions with comprehensive safety checks.

## Quick Start

### Using Safe Functions Directly

```python
from kalki_agent_integration import safe_ask_kalki_sync, safe_ingest_pdf_file_sync

# Safe query with full safety orchestration
result = safe_ask_kalki_sync("What is AI safety?", {"user_id": "user123"})
if result["status"] == "success":
    print(f"Answer: {result['answer']}")
    print(f"Safety Score: {result['safety_assessment']['ethical_score']}")
```

### Using CLI with Safety

```bash
# Safe query via CLI
python -m modules.cli --safe-query "What are AI ethics?"

# Safe ingestion via CLI
python -m modules.cli --safe-ingest /path/to/document.pdf --domain academic

# Check safety system status
python -m modules.cli --safety-status
```

## Safety Features

### Query Safety Orchestration
- **Ethical Review**: Multi-framework ethical analysis
- **Risk Assessment**: Dynamic risk scoring with pattern analysis
- **Resource Checks**: System resource validation before LLM calls
- **Answer Verification**: Simulation-based answer safety validation
- **Event Logging**: All actions logged to EventBus for audit trails

### Ingestion Safety Orchestration
- **Content Ethics**: Document content ethical evaluation
- **Processing Risk**: Data processing safety verification
- **Resource Management**: Memory/CPU monitoring for large files
- **Containment Checks**: Isolation and rollback capabilities
- **Approval Workflows**: EventBus notifications for high-risk operations

## Integration Points

### Global Integration Instance
```python
from kalki_agent_integration import get_global_integration
integration = await get_global_integration()
```

### Synchronous Wrappers
For environments that need synchronous operation:
- `safe_ask_kalki_sync()` - Synchronous safe query
- `safe_ingest_pdf_file_sync()` - Synchronous safe ingestion

### CLI Integration
The CLI now supports:
- `--safe-query`: Query with safety orchestration
- `--safe-ingest`: Ingest with safety orchestration
- `--safety-status`: Check safety system health
- `--domain`: Specify content domain for better evaluation

## Response Schema

All safe functions return structured responses:

```python
{
    "status": "success|blocked|deferred|error",
    "answer": "...",  # For queries
    "safety_assessment": {
        "ethical_score": 0.0-1.0,
        "risk_score": 0.0-10.0,
        "risk_level": "low|medium|high|critical",
        "recommendations": [...]
    }
}
```

## Status Codes

- **success**: Operation completed safely
- **blocked**: Operation denied by safety agents
- **deferred**: Operation queued (insufficient resources)
- **error**: System or processing error

## Configuration

Safety agents are configured in `KalkiAgentIntegration.__init__()`:

```python
self.ethics_agent = EthicsAgent({
    "ethical_framework": "hybrid",
    "max_concurrent": 3
})
```

## Testing

Run the integration test:
```bash
python test_live_integration.py
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐
│   CLI/API       │───▶│  Safe Wrappers   │
└─────────────────┘    └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐
│ KalkiAgent      │◀──▶│  Safety Agents   │
│ Integration     │    │                  │
└─────────────────┘    │ • EthicsAgent     │
                       │ • RiskAgent       │
                       │ • SimulationAgent │
                       └──────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌──────────────────┐
│ Core Kalki      │    │  Support Agents  │
│ Functions       │    │                  │
│ • ask_kalki()   │    │ • ComputeOpt.    │
│ • ingest_pdf()  │    │ • KnowledgeLife. │
└─────────────────┘    │ • PerformanceMon.│
                       └──────────────────┘
```

## EventBus Integration

All safety decisions are published to EventBus:
- `action.requested`: Before high-risk operations
- `ethics.evaluated`: After ethical reviews
- `risk.assessed`: After risk assessments

## Performance Monitoring

All operations are automatically monitored:
- Query duration and success rates
- Resource utilization
- Error rates and types
- Safety decision metrics

## Version Control

Safety evaluations are versioned using KnowledgeLifecycleAgent for:
- Audit trails
- Performance analysis
- Continuous improvement

## Error Handling

Structured error propagation ensures:
- No exceptions bubble up to user
- Clear error categorization
- Recovery suggestions provided
- Telemetry collection for debugging