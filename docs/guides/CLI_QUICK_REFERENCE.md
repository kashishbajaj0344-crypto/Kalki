# KALKI CLI Quick Reference

**Version**: 3.0.0 (Multi-Domain Architecture)  
**Updated**: 2025-01-19

---

## Overview

KALKI CLI provides a unified interface to interact with KALKI's multi-domain intelligence system. Use it to manage domains, create projects, query knowledge, and ingest learning materials.

---

## Domain Commands

### `kalki domains list`
**Purpose**: List all available domain modules  
**Output**: Shows each domain with name, description, and knowledge item counts

```bash
kalki domains list
```

**Example Output**:
```
Available domains:
  • construction: Expert in construction, architecture, structural engineering
    Knowledge: span_tables(0), procedures(0), inspection_criteria(0)...
```

---

### `kalki domains info <domain_name>`
**Purpose**: Show detailed information about a specific domain  
**Arguments**:
- `domain_name`: Name of domain (e.g., construction, game_dev, robotics)

```bash
kalki domains info construction
```

**Example Output**:
```
Domain: construction
Description: Expert in construction, architecture, structural engineering...

Knowledge Types:
  • span_tables: 0 items
  • procedures: 0 items
  • inspection_criteria: 0 items
  ...

Available Deliverables:
  • construction_drawings: Architectural and structural drawings
  • bill_of_materials: Detailed materials list with quantities
  ...
```

---

### `kalki domains stats`
**Purpose**: Show system-wide domain statistics  
**Output**: Total domains, total knowledge items, deliverables available

```bash
kalki domains stats
```

---

## Project Commands

### `kalki project create <description>`
**Purpose**: Create a new project (domain auto-inferred from description)  
**Arguments**:
- `description`: Natural language project description
- `--domain <name>`: (Optional) Force specific domain
- `--requirements <json>`: (Optional) Structured requirements JSON

```bash
# Auto-infer domain
kalki project create "Build a 3-story home in Sechelt, BC"

# Force domain
kalki project create "Design a house" --domain construction

# With requirements
kalki project create "Design a garage" --domain construction --requirements '{"location":"Vancouver","size_sqft":800,"budget":50000}'
```

**Output**: Returns project ID for future reference

---

### `kalki project list`
**Purpose**: List all saved projects  
**Filters**:
- `--domain <name>`: Show only projects from specific domain
- `--phase <phase>`: Show only projects in specific phase

```bash
# All projects
kalki project list

# Construction projects only
kalki project list --domain construction

# Projects in FRAMING phase
kalki project list --phase FRAMING
```

---

### `kalki project status <project_id>`
**Purpose**: Show detailed status of a specific project  
**Arguments**:
- `project_id`: UUID returned from `project create`

```bash
kalki project status 1e814abc-217c-4818-9260-629b3294e3b8
```

**Output**:
```
Project: Build a 3-story home in Sechelt, BC
ID: 1e814abc-217c-4818-9260-629b3294e3b8
Domain: construction
Phase: REQUIREMENTS
Complexity: 75/100
Created: 2025-01-19 14:23:45
Requirements:
  • location: Sechelt, BC
  • stories: 3
```

---

### `kalki project advance-phase <project_id>`
**Purpose**: Move project to next phase  
**Arguments**:
- `project_id`: UUID of project

```bash
kalki project advance-phase 1e814abc-217c-4818-9260-629b3294e3b8
```

---

### `kalki project query <project_id> <query>`
**Purpose**: Ask domain-specific question about a project  
**Arguments**:
- `project_id`: UUID of project
- `query`: Natural language question

```bash
kalki project query 1e814abc-217c-4818-9260-629b3294e3b8 "What foundation type should I use?"
```

---

## Query Commands

### `kalki ask <query>`
**Purpose**: Unified query interface (auto-infers domain)  
**Arguments**:
- `query`: Natural language question

```bash
kalki ask "What size joists do I need for a 16 foot span?"
kalki ask "How do I create a health regeneration system in Unity?"
kalki ask "What's the best propulsion system for a flying suit?"
```

**How It Works**:
1. Analyzes query keywords
2. Infers relevant domain(s)
3. Retrieves domain-specific knowledge
4. Synthesizes quantum-enhanced response

---

## Learning Commands

### `kalki learn ingest <pdf_path>`
**Purpose**: Ingest PDF into knowledge base  
**Arguments**:
- `pdf_path`: Path to PDF file
- `--domain <name>`: (Optional) Associate with specific domain
- `--knowledge-type <type>`: (Optional) Specify knowledge type (span_tables, procedures, etc.)

```bash
# Auto-categorize
kalki learn ingest "BC_Building_Code.pdf"

# Domain-specific
kalki learn ingest "Wood_Design_Manual.pdf" --domain construction

# Specific knowledge type
kalki learn ingest "Span_Tables.pdf" --domain construction --knowledge-type span_tables
```

**Supported Formats**: PDF only (auto-extracts text, tables, images)

---

### `kalki learn query <knowledge_type>`
**Purpose**: Query specific knowledge type  
**Arguments**:
- `knowledge_type`: Type to query (span_tables, procedures, etc.)
- `--domain <name>`: (Optional) Filter by domain
- `--query <text>`: (Optional) Semantic search query

```bash
# All span tables
kalki learn query span_tables

# Construction procedures only
kalki learn query procedures --domain construction

# Semantic search
kalki learn query span_tables --query "16 foot span floor joists"
```

---

### `kalki learn status`
**Purpose**: Show learning system status and statistics

```bash
kalki learn status
```

---

### `kalki learn evolve`
**Purpose**: Trigger self-evolution and knowledge synthesis

```bash
kalki learn evolve
```

---

## Session Commands

### `kalki session new <name>`
**Purpose**: Start a new conversation session  
**Arguments**:
- `name`: Session name

```bash
kalki session new "House Design Project"
```

---

### `kalki session list`
**Purpose**: List all saved sessions

```bash
kalki session list
```

---

### `kalki session load <session_id>`
**Purpose**: Load previous session  
**Arguments**:
- `session_id`: ID from `session list`

```bash
kalki session load abc123
```

---

## Chat Commands

### `kalki chat <message>`
**Purpose**: Interactive conversation with KALKI  
**Arguments**:
- `message`: Natural language message

```bash
kalki chat "I'm building a house in BC, can you help me?"
```

---

### `kalki chat --interactive`
**Purpose**: Start interactive chat mode

```bash
kalki chat --interactive
```

**Usage**: Type messages, press Enter. Type `exit` to quit.

---

## Memory Commands

### `kalki memory save <key> <value>`
**Purpose**: Save fact to memory  
**Arguments**:
- `key`: Memory key
- `value`: Value to store

```bash
kalki memory save "client_name" "John Smith"
kalki memory save "project_budget" "500000"
```

---

### `kalki memory recall <key>`
**Purpose**: Retrieve saved memory  
**Arguments**:
- `key`: Memory key

```bash
kalki memory recall "client_name"
```

---

### `kalki memory search <query>`
**Purpose**: Semantic search across all memories  
**Arguments**:
- `query`: Search query

```bash
kalki memory search "budget"
```

---

## Status Commands

### `kalki status`
**Purpose**: Show complete system status

```bash
kalki status
```

**Output Includes**:
- Quantum state
- Neural network status
- Memory statistics
- Domain knowledge counts
- Active sessions
- Recent activity

---

## Common Workflows

### Starting a Construction Project

```bash
# 1. Create project
kalki project create "Build 3-story home in Vancouver, BC" --requirements '{"budget":800000,"size_sqft":2500}'

# 2. Get project ID (e.g., abc-123)

# 3. Check status
kalki project status abc-123

# 4. Ask questions
kalki project query abc-123 "What foundation type should I use?"

# 5. Advance to next phase
kalki project advance-phase abc-123

# 6. Generate deliverables
kalki project query abc-123 "Generate construction drawings"
```

---

### Building Knowledge Base

```bash
# 1. Ingest PDFs
kalki learn ingest "BC_Building_Code.pdf" --domain construction
kalki learn ingest "Wood_Design_Manual.pdf" --domain construction
kalki learn ingest "Span_Tables.pdf" --domain construction --knowledge-type span_tables

# 2. Check progress
kalki domains stats
kalki domains info construction

# 3. Query knowledge
kalki ask "What size joists for 16 foot span?"

# 4. Verify learning
kalki learn query span_tables --query "16 foot span"
```

---

### Multi-Domain Usage

```bash
# Construction query
kalki ask "What's the load capacity of a 2x10 joist?"

# Game dev query
kalki ask "How do I implement a health bar in Unity?"

# Robotics query
kalki ask "What sensors do I need for autonomous navigation?"

# Aerospace query
kalki ask "What's the thrust-to-weight ratio for VTOL?"

# KALKI auto-infers domain and retrieves relevant knowledge
```

---

## Tips & Best Practices

**1. Domain Auto-Inference**  
Let KALKI infer the domain from your query - it's usually accurate. Only force `--domain` if needed.

**2. Structured Requirements**  
Use `--requirements` JSON for complex projects to ensure all validation passes.

**3. Progressive Learning**  
Ingest PDFs incrementally. Check `domains stats` after each ingestion to monitor progress.

**4. Project IDs**  
Save project IDs! They're needed for all project operations. Use `project list` to retrieve them.

**5. Knowledge Types**  
Match `--knowledge-type` to PDF content for better categorization:
- Construction: `span_tables`, `procedures`, `inspection_criteria`, `cost_data`
- Game Dev: `design_patterns`, `api_docs`, `optimization`
- Robotics: `kinematics`, `sensors`, `control_systems`

**6. Query Specificity**  
More specific queries get better results:
- ❌ "Tell me about joists"
- ✅ "What size joists for 16 foot span with 40 PSF live load?"

**7. Session Management**  
Use sessions for multi-day projects to preserve context.

**8. Memory Usage**  
Store critical facts in memory for quick recall across sessions.

---

## Troubleshooting

**"No domain found for query"**  
→ Query too generic. Add domain-specific keywords or use `--domain` flag.

**"Knowledge base empty"**  
→ Need to ingest PDFs first. Run `learn ingest` commands.

**"Project ID not found"**  
→ Check `project list` for correct UUID.

**"Invalid requirements"**  
→ Requirements JSON must match domain's validation schema. Check `domains info <domain>`.

**"Phase transition failed"**  
→ Current phase may have validation failures. Run `project status <id>` to check.

---

## Advanced Usage

### Batch Operations

```bash
# Create multiple projects
for desc in "House A" "House B" "House C"; do
  kalki project create "Build $desc" --domain construction
done

# Ingest multiple PDFs
for pdf in *.pdf; do
  kalki learn ingest "$pdf" --domain construction
done
```

### JSON Piping

```bash
# Export project data
kalki project status abc-123 --format json > project.json

# Query with complex requirements
cat requirements.json | kalki project create "New Project" --domain construction --requirements-stdin
```

### Environment Variables

```bash
# Set default domain
export KALKI_DEFAULT_DOMAIN=construction

# Set API keys
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

---

## System Requirements

- **Python**: 3.9+
- **OS**: macOS, Linux, Windows
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB for knowledge base
- **Network**: Required for LLM API calls

---

## Support & Resources

- **Documentation**: `docs/README_COMPLETE.md`
- **Architecture**: `DOMAIN_ARCHITECTURE.md`
- **Build Roadmap**: `BUILD_ROADMAP.md`
- **Issues**: GitHub Issues (if applicable)

---

## Version History

**v3.0.0** (2025-01-19)
- Multi-domain architecture
- Domain registry and auto-discovery
- Project persistence system
- Enhanced CLI with 15+ new commands

**v2.0.0** (Earlier)
- Supreme Control Hub
- Professional deliverables
- Quantum consciousness

**v1.0.0** (Initial)
- Basic construction intelligence
- PDF ingestion
- Memory system

---

**Quick Start**: Run `kalki domains list` to see available capabilities, then `kalki ask "Your question"` to get started!
