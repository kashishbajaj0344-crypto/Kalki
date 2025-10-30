# Kalki 2.0 - Personal AI Faculty

---
## 1. Project Directory Structure

```plaintext
Kalki/
├─ kalki_env/                 # Python venv (not in repo)
├─ modules/
│  ├─ ingest.py
│  ├─ llm.py
│  ├─ vectordb.py
│  ├─ copilot.py
│  ├─ agents/
│  │  ├─ agent_manager.py
│  │  ├─ base_agent.py
│  │  ├─ physics_agent.py
│  │  └─ ... (other agents)
├─ agents/                    # High-level agent configs & manifests
├─ pdfs/                      # User PDFs (engineering, cs, arts)
│  ├─ engineering/
│  ├─ arts/
│  └─ computer_science/
├─ vector_db/                 # Persistent vector DB
│  └─ chroma.sqlite3
├─ connectors/                # API connectors (arxiv, pubmed, github, etc.)
├─ pipeline/                  # Ingestion pipeline tasks, chunkers
├─ ui/                        # React/Tauri front-end project (optional)
├─ scripts/
│  ├─ setup_dev.sh
│  └─ ingest_run.sh
├─ kalki.py                   # Main orchestrator/CLI
├─ kalki_resources.json
├─ query_cost.json
└─ README.md
```
## 2. Overview

Kalki 2.0 is a **modular, locally-run, research-grade AI faculty** for deep reasoning, knowledge retrieval, and original research.  
It orchestrates domain-specialized agents, a vector database, and modern local LLM/embedding models to automate ingestion, querying, and synthesis across science, tech, humanities, and more.

**Key Architecture:**
- **Agents:** Each agent is a domain expert (Physics, Medicine, Law, etc.) with its own retrieval and reasoning.
- **Vector DB:** Central knowledge store (ChromaDB recommended) for all embeddings and metadata.
- **Ingestion Pipeline:** Automated chunking, deduplication, and embedding of local files and open-source connectors (arXiv, PubMed, SSRN, GitHub, ...).
- **CLI:** Unified command line for ingestion, querying, agent management, and status.

---

## 3. Setup Instructions

### Python & Environment

- Requires **Python 3.10+** (tested on 3.13.x)
- Recommended: Apple Silicon (M1/M2/M4) or CUDA GPU for fastest embeddings.

### Install Dependencies

```bash
python -m venv kalki_env
source kalki_env/bin/activate
pip install -r requirements.txt
```

### Config

- Main config: `kalki_resources.json` (see below for structure)
- Place your PDFs in `pdfs/` (with subfolders as needed)
- Run from project root (`Kalki/`)

---

## 4. Usage

### CLI Commands

```bash
# Ingest a folder of PDFs
python kalki.py --ingest pdfs/engineering/

# Query an agent
python kalki.py --agent PhysicsAgent --ask "Explain quantum tunneling"

# Batch query
python kalki.py --agent LawAgent --batch_query law_questions.txt

# Status report
python kalki.py --status

# Interactive mode
python kalki.py --interactive
```

### Example Agent Query

```
Q: How does mRNA vaccine technology work?
A: [MedicineAgent] ... (context + LLM-based synthesis)
```

---

## 5. Configuration

**Example: `kalki_resources.json` v0.3**
```json
{
  "config_version": "0.3",
  "embedding_model": "BAAI/bge-large-en-v1.5",
  "vector_db_path": "vector_db/chroma.sqlite3",
  "vector_db_persistent": true,
  "pdfs_path": "pdfs/",
  "agents_path": "modules/agents/",
  "connectors_path": "connectors/",
  "device": "auto",
  "log_path": "kalki.log",
  "embedding_batch_size": 16,
  "vector_db_top_k": 10,
  "chunk_size": 2000,
  "chunk_overlap": 300,
  "max_tokens_per_query": 4096,
  "query_timeout": 60,
  "num_workers": 4
}
```
- **device:** "auto" (auto-detects cpu, mps, cuda)
- **embedding_model:** Local BGE large recommended (fastest on Apple Silicon)
- **batch_size, chunk_size:** Tune for performance vs. memory

---

## 6. Modules

- **AgentManager:** Loads and manages all agent classes, routes queries, handles ingestion.
- **vectordb:** Abstraction for ChromaDB or other vector DBs. Handles indexing, search, and metadata.
- **Ingestion pipeline:** Detects file types, chunks, deduplicates, embeds, and stores in DB.
- **Connectors:** Domain-specific data fetchers (arXiv, PubMed, SSRN, GitHub, etc.); extensible for more.
- **llm.py:** Handles all embedding and reasoning LLM calls, device selection, batching.

---

## 7. Notes & Recommendations

- **Device selection:** Use "auto" or set manually; M1/M2/M4 is fastest for BGE.
- **Batch size/chunking:** Larger batches == better throughput, up to VRAM limits.
- **Fallback models:** Supported in config; see query_cost.json for performance.
- **Vector DB Persistence:** `vector_db_persistent: true` for long-term storage; false for RAM-only.
- **Parallel ingestion:** Use `num_workers` for multi-threaded file processing.
- **Status reporting:** Use `--status` for loaded agents, DB size, active config.

---

## 8. Changelog

- **v0.1**: Project skeleton, CLI, config, README, initial agent/ingestion/vector DB modules.
- **v0.2**: Device auto-detection, batch size, chunking, status reporting.
- **v0.3**: Persistence options, query timeout, parallel ingestion, performance metadata, fallback models.

---

**Future Enhancements / TODOs**
- Hybrid embedding strategy (per agent/domain)
- Multi-agent collaboration (cross-domain synthesis)
- Mobile bridge and browser extension for on-the-fly data ingestion
- Full UI/dashboard with agent status and knowledge visualization
- Live LLM streaming/interactive chat with agents

---

**Kalki 2.0 is a living, modular research faculty—expandable, explainable, and uniquely yours.**