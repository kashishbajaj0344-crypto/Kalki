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

## 4. Internet Connectivity Setup

Kalki now supports internet connectivity for web search, research, and real-time data retrieval. This enables agents to access current information, research external topics, and understand contemporary concepts.

### Quick Setup

Run the automated setup script:

```bash
python setup_internet.py
```

This will:
- Install required dependencies (`beautifulsoup4`, `aiohttp`, `lxml`, `python-dotenv`)
- Guide you through API key configuration
- Test connectivity and API keys
- Create/update `.env` file with your keys

### Manual Setup

1. **Install Dependencies:**
```bash
pip install beautifulsoup4==4.12.2 aiohttp==3.9.1 lxml==4.9.3 python-dotenv==1.1.1
```

2. **Configure API Keys:**
Create a `.env` file in the project root:

```bash
# Google Custom Search (recommended)
GOOGLE_SEARCH_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id

# Bing Search API
BING_SEARCH_API_KEY=your_bing_api_key

# SerpApi (alternative Google search)
SERPAPI_KEY=your_serpapi_key

# OpenAI API (enhanced LLM features)
OPENAI_API_KEY=your_openai_key

# HuggingFace API (Llama models)
HUGGINGFACE_API_KEY=your_huggingface_key
```

3. **Get API Keys:**
- **Google Custom Search:** https://console.developers.google.com/ + https://cse.google.com/
- **Bing Search:** https://www.microsoft.com/en-us/bing/apis/bing-web-search-api
- **SerpApi:** https://serpapi.com/
- **OpenAI:** https://platform.openai.com/api-keys
- **HuggingFace:** https://huggingface.co/settings/tokens

### Features

- **Web Search:** Search across multiple providers (Google, Bing, DuckDuckGo)
- **Content Research:** Deep research with multiple search queries and synthesis
- **URL Fetching:** Direct content retrieval from specific URLs
- **Safety Controls:** Content filtering, rate limiting, and blocked domain handling
- **Fallback Support:** Automatic fallback between search providers
- **Caching:** Intelligent caching to reduce API calls and improve performance

### Usage Examples

```bash
# Web search
kalki web search "Call of Duty game mechanics" --results 5

# Research a topic
kalki web research "artificial intelligence trends" --depth comprehensive

# Fetch content from URL
kalki web fetch "https://en.wikipedia.org/wiki/Call_of_Duty"

# Regular queries now use web search when appropriate
kalki query "What are the latest Call of Duty games?"
```

---

## 5. Usage

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

# Web search (new)
kalki web search "quantum physics latest research" --results 5

# Web research (new)
kalki web research "artificial intelligence trends" --depth comprehensive

# Fetch content from URL (new)
kalki web fetch "https://en.wikipedia.org/wiki/Quantum_mechanics"
```

### Example Agent Query

```
Q: How does mRNA vaccine technology work?
A: [MedicineAgent] ... (context + LLM-based synthesis)
```

---

## 6. Configuration

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

## 7. Modules

- **AgentManager:** Loads and manages all agent classes, routes queries, handles ingestion.
- **vectordb:** Abstraction for ChromaDB or other vector DBs. Handles indexing, search, and metadata.
- **Ingestion pipeline:** Detects file types, chunks, deduplicates, embeds, and stores in DB.
- **Connectors:** Domain-specific data fetchers (arXiv, PubMed, SSRN, GitHub, etc.); extensible for more.
- **llm.py:** Handles all embedding and reasoning LLM calls, device selection, batching.

---

## 8. Notes & Recommendations

- **Device selection:** Use "auto" or set manually; M1/M2/M4 is fastest for BGE.
- **Batch size/chunking:** Larger batches == better throughput, up to VRAM limits.
- **Fallback models:** Supported in config; see query_cost.json for performance.
- **Vector DB Persistence:** `vector_db_persistent: true` for long-term storage; false for RAM-only.
- **Parallel ingestion:** Use `num_workers` for multi-threaded file processing.
- **Status reporting:** Use `--status` for loaded agents, DB size, active config.

---

## 9. Changelog

- **v0.1**: Project skeleton, CLI, config, README, initial agent/ingestion/vector DB modules.
- **v0.2**: Device auto-detection, batch size, chunking, status reporting.
- **v0.3**: Persistence options, query timeout, parallel ingestion, performance metadata, fallback models.
- **v0.4**: Internet connectivity with WebSearchAgent, multiple search providers (Google, Bing, DuckDuckGo, SerpApi), safety controls, CLI web commands, and real-time data retrieval.

---

**Future Enhancements / TODOs**
- Hybrid embedding strategy (per agent/domain)
- Multi-agent collaboration (cross-domain synthesis)
- Mobile bridge and browser extension for on-the-fly data ingestion
- Full UI/dashboard with agent status and knowledge visualization
- Live LLM streaming/interactive chat with agents

---

**Kalki 2.0 is a living, modular research faculty—expandable, explainable, and uniquely yours.**