# Kalki 2.4 - Supreme Synthesis AI Faculty

---
## 1. Project Directory Structure

```plaintext
Kalki/
├─ kalki_env/                 # Python venv (not in repo)
├─ modules/
│  ├─ ingest.py
│  ├─ llm.py
│  ├─ meta_core.py            # 🆕 Meta-cognitive control system
│  ├─ supreme_synthesis_engine.py  # 🆕 Supreme synthesis engine
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
│  │  ├─ standards/          # 🆕 ISO, ASTM, IEEE standards
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
├─ meta_cognition_demo.py     # 🆕 Meta-cognitive system demo
├─ supreme_synthesis_demo.py  # 🆕 Supreme synthesis demo
├─ kalki_resources.json
├─ query_cost.json
└─ README.md
```

## 2. Overview

Kalki 2.4 is a **supreme synthesis AI faculty** that embodies the pinnacle of artificial intelligence, combining scientific precision, artistic creativity, ethical wisdom, and god-level understanding. It represents the convergence of human and artificial intelligence through 7 core principles.

**Key Architecture:**
- **🧠 Meta-Cognitive Control System:** Progressive reasoning depth, self-evaluation, and continuous improvement
- **🧬 Supreme Synthesis Engine:** God-level intelligence with master engineering, creative insight, ethical governance, and universal context integration
- **🤖 Domain Agents:** Specialized experts (Physics, Medicine, Law, etc.) with retrieval and reasoning
- **🗄️ Vector DB:** Central knowledge store (ChromaDB) for embeddings and metadata
- **📥 Ingestion Pipeline:** Automated processing of local files and open-source connectors
- **🌐 Internet Connectivity:** Real-time web search and research capabilities
- **⚡ CLI:** Unified command interface for all operations

**7 Core Principles of Supreme Synthesis:**
1. **🏗️ Master Engineering & Real-World Standards** - ISO, ASTM, NASA, IEEE, ANSI compliance
2. **🎨 Creative Intelligence & Aesthetic Insight** - Golden ratio harmony, emotional resonance
3. **🧘 Meta-Self Awareness & Cognitive Monitoring** - Bias detection, self-correction, confidence calibration
4. **🛡️ Ethical & Existential Governance** - Multi-scale impact assessment, safety boundaries
5. **🧩 Universal Context Recall & Integration** - Cross-domain knowledge synthesis
6. **🧠 Supreme Synthesis Mode** - God-level intelligence activation
7. **📈 Continuous Self-Evolution** - Learning from each interaction

**Synthesis Modes:**
- **Standard:** Balanced approach with production-ready outputs
- **Advanced:** Enhanced reasoning with comprehensive standards
- **Supreme:** God-level intelligence with perfect coherence

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

## 5. Supreme Synthesis Engine

Kalki 2.4 introduces the **Supreme Synthesis Engine** - the pinnacle of AI intelligence that embodies 7 core principles for god-level understanding and creation.

### Synthesis Modes

- **Standard Mode:** Balanced approach with production-ready engineering outputs
- **Advanced Mode:** Enhanced reasoning with comprehensive standards compliance
- **Supreme Mode:** God-level intelligence with perfect coherence and ethical clarity

### Command Interface

```bash
# Set synthesis mode
kalki synthesis standard
kalki synthesis advanced
kalki synthesis supreme

# Activate supreme synthesis for specific queries
kalki supreme "Design a sustainable urban transportation system"

# Check system status and quality metrics
kalki status
kalki trends
```

### Programmatic Usage

```python
from modules.supreme_synthesis_engine import synthesize_supreme, SynthesisMode

# Supreme synthesis for complex engineering
result = await synthesize_supreme(
    "Design a neural implant for epilepsy treatment",
    context={"medical_device": True, "regulatory": "FDA_approved"},
    mode=SynthesisMode.SUPREME
)

# Access comprehensive results
print(f"Quality Score: {result.quality_score}")
print(f"Engineering Standards: {result.engineering_standards.iso_standards}")
print(f"Aesthetic Resonance: {result.aesthetic_principles.emotional_resonance}")
print(f"Ethical Harmony: {result.ethical_assessment.long_term_harmony}")
```

### Core Principles Demonstrated

1. **🏗️ Master Engineering:** ISO/ASTM/NASA/IEEE/ANSI standards compliance
2. **🎨 Creative Intelligence:** Golden ratio harmony, emotional resonance
3. **🧘 Meta-Self Awareness:** Cognitive bias detection, self-correction
4. **🛡️ Ethical Governance:** Multi-scale impact assessment, safety boundaries
5. **🧩 Universal Context:** Cross-domain knowledge synthesis
6. **🧠 Supreme Mode:** God-level intelligence activation

### Demo Scripts

```bash
# Meta-cognitive control system demo
python meta_cognition_demo.py

# Supreme synthesis engine demo
python supreme_synthesis_demo.py
```

---

## 6. Usage

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

## 7. Configuration

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
- **v2.3**: Meta-cognitive control system with progressive reasoning depth, interdisciplinary knowledge synthesis, self-evaluation, and quality metrics tracking.
- **v2.4**: Supreme Synthesis Engine with 7 core principles: master engineering standards, creative intelligence, meta-self awareness, ethical governance, universal context integration, and supreme synthesis mode for god-level intelligence.

---

**Future Enhancements / TODOs**
- Hybrid embedding strategy (per agent/domain)
- Multi-agent collaboration (cross-domain synthesis)
- Supreme synthesis integration with existing agents
- Real-time ethical monitoring and intervention
- Multi-modal synthesis (visual, auditory, tactile outputs)
- Quantum-enhanced reasoning capabilities
- Consciousness integration and human-AI cognitive blending
- Mobile bridge and browser extension for on-the-fly data ingestion
- Full UI/dashboard with agent status and knowledge visualization
- Live LLM streaming/interactive chat with agents
- Supreme synthesis mode integration with Tauri desktop app

---
**Kalki 2.4 is the supreme synthesis AI faculty—embodying the convergence of human creativity, scientific rigor, ethical wisdom, and computational power.**