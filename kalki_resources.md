Kalki/
├─ kalki_env/                 # python venv (not in repo)
├─ modules/                   # core python modules (ingest, llm, vectordb, copilot, agents...)
│  ├─ ingest.py
│  ├─ llm.py
│  ├─ vectordb.py
│  ├─ copilot.py
│  ├─ agents/
│  │  ├─ core_agent.py
│  │  ├─ ai_agent.py
│  │  ├─ physics_agent.py
│  │  └─ ... (all other agents)
├─ agents/                    # high-level agent configs & manifests
├─ pdfs/                      # user PDFs (engineering, cs, arts)
│  ├─ engineering/
│  ├─ arts/
│  └─ computer_science/
├─ vector_db/                 # e.g. chroma.sqlite3
│  └─ chroma.sqlite3
├─ connectors/                # API connectors (arxiv, pubmed, github, etc.)
├─ pipeline/                  # ingestion pipeline tasks, chunkers
├─ ui/                        # React/Tauri front-end project (optional)
├─ scripts/
│  ├─ setup_dev.sh
│  └─ ingest_run.sh
├─ kalki.py                   # main orchestrator/CLI
├─ kalki_resources.json
└─ query_cost.json
