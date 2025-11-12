# 📁 Kalki Project Structure

This document describes the reorganized project structure.

## Directory Layout

```
Kalki/
├── README.md                 # Main project README
├── src/                      # Main application entry points
│   ├── kalki_cli.py         # CLI interface
│   ├── kalki_complete.py    # Main orchestrator
│   ├── kalki_api_server.py # API server
│   └── chat_with_kalki.py   # Chat interface
│
├── apps/                     # Application variants
│   ├── kalki_app.py         # Basic Streamlit app
│   ├── kalki_app_enhanced.py # Enhanced Streamlit app
│   ├── kalki_app_proactive.py # Proactive Streamlit app
│   ├── kalki_app_ai.py      # AI-focused app
│   └── kalki_unified_chat.py # Unified chatbot
│
├── modules/                   # Core modules (unchanged)
│   ├── agents/              # AI agents
│   ├── domains/             # Domain-specific modules
│   ├── learning/            # Learning systems
│   ├── utils/               # Utilities
│   └── ...
│
├── tests/                    # Test suites
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   ├── e2e/                 # End-to-end tests
│   └── ...                  # Other test files
│
├── docs/                     # Documentation
│   ├── guides/              # User guides and quick starts
│   ├── architecture/        # Architecture documentation
│   ├── status/              # Status and summary documents
│   └── archive/             # Historical documentation
│
├── scripts/                  # Utility scripts
│   ├── batch_ingest_pdfs.py
│   ├── ingest_folder.py
│   ├── quick_start_vision.py
│   └── ...
│
├── config/                   # Configuration files
│   ├── models_config.py
│   ├── requirements.txt
│   └── requirements_api.txt
│
├── frontend/                 # Frontend application
├── data/                     # Data files
├── output/                   # Generated outputs
├── logs/                     # Log files
└── modules/                  # Core modules (see above)
```

## Key Changes

### 1. Main Application Files → `src/`
- `kalki_cli.py` → `src/kalki_cli.py`
- `kalki_complete.py` → `src/kalki_complete.py`
- `kalki_api_server.py` → `src/kalki_api_server.py`

### 2. App Variants → `apps/`
- All `kalki_app*.py` files → `apps/`
- `kalki_unified_chat.py` → `apps/`

### 3. Documentation → `docs/`
- User guides → `docs/guides/`
- Architecture docs → `docs/architecture/`
- Status/summary docs → `docs/status/`

### 4. Tests → `tests/`
- Unit tests → `tests/unit/`
- Integration tests → `tests/integration/`
- E2E tests → `tests/e2e/`

### 5. Scripts → `scripts/`
- Utility scripts moved to `scripts/`

### 6. Config → `config/`
- Configuration files moved to `config/`

## Usage

### Running Applications

```bash
# CLI
python3 src/kalki_cli.py chat

# Unified Chatbot
python3 apps/kalki_unified_chat.py

# Streamlit Apps
streamlit run apps/kalki_app_enhanced.py

# API Server
python3 src/kalki_api_server.py
```

### Running Scripts

```bash
# Batch PDF ingestion
python3 scripts/batch_ingest_pdfs.py

# Quick start
python3 scripts/quick_start_vision.py
```

### Running Tests

```bash
# Unit tests
python3 -m pytest tests/unit/

# Integration tests
python3 -m pytest tests/integration/

# All tests
python3 -m pytest tests/
```

## Import Updates

If you have custom code that imports from moved files, update imports:

```python
# Old
from kalki_cli import KalkiCLI
from models_config import get_model_path

# New
from src.kalki_cli import KalkiCLI
from config.models_config import get_model_path
```

## Notes

- The `modules/` directory structure remains unchanged
- All file paths in documentation have been updated
- Scripts and shell files have been updated with new paths
- Python imports may need updating if you have custom code

