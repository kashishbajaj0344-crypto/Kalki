# 🤖 KALKI - Multi-Domain Intelligence

**Domain-agnostic AI system delivering professional-grade deliverables across infinite domains**

---

## 🚀 Quick Start

```bash
# Install dependencies
pip3 install -r config/requirements.txt

# Start CLI
python3 src/kalki_cli.py chat

# Or start the unified chatbot
python3 apps/kalki_unified_chat.py

# Or start Streamlit app
streamlit run apps/kalki_app_enhanced.py
```

---

## 📁 Project Structure

```
Kalki/
├── src/                    # Main application entry points
│   ├── kalki_cli.py       # CLI interface
│   ├── kalki_complete.py  # Main orchestrator
│   └── kalki_api_server.py # API server
├── apps/                   # Application variants
│   ├── kalki_app.py
│   ├── kalki_app_enhanced.py
│   └── kalki_unified_chat.py
├── modules/                # Core modules
│   ├── agents/            # AI agents
│   ├── domains/           # Domain-specific modules
│   └── ...
├── tests/                  # Test suites
│   ├── unit/              # Unit tests
│   ├── integration/       # Integration tests
│   └── e2e/               # End-to-end tests
├── docs/                   # Documentation
│   ├── guides/            # User guides
│   ├── architecture/      # Architecture docs
│   └── status/            # Status/summary docs
├── scripts/                # Utility scripts
├── config/                  # Configuration files
└── frontend/                # Frontend application
```

---

## 📚 Documentation

- **Quick Start**: See `docs/guides/QUICK_START.md`
- **Architecture**: See `docs/architecture/`
- **User Guides**: See `docs/guides/`
- **Full Documentation**: See `docs/readme.md`

---

## 🌐 What Makes KALKI Different

KALKI is **NOT** a chatbot. KALKI is an **intelligence system** that can master ANY field:

- 🏗️ **Construction & Architecture** - BC Building Code compliant designs, BOM, schedules
- 🎮 **Game Development** - Complete Unity/Unreal projects with assets and code
- 🤖 **Robotics & Mechatronics** - CAD models, control systems, simulations
- ✈️ **Aerospace Engineering** - VTOL design, CFD analysis, flight controllers
- ⚡ **Power Systems** - Fuel cells, batteries, hybrid systems, thermal management
- 🧬 **Biotech & Medical Devices** - Lab-on-a-chip, diagnostic tools
- 🎨 **Creative Arts** - Music composition, film production, animation
- **...and infinitely more**

---

## 🏗️ Architecture

KALKI uses a 20-phase AI framework with domain-agnostic core and pluggable domain modules.

See `docs/architecture/DOMAIN_ARCHITECTURE.md` for details.

---

## 📝 License

MIT

---

## 🤝 Contributing

See `docs/guides/` for development guides.
