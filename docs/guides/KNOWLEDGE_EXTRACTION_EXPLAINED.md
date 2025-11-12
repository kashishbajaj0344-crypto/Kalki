# 🧠 KALKI Knowledge Extraction System Explained
## Dual-Track System: Structured Knowledge + Vector Embeddings

---

## Overview: Two Parallel Knowledge Systems

When you ingest PDFs into KALKI, it runs **TWO SEPARATE extraction processes simultaneously**:

1. **📊 Structured Knowledge Extraction** → Domain-specific databases (SQLite)
2. **🔍 Vector Embeddings** → Semantic search database (ChromaDB)

**They are completely separate but complementary!**

---

## System Architecture

```
PDF Document
     |
     v
[KALKI Ingestion Pipeline]
     |
     ├──→ [Text Extraction + OCR + Table Extraction]
     |           |
     |           v
     |    ┌─────────────────────────────────┐
     |    │   Full Text Chunks (512 tokens) │
     |    └─────────────────────────────────┘
     |           |
     |           ├─────────────────────────────────────────────┐
     |           |                                             |
     |           v                                             v
     |    ┌───────────────────┐                    ┌──────────────────────┐
     |    │  TRACK 1: VECTOR  │                    │ TRACK 2: STRUCTURED  │
     |    │    EMBEDDINGS     │                    │    KNOWLEDGE         │
     |    └───────────────────┘                    └──────────────────────┘
     |           |                                             |
     |           v                                             v
     |    ┌──────────────────────┐              ┌───────────────────────────┐
     |    │ BGE-Large Embedder   │              │ Domain-Specific Extractors│
     |    │ (768-dim vectors)    │              │ (Regex + NLP patterns)    │
     |    └──────────────────────┘              └───────────────────────────┘
     |           |                                             |
     |           v                                             v
     |    ┌──────────────────────┐              ┌───────────────────────────┐
     |    │    ChromaDB          │              │   SQLite Databases        │
     |    │  (vector_db/)        │              │   (data/knowledge_db/)    │
     |    │                      │              │                           │
     |    │ • Semantic search    │              │ • span_tables.db          │
     |    │ • RAG retrieval      │              │ • procedures.db           │
     |    │ • Context matching   │              │ • inspection_criteria.db  │
     |    │ • Full-text chunks   │              │ • cost_data.db            │
     |    └──────────────────────┘              │ • load_parameters.db      │
     |                                          │ • decision_trees.db       │
     |                                          └───────────────────────────┘
     |
     v
[KALKI Control Hub]
     |
     ├──→ Semantic Query? → Query ChromaDB vector embeddings
     └──→ Specific Data? → Query structured knowledge databases
```

---

## Track 1: Vector Embeddings (ChromaDB)

### What Gets Stored

**Every chunk of text** from the PDF gets converted to a 768-dimensional vector using BGE-Large model.

### Example from Construction PDF:

**Original Text Chunk:**
```
"2x10 joists at 16" O.C. can span up to 16'-1" for residential floor loads 
(40 PSF live load + 10 PSF dead load) when using #2 grade SPF lumber."
```

**Gets Converted To:**
```python
{
    "text": "2x10 joists at 16\" O.C. can span up to 16'-1\" for residential...",
    "embedding": [0.023, -0.145, 0.891, ..., 0.234],  # 768 numbers!
    "metadata": {
        "source": "BC_Building_Code_Part9_2018.pdf",
        "page": 142,
        "chunk_id": "abc123_chunk_45",
        "tags": ["structural", "joists", "span_tables", "residential"]
    }
}
```

### What It's Used For

1. **Semantic Search** - Find relevant context when user asks questions:
   ```bash
   User: "How far can I span a 2x10 joist?"
   
   KALKI queries vector DB with semantic similarity
   → Returns top 10 most relevant chunks
   → Uses them as context for LLM response
   ```

2. **RAG (Retrieval-Augmented Generation)** - Enrich responses with PDF knowledge:
   ```python
   query = "Design a 2-story residential home"
   
   # KALKI searches vector DB
   relevant_chunks = vector_db.search_similar(query, top_k=10)
   
   # Injects chunks into LLM prompt
   prompt = f"Using this context: {relevant_chunks}\n\nDesign: {query}"
   ```

3. **Full-Text Context** - Preserve ALL information for later retrieval

### Storage Location
- **Directory**: `vector_db/` (ChromaDB files)
- **Size**: Large (~1-2 GB per 1000 pages)
- **Format**: Binary vector embeddings + metadata

---

## Track 2: Structured Knowledge Extraction

### What Gets Extracted

**Domain-specific data patterns** get parsed and stored in **structured databases** for instant lookup.

KALKI uses **24 specialized extractors** across 5 domains:

### Construction Domain (6 Extractors)

#### 1. **Span Tables** (`span_tables.db`)

**What It Extracts:**
```
Regex: r"(\d+x\d+)\s+@\s+(\d+)\""
```

**Example From PDF:**
```
Table 9.23.4.2
Joist Span Tables for Residential Floors

2x8  @ 12" O.C. → 13'-7"
2x8  @ 16" O.C. → 12'-4"
2x8  @ 24" O.C. → 10'-8"
2x10 @ 12" O.C. → 17'-4"
2x10 @ 16" O.C. → 16'-1"
2x10 @ 24" O.C. → 13'-7"
```

**Gets Stored As:**
```sql
CREATE TABLE span_tables (
    id INTEGER PRIMARY KEY,
    member_size TEXT,      -- "2x10"
    spacing_inches INTEGER, -- 16
    max_span_feet REAL,    -- 16.08
    load_type TEXT,        -- "residential_floor"
    grade TEXT,            -- "#2 SPF"
    source_pdf TEXT,
    page_number INTEGER
);

INSERT INTO span_tables VALUES 
(1, '2x10', 16, 16.08, 'residential_floor', '#2 SPF', 'BC_Building_Code_Part9_2018.pdf', 142);
```

**How It's Used:**
```python
# When generating BOM for a 20-foot floor span
query = "SELECT member_size FROM span_tables 
         WHERE max_span_feet >= 20 AND load_type = 'residential_floor' 
         ORDER BY member_size ASC LIMIT 1"

# Result: "2x12" → KALKI uses this in the construction drawings!
```

---

#### 2. **Procedures** (`procedures.db`)

**What It Extracts:**
```
Regex: r"Step\s+(\d+)[:.]\s+([A-Z][^\n\.]+)"
```

**Example From PDF:**
```
Foundation Installation Procedure:

Step 1: Excavate to required depth per soil report
Step 2: Install 4" gravel base, compact to 95% density
Step 3: Place #4 rebar on 12" grid, 2" clear from bottom
Step 4: Pour concrete to top of forms, vibrate to remove air
Step 5: Cure for minimum 7 days before loading
```

**Gets Stored As:**
```sql
CREATE TABLE procedures (
    id INTEGER PRIMARY KEY,
    procedure_name TEXT,   -- "Foundation Installation"
    step_number INTEGER,   -- 1, 2, 3, 4, 5
    step_description TEXT, -- "Excavate to required depth..."
    domain TEXT,           -- "construction"
    category TEXT,         -- "foundations"
    source_pdf TEXT
);
```

**How It's Used:**
```python
# When generating construction schedule
procedures = db.query("SELECT * FROM procedures WHERE procedure_name LIKE '%foundation%'")

# KALKI generates step-by-step schedule in deliverables!
# Week 1: Foundation Installation
#   Day 1-2: Excavation
#   Day 3: Gravel base
#   Day 4: Rebar placement
#   Day 5: Concrete pour
```

---

#### 3. **Inspection Criteria** (`inspection_criteria.db`)

**What It Extracts:**
```
Regex: r"[Ii]nspect\s+([^for]+?)\s+for\s+([^\n\.]+)"
```

**Example From PDF:**
```
9.3.2.9 Quality Assurance Inspections

Inspect foundation formwork for alignment, bracing, and cleanliness
Inspect rebar placement for proper spacing, cover, and lap splices
Inspect concrete for proper slump, temperature, and placement technique
Inspect framing connections for proper nailing, bolt size, and spacing
```

**Gets Stored As:**
```sql
CREATE TABLE inspection_criteria (
    id INTEGER PRIMARY KEY,
    inspection_item TEXT,     -- "foundation formwork"
    criteria_list TEXT,       -- "alignment, bracing, cleanliness"
    phase TEXT,               -- "foundation"
    code_reference TEXT,      -- "9.3.2.9"
    source_pdf TEXT
);
```

**How It's Used:**
```python
# When generating inspection checklists deliverable
inspections = db.query("SELECT * FROM inspection_criteria WHERE phase = 'foundation'")

# KALKI creates professional QC checklist:
# ☐ Foundation formwork: alignment, bracing, cleanliness
# ☐ Rebar placement: spacing, cover, lap splices
# ☐ Concrete: slump, temperature, placement
```

---

#### 4. **Cost Data** (`cost_data.db`)

**What It Extracts:**
```
Regex: r"([A-Za-z0-9][^\n:$]+?):\s*\$(\d+\.?\d*)"
```

**Example From PDF (RSMeans Data):**
```
03 31 13.20 - Structural Concrete
  Concrete, 4000 PSI: $142.00 per cubic yard
  Rebar #4, installed: $1.85 per pound
  Formwork, foundation walls: $8.50 per square foot

06 11 10.10 - Wood Framing
  2x10 #2 SPF lumber: $12.45 per linear foot
  2x4 studs: $3.25 each
  Plywood sheathing 1/2": $28.50 per sheet
```

**Gets Stored As:**
```sql
CREATE TABLE cost_data (
    id INTEGER PRIMARY KEY,
    item_description TEXT,  -- "2x10 #2 SPF lumber"
    unit_cost REAL,        -- 12.45
    unit TEXT,             -- "linear foot"
    category TEXT,         -- "wood_framing"
    source TEXT,           -- "RSMeans_2024.pdf"
    year INTEGER           -- 2024
);
```

**How It's Used:**
```python
# When generating cost estimate deliverable
lumber_cost = db.query("SELECT unit_cost FROM cost_data WHERE item_description LIKE '%2x10%'")
total_lumber = 450  # linear feet needed
estimated_cost = lumber_cost * total_lumber  # $5,602.50

# KALKI generates professional cost estimate:
# Material: 2x10 #2 SPF Lumber
# Quantity: 450 LF
# Unit Cost: $12.45/LF
# Total: $5,602.50
```

---

#### 5. **Load Parameters** (`load_parameters.db`)

**What It Extracts:**
```
Regex: r"([A-Za-z\s]+load):\s*(\d+\.?\d*)\s*(PSF|PSI|kN|kPa)"
```

**Example From PDF:**
```
4.1.5.3 Residential Design Loads

Live load: 40 PSF (1.9 kPa)
Dead load: 10 PSF (0.48 kPa)
Snow load: 35 PSF (1.68 kPa) for Vancouver
Roof live load: 20 PSF (0.96 kPa)
Wind load: 28 PSF (1.34 kPa) for Vancouver coastal
```

**Gets Stored As:**
```sql
CREATE TABLE load_parameters (
    id INTEGER PRIMARY KEY,
    load_type TEXT,        -- "snow load"
    value REAL,           -- 35
    unit TEXT,            -- "PSF"
    location TEXT,        -- "Vancouver"
    building_type TEXT,   -- "residential"
    code_reference TEXT   -- "4.1.5.3"
);
```

**How It's Used:**
```python
# When calculating structural requirements
loads = db.query("SELECT * FROM load_parameters WHERE location = 'Vancouver' AND building_type = 'residential'")

# KALKI uses these for structural calculations:
total_load = live_load + dead_load + snow_load  # 40 + 10 + 35 = 85 PSF
# → Selects appropriate beam size from span_tables
```

---

#### 6. **Decision Trees** (`decision_trees.db`)

**What It Extracts:**
```
Regex: r"[Ii]f\s+([^,]+?)\s*([<>=]+)\s*([^,]+?),\s*then"
```

**Example From PDF:**
```
9.4.2.2 Foundation Depth Requirements

If frost penetration depth > 4 feet, then foundation must extend to 4.5 feet
If soil bearing capacity < 1500 PSF, then require engineered foundation design
If building height > 3 stories, then prescriptive path not permitted
If seismic zone >= 0.23g, then require special seismic detailing
```

**Gets Stored As:**
```sql
CREATE TABLE decision_trees (
    id INTEGER PRIMARY KEY,
    condition_param TEXT,     -- "frost penetration depth"
    condition_operator TEXT,  -- ">"
    condition_value TEXT,     -- "4 feet"
    action_required TEXT,     -- "foundation must extend to 4.5 feet"
    code_section TEXT,        -- "9.4.2.2"
    domain TEXT              -- "construction"
);
```

**How It's Used:**
```python
# When validating project requirements
frost_depth = project.get_frost_depth()  # 4.5 feet

# KALKI checks decision trees
rules = db.query("SELECT * FROM decision_trees WHERE condition_param = 'frost penetration depth'")
for rule in rules:
    if eval(f"{frost_depth} {rule.operator} {rule.value}"):
        project.add_requirement(rule.action_required)

# Result: Project requirements updated with "Foundation must extend to 4.5 feet"
```

---

### Game Development Domain (6 Extractors)

#### 1. **Game Mechanics** (`game_mechanics.db`)
- Extracts: Movement systems, combat mechanics, inventory systems
- Pattern: `r"Mechanic:\s+([A-Z][^\n]+)"`
- Use: Suggests appropriate mechanics for new game designs

#### 2. **Engine Features** (`engine_features.db`)
- Extracts: Unity/Unreal/Godot capabilities, APIs, performance tips
- Pattern: `r"(Unity|Unreal|Godot):\s+([^\n]+)"`
- Use: Generates technical specs with engine-specific implementations

#### 3. **Optimization Techniques** (`optimization.db`)
- Extracts: Performance tips, rendering strategies, memory management
- Pattern: `r"Optimize\s+([^by]+?)\s+by\s+([^\n\.]+)"`
- Use: Adds optimization recommendations to technical specs

#### 4. **Monetization Strategies** (`monetization.db`)
- Extracts: IAP strategies, pricing models, retention tactics
- Pattern: `r"Monetization:\s+([^\n]+)"`
- Use: Generates monetization plans for game projects

#### 5. **Multiplayer Patterns** (`multiplayer.db`)
- Extracts: Networking architectures, sync strategies, matchmaking
- Pattern: `r"Multiplayer:\s+([^\n]+)"`
- Use: Designs multiplayer systems in technical specs

#### 6. **Publishing Guidelines** (`publishing.db`)
- Extracts: Platform requirements, certification processes, marketing
- Pattern: `r"Platform:\s+([^-]+?)\s+-\s+([^\n]+)"`
- Use: Creates marketing plans and launch checklists

---

### Robotics Domain (4 Extractors)

#### 1. **Kinematics Formulas** (`kinematics.db`)
- Extracts: DH parameters, forward/inverse kinematics, Jacobians
- Pattern: `r"DH\s+parameters?:\s+([^\n]+)"`
- Use: Calculates robot arm reach and motion planning

#### 2. **Control Systems** (`control_systems.db`)
- Extracts: PID tuning, MPC parameters, adaptive control
- Pattern: `r"(PID|MPC|Control)\s+tuning:\s+([^\n]+)"`
- Use: Designs control code for actuators

#### 3. **SLAM Algorithms** (`slam.db`)
- Extracts: Mapping techniques, localization methods, sensor fusion
- Pattern: `r"SLAM:\s+([^\n]+)"`
- Use: Generates navigation systems for mobile robots

#### 4. **Sensor Specifications** (`sensors.db`)
- Extracts: LiDAR specs, camera parameters, IMU characteristics
- Pattern: `r"Sensor:\s+([^-]+?)\s+-\s+([^\n]+)"`
- Use: Creates BOM with appropriate sensor selections

---

### Aerospace Domain (4 Extractors)

#### 1. **Aerodynamics Data** (`aerodynamics.db`)
- Extracts: Airfoil coefficients, lift/drag curves, CFD results
- Pattern: `r"Airfoil:\s+([^-]+?)\s+-\s+([^\n]+)"`
- Use: Selects wing profiles for UAV designs

#### 2. **Propulsion Systems** (`propulsion.db`)
- Extracts: Motor specs, propeller data, thrust calculations
- Pattern: `r"Motor:\s+([^-]+?)\s+-\s+([^\n]+)"`
- Use: Sizes propulsion system for target performance

#### 3. **Flight Control** (`flight_control.db`)
- Extracts: Autopilot tuning, stability augmentation, control laws
- Pattern: `r"Flight\s+control:\s+([^\n]+)"`
- Use: Configures autopilot systems (Pixhawk, ArduPilot)

#### 4. **Aviation Regulations** (`regulations.db`)
- Extracts: FAA/Transport Canada/EASA rules, certification requirements
- Pattern: `r"Regulation\s+(\d+\.\d+):\s+([^\n]+)"`
- Use: Ensures compliance in designs and test reports

---

### Power Systems Domain (4 Extractors)

#### 1. **Battery Technology** (`batteries.db`)
- Extracts: Chemistry specs, capacity curves, safety parameters
- Pattern: `r"Battery:\s+([^-]+?)\s+-\s+([^\n]+)"`
- Use: Selects battery cells and designs BMS

#### 2. **Solar PV Data** (`solar_pv.db`)
- Extracts: Panel efficiency, MPPT algorithms, degradation rates
- Pattern: `r"Solar\s+panel:\s+([^-]+?)\s+-\s+([^\n]+)"`
- Use: Sizes solar arrays and charge controllers

#### 3. **Power Electronics** (`power_electronics.db`)
- Extracts: Inverter specs, converter topologies, efficiency curves
- Pattern: `r"Inverter:\s+([^-]+?)\s+-\s+([^\n]+)"`
- Use: Designs power conversion stages

#### 4. **Energy Storage Systems** (`energy_storage.db`)
- Extracts: ESS architectures, grid integration, control strategies
- Pattern: `r"ESS:\s+([^\n]+)"`
- Use: Designs battery storage systems with grid-tie capabilities

---

## How They Work Together

### Scenario: User Creates Construction Project

```python
# User command
kalki project create "2-story residential home in Vancouver"

# Step 1: Domain inference identifies "construction"
domain = domain_registry.infer_domain("2-story residential home in Vancouver")
# Result: construction

# Step 2: KALKI queries BOTH systems in parallel

# Track 1: Vector DB Query (Semantic Context)
vector_results = vector_db.search_similar(
    "2-story residential home Vancouver building design",
    top_k=20
)
# Returns: Relevant text chunks from BC Building Code, residential design guides, etc.

# Track 2: Structured Knowledge Queries (Specific Data)
span_data = span_tables_db.query("SELECT * FROM span_tables WHERE load_type = 'residential_floor'")
load_data = load_parameters_db.query("SELECT * FROM load_parameters WHERE location = 'Vancouver'")
costs = cost_data_db.query("SELECT * FROM cost_data WHERE category = 'residential_framing'")
procedures = procedures_db.query("SELECT * FROM procedures WHERE building_type = 'residential'")

# Step 3: KALKI combines both knowledge sources

# Vector results give CONTEXT and UNDERSTANDING
# Structured data gives PRECISE CALCULATIONS and COMPLIANCE

# Step 4: Generate deliverables using BOTH

# Construction Drawings: Uses span_tables + vector context
# BOM: Uses cost_data + material specs from vectors
# Schedule: Uses procedures + timeline estimates from vectors
# Inspection Checklist: Uses inspection_criteria + vector context
```

---

## Why Two Separate Systems?

### Vector DB (ChromaDB) - Strengths:
✅ **Semantic understanding** - "What size beam?" matches "joist span capacity"  
✅ **Flexible queries** - No exact keyword matching needed  
✅ **Rich context** - Preserves full paragraphs and explanations  
✅ **LLM-ready** - Direct injection into prompts for RAG  

### Vector DB - Limitations:
❌ **No structured queries** - Can't say "SELECT all spans > 20 feet"  
❌ **Approximate** - Returns "similar" content, not exact matches  
❌ **Slow for lookups** - Need to search thousands of vectors  
❌ **No calculations** - Can't compute SUM, AVG, or JOIN data  

### Structured DBs (SQLite) - Strengths:
✅ **Precise lookups** - Exact SQL queries for specific data  
✅ **Fast retrieval** - Indexed queries in milliseconds  
✅ **Calculations** - Can aggregate, filter, sort, join  
✅ **Compact storage** - Tables more efficient than vectors  

### Structured DBs - Limitations:
❌ **Exact matches only** - Must know exact column/value  
❌ **No semantic understanding** - "beam size" ≠ "joist dimensions"  
❌ **Manual extraction** - Need regex patterns for each data type  
❌ **Domain-specific** - Different schema per domain  

---

## Storage Locations

### Vector Database (ChromaDB)
```
vector_db/
├── chroma.sqlite3           # Metadata index
├── embeddings/              # Binary vector storage
│   ├── collection_0.bin
│   ├── collection_1.bin
│   └── ...
└── known_hashes.json        # Deduplication tracking
```

**Size**: ~1-2 GB per 1,000 PDF pages

### Structured Knowledge Databases
```
data/knowledge_db/
├── construction/
│   ├── span_tables.db          # ~5 MB (500 tables)
│   ├── procedures.db           # ~2 MB (200 procedures)
│   ├── inspection_criteria.db  # ~1 MB (150 criteria)
│   ├── cost_data.db           # ~10 MB (1000+ items)
│   ├── load_parameters.db     # ~500 KB (100 loads)
│   └── decision_trees.db      # ~1 MB (200 rules)
│
├── game_dev/
│   ├── game_mechanics.db
│   ├── engine_features.db
│   └── ...
│
├── robotics/
│   ├── kinematics.db
│   ├── control_systems.db
│   └── ...
│
├── aerospace/
│   ├── aerodynamics.db
│   ├── propulsion.db
│   └── ...
│
└── power_systems/
    ├── batteries.db
    ├── solar_pv.db
    └── ...
```

**Size**: ~50-100 MB total across all domains

---

## Extraction Statistics (After 50 PDFs)

### Construction Domain
| Knowledge Type | Target | Current | Example Query |
|---------------|--------|---------|---------------|
| Span Tables | 500+ | 0 | "What size beam for 20ft span?" |
| Procedures | 200+ | 0 | "Foundation installation steps" |
| Inspection Criteria | 150+ | 0 | "Framing inspection checklist" |
| Cost Data | 1000+ | 0 | "Cost of 2x10 lumber per LF" |
| Load Parameters | 100+ | 0 | "Snow load for Vancouver" |
| Decision Trees | 200+ | 0 | "When is engineer required?" |

### Vector Embeddings
| Metric | Value |
|--------|-------|
| Total Chunks | 0 |
| Average Chunk Size | 512 tokens |
| Embedding Dimensions | 768 |
| Total Vectors | 0 |
| Storage Size | 0 MB |

---

## Example Queries & Which System Handles Them

### Semantic Questions → Vector DB
```bash
kalki ask "What are best practices for concrete curing?"
# → Searches vector DB for similar content
# → Returns paragraphs from PDF about curing methods
```

### Specific Data → Structured DB
```bash
kalki ask "What's the max span for a 2x10 joist at 16 inch spacing?"
# → Queries span_tables.db with exact parameters
# → Returns: 16'-1" from structured table
```

### Complex Queries → BOTH Systems
```bash
kalki ask "Design a floor system for a 22-foot span residential room"
# → Vector DB: Gets context about floor design best practices
# → Structured DB: Queries span_tables for members that span 22+ feet
# → KALKI combines: "Use 2x12 joists at 12\" O.C. with proper blocking per BC Building Code 9.23.4.2"
```

---

## Summary

**Vector DB (ChromaDB):**
- Stores: Full text chunks with semantic embeddings
- Purpose: Semantic search, RAG, context retrieval
- Format: 768-dimensional vectors
- Size: Large (~1-2 GB/1000 pages)
- Query: Similarity search
- Separate: YES - completely independent storage

**Structured DB (SQLite):**
- Stores: Extracted structured data (tables, procedures, costs, etc.)
- Purpose: Precise lookups, calculations, aggregations
- Format: SQL tables with typed columns
- Size: Compact (~50-100 MB total)
- Query: SQL SELECT statements
- Separate: YES - domain-specific databases

**Both Work Together:**
1. Vector DB provides **context and understanding**
2. Structured DB provides **precise data and calculations**
3. KALKI Control Hub orchestrates **both simultaneously**
4. Result: **Professional deliverables** combining semantic AI with structured engineering data

---

## Next Steps

1. **Ingest your PDFs**: `kalki learn ingest path/to/pdf`
2. **Check extraction stats**: `kalki learn stats`
3. **Test queries**: `kalki ask "your question"`
4. **Generate deliverables**: `kalki project deliverable <id> <type>`

Both systems will populate automatically as you ingest PDFs! 🚀
