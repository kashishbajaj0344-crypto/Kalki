# 🎯 Quick Reference: KALKI Knowledge Extraction

## TL;DR

**KALKI extracts knowledge in TWO COMPLETELY SEPARATE ways:**

1. **Vector Embeddings** → Semantic search (ChromaDB)
2. **Structured Data** → Precise lookups (SQLite)

**They are separate but work together!**

---

## Visual Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     PDF INGESTION                            │
│                                                              │
│  "BC Building Code Part 9" (500 pages)                      │
└──────────────────┬──────────────────────────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Text Extraction     │
        │  + OCR + Tables      │
        └──────────┬───────────┘
                   │
                   │ Split into 512-token chunks
                   │
                   ▼
    ┌──────────────────────────────────┐
    │  Chunk 1: "2x10 joists at..."   │
    │  Chunk 2: "Foundation depth..."  │
    │  Chunk 3: "Snow load for..."     │
    │  ...                              │
    │  Chunk 847: "Inspection of..."   │
    └──────────────┬───────────────────┘
                   │
                   │ SPLIT INTO TWO TRACKS
                   │
       ┌───────────┴───────────┐
       │                       │
       ▼                       ▼
┌─────────────┐        ┌──────────────┐
│  TRACK 1    │        │  TRACK 2     │
│  VECTOR     │        │  STRUCTURED  │
│  EMBEDDINGS │        │  EXTRACTION  │
└─────────────┘        └──────────────┘
       │                       │
       ▼                       ▼
┌─────────────────┐    ┌──────────────────────┐
│ BGE-Large       │    │ Regex + NLP Patterns │
│ Transformer     │    │                      │
│ "2x10 joists.." │    │ Pattern: (\d+x\d+)   │
│      ↓          │    │ "2x10" → span_tables │
│ [768 numbers]   │    │                      │
│ [0.02, -0.14..] │    │ Pattern: Step \d+    │
└────────┬────────┘    │ "Step 1: Excavate"   │
         │             │      → procedures     │
         ▼             └──────────┬───────────┘
┌─────────────────┐              │
│   ChromaDB      │              ▼
│   vector_db/    │    ┌──────────────────────┐
│                 │    │   SQLite Databases   │
│ Stores:         │    │   data/knowledge_db/ │
│ • Text chunks   │    │                      │
│ • Embeddings    │    │ span_tables.db       │
│ • Metadata      │    │ procedures.db        │
│ • Tags          │    │ inspection_criteria.db│
│                 │    │ cost_data.db         │
│ Size: ~2 GB     │    │ load_parameters.db   │
│ Purpose:        │    │ decision_trees.db    │
│ Semantic search │    │                      │
│ RAG retrieval   │    │ Size: ~50 MB         │
│                 │    │ Purpose:             │
│                 │    │ Precise lookups      │
│                 │    │ SQL queries          │
└────────┬────────┘    └──────────┬───────────┘
         │                        │
         │    ┌───────────────────┘
         │    │
         ▼    ▼
    ┌────────────────────┐
    │  KALKI QUERY       │
    │  Supreme Hub       │
    └─────────┬──────────┘
              │
              ▼
    ┌─────────────────────────────┐
    │ "What size beam for 20ft?"  │
    └─────────────────────────────┘
              │
              ├──→ Vector: Get context about beam design
              │    Returns: Paragraphs explaining loads, spans, codes
              │
              └──→ Structured: Query span_tables WHERE span >= 20
                   Returns: "2x12 at 12\" O.C. → 20'-6\""
                   
              ▼
    ┌─────────────────────────────┐
    │  COMBINED ANSWER:           │
    │  "Use 2x12 joists at 12\"   │
    │   spacing for 20ft span.    │
    │   Per BC Building Code      │
    │   9.23.4.2, max span is     │
    │   20'-6\" for residential    │
    │   floor loads."              │
    └─────────────────────────────┘
```

---

## Key Differences

| Aspect | Vector Embeddings | Structured Extraction |
|--------|-------------------|----------------------|
| **Storage** | ChromaDB (vector_db/) | SQLite (data/knowledge_db/) |
| **Format** | 768-dim vectors + text | SQL tables with typed columns |
| **Size** | ~2 GB per 1000 pages | ~50 MB total |
| **Query Type** | Semantic similarity | SQL SELECT |
| **Example Query** | "best practices for concrete" | "SELECT * WHERE span >= 20" |
| **Purpose** | Understanding, context, RAG | Calculations, lookups, precision |
| **Extraction** | Automatic (all chunks) | Pattern-based (regex + NLP) |
| **Accuracy** | Approximate (similarity) | Exact (structured data) |
| **Speed** | Slower (vector search) | Fast (indexed tables) |
| **Maintenance** | Automatic | Requires extractor patterns |

---

## Extraction Examples

### From BC Building Code PDF

**Original Text:**
```
Table 9.23.4.2 - Maximum Spans for Joists

Joist spacing 400 mm (16") on centre

2x8  #2 grade SPF: 3.76 m (12'-4")
2x10 #2 grade SPF: 4.88 m (16'-0") 
2x12 #2 grade SPF: 6.25 m (20'-6")

For residential floor loading (1.9 kPa live + 0.5 kPa dead)
```

**Track 1: Vector Embedding**
```python
{
    "chunk_text": "Table 9.23.4.2 - Maximum Spans for Joists\n\nJoist spacing 400 mm (16\") on centre\n\n2x8 #2 grade SPF: 3.76 m (12'-4\")\n2x10 #2 grade SPF: 4.88 m (16'-0\") \n2x12 #2 grade SPF: 6.25 m (20'-6\")\n\nFor residential floor loading (1.9 kPa live + 0.5 kPa dead)",
    
    "embedding": [0.023, -0.145, 0.891, 0.456, -0.234, ... 768 numbers total],
    
    "metadata": {
        "source": "BC_Building_Code_Part9_2018.pdf",
        "page": 142,
        "chunk_id": "abc123_142_chunk_5",
        "tags": ["structural", "joists", "span_tables", "residential", "wood_framing"]
    }
}
```
**Stored in**: `vector_db/chroma.sqlite3` + `vector_db/embeddings/`

---

**Track 2: Structured Extraction**

**Extractor 1: Span Tables**
```sql
-- Pattern matches: "2x10 #2 grade SPF: 4.88 m"
INSERT INTO span_tables (
    member_size, grade, species, spacing_mm, 
    max_span_m, load_type, code_ref, source_pdf, page
) VALUES 
    ('2x8',  '#2', 'SPF', 400, 3.76, 'residential_floor', '9.23.4.2', 'BC_Building_Code_Part9_2018.pdf', 142),
    ('2x10', '#2', 'SPF', 400, 4.88, 'residential_floor', '9.23.4.2', 'BC_Building_Code_Part9_2018.pdf', 142),
    ('2x12', '#2', 'SPF', 400, 6.25, 'residential_floor', '9.23.4.2', 'BC_Building_Code_Part9_2018.pdf', 142);
```

**Extractor 2: Load Parameters**
```sql
-- Pattern matches: "1.9 kPa live"
INSERT INTO load_parameters (
    load_type, value_kpa, building_type, code_ref, source_pdf
) VALUES
    ('live_load',  1.9, 'residential', '9.23.4.2', 'BC_Building_Code_Part9_2018.pdf'),
    ('dead_load',  0.5, 'residential', '9.23.4.2', 'BC_Building_Code_Part9_2018.pdf');
```

**Stored in**: 
- `data/knowledge_db/construction/span_tables.db`
- `data/knowledge_db/construction/load_parameters.db`

---

## Query Examples

### Semantic Query (Uses Vector DB)

```bash
$ kalki ask "What are the requirements for concrete foundations?"

# KALKI searches vector embeddings for similar content
# Returns top 10 chunks about foundations from multiple PDFs
# LLM synthesizes answer from context

Answer: "According to BC Building Code 9.15.3, concrete foundations 
must extend below frost depth (1.2m in Vancouver), have minimum 
4000 PSI concrete, #4 rebar on 12\" grid, and be cured for 7 days 
before loading..."
```

### Structured Query (Uses SQLite)

```bash
$ kalki ask "What size joist for 18 foot span?"

# KALKI queries span_tables.db
SELECT member_size, max_span_m 
FROM span_tables 
WHERE max_span_m >= 5.49 
  AND spacing_mm = 400 
  AND load_type = 'residential_floor'
ORDER BY member_size ASC 
LIMIT 1;

# Result: 2x10 (spans up to 4.88m / 16'-0")
# Next size up: 2x12 (spans up to 6.25m / 20'-6")

Answer: "For an 18-foot span, you need 2x12 joists at 16\" spacing. 
This provides a maximum span of 20'-6\", giving you adequate capacity 
for your 18-foot requirement per BC Building Code Table 9.23.4.2."
```

### Hybrid Query (Uses BOTH)

```bash
$ kalki project deliverable <project-id> bill_of_materials

# Step 1: Vector DB provides design context
vector_context = search_similar("residential framing materials lumber")
# Returns: Chapters about framing, material specs, installation methods

# Step 2: Structured DB provides exact data
span_requirements = query("SELECT * FROM span_tables WHERE...")
cost_data = query("SELECT unit_cost FROM cost_data WHERE item LIKE '%2x10%'")
material_specs = query("SELECT * FROM material_properties WHERE...")

# Step 3: KALKI generates BOM combining both
# Vector context: Ensures complete material list (nails, fasteners, etc.)
# Structured data: Exact quantities, costs, specifications

Generated BOM:
┌──────────────────┬──────┬──────┬────────┬──────────┐
│ Item             │ Qty  │ Unit │ Cost   │ Total    │
├──────────────────┼──────┼──────┼────────┼──────────┤
│ 2x10 #2 SPF     │ 450  │ LF   │ $12.45 │ $5,602.50│
│ 2x4 Studs       │ 180  │ EA   │ $3.25  │ $585.00  │
│ 3/4" Plywood    │ 42   │ SHT  │ $45.50 │ $1,911.00│
│ 16d Nails       │ 50   │ LB   │ $2.85  │ $142.50  │
└──────────────────┴──────┴──────┴────────┴──────────┘
```

---

## Storage Hierarchy

```
/Users/kashish/Desktop/Kalki/
│
├── vector_db/                    # TRACK 1: Vector Embeddings
│   ├── chroma.sqlite3           # Metadata index
│   ├── embeddings/              # Binary vector storage
│   │   ├── collection_0.bin    # Compressed embeddings
│   │   └── collection_1.bin
│   └── known_hashes.json        # Deduplication
│
└── data/
    └── knowledge_db/            # TRACK 2: Structured Data
        │
        ├── construction/
        │   ├── span_tables.db
        │   ├── procedures.db
        │   ├── inspection_criteria.db
        │   ├── cost_data.db
        │   ├── load_parameters.db
        │   └── decision_trees.db
        │
        ├── game_dev/
        │   ├── game_mechanics.db
        │   ├── engine_features.db
        │   ├── optimization.db
        │   ├── monetization.db
        │   ├── multiplayer.db
        │   └── publishing.db
        │
        ├── robotics/
        │   ├── kinematics.db
        │   ├── control_systems.db
        │   ├── slam.db
        │   └── sensors.db
        │
        ├── aerospace/
        │   ├── aerodynamics.db
        │   ├── propulsion.db
        │   ├── flight_control.db
        │   └── regulations.db
        │
        └── power_systems/
            ├── batteries.db
            ├── solar_pv.db
            ├── power_electronics.db
            └── energy_storage.db
```

---

## 24 Knowledge Extractors Across 5 Domains

### Construction (6)
1. **span_tables** - Structural member sizing
2. **procedures** - Step-by-step construction sequences
3. **inspection_criteria** - QC validation points
4. **cost_data** - Material/labor costs
5. **load_parameters** - Design loads (live, dead, snow, wind)
6. **decision_trees** - Code compliance logic

### Game Development (6)
7. **game_mechanics** - Gameplay systems
8. **engine_features** - Unity/Unreal capabilities
9. **optimization** - Performance techniques
10. **monetization** - Revenue strategies
11. **multiplayer** - Networking patterns
12. **publishing** - Platform requirements

### Robotics (4)
13. **kinematics** - Motion math and DH parameters
14. **control_systems** - PID/MPC tuning
15. **slam** - Mapping and localization
16. **sensors** - Sensor specifications

### Aerospace (4)
17. **aerodynamics** - Airfoil data and CFD
18. **propulsion** - Motor/propeller specs
19. **flight_control** - Autopilot configuration
20. **regulations** - FAA/EASA rules

### Power Systems (4)
21. **batteries** - Cell chemistry and BMS
22. **solar_pv** - Panel specs and MPPT
23. **power_electronics** - Inverter designs
24. **energy_storage** - ESS architectures

---

## How to Check What's Extracted

### View Vector DB Stats
```bash
$ kalki learn stats

Vector Database Statistics:
  Total documents: 0
  Total chunks: 0
  Total embeddings: 0
  Storage size: 0 MB
  Domains covered: []
```

### View Structured DB Stats
```bash
$ python3 -c "
import sqlite3
from pathlib import Path

db_dir = Path('data/knowledge_db/construction')
for db_file in db_dir.glob('*.db'):
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM sqlite_master WHERE type=\"table\"')
    tables = cursor.fetchone()[0]
    print(f'{db_file.name}: {tables} tables')
    conn.close()
"
```

### Test Queries
```bash
# Semantic search (vector DB)
$ kalki ask "What are foundation design requirements?"

# Structured lookup (SQLite)
$ kalki ask "Show me span table for 2x10 joists"

# Hybrid (both systems)
$ kalki project deliverable <id> construction_drawings
```

---

## Summary

✅ **Two completely separate systems**  
✅ **Vector DB**: Semantic understanding + RAG  
✅ **Structured DB**: Precise calculations + lookups  
✅ **24 extractors** across 5 domains  
✅ **Both populated automatically** from PDFs  
✅ **Work together** for professional deliverables  

**Start ingesting PDFs to populate both!** 🚀
