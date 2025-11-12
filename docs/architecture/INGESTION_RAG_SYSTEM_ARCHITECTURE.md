# KALKI Ingestion Pipeline & RAG System Architecture

## Complete System Interaction Flow

This document explains how KALKI's various subsystems interact with the document ingestion pipeline and RAG (Retrieval-Augmented Generation) system.

---

## 🏗️ **Architecture Overview**

```
┌─────────────────────────────────────────────────────────────────────┐
│                      USER INPUT / PDF FILES                          │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  📥 INGESTION PIPELINE (Multi-Stage Knowledge Extraction)            │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                       │
│  Stage 1: DocumentIngestAgent + DocumentIngestor                     │
│  ├─ Discover PDFs (drag & drop, folder scan, CLI)                   │
│  ├─ Extract text (pdfplumber with OCR fallback)                     │
│  ├─ Extract tables (structured data preservation)                   │
│  ├─ Extract metadata (title, author, date, page count)              │
│  └─ Generate file hash (SHA256 deduplication)                       │
│                                                                       │
│  Stage 2: KnowledgeExtractor (Hybrid Learning System)                │
│  ├─ Regex pattern matching for structured knowledge                 │
│  ├─ Formula extraction (M = wL²/8, A = πr², SF = yield/stress)     │
│  ├─ Material properties (6061, 4140, concrete specs)                │
│  ├─ Design rules (shall/must/should imperatives)                    │
│  └─ Code requirements (Section X.X format)                          │
│                                                                       │
│  Stage 3: Dual Storage Strategy                                      │
│  ├─ Vector DB (ChromaDB) → Semantic search                          │
│  │  └─ Embeddings: BAAI/bge-large-en-v1.5 (1024-dim vectors)       │
│  └─ Structured DBs (4 SQLite databases) → Fast lookup               │
│     ├─ formulas.db (4,896 engineering formulas)                     │
│     ├─ materials.db (material properties + standards)               │
│     ├─ design_rules.db (best practices + imperatives)               │
│     └─ code_requirements.db (building codes + compliance)           │
│                                                                       │
│  Stage 4: Archive Management                                         │
│  └─ data/pdf_archive/ → Original PDFs preserved                     │
│                                                                       │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  🔍 RAG QUERY SYSTEM (Retrieval-Augmented Generation)                │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                       │
│  ┌─ Hybrid Retrieval Strategy ─────────────────────────────────┐    │
│  │                                                               │    │
│  │  1. SEMANTIC SEARCH (VectorDBAdapter)                        │    │
│  │     ├─ Query → BAAI/bge embeddings → ChromaDB                │    │
│  │     ├─ Cosine similarity search (top_k=10)                   │    │
│  │     ├─ Metadata filters (domain, date, author)               │    │
│  │     └─ Returns: RAGResult[] with similarity scores           │    │
│  │                                                               │    │
│  │  2. STRUCTURED LOOKUP (KnowledgeExtractor)                   │    │
│  │     ├─ Query formulas by domain (structural, electrical)     │    │
│  │     ├─ Query materials by name (aluminum, steel, concrete)   │    │
│  │     ├─ Query design rules by category (safety, efficiency)   │    │
│  │     └─ Query codes by type (building, electrical, plumbing)  │    │
│  │                                                               │    │
│  │  3. HYBRID SCORING (Configurable Weights)                    │    │
│  │     ├─ Score = (α × semantic_score) + (β × text_match)      │    │
│  │     ├─ Default: α=0.5, β=0.5 (user-adjustable)              │    │
│  │     └─ Normalization + deduplication                         │    │
│  │                                                               │    │
│  │  4. LLM RERANKING (Optional Enhancement)                     │    │
│  │     ├─ Top-K results → LLM relevance assessment              │    │
│  │     ├─ Semantic coherence + query alignment                  │    │
│  │     └─ Final ranking with rerank_score                       │    │
│  │                                                               │    │
│  └───────────────────────────────────────────────────────────────┘    │
│                                                                       │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ⚡ SUPREME CONTROL HUB (Intelligence Orchestration)                 │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                       │
│  Process Flow for User Query:                                        │
│                                                                       │
│  1. CONSCIOUSNESS ASSESSMENT                                         │
│     └─ ConsciousnessEngine.achieve_consciousness()                  │
│        → awareness_level (0.0-1.0)                                   │
│                                                                       │
│  2. META-COGNITIVE DEPTH SELECTION                                   │
│     └─ MetaCore.assess_task_complexity()                            │
│        → ReasoningDepth (STANDARD, ADVANCED, DEEP)                  │
│                                                                       │
│  3. HYBRID KNOWLEDGE RETRIEVAL ★ RAG Integration Point               │
│     ├─ HybridLearningSystem.query_formulas()                        │
│     │  └─ Returns: Top 20 relevant formulas from formulas.db        │
│     ├─ HybridLearningSystem.query_materials()                       │
│     │  └─ Returns: Top 10 material specs from materials.db          │
│     ├─ HybridLearningSystem.query_design_rules()                    │
│     │  └─ Returns: Top 15 design rules from design_rules.db         │
│     ├─ HybridLearningSystem.query_code_requirements()               │
│     │  └─ Returns: Top 10 code requirements from codes.db           │
│     └─ VectorDBManager.search_similar(query, top_k=10)              │
│        └─ Returns: Top 10 semantic chunks from ChromaDB             │
│                                                                       │
│  4. SUPREME SYNTHESIS                                                │
│     └─ SupremeSynthesisEngine.synthesize()                          │
│        Input: query + full knowledge context                         │
│        Process:                                                      │
│        ├─ Engineering analysis (formulas + materials)                │
│        ├─ Aesthetic evaluation (golden ratio + proportion)           │
│        ├─ Ethical assessment (safety + sustainability)               │
│        └─ Semantic context integration (RAG chunks)                  │
│        Output: Conceptual blueprint                                  │
│                                                                       │
│  5. DESIGN GENERATION (If applicable)                                │
│     └─ DesignBrain.generate_from_intent()                           │
│        → Detailed 3D models + specs                                  │
│                                                                       │
│  6. SELF-EVOLUTION FEEDBACK                                          │
│     └─ SelfEvolutionManager.record_performance()                    │
│        → Learning loop for system improvement                        │
│                                                                       │
└────────────┬────────────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────────────┐
│  🎯 KALKI ORCHESTRATOR (User-Facing Interface)                       │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  │
│                                                                       │
│  Entry Points:                                                       │
│  ├─ KalkiOrchestrator.process_user_query()                          │
│  │  └─ Routes to appropriate agent based on query analysis          │
│  ├─ KalkiOrchestrator.create_design()                               │
│  │  └─ Full design pipeline with RAG knowledge integration          │
│  └─ KalkiCLI commands                                                │
│     ├─ kalki learn ingest <pdf>      → DocumentIngestAgent          │
│     ├─ kalki learn query <topic>     → RAG search                   │
│     ├─ kalki query <task>    → Control Hub          │
│     └─ kalki design create <request> → Design generation            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 **Example User Journey: "Design a robotics arm"**

### **Step 1: Knowledge Retrieval (RAG System)**

```python
# User Query: "Design a 6-DOF robotics arm with 5kg payload"

# Semantic Search (Vector DB)
vector_results = vector_db.search_similar(
    "robotics arm 6 DOF payload actuator", 
    top_k=10
)
# Returns:
# - "Robotics arm design principles" (similarity: 0.89)
# - "Actuator selection for industrial robots" (similarity: 0.84)
# - "Payload capacity calculations" (similarity: 0.81)
# - ... 7 more chunks

# Structured Knowledge Lookup
formulas = hybrid_system.query_formulas(domain="mechanical")
# Returns:
# - Torque formula: τ = F × r × sin(θ)
# - Joint load: L = m × g × d
# - Moment of inertia: I = Σ(m × r²)
# - ... 17 more formulas

materials = hybrid_system.query_materials()
# Returns:
# - Aluminum 6061: yield_strength=276 MPa
# - Steel 4140: yield_strength=415 MPa
# - Carbon fiber composite: density=1.6 g/cm³
# - ... 7 more materials

design_rules = hybrid_system.query_design_rules(category="safety")
# Returns:
# - "Safety factor shall be minimum 2.0 for dynamic loads"
# - "Joint limits must include mechanical stops"
# - "Emergency stop shall halt motion within 0.5 seconds"
# - ... 12 more rules
```

### **Step 2: Supreme Synthesis (Intelligence Integration)**

```python
# Supreme Control Hub receives full context
synthesis_result = await supreme_hub.process_supreme_task(
    task="Design 6-DOF robotics arm with 5kg payload",
    mode="supreme"
)

# Internal Processing:
# 1. Consciousness → awareness_level: 0.87 (high complexity task)
# 2. Reasoning Depth → DEEP (multi-domain engineering)
# 3. Engineering Analysis:
#    - Calculate torque requirements using retrieved formulas
#    - Select materials based on strength/weight ratio
#    - Apply safety factors from design rules
# 4. Aesthetic Evaluation:
#    - Optimize for visual balance
#    - Consider industrial design principles
# 5. Ethics Assessment:
#    - Safety factor compliance ✓
#    - Environmental impact (material selection) ✓

# Output:
{
    "conceptual_blueprint": {
        "joint_count": 6,
        "materials": ["aluminum_6061", "steel_4140"],
        "actuators": [
            {"joint": 1, "torque_nm": 120, "type": "brushless_dc"},
            {"joint": 2, "torque_nm": 90, "type": "brushless_dc"},
            # ... 4 more joints
        ],
        "safety_factor": 2.5,
        "total_weight_kg": 24.3,
        "estimated_cost_usd": 4850
    },
    "knowledge_used": {
        "formulas": 8,  # Applied 8 mechanical formulas
        "materials": 3,  # Evaluated 3 material options
        "design_rules": 5,  # Validated against 5 rules
        "semantic_chunks": 4  # Used 4 RAG chunks
    },
    "quality_score": 0.91
}
```

### **Step 3: Design Brain Generation**

```python
# Design Brain uses synthesis blueprint
design = await design_brain.generate_from_intent(
    intent=synthesis_result["conceptual_blueprint"]
)

# Generates:
# - 3D CAD models (FreeCAD .FCStd files)
# - Bill of Materials (BOM with costs from cost_data.db)
# - Assembly instructions (step-by-step)
# - QC checklists (from design_rules.db)
# - Technical drawings (2D projections)
```

---

## 📊 **System Statistics & Current Capabilities**

### **Ingestion Pipeline (Current State)**

| Component | Status | Details |
|-----------|--------|---------|
| **PDFs Processed** | ✅ 446 PDFs | From academia.edu + construction manuals |
| **Vector DB Size** | ✅ ~10,000 chunks | ChromaDB with BGE-large-en-v1.5 embeddings |
| **Formulas Extracted** | ✅ 4,896 formulas | Engineering equations (structural, mechanical, electrical) |
| **Materials Database** | ⚠️ 2 materials | **NEEDS EXPANSION** (target: 500+ materials) |
| **Design Rules** | ⚠️ 9 rules | **NEEDS EXPANSION** (target: 200+ rules) |
| **Code Requirements** | ⚠️ 2 codes | **NEEDS EXPANSION** (target: 1,000+ code sections) |

### **RAG Query Performance**

| Metric | Current | Target (v2.5) |
|--------|---------|---------------|
| **Semantic Search Latency** | 120ms | 80ms |
| **Hybrid Search Accuracy** | 72% | 90% |
| **Reranking Quality** | N/A | 95% relevance |
| **Knowledge Coverage** | 10% construction domain | 70% construction domain |

---

## 🔧 **KALKI v2.5 Enhancement Plan**

### **Priority 1: Enhanced PDF Extraction** (Months 1-2)

**Current Gaps:**
- ❌ Only extracts formulas (regex-based)
- ❌ Missing span tables (critical for structural design)
- ❌ Missing procedural knowledge (construction sequences)
- ❌ Missing inspection criteria (QC validation)
- ❌ Missing cost data (material/labor unit costs)

**Enhanced Extraction Strategy:**

```python
# Add 6 new extraction methods to KnowledgeExtractor:

def _extract_span_tables(self, content: str) -> List[SpanTableEntry]:
    """
    Extract structural member sizing tables
    Example: "2x6 joists @ 16" O.C. can span 12'3" for 40 PSF live load"
    """
    pattern = r'(\d+x\d+)\s+.*?span\s+(\d+[\'"]?\d*[\""]?)'
    # Store in span_tables.db

def _extract_procedures(self, content: str) -> List[Procedure]:
    """
    Extract step-by-step construction sequences
    Example: "1. Install vapor barrier 2. Frame walls 3. Install sheathing"
    """
    pattern = r'(\d+)\.\s+([A-Z][^.]+)'
    # Store in procedures.db

def _extract_inspection_criteria(self, content: str) -> List[InspectionCriteria]:
    """
    Extract quality control validation points
    Example: "Inspect foundation for cracks > 1/4 inch"
    """
    pattern = r'[Ii]nspect.*?for\s+([^.]+)'
    # Store in inspection_criteria.db

def _extract_cost_data(self, content: str) -> List[CostData]:
    """
    Extract material/labor unit costs
    Example: "2x4 studs: $3.50/ea, Framing labor: $45/hr"
    """
    pattern = r'(\w+[\w\s]+):\s*\$(\d+\.?\d*)'
    # Store in cost_data.db

def _extract_load_parameters(self, content: str) -> List[LoadParameter]:
    """
    Extract structural load values
    Example: "Residential floor live load: 40 PSF"
    """
    pattern = r'(\w+[\w\s]+load):\s*(\d+)\s*(PSF|PSI|kN)'
    # Store in load_parameters.db

def _extract_decision_trees(self, content: str) -> List[DecisionTree]:
    """
    Extract conditional code compliance logic
    Example: "If height > 35 feet, then require sprinkler system"
    """
    pattern = r'[Ii]f\s+([^,]+),\s+then\s+([^.]+)'
    # Store in decision_trees.db
```

### **Priority 2: Advanced RAG Enhancements** (Months 3-4)

**Semantic Reranking with LLM:**
```python
# Current: Basic cosine similarity
# Enhanced: LLM-based relevance scoring

async def semantic_rerank(results: List[RAGResult], query: str):
    """
    Use Llama-3.1-8B to rerank results by true semantic relevance
    
    Prompt: "Rate how relevant this passage is to the query..."
    Returns: Confidence scores 0-1 for each result
    """
    rerank_scores = await llm_engine.generate(
        prompt=f"Rate relevance of passages to: {query}...",
        context=results
    )
    # Resort results by rerank_score
```

**Multi-Query Fusion:**
```python
# Generate multiple query variations for better recall
queries = [
    "robotics arm payload capacity",
    "robot manipulator weight limits",
    "industrial arm strength specifications"
]
results = await rag_query.batch_query_embeddings(queries)
# Merge and deduplicate results
```

### **Priority 3: Construction-Specific Knowledge Graph** (Months 5-6)

**Current:** Flat document chunks + isolated structured data  
**Target:** Interconnected knowledge graph

```
┌─────────────────────────────────────────────────┐
│     CONSTRUCTION KNOWLEDGE GRAPH (Neo4j)         │
├─────────────────────────────────────────────────┤
│                                                  │
│  [Material: Concrete]                            │
│     │                                            │
│     ├─ requires → [Procedure: Curing]           │
│     ├─ governed_by → [Code: ACI 318]            │
│     ├─ strength → [Formula: f'c = 4000 PSI]     │
│     └─ inspected_by → [Criteria: Slump test]    │
│                                                  │
│  [Component: Foundation]                         │
│     │                                            │
│     ├─ uses → [Material: Concrete]              │
│     ├─ sized_by → [SpanTable: Footing sizes]    │
│     ├─ cost → [CostData: $8.50/cu.ft]          │
│     └─ validated_by → [Inspection: Rebar check] │
│                                                  │
└─────────────────────────────────────────────────┘
```

---

## 💡 **Key Integration Points for v2.5 Features**

### **Construction Companion (Continuous Support)**

```python
# RAG system provides contextual knowledge at each stage

# Week 1: Foundation Design
knowledge = rag_query.query_with_context(
    query="foundation design for 3-story building",
    context={"project_phase": "foundation", "location": "BC"},
    filters={"code_type": "building", "domain": "structural"}
)
# Returns: BC Building Code Part 4 + soil bearing capacity tables

# Week 8: Framing Inspection
knowledge = rag_query.query_with_context(
    query="framing inspection checklist",
    context={"project_phase": "framing_inspection"},
    filters={"category": "inspection_criteria"}
)
# Returns: Stud spacing requirements + nailing schedules + inspection photos
```

### **Computer Vision Quality Control**

```python
# RAG provides reference images + inspection criteria

inspection_context = rag_query.query_multimodal(
    query="foundation crack inspection",
    modalities=["text", "image"],
    filters={"content_type": "inspection_criteria"}
)
# Returns:
# - Text: "Cracks > 1/4 inch require engineer review"
# - Images: Reference photos of acceptable vs. unacceptable cracks

# Computer vision compares site photo against references
vision_result = vision_agent.analyze(
    site_photo=user_upload,
    reference_context=inspection_context
)
```

### **Professional Marketplace Integration**

```python
# RAG helps match professionals to project needs

project_requirements = rag_query.analyze_project(
    project_data=project_state
)
# Returns:
# - Required professional licenses: ["P.Eng structural", "Architect AIBC"]
# - Specialized expertise: ["seismic design", "timber framing"]
# - Typical billing rates: "$150-200/hr for structural P.Eng"

# Match against professional database
matches = professional_marketplace.find_matches(
    requirements=project_requirements,
    location=project.location
)
```

---

## 🚀 **Performance Optimization Strategies**

### **1. Vector DB Caching**
```python
# Cache frequently accessed embeddings
vector_cache = LRUCache(max_size=1000)

def search_similar_cached(query: str, top_k: int = 10):
    cache_key = f"{query}_{top_k}"
    if cache_key in vector_cache:
        return vector_cache[cache_key]
    
    results = vector_db.search_similar(query, top_k)
    vector_cache[cache_key] = results
    return results
```

### **2. Structured DB Indexing**
```sql
-- Add indexes to speed up structured queries
CREATE INDEX idx_formulas_domain ON formulas(domain);
CREATE INDEX idx_materials_name ON materials(material_name);
CREATE INDEX idx_rules_category ON design_rules(category);
CREATE INDEX idx_codes_type ON code_requirements(code_type);
```

### **3. Parallel Query Execution**
```python
# Run semantic + structured queries in parallel
async def hybrid_query_parallel(query: str):
    semantic_task = vector_db.search_similar(query, top_k=10)
    formulas_task = hybrid_system.query_formulas()
    materials_task = hybrid_system.query_materials()
    
    # Execute all queries concurrently
    results = await asyncio.gather(
        semantic_task, formulas_task, materials_task
    )
    return merge_and_rank(results)
```

---

## 📈 **Success Metrics for RAG System**

| Metric | Current | Month 3 Target | Month 6 Target | Month 12 Target |
|--------|---------|----------------|----------------|-----------------|
| **Knowledge Coverage** | 10% | 40% | 70% | 90% |
| **Query Response Time** | 500ms | 300ms | 150ms | 80ms |
| **Relevance Accuracy** | 72% | 85% | 92% | 97% |
| **Extraction Accuracy** | 65% | 80% | 90% | 95% |
| **User Satisfaction** | N/A | 3.5/5 | 4.2/5 | 4.7/5 |

---

## 🎯 **Next Steps for Implementation**

### **Week 1-2: Enhanced Extraction**
1. Implement 6 new extraction methods in `KnowledgeExtractor`
2. Create 6 new SQLite database schemas
3. Download BC Building Code PDF + 20 construction handbooks
4. Run full re-extraction on 446 existing PDFs
5. Validate extraction accuracy (target: 85%+)

### **Week 3-4: RAG Enhancement**
1. Implement LLM-based semantic reranking
2. Add multi-query fusion
3. Optimize vector DB queries (add HNSW index)
4. Build RAG evaluation benchmark (50 test queries)
5. Achieve 85%+ relevance accuracy

### **Week 5-6: Construction Knowledge Graph**
1. Design Neo4j schema for construction domain
2. Migrate structured data → graph format
3. Build relationship extraction pipeline
4. Create graph query API for Supreme Control Hub
5. Test complex multi-hop queries

### **Week 7-8: Integration Testing**
1. Test full stack: Ingestion → RAG → Supreme Hub → Design Brain
2. Run 10 complete design projects end-to-end
3. Measure knowledge utilization rates
4. Validate professional deliverables accuracy
5. Collect user feedback from 5 beta testers

---

## 📚 **Resources & References**

**Current Implementation Files:**
- `modules/hybrid_learning_system.py` - Knowledge extraction + structured storage
- `modules/learning/rag_query.py` - RAG search with hybrid scoring
- `modules/agents/core/document_ingest.py` - Document ingestion agent
- `modules/ingest.py` - PDF parsing + chunking pipeline
- `modules/supreme_control_hub.py` - Intelligence orchestration
- `modules/learning/vectordb.py` - ChromaDB vector storage

**Training Data Sources:**
- academia.edu - Construction handbooks, structural engineering textbooks
- BC Building Code Part 9 - Residential building code (free download)
- RSMeans Cost Data 2024 - Construction cost estimating ($429)
- ASHRAE Standards - Energy efficiency guidelines
- Municipal bylaws - Local building regulations (BC cities)

**Technologies:**
- Vector DB: ChromaDB (BGE-large-en-v1.5 embeddings, 1024-dim)
- Structured DB: SQLite (4 knowledge databases)
- LLM: meta-llama/Llama-3.1-8B-Instruct (local on M4 Max)
- PDF Parsing: pdfplumber + Tesseract OCR
- Knowledge Graph: Neo4j (planned for Month 5)

---

**Document Version:** 1.0  
**Last Updated:** 2024-11-07  
**Author:** KALKI Development Team  
**Status:** Living document - will be updated as system evolves
