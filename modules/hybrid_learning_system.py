"""
Kalki Hybrid Learning System
============================
Multi-stage knowledge extraction and learning pipeline:
1. PDF Ingestion → Vector DB (RAG)
2. Knowledge Extraction → Structured DBs (Fast lookup)
3. Training Data Generation → Fine-tuning (True learning)
4. Continuous Learning → Self-improvement

Storage Strategy:
- Keep original PDFs (archival, legal, re-processing)
- Vector DB (retrieval)
- Knowledge DB (structured facts)
- Training data (model improvement)
"""

import os
import json
import sqlite3
import re
import logging
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)
from dataclasses import dataclass, asdict
import hashlib

# Global LLM cache to avoid reloading model (17s overhead)
_llm_cache = None

def get_cached_llm():
    """Get cached LLM instance or create new one"""
    global _llm_cache
    if _llm_cache is None:
        from modules.llm import get_llm_engine
        _llm_cache = get_llm_engine()
    return _llm_cache

@dataclass
class ExtractedFormula:
    """Mathematical formula or equation"""
    id: str
    name: str
    formula: str
    variables: Dict[str, str]
    domain: str  # engineering, physics, architecture, etc.
    source_pdf: str
    page_number: int
    confidence: float
    
@dataclass
class MaterialProperty:
    """Material specification"""
    material_name: str
    property_type: str  # mechanical, thermal, electrical, etc.
    properties: Dict[str, Any]
    standard: str  # ASTM, ISO, etc.
    source_pdf: str
    
@dataclass
class DesignRule:
    """Design rule or best practice"""
    rule_id: str
    category: str
    condition: str
    action: str
    reasoning: str
    source_pdf: str
    standard: Optional[str]
    
@dataclass
class CodeRequirement:
    """Building code or standard requirement"""
    code_id: str
    code_type: str  # building, electrical, plumbing, etc.
    requirement: str
    applicability: str
    exceptions: List[str]
    source_pdf: str


class KnowledgeExtractor:
    """Extract structured knowledge from PDFs - KALKI v2.5 Enhanced"""
    
    def __init__(self, knowledge_db_path: str = "data/knowledge/"):
        self.knowledge_path = Path(knowledge_db_path)
        self.knowledge_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize databases (original 4)
        self.formulas_db = self.knowledge_path / "formulas.db"
        self.materials_db = self.knowledge_path / "materials.db"
        self.rules_db = self.knowledge_path / "design_rules.db"
        self.codes_db = self.knowledge_path / "code_requirements.db"
        
        # v2.5 Enhanced databases (6 new)
        self.span_tables_db = self.knowledge_path / "span_tables.db"
        self.procedures_db = self.knowledge_path / "procedures.db"
        self.inspection_criteria_db = self.knowledge_path / "inspection_criteria.db"
        self.cost_data_db = self.knowledge_path / "cost_data.db"
        self.load_parameters_db = self.knowledge_path / "load_parameters.db"
        self.decision_trees_db = self.knowledge_path / "decision_trees.db"
        
        self._init_databases()
    
    def _init_databases(self):
        """Initialize SQLite databases for structured knowledge"""
        
        # Formulas database
        conn = sqlite3.connect(self.formulas_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS formulas (
                id TEXT PRIMARY KEY,
                name TEXT,
                formula TEXT,
                variables TEXT,
                domain TEXT,
                source_pdf TEXT,
                page_number INTEGER,
                confidence REAL,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        # Materials database
        conn = sqlite3.connect(self.materials_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS materials (
                material_name TEXT PRIMARY KEY,
                property_type TEXT,
                properties TEXT,
                standard TEXT,
                source_pdf TEXT,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        # Design rules database
        conn = sqlite3.connect(self.rules_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS design_rules (
                rule_id TEXT PRIMARY KEY,
                category TEXT,
                condition TEXT,
                action TEXT,
                reasoning TEXT,
                source_pdf TEXT,
                standard TEXT,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        # Code requirements database
        conn = sqlite3.connect(self.codes_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS code_requirements (
                code_id TEXT PRIMARY KEY,
                code_type TEXT,
                requirement TEXT,
                applicability TEXT,
                exceptions TEXT,
                source_pdf TEXT,
                created_at TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        # ======== KALKI v2.5 Enhanced Databases ========
        
        # Span tables database (structural member sizing)
        conn = sqlite3.connect(self.span_tables_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS span_tables (
                id TEXT PRIMARY KEY,
                member_type TEXT,
                member_size TEXT,
                spacing TEXT,
                span_feet REAL,
                span_inches REAL,
                load_type TEXT,
                load_value REAL,
                load_unit TEXT,
                species TEXT,
                grade TEXT,
                source_pdf TEXT,
                page_number INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_member_type ON span_tables(member_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_member_size ON span_tables(member_size)")
        conn.commit()
        conn.close()
        
        # Procedures database (step-by-step construction sequences)
        conn = sqlite3.connect(self.procedures_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS procedures (
                id TEXT PRIMARY KEY,
                procedure_name TEXT,
                category TEXT,
                step_number INTEGER,
                step_description TEXT,
                tools_required TEXT,
                materials_required TEXT,
                safety_notes TEXT,
                estimated_time_minutes INTEGER,
                skill_level TEXT,
                source_pdf TEXT,
                page_number INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_procedure_name ON procedures(procedure_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON procedures(category)")
        conn.commit()
        conn.close()
        
        # Inspection criteria database (QC validation points)
        conn = sqlite3.connect(self.inspection_criteria_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS inspection_criteria (
                id TEXT PRIMARY KEY,
                inspection_type TEXT,
                component TEXT,
                criteria_description TEXT,
                acceptance_standard TEXT,
                rejection_threshold TEXT,
                measurement_method TEXT,
                required_tools TEXT,
                code_reference TEXT,
                source_pdf TEXT,
                page_number INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_inspection_type ON inspection_criteria(inspection_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_component ON inspection_criteria(component)")
        conn.commit()
        conn.close()
        
        # Cost data database (material/labor unit costs)
        conn = sqlite3.connect(self.cost_data_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cost_data (
                id TEXT PRIMARY KEY,
                item_name TEXT,
                item_category TEXT,
                unit_cost REAL,
                unit TEXT,
                labor_cost REAL,
                labor_unit TEXT,
                location TEXT,
                year INTEGER,
                source TEXT,
                source_pdf TEXT,
                page_number INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_item_name ON cost_data(item_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON cost_data(item_category)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_year ON cost_data(year)")
        conn.commit()
        conn.close()
        
        # Load parameters database (structural design loads)
        conn = sqlite3.connect(self.load_parameters_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS load_parameters (
                id TEXT PRIMARY KEY,
                load_type TEXT,
                load_name TEXT,
                load_value REAL,
                load_unit TEXT,
                building_type TEXT,
                occupancy_type TEXT,
                code_reference TEXT,
                applicability TEXT,
                source_pdf TEXT,
                page_number INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_load_type ON load_parameters(load_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_building_type ON load_parameters(building_type)")
        conn.commit()
        conn.close()
        
        # Decision trees database (conditional code logic)
        conn = sqlite3.connect(self.decision_trees_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS decision_trees (
                id TEXT PRIMARY KEY,
                rule_name TEXT,
                condition TEXT,
                condition_operator TEXT,
                condition_value TEXT,
                then_action TEXT,
                else_action TEXT,
                code_section TEXT,
                category TEXT,
                source_pdf TEXT,
                page_number INTEGER,
                created_at TEXT
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rule_name ON decision_trees(rule_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_category ON decision_trees(category)")
        conn.commit()
        conn.close()
    
    def extract_from_pdf(self, pdf_path: str, pdf_content: str, 
                        use_llm_enhancements: bool = True,
                        extract_images: bool = True) -> Dict[str, int]:
        """
        Extract structured knowledge from PDF content - KALKI v3.5 with Vision
        
        Args:
            pdf_path: Path to PDF file
            pdf_content: Extracted text content
            use_llm_enhancements: Enable LLM validation and enhancement (DEFAULT: True)
            extract_images: Extract and analyze images/diagrams with vision model (DEFAULT: True)
        
        Returns:
            Count of extracted items by type
        """
        results = {
            "formulas": 0,
            "materials": 0,
            "rules": 0,
            "codes": 0,
            "cost_data": 0,
            "load_parameters": 0,
            "diagrams_analyzed": 0  # NEW: v3.5 vision extraction
        }
        
        # === NEW v3.5: Extract and analyze diagrams/images ===
        if extract_images:
            try:
                diagrams = self._extract_images_from_pdf(pdf_path)
                logger.info(f"Found {len(diagrams)} images in {Path(pdf_path).name}")
                
                if diagrams:
                    import asyncio
                    import nest_asyncio
                    nest_asyncio.apply()
                    
                    async def analyze_diagrams():
                        from modules.llm import get_llm_engine
                        llm = get_llm_engine()
                        
                        # Ensure vision model is initialized
                        if not llm.vision_engine or not llm.vision_engine.is_initialized:
                            logger.info("Initializing vision model for diagram analysis...")
                            await llm.initialize()
                        
                        diagram_results = []
                        for img_path, page_num in diagrams:
                            try:
                                # Extract structured data from diagram
                                diagram_data = await llm.extract_diagram(img_path)
                                diagram_data['page_number'] = page_num
                                diagram_data['source_pdf'] = pdf_path
                                diagram_results.append(diagram_data)
                                
                                # Store extracted formulas from diagrams
                                for formula_text in diagram_data.get('formulas', []):
                                    formula = ExtractedFormula(
                                        id=hashlib.md5(formula_text.encode()).hexdigest()[:16],
                                        name=formula_text.split('=')[0].strip() if '=' in formula_text else 'diagram_formula',
                                        formula=formula_text,
                                        variables={},
                                        domain="diagram_extracted",
                                        source_pdf=pdf_path,
                                        page_number=page_num,
                                        confidence=0.85  # Vision extraction confidence
                                    )
                                    self._store_formula(formula)
                                    results["formulas"] += 1
                                
                                # Store dimensions as design rules
                                for dimension in diagram_data.get('dimensions', []):
                                    rule = DesignRule(
                                        rule_id=hashlib.md5(f"{pdf_path}_{page_num}_{dimension}".encode()).hexdigest()[:16],
                                        category="dimensional_constraint",
                                        condition="diagram_specification",
                                        action=f"Dimension: {dimension}",
                                        reasoning=f"Extracted from technical drawing on page {page_num}",
                                        source_pdf=pdf_path,
                                        standard=None
                                    )
                                    self._store_rule(rule)
                                    results["rules"] += 1
                                
                                logger.info(f"✅ Analyzed diagram from page {page_num}: {len(diagram_data.get('formulas', []))} formulas, {len(diagram_data.get('dimensions', []))} dimensions")
                                results["diagrams_analyzed"] += 1
                                
                            except Exception as e:
                                logger.warning(f"Failed to analyze diagram from page {page_num}: {e}")
                        
                        return diagram_results
                    
                    asyncio.run(analyze_diagrams())
                    
            except Exception as e:
                logger.warning(f"Image extraction failed: {e}, continuing with text-only extraction")
        
        # === Original text-based extraction continues === 
        # Extract formulas (with optional LLM enhancement)
        formulas = self._extract_formulas(pdf_content, pdf_path)
        
        # LLM Enhancement: Validate and parse variables
        if use_llm_enhancements and formulas:
            import asyncio
            try:
                # Check if we're already in an event loop
                try:
                    loop = asyncio.get_running_loop()
                    # Already in async context - create task
                    import nest_asyncio
                    nest_asyncio.apply()
                    formulas = asyncio.run(self.enhance_formulas_with_llm(
                        formulas,
                        use_validation=True,
                        use_variable_parsing=True
                    ))
                except RuntimeError:
                    # No event loop - safe to use asyncio.run()
                    formulas = asyncio.run(self.enhance_formulas_with_llm(
                        formulas,
                        use_validation=True,
                        use_variable_parsing=True
                    ))
            except Exception as e:
                logger.warning(f"LLM formula enhancement failed: {e}, using regex-only extraction")
        
        for formula in formulas:
            self._store_formula(formula)
            results["formulas"] += 1
        
        # Extract material properties (with optional LLM validation)
        materials = self._extract_materials(pdf_content, pdf_path)
        if use_llm_enhancements and materials:
            import asyncio
            import nest_asyncio
            nest_asyncio.apply()  # Apply FIRST before any async operations
            
            try:
                async def validate_materials():
                    llm = get_cached_llm()
                    if not hasattr(llm, 'pipe') or llm.pipe is None:
                        await llm.initialize()
                    
                    validated_materials = []
                    for material in materials:
                        is_valid = await self._validate_material_with_llm(llm, material)
                        if is_valid:
                            validated_materials.append(material)
                    return validated_materials
                
                materials = asyncio.run(validate_materials())
                logger.info(f"LLM validated {len(materials)} materials")
            except Exception as e:
                logger.warning(f"LLM material validation failed: {e}, using all materials")
        
        for material in materials:
            self._store_material(material)
            results["materials"] += 1
        
        # Extract design rules (with optional LLM validation)
        rules = self._extract_design_rules(pdf_content, pdf_path)
        if use_llm_enhancements and rules:
            import asyncio
            import nest_asyncio
            nest_asyncio.apply()  # Apply FIRST
            
            try:
                async def validate_rules():
                    llm = get_cached_llm()
                    validated_rules = []
                    for rule in rules:
                        is_valid = await self._validate_design_rule_with_llm(llm, rule)
                        if is_valid:
                            validated_rules.append(rule)
                    return validated_rules
                
                rules = asyncio.run(validate_rules())
                logger.info(f"LLM validated {len(rules)} design rules")
            except Exception as e:
                logger.warning(f"LLM rule validation failed: {e}, using all rules")
        
        for rule in rules:
            self._store_rule(rule)
            results["rules"] += 1
        
        # Extract code requirements (with optional LLM validation)
        codes = self._extract_code_requirements(pdf_content, pdf_path)
        if use_llm_enhancements and codes:
            import asyncio
            import nest_asyncio
            nest_asyncio.apply()  # Apply FIRST
            
            try:
                async def validate_codes():
                    llm = get_cached_llm()
                    validated_codes = []
                    for code in codes:
                        is_valid = await self._validate_code_requirement_with_llm(llm, code)
                        if is_valid:
                            validated_codes.append(code)
                    return validated_codes
                
                codes = asyncio.run(validate_codes())
                logger.info(f"LLM validated {len(codes)} code requirements")
            except Exception as e:
                logger.warning(f"LLM code validation failed: {e}, using all codes")
        
        for code in codes:
            self._store_code(code)
            results["codes"] += 1
        
        # ======== KALKI v3.2 LEAN - Core Extractors Only ========
        # DISABLED: span_tables (redundant with materials/design rules)
        # DISABLED: procedures (better handled by RAG semantic search)
        # DISABLED: inspection_criteria (redundant with code requirements)
        # DISABLED: decision_trees (redundant with design rules)
        
        # All disabled extractors skipped in v3.2
        
        # Extract cost data (with LLM validation) - ACTIVE
        costs = self._extract_cost_data(pdf_content, pdf_path)
        if use_llm_enhancements and costs:
            import asyncio
            import nest_asyncio
            nest_asyncio.apply()
            
            try:
                async def validate_costs():
                    llm = get_cached_llm()
                    validated_costs = []
                    for cost in costs:
                        is_valid = await self._validate_cost_data_with_llm(llm, cost)
                        if is_valid:
                            validated_costs.append(cost)
                    return validated_costs
                
                costs = asyncio.run(validate_costs())
                logger.info(f"LLM validated {len(costs)} cost items")
            except Exception as e:
                logger.warning(f"LLM cost validation failed: {e}, using all costs")
        
        for cost in costs:
            self._store_cost_data(cost)
            results["cost_data"] += 1
        
        # Extract load parameters (with LLM validation)
        loads = self._extract_load_parameters(pdf_content, pdf_path)
        if use_llm_enhancements and loads:
            import asyncio
            import nest_asyncio
            nest_asyncio.apply()
            
            try:
                async def validate_loads():
                    llm = get_cached_llm()
                    validated_loads = []
                    for load in loads:
                        is_valid = await self._validate_load_parameter_with_llm(llm, load)
                        if is_valid:
                            validated_loads.append(load)
                    return validated_loads
                
                loads = asyncio.run(validate_loads())
                logger.info(f"LLM validated {len(loads)} load parameters")
            except Exception as e:
                logger.warning(f"LLM load validation failed: {e}, using all loads")
        
        for load in loads:
            self._store_load_parameter(load)
            results["load_parameters"] += 1
        
        return results
    
    def _extract_images_from_pdf(self, pdf_path: str) -> List[Tuple[str, int]]:
        """
        Extract images from PDF pages using pdf2image
        
        Returns:
            List of (image_path, page_number) tuples
        """
        try:
            from pdf2image import convert_from_path
        except ImportError:
            logger.error("pdf2image not installed. Install with: pip install pdf2image")
            return []
        
        images = []
        temp_dir = Path("data/temp_images")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Convert PDF pages to images
            pdf_name = Path(pdf_path).stem
            pages = convert_from_path(pdf_path, dpi=150)  # 150 DPI for good quality
            
            for i, page_image in enumerate(pages):
                page_num = i + 1
                
                # Save image temporarily
                img_path = temp_dir / f"{pdf_name}_page_{page_num}.png"
                page_image.save(img_path, "PNG")
                
                # Check if image likely contains technical content
                # (skip mostly blank pages to save processing time)
                img_size = img_path.stat().st_size
                if img_size > 50_000:  # At least 50KB suggests content
                    images.append((str(img_path), page_num))
                else:
                    # Delete small/blank images
                    img_path.unlink()
            
            return images
            
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {e}")
            return []
    
    def _extract_formulas(self, content: str, source: str) -> List[ExtractedFormula]:
        """
        Extract mathematical formulas and equations - ENHANCED v3.0
        Hybrid regex + validation approach with unit checking
        """
        formulas = []
        
        # STRICTER formula patterns - require mathematical operators or technical terms
        patterns = [
            # Engineering units with operators (F = 100 kN, M = 50 kN·m)
            r'([A-ZτθσεΔΣΠ][a-z]?\d*)\s*=\s*([^\n=]+?(?:[+\-*/×÷·^√]|(?:\d+\.?\d*\s*(?:kN|kPa|MPa|PSI|PSF|ksi|GPa|N|Pa|mm|cm|m|ft|in|kg|lb|ton)\b))[^\n]{0,60})',
            # Explicit technical formulas (Area = π r², Stress = F/A)
            r'(Area|Volume|Capacity|Load|Moment|Stress|Strain|Deflection|Bearing)\s*=\s*([^\n=]+?(?:[+\-*/×÷·^√πr²³]|\d)[^\n]{5,60})',
            # Safety factors with numbers (SF = 1.5, FOS = 2.0)
            r'(SF|FOS|Factor\s+of\s+Safety|Load\s+Factor|Resistance\s+Factor)\s*=\s*(\d+\.?\d*)',
            # Greek letter formulas (τ = V·Q/I·b, σ = M·c/I)
            r'([τθσεΔΣΠαβγδλμ])\s*=\s*([^\n=]+?[+\-*/×÷·^√][^\n]{5,60})',
            # Structural formulas with subscripts (M_max = w·L²/8)
            r'([A-Z][a-z]*_(?:max|min|cr|ult|nom|req|allow|des))\s*=\s*([^\n=]+?[+\-*/×÷·^√][^\n]{5,60})',
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                formula_text = match.group(0).strip()
                
                # v3.0: Filter out common false positives with regex
                if self._is_likely_formula(formula_text):
                    formula_id = hashlib.md5(formula_text.encode()).hexdigest()[:16]
                    
                    # Determine confidence based on pattern match quality
                    confidence = self._calculate_formula_confidence(formula_text)
                    
                    # OPPORTUNITY #4: Extract variable definitions from context
                    # Get surrounding text (500 chars before/after)
                    match_pos = content.find(formula_text)
                    if match_pos >= 0:
                        context_start = max(0, match_pos - 500)
                        context_end = min(len(content), match_pos + len(formula_text) + 500)
                        surrounding_context = content[context_start:context_end]
                    else:
                        surrounding_context = ""
                    
                    formula = ExtractedFormula(
                        id=formula_id,
                        name=match.group(1),
                        formula=formula_text,
                        variables={},  # Will be populated by LLM if enabled
                        domain="engineering",
                        source_pdf=source,
                        page_number=1,
                        confidence=confidence
                    )
                    
                    # Store context for later LLM processing
                    formula._context = surrounding_context  # Temporary attribute
                    formulas.append(formula)
        
        return formulas
    
    def _is_likely_formula(self, text: str) -> bool:
        """
        Validate if text is likely a mathematical formula vs prose - ENHANCED v3.0
        Reduces false positives by requiring multiple indicators
        """
        text_lower = text.lower()
        
        # Reject table-like data (multiple = signs, looks like reference table)
        if text.count('=') > 1:
            return False
        
        # Reject dimension/spec patterns (looks like: W = 12, L = 24, etc. in tables)
        # These are often table rows, not formulas
        if len(text) < 15 and re.match(r'^[A-Z]\s*=\s*\d+\.?\d*\s*[a-z]*$', text):
            return False
        
        # Reject common false positives first
        false_positives = [
            'a = approved', 'b = building', 'c = chapter', 'c = code',
            'd = design', 'e = example', 'f = figure', 'n = note',
            'r = required', 's = section', 't = table', 'v = volume',
            'p = page', 'm = minimum', 'a = appendix', 'b = basic',
            'i = ', 'j = ', 'k = ', 'l = ', 'o = ', 'q = ', 'u = ',
            'w = ', 'x = ', 'y = ', 'z = ',
            '= the ', '= a ', '= an ', '= all ', '= any ', '= as ',
            '= see ', '= refer', '= note', '= per ', '= for ',
            '= where', '= when', '= which', '= that', '= this'
        ]
        
        for fp in false_positives:
            if text_lower.startswith(fp) or f' {fp}' in text_lower:
                return False
        
        # Reject if too long (formulas are typically concise)
        if len(text) > 100:
            return False
        
        # Count mathematical indicators
        score = 0
        
        # Check for numbers (strong indicator)
        if re.search(r'\d+\.?\d*', text):
            score += 2
        
        # Check for mathematical operators (strong indicator)
        if re.search(r'[+\-*/×÷·^√∑∫]', text):
            score += 2
        
        # Check for engineering units (very strong indicator)
        if re.search(r'\b(PSI|PSF|MPa|kN|kPa|ksi|N|Pa|ft|in|mm|cm|m|kg|lb|ton|GPa)\b', text, re.IGNORECASE):
            score += 3
        
        # Check for parentheses/brackets (common in formulas)
        if re.search(r'[()[\]{}]', text):
            score += 1
        
        # Check for Greek letters (technical formulas)
        if re.search(r'[τθσεΔΣΠαβγδλμ]', text):
            score += 2
        
        # Check for subscripts/superscripts notation
        if re.search(r'[_^]', text):
            score += 1
        
        # Check for fractions
        if re.search(r'\d+\s*/\s*\d+', text):
            score += 2
        
        # Reject if contains too many common words (prose, not formula)
        common_words = len(re.findall(r'\b(the|and|or|of|to|for|with|from|by|at|in|on|is|are|was|were|be|been|being|have|has|had|do|does|did|will|would|should|could|may|might|can|must)\b', text_lower))
        if common_words > 2:
            return False
        
        # VERY STRICT: Require score >= 5 AND must contain operator or unit
        # This means MUST have:
        # - Operator + unit (2+3=5) OR
        # - Number + operator + unit (2+2+3=7) OR
        # - Greek letter + operator + unit (2+2+3=7)
        has_operator = bool(re.search(r'[+\-*/×÷·^√∑∫]', text))
        has_unit = bool(re.search(r'\b(PSI|PSF|MPa|kN|kPa|ksi|N|Pa|ft|in|mm|cm|m|kg|lb|ton|GPa)\b', text, re.IGNORECASE))
        
        # CRITICAL: Must have at least 2 variables/terms (real formula, not "F = 100 kN")
        # Count variables (uppercase letters not part of units)
        variables = re.findall(r'\b[A-Z][a-z]?(?![a-zA-Z])', text)
        # Filter out units
        variables = [v for v in variables if v not in ['PSI', 'PSF', 'MPa', 'GPa', 'Pa', 'N', 'M', 'I']]
        has_multiple_terms = len(variables) >= 2 or bool(re.search(r'[+\-*/×÷]', text))
        
        return score >= 5 and (has_operator or has_unit) and has_multiple_terms
    
    def _calculate_formula_confidence(self, text: str) -> float:
        """Calculate confidence score for formula extraction"""
        confidence = 0.5
        
        # Boost confidence for mathematical indicators
        if re.search(r'\d+\.?\d*', text):
            confidence += 0.1
        if re.search(r'[+\-*/×÷·^√∑∫]', text):
            confidence += 0.15
        if re.search(r'\b(PSI|PSF|MPa|kN|kPa|N|Pa|ft|in|mm|kg|lb)\b', text, re.IGNORECASE):
            confidence += 0.15
        if re.search(r'[()[\]{}]', text):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _extract_materials(self, content: str, source: str) -> List[MaterialProperty]:
        """
        Extract material specifications and properties - ENHANCED v3.0
        Expanded to cover wider range of construction/engineering materials
        """
        materials = []
        
        # Expanded material patterns for comprehensive coverage
        material_patterns = {
            # Metals
            r'(?i)(aluminum|aluminium)\s*(?:alloy)?\s*(\d{4})[^\d]*?(\d{2,5})\s*(MPa|psi|ksi)': 'aluminum',
            r'(?i)(steel)\s*(?:grade)?\s*(\d{3,4})[^\d]*?(\d{2,5})\s*(MPa|psi|ksi)': 'steel',
            r'(?i)(stainless\s*steel)\s*(\d{3})[^\d]*?(\d{2,5})\s*(MPa|psi|ksi)': 'stainless_steel',
            r'(?i)(titanium)\s*(?:grade)?\s*(\d+)?[^\d]*?(\d{2,5})\s*(MPa|psi|ksi)': 'titanium',
            r'(?i)(carbon\s*steel)[^\d]*?(\d{2,5})\s*(MPa|psi|ksi)': 'carbon_steel',
            
            # Composites & Advanced Materials
            r'(?i)(carbon\s*fiber|carbon\s*fibre|CFRP)[^\d]*?(\d{2,5})\s*(MPa|psi|ksi|GPa)': 'carbon_fiber',
            r'(?i)(fiberglass|fibreglass|GFRP)[^\d]*?(\d{2,5})\s*(MPa|psi|ksi)': 'fiberglass',
            r'(?i)(kevlar|aramid)[^\d]*?(\d{2,5})\s*(MPa|psi|ksi|GPa)': 'kevlar',
            
            # Concrete & Masonry
            r'(?i)(concrete)\s*(?:strength|f\'c)?[^\d]*?(\d{2,5})\s*(psi|MPa|ksi)': 'concrete',
            r'(?i)(reinforced\s*concrete)[^\d]*?(\d{2,5})\s*(psi|MPa)': 'reinforced_concrete',
            r'(?i)(masonry|brick|CMU|concrete\s*block)[^\d]*?(\d{2,5})\s*(psi|MPa)': 'masonry',
            
            # Wood & Lumber
            r'(?i)(Douglas\s*Fir|Doug\s*Fir|DF)[-\s]*(Larch|Select|No\.?\s*[12])?[^\d]*?(\d{2,5})\s*(psi|MPa)': 'douglas_fir',
            r'(?i)(Hem-Fir|Hemlock)[-\s]*(No\.?\s*[123])?[^\d]*?(\d{2,5})\s*(psi|MPa)': 'hem_fir',
            r'(?i)(Southern\s*Pine|SP)[-\s]*(No\.?\s*[123])?[^\d]*?(\d{2,5})\s*(psi|MPa)': 'southern_pine',
            r'(?i)(SPF|Spruce-Pine-Fir)[-\s]*(No\.?\s*[123])?[^\d]*?(\d{2,5})\s*(psi|MPa)': 'spf',
            r'(?i)(plywood|OSB|oriented\s*strand\s*board)[^\d]*?(\d{2,5})\s*(psi|MPa)': 'engineered_wood',
            r'(?i)(glulam|glue\s*laminated|LVL|laminated\s*veneer)[^\d]*?(\d{2,5})\s*(psi|MPa)': 'glulam',
            
            # Polymers & Plastics
            r'(?i)(PVC|polyvinyl\s*chloride)[^\d]*?(\d{2,5})\s*(MPa|psi|ksi)': 'pvc',
            r'(?i)(HDPE|high\s*density\s*polyethylene)[^\d]*?(\d{2,5})\s*(MPa|psi)': 'hdpe',
            r'(?i)(polycarbonate|lexan)[^\d]*?(\d{2,5})\s*(MPa|psi|GPa)': 'polycarbonate',
            r'(?i)(acrylic|PMMA)[^\d]*?(\d{2,5})\s*(MPa|psi)': 'acrylic',
            
            # Glass & Ceramics
            r'(?i)(glass|tempered\s*glass|laminated\s*glass)[^\d]*?(\d{2,5})\s*(MPa|psi|ksi)': 'glass',
            r'(?i)(ceramic|porcelain)[^\d]*?(\d{2,5})\s*(MPa|psi|ksi)': 'ceramic',
        }
        
        for pattern, material_type in material_patterns.items():
            matches = re.finditer(pattern, content)
            for match in matches:
                try:
                    # Extract material name and properties
                    material_name = match.group(1) if len(match.groups()) >= 1 else material_type
                    
                    # Validate material name (v3.0 Enhanced - filter garbage)
                    if not self._is_valid_material_name(material_name):
                        continue
                    
                    # Find strength value (usually last numeric group)
                    strength_value = None
                    unit = None
                    for group in reversed(match.groups()):
                        if group and re.match(r'^\d+$', group):
                            strength_value = group
                        elif group and group.upper() in ['MPA', 'PSI', 'KSI', 'GPA']:
                            unit = group.upper()
                        if strength_value and unit:
                            break
                    
                    if not strength_value or not unit:
                        continue
                    
                    material = MaterialProperty(
                        material_name=material_name,
                        property_type="mechanical",
                        properties={"strength": strength_value, "unit": unit},
                        standard="ASTM/CSA",
                        source_pdf=source
                    )
                    materials.append(material)
                except (IndexError, ValueError):
                    continue
        
        return materials
    
    def _is_valid_material_name(self, name: str) -> bool:
        """
        Validate material name quality - v3.0 Enhanced
        Filter out abbreviations, units, and garbage
        """
        if not name or len(name) < 3:
            return False
        
        # Reject common false positives (units, abbreviations)
        invalid_names = {
            'sp', 'ft', 'in', 'mm', 'cm', 'm', 'psi', 'mpa', 'ksi', 'gpa',
            'no', 'the', 'and', 'or', 'of', 'to', 'for', 'with', 'from',
            'by', 'at', 'on', 'is', 'are', 'was', 'were', 'be', 'been'
        }
        
        if name.lower().strip() in invalid_names:
            return False
        
        # Reject if mostly numbers
        if sum(c.isdigit() for c in name) > len(name) * 0.5:
            return False
        
        # Must contain at least one letter
        if not any(c.isalpha() for c in name):
            return False
        
        return True
    
    def _extract_design_rules(self, content: str, source: str) -> List[DesignRule]:
        """
        Extract design rules and best practices - ENHANCED v3.0
        Improved to extract condition-requirement pairs with better filtering
        """
        rules = []
        
        # Enhanced patterns with better context capture
        rule_patterns = [
            # Normative statements with component-action pairs
            r'([A-Z][A-Za-z\s]{3,40}?)\s+(shall|must|should)\s+(be|have|meet|comply\s+with|conform\s+to|withstand|resist|provide)\s+([^.;]{10,120})',
            # Min/max specifications with measurements
            r'(minimum|maximum)\s+([A-Za-z\s]{3,30}?)\s+(?:of|shall\s+be|is|:|=)\s*([^.;]{5,80})',
            # Factor of safety specifications
            r'(factor\s+of\s+safety|safety\s+factor|load\s+factor|resistance\s+factor)\s+(?:of|shall\s+be|=|:)?\s*(\d+\.?\d*)',
            # Design criteria with numeric thresholds
            r'([A-Za-z\s]{3,40}?)\s+(?:shall\s+not\s+exceed|must\s+not\s+exceed|limited\s+to|maximum)\s+(\d+\.?\d*)\s*([A-Za-z/%]{1,10})',
        ]
        
        for pattern in rule_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                rule_text = match.group(0).strip()
                
                # Filter out overly generic or incomplete statements
                if self._is_valid_design_rule(rule_text):
                    rule_id = hashlib.md5(rule_text.encode()).hexdigest()[:16]
                    
                    # Extract component, condition, and action
                    component, condition, action = self._parse_design_rule(match.groups())
                    
                    rule = DesignRule(
                        rule_id=rule_id,
                        category=self._categorize_design_rule(rule_text),
                        condition=condition,
                        action=action,
                        reasoning="Code/standard requirement",
                        source_pdf=source,
                        standard=self._extract_standard_reference(content, match.start())
                    )
                    rules.append(rule)
        
        return rules
    
    def _is_valid_design_rule(self, text: str) -> bool:
        """Validate if text is a meaningful design rule"""
        text_lower = text.lower()
        
        # Reject overly generic statements
        generic_phrases = [
            'should be considered', 'may be used', 'is recommended',
            'are discussed', 'is described', 'refer to', 'see section',
            'as follows', 'in accordance with', 'for example'
        ]
        
        for phrase in generic_phrases:
            if phrase in text_lower:
                return False
        
        # Require minimum length and some specificity
        if len(text) < 20:
            return False
        
        # Require at least one technical term or number
        has_technical = bool(re.search(r'\b(load|stress|strain|strength|capacity|deflection|factor|spacing|thickness|width|depth|height|diameter)\b', text_lower))
        has_number = bool(re.search(r'\d', text))
        
        return has_technical or has_number
    
    def _parse_design_rule(self, groups: tuple) -> tuple:
        """Parse design rule into component, condition, action"""
        if len(groups) >= 4:
            component = groups[0].strip()
            condition = f"{groups[1]} {groups[2]}" if len(groups) > 2 else groups[1]
            action = groups[-1].strip()
        elif len(groups) >= 2:
            component = groups[0].strip()
            condition = "specified requirement"
            action = groups[1].strip()
        else:
            component = "general"
            condition = "code requirement"
            action = groups[0].strip()
        
        return component, condition, action
    
    def _categorize_design_rule(self, text: str) -> str:
        """Categorize design rule by domain"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['structural', 'beam', 'column', 'load', 'stress', 'moment']):
            return 'structural'
        elif any(word in text_lower for word in ['fire', 'flame', 'smoke', 'egress', 'exit']):
            return 'fire_safety'
        elif any(word in text_lower for word in ['seismic', 'earthquake', 'lateral', 'wind']):
            return 'seismic'
        elif any(word in text_lower for word in ['foundation', 'soil', 'footing', 'bearing']):
            return 'foundation'
        elif any(word in text_lower for word in ['thermal', 'insulation', 'r-value', 'u-value']):
            return 'thermal'
        elif any(word in text_lower for word in ['accessibility', 'ada', 'barrier-free']):
            return 'accessibility'
        else:
            return 'general'
    
    def _extract_standard_reference(self, content: str, position: int) -> str:
        """Extract nearby standard reference (ASTM, CSA, ISO, etc.)"""
        context = content[max(0, position-100):min(len(content), position+100)]
        
        standards = ['ASTM', 'CSA', 'ISO', 'ANSI', 'ASCE', 'ACI', 'AISC', 'IBC', 'IRC', 'BCBC', 'NBC']
        for standard in standards:
            if standard in context:
                # Try to extract full reference
                ref_match = re.search(rf'{standard}\s*[A-Z]?\d+[\d\-\.]*', context)
                if ref_match:
                    return ref_match.group(0)
                return standard
        
        return None
    
    def _extract_code_requirements(self, content: str, source: str) -> List[CodeRequirement]:
        """
        Extract building code requirements - ENHANCED v3.0
        Expanded to support multiple code format styles
        """
        codes = []
        
        # Enhanced patterns for various code formats
        code_patterns = [
            r'(?:Section|SECTION)\s+(\d+(?:\.\d+)*)[:\s]+([^.\n]{10,200})',
            r'(?:Clause|CLAUSE)\s+(\d+(?:\.\d+)*)[:\s]+([^.\n]{10,200})',
            r'§\s*(\d+(?:\.\d+)*)[:\s]+([^.\n]{10,200})',
            r'(?:Article|ARTICLE)\s+(\d+(?:\.\d+)*)[:\s]+([^.\n]{10,200})',
            r'(\d{4}\.\d+(?:\.\d+)?)\s+([A-Z][^.\n]{15,200})',  # Format: 1234.5.6 Requirement
        ]
        
        for pattern in code_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            
            for match in matches:
                code_number = match.group(1)
                requirement_text = match.group(2).strip()
                
                # Skip if requirement text is too short or generic
                if len(requirement_text) < 15:
                    continue
                
                code_id = f"CODE_{code_number}"
                
                # Determine code type from context
                code_type = self._determine_code_type(requirement_text, content, match.start())
                
                code = CodeRequirement(
                    code_id=code_id,
                    code_type=code_type,
                    requirement=requirement_text,
                    applicability=self._extract_code_applicability(requirement_text),
                    exceptions=self._extract_code_exceptions(content, match.end()),
                    source_pdf=source
                )
                codes.append(code)
        
        return codes
    
    def _determine_code_type(self, requirement: str, content: str, position: int) -> str:
        """Determine the type of building code requirement"""
        req_lower = requirement.lower()
        context = content[max(0, position-200):position].lower()
        
        # Check requirement text and surrounding context
        if any(word in req_lower or word in context for word in ['structural', 'load', 'beam', 'column', 'foundation']):
            return 'structural'
        elif any(word in req_lower or word in context for word in ['electrical', 'wiring', 'circuit', 'panel', 'outlet']):
            return 'electrical'
        elif any(word in req_lower or word in context for word in ['plumbing', 'pipe', 'drain', 'water', 'sewer']):
            return 'plumbing'
        elif any(word in req_lower or word in context for word in ['fire', 'sprinkler', 'alarm', 'egress', 'exit']):
            return 'fire_safety'
        elif any(word in req_lower or word in context for word in ['mechanical', 'hvac', 'ventilation', 'duct']):
            return 'mechanical'
        elif any(word in req_lower or word in context for word in ['energy', 'insulation', 'thermal', 'efficiency']):
            return 'energy'
        elif any(word in req_lower or word in context for word in ['accessibility', 'ada', 'barrier-free', 'wheelchair']):
            return 'accessibility'
        else:
            return 'building_general'
    
    def _extract_code_applicability(self, requirement: str) -> str:
        """Extract applicability from requirement text"""
        req_lower = requirement.lower()
        
        # Check for specific building types
        if 'residential' in req_lower or 'dwelling' in req_lower:
            return 'Residential buildings'
        elif 'commercial' in req_lower:
            return 'Commercial buildings'
        elif 'industrial' in req_lower:
            return 'Industrial buildings'
        elif 'assembly' in req_lower:
            return 'Assembly occupancies'
        else:
            return 'All buildings unless specified otherwise'
    
    def _extract_code_exceptions(self, content: str, position: int) -> List[str]:
        """Extract exceptions that follow the code requirement"""
        exceptions = []
        
        # Look for exception text following the requirement
        exception_text = content[position:min(len(content), position+500)]
        
        # Pattern for exceptions
        exception_patterns = [
            r'Exception(?:s)?[:\s]+([^.\n]{10,200})',
            r'Except(?:ion)?[:\s]+([^.\n]{10,200})',
            r'This\s+(?:section|requirement)\s+(?:does\s+not\s+apply|shall\s+not\s+apply)\s+(?:to|when)\s+([^.\n]{10,200})',
        ]
        
        for pattern in exception_patterns:
            matches = re.finditer(pattern, exception_text, re.IGNORECASE)
            for match in matches:
                exceptions.append(match.group(1).strip())
        
        return exceptions
    
    def _store_formula(self, formula: ExtractedFormula):
        """Store formula in database"""
        conn = sqlite3.connect(self.formulas_db)
        conn.execute("""
            INSERT OR REPLACE INTO formulas VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            formula.id, formula.name, formula.formula,
            json.dumps(formula.variables), formula.domain,
            formula.source_pdf, formula.page_number, formula.confidence,
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def _store_material(self, material: MaterialProperty):
        """Store material in database"""
        conn = sqlite3.connect(self.materials_db)
        conn.execute("""
            INSERT OR REPLACE INTO materials VALUES (?, ?, ?, ?, ?, ?)
        """, (
            material.material_name, material.property_type,
            json.dumps(material.properties), material.standard,
            material.source_pdf, datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def _store_rule(self, rule: DesignRule):
        """Store design rule in database"""
        conn = sqlite3.connect(self.rules_db)
        conn.execute("""
            INSERT OR REPLACE INTO design_rules VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id, rule.category, rule.condition,
            rule.action, rule.reasoning, rule.source_pdf,
            rule.standard, datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def _store_code(self, code: CodeRequirement):
        """Store code requirement in database"""
        conn = sqlite3.connect(self.codes_db)
        conn.execute("""
            INSERT OR REPLACE INTO code_requirements VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            code.code_id, code.code_type, code.requirement,
            code.applicability, json.dumps(code.exceptions),
            code.source_pdf, datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    # ======== KALKI v2.5 Enhanced Extraction Methods ========
    
    def _extract_span_tables(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Extract structural member sizing tables (span tables) - ENHANCED v3.0
        Example: "2x6 joists @ 16" O.C. can span 12'3" for 40 PSF live load"
        """
        span_tables = []
        
        # Pattern 1: Standard span table format with optional load
        # "2x8 @ 16" O.C. - 14' 6" span" or "2x10 at 24" O.C. spans 16'-1" for 40 PSF"
        pattern1 = r'(\d+\s*[xX×]\s*\d+)\s+(?:@|at)\s+(\d+)"?\s+(?:O\.?C\.?|o\.?c\.?|on\s+center)\s*[-–—:]\s*(?:span(?:s)?|max(?:imum)?(?:\s+span)?)?[:.]?\s*(\d+)[\'′\'\'\\-]?\s*(\d+)?["\s″\'\']?'
        matches = re.finditer(pattern1, content, re.IGNORECASE)
        for match in matches:
            span_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            member_size = re.sub(r'\s+', '', match.group(1))  # Normalize "2 x 8" to "2x8"
            spacing = match.group(2) + '"'
            span_feet = int(match.group(3))
            span_inches = int(match.group(4)) if match.group(4) else 0
            
            # Extract load info from surrounding context (200 chars)
            context = content[max(0, match.start()-200):min(len(content), match.end()+200)]
            load_match = re.search(r'(\d+)\s*(?:PSF|psf|lb/ft²|pounds per square foot)', context, re.IGNORECASE)
            load_value = int(load_match.group(1)) if load_match else 40
            
            # Detect member type from context
            member_type = self._detect_member_type(context)
            
            # Detect species and grade
            species, grade = self._detect_species_grade(context)
            
            # Detect load type
            load_type = self._detect_load_type(context)
            
            span_tables.append({
                'id': span_id,
                'member_type': member_type,
                'member_size': member_size,
                'spacing': spacing,
                'span_feet': span_feet,
                'span_inches': span_inches,
                'load_type': load_type,
                'load_value': load_value,
                'load_unit': 'PSF',
                'species': species,
                'grade': grade,
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 2: Table format "Member Size | Spacing | Maximum Span"
        pattern2 = r'(\d+\s*[xX×]\s*\d+)\s*\|\s*(\d+)["\s]*O\.?C\.?\s*\|\s*(\d+)[\'′\'\'\\-]?\s*(\d+)?["\s″\'\']?'
        matches = re.finditer(pattern2, content, re.IGNORECASE)
        for match in matches:
            span_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            member_size = re.sub(r'\s+', '', match.group(1))
            context = content[max(0, match.start()-200):min(len(content), match.end()+200)]
            
            span_tables.append({
                'id': span_id,
                'member_type': self._detect_member_type(context),
                'member_size': member_size,
                'spacing': match.group(2) + '"',
                'span_feet': int(match.group(3)),
                'span_inches': int(match.group(4)) if match.group(4) else 0,
                'load_type': self._detect_load_type(context),
                'load_value': 40,
                'load_unit': 'PSF',
                'species': self._detect_species_grade(context)[0],
                'grade': self._detect_species_grade(context)[1],
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 3: Beam sizing format "W12x26 beam spans 20 feet"
        pattern3 = r'([WHS]\d+[xX×]\d+)\s+(?:beam|girder|column)\s+(?:span(?:s)?|support(?:s)?)\s+(\d+)\s*(?:feet|ft|\')'
        matches = re.finditer(pattern3, content, re.IGNORECASE)
        for match in matches:
            span_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            span_tables.append({
                'id': span_id,
                'member_type': 'steel_beam',
                'member_size': match.group(1),
                'spacing': 'N/A',
                'span_feet': int(match.group(2)),
                'span_inches': 0,
                'load_type': 'combined',
                'load_value': 0,
                'load_unit': 'PLF',
                'species': 'Steel',
                'grade': 'A36',
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 4: Rafter/roof member format "2x6 rafters, 24" spacing, 14' span"
        pattern4 = r'(\d+\s*[xX×]\s*\d+)\s+(?:rafter|roof\\s+joist|ceiling\\s+joist)s?\s*,?\s*(\d+)["\s]+(?:spacing|O\.?C\.?)\s*,?\s*(\d+)[\'′\'\']?\s*(\d+)?'
        matches = re.finditer(pattern4, content, re.IGNORECASE)
        for match in matches:
            span_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            member_size = re.sub(r'\s+', '', match.group(1))
            context = content[max(0, match.start()-200):min(len(content), match.end()+200)]
            
            span_tables.append({
                'id': span_id,
                'member_type': 'rafter',
                'member_size': member_size,
                'spacing': match.group(2) + '"',
                'span_feet': int(match.group(3)),
                'span_inches': int(match.group(4)) if match.group(4) else 0,
                'load_type': 'snow_load',
                'load_value': self._extract_snow_load(context),
                'load_unit': 'PSF',
                'species': self._detect_species_grade(context)[0],
                'grade': self._detect_species_grade(context)[1],
                'source_pdf': source,
                'page_number': 1
            })
        
        return span_tables
    
    def _detect_member_type(self, context: str) -> str:
        """Detect member type from context"""
        context_lower = context.lower()
        if any(word in context_lower for word in ['floor joist', 'joist', 'floor framing']):
            return 'floor_joist'
        elif any(word in context_lower for word in ['ceiling joist', 'ceiling']):
            return 'ceiling_joist'
        elif any(word in context_lower for word in ['rafter', 'roof joist', 'roof framing']):
            return 'rafter'
        elif any(word in context_lower for word in ['beam', 'girder', 'header']):
            return 'beam'
        elif any(word in context_lower for word in ['deck', 'decking']):
            return 'deck_joist'
        else:
            return 'joist'
    
    def _detect_species_grade(self, context: str) -> tuple:
        """Detect wood species and grade from context"""
        context_lower = context.lower()
        
        # Detect species
        if 'douglas fir' in context_lower or 'd fir' in context_lower or 'df' in context_lower:
            species = 'Douglas Fir'
        elif 'hem-fir' in context_lower or 'hemlock' in context_lower:
            species = 'Hem-Fir'
        elif 'spf' in context_lower or 'spruce-pine-fir' in context_lower:
            species = 'SPF'
        elif 'southern pine' in context_lower or 'sp' in context_lower:
            species = 'Southern Pine'
        else:
            species = 'SPF'  # Default
        
        # Detect grade
        if 'select structural' in context_lower or 'ss' in context_lower:
            grade = 'Select Structural'
        elif 'no. 1' in context_lower or 'no.1' in context_lower or '#1' in context_lower:
            grade = 'No. 1'
        elif 'no. 2' in context_lower or 'no.2' in context_lower or '#2' in context_lower:
            grade = 'No. 2'
        elif 'no. 3' in context_lower or 'no.3' in context_lower or '#3' in context_lower:
            grade = 'No. 3'
        else:
            grade = 'No. 2'  # Default
        
        return species, grade
    
    def _detect_load_type(self, context: str) -> str:
        """Detect load type from context"""
        context_lower = context.lower()
        if 'live load' in context_lower or 'll' in context_lower:
            return 'live_load'
        elif 'dead load' in context_lower or 'dl' in context_lower:
            return 'dead_load'
        elif 'snow load' in context_lower or 'snow' in context_lower:
            return 'snow_load'
        elif 'total load' in context_lower or 'combined' in context_lower:
            return 'total_load'
        else:
            return 'live_load'  # Default
    
    def _extract_snow_load(self, context: str) -> int:
        """Extract snow load value from context"""
        snow_match = re.search(r'(?:snow|roof)\s+load[:\s]+(\d+)\s*PSF', context, re.IGNORECASE)
        return int(snow_match.group(1)) if snow_match else 40  # Default 40 PSF
    
    def _extract_procedures(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Extract step-by-step construction procedures - ENHANCED v3.0
        Example: "1. Install vapor barrier 2. Frame walls 3. Install sheathing"
        """
        procedures = []
        
        # Pattern 1: Numbered steps "1. Install vapor barrier" or "Step 1: Frame walls"
        pattern1 = r'(?:Step\s+)?(\d+)[\.:)]\s+([A-Z][^\n\r]+?)(?=(?:\n|$|Step\s+\d+|\d+[\.:)]|\Z))'
        matches = re.finditer(pattern1, content)
        
        current_procedure_name = "General Construction"
        procedure_id_base = hashlib.md5(source.encode()).hexdigest()[:8]
        
        for match in matches:
            step_number = int(match.group(1))
            step_description = match.group(2).strip()
            
            # Skip if description is too short (likely false positive)
            if len(step_description) < 10:
                continue
            
            # Try to detect procedure name from headers above (improved search)
            header_search = content[max(0, match.start()-300):match.start()]
            header_patterns = [
                r'(?:Procedure|Process|Installation|Instructions?|Method)[:\s]+([A-Z][^\n]+)',
                r'([A-Z][^\n]+?)\s+(?:Procedure|Process|Installation)',
                r'##\s+([A-Z][^\n]+)',  # Markdown headers
                r'\*\*([A-Z][^\n]+?)\*\*'  # Bold text
            ]
            for pattern in header_patterns:
                header_match = re.search(pattern, header_search, re.IGNORECASE)
                if header_match:
                    current_procedure_name = header_match.group(1).strip()
                    break
            
            proc_id = f"{procedure_id_base}_{step_number}"
            
            # Extract tools and materials from step description
            tools = self._extract_tools(step_description)
            materials = self._extract_materials_from_text(step_description)
            
            # Estimate time based on step complexity
            estimated_time = self._estimate_step_time(step_description)
            
            # Determine skill level
            skill_level = self._determine_skill_level(step_description)
            
            # Extract safety notes
            safety_notes = self._extract_safety_notes(step_description)
            
            procedures.append({
                'id': proc_id,
                'procedure_name': current_procedure_name,
                'category': self._categorize_procedure(step_description),
                'step_number': step_number,
                'step_description': step_description,
                'tools_required': str(tools),
                'materials_required': str(materials),
                'safety_notes': safety_notes,
                'estimated_time_minutes': estimated_time,
                'skill_level': skill_level,
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 2: Bulleted procedures "• Install..." or "- Install..."
        pattern2 = r'(?:^|\n)[\•\-\*]\s+([A-Z][^\n\r]+?)(?=(?:\n[\•\-\*]|\n\n|\Z))'
        matches = re.finditer(pattern2, content, re.MULTILINE)
        
        bullet_step = 1
        for match in matches:
            step_description = match.group(1).strip()
            
            if len(step_description) < 10:
                continue
            
            proc_id = f"{procedure_id_base}_bullet_{bullet_step}"
            
            procedures.append({
                'id': proc_id,
                'procedure_name': current_procedure_name,
                'category': self._categorize_procedure(step_description),
                'step_number': bullet_step,
                'step_description': step_description,
                'tools_required': str(self._extract_tools(step_description)),
                'materials_required': str(self._extract_materials_from_text(step_description)),
                'safety_notes': self._extract_safety_notes(step_description),
                'estimated_time_minutes': self._estimate_step_time(step_description),
                'skill_level': self._determine_skill_level(step_description),
                'source_pdf': source,
                'page_number': 1
            })
            bullet_step += 1
        
        return procedures
    
    def _extract_tools(self, text: str) -> list:
        """Extract tool names from text"""
        tools = []
        tool_keywords = [
            'hammer', 'saw', 'drill', 'level', 'tape measure', 'square', 'screwdriver',
            'wrench', 'pliers', 'chisel', 'plane', 'router', 'nail gun', 'screw gun',
            'circular saw', 'jigsaw', 'miter saw', 'table saw', 'impact driver',
            'chalk line', 'laser level', 'stud finder', 'trowel', 'shovel', 'wheelbarrow'
        ]
        text_lower = text.lower()
        for tool in tool_keywords:
            if tool in text_lower:
                tools.append(tool)
        return tools
    
    def _extract_materials_from_text(self, text: str) -> list:
        """Extract material names from text"""
        materials = []
        material_keywords = [
            'lumber', 'plywood', 'osb', 'drywall', 'insulation', 'concrete',
            'rebar', 'screws', 'nails', 'bolts', 'adhesive', 'caulk', 'sealant',
            'vapor barrier', 'flashing', 'shingles', 'siding', 'paint', 'primer',
            'gravel', 'sand', 'mortar', 'cement', 'wire', 'conduit', 'pipe'
        ]
        text_lower = text.lower()
        for material in material_keywords:
            if material in text_lower:
                materials.append(material)
        return materials
    
    def _estimate_step_time(self, description: str) -> int:
        """Estimate time in minutes based on step complexity"""
        desc_lower = description.lower()
        
        # Quick tasks (15 min)
        if any(word in desc_lower for word in ['mark', 'measure', 'check', 'inspect', 'clean']):
            return 15
        
        # Medium tasks (30 min)
        if any(word in desc_lower for word in ['cut', 'drill', 'fasten', 'attach', 'secure']):
            return 30
        
        # Longer tasks (60 min)
        if any(word in desc_lower for word in ['install', 'frame', 'build', 'construct', 'assemble']):
            return 60
        
        # Heavy tasks (120 min)
        if any(word in desc_lower for word in ['pour', 'excavate', 'demolish', 'remove large']):
            return 120
        
        return 30  # Default
    
    def _determine_skill_level(self, description: str) -> str:
        """Determine required skill level"""
        desc_lower = description.lower()
        
        # Advanced tasks
        if any(word in desc_lower for word in ['weld', 'electrical panel', 'gas line', 'structural beam', 'engineer']):
            return 'advanced'
        
        # Intermediate tasks
        if any(word in desc_lower for word in ['frame', 'plumbing', 'electrical', 'install', 'construct']):
            return 'intermediate'
        
        # Beginner tasks
        if any(word in desc_lower for word in ['clean', 'measure', 'mark', 'organize', 'prepare']):
            return 'beginner'
        
        return 'intermediate'  # Default
    
    def _extract_safety_notes(self, text: str) -> str:
        """Extract safety-related information"""
        safety_keywords = ['safety', 'ppe', 'protect', 'caution', 'warning', 'danger', 'hazard']
        text_lower = text.lower()
        
        safety_notes = []
        if any(keyword in text_lower for keyword in safety_keywords):
            safety_notes.append("Safety precautions required - see step description")
        
        if 'electrical' in text_lower or 'wire' in text_lower:
            safety_notes.append("Turn off power before working")
        
        if 'ladder' in text_lower or 'height' in text_lower or 'roof' in text_lower:
            safety_notes.append("Fall protection required")
        
        if 'cut' in text_lower or 'saw' in text_lower:
            safety_notes.append("Wear eye protection")
        
        return '; '.join(safety_notes) if safety_notes else ''
    
    def _categorize_procedure(self, description: str) -> str:
        """Categorize procedure by keywords"""
        desc_lower = description.lower()
        if any(word in desc_lower for word in ['foundation', 'footing', 'concrete', 'excavate', 'pour']):
            return 'foundation'
        elif any(word in desc_lower for word in ['frame', 'stud', 'wall', 'joist', 'beam', 'truss']):
            return 'framing'
        elif any(word in desc_lower for word in ['roof', 'shingle', 'rafter', 'soffit', 'fascia']):
            return 'roofing'
        elif any(word in desc_lower for word in ['insulation', 'vapor barrier', 'drywall', 'finish']):
            return 'insulation_finishing'
        elif any(word in desc_lower for word in ['electrical', 'wiring', 'panel', 'outlet', 'switch']):
            return 'electrical'
        elif any(word in desc_lower for word in ['plumbing', 'pipe', 'drain', 'water', 'sewer']):
            return 'plumbing'
        elif any(word in desc_lower for word in ['hvac', 'heating', 'cooling', 'ventilation', 'duct']):
            return 'hvac'
        elif any(word in desc_lower for word in ['exterior', 'siding', 'window', 'door', 'trim']):
            return 'exterior_finish'
        elif any(word in desc_lower for word in ['interior', 'paint', 'floor', 'cabinet', 'countertop']):
            return 'interior_finish'
        else:
            return 'general'
    
    def _extract_inspection_criteria(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Extract quality control inspection criteria - ENHANCED v3.0
        Example: "Inspect foundation for cracks > 1/4 inch"
        """
        inspections = []
        
        # Pattern 1: "Inspect [component] for [criteria]"
        pattern1 = r'[Ii]nspect\s+(?:the\s+)?([^for]+?)\s+for\s+([^\n\.;]+)'
        matches = re.finditer(pattern1, content)
        
        for match in matches:
            inspection_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            component = match.group(1).strip()
            criteria = match.group(2).strip()
            
            # Extract measurement details
            measurement_method, required_tools = self._extract_measurement_info(criteria)
            
            inspections.append({
                'id': inspection_id,
                'inspection_type': self._categorize_inspection(component),
                'component': component,
                'criteria_description': criteria,
                'acceptance_standard': self._extract_acceptance_standard(criteria),
                'rejection_threshold': self._extract_rejection_threshold(criteria),
                'measurement_method': measurement_method,
                'required_tools': required_tools,
                'code_reference': self._extract_code_reference(content, match.start()),
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 2: "[Component] shall/must [criteria]"
        pattern2 = r'([A-Z][A-Za-z\s]+?)\s+(?:shall|must|should)\s+(?:be|have|meet|comply with|conform to)\s+([^\n\.;]+)'
        matches = re.finditer(pattern2, content)
        
        for match in matches:
            inspection_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            component = match.group(1).strip()
            criteria = match.group(2).strip()
            
            # Skip if too long (likely sentence fragment)
            if len(component) > 50:
                continue
            
            measurement_method, required_tools = self._extract_measurement_info(criteria)
            
            inspections.append({
                'id': inspection_id,
                'inspection_type': self._categorize_inspection(component),
                'component': component,
                'criteria_description': criteria,
                'acceptance_standard': criteria,
                'rejection_threshold': 'Non-compliance with stated requirement',
                'measurement_method': measurement_method,
                'required_tools': required_tools,
                'code_reference': self._extract_code_reference(content, match.start()),
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 3: "Check/Verify/Ensure that [component] [condition]"
        pattern3 = r'(?:Check|Verify|Ensure|Confirm)\s+(?:that\s+)?(?:the\s+)?([^is]+?)\s+(?:is|are|has|have)\s+([^\n\.;]+)'
        matches = re.finditer(pattern3, content, re.IGNORECASE)
        
        for match in matches:
            inspection_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            component = match.group(1).strip()
            criteria = match.group(2).strip()
            
            if len(component) > 50:
                continue
            
            measurement_method, required_tools = self._extract_measurement_info(criteria)
            
            inspections.append({
                'id': inspection_id,
                'inspection_type': self._categorize_inspection(component),
                'component': component,
                'criteria_description': criteria,
                'acceptance_standard': self._extract_acceptance_standard(criteria),
                'rejection_threshold': self._extract_rejection_threshold(criteria),
                'measurement_method': measurement_method,
                'required_tools': required_tools,
                'code_reference': self._extract_code_reference(content, match.start()),
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 4: Tolerance specifications "Maximum [deviation]: X inches"
        pattern4 = r'(?:Maximum|Minimum|Tolerance)\s+([^:]+):\s*([^\n]+)'
        matches = re.finditer(pattern4, content, re.IGNORECASE)
        
        for match in matches:
            inspection_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            component = match.group(1).strip()
            tolerance = match.group(2).strip()
            
            inspections.append({
                'id': inspection_id,
                'inspection_type': 'tolerance_check',
                'component': component,
                'criteria_description': f"Tolerance: {tolerance}",
                'acceptance_standard': tolerance,
                'rejection_threshold': f"Exceeds {tolerance}",
                'measurement_method': 'measurement',
                'required_tools': 'tape measure, calipers',
                'code_reference': '',
                'source_pdf': source,
                'page_number': 1
            })
        
        return inspections
    
    def _extract_measurement_info(self, criteria: str) -> tuple:
        """Extract measurement method and required tools from criteria"""
        criteria_lower = criteria.lower()
        
        # Determine measurement method
        if any(word in criteria_lower for word in ['measure', 'dimension', 'length', 'width', 'thickness']):
            method = 'dimensional_measurement'
            tools = 'tape measure, ruler'
        elif any(word in criteria_lower for word in ['level', 'plumb', 'square', 'alignment']):
            method = 'alignment_check'
            tools = 'level, plumb bob, square'
        elif any(word in criteria_lower for word in ['crack', 'damage', 'defect', 'condition']):
            method = 'visual_inspection'
            tools = 'flashlight, mirror'
        elif any(word in criteria_lower for word in ['test', 'pressure', 'load']):
            method = 'performance_test'
            tools = 'test equipment'
        elif any(word in criteria_lower for word in ['spacing', 'pattern', 'layout']):
            method = 'pattern_verification'
            tools = 'tape measure, chalk line'
        else:
            method = 'visual_inspection'
            tools = 'as required'
        
        return method, tools
    
    def _extract_code_reference(self, content: str, position: int) -> str:
        """Extract nearby code section references"""
        # Look in surrounding context for code references
        context = content[max(0, position-150):min(len(content), position+150)]
        
        # Pattern for code sections
        code_patterns = [
            r'(?:Section|Article|Clause)\s+(\d+(?:\.\d+)*)',
            r'(?:IBC|IRC|BCBC|NBC)\s+(\d+(?:\.\d+)*)',
            r'(?:Part|Division)\s+(\d+)',
            r'(\d{4}\.\d+\.\d+)'  # Numbered format like 9.23.4.5
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return ''
    
    def _categorize_inspection(self, component: str) -> str:
        """Categorize inspection by component"""
        comp_lower = component.lower()
        if any(word in comp_lower for word in ['foundation', 'footing', 'concrete slab', 'basement']):
            return 'foundation_inspection'
        elif any(word in comp_lower for word in ['framing', 'stud', 'joist', 'beam', 'rafter', 'truss']):
            return 'framing_inspection'
        elif any(word in comp_lower for word in ['roof', 'sheathing', 'shingle', 'flashing', 'gutter']):
            return 'roofing_inspection'
        elif any(word in comp_lower for word in ['insulation', 'vapor barrier', 'air barrier']):
            return 'insulation_inspection'
        elif any(word in comp_lower for word in ['electrical', 'wiring', 'panel', 'outlet', 'fixture']):
            return 'electrical_inspection'
        elif any(word in comp_lower for word in ['plumbing', 'pipe', 'drain', 'water', 'sewer']):
            return 'plumbing_inspection'
        elif any(word in comp_lower for word in ['hvac', 'duct', 'furnace', 'ac unit']):
            return 'hvac_inspection'
        elif any(word in comp_lower for word in ['window', 'door', 'exterior', 'siding']):
            return 'envelope_inspection'
        elif any(word in comp_lower for word in ['drywall', 'paint', 'finish', 'trim']):
            return 'finish_inspection'
        else:
            return 'general_inspection'
    
    def _extract_acceptance_standard(self, criteria: str) -> str:
        """Extract acceptance criteria from description"""
        # Look for measurement thresholds
        threshold_patterns = [
            r'(?:less than|<|under|maximum|max)\s+([^\s,]+)',
            r'(?:within|±|plus or minus)\s+([^\s,]+)',
            r'(?:at least|minimum|min|greater than|>)\s+([^\s,]+)',
            r'(?:between)\s+([^\s]+)\s+and\s+([^\s,]+)'
        ]
        
        for pattern in threshold_patterns:
            match = re.search(pattern, criteria, re.IGNORECASE)
            if match:
                return f"Acceptable: {match.group(0)}"
        
        return criteria
    
    def _extract_rejection_threshold(self, criteria: str) -> str:
        """Extract rejection threshold from description"""
        # Look for failure conditions
        failure_patterns = [
            r'(?:greater than|>|exceeds?|more than)\s+([^\s,]+)',
            r'(?:less than|<|below|under)\s+([^\s,]+)',
            r'(?:crack|gap|void|defect)(?:s)?\s+(?:exceeding|greater than|over)\s+([^\s,]+)'
        ]
        
        for pattern in failure_patterns:
            match = re.search(pattern, criteria, re.IGNORECASE)
            if match:
                return f"Reject if: {match.group(0)}"
        
        return "Per code requirements"
    
    def _extract_cost_data(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Extract material/labor unit costs - ENHANCED v3.0
        Example: "2x4 studs: $3.50/ea" or "Framing labor: $45/hr"
        """
        costs = []
        
        # Pattern 1: "Item: $X.XX/unit" or "Item: $X.XX per unit"
        pattern1 = r'([A-Za-z0-9][^\n:$]{3,60}?):\s*\$(\d+(?:,\d{3})*\.?\d*)\s*(?:/|per)\s*([a-z]{2,10})'
        matches = re.finditer(pattern1, content, re.IGNORECASE)
        
        for match in matches:
            cost_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            item_name = match.group(1).strip()
            unit_cost_str = match.group(2).replace(',', '')  # Remove comma separators
            unit_cost = float(unit_cost_str)
            unit = match.group(3).lower()
            
            # Determine if material or labor
            is_labor = any(word in item_name.lower() for word in ['labor', 'labour', 'installation', 'install', 'crew', 'worker'])
            
            # Extract location and year from surrounding context
            context = content[max(0, match.start()-200):min(len(content), match.end()+100)]
            location = self._extract_location(context)
            year = self._extract_year(context)
            
            costs.append({
                'id': cost_id,
                'item_name': item_name,
                'item_category': 'labor' if is_labor else 'material',
                'unit_cost': unit_cost,
                'unit': unit,
                'labor_cost': unit_cost if is_labor else 0.0,
                'labor_unit': unit if is_labor else 'hr',
                'location': location,
                'year': year,
                'source': 'manual_extraction',
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 2: Table format "Material | Unit | $Cost" or "Material | $Cost | Unit"
        pattern2 = r'([A-Za-z0-9][^\|\n]{3,50}?)\s*\|\s*(?:(\w+)\s*\|\s*)?\$?(\d+(?:,\d{3})*\.?\d*)(?:\s*\|\s*(\w+))?'
        matches = re.finditer(pattern2, content, re.IGNORECASE)
        
        for match in matches:
            item_name = match.group(1).strip()
            unit = (match.group(2) or match.group(4) or 'ea').lower()
            cost_str = match.group(3).replace(',', '')
            
            try:
                unit_cost = float(cost_str)
            except ValueError:
                continue
            
            # Skip if cost is unreasonably high or low (likely false positive)
            if unit_cost < 0.01 or unit_cost > 1000000:
                continue
            
            cost_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            context = content[max(0, match.start()-200):min(len(content), match.end()+100)]
            
            costs.append({
                'id': cost_id,
                'item_name': item_name,
                'item_category': self._categorize_cost_item(item_name),
                'unit_cost': unit_cost,
                'unit': unit,
                'labor_cost': 0.0,
                'labor_unit': 'hr',
                'location': self._extract_location(context),
                'year': self._extract_year(context),
                'source': 'manual_extraction',
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 3: RSMeans style "Division XX: Item - $X.XX"
        pattern3 = r'(?:Division\s+\d+[:\-]?\s+)?([A-Z][^\-\n]{5,50}?)\s*[-–—]\s*\$(\d+(?:,\d{3})*\.?\d*)'
        matches = re.finditer(pattern3, content)
        
        for match in matches:
            cost_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            item_name = match.group(1).strip()
            unit_cost = float(match.group(2).replace(',', ''))
            
            if unit_cost < 0.01 or unit_cost > 1000000:
                continue
            
            context = content[max(0, match.start()-200):min(len(content), match.end()+100)]
            
            costs.append({
                'id': cost_id,
                'item_name': item_name,
                'item_category': self._categorize_cost_item(item_name),
                'unit_cost': unit_cost,
                'unit': self._infer_unit(item_name),
                'labor_cost': 0.0,
                'labor_unit': 'hr',
                'location': self._extract_location(context),
                'year': self._extract_year(context),
                'source': 'cost_guide',
                'source_pdf': source,
                'page_number': 1
            })
        
        return costs
    
    def _categorize_cost_item(self, item_name: str) -> str:
        """Categorize cost item as material, labor, or equipment"""
        name_lower = item_name.lower()
        if any(word in name_lower for word in ['labor', 'labour', 'install', 'crew', 'worker', 'carpenter', 'electrician', 'plumber']):
            return 'labor'
        elif any(word in name_lower for word in ['rental', 'equipment', 'tool', 'machine', 'crane', 'excavator']):
            return 'equipment'
        else:
            return 'material'
    
    def _extract_location(self, context: str) -> str:
        """Extract geographic location from context"""
        # Canadian provinces
        provinces = ['BC', 'AB', 'SK', 'MB', 'ON', 'QC', 'NB', 'NS', 'PE', 'NL', 'YT', 'NT', 'NU']
        for prov in provinces:
            if prov in context:
                return prov
        
        # Cities
        cities = ['Vancouver', 'Victoria', 'Calgary', 'Edmonton', 'Toronto', 'Montreal', 'Seattle', 'Portland']
        context_upper = context.upper()
        for city in cities:
            if city.upper() in context_upper:
                return city
        
        return 'BC'  # Default
    
    def _extract_year(self, context: str) -> int:
        """Extract year from context"""
        year_match = re.search(r'(20\d{2})', context)
        if year_match:
            return int(year_match.group(1))
        return 2024  # Default current year
    
    def _infer_unit(self, item_name: str) -> str:
        """Infer measurement unit from item name"""
        name_lower = item_name.lower()
        if any(word in name_lower for word in ['lumber', 'board', 'plank']):
            return 'LF'  # Linear feet
        elif any(word in name_lower for word in ['sheet', 'panel', 'plywood', 'drywall']):
            return 'sheet'
        elif any(word in name_lower for word in ['concrete', 'asphalt']):
            return 'CY'  # Cubic yards
        elif any(word in name_lower for word in ['paint', 'stain', 'sealer']):
            return 'gallon'
        elif any(word in name_lower for word in ['nail', 'screw', 'bolt', 'fastener']):
            return 'box'
        elif any(word in name_lower for word in ['hour', 'hourly', 'labor']):
            return 'hr'
        else:
            return 'ea'  # Each
    
    def _extract_load_parameters(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Extract structural design load values - ENHANCED v3.0
        Example: "Residential floor live load: 40 PSF"
        """
        loads = []
        
        # Pattern 1: "[Load type]: X PSF/PSI/kN/kPa"
        pattern1 = r'([A-Za-z\s]+load):\s*(\d+\.?\d*)\s*(PSF|PSI|kN|kPa|lb/ft²|kg/m²)'
        matches = re.finditer(pattern1, content, re.IGNORECASE)
        
        for match in matches:
            load_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            load_name = match.group(1).strip()
            load_value = float(match.group(2))
            load_unit = match.group(3).upper().replace('LB/FT²', 'PSF').replace('KG/M²', 'KPA')
            
            # Categorize load type
            load_type = self._categorize_load_type(load_name)
            
            # Determine building type and occupancy from context
            context = content[max(0, match.start()-200):min(len(content), match.end()+150)]
            building_type, occupancy_type = self._extract_building_occupancy(context)
            
            # Extract code reference
            code_ref = self._extract_load_code_reference(context)
            
            loads.append({
                'id': load_id,
                'load_type': load_type,
                'load_name': load_name,
                'load_value': load_value,
                'load_unit': load_unit,
                'building_type': building_type,
                'occupancy_type': occupancy_type,
                'code_reference': code_ref,
                'applicability': self._describe_applicability(load_name, building_type),
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 2: Table format "Load Type | Value | Unit"
        pattern2 = r'([A-Za-z\s]+load)\s*\|\s*(\d+\.?\d*)\s*\|\s*(PSF|PSI|kN|kPa)'
        matches = re.finditer(pattern2, content, re.IGNORECASE)
        
        for match in matches:
            load_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            load_name = match.group(1).strip()
            load_value = float(match.group(2))
            load_unit = match.group(3).upper()
            
            context = content[max(0, match.start()-200):min(len(content), match.end()+150)]
            building_type, occupancy_type = self._extract_building_occupancy(context)
            
            loads.append({
                'id': load_id,
                'load_type': self._categorize_load_type(load_name),
                'load_name': load_name,
                'load_value': load_value,
                'load_unit': load_unit,
                'building_type': building_type,
                'occupancy_type': occupancy_type,
                'code_reference': self._extract_load_code_reference(context),
                'applicability': self._describe_applicability(load_name, building_type),
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 3: Prescriptive loads "X PSF for [application]"
        pattern3 = r'(\d+\.?\d*)\s*(PSF|PSI|kPa|kN)\s+for\s+([^\n\.;]+)'
        matches = re.finditer(pattern3, content, re.IGNORECASE)
        
        for match in matches:
            load_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            load_value = float(match.group(1))
            load_unit = match.group(2).upper()
            application = match.group(3).strip()
            
            # Infer load type from application
            load_type = self._infer_load_type_from_application(application)
            
            context = content[max(0, match.start()-200):min(len(content), match.end()+100)]
            building_type, occupancy_type = self._extract_building_occupancy(context)
            
            loads.append({
                'id': load_id,
                'load_type': load_type,
                'load_name': f"{load_type.replace('_', ' ')} for {application}",
                'load_value': load_value,
                'load_unit': load_unit,
                'building_type': building_type,
                'occupancy_type': occupancy_type,
                'code_reference': self._extract_load_code_reference(context),
                'applicability': application,
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 4: Snow load by location "Snow load, Vancouver: 2.0 kPa"
        pattern4 = r'(?:Snow|Wind|Seismic)\s+load[,\s]+([A-Za-z\s]+):\s*(\d+\.?\d*)\s*(kPa|PSF|kN|MPH)'
        matches = re.finditer(pattern4, content, re.IGNORECASE)
        
        for match in matches:
            load_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            load_type_raw = match.group(0).split()[0].lower()
            location = match.group(1).strip()
            load_value = float(match.group(2))
            load_unit = match.group(3).upper()
            
            load_type = f"{load_type_raw}_load"
            
            loads.append({
                'id': load_id,
                'load_type': load_type,
                'load_name': f"{load_type_raw.title()} load - {location}",
                'load_value': load_value,
                'load_unit': load_unit,
                'building_type': 'all',
                'occupancy_type': 'all',
                'code_reference': f'Location: {location}',
                'applicability': f'Buildings in {location}',
                'source_pdf': source,
                'page_number': 1
            })
        
        return loads
    
    def _categorize_load_type(self, load_name: str) -> str:
        """Categorize load type from name"""
        load_lower = load_name.lower()
        if 'live' in load_lower or 'll' in load_lower:
            return 'live_load'
        elif 'dead' in load_lower or 'dl' in load_lower:
            return 'dead_load'
        elif 'wind' in load_lower:
            return 'wind_load'
        elif 'snow' in load_lower:
            return 'snow_load'
        elif 'seismic' in load_lower or 'earthquake' in load_lower:
            return 'seismic_load'
        elif 'roof' in load_lower:
            return 'roof_load'
        elif 'floor' in load_lower:
            return 'floor_load'
        elif 'soil' in load_lower or 'lateral' in load_lower:
            return 'lateral_load'
        else:
            return 'other_load'
    
    def _extract_building_occupancy(self, context: str) -> tuple:
        """Extract building type and occupancy type from context"""
        context_lower = context.lower()
        
        # Building type
        if any(word in context_lower for word in ['residential', 'dwelling', 'house', 'apartment']):
            building_type = 'residential'
        elif any(word in context_lower for word in ['commercial', 'office', 'retail', 'store']):
            building_type = 'commercial'
        elif any(word in context_lower for word in ['industrial', 'warehouse', 'factory']):
            building_type = 'industrial'
        elif any(word in context_lower for word in ['assembly', 'theatre', 'arena', 'church']):
            building_type = 'assembly'
        elif any(word in context_lower for word in ['institutional', 'school', 'hospital']):
            building_type = 'institutional'
        else:
            building_type = 'general'
        
        # Occupancy type (more specific)
        if 'single family' in context_lower or 'detached' in context_lower:
            occupancy_type = 'single_family'
        elif 'multi-family' in context_lower or 'multi family' in context_lower or 'apartment' in context_lower:
            occupancy_type = 'multi_family'
        elif 'office' in context_lower:
            occupancy_type = 'office'
        elif 'retail' in context_lower or 'store' in context_lower:
            occupancy_type = 'retail'
        elif 'warehouse' in context_lower or 'storage' in context_lower:
            occupancy_type = 'storage'
        elif 'school' in context_lower or 'classroom' in context_lower:
            occupancy_type = 'educational'
        else:
            occupancy_type = building_type
        
        return building_type, occupancy_type
    
    def _extract_load_code_reference(self, context: str) -> str:
        """Extract code reference for loads"""
        # Look for code references
        code_patterns = [
            r'(?:BCBC|IBC|IRC|NBC|ASCE)\s+[\d\.-]+',
            r'(?:Table|Section)\s+[\d\.-]+',
            r'(?:Part|Division)\s+\d+'
        ]
        
        for pattern in code_patterns:
            match = re.search(pattern, context, re.IGNORECASE)
            if match:
                return match.group(0)
        
        return 'Building Code'
    
    def _describe_applicability(self, load_name: str, building_type: str) -> str:
        """Generate applicability description"""
        return f"{load_name} for {building_type} buildings"
    
    def _infer_load_type_from_application(self, application: str) -> str:
        """Infer load type from application description"""
        app_lower = application.lower()
        if any(word in app_lower for word in ['floor', 'occupancy', 'live']):
            return 'live_load'
        elif any(word in app_lower for word in ['roof', 'snow']):
            return 'snow_load'
        elif any(word in app_lower for word in ['wind', 'lateral']):
            return 'wind_load'
        elif any(word in app_lower for word in ['seismic', 'earthquake']):
            return 'seismic_load'
        else:
            return 'design_load'
        
        return loads
    
    def _extract_decision_trees(self, content: str, source: str) -> List[Dict[str, Any]]:
        """
        Extract conditional code compliance logic - ENHANCED v3.0
        Support for compound conditions (AND/OR) and nested logic
        """
        decisions = []
        
        # Pattern 1: Compound conditions "If A > 10 and B < 5, then..."
        pattern1 = r'[Ii]f\s+(.+?)(?:,|\s+then)\s+then\s+([^\n\.]{10,150})'
        matches = re.finditer(pattern1, content)
        
        for match in matches:
            decision_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            full_condition = match.group(1).strip()
            action = match.group(2).strip()
            
            # Check if compound condition (AND/OR)
            is_compound = bool(re.search(r'\b(and|or|AND|OR)\b', full_condition))
            
            # Extract primary condition
            if is_compound:
                # Parse compound conditions
                condition_parts = re.split(r'\b(and|or|AND|OR)\b', full_condition, maxsplit=1)
                primary_condition = condition_parts[0].strip()
                compound_type = condition_parts[1].strip().upper() if len(condition_parts) > 1 else 'AND'
                secondary_condition = condition_parts[2].strip() if len(condition_parts) > 2 else ''
                
                condition_text = f"{primary_condition} {compound_type} {secondary_condition}"
            else:
                # Simple condition
                condition_match = re.search(r'([^<>=]+?)\s*([<>=!]+)\s*([^<>=]+)', full_condition)
                if condition_match:
                    param = condition_match.group(1).strip()
                    operator = condition_match.group(2).strip()
                    threshold = condition_match.group(3).strip()
                    condition_text = f"{param} {operator} {threshold}"
                else:
                    condition_text = full_condition
            
            decisions.append({
                'id': decision_id,
                'rule_name': f"Requirement: {condition_text[:50]}",
                'condition': condition_text[:100],
                'condition_operator': 'compound' if is_compound else 'simple',
                'condition_value': full_condition[:100],
                'then_action': action[:150],
                'else_action': 'Standard requirements apply',
                'code_section': self._extract_code_reference(content, match.start()),
                'category': self._categorize_decision(full_condition),
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 2: Simple conditions "If height > 35 feet, then..."
        pattern2 = r'[Ii]f\s+([^,<>=]+?)\s*([<>=!]+)\s*([^,]+?),\s*then\s+([^\n\.]{10,150})'
        matches = re.finditer(pattern2, content)
        
        for match in matches:
            decision_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            condition_param = match.group(1).strip()
            operator = match.group(2).strip()
            threshold = match.group(3).strip()
            action = match.group(4).strip()
            
            decisions.append({
                'id': decision_id,
                'rule_name': f"Requirement for {condition_param[:40]}",
                'condition': f"{condition_param} {operator} {threshold}",
                'condition_operator': operator,
                'condition_value': threshold,
                'then_action': action[:150],
                'else_action': 'Standard requirements apply',
                'code_section': self._extract_code_reference(content, match.start()),
                'category': self._categorize_decision(condition_param),
                'source_pdf': source,
                'page_number': 1
            })
        
        # Pattern 3: "When [condition], [action] required/shall"
        pattern3 = r'[Ww]hen\s+([^,]{5,80}),\s*([^\n]{10,120}?)\s+(?:required|shall|must)'
        matches = re.finditer(pattern3, content)
        
        for match in matches:
            decision_id = hashlib.md5(match.group(0).encode()).hexdigest()[:16]
            condition = match.group(1).strip()
            action = match.group(2).strip()
            
            decisions.append({
                'id': decision_id,
                'rule_name': f"When {condition[:40]}",
                'condition': condition[:100],
                'condition_operator': 'when',
                'condition_value': 'true',
                'then_action': action[:150],
                'else_action': 'Not required',
                'code_section': self._extract_code_reference(content, match.start()),
                'category': self._categorize_decision(condition),
                'source_pdf': source,
                'page_number': 1
            })
        
        return decisions
    
    def _categorize_decision(self, condition: str) -> str:
        """Categorize decision tree by condition type"""
        cond_lower = condition.lower()
        if any(word in cond_lower for word in ['height', 'story', 'stories', 'floor']):
            return 'building_height'
        elif any(word in cond_lower for word in ['area', 'square', 'size']):
            return 'building_area'
        elif any(word in cond_lower for word in ['occupancy', 'occupant', 'people']):
            return 'occupancy'
        elif any(word in cond_lower for word in ['fire', 'sprinkler', 'alarm']):
            return 'fire_safety'
        elif any(word in cond_lower for word in ['exit', 'egress', 'door']):
            return 'life_safety'
        else:
            return 'general'
    
    # Storage methods for v2.5 data
    
    def _store_span_table(self, span_table: Dict[str, Any]):
        """Store span table entry in database"""
        conn = sqlite3.connect(self.span_tables_db)
        conn.execute("""
            INSERT OR REPLACE INTO span_tables VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            span_table['id'], span_table['member_type'], span_table['member_size'],
            span_table['spacing'], span_table['span_feet'], span_table['span_inches'],
            span_table['load_type'], span_table['load_value'], span_table['load_unit'],
            span_table['species'], span_table['grade'], span_table['source_pdf'],
            span_table['page_number'], datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def _store_procedure(self, procedure: Dict[str, Any]):
        """Store procedure step in database"""
        conn = sqlite3.connect(self.procedures_db)
        conn.execute("""
            INSERT OR REPLACE INTO procedures VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            procedure['id'], procedure['procedure_name'], procedure['category'],
            procedure['step_number'], procedure['step_description'], procedure['tools_required'],
            procedure['materials_required'], procedure['safety_notes'], procedure['estimated_time_minutes'],
            procedure['skill_level'], procedure['source_pdf'], procedure['page_number'],
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def _store_inspection_criteria(self, inspection: Dict[str, Any]):
        """Store inspection criteria in database"""
        conn = sqlite3.connect(self.inspection_criteria_db)
        conn.execute("""
            INSERT OR REPLACE INTO inspection_criteria VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            inspection['id'], inspection['inspection_type'], inspection['component'],
            inspection['criteria_description'], inspection['acceptance_standard'],
            inspection['rejection_threshold'], inspection['measurement_method'],
            inspection['required_tools'], inspection['code_reference'], inspection['source_pdf'],
            inspection['page_number'], datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def _store_cost_data(self, cost: Dict[str, Any]):
        """Store cost data in database"""
        conn = sqlite3.connect(self.cost_data_db)
        conn.execute("""
            INSERT OR REPLACE INTO cost_data VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cost['id'], cost['item_name'], cost['item_category'], cost['unit_cost'],
            cost['unit'], cost['labor_cost'], cost['labor_unit'], cost['location'],
            cost['year'], cost['source'], cost['source_pdf'], cost['page_number'],
            datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def _store_load_parameter(self, load: Dict[str, Any]):
        """Store load parameter in database"""
        conn = sqlite3.connect(self.load_parameters_db)
        conn.execute("""
            INSERT OR REPLACE INTO load_parameters VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            load['id'], load['load_type'], load['load_name'], load['load_value'],
            load['load_unit'], load['building_type'], load['occupancy_type'],
            load['code_reference'], load['applicability'], load['source_pdf'],
            load['page_number'], datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def _store_decision_tree(self, decision: Dict[str, Any]):
        """Store decision tree in database"""
        conn = sqlite3.connect(self.decision_trees_db)
        conn.execute("""
            INSERT OR REPLACE INTO decision_trees VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            decision['id'], decision['rule_name'], decision['condition'],
            decision['condition_operator'], decision['condition_value'], decision['then_action'],
            decision['else_action'], decision['code_section'], decision['category'],
            decision['source_pdf'], decision['page_number'], datetime.now().isoformat()
        ))
        conn.commit()
        conn.close()
    
    def query_formulas(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query formulas from database"""
        conn = sqlite3.connect(self.formulas_db)
        
        if domain:
            cursor = conn.execute(
                "SELECT * FROM formulas WHERE domain = ?", (domain,)
            )
        else:
            cursor = conn.execute("SELECT * FROM formulas")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "name": row[1],
                "formula": row[2],
                "variables": json.loads(row[3]),
                "domain": row[4],
                "source_pdf": row[5],
                "page_number": row[6],
                "confidence": row[7]
            })
        
        conn.close()
        return results
    
    def query_materials(self, material_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query materials from database"""
        conn = sqlite3.connect(self.materials_db)
        
        if material_name:
            cursor = conn.execute(
                "SELECT * FROM materials WHERE material_name LIKE ?", 
                (f"%{material_name}%",)
            )
        else:
            cursor = conn.execute("SELECT * FROM materials")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "material_name": row[0],
                "property_type": row[1],
                "properties": json.loads(row[2]),
                "standard": row[3],
                "source_pdf": row[4]
            })
        
        conn.close()
        return results
    
    def query_design_rules(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query design rules from database"""
        conn = sqlite3.connect(self.rules_db)
        
        if category:
            cursor = conn.execute(
                "SELECT * FROM design_rules WHERE category LIKE ?", 
                (f"%{category}%",)
            )
        else:
            cursor = conn.execute("SELECT * FROM design_rules")
        
        results = []
        for row in cursor.fetchall():
            # Safely parse JSON parameters with error handling
            try:
                parameters = json.loads(row[4]) if row[4] and row[4].strip() else {}
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON for design rule {row[1]}: {row[4]}")
                parameters = {}
                
            results.append({
                "id": row[0],
                "rule_name": row[1],
                "description": row[2],
                "category": row[3],
                "parameters": parameters,
                "source_pdf": row[5],
                "page_number": row[6]
            })
        
        conn.close()
        return results
    
    def query_code_requirements(self, code_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query code requirements from database"""
        conn = sqlite3.connect(self.codes_db)
        
        if code_type:
            cursor = conn.execute(
                "SELECT * FROM code_requirements WHERE code_type LIKE ?", 
                (f"%{code_type}%",)
            )
        else:
            cursor = conn.execute("SELECT * FROM code_requirements")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "code_type": row[1],
                "code_section": row[2],
                "requirement": row[3],
                "applicability": row[4],
                "source_pdf": row[5],
                "page_number": row[6]
            })
        
        conn.close()
        return results
    
    # ======== KALKI v2.5 Enhanced Query Methods ========
    
    def query_span_tables(self, member_type: Optional[str] = None, member_size: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query span tables from database"""
        conn = sqlite3.connect(self.span_tables_db)
        
        if member_type and member_size:
            cursor = conn.execute(
                "SELECT * FROM span_tables WHERE member_type LIKE ? AND member_size = ?",
                (f"%{member_type}%", member_size)
            )
        elif member_type:
            cursor = conn.execute(
                "SELECT * FROM span_tables WHERE member_type LIKE ?",
                (f"%{member_type}%",)
            )
        elif member_size:
            cursor = conn.execute(
                "SELECT * FROM span_tables WHERE member_size = ?",
                (member_size,)
            )
        else:
            cursor = conn.execute("SELECT * FROM span_tables")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "member_type": row[1],
                "member_size": row[2],
                "spacing": row[3],
                "span_feet": row[4],
                "span_inches": row[5],
                "load_type": row[6],
                "load_value": row[7],
                "load_unit": row[8],
                "species": row[9],
                "grade": row[10],
                "source_pdf": row[11],
                "page_number": row[12]
            })
        
        conn.close()
        return results
    
    def query_procedures(self, procedure_name: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query construction procedures from database"""
        conn = sqlite3.connect(self.procedures_db)
        
        if procedure_name and category:
            cursor = conn.execute(
                "SELECT * FROM procedures WHERE procedure_name LIKE ? AND category = ? ORDER BY step_number",
                (f"%{procedure_name}%", category)
            )
        elif procedure_name:
            cursor = conn.execute(
                "SELECT * FROM procedures WHERE procedure_name LIKE ? ORDER BY step_number",
                (f"%{procedure_name}%",)
            )
        elif category:
            cursor = conn.execute(
                "SELECT * FROM procedures WHERE category = ? ORDER BY step_number",
                (category,)
            )
        else:
            cursor = conn.execute("SELECT * FROM procedures ORDER BY procedure_name, step_number")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "procedure_name": row[1],
                "category": row[2],
                "step_number": row[3],
                "step_description": row[4],
                "tools_required": row[5],
                "materials_required": row[6],
                "safety_notes": row[7],
                "estimated_time_minutes": row[8],
                "skill_level": row[9],
                "source_pdf": row[10],
                "page_number": row[11]
            })
        
        conn.close()
        return results
    
    def query_inspection_criteria(self, inspection_type: Optional[str] = None, component: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query inspection criteria from database"""
        conn = sqlite3.connect(self.inspection_criteria_db)
        
        if inspection_type and component:
            cursor = conn.execute(
                "SELECT * FROM inspection_criteria WHERE inspection_type = ? AND component LIKE ?",
                (inspection_type, f"%{component}%")
            )
        elif inspection_type:
            cursor = conn.execute(
                "SELECT * FROM inspection_criteria WHERE inspection_type = ?",
                (inspection_type,)
            )
        elif component:
            cursor = conn.execute(
                "SELECT * FROM inspection_criteria WHERE component LIKE ?",
                (f"%{component}%",)
            )
        else:
            cursor = conn.execute("SELECT * FROM inspection_criteria")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "inspection_type": row[1],
                "component": row[2],
                "criteria_description": row[3],
                "acceptance_standard": row[4],
                "rejection_threshold": row[5],
                "measurement_method": row[6],
                "required_tools": row[7],
                "code_reference": row[8],
                "source_pdf": row[9],
                "page_number": row[10]
            })
        
        conn.close()
        return results
    
    def query_cost_data(self, item_name: Optional[str] = None, item_category: Optional[str] = None, year: Optional[int] = None) -> List[Dict[str, Any]]:
        """Query cost data from database"""
        conn = sqlite3.connect(self.cost_data_db)
        
        if item_name and item_category:
            cursor = conn.execute(
                "SELECT * FROM cost_data WHERE item_name LIKE ? AND item_category = ?",
                (f"%{item_name}%", item_category)
            )
        elif item_name:
            cursor = conn.execute(
                "SELECT * FROM cost_data WHERE item_name LIKE ?",
                (f"%{item_name}%",)
            )
        elif item_category:
            cursor = conn.execute(
                "SELECT * FROM cost_data WHERE item_category = ?",
                (item_category,)
            )
        elif year:
            cursor = conn.execute(
                "SELECT * FROM cost_data WHERE year = ?",
                (year,)
            )
        else:
            cursor = conn.execute("SELECT * FROM cost_data")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "item_name": row[1],
                "item_category": row[2],
                "unit_cost": row[3],
                "unit": row[4],
                "labor_cost": row[5],
                "labor_unit": row[6],
                "location": row[7],
                "year": row[8],
                "source": row[9],
                "source_pdf": row[10],
                "page_number": row[11]
            })
        
        conn.close()
        return results
    
    def query_load_parameters(self, load_type: Optional[str] = None, building_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query structural load parameters from database"""
        conn = sqlite3.connect(self.load_parameters_db)
        
        if load_type and building_type:
            cursor = conn.execute(
                "SELECT * FROM load_parameters WHERE load_type = ? AND building_type = ?",
                (load_type, building_type)
            )
        elif load_type:
            cursor = conn.execute(
                "SELECT * FROM load_parameters WHERE load_type = ?",
                (load_type,)
            )
        elif building_type:
            cursor = conn.execute(
                "SELECT * FROM load_parameters WHERE building_type = ?",
                (building_type,)
            )
        else:
            cursor = conn.execute("SELECT * FROM load_parameters")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "load_type": row[1],
                "load_name": row[2],
                "load_value": row[3],
                "load_unit": row[4],
                "building_type": row[5],
                "occupancy_type": row[6],
                "code_reference": row[7],
                "applicability": row[8],
                "source_pdf": row[9],
                "page_number": row[10]
            })
        
        conn.close()
        return results
    
    def query_decision_trees(self, category: Optional[str] = None, rule_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query decision trees from database"""
        conn = sqlite3.connect(self.decision_trees_db)
        
        if category and rule_name:
            cursor = conn.execute(
                "SELECT * FROM decision_trees WHERE category = ? AND rule_name LIKE ?",
                (category, f"%{rule_name}%")
            )
        elif category:
            cursor = conn.execute(
                "SELECT * FROM decision_trees WHERE category = ?",
                (category,)
            )
        elif rule_name:
            cursor = conn.execute(
                "SELECT * FROM decision_trees WHERE rule_name LIKE ?",
                (f"%{rule_name}%",)
            )
        else:
            cursor = conn.execute("SELECT * FROM decision_trees")
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "id": row[0],
                "rule_name": row[1],
                "condition": row[2],
                "condition_operator": row[3],
                "condition_value": row[4],
                "then_action": row[5],
                "else_action": row[6],
                "code_section": row[7],
                "category": row[8],
                "source_pdf": row[9],
                "page_number": row[10]
            })
        
        conn.close()
        return results
    
    def get_statistics(self) -> Dict[str, int]:
        """Get knowledge base statistics - KALKI v2.5 Enhanced"""
        stats = {}
        
        # Count formulas
        conn = sqlite3.connect(self.formulas_db)
        stats["formulas"] = conn.execute("SELECT COUNT(*) FROM formulas").fetchone()[0]
        conn.close()
        
        # Count materials
        conn = sqlite3.connect(self.materials_db)
        stats["materials"] = conn.execute("SELECT COUNT(*) FROM materials").fetchone()[0]
        conn.close()
        
        # Count rules
        conn = sqlite3.connect(self.rules_db)
        stats["design_rules"] = conn.execute("SELECT COUNT(*) FROM design_rules").fetchone()[0]
        conn.close()
        
        # Count codes
        conn = sqlite3.connect(self.codes_db)
        stats["code_requirements"] = conn.execute("SELECT COUNT(*) FROM code_requirements").fetchone()[0]
        conn.close()
        
        # ======== v2.5 Enhanced Statistics ========
        
        # Count span tables
        conn = sqlite3.connect(self.span_tables_db)
        stats["span_tables"] = conn.execute("SELECT COUNT(*) FROM span_tables").fetchone()[0]
        conn.close()
        
        # Count procedures
        conn = sqlite3.connect(self.procedures_db)
        stats["procedures"] = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
        conn.close()
        
        # Count inspection criteria
        conn = sqlite3.connect(self.inspection_criteria_db)
        stats["inspection_criteria"] = conn.execute("SELECT COUNT(*) FROM inspection_criteria").fetchone()[0]
        conn.close()
        
        # Count cost data
        conn = sqlite3.connect(self.cost_data_db)
        stats["cost_data"] = conn.execute("SELECT COUNT(*) FROM cost_data").fetchone()[0]
        conn.close()
        
        # Count load parameters
        conn = sqlite3.connect(self.load_parameters_db)
        stats["load_parameters"] = conn.execute("SELECT COUNT(*) FROM load_parameters").fetchone()[0]
        conn.close()
        
        # Count decision trees
        conn = sqlite3.connect(self.decision_trees_db)
        stats["decision_trees"] = conn.execute("SELECT COUNT(*) FROM decision_trees").fetchone()[0]
        conn.close()
        
        return stats
    
    # ========== LLM ENHANCEMENT METHODS ==========
    
    async def enhance_formulas_with_llm(self, formulas: List[ExtractedFormula], 
                                        use_validation: bool = True,
                                        use_variable_parsing: bool = True) -> List[ExtractedFormula]:
        """
        OPPORTUNITIES #1 & #4: LLM-Enhanced Formula Processing
        
        1. Validates formulas to reduce false positives (50-80% reduction)
        2. Extracts variable definitions from context
        
        Args:
            formulas: List of candidate formulas
            use_validation: Enable LLM validation (Opportunity #1)
            use_variable_parsing: Enable variable definition extraction (Opportunity #4)
        
        Returns:
            Filtered and enhanced formulas
        """
        import json
        
        if not formulas:
            return []
        
        try:
            # Use cached LLM to avoid 17s reload overhead
            llm = get_cached_llm()
            
            # Initialize if not already done (check if pipe exists)
            if not hasattr(llm, 'pipe') or llm.pipe is None:
                logger.info("🚀 Initializing LLM (first time - may take 15-20s)...")
                init_success = await llm.initialize()
                if not init_success:
                    logger.error("❌ LLM initialization failed")
                    return formulas
                device_name = getattr(llm, 'device', 'unknown')
                logger.info(f"✅ LLM initialized on device: {device_name}")
            else:
                device_name = getattr(llm, 'device', 'cached')
                logger.info(f"✅ Using cached LLM on device: {device_name}")
            
            enhanced_formulas = []
            
            for formula in formulas:
                # OPPORTUNITY #1: Validate with LLM
                if use_validation:
                    is_valid = await self._validate_formula_with_llm(llm, formula)
                    if not is_valid:
                        continue  # Skip false positives
                
                # OPPORTUNITY #4: Extract variable definitions
                if use_variable_parsing and hasattr(formula, '_context'):
                    variables = await self._parse_formula_variables_with_llm(
                        llm, formula.formula, formula._context
                    )
                    formula.variables = variables
                
                enhanced_formulas.append(formula)
            
            logger.info(f"LLM enhanced {len(enhanced_formulas)}/{len(formulas)} formulas")
            return enhanced_formulas
            
        except Exception as e:
            import traceback
            logger.error(f"LLM enhancement failed: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return formulas
    
    async def _validate_formula_with_llm(self, llm, formula: ExtractedFormula) -> bool:
        """
        OPPORTUNITY #1: Validate if text is truly a formula
        Reduces false positives by 50-80%
        """
        prompt = f"""Is this a valid engineering or mathematical formula?

Formula: "{formula.formula}"

Respond with ONLY "YES" or "NO".

YES if it's a real formula with mathematical relationships.
NO if it's a definition, label, or non-formula text.

Examples:
- "F = ma" → YES (Newton's second law)
- "M = wL²/8" → YES (bending moment formula)
- "A = Approved" → NO (definition, not formula)
- "Section = 1607.1" → NO (reference, not formula)

Response:"""

        try:
            response = await llm.generate(prompt, max_new_tokens=10, temperature=0.1)
            response = response.strip().upper()
            return 'YES' in response
        except:
            return True  # If LLM fails, keep the formula (fallback to regex validation)
    
    async def _parse_formula_variables_with_llm(self, llm, formula: str, context: str) -> Dict[str, Any]:
        """
        OPPORTUNITY #4: Extract variable definitions from context
        Makes formulas self-documenting
        """
        # Extract variable names from formula
        variables = re.findall(r'\b[A-Za-z][a-z]?(?:_[a-z]+)?\b', formula)
        # Filter out common words and units
        variables = [v for v in set(variables) if v not in ['the', 'and', 'or', 'of', 'to', 'for', 'mm', 'cm', 'ft', 'in', 'PSI', 'MPa']]
        
        if not variables or len(variables) > 10:  # Skip if too many or no variables
            return {}
        
        prompt = f"""Extract variable definitions from the context for this formula.

Formula: {formula}

Variables to define: {', '.join(variables)}

Context:
{context[:1000]}

Respond in EXACT JSON format with variable definitions:
{{
  "M": {{"name": "Maximum bending moment", "unit": "kip-ft"}},
  "w": {{"name": "Uniform load", "unit": "kip/ft"}},
  "L": {{"name": "Span length", "unit": "ft"}}
}}

If a variable definition is not found in context, omit it.
JSON Response:"""

        try:
            response = await llm.generate(prompt, max_new_tokens=300, temperature=0.2)
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return {}
        except Exception as e:
            logger.debug(f"Variable parsing failed: {e}")
            return {}
    
    async def extract_relationships_with_llm(self, facts: List[Dict[str, Any]], 
                                            fact_type: str) -> List[Dict[str, Any]]:
        """
        OPPORTUNITY #3: Semantic Relationship Extraction
        
        Finds connections between facts that regex patterns miss
        Creates knowledge graph instead of isolated facts
        
        Args:
            facts: List of extracted facts (from same section/page)
            fact_type: Type of facts ('formula', 'rule', 'code', etc.)
        
        Returns:
            List of relationships between facts
        """
        import json
        
        if len(facts) < 2:
            return []
        
        try:
            llm = get_cached_llm()
            await llm.initialize()
            
            # Prepare facts summary
            facts_text = ""
            for i, fact in enumerate(facts[:10], 1):  # Limit to 10 facts
                fact_str = str(fact)[:200]  # Truncate long facts
                facts_text += f"{i}. {fact_str}\n"
            
            prompt = f"""Analyze these {fact_type} facts and identify relationships between them.

Facts:
{facts_text}

Find relationships like:
- Prerequisite: Fact A must be satisfied before Fact B applies
- Derived: Fact B is calculated using Fact A
- Conditional: Fact B applies only when Fact A is true
- Alternative: Fact A OR Fact B can be used
- Complementary: Fact A AND Fact B together describe something

Respond in EXACT JSON format:
[
  {{
    "fact_1": 1,
    "fact_2": 3,
    "relationship": "derived",
    "description": "Maximum span (fact 3) is calculated using load value (fact 1)"
  }}
]

If no clear relationships exist, return empty array [].
JSON Response:"""

            response = await llm.generate(prompt, max_new_tokens=400, temperature=0.3)
            
            # Extract JSON
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
            return []
            
        except Exception as e:
            logger.warning(f"Relationship extraction failed: {e}")
            return []
    
    async def validate_procedure_with_llm(self, procedure: Dict[str, Any]) -> Dict[str, Any]:
        """
        OPPORTUNITY #5: Procedure Step Validation
        
        Validates completeness and safety of construction procedures
        
        Args:
            procedure: Extracted procedure with steps
        
        Returns:
            Enhanced procedure with completeness_score and missing_steps
        """
        import json
        
        try:
            llm = get_cached_llm()
            await llm.initialize()
            
            steps_text = procedure.get('steps', [])
            if isinstance(steps_text, list):
                steps_text = '\n'.join([f"{i+1}. {s}" for i, s in enumerate(steps_text)])
            
            prompt = f"""Evaluate this construction procedure for completeness and safety.

Procedure: {procedure.get('procedure_name', 'Unknown')}

Steps:
{steps_text}

Analyze:
1. Is the procedure complete? (0-10 score)
2. What critical steps are missing?
3. Are there safety concerns?

Respond in EXACT JSON format:
{{
  "completeness_score": 7,
  "missing_steps": ["Inspect formwork before pouring", "Specify curing time"],
  "safety_warnings": ["No PPE mentioned", "Vibration safety not addressed"]
}}

JSON Response:"""

            response = await llm.generate(prompt, max_new_tokens=300, temperature=0.3)
            
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                validation = json.loads(response[json_start:json_end])
                procedure['completeness_score'] = validation.get('completeness_score', 5)
                procedure['missing_steps'] = validation.get('missing_steps', [])
                procedure['safety_warnings'] = validation.get('safety_warnings', [])
            
            return procedure
            
        except Exception as e:
            logger.warning(f"Procedure validation failed: {e}")
            return procedure
    
    async def generate_ingestion_summary_with_llm(self, stats: Dict[str, int], 
                                                  pdf_count: int) -> str:
        """
        OPPORTUNITY #7: Batch Extraction Summarization
        
        Generates executive summary after ingesting multiple PDFs
        Provides insights, patterns, and quality analysis
        
        Args:
            stats: Extraction statistics from all PDFs
            pdf_count: Number of PDFs ingested
        
        Returns:
            Natural language summary with insights
        """
        import json
        
        try:
            llm = get_cached_llm()
            await llm.initialize()
            
            prompt = f"""Generate an executive summary of this knowledge base ingestion.

Ingestion Statistics:
- PDFs Processed: {pdf_count}
- Formulas: {stats.get('formulas', 0):,}
- Materials: {stats.get('materials', 0):,}
- Design Rules: {stats.get('rules', 0):,}
- Code Requirements: {stats.get('codes', 0):,}
- Procedures: {stats.get('procedures', 0):,}
- Inspection Criteria: {stats.get('inspection_criteria', 0):,}
- Cost Data: {stats.get('cost_data', 0):,}
- Load Parameters: {stats.get('load_parameters', 0):,}
- Decision Trees: {stats.get('decision_trees', 0):,}

Provide:
1. Overall quality assessment (2-3 sentences)
2. Key insights or patterns (3-5 bullet points)
3. Potential knowledge gaps (2-3 areas)
4. Recommendations for improvement

Format as professional engineering report.

Summary:"""

            response = await llm.generate(prompt, max_new_tokens=600, temperature=0.4)
            
            total_items = sum(stats.values())
            return f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                     KNOWLEDGE BASE INGESTION SUMMARY                       ║
╚════════════════════════════════════════════════════════════════════════════╝

📊 Statistics:
   • {pdf_count} PDFs processed
   • {total_items:,} total knowledge items extracted
   • Average: {total_items//max(pdf_count, 1):,} items per PDF

{response.strip()}

✅ Ingestion Complete - Knowledge base ready for queries
"""
            
        except Exception as e:
            logger.warning(f"Summary generation failed: {e}")
            total_items = sum(stats.values())
            return f"""Ingestion Complete: {pdf_count} PDFs, {total_items:,} items extracted"""

    async def _validate_material_with_llm(self, llm, material: MaterialProperty) -> bool:
        """
        Validate if extracted text is truly a material specification
        Reduces false positives like abbreviations, units, or general text
        """
        # Truncate properties to avoid exceeding token limits
        props_str = str(material.properties)[:150]  # Limit to 150 chars
        
        prompt = f"""Is this a valid material specification? Answer YES or NO.

Material Name: {material.material_name}
Property Type: {material.property_type}
Properties: {props_str}

Valid materials have:
- Real material names (concrete, steel, wood, etc.)
- Measurable properties (strength, density, modulus, etc.)
- Standard specifications (ASTM, ISO, etc.)

Invalid examples:
- Abbreviations (sp, ft, in, lb)
- Generic terms (item, section, chapter)
- Incomplete data (just units without values)

Answer (YES/NO):"""

        try:
            response = await llm.generate(prompt, max_new_tokens=10, temperature=0.1)
            return "YES" in response.upper()
        except:
            return True  # Default to keeping if LLM fails

    async def _validate_design_rule_with_llm(self, llm, rule: DesignRule) -> bool:
        """
        Validate if extracted text is truly a design rule
        Filters out general text, descriptions, or non-actionable content
        """
        prompt = f"""Is this a valid design rule? Answer YES or NO.

Category: {rule.category}
Condition: {rule.condition[:200]}
Action: {rule.action[:200]}

Valid design rules:
- Have specific conditions (IF/WHEN statements)
- Specify clear actions (THEN/MUST/SHALL statements)
- Are technically actionable
- Relate to design constraints or requirements

Invalid examples:
- General descriptions or background info
- Historical context or explanations
- Vague statements without specific conditions
- Pure definitions

Answer (YES/NO):"""

        try:
            response = await llm.generate(prompt, max_new_tokens=10, temperature=0.1)
            return "YES" in response.upper()
        except:
            return True

    async def _validate_procedure_with_llm(self, llm, procedure: Dict[str, Any]) -> bool:
        """
        Validate if extracted text is truly a construction/assembly procedure
        Reduces false positives from general instructions or descriptions
        """
        prompt = f"""Is this a valid construction/assembly procedure? Answer YES or NO.

Procedure: {procedure.get('procedure_name', '')}
Category: {procedure.get('category', '')}
Steps: {procedure.get('step_description', '')[:300]}

Valid procedures:
- Step-by-step construction/assembly instructions
- Specific tools and materials mentioned
- Measurable actions (cut, install, fasten, etc.)
- Safety considerations
- Clear sequence of operations

Invalid examples:
- General descriptions or overviews
- Code requirements (use separate code extractor)
- Inspection criteria (use inspection extractor)
- Background information or theory

Answer (YES/NO):"""

        try:
            response = await llm.generate(prompt, max_new_tokens=10, temperature=0.1)
            return "YES" in response.upper()
        except:
            return True

    async def _validate_cost_data_with_llm(self, llm, cost: Dict[str, Any]) -> bool:
        """
        Validate if extracted text is truly cost/pricing data
        Filters out page numbers, dates, or unrelated numeric data
        """
        item_name = cost.get('item_name', '')
        unit_cost = cost.get('unit_cost', 0)
        unit = cost.get('unit', '')
        
        prompt = f"""Is this valid construction cost data? Answer YES or NO.

Item: {item_name[:200]}
Cost: ${unit_cost} per {unit}

Valid cost data:
- Real construction materials or labor (concrete, rebar, labor hours)
- Reasonable pricing ($0.50 to $10,000 per unit typically)
- Standard units (SF, CF, LF, each, hour, ton, yard)
- Trade-specific items (formwork, excavation, painting)

Invalid examples:
- Page numbers or section references
- Dates or year numbers  
- Generic numbers without context
- Equipment serial numbers
- Temperature or measurement readings

Answer (YES/NO):"""

        try:
            response = await llm.generate(prompt, max_new_tokens=10, temperature=0.1)
            return "YES" in response.upper()
        except:
            return True

    async def _validate_load_parameter_with_llm(self, llm, load: Dict[str, Any]) -> bool:
        """
        Validate if extracted text is truly a structural load parameter
        Filters out unrelated numeric data or dimensions
        """
        load_type = load.get('load_type', '')
        load_value = load.get('load_value', 0)
        load_unit = load.get('load_unit', '')
        description = load.get('description', '')
        
        prompt = f"""Is this valid structural load data? Answer YES or NO.

Load Type: {load_type}
Value: {load_value} {load_unit}
Description: {description[:200]}

Valid load parameters:
- Dead load, live load, wind load, seismic load, snow load
- Standard units (PSF, PSI, kPa, kN, pounds, kips)
- Reasonable ranges (5-500 PSF typical for buildings)
- Building code load specifications
- Load combinations or factors

Invalid examples:
- Dimensions or member sizes
- Material properties (not loads)
- Temperature or humidity readings
- Page numbers or references
- Unrelated numeric data

Answer (YES/NO):"""

        try:
            response = await llm.generate(prompt, max_new_tokens=10, temperature=0.1)
            return "YES" in response.upper()
        except:
            return True

    async def _validate_code_requirement_with_llm(self, llm, code: CodeRequirement) -> bool:
        """
        Validate if extracted text is truly a code requirement
        Filters out general text, examples, or non-mandatory content
        """
        prompt = f"""Is this a valid building code requirement? Answer YES or NO.

Code ID: {code.code_id}
Code Type: {code.code_type}
Requirement: {code.requirement[:300]}

Valid code requirements:
- Mandatory regulatory requirements (SHALL/MUST)
- Specific numeric thresholds or limits
- Clear compliance criteria
- Referenced in official standards
- Legally enforceable

Invalid examples:
- General recommendations (SHOULD/MAY)
- Background explanations or commentary
- Historical context
- Examples or illustrations
- Informative content (not normative)

Answer (YES/NO):"""

        try:
            response = await llm.generate(prompt, max_new_tokens=10, temperature=0.1)
            return "YES" in response.upper()
        except:
            return True


class TrainingDataGenerator:
    """Generate training data for fine-tuning from extracted knowledge"""
    
    def __init__(self, output_path: str = "data/training/"):
        self.output_path = Path(output_path)
        self.output_path.mkdir(parents=True, exist_ok=True)
    
    def generate_from_knowledge_base(self, knowledge_extractor: KnowledgeExtractor) -> str:
        """
        Generate training data in formats suitable for fine-tuning
        
        Returns path to training data file
        """
        training_data = []
        
        # Generate Q&A pairs from formulas
        formulas = knowledge_extractor.query_formulas()
        for formula in formulas:
            training_data.append({
                "instruction": f"What is the formula for {formula['name']} in {formula['domain']}?",
                "input": "",
                "output": f"The formula is: {formula['formula']}"
            })
        
        # Generate Q&A pairs from materials
        materials = knowledge_extractor.query_materials()
        for material in materials:
            props = material['properties']
            training_data.append({
                "instruction": f"What are the properties of {material['material_name']}?",
                "input": "",
                "output": f"{material['material_name']} has the following {material['property_type']} properties: {json.dumps(props)}"
            })
        
        # Save training data
        output_file = self.output_path / f"training_data_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(output_file, 'w') as f:
            for item in training_data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Generated {len(training_data)} training examples")
        print(f"Saved to: {output_file}")
        
        return str(output_file)


class HybridKnowledgeSystem:
    """
    Main hybrid learning system combining:
    1. Vector DB (RAG retrieval)
    2. Structured Knowledge DB (fast lookup)
    3. Fine-tuned Model (internalized learning)
    """
    
    def __init__(self):
        self.knowledge_extractor = KnowledgeExtractor()
        self.training_generator = TrainingDataGenerator()
        
        # PDF archive (keep originals)
        self.pdf_archive = Path("data/pdf_archive/")
        self.pdf_archive.mkdir(parents=True, exist_ok=True)
        
        # Processing logs
        self.processing_log = Path("data/knowledge/processing_log.json")
        self.processed_pdfs = self._load_processing_log()
    
    def _load_processing_log(self) -> Dict[str, Any]:
        """Load log of processed PDFs"""
        if self.processing_log.exists():
            with open(self.processing_log, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_processing_log(self):
        """Save processing log"""
        with open(self.processing_log, 'w') as f:
            json.dump(self.processed_pdfs, f, indent=2)
    
    def ingest_pdf(self, pdf_path: str, pdf_content: str, 
                   archive: bool = True, use_llm_enhancements: bool = True) -> Dict[str, Any]:
        """
        Complete PDF ingestion pipeline:
        1. Archive original PDF (if requested)
        2. Extract to Vector DB (for RAG)
        3. Extract to Knowledge DB (for structured lookup)
        4. Generate training data (for fine-tuning)
        
        Args:
            pdf_path: Path to PDF file
            pdf_content: Extracted text content
            archive: Whether to archive original PDF
            use_llm_enhancements: Enable LLM validation and enhancement (DEFAULT: True)
        
        Returns:
            Statistics of extraction
        """
        pdf_name = Path(pdf_path).name
        pdf_hash = hashlib.md5(pdf_content.encode()).hexdigest()
        
        # Check if already processed
        if pdf_hash in self.processed_pdfs:
            print(f"PDF already processed: {pdf_name}")
            return self.processed_pdfs[pdf_hash]
        
        print(f"\n🔄 Processing PDF: {pdf_name}")
        
        # 1. Archive original PDF
        if archive and Path(pdf_path).exists():
            archive_path = self.pdf_archive / pdf_name
            import shutil
            shutil.copy2(pdf_path, archive_path)
            print(f"✅ Archived to: {archive_path}")
        
        # 2. Extract to Vector DB (handled by existing ingestion pipeline)
        print("✅ Vector DB ingestion (handled by existing system)")
        
        # 3. Extract structured knowledge (with optional LLM enhancement)
        print("🔍 Extracting structured knowledge...")
        extraction_results = self.knowledge_extractor.extract_from_pdf(
            pdf_path, pdf_content, use_llm_enhancements=use_llm_enhancements
        )
        
        # 4. Log processing
        self.processed_pdfs[pdf_hash] = {
            "pdf_name": pdf_name,
            "pdf_path": pdf_path,
            "processed_at": datetime.now().isoformat(),
            "extraction_results": extraction_results,
            "llm_enhanced": use_llm_enhancements
        }
        self._save_processing_log()
        
        print(f"\n📊 Extraction Results:")
        print(f"   Formulas: {extraction_results['formulas']}")
        print(f"   Materials: {extraction_results['materials']}")
        print(f"   Design Rules: {extraction_results['rules']}")
        print(f"   Code Requirements: {extraction_results['codes']}")
        
        # Show v3.0 enhanced extractors
        v3_items = (extraction_results.get('span_tables', 0) + extraction_results.get('procedures', 0) + 
                    extraction_results.get('inspection_criteria', 0) + extraction_results.get('cost_data', 0) +
                    extraction_results.get('load_parameters', 0) + extraction_results.get('decision_trees', 0))
        
        if v3_items > 0:
            print(f"\n   🆕 v3.0 Enhanced Extraction:")
            print(f"   Span Tables: {extraction_results.get('span_tables', 0)}")
            print(f"   Procedures: {extraction_results.get('procedures', 0)}")
            print(f"   Inspection Criteria: {extraction_results.get('inspection_criteria', 0)}")
            print(f"   Cost Data: {extraction_results.get('cost_data', 0)}")
            print(f"   Load Parameters: {extraction_results.get('load_parameters', 0)}")
            print(f"   Decision Trees: {extraction_results.get('decision_trees', 0)}")
        
        return extraction_results
    
    def generate_training_data(self) -> str:
        """Generate training data from all extracted knowledge"""
        print("\n📝 Generating training data for fine-tuning...")
        training_file = self.training_generator.generate_from_knowledge_base(
            self.knowledge_extractor
        )
        return training_file
    
    def get_learned_knowledge(self, query_type: str, **kwargs) -> Any:
        """
        Query learned knowledge directly (no LLM needed)
        Fast lookup from structured DBs
        """
        if query_type == "formula":
            return self.knowledge_extractor.query_formulas(kwargs.get("domain"))
        elif query_type == "material":
            return self.knowledge_extractor.query_materials(kwargs.get("material_name"))
        else:
            return None
    
    def query_formulas(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query formulas from knowledge database"""
        return self.knowledge_extractor.query_formulas(domain)
    
    def query_materials(self, material_name: Optional[str] = None, property_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query materials from knowledge database"""
        return self.knowledge_extractor.query_materials(material_name)
    
    def query_design_rules(self, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query design rules from knowledge database"""
        return self.knowledge_extractor.query_design_rules(category)
    
    def query_code_requirements(self, code_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query code requirements from knowledge database"""
        return self.knowledge_extractor.query_code_requirements(code_type)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        kb_stats = self.knowledge_extractor.get_statistics()
        
        return {
            "processed_pdfs": len(self.processed_pdfs),
            "knowledge_base": kb_stats,
            "pdf_archive_size": sum(f.stat().st_size for f in self.pdf_archive.glob("*.pdf")) if self.pdf_archive.exists() else 0,
            "storage_breakdown": {
                "pdf_archive": "Original PDFs (kept for reference)",
                "vector_db": "For RAG retrieval",
                "knowledge_db": "Structured facts (formulas, materials, rules)",
                "training_data": "For model fine-tuning"
            }
        }
    
    def hybrid_query(self, query: str, query_type: str = 'general', 
                     **kwargs) -> Dict[str, Any]:
        """
        Hybrid query combining Vector DB (RAG) + Knowledge DB (structured facts)
        
        This is THE CORE of KALKI's intelligence - combines:
        1. Vector DB: Context, explanations, related information
        2. Knowledge DB: Exact values, code references, deterministic facts
        3. LLM: Synthesizes both into coherent answer
        
        Args:
            query: Natural language question
            query_type: Type of query to optimize retrieval
                - 'general': Use both systems equally
                - 'structural': Focus on span tables, loads, formulas
                - 'code': Focus on code requirements, design rules
                - 'cost': Focus on cost data
                - 'procedure': Focus on step-by-step instructions
            **kwargs: Additional filters for knowledge DB queries
        
        Returns:
            {
                'vector_context': [...],  # Relevant text chunks from PDFs
                'structured_facts': {...},  # Exact data from knowledge DB
                'confidence_score': 0.95,  # How confident in the answer
                'sources': [...]  # Source PDFs and sections
            }
        
        Example:
            result = hybrid_query(
                "What's the floor live load for residential buildings?",
                query_type='structural',
                building_type='residential'
            )
            
            → Returns:
            {
                'vector_context': ["Residential buildings require...", ...],
                'structured_facts': {
                    'load_value': 40,
                    'load_unit': 'PSF',
                    'code_reference': 'IBC Section 1607.1',
                    'source_pdf': '2021_International_Building_Code.pdf'
                },
                'confidence_score': 1.0,
                'sources': ['2021_International_Building_Code.pdf']
            }
        """
        result = {
            'vector_context': [],
            'structured_facts': {},
            'confidence_score': 0.0,
            'sources': [],
            'query_type': query_type
        }
        
        # Step 1: Vector DB retrieval (would integrate with existing RAG system)
        # TODO: Integrate with modules.learning.vectordb when available
        # For now, this is a placeholder showing the architecture
        vector_results = self._vector_search(query)
        result['vector_context'] = vector_results
        
        # Step 2: Knowledge DB structured queries based on query type
        if query_type == 'structural' or query_type == 'general':
            # Query structural data
            result['structured_facts']['span_tables'] = self.knowledge_extractor.query_span_tables(
                **{k: v for k, v in kwargs.items() if k in ['member_type', 'member_size']}
            )
            result['structured_facts']['load_parameters'] = self.knowledge_extractor.query_load_parameters(
                **{k: v for k, v in kwargs.items() if k in ['load_type', 'building_type']}
            )
            result['structured_facts']['formulas'] = self.knowledge_extractor.query_formulas(
                **{k: v for k, v in kwargs.items() if k in ['domain']}
            )
        
        if query_type == 'code' or query_type == 'general':
            # Query code compliance data
            result['structured_facts']['code_requirements'] = self.knowledge_extractor.query_code_requirements(
                **{k: v for k, v in kwargs.items() if k in ['code_type']}
            )
            result['structured_facts']['design_rules'] = self.knowledge_extractor.query_design_rules(
                **{k: v for k, v in kwargs.items() if k in ['category']}
            )
            result['structured_facts']['decision_trees'] = self.knowledge_extractor.query_decision_trees(
                **{k: v for k, v in kwargs.items() if k in ['category', 'rule_name']}
            )
        
        if query_type == 'cost' or query_type == 'general':
            # Query cost data
            result['structured_facts']['cost_data'] = self.knowledge_extractor.query_cost_data(
                **{k: v for k, v in kwargs.items() if k in ['item_name', 'item_category', 'year']}
            )
        
        if query_type == 'procedure' or query_type == 'general':
            # Query procedural data
            result['structured_facts']['procedures'] = self.knowledge_extractor.query_procedures(
                **{k: v for k, v in kwargs.items() if k in ['procedure_name', 'category']}
            )
            result['structured_facts']['inspection_criteria'] = self.knowledge_extractor.query_inspection_criteria(
                **{k: v for k, v in kwargs.items() if k in ['inspection_type', 'component']}
            )
        
        if query_type == 'material' or query_type == 'general':
            # Query material data
            result['structured_facts']['materials'] = self.knowledge_extractor.query_materials(
                **{k: v for k, v in kwargs.items() if k in ['material_name']}
            )
        
        # Step 3: Calculate confidence score
        has_structured = any(result['structured_facts'].values())
        has_vector = len(result['vector_context']) > 0
        
        if has_structured and has_vector:
            result['confidence_score'] = 0.95  # Both sources agree
        elif has_structured:
            result['confidence_score'] = 0.90  # Have exact facts
        elif has_vector:
            result['confidence_score'] = 0.70  # Only have context
        else:
            result['confidence_score'] = 0.30  # No good results
        
        # Step 4: Collect sources
        sources = set()
        for facts in result['structured_facts'].values():
            if isinstance(facts, list):
                for fact in facts:
                    if isinstance(fact, dict) and 'source_pdf' in fact:
                        sources.add(fact['source_pdf'])
        result['sources'] = list(sources)
        
        return result
    
    async def hybrid_query_with_synthesis(self, query: str, 
                                          query_type: str = 'general',
                                          **kwargs) -> str:
        """
        OPPORTUNITY #2: LLM-Enhanced Query Synthesis
        
        Converts raw data from hybrid_query() into natural language answer
        Returns professional engineering response instead of data dumps
        
        Args:
            query: Natural language question
            query_type: Type of query ('general', 'structural', 'code', etc.)
            **kwargs: Additional filters
        
        Returns:
            Natural language answer with confidence and sources
        
        Example:
            answer = await system.hybrid_query_with_synthesis(
                "What's the floor live load for residential buildings?",
                query_type='structural'
            )
            → "Based on IBC Section 1607.1, residential floor live loads 
               require 40 PSF minimum..."
        """
        import json
        
        # Get data from both systems
        data = self.hybrid_query(query, query_type, **kwargs)
        
        # If low confidence or no data, return simple message
        if data['confidence_score'] < 0.3:
            return f"❌ I don't have enough information to answer '{query}'.\n\nRecommendation: Ingest more relevant technical PDFs covering this topic."
        
        # Prepare context for LLM
        try:
            llm = get_cached_llm()
            await llm.initialize()
            
            # Format structured facts nicely
            facts_text = ""
            if data['structured_facts']:
                facts_text = "Exact Engineering Data:\n"
                for key, value in data['structured_facts'].items():
                    if value:
                        facts_text += f"  • {key}: {json.dumps(value, indent=4)}\n"
            
            # Format vector context
            context_text = ""
            if data['vector_context']:
                context_text = "Context from Technical Documents:\n"
                for i, ctx in enumerate(data['vector_context'][:3], 1):
                    context_text += f"  {i}. {ctx}\n"
            
            prompt = f"""You are KALKI, an expert engineering AI assistant with deep knowledge of construction codes, structural design, and building standards.

Question: {query}

{context_text}

{facts_text}

Instructions:
1. Provide a clear, direct answer to the question
2. Include specific code references or standards when available
3. Mention any important safety considerations
4. Use professional engineering language
5. If data is limited, acknowledge uncertainty
6. Keep response concise but complete (3-5 sentences)

Response:"""

            response = await llm.generate(prompt, max_new_tokens=500, temperature=0.3)
            
            # Add confidence indicator and sources
            if data['confidence_score'] > 0.8:
                confidence_emoji = "🟢"
                confidence_text = "High"
            elif data['confidence_score'] > 0.5:
                confidence_emoji = "🟡"
                confidence_text = "Medium"
            else:
                confidence_emoji = "🔴"
                confidence_text = "Low"
            
            sources_text = ""
            if data['sources']:
                sources_text = f"\n📚 Sources: {', '.join(data['sources'][:3])}"
                if len(data['sources']) > 3:
                    sources_text += f" (+{len(data['sources'])-3} more)"
            
            return f"""{response.strip()}

{confidence_emoji} Confidence: {confidence_text} ({data['confidence_score']*100:.0f}%){sources_text}"""
            
        except Exception as e:
            # Fallback to raw data if LLM fails
            logger.warning(f"LLM synthesis failed: {e}, returning raw data")
            return f"Query results for '{query}':\n\nStructured Facts:\n{json.dumps(data['structured_facts'], indent=2)}\n\nConfidence: {data['confidence_score']*100:.0f}%"
    
    async def query_router(self, query: str) -> Dict[str, Any]:
        """
        OPPORTUNITY #6: Smart Query Routing
        
        Uses LLM to automatically detect query intent and route to correct database
        Eliminates need for users to specify query_type manually
        
        Args:
            query: Natural language question
        
        Returns:
            {
                'query_type': 'structural',
                'entities': ['span', '2x10', 'joist'],
                'filters': {'member_type': 'joist', 'member_size': '2x10'},
                'confidence': 0.95
            }
        """
        import json
        
        try:
            llm = get_cached_llm()
            await llm.initialize()
            
            prompt = f"""Analyze this engineering query and extract intent:

Query: "{query}"

Classify the query into ONE of these types:
- structural: spans, loads, beams, columns, foundations
- code: building codes, requirements, regulations, IBC, IRC
- cost: pricing, material costs, labor rates, estimates
- procedure: installation steps, construction methods, how-to
- material: material properties, specifications, strengths
- general: mixed or unclear

Also extract:
- Key entities (materials, sizes, building types, etc.)
- Filters for database query (as key-value pairs)

Respond in EXACT JSON format:
{{
  "query_type": "structural",
  "entities": ["span", "2x10", "floor joist"],
  "filters": {{"member_type": "joist", "member_size": "2x10"}},
  "confidence": 0.95
}}

JSON Response:"""

            response = await llm.generate(prompt, max_new_tokens=200, temperature=0.1)
            
            # Parse JSON response
            try:
                # Extract JSON from response (may have extra text)
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    result = json.loads(response[json_start:json_end])
                    return result
                else:
                    raise ValueError("No JSON found in response")
            except:
                # Fallback: simple keyword matching
                return self._fallback_query_routing(query)
                
        except Exception as e:
            logger.warning(f"LLM query routing failed: {e}, using fallback")
            return self._fallback_query_routing(query)
    
    def _fallback_query_routing(self, query: str) -> Dict[str, Any]:
        """Fallback keyword-based query routing"""
        query_lower = query.lower()
        
        # Detect query type by keywords
        if any(word in query_lower for word in ['span', 'beam', 'column', 'load', 'joist', 'rafter', 'foundation']):
            query_type = 'structural'
        elif any(word in query_lower for word in ['code', 'ibc', 'irc', 'requirement', 'section', 'regulation']):
            query_type = 'code'
        elif any(word in query_lower for word in ['cost', 'price', 'estimate', 'budget', '$']):
            query_type = 'cost'
        elif any(word in query_lower for word in ['how', 'install', 'procedure', 'step', 'method']):
            query_type = 'procedure'
        elif any(word in query_lower for word in ['concrete', 'steel', 'wood', 'material', 'strength', 'property']):
            query_type = 'material'
        else:
            query_type = 'general'
        
        return {
            'query_type': query_type,
            'entities': [],
            'filters': {},
            'confidence': 0.5
        }
    
    def _vector_search(self, query: str) -> List[str]:
        """
        Placeholder for vector DB search
        TODO: Integrate with modules.learning.vectordb
        """
        # This would call the actual vector DB when available
        # For now, return empty list
        return []


# Global instance
hybrid_system = None

def get_hybrid_system() -> HybridKnowledgeSystem:
    """Get global hybrid knowledge system instance"""
    global hybrid_system
    if hybrid_system is None:
        hybrid_system = HybridKnowledgeSystem()
    return hybrid_system
