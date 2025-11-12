# ============================================================
# Kalki v2.3 — technical_standards_ingestor.py
# ------------------------------------------------------------
# Technical Standards and Documentation Ingestor for RAG
# - Ingest mechanical engineering PDFs and ISO standards
# - Process technical drawings and specifications
# - Extract design guidelines and manufacturing standards
# - Integrate with Kalki's RAG vector database
# ============================================================

import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import json
import re
from datetime import datetime

from modules.utils.config import get_config
from modules.utils.logging_config import get_logger
from modules.ingest import DocumentIngestor
from modules.learning.vectordb import VectorDBManager

logger = get_logger("Kalki.TechnicalStandards")

class TechnicalStandardsIngestor:
    """Ingestor for technical standards, PDFs, and engineering documentation"""

    def __init__(self):
        self.ingestor = DocumentIngestor()
        self.vector_db = VectorDBManager()
        self.standards_dir = Path("data/technical_standards")
        self.standards_dir.mkdir(parents=True, exist_ok=True)

        # Standard document categories
        self.categories = {
            'iso_standards': 'ISO International Standards',
            'astm_standards': 'ASTM Standards',
            'ansi_standards': 'ANSI Standards',
            'din_standards': 'DIN Standards',
            'engineering_handbooks': 'Engineering Handbooks',
            'material_specifications': 'Material Specifications',
            'design_guidelines': 'Design Guidelines',
            'manufacturing_standards': 'Manufacturing Standards',
            'safety_standards': 'Safety Standards',
            'tolerance_standards': 'Tolerance and Precision Standards'
        }

    async def initialize(self) -> bool:
        """Initialize the technical standards ingestor"""
        try:
            # VectorDB is already initialized in constructor
            logger.info("Technical standards ingestor initialized successfully")
            return True

        except Exception as e:
            logger.exception(f"Error initializing technical standards ingestor: {e}")
            return False

    async def ingest_iso_standards(self, standards_path: str,
                                 standard_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ingest ISO standards from PDF files

        Args:
            standards_path: Path to directory containing ISO standard PDFs
            standard_ids: Specific ISO standard IDs to ingest (optional)

        Returns:
            Dict with ingestion results
        """
        standards_dir = Path(standards_path)
        if not standards_dir.exists():
            return {
                "status": "error",
                "error": f"ISO standards directory not found: {standards_path}"
            }

        # Find ISO PDF files
        iso_files = []
        for pdf_file in standards_dir.glob("**/*.pdf"):
            filename = pdf_file.name.lower()
            if 'iso' in filename or re.search(r'iso\s*\d+', filename):
                iso_files.append(pdf_file)

        if not iso_files:
            return {
                "status": "error",
                "error": "No ISO standard PDF files found"
            }

        # Filter by specific standard IDs if provided
        if standard_ids:
            filtered_files = []
            for pdf_file in iso_files:
                filename = pdf_file.name
                for std_id in standard_ids:
                    if std_id.lower() in filename.lower():
                        filtered_files.append(pdf_file)
                        break
            iso_files = filtered_files

        logger.info(f"Found {len(iso_files)} ISO standard files to ingest")

        # Ingest each standard
        results = []
        for pdf_file in iso_files:
            result = await self._ingest_technical_document(
                str(pdf_file),
                'iso_standards',
                self._extract_iso_metadata(str(pdf_file))
            )
            results.append(result)

        # Summarize results
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]

        return {
            "status": "completed",
            "category": "iso_standards",
            "total_files": len(results),
            "successful_ingestions": len(successful),
            "failed_ingestions": len(failed),
            "results": results
        }

    async def ingest_engineering_handbooks(self, handbooks_path: str,
                                         subjects: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ingest engineering handbook PDFs

        Args:
            handbooks_path: Path to directory containing handbook PDFs
            subjects: Specific subjects to focus on (optional)

        Returns:
            Dict with ingestion results
        """
        handbooks_dir = Path(handbooks_path)
        if not handbooks_dir.exists():
            return {
                "status": "error",
                "error": f"Handbooks directory not found: {handbooks_path}"
            }

        # Find handbook PDF files
        handbook_files = list(handbooks_dir.glob("**/*.pdf"))

        if not handbook_files:
            return {
                "status": "error",
                "error": "No handbook PDF files found"
            }

        logger.info(f"Found {len(handbook_files)} handbook files to ingest")

        # Ingest each handbook
        results = []
        for pdf_file in handbook_files:
            metadata = self._extract_handbook_metadata(str(pdf_file))
            if subjects:
                # Check if handbook covers requested subjects
                handbook_subjects = metadata.get('subjects', [])
                if any(subj.lower() in ' '.join(handbook_subjects).lower() for subj in subjects):
                    result = await self._ingest_technical_document(
                        str(pdf_file), 'engineering_handbooks', metadata
                    )
                    results.append(result)
            else:
                result = await self._ingest_technical_document(
                    str(pdf_file), 'engineering_handbooks', metadata
                )
                results.append(result)

        # Summarize results
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]

        return {
            "status": "completed",
            "category": "engineering_handbooks",
            "total_files": len(results),
            "successful_ingestions": len(successful),
            "failed_ingestions": len(failed),
            "results": results
        }

    async def ingest_material_specifications(self, specs_path: str,
                                          material_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Ingest material specification documents

        Args:
            specs_path: Path to material specifications
            material_types: Specific material types to focus on

        Returns:
            Dict with ingestion results
        """
        specs_dir = Path(specs_path)
        if not specs_dir.exists():
            return {
                "status": "error",
                "error": f"Material specifications directory not found: {specs_path}"
            }

        # Find specification files (PDF, DOC, etc.)
        spec_files = []
        for ext in ['*.pdf', '*.doc', '*.docx', '*.txt']:
            spec_files.extend(list(specs_dir.glob(f"**/{ext}")))

        if not spec_files:
            return {
                "status": "error",
                "error": "No material specification files found"
            }

        logger.info(f"Found {len(spec_files)} material specification files to ingest")

        # Ingest each specification
        results = []
        for spec_file in spec_files:
            metadata = self._extract_material_metadata(str(spec_file))
            if material_types:
                # Check if spec covers requested material types
                spec_materials = metadata.get('materials', [])
                if any(mat.lower() in ' '.join(spec_materials).lower() for mat in material_types):
                    result = await self._ingest_technical_document(
                        str(spec_file), 'material_specifications', metadata
                    )
                    results.append(result)
            else:
                result = await self._ingest_technical_document(
                    str(spec_file), 'material_specifications', metadata
                )
                results.append(result)

        # Summarize results
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]

        return {
            "status": "completed",
            "category": "material_specifications",
            "total_files": len(results),
            "successful_ingestions": len(successful),
            "failed_ingestions": len(failed),
            "results": results
        }

    async def _ingest_technical_document(self, file_path: str, category: str,
                                       metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ingest a single technical document into the RAG system

        Args:
            file_path: Path to the document file
            category: Document category
            metadata: Document metadata

        Returns:
            Dict with ingestion result
        """
        try:
            # Prepare document for ingestion
            doc_data = {
                "file_path": file_path,
                "category": category,
                "metadata": metadata,
                "ingestion_timestamp": datetime.now().isoformat(),
                "document_type": "technical_standard"
            }

            # Use the unified document ingestor
            result = await self.ingestor.ingest_document(doc_data)

            if result.get("status") == "success":
                logger.info(f"Successfully ingested technical document: {Path(file_path).name}")
                return {
                    "status": "success",
                    "file_path": file_path,
                    "category": category,
                    "chunks_ingested": result.get("chunks_ingested", 0),
                    "vectors_created": result.get("vectors_created", 0)
                }
            else:
                logger.error(f"Failed to ingest technical document: {file_path}")
                return {
                    "status": "error",
                    "file_path": file_path,
                    "error": result.get("error", "Unknown ingestion error")
                }

        except Exception as e:
            logger.exception(f"Error ingesting technical document {file_path}: {e}")
            return {
                "status": "error",
                "file_path": file_path,
                "error": str(e)
            }

    def _extract_iso_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from ISO standard filename"""
        filename = Path(file_path).name

        # Extract ISO number
        iso_match = re.search(r'iso[\s\-_]*(\d+)', filename, re.IGNORECASE)
        iso_number = iso_match.group(1) if iso_match else None

        # Extract year if present
        year_match = re.search(r'(\d{4})', filename)
        year = year_match.group(1) if year_match else None

        # Determine standard type
        standard_type = "Unknown"
        if iso_number:
            if iso_number.startswith('2768'):
                standard_type = "Surface Roughness"
            elif iso_number.startswith('2768'):
                standard_type = "Tolerances"
            elif iso_number.startswith('1302'):
                standard_type = "Technical Drawings"
            elif iso_number.startswith('5457'):
                standard_type = "Technical Product Documentation"

        return {
            "standard_organization": "ISO",
            "standard_number": iso_number,
            "publication_year": year,
            "standard_type": standard_type,
            "document_category": "international_standard",
            "applicability": ["mechanical_engineering", "manufacturing", "quality_control"]
        }

    def _extract_handbook_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from engineering handbook filename"""
        filename = Path(file_path).stem.lower()

        # Determine subjects based on filename keywords
        subjects = []
        subject_keywords = {
            'mechanical': ['mechanical', 'machine', 'machinery'],
            'electrical': ['electrical', 'electronic', 'circuit'],
            'civil': ['civil', 'structural', 'construction'],
            'materials': ['material', 'alloy', 'composite'],
            'manufacturing': ['manufactur', 'process', 'production'],
            'design': ['design', 'engineer'],
            'thermodynamics': ['thermo', 'heat', 'energy'],
            'fluid': ['fluid', 'hydraulic', 'pneumatic']
        }

        for subject, keywords in subject_keywords.items():
            if any(keyword in filename for keyword in keywords):
                subjects.append(subject.title())

        return {
            "document_type": "engineering_handbook",
            "subjects": subjects,
            "applicability": ["education", "reference", "design_guidance"],
            "content_type": "comprehensive_reference"
        }

    def _extract_material_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from material specification filename"""
        filename = Path(file_path).stem.lower()

        # Determine material types
        materials = []
        material_keywords = {
            'steel': ['steel', 'carbon', 'alloy'],
            'aluminum': ['aluminum', 'aluminium'],
            'plastic': ['plastic', 'polymer', 'resin'],
            'composite': ['composite', 'fiber', 'carbon_fiber'],
            'ceramic': ['ceramic', 'silicon'],
            'titanium': ['titanium'],
            'copper': ['copper', 'brass', 'bronze']
        }

        for material, keywords in material_keywords.items():
            if any(keyword in filename for keyword in keywords):
                materials.append(material.title())

        return {
            "document_type": "material_specification",
            "materials": materials,
            "applicability": ["material_selection", "design_validation", "manufacturing"],
            "content_type": "technical_specification"
        }

    async def search_technical_standards(self, query: str,
                                       category: Optional[str] = None,
                                       limit: int = 10) -> Dict[str, Any]:
        """
        Search technical standards using RAG

        Args:
            query: Search query
            category: Specific category to search in
            limit: Maximum number of results

        Returns:
            Dict with search results
        """
        try:
            # Prepare search filters
            filters = {"document_type": "technical_standard"}
            if category:
                filters["category"] = category

            # Perform semantic search
            search_results = await self.vector_db.search(
                query=query,
                filters=filters,
                limit=limit
            )

            return {
                "status": "success",
                "query": query,
                "category": category,
                "total_results": len(search_results),
                "results": search_results
            }

        except Exception as e:
            logger.exception(f"Error searching technical standards: {e}")
            return {
                "status": "error",
                "query": query,
                "error": str(e)
            }

    async def get_standards_summary(self) -> Dict[str, Any]:
        """Get summary of ingested technical standards"""
        try:
            # Query vector database for statistics
            stats = await self.vector_db.get_collection_stats()

            # Get category breakdown
            category_stats = {}
            for category in self.categories.keys():
                # This would require additional vector DB queries to count by category
                # For now, return basic stats
                pass

            return {
                "status": "success",
                "total_documents": stats.get("total_documents", 0),
                "total_chunks": stats.get("total_chunks", 0),
                "categories": list(self.categories.keys()),
                "categories_descriptions": self.categories,
                "last_updated": datetime.now().isoformat()
            }

        except Exception as e:
            logger.exception(f"Error getting standards summary: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Global technical standards ingestor instance
_technical_standards_ingestor = None

def get_technical_standards_ingestor() -> TechnicalStandardsIngestor:
    """Get the global technical standards ingestor instance"""
    global _technical_standards_ingestor
    if _technical_standards_ingestor is None:
        _technical_standards_ingestor = TechnicalStandardsIngestor()
    return _technical_standards_ingestor

async def ingest_iso_standards(standards_path: str, standard_ids: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to ingest ISO standards

    Args:
        standards_path: Path to ISO standards directory
        standard_ids: Specific standard IDs to ingest

    Returns:
        Ingestion results
    """
    ingestor = get_technical_standards_ingestor()
    await ingestor.initialize()
    return await ingestor.ingest_iso_standards(standards_path, standard_ids)

async def ingest_engineering_handbooks(handbooks_path: str, subjects: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to ingest engineering handbooks

    Args:
        handbooks_path: Path to handbooks directory
        subjects: Specific subjects to focus on

    Returns:
        Ingestion results
    """
    ingestor = get_technical_standards_ingestor()
    await ingestor.initialize()
    return await ingestor.ingest_engineering_handbooks(handbooks_path, subjects)


async def search_technical_standards(query: str, category: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    """
    Convenience function to search technical standards

    Args:
        query: Search query
        category: Category to search in
        limit: Maximum results

    Returns:
        Search results
    """
    ingestor = get_technical_standards_ingestor()
    await ingestor.initialize()
    return await ingestor.search_technical_standards(query, category, limit)
