"""
KALKI v2.3 — OCR Module v1.5
------------------------------------------------------------
Enterprise-grade OCR subsystem for Kalki ingestion pipeline.
- Multi-format: PDFs, images, scanned docs
- Multi-strategy: pdfplumber first, fallback to pdf2image + pytesseract
- Async-safe I/O wrappers for pipeline scaling
- Error-tolerant, dependency aware, robust logging
- Supports multi-page extraction, language selection
- Registers version for audit and traceability
- Self-test CLI for module verification
"""

import os
import asyncio
from pathlib import Path
from typing import List, Optional, Union
from modules.config import register_module_version
from modules.logger import get_logger

__version__ = "KALKI v2.3 — ocr.py v1.5"
register_module_version("ocr.py", __version__)
logger = get_logger("OCR")

# --- Dependency Management ---
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
    logger.warning("pdfplumber not installed; PDF text extraction may fallback to OCR.")

try:
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:
    convert_from_path = None
    pytesseract = None
    logger.warning("pdf2image or pytesseract not installed; OCR fallback unavailable.")

POPPLER_PATH = os.getenv("POPPLER_PATH", None)

# ------------------------------------------------------------
# Main OCR Extraction API
# ------------------------------------------------------------
def extract_text_pdf(file_path: Path, lang: str = "eng") -> Optional[str]:
    """
    Extract text from PDF using:
    1. pdfplumber (native text layer)
    2. pdf2image + pytesseract (OCR fallback)
    Returns text or None.
    """
    if not file_path.exists() or not file_path.is_file():
        logger.error("File does not exist or is not a file: %s", file_path)
        return None

    # --- Primary: pdfplumber ---
    if pdfplumber:
        try:
            text = ""
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            if text.strip():
                logger.debug("Text extracted via pdfplumber for %s", file_path.name)
                return text.strip()
        except Exception as e:
            logger.warning("pdfplumber extraction failed for %s: %s", file_path.name, e)

    # --- Fallback: pdf2image + pytesseract ---
    if convert_from_path and pytesseract:
        try:
            logger.info(f"OCR fallback activated for: {file_path.name}")
            images = convert_from_path(str(file_path), poppler_path=POPPLER_PATH)
            text_pages: List[str] = []
            for idx, image in enumerate(images):
                page_text = pytesseract.image_to_string(image, lang=lang)
                text_pages.append(page_text)
                logger.debug(f"OCR processed page {idx + 1}/{len(images)}")
            full_text = "\n".join(text_pages)
            if full_text.strip():
                logger.info(f"OCR completed for {file_path.name} ({len(full_text)} chars)")
                return full_text.strip()
        except Exception as e:
            logger.error("OCR extraction failed for %s: %s", file_path.name, e)

    logger.warning("Failed to extract text from %s", file_path.name)
    return None

def ocr_image(image_path: Path, lang: str = "eng") -> Optional[str]:
    """Extract text directly from an image file using pytesseract."""
    if not pytesseract:
        logger.error("pytesseract not available.")
        return None
    try:
        from PIL import Image
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img, lang=lang)
        logger.info(f"OCR completed for image: {image_path.name}")
        return text.strip() if text else None
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return None

def extract_text_file(file_path: Union[str, Path], lang: str = "eng") -> Optional[str]:
    """
    Unified dispatcher for OCR/text extraction from PDF/image files.
    Returns all extracted text as a single string (or None).
    """
    path = Path(file_path)
    ext = path.suffix.lower()
    if ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
        return ocr_image(path, lang)
    elif ext == ".pdf":
        return extract_text_pdf(path, lang)
    else:
        logger.warning(f"Unsupported file type for OCR: {path}")
        return None

# ------------------------------------------------------------
# Async wrappers for ingestion pipelines
# ------------------------------------------------------------
async def async_ocr_image(image_path: Path, lang: str = "eng") -> Optional[str]:
    """Async wrapper for OCR on images."""
    return await asyncio.to_thread(ocr_image, image_path, lang)

async def async_ocr_pdf(pdf_path: Path, lang: str = "eng") -> Optional[str]:
    """Async wrapper for OCR on PDFs."""
    return await asyncio.to_thread(extract_text_pdf, pdf_path, lang)

async def async_extract_text_file(file_path: Union[str, Path], lang: str = "eng") -> Optional[str]:
    """Async wrapper for generic OCR extraction."""
    return await asyncio.to_thread(extract_text_file, file_path, lang)

def get_version() -> str:
    return __version__

# ------------------------------------------------------------
# Self-test / CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Kalki OCR Extractor")
    parser.add_argument("file", type=str, help="Path to PDF or image file")
    parser.add_argument("--lang", type=str, default="eng", help="OCR language (default: eng)")
    args = parser.parse_args()
    path = Path(args.file)
    text = extract_text_file(path, args.lang)
    if text:
        print(text)
    else:
        print(f"OCR failed for {path}")