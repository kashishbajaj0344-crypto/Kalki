#!/usr/bin/env python3
"""
Thread-safe ingestion module with parallel embeddings for Kalki v3.0
"""
import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional, List
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
import requests
import portalocker
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config import PDF_DIR, RESOURCES_JSON, INGEST_LOCK, POPPLER_PATH, EMBED_CHUNK_WORDS, EMBED_OVERLAP_WORDS
from .vectordb import add_precomputed_embeddings, add_documents, embeddings as embeddings_client
from .utils import now_ts, load_json, save_json, sha256_file

logger = logging.getLogger("kalki.ingest")

INGEST_INDEX = Path(PDF_DIR.parent) / "kalki_ingest_index.json"

def acquire_ingest_lock(timeout: int = 60):
    f = open(str(INGEST_LOCK), "w+")
    try:
        portalocker.lock(f, portalocker.LockFlags.EXCLUSIVE | portalocker.LockFlags.NON_BLOCKING)
        return f
    except Exception:
        import time
        start = time.time()
        while time.time() - start < timeout:
            try:
                portalocker.lock(f, portalocker.LockFlags.EXCLUSIVE)
                return f
            except Exception:
                time.sleep(0.5)
        f.close()
        raise TimeoutError("Failed to acquire ingest lock within timeout")

def release_ingest_lock(handle):
    try:
        portalocker.unlock(handle)
        handle.close()
    except Exception as e:
        logger.warning(f"Failed to release ingest lock cleanly: {e}")

def ocr_extract_from_pdf(path: Path) -> str:
    try:
        convert_kwargs = {}
        if POPPLER_PATH:
            convert_kwargs["poppler_path"] = POPPLER_PATH
        images = convert_from_path(str(path), **convert_kwargs)
        ocr_texts = [pytesseract.image_to_string(img) for img in images if pytesseract.image_to_string(img)]
        return "\n".join(ocr_texts)
    except Exception:
        logger.exception("OCR extraction failed")
        return ""

def extract_text_from_pdf(path: Path) -> str:
    try:
        parts = []
        with pdfplumber.open(str(path)) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    parts.append(txt)
        text = "\n".join(parts)
        if text.strip():
            return text
        logger.info("No PDF text found; using OCR fallback")
        return ocr_extract_from_pdf(path)
    except Exception:
        logger.exception("extract_text_from_pdf failed")
        return ocr_extract_from_pdf(path)

def load_ingest_index() -> dict:
    return load_json(INGEST_INDEX, {}) or {}

def save_ingest_index(idx: dict):
    save_json(INGEST_INDEX, idx)

def should_skip_file(path: Path, force: bool = False) -> bool:
    if force:
        return False
    file_hash = sha256_file(path)
    idx = load_ingest_index()
    rec = idx.get(file_hash)
    if not rec:
        return False
    if rec.get("path") == str(path) and rec.get("mtime") == path.stat().st_mtime:
        return True
    return False

def chunk_text(text: str, chunk_words: int = EMBED_CHUNK_WORDS, overlap: int = EMBED_OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i+chunk_words]))
        i += max(1, chunk_words - overlap)
    return chunks

def compute_embeddings_parallel(chunks: List[str], max_workers: int = 4) -> List[List[float]]:
    embeddings_out = [None] * len(chunks)
    def worker(i, text):
        return i, embeddings_client.embed_query(text)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, i, c) for i, c in enumerate(chunks)]
        for fut in as_completed(futures):
            try:
                i, emb = fut.result()
                embeddings_out[i] = emb
            except Exception:
                logger.exception("Embedding worker failed")
                embeddings_out[i] = []
    return embeddings_out

def ingest_pdf_file(pdf_path: Path, domain: str = "general", title: Optional[str] = None, author: Optional[str] = None, force: bool = False):
    title = title or pdf_path.stem
    author = author or ""
    if should_skip_file(pdf_path, force=force):
        logger.info(f"Skipping already ingested file: {pdf_path}")
        return False
    text = extract_text_from_pdf(pdf_path)
    if not text:
        logger.warning(f"No text extracted from {pdf_path}")
        return False
    chunks = chunk_text(text)
    if not chunks:
        logger.warning(f"No chunks produced for {pdf_path}")
        return False
    metadatas = []
    ids = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = {"title": title, "author": author, "source": str(pdf_path), "domain": domain, "chunk_id": f"{pdf_path.name}chunk{idx}", "ingested_at": now_ts()}
        metadatas.append(meta)
        ids.append(meta["chunk_id"])
    embeddings_list = compute_embeddings_parallel(chunks, max_workers=4)
    add_precomputed_embeddings(chunks, embeddings_list, metadatas=metadatas, ids=ids)
    file_hash = sha256_file(pdf_path)
    idx = load_ingest_index()
    idx[file_hash] = {"path": str(pdf_path), "mtime": pdf_path.stat().st_mtime, "chunk_count": len(chunks), "last_ingested_at": now_ts()}
    save_ingest_index(idx)
    logger.info(f"Ingested {pdf_path} ({len(chunks)} chunks)")
    logger.info(f"INGEST_OK: {pdf_path} ({len(chunks)} chunks)")
    return True

def ingest_resources_file(resources_path: Path = RESOURCES_JSON, force: bool = False):
    lock_handle = None
    try:
        lock_handle = acquire_ingest_lock()
    except Exception:
        logger.warning("Could not acquire ingest lock")
        return
    try:
        resources = load_json(resources_path, [])
        if not resources:
            logger.info("No resources to ingest.")
            return
        for r in resources:
            local = r.get("local_path")
            if local:
                p = Path(local)
                if p.exists():
                    domain = r.get("domain", "general")
                    target_dir = Path(PDF_DIR) / domain
                    target_dir.mkdir(parents=True, exist_ok=True)
                    target = target_dir / p.name
                    if not target.exists():
                        shutil.copy(p, target)
                    ingest_pdf_file(target, domain=domain, title=r.get("title"), author=r.get("author"), force=force)
            else:
                link = r.get("link")
                if link:
                    domain = r.get("domain", "general")
                    target_dir = Path(PDF_DIR) / domain
                    target_dir.mkdir(parents=True, exist_ok=True)
                    filename = link.split("/")[-1].split("?")[0] or f"{uuid.uuid4()}.pdf"
                    target = target_dir / filename
                    try:
                        resp = requests.get(link, timeout=30)
                        resp.raise_for_status()
                        target.write_bytes(resp.content)
                        ingest_pdf_file(target, domain=domain, title=r.get("title"), author=r.get("author"), force=force)
                    except Exception:
                        logger.exception(f"Failed download {link}")
    except Exception:
        logger.exception("ingest_resources_file failed")
    finally:
        if lock_handle:
            release_ingest_lock(lock_handle)
