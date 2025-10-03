#!/usr/bin/env python3
"""
Thread-safe vectordb helpers for Kalki v3.0
"""
import threading
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from .config import VECTOR_DB_DIR, get_openai_api_key

logger = logging.getLogger("kalki.vectordb")

# Try to import compatible langchain/chroma APIs
try:
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    LC_USE = "community"
except Exception:
    try:
        from langchain.embeddings import OpenAIEmbeddings
        from langchain.vectorstores import Chroma
        LC_USE = "langchain"
    except Exception as e:
        logger.exception("No suitable langchain embeddings/vectorstores found")
        raise

# Lock to protect vectordb operations
vectordb_lock = threading.Lock()

# Init embeddings & vectordb
OPENAI_API_KEY = get_openai_api_key()
if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not found (keyring/env). Embeddings calls may fail until set.")

try:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
except Exception as e:
    logger.exception(f"Failed to init OpenAIEmbeddings: {e}")
    raise

try:
    # Try typical constructor
    try:
        vectordb = Chroma(persist_directory=str(VECTOR_DB_DIR), embedding_function=embeddings)
    except TypeError:
        # fallback factory forms
        try:
            vectordb = Chroma.from_texts([], embedding=embeddings, persist_directory=str(VECTOR_DB_DIR))
        except Exception as e:
            logger.exception(f"Chroma fallback init failed: {e}")
            raise
except Exception as e:
    logger.exception(f"Failed to initialize Chroma vectorstore: {e}")
    raise

def safe_query_raw(query: str, top_k: int = 5) -> List[Any]:
    """Return list of Documents (or empty list on failure)."""
    try:
        with vectordb_lock:
            if hasattr(vectordb, "similarity_search"):
                res = vectordb.similarity_search(query, k=top_k)
            else:
                res = vectordb.query(query_texts=[query], n_results=top_k)
                if isinstance(res, dict):
                    docs = res.get("documents", [[]])[0]
                    return docs
            return res or []
    except Exception as e:
        logger.exception(f"safe_query_raw failed: {e}")
        return []

def get_top_k_chunks(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Normalize results into list of dicts with chunk text and metadata."""
    raw = safe_query_raw(query, top_k=top_k)
    out = []
    for item in raw:
        try:
            text = getattr(item, "page_content", None) or getattr(item, "content", None) or str(item)
            meta = getattr(item, "metadata", None) or {}
        except Exception:
            text = str(item)
            meta = {}
        out.append({"chunk": text, "metadata": meta, "score": None})
    return out

def add_documents(chunks: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
    """Let Chroma compute embeddings and add documents (thread-safe)."""
    metadatas = metadatas or [{} for _ in chunks]
    try:
        with vectordb_lock:
            if hasattr(vectordb, "add_texts"):
                vectordb.add_texts(texts=chunks, metadatas=metadatas, ids=ids)
            elif hasattr(vectordb, "add_documents"):
                from langchain.schema import Document
                docs = [Document(page_content=c, metadata=m) for c, m in zip(chunks, metadatas)]
                vectordb.add_documents(documents=docs, ids=ids)
            else:
                vectordb.add(documents=chunks, metadatas=metadatas, ids=ids)
            try:
                vectordb.persist()
            except Exception as e:
                logger.warning(f"vectordb.persist failed after add_documents: {e}")
    except Exception:
        logger.exception("add_documents failed")
        raise

def add_precomputed_embeddings(chunks: List[str], embeddings_list: List[List[float]], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
    """Add precomputed embeddings if supported; otherwise fallback."""
    metadatas = metadatas or [{} for _ in chunks]
    try:
        with vectordb_lock:
            try:
                vectordb.add(documents=chunks, embeddings=embeddings_list, metadatas=metadatas, ids=ids)
                try:
                    vectordb.persist()
                except Exception as e:
                    logger.warning(f"vectordb.persist failed after add_precomputed_embeddings: {e}")
                return
            except Exception:
                pass
        add_documents(chunks, metadatas=metadatas, ids=ids)
    except Exception:
        logger.exception("add_precomputed_embeddings failed")
        raise

def snapshot_vectordb(destination: Path):
    """Copy vector DB directory to destination (must not exist)."""
    try:
        destination = Path(destination)
        if destination.exists():
            raise FileExistsError(f"{destination} exists")
        with vectordb_lock:
            import shutil
            shutil.copytree(VECTOR_DB_DIR, destination)
        logger.info(f"Snapshot created at {destination}")
    except Exception:
        logger.exception("snapshot_vectordb failed")
        raise

if __name__ == "__main__":
    print("vectordb self-test: safe_query_raw on empty index")
    res = safe_query_raw("test", top_k=1)
    print("safe_query_raw returned", len(res))
