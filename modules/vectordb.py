"""
KALKI v2.3 — Vector Database Manager v15.0
------------------------------------------------------------
- Local embedding engine using BGE-Large (BAAI/bge-large-en-v1.5)
- Integrates with ChromaDB for vector storage and semantic retrieval.
- Handles deduplication, metadata validation, async ingestion, and robust retries.
- Fully replaces OpenAIEmbeddings with transformer-based embeddings.
"""

import os
import time
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Union

import torch
from transformers import AutoTokenizer, AutoModel
import json

try:
    from langchain_chroma import Chroma
    from langchain_community.vectorstores.utils import filter_complex_metadata
except ImportError:
    Chroma = None
    def filter_complex_metadata(meta): return meta

try:
    from modules.config import CONFIG, register_module_version
except ImportError:
    CONFIG = {"vector_db_dir": "db/chroma"}
    def register_module_version(module, version): pass

try:
    from modules.logger import get_logger
except ImportError:
    import logging
    def get_logger(name): return logging.getLogger(name)
logger = get_logger("vectordb")

try:
    from modules.filehash import compute_sha256
except ImportError:
    def compute_sha256(x): return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# Local Embedding Model (BGE Large)
# ------------------------------------------------------------
class BGEEmbedder:
    """Local semantic embedding generator using BGE Large."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"[BGE] Loading model '{model_name}' on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]
        encoded = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        model_output = self.model(**encoded)
        embeddings = model_output.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()


class BGEEmbeddingsAdapter:
    """Adapter to expose the embedder with the LangChain Embeddings API.

    Chroma/langchain expects an object with `embed_documents` and
    `embed_query` methods. The local `BGEEmbedder.embed` method returns a
    list of embeddings for a list of texts; this adapter forwards calls and
    normalizes the return shape.
    """
    def __init__(self, embedder: BGEEmbedder):
        self._embedder = embedder

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.embed(texts)

    def embed_query(self, text: str) -> List[float]:
        embs = self._embedder.embed(text)
        return embs[0] if isinstance(embs, list) and embs else embs


__version__ = "KALKI v2.3 — vectordb.py v15.0"
register_module_version("vectordb.py", __version__)


# ------------------------------------------------------------
# Vector Database Manager
# ------------------------------------------------------------
class VectorDBManager:
    """Handles embedding, storage, semantic search, deduplication for document chunks."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self.persist_dir = Path(persist_dir) if persist_dir else Path(CONFIG.get("vector_db_dir", "db/chroma"))
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = BGEEmbedder()
        # Wrap embedder in an adapter that implements the LangChain Embeddings
        # interface so Chroma can call embed_documents/embed_query safely.
        self.embedding_adapter = BGEEmbeddingsAdapter(self.embedder)
        if Chroma:
            self.db = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embedding_adapter
            )
        else:
            self.db = None
            logger.warning("Chroma is not installed or import failed.")

        self.known_hashes: Set[str] = self._load_known_hashes()
        logger.info("VectorDBManager initialized at %s", self.persist_dir)

    def _load_known_hashes(self) -> Set[str]:
        known = set()
        if not self.db:
            return known
        try:
            all_metadatas = self.db.get()["metadatas"]
            for meta in all_metadatas:
                if meta and isinstance(meta, dict):
                    if "hash" in meta:
                        known.add(meta["hash"])
                    if "chunk_id" in meta:
                        known.add(meta["chunk_id"])
            logger.debug("Loaded %d known hashes/chunk_ids.", len(known))
        except Exception as e:
            logger.warning("Could not load known hashes: %s", e)
        return known

    def _validate_metadata(self, metadata: Any) -> Dict[str, Any]:
        if not isinstance(metadata, dict):
            logger.warning(f"[VectorDB] Metadata type is {type(metadata)}, converting to empty dict.")
            metadata = {}
        def _coerce_value(v: Any) -> Any:
            # Allowed primitive types: str, int, float, bool, None
            if v is None:
                return None
            if isinstance(v, (str, int, float, bool)):
                return v
            if isinstance(v, list):
                # If list contains only simple primitives, join into string
                if all(isinstance(i, (str, int, float, bool)) for i in v):
                    return ", ".join(map(str, v))
                # Otherwise JSON-serialize (langchain/Chroma expects simple types)
                try:
                    return json.dumps(v, default=repr, ensure_ascii=False)
                except Exception:
                    return repr(v)
            if isinstance(v, dict):
                try:
                    return json.dumps(v, default=repr, ensure_ascii=False)
                except Exception:
                    return repr(v)
            # Fallback: string representation
            return repr(v)

        meta_clean = {k: _coerce_value(v) for k, v in metadata.items()}

        # Try to use external helper if it accepts our dict; otherwise fall back
        # to the coerced metadata which only contains allowed primitive types.
        try:
            result = filter_complex_metadata(meta_clean)
            return result if isinstance(result, dict) else meta_clean
        except Exception:
            return meta_clean

    def add_document(
        self,
        file_path: Path,
        texts: Union[str, List[str]],
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        retry: int = 3,
        retry_delay: float = 2.0,
    ) -> bool:
        if not self.db:
            logger.warning("VectorDB not available.")
            return False
        if isinstance(texts, str):
            texts = [texts]
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif isinstance(metadatas, dict):
            metadatas = [metadatas]
        elif len(metadatas) != len(texts):
            logger.error("metadatas and texts length mismatch")
            return False

        filtered_texts, filtered_metas = [], []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            if not isinstance(meta, dict):
                meta = {}
            chunk_id = meta.get("chunk_id") or meta.get("hash") or compute_sha256(file_path)
            meta["hash"] = chunk_id
            meta["chunk_id"] = chunk_id
            if chunk_id in self.known_hashes:
                logger.info("Duplicate chunk detected, skipping: %s", chunk_id)
                continue
            meta_clean = self._validate_metadata(meta)
            filtered_texts.append(text)
            filtered_metas.append(meta_clean)

        if not filtered_texts:
            logger.info("No new chunks to add for %s", file_path)
            return False

        for attempt in range(1, retry + 1):
            try:
                self.db.add_texts(filtered_texts, metadatas=filtered_metas)
                for meta in filtered_metas:
                    if "chunk_id" in meta:
                        self.known_hashes.add(meta["chunk_id"])
                # Some Chroma/langchain versions expose a persist() method;
                # others do not. Call persist() only when available to avoid
                # attribute errors on different library versions.
                if hasattr(self.db, "persist") and callable(getattr(self.db, "persist")):
                    try:
                        self.db.persist()
                    except Exception as e:
                        logger.warning("VectorDB persist() raised an exception: %s", e)
                else:
                    logger.debug("VectorDB persist() not available on this Chroma object; skipping persist step.")
                logger.info("Added %d chunks from %s", len(filtered_texts), file_path)
                return True
            except Exception as e:
                logger.error("Attempt %d failed: %s", attempt, e)
                if attempt < retry:
                    time.sleep(retry_delay * attempt)
        return False

    async def add_document_async(
        self,
        file_path: Path,
        texts: Union[str, List[str]],
        metadatas: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        retry: int = 3,
        retry_delay: float = 2.0,
    ) -> bool:
        return await asyncio.to_thread(self.add_document, file_path, texts, metadatas, retry, retry_delay)

    def query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        if not self.db:
            return []
        try:
            results = self.db.similarity_search_with_score(query_text, k=k)
            return [{"text": r[0].page_content, "metadata": r[0].metadata, "score": r[1]} for r in results]
        except Exception as e:
            logger.error("Query failed: %s", e)
            return []

    def rebuild_index(self):
        if not self.db:
            return
        try:
            self.db.delete_collection()
            self.db = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embedding_adapter
            )
            self.known_hashes.clear()
            logger.info("Vector DB index rebuilt.")
        except Exception as e:
            logger.error("Failed to rebuild DB: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = len(self.db.get()["ids"]) if self.db else 0
            return {"collection": str(self.persist_dir), "count": count}
        except Exception as e:
            logger.error(f"[VectorDB] Stats failed: {e}")
            return {"collection": str(self.persist_dir), "count": 0}

    def get_top_k_chunks(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Compatibility wrapper expected by SearchAgent: returns top-k results as list of dicts.

        Uses the existing query() method which returns [{'text', 'metadata', 'score'}, ...].
        """
        return self.query(query_text, k=top_k)

    def search_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar documents in the vector database"""
        try:
            results = self.query(query, k=top_k)
            # Format results for the orchestrator
            formatted_results = []
            for result in results:
                formatted_results.append({
                    "content": result.get("text", ""),
                    "metadata": result.get("metadata", {}),
                    "score": result.get("score", 0.0),
                    "source": "vector_db"
                })
            return formatted_results
        except Exception as e:
            logger.error(f"VectorDB search failed: {e}")
            return []


# ------------------------------------------------------------
# VectorDB Adapter for rag_query.py interface
# ------------------------------------------------------------
class ChromaVectorDBAdapter:
    """Adapter to implement VectorDBAdapter interface for ChromaDB"""

    def __init__(self, collection_name: str = "default"):
        self.manager = VectorDBManager()
        self.collection_name = collection_name

    async def query_embeddings(
        self, embedding: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        # Convert embedding query to text query using the manager's query method
        # This is a simplified implementation - in practice you'd need to implement
        # proper embedding-based search
        results = self.manager.query("", k=top_k)  # Empty query for now
        return [
            {
                "doc_id": r.get("doc_id", ""),
                "text": r.get("text", ""),
                "metadata": r.get("metadata", {}),
                "similarity": r.get("score", 0.0)
            }
            for r in results
        ]

    async def query_text(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        results = self.manager.query(query, k=top_k)
        return [
            {
                "doc_id": r.get("doc_id", ""),
                "text": r.get("text", ""),
                "metadata": r.get("metadata", {}),
                "similarity": r.get("score", 0.0),
                "text_score": r.get("score", 0.0)  # For hybrid scoring
            }
            for r in results
        ]

    async def batch_query_embeddings(
        self, embeddings: List[List[float]], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        # Simplified batch implementation
        results = []
        for _ in embeddings:
            results.append(await self.query_embeddings([], top_k=top_k, filters=filters))
        return results

    async def add_documents(self, docs: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Add multiple documents with their embeddings"""
        for doc, emb in zip(docs, embeddings):
            await self.manager.add_document_async(
                text=doc.get("text", ""),
                metadata=doc.get("metadata", {}),
                embedding=emb
            )


def get_vector_db_adapter(backend: str = "chroma", collection_name: str = "default", config: Optional[Dict[str, Any]] = None) -> ChromaVectorDBAdapter:
    """Factory function to create vector DB adapter"""
    if backend.lower() == "chroma":
        return ChromaVectorDBAdapter(collection_name)
    else:
        raise ValueError(f"Unsupported vector DB backend: {backend}")


def get_version() -> str:
    return __version__


# ------------------------------------------------------------
# Example Usage
# ------------------------------------------------------------
if __name__ == "__main__":
    vdb = VectorDBManager()
    print(vdb.get_stats())
    query_text = "solar energy systems"
    results = vdb.query(query_text)
    for r in results:
        print(f"\nScore: {r['score']:.3f}\nText: {r['text'][:200]}...")