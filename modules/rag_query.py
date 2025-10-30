# ============================================================
# Kalki v2.0 — rag_query.py v2.2
# ------------------------------------------------------------
# - Vector DB agnostic: plug Chroma, FAISS, Weaviate, etc.
# - Leverages doc_parser.py metadata for filters (headings, tags, keywords, word_count)
# - Customizable scoring: user-adjustable hybrid vs embedding weights
# - Optional semantic reranking via LLM (on top-K results)
# - Typed RAGResult dataclass for structured, rich results
# - Async/sync, batch querying, flexible filters
# ============================================================

import asyncio
import logging
from typing import List, Dict, Any, Optional, Callable, Union, Type
from dataclasses import dataclass
from modules.logging_config import get_logger
from modules.utils import retry

logger = get_logger("Kalki.RAGQuery")

@dataclass
class RAGResult:
    doc_id: str
    text: str
    metadata: Dict[str, Any]
    similarity: float
    relevance_score: Optional[float] = None
    rerank_score: Optional[float] = None
    chunk_id: Optional[int] = None

# --- Embedding Model Wrapper ---
def wrap_embedder(embed_func: Callable[[str], Any]):
    """Accepts sync or async embedding function and returns a standard async callable."""
    if asyncio.iscoroutinefunction(embed_func):
        return embed_func
    async def async_wrapper(text: str):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, embed_func, text)
    return async_wrapper

# --- Vector DB Backend Adapter Interface ---
class VectorDBAdapter:
    """Interface for a pluggable vector DB backend."""
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
    
    async def query_embeddings(
        self, embedding: List[float], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def query_text(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    async def batch_query_embeddings(
        self, embeddings: List[List[float]], top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[List[Dict[str, Any]]]:
        raise NotImplementedError

# --- Scoring and Normalization ---
def score_and_normalize(
    results: List[Dict[str, Any]],
    text_scores: Optional[List[float]] = None,
    hybrid_weight: float = 0.5,
    embedding_weight: float = 0.5
) -> List[RAGResult]:
    """
    Combine embedding similarity with text/hybrid scores for unified ranking.
    User can control hybrid vs embedding contribution.
    """
    sim_scores = [r.get("similarity", 0.0) for r in results]
    sim_min, sim_max = min(sim_scores, default=0), max(sim_scores, default=1)
    norm = lambda s: (s - sim_min) / (sim_max - sim_min) if sim_max > sim_min else s
    rag_results = []
    for idx, r in enumerate(results):
        norm_sim = norm(r.get("similarity", 0.0))
        hybrid_score = text_scores[idx] if text_scores and idx < len(text_scores) else 0.0
        # User-adjustable weights for hybrid (text/metadata) vs embedding
        relevance = hybrid_weight * hybrid_score + embedding_weight * norm_sim
        rag_results.append(
            RAGResult(
                doc_id=r.get("doc_id", ""),
                text=r.get("text", ""),
                metadata=r.get("metadata", {}),
                similarity=norm_sim,
                relevance_score=relevance,
                chunk_id=r.get("chunk_id", None)
            )
        )
    rag_results.sort(key=lambda x: x.relevance_score if x.relevance_score is not None else x.similarity, reverse=True)
    return rag_results

# --- Optional LLM-based semantic reranking ---
async def semantic_rerank(
    results: List[RAGResult],
    query: str,
    llm_rerank_func: Optional[Callable[[str, List[str]], List[float]]] = None
) -> List[RAGResult]:
    """
    Use an LLM to rerank top-k retrieved passages.
    llm_rerank_func: callable that takes (query, [candidate_texts]) and returns [rerank_scores]
    """
    if not llm_rerank_func or not results:
        return results
    texts = [r.text for r in results]
    if asyncio.iscoroutinefunction(llm_rerank_func):
        scores = await llm_rerank_func(query, texts)
    else:
        loop = asyncio.get_event_loop()
        scores = await loop.run_in_executor(None, llm_rerank_func, query, texts)
    for r, score in zip(results, scores):
        r.rerank_score = score
    # Sort by rerank_score if available
    results.sort(key=lambda x: x.rerank_score if x.rerank_score is not None else x.relevance_score, reverse=True)
    return results

# --- Main async RAG Query ---
async def rag_query(
    query_text: str,
    embed_func: Callable[[str], Any],
    vector_db: VectorDBAdapter,
    top_k: int = 10,
    tag_filters: Optional[List[str]] = None,
    keyword_filters: Optional[List[str]] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    doc_metadata: Optional[Dict[str, Any]] = None,
    hybrid: bool = True,
    chunk_level: bool = False,
    hybrid_weight: float = 0.5,
    embedding_weight: float = 0.5,
    llm_rerank_func: Optional[Callable[[str, List[str]], List[float]]] = None
) -> List[RAGResult]:
    """
    Perform a hybrid RAG query: embed the query, search by vector, tags, and doc_parser metadata.
    Returns a ranked list of RAGResult objects.
    """
    logger.info(f"RAG query: '{query_text}' | hybrid={hybrid} | top_k={top_k}")
    filters = {}
    # Integrate doc_parser.py metadata for smarter filtering
    if doc_metadata:
        if doc_metadata.get("headings"):
            filters["headings"] = [h["heading"] for h in doc_metadata["headings"]]
        if doc_metadata.get("tags"):
            filters["tags"] = doc_metadata["tags"]
        if doc_metadata.get("keywords"):
            filters["keywords"] = doc_metadata["keywords"]
        if doc_metadata.get("word_count"):
            filters["word_count"] = doc_metadata["word_count"]
    if tag_filters:
        filters.setdefault("tags", []).extend(tag_filters)
    if keyword_filters:
        filters.setdefault("keywords", []).extend(keyword_filters)
    if metadata_filters:
        filters.update(metadata_filters)

    results = []
    text_scores = None
    if hybrid:
        try:
            results = await vector_db.query_text(query_text, top_k=top_k, filters=filters)
            text_scores = [r.get("text_score", 0.0) for r in results]
        except NotImplementedError:
            logger.warning("Hybrid query not implemented in backend, falling back to embedding search.")
    if not results:
        embedder = wrap_embedder(embed_func)
        embedding = await embedder(query_text)
        results = await vector_db.query_embeddings(embedding, top_k=top_k, filters=filters)
    # Optionally split results into finer chunks
    if chunk_level:
        expanded = []
        for r in results:
            text_chunks = r.get("text_chunks", [])
            for i, chunk in enumerate(text_chunks):
                expanded.append({
                    "doc_id": r.get("doc_id", ""),
                    "text": chunk,
                    "metadata": r.get("metadata", {}),
                    "similarity": r.get("similarity", 0.0),
                    "chunk_id": i
                })
        if expanded:
            results = expanded
    rag_results = score_and_normalize(results, text_scores, hybrid_weight, embedding_weight)
    # Optional: rerank with LLM
    if llm_rerank_func:
        rag_results = await semantic_rerank(rag_results, query_text, llm_rerank_func)
    logger.info(f"RAG query returned {len(rag_results)} results.")
    return rag_results

# --- Batch async RAG Query ---
async def batch_rag_query(
    queries: List[str],
    embed_func: Callable[[str], Any],
    vector_db: VectorDBAdapter,
    top_k: int = 10,
    tag_filters: Optional[List[str]] = None,
    keyword_filters: Optional[List[str]] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    doc_metadatas: Optional[List[Dict[str, Any]]] = None,
    hybrid: bool = True,
    chunk_level: bool = False,
    hybrid_weight: float = 0.5,
    embedding_weight: float = 0.5,
    llm_rerank_func: Optional[Callable[[str, List[str]], List[float]]] = None
) -> List[List[RAGResult]]:
    logger.info(f"Batch RAG query: {len(queries)} queries | hybrid={hybrid} | top_k={top_k}")
    filters = {}
    # doc_metadatas: provide one per query or a shared dict
    if hybrid:
        logger.warning("Batch hybrid query not implemented, using embedding batch.")
    embedder = wrap_embedder(embed_func)
    if asyncio.iscoroutinefunction(embed_func):
        embeddings = await asyncio.gather(*[embed_func(q) for q in queries])
    else:
        embeddings = [embed_func(q) for q in queries]
    results_batches = await vector_db.batch_query_embeddings(embeddings, top_k=top_k, filters=filters)
    batch_results = []
    for i, results in enumerate(results_batches):
        meta = doc_metadatas[i] if doc_metadatas and i < len(doc_metadatas) else None
        batch_results.append(
            await rag_query(
                queries[i], 
                embed_func, 
                vector_db, 
                top_k, 
                tag_filters, 
                keyword_filters, 
                metadata_filters, 
                doc_metadata=meta,
                hybrid=hybrid, 
                chunk_level=chunk_level, 
                hybrid_weight=hybrid_weight, 
                embedding_weight=embedding_weight, 
                llm_rerank_func=llm_rerank_func
            )
        )
    logger.info(f"Batch RAG query completed.")
    return batch_results

# --- Sync wrapper for non-async code ---
def rag_query_sync(
    query_text: str,
    embed_func: Callable[[str], Any],
    vector_db: VectorDBAdapter,
    top_k: int = 10,
    tag_filters: Optional[List[str]] = None,
    keyword_filters: Optional[List[str]] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    doc_metadata: Optional[Dict[str, Any]] = None,
    hybrid: bool = True,
    chunk_level: bool = False,
    hybrid_weight: float = 0.5,
    embedding_weight: float = 0.5,
    llm_rerank_func: Optional[Callable[[str, List[str]], List[float]]] = None
) -> List[RAGResult]:
    return asyncio.run(rag_query(
        query_text, embed_func, vector_db, top_k, tag_filters, keyword_filters, 
        metadata_filters, doc_metadata, hybrid, chunk_level, 
        hybrid_weight, embedding_weight, llm_rerank_func
    ))

# Kalki v2.0 — rag_query.py v2.2