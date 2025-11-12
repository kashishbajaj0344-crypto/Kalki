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
from pathlib import Path
from modules.utils.logging_config import get_logger
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
    related_images: Optional[List[Dict[str, Any]]] = None  # NEW: Related diagrams/images
    visual_similarity: Optional[float] = None  # NEW: Visual similarity score

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

# --- Enhanced LLM-based semantic reranking ---
async def semantic_rerank(
    results: List[RAGResult],
    query: str,
    llm_rerank_func: Optional[Callable[[str, List[str]], List[float]]] = None
) -> List[RAGResult]:
    """
    Use an LLM to rerank top-k retrieved passages with enhanced analysis.
    llm_rerank_func: callable that takes (query, [candidate_texts]) and returns [rerank_scores]
    """
    if not llm_rerank_func or not results:
        return results
    
    # Use enhanced LLM reranking if available
    try:
        enhanced_scores = await _enhanced_llm_rerank(results, query)
        for r, score in zip(results, enhanced_scores):
            r.rerank_score = score
    except Exception as e:
        logger.warning(f"Enhanced reranking failed, falling back to basic: {e}")
        # Fallback to basic reranking
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

async def _enhanced_llm_rerank(results: List[RAGResult], query: str) -> List[float]:
    """
    Enhanced LLM reranking with detailed analysis and scoring.
    
    Args:
        results: List of RAG results to rerank
        query: Original query string
        
    Returns:
        List of reranking scores (0-1 scale)
    """
    try:
        from modules.llm import LLMEngine
        
        llm_engine = LLMEngine()
        
        # Build reranking prompt
        rerank_prompt = f"""
You are an expert document relevance assessor. Evaluate how well each retrieved document passage matches the user's query.

Query: {query}

Passages to evaluate:
{chr(10).join([f"Passage {i+1}: {r.text[:500]}{'...' if len(r.text) > 500 else ''}" for i, r in enumerate(results)])}

For each passage, provide:
1. Relevance score (0.0 to 1.0) - how well it answers the query
2. Key matching elements - specific information that relates to the query
3. Completeness score (0.0 to 1.0) - how complete the answer is

Format your response as:
Passage 1: score=relevance_score, completeness=completeness_score, matches="[key elements]"
Passage 2: score=relevance_score, completeness=completeness_score, matches="[key elements]"
...

Be precise and focus on factual relevance to the query.
"""
        
        response = llm_engine.generate(rerank_prompt, max_tokens=1000, temperature=0.1)
        
        if not response:
            # Fallback to simple scoring
            return [0.5] * len(results)
        
        # Parse the response
        scores = []
        lines = response.strip().split('\n')
        
        for i, line in enumerate(lines):
            if i >= len(results):
                break
                
            try:
                if 'score=' in line and 'completeness=' in line:
                    # Extract scores
                    score_part = line.split('score=')[1].split(',')[0].strip()
                    completeness_part = line.split('completeness=')[1].split(',')[0].strip()
                    
                    relevance_score = float(score_part)
                    completeness_score = float(completeness_part)
                    
                    # Combine scores (weighted average)
                    combined_score = 0.7 * relevance_score + 0.3 * completeness_score
                    scores.append(min(1.0, max(0.0, combined_score)))
                else:
                    scores.append(0.5)  # Default score
            except (ValueError, IndexError):
                scores.append(0.5)  # Default score on parsing error
        
        # Ensure we have scores for all results
        while len(scores) < len(results):
            scores.append(0.5)
        
        return scores[:len(results)]
        
    except Exception as e:
        logger.warning(f"Enhanced LLM reranking failed: {e}")
        return [0.5] * len(results)

# --- Enhanced Main async RAG Query with LLM features ---
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
    llm_rerank_func: Optional[Callable[[str, List[str]], List[float]]] = None,
    use_query_expansion: bool = True,
    generate_answer: bool = True
) -> Dict[str, Any]:
    """
    Perform an enhanced RAG query with LLM-powered features.
    
    Args:
        query_text: The search query
        embed_func: Embedding function
        vector_db: Vector database adapter
        top_k: Number of top results to retrieve
        tag_filters: Tag-based filters
        keyword_filters: Keyword-based filters
        metadata_filters: Additional metadata filters
        doc_metadata: Document metadata for filtering
        hybrid: Whether to use hybrid search
        chunk_level: Whether to search at chunk level
        hybrid_weight: Weight for hybrid scoring
        embedding_weight: Weight for embedding scoring
        llm_rerank_func: LLM reranking function
        use_query_expansion: Whether to expand query with LLM
        generate_answer: Whether to generate answer from results
        
    Returns:
        Dictionary with results and optional generated answer
    """
    logger.info(f"Enhanced RAG query: '{query_text}' | expansion={use_query_expansion} | answer_gen={generate_answer}")
    
    # Step 1: Query expansion (optional)
    query_info = {"original_query": query_text}
    if use_query_expansion:
        try:
            query_info = await expand_query_with_llm(query_text)
            logger.info(f"Query expanded: '{query_info['expanded_query']}'")
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
    
    # Use expanded query for retrieval
    search_query = query_info.get("expanded_query", query_text)
    
    # Build filters including expanded concepts
    filters = {}
    if doc_metadata:
        if doc_metadata.get("headings"):
            filters["headings"] = [h["heading"] for h in doc_metadata["headings"]]
        if doc_metadata.get("tags"):
            filters["tags"] = doc_metadata["tags"]
        if doc_metadata.get("keywords"):
            filters["keywords"] = doc_metadata["keywords"]
        if doc_metadata.get("word_count"):
            filters["word_count"] = doc_metadata["word_count"]
    
    # Add expanded concepts and related terms to filters
    if query_info.get("key_concepts"):
        filters.setdefault("keywords", []).extend(query_info["key_concepts"])
    if query_info.get("related_terms"):
        filters.setdefault("keywords", []).extend(query_info["related_terms"])
    
    if tag_filters:
        filters.setdefault("tags", []).extend(tag_filters)
    if keyword_filters:
        filters.setdefault("keywords", []).extend(keyword_filters)
    if metadata_filters:
        filters.update(metadata_filters)

    # Step 2: Document retrieval
    results = []
    text_scores = None
    if hybrid:
        try:
            results = await vector_db.query_text(search_query, top_k=top_k, filters=filters)
            text_scores = [r.get("text_score", 0.0) for r in results]
        except NotImplementedError:
            logger.warning("Hybrid query not implemented, falling back to embedding search.")
    
    if not results:
        embedder = wrap_embedder(embed_func)
        embedding = await embedder(search_query)
        results = await vector_db.query_embeddings(embedding, top_k=top_k, filters=filters)
    
    # Handle chunk-level results
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
    
    # Step 3: Scoring and normalization
    rag_results = score_and_normalize(results, text_scores, hybrid_weight, embedding_weight)
    
    # Step 4: LLM reranking (enhanced)
    if llm_rerank_func:
        rag_results = await semantic_rerank(rag_results, query_text, llm_rerank_func)
    
    # Step 5: Answer generation (optional)
    answer_result = None
    if generate_answer and rag_results:
        try:
            answer_result = await generate_answer_from_documents(query_text, rag_results)
        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
    
    logger.info(f"Enhanced RAG query completed: {len(rag_results)} results, answer_generated={answer_result is not None}")
    
    return {
        "query_info": query_info,
        "results": rag_results,
        "answer": answer_result,
        "total_results": len(rag_results),
        "search_query_used": search_query
    }

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
    llm_rerank_func: Optional[Callable[[str, List[str]], List[float]]] = None,
    use_query_expansion: bool = True,
    generate_answer: bool = True
) -> List[Dict[str, Any]]:
    logger.info(f"Batch enhanced RAG query: {len(queries)} queries | expansion={use_query_expansion}")
    
    # For batch queries, we'll use basic embedding search for efficiency
    embedder = wrap_embedder(embed_func)
    if asyncio.iscoroutinefunction(embed_func):
        embeddings = await asyncio.gather(*[embed_func(q) for q in queries])
    else:
        embeddings = [embed_func(q) for q in queries]
    
    filters = {}
    if tag_filters:
        filters["tags"] = tag_filters
    if keyword_filters:
        filters["keywords"] = keyword_filters
    if metadata_filters:
        filters.update(metadata_filters)
    
    results_batches = await vector_db.batch_query_embeddings(embeddings, top_k=top_k, filters=filters)
    
    batch_results = []
    for i, results in enumerate(results_batches):
        query = queries[i]
        meta = doc_metadatas[i] if doc_metadatas and i < len(doc_metadatas) else None
        
        # Process each result in the batch
        text_scores = None
        if hybrid and results:
            text_scores = [r.get("text_score", 0.0) for r in results]
        
        if chunk_level:
            expanded = []
            for r in results:
                text_chunks = r.get("text_chunks", [])
                for j, chunk in enumerate(text_chunks):
                    expanded.append({
                        "doc_id": r.get("doc_id", ""),
                        "text": chunk,
                        "metadata": r.get("metadata", {}),
                        "similarity": r.get("similarity", 0.0),
                        "chunk_id": j
                    })
            if expanded:
                results = expanded
        
        rag_results = score_and_normalize(results, text_scores, hybrid_weight, embedding_weight)
        
        if llm_rerank_func:
            rag_results = await semantic_rerank(rag_results, query, llm_rerank_func)
        
        # Query expansion for batch (simplified)
        query_info = {"original_query": query}
        if use_query_expansion:
            try:
                query_info = await expand_query_with_llm(query)
            except:
                pass
        
        # Answer generation
        answer_result = None
        if generate_answer and rag_results:
            try:
                answer_result = await generate_answer_from_documents(query, rag_results)
            except:
                pass
        
        batch_results.append({
            "query_info": query_info,
            "results": rag_results,
            "answer": answer_result,
            "total_results": len(rag_results),
            "search_query_used": query_info.get("expanded_query", query)
        })
    
    logger.info(f"Batch enhanced RAG query completed.")
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
    llm_rerank_func: Optional[Callable[[str, List[str]], List[float]]] = None,
    use_query_expansion: bool = True,
    generate_answer: bool = True
) -> Dict[str, Any]:
    """
    Synchronous wrapper for the enhanced RAG query function.
    """
    return asyncio.run(rag_query(
        query_text, embed_func, vector_db, top_k, tag_filters, keyword_filters, 
        metadata_filters, doc_metadata, hybrid, chunk_level, 
        hybrid_weight, embedding_weight, llm_rerank_func,
        use_query_expansion, generate_answer
    ))

# Kalki v2.0 — rag_query.py v2.2

async def expand_query_with_llm(query: str) -> Dict[str, Any]:
    """
    Use LLM to understand and expand the query for better retrieval.
    
    Args:
        query: Original user query
        
    Returns:
        Dictionary with expanded query information
    """
    try:
        from modules.llm import LLMEngine
        
        llm_engine = LLMEngine()
        
        expansion_prompt = f"""
You are an expert information retrieval assistant. Analyze the user's query and expand it for better document retrieval.

Original Query: {query}

Provide the following expansions:
1. Expanded Query: A more detailed version of the query with additional context and related terms
2. Key Concepts: Main concepts that should be searched for
3. Related Terms: Synonyms and related terms that might appear in relevant documents
4. Question Type: What type of question this is (factual, analytical, comparative, etc.)
5. Expected Answer Type: What kind of answer is expected (definition, explanation, comparison, etc.)

Format your response as:
EXPANDED_QUERY: [expanded query text]
KEY_CONCEPTS: [concept1, concept2, concept3]
RELATED_TERMS: [term1, term2, term3]
QUESTION_TYPE: [type]
EXPECTED_ANSWER_TYPE: [answer type]
"""
        
        response = await llm_engine.generate(expansion_prompt, max_tokens=500, temperature=0.2)
        
        # Parse response
        expanded_info = {
            "expanded_query": query,
            "key_concepts": [],
            "related_terms": [],
            "question_type": "general",
            "expected_answer_type": "information"
        }
        
        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('EXPANDED_QUERY:'):
                expanded_info["expanded_query"] = line.replace('EXPANDED_QUERY:', '').strip()
            elif line.startswith('KEY_CONCEPTS:'):
                concepts_str = line.replace('KEY_CONCEPTS:', '').strip()
                expanded_info["key_concepts"] = [c.strip() for c in concepts_str.split(',') if c.strip()]
            elif line.startswith('RELATED_TERMS:'):
                terms_str = line.replace('RELATED_TERMS:', '').strip()
                expanded_info["related_terms"] = [t.strip() for t in terms_str.split(',') if t.strip()]
            elif line.startswith('QUESTION_TYPE:'):
                expanded_info["question_type"] = line.replace('QUESTION_TYPE:', '').strip()
            elif line.startswith('EXPECTED_ANSWER_TYPE:'):
                expanded_info["expected_answer_type"] = line.replace('EXPECTED_ANSWER_TYPE:', '').strip()
        
        return expanded_info
        
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return {
            "expanded_query": query,
            "key_concepts": [],
            "related_terms": [],
            "question_type": "general",
            "expected_answer_type": "information"
        }

async def generate_answer_from_documents(
    query: str, 
    documents: List[RAGResult],
    max_answer_length: int = 1000
) -> Dict[str, Any]:
    """
    Use LLM to generate a comprehensive answer from retrieved documents.
    
    Args:
        query: Original user query
        documents: Retrieved relevant documents
        max_answer_length: Maximum length of generated answer
        
    Returns:
        Dictionary with generated answer and metadata
    """
    try:
        from modules.llm import LLMEngine
        
        llm_engine = LLMEngine()
        
        # Prepare context from top documents
        context_docs = documents[:5]  # Use top 5 documents
        context_text = "\n\n".join([
            f"Document {i+1} (Relevance: {doc.rerank_score or doc.relevance_score or doc.similarity:.3f}):\n{doc.text}"
            for i, doc in enumerate(context_docs)
        ])
        
        generation_prompt = f"""
You are an expert assistant providing comprehensive answers based on retrieved documents.

User Query: {query}

Retrieved Documents:
{context_text}

Instructions:
1. Provide a comprehensive, accurate answer based ONLY on the information in the retrieved documents
2. If the documents don't contain enough information to fully answer the query, clearly state what information is missing
3. Cite specific documents when relevant information comes from them
4. Be concise but thorough - aim for clarity and completeness
5. If there are conflicting pieces of information, note the discrepancies
6. Structure your answer logically with clear sections if appropriate

Answer the query using the provided documents:
"""
        
        answer = await llm_engine.generate(generation_prompt, max_tokens=max_answer_length, temperature=0.1)
        
        if not answer:
            return {
                "answer": "I apologize, but I was unable to generate an answer from the retrieved documents.",
                "confidence": 0.0,
                "sources_used": len(context_docs),
                "generation_status": "failed"
            }
        
        # Generate confidence score based on document relevance
        avg_relevance = sum([
            doc.rerank_score or doc.relevance_score or doc.similarity 
            for doc in context_docs
        ]) / len(context_docs) if context_docs else 0.0
        
        # Assess answer quality
        quality_prompt = f"""
Rate the quality of this answer on a scale of 0.0 to 1.0:

Query: {query}
Answer: {answer}

Consider:
- How well it addresses the query
- Factual accuracy based on documents
- Completeness of information
- Clarity and coherence

Provide only a numerical score between 0.0 and 1.0:
"""
        
        quality_response = await llm_engine.generate(quality_prompt, max_tokens=10, temperature=0.0)
        confidence = 0.5  # Default
        try:
            if quality_response:
                confidence = min(1.0, max(0.0, float(quality_response.strip().split()[0])))
        except:
            pass
        
        return {
            "answer": answer,
            "confidence": confidence,
            "sources_used": len(context_docs),
            "avg_source_relevance": avg_relevance,
            "generation_status": "success",
            "source_documents": [
                {
                    "doc_id": doc.doc_id,
                    "relevance": doc.rerank_score or doc.relevance_score or doc.similarity,
                    "text_preview": doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
                }
                for doc in context_docs
            ]
        }
        
    except Exception as e:
        logger.exception(f"Answer generation failed: {e}")
        return {
            "answer": f"I encountered an error while generating the answer: {str(e)}",
            "confidence": 0.0,
            "sources_used": 0,
            "generation_status": "error"
        }


# ============================================================
# MULTI-MODAL RAG QUERY SYSTEM v1.0
# ------------------------------------------------------------
# Revolutionary multi-modal retrieval combining:
# - Text-based semantic search (traditional RAG)
# - Visual knowledge graph integration
# - Image similarity search with CLIP embeddings
# - Cross-modal ranking (text + visual relevance)
# - Unified result presentation (text + diagrams)
# ============================================================

@dataclass
class MultiModalRAGResult:
    """Enhanced RAG result with visual components."""
    text_results: List[RAGResult]
    visual_results: List[Dict[str, Any]]
    combined_ranking: List[Dict[str, Any]]
    query_info: Dict[str, Any]
    answer: Optional[Dict[str, Any]] = None
    visual_answer: Optional[Dict[str, Any]] = None  # Answer with visual support


async def multimodal_rag_query(
    query_text: str,
    embed_func: Callable[[str], Any],
    vector_db: VectorDBAdapter,
    top_k: int = 10,
    visual_top_k: int = 5,
    include_visual: bool = True,
    visual_weight: float = 0.3,
    text_weight: float = 0.7,
    tag_filters: Optional[List[str]] = None,
    keyword_filters: Optional[List[str]] = None,
    metadata_filters: Optional[Dict[str, Any]] = None,
    doc_metadata: Optional[Dict[str, Any]] = None,
    hybrid: bool = True,
    chunk_level: bool = False,
    hybrid_weight: float = 0.5,
    embedding_weight: float = 0.5,
    llm_rerank_func: Optional[Callable[[str, List[str]], List[float]]] = None,
    use_query_expansion: bool = True,
    generate_answer: bool = True,
    generate_visual_answer: bool = True,
    domain_filter: Optional[str] = None,
    vision_engine: Optional[Any] = None,
    visual_knowledge_graph: Optional[Any] = None,
    visual_knowledge_base: Optional[Any] = None
) -> MultiModalRAGResult:
    """
    Perform a multi-modal RAG query combining text and visual retrieval.
    
    This revolutionary system:
    - Retrieves relevant text documents (traditional RAG)
    - Retrieves relevant diagrams/images (visual search)
    - Links text and visual results via knowledge graph
    - Ranks results by combined text+visual relevance
    - Generates answer with visual support
    
    Args:
        query_text: The search query
        embed_func: Embedding function for text
        vector_db: Vector database adapter for text
        top_k: Number of top text results
        visual_top_k: Number of top visual results
        include_visual: Whether to include visual search
        visual_weight: Weight for visual results in combined ranking (0-1)
        text_weight: Weight for text results in combined ranking (0-1)
        [... standard RAG args ...]
        domain_filter: Filter by domain (construction, game_dev, etc.)
        vision_engine: Vision model for image analysis
        visual_knowledge_graph: Cross-modal knowledge graph
        visual_knowledge_base: CLIP-based visual KB
        
    Returns:
        MultiModalRAGResult with text, visual, and combined results
    """
    logger.info(f"🎨 Multi-Modal RAG Query: '{query_text}' | visual={include_visual} | domain={domain_filter}")
    
    # ========================================
    # PHASE 1: Text Retrieval (Standard RAG)
    # ========================================
    text_results_dict = await rag_query(
        query_text=query_text,
        embed_func=embed_func,
        vector_db=vector_db,
        top_k=top_k,
        tag_filters=tag_filters,
        keyword_filters=keyword_filters,
        metadata_filters=metadata_filters,
        doc_metadata=doc_metadata,
        hybrid=hybrid,
        chunk_level=chunk_level,
        hybrid_weight=hybrid_weight,
        embedding_weight=embedding_weight,
        llm_rerank_func=llm_rerank_func,
        use_query_expansion=use_query_expansion,
        generate_answer=generate_answer
    )
    
    text_results = text_results_dict.get("results", [])
    query_info = text_results_dict.get("query_info", {"original_query": query_text})
    text_answer = text_results_dict.get("answer", None)
    
    logger.info(f"📄 Text retrieval: {len(text_results)} documents found")
    
    # ========================================
    # PHASE 2: Visual Retrieval (if enabled)
    # ========================================
    visual_results = []
    if include_visual:
        try:
            visual_results = await _retrieve_visual_results(
                query_text=query_text,
                query_info=query_info,
                top_k=visual_top_k,
                domain_filter=domain_filter,
                visual_knowledge_graph=visual_knowledge_graph,
                visual_knowledge_base=visual_knowledge_base
            )
            logger.info(f"🖼️  Visual retrieval: {len(visual_results)} diagrams found")
        except Exception as e:
            logger.warning(f"⚠️  Visual retrieval failed: {e}")
    
    # ========================================
    # PHASE 3: Cross-Modal Linking
    # ========================================
    if visual_knowledge_graph:
        try:
            text_results = await _link_text_to_images(
                text_results=text_results,
                visual_knowledge_graph=visual_knowledge_graph
            )
            logger.info(f"🔗 Cross-modal linking complete")
        except Exception as e:
            logger.warning(f"⚠️  Cross-modal linking failed: {e}")
    
    # ========================================
    # PHASE 4: Combined Ranking
    # ========================================
    combined_ranking = _combine_text_and_visual_results(
        text_results=text_results,
        visual_results=visual_results,
        text_weight=text_weight,
        visual_weight=visual_weight
    )
    
    logger.info(f"📊 Combined ranking: {len(combined_ranking)} total results")
    
    # ========================================
    # PHASE 5: Visual Answer Generation
    # ========================================
    visual_answer = None
    if generate_visual_answer and visual_results and vision_engine:
        try:
            visual_answer = await _generate_visual_answer(
                query=query_text,
                text_answer=text_answer,
                visual_results=visual_results,
                vision_engine=vision_engine
            )
            logger.info(f"🎨 Visual answer generated with {visual_answer.get('diagrams_used', 0)} diagrams")
        except Exception as e:
            logger.warning(f"⚠️  Visual answer generation failed: {e}")
    
    # ========================================
    # Return Multi-Modal Results
    # ========================================
    result = MultiModalRAGResult(
        text_results=text_results,
        visual_results=visual_results,
        combined_ranking=combined_ranking,
        query_info=query_info,
        answer=text_answer,
        visual_answer=visual_answer
    )
    
    logger.info(f"✅ Multi-Modal RAG completed: {len(text_results)} text + {len(visual_results)} visual")
    return result


async def _retrieve_visual_results(
    query_text: str,
    query_info: Dict[str, Any],
    top_k: int = 5,
    domain_filter: Optional[str] = None,
    visual_knowledge_graph: Optional[Any] = None,
    visual_knowledge_base: Optional[Any] = None
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant diagrams/images for the query.
    
    Uses multiple strategies:
    1. Visual Knowledge Graph: Text → Images via links
    2. Visual Knowledge Base: CLIP semantic similarity
    3. Metadata matching: Domain, tags, keywords
    """
    visual_results = []
    
    # Strategy 1: Query visual knowledge graph by text
    if visual_knowledge_graph:
        try:
            graph_results = visual_knowledge_graph.query_with_text(
                query_text=query_text,
                include_text=False,  # Only images
                include_images=True,
                domain_filter=domain_filter
            )
            
            for img in graph_results.get("image_results", []):
                visual_results.append({
                    "image_id": img["id"],
                    "image_path": img["path"],
                    "description": img.get("description", ""),
                    "type": img.get("type", "diagram"),
                    "domain": img.get("domain", ""),
                    "relevance_score": img.get("relevance", 0.7),
                    "source": "knowledge_graph",
                    "linked_from_text": img.get("linked_from_text", [])
                })
        except Exception as e:
            logger.warning(f"Knowledge graph visual retrieval failed: {e}")
    
    # Strategy 2: Query visual knowledge base with CLIP
    if visual_knowledge_base:
        try:
            # Search by text description
            vkb_results = visual_knowledge_base.search_by_text(
                query_text=query_text,
                top_k=top_k,
                domain_filter=domain_filter
            )
            
            for vkb_result in vkb_results:
                # Avoid duplicates
                if not any(v["image_path"] == vkb_result["image_path"] for v in visual_results):
                    visual_results.append({
                        "image_id": vkb_result.get("image_id", ""),
                        "image_path": vkb_result["image_path"],
                        "description": vkb_result.get("description", ""),
                        "type": vkb_result.get("type", "diagram"),
                        "domain": vkb_result.get("domain", ""),
                        "relevance_score": vkb_result.get("similarity", 0.6),
                        "source": "visual_knowledge_base",
                        "clip_similarity": vkb_result.get("similarity", 0.0)
                    })
        except Exception as e:
            logger.warning(f"Visual KB retrieval failed: {e}")
    
    # Strategy 3: Use expanded query concepts
    if visual_knowledge_graph and query_info.get("key_concepts"):
        try:
            for concept in query_info["key_concepts"][:3]:  # Top 3 concepts
                concept_results = visual_knowledge_graph.query_with_text(
                    query_text=concept,
                    include_images=True,
                    include_text=False,
                    domain_filter=domain_filter
                )
                
                for img in concept_results.get("image_results", [])[:2]:  # Top 2 per concept
                    if not any(v["image_path"] == img["path"] for v in visual_results):
                        visual_results.append({
                            "image_id": img["id"],
                            "image_path": img["path"],
                            "description": img.get("description", ""),
                            "type": img.get("type", "diagram"),
                            "domain": img.get("domain", ""),
                            "relevance_score": img.get("relevance", 0.5) * 0.8,  # Lower weight for concept match
                            "source": "concept_expansion",
                            "matched_concept": concept
                        })
        except Exception as e:
            logger.warning(f"Concept-based visual retrieval failed: {e}")
    
    # Sort by relevance and limit
    visual_results.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    return visual_results[:top_k]


async def _link_text_to_images(
    text_results: List[RAGResult],
    visual_knowledge_graph: Any
) -> List[RAGResult]:
    """
    Link text results to related images via knowledge graph.
    
    For each text result:
    - Find linked images in knowledge graph
    - Calculate visual similarity
    - Attach to RAGResult
    """
    for result in text_results:
        try:
            # Try to find this text node in knowledge graph
            # (would need text node ID - simplified here)
            
            # Search for images related to this text content
            related_images_result = visual_knowledge_graph.query_with_text(
                query_text=result.text[:500],  # Use beginning of text
                include_images=True,
                include_text=False
            )
            
            related_images = []
            for img in related_images_result.get("image_results", [])[:3]:  # Top 3 images
                related_images.append({
                    "image_id": img["id"],
                    "image_path": img["path"],
                    "description": img.get("description", ""),
                    "relevance": img.get("relevance", 0.5)
                })
            
            result.related_images = related_images
            result.visual_similarity = (
                sum(img["relevance"] for img in related_images) / len(related_images)
                if related_images else 0.0
            )
            
        except Exception as e:
            logger.debug(f"Could not link text to images: {e}")
            result.related_images = []
            result.visual_similarity = 0.0
    
    return text_results


def _combine_text_and_visual_results(
    text_results: List[RAGResult],
    visual_results: List[Dict[str, Any]],
    text_weight: float = 0.7,
    visual_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    Combine text and visual results into a unified ranking.
    
    Strategy:
    1. Normalize text relevance scores
    2. Normalize visual relevance scores
    3. Create combined list with weighted scores
    4. Sort by combined score
    """
    combined = []
    
    # Add text results
    for result in text_results:
        score = result.rerank_score or result.relevance_score or result.similarity or 0.0
        visual_boost = result.visual_similarity or 0.0
        combined_score = text_weight * score + visual_weight * visual_boost
        
        combined.append({
            "type": "text",
            "content": result.text,
            "doc_id": result.doc_id,
            "metadata": result.metadata,
            "text_score": score,
            "visual_score": visual_boost,
            "combined_score": combined_score,
            "related_images": result.related_images or [],
            "result_object": result
        })
    
    # Add visual results
    for visual in visual_results:
        # Check if this visual is already linked to a text result
        already_linked = any(
            visual["image_path"] in [img["image_path"] for img in (r.get("related_images") or [])]
            for r in text_results
        )
        
        if not already_linked:
            visual_score = visual.get("relevance_score", 0.5)
            combined_score = visual_weight * visual_score
            
            combined.append({
                "type": "visual",
                "image_path": visual["image_path"],
                "image_id": visual.get("image_id", ""),
                "description": visual.get("description", ""),
                "domain": visual.get("domain", ""),
                "text_score": 0.0,
                "visual_score": visual_score,
                "combined_score": combined_score,
                "source": visual.get("source", ""),
                "result_object": visual
            })
    
    # Sort by combined score
    combined.sort(key=lambda x: x["combined_score"], reverse=True)
    
    return combined


async def _generate_visual_answer(
    query: str,
    text_answer: Optional[Dict[str, Any]],
    visual_results: List[Dict[str, Any]],
    vision_engine: Any
) -> Dict[str, Any]:
    """
    Generate an answer enhanced with visual diagrams.
    
    This creates a multi-modal answer that:
    - Starts with text answer (if available)
    - References specific diagrams
    - Uses vision model to analyze diagrams
    - Explains what diagrams show
    - Creates cohesive text+visual explanation
    """
    try:
        from modules.llm import LLMEngine
        
        llm_engine = LLMEngine()
        
        # Get top diagrams
        top_diagrams = visual_results[:3]
        
        # Analyze each diagram with vision model
        diagram_analyses = []
        for diagram in top_diagrams:
            try:
                analysis_prompt = f"""
Analyze this diagram in context of the query: "{query}"

Provide:
1. What the diagram shows
2. How it relates to the query
3. Key elements visible in the diagram
4. What information it provides to answer the query

Be specific and concise.
"""
                
                vision_response = vision_engine.analyze_image(
                    image_path=diagram["image_path"],
                    prompt=analysis_prompt
                )
                
                diagram_analyses.append({
                    "diagram_path": diagram["image_path"],
                    "description": diagram.get("description", ""),
                    "analysis": vision_response,
                    "relevance": diagram.get("relevance_score", 0.5)
                })
                
            except Exception as e:
                logger.warning(f"Could not analyze diagram {diagram['image_path']}: {e}")
        
        # Generate unified answer with visual references
        text_answer_content = ""
        if text_answer:
            text_answer_content = text_answer.get("answer", "")
        
        diagram_summaries = "\n\n".join([
            f"Diagram {i+1}: {a['description']}\nAnalysis: {a['analysis']}"
            for i, a in enumerate(diagram_analyses)
        ])
        
        unified_prompt = f"""
Create a comprehensive answer that combines text information and visual diagrams.

Query: {query}

Text Answer:
{text_answer_content if text_answer_content else "No text answer available."}

Available Diagrams:
{diagram_summaries if diagram_summaries else "No diagrams available."}

Instructions:
1. Synthesize information from both text and visual sources
2. Reference specific diagrams when explaining concepts
3. Explain what each diagram shows and how it supports the answer
4. Create a cohesive multi-modal explanation
5. Use diagram references like "As shown in Diagram 1..." or "The diagram illustrates..."

Provide a comprehensive multi-modal answer:
"""
        
        unified_answer = await llm_engine.generate(
            unified_prompt,
            max_tokens=1200,
            temperature=0.2
        )
        
        return {
            "answer": unified_answer,
            "text_answer": text_answer_content,
            "diagrams_used": len(diagram_analyses),
            "diagram_analyses": diagram_analyses,
            "confidence": text_answer.get("confidence", 0.5) if text_answer else 0.5,
            "answer_type": "multi_modal"
        }
        
    except Exception as e:
        logger.exception(f"Visual answer generation failed: {e}")
        return {
            "answer": text_answer.get("answer", "") if text_answer else "Could not generate answer.",
            "text_answer": text_answer.get("answer", "") if text_answer else "",
            "diagrams_used": 0,
            "diagram_analyses": [],
            "confidence": 0.3,
            "answer_type": "text_only",
            "error": str(e)
        }


# Sync wrapper for multi-modal RAG
def multimodal_rag_query_sync(
    query_text: str,
    embed_func: Callable[[str], Any],
    vector_db: VectorDBAdapter,
    **kwargs
) -> MultiModalRAGResult:
    """Synchronous wrapper for multi-modal RAG query."""
    return asyncio.run(multimodal_rag_query(query_text, embed_func, vector_db, **kwargs))


# ============================================================
# USAGE EXAMPLE
# ============================================================
if __name__ == "__main__":
    """
    Example usage of Multi-Modal RAG Query System:
    
    from modules.learning.rag_query import multimodal_rag_query
    from modules.visual_knowledge_graph import VisualKnowledgeGraph
    from modules.visual_knowledge_base import VisualKnowledgeBase
    from modules.llm import get_llm_engine, get_vision_engine
    
    # Initialize components
    graph = VisualKnowledgeGraph()
    vkb = VisualKnowledgeBase()
    vision = get_vision_engine()
    
    # Query with both text and visual retrieval
    result = await multimodal_rag_query(
        query_text="How do I calculate beam loads?",
        embed_func=embed_function,
        vector_db=vector_db_adapter,
        include_visual=True,
        visual_top_k=5,
        generate_visual_answer=True,
        visual_knowledge_graph=graph,
        visual_knowledge_base=vkb,
        vision_engine=vision,
        domain_filter="construction"
    )
    
    # Access results
    print(f"Text results: {len(result.text_results)}")
    print(f"Visual results: {len(result.visual_results)}")
    print(f"Combined ranking: {len(result.combined_ranking)}")
    
    # Display answer
    if result.visual_answer:
        print(result.visual_answer["answer"])
        print(f"Answer supported by {result.visual_answer['diagrams_used']} diagrams")
    
    # Access individual results
    for item in result.combined_ranking[:5]:
        if item["type"] == "text":
            print(f"Text: {item['content'][:100]}...")
            if item.get("related_images"):
                print(f"  Related images: {len(item['related_images'])}")
        elif item["type"] == "visual":
            print(f"Image: {item['image_path']}")
            print(f"  Description: {item['description']}")
    """
    print("Multi-Modal RAG Query System v1.0")
    print("="*60)
    print("Revolutionary features:")
    print("✅ Text + Visual retrieval")
    print("✅ Cross-modal knowledge graph integration")
    print("✅ CLIP semantic similarity")
    print("✅ Unified text+visual ranking")
    print("✅ Multi-modal answer generation")
    print("✅ Diagram analysis with vision model")
    print("="*60)