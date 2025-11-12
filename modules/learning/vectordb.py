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
import threading

import torch
from transformers import AutoTokenizer, AutoModel
import json

try:
    import chromadb
    from chromadb.utils import embedding_functions
except ImportError:
    chromadb = None
    embedding_functions = None

# Simple metadata filter function
def filter_complex_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Filter out complex metadata types that ChromaDB can't handle"""
    filtered = {}
    for k, v in metadata.items():
        if isinstance(v, (str, int, float, bool)):
            filtered[k] = v
        elif isinstance(v, list) and all(isinstance(item, (str, int, float, bool)) for item in v):
            filtered[k] = v
        elif isinstance(v, dict):
            # Convert dict to JSON string
            try:
                import json
                filtered[k] = json.dumps(v)
            except:
                filtered[k] = str(v)
        else:
            # Convert other types to string
            filtered[k] = str(v)
    return filtered

try:
    from modules.utils.config import CONFIG, register_module_version
except ImportError:
    CONFIG = {"vector_db_dir": "db/chroma"}
    def register_module_version(module, version): pass

try:
    from modules.utils.logger import get_logger
except ImportError:
    import logging
    def get_logger(name): return logging.getLogger(name)
logger = get_logger("vectordb")

try:
    from modules.utils.filehash import compute_sha256
except ImportError:
    def compute_sha256(x): return hashlib.sha256(str(x).encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# Local Embedding Model (BGE Large)
# ------------------------------------------------------------
class BGEEmbedder:
    """Local semantic embedding generator using BGE Large with domain adaptation."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: Optional[str] = None, cache_enabled: bool = True, domain: str = "general"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        logger.info(f"[BGE] Loading model '{model_name}' on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Domain adaptation
        self.domain = domain
        self.domain_instructions = self._get_domain_instructions()
        
        # Thread safety for parallel processing
        self.tokenizer_lock = threading.Lock()
        
        # Embedding cache
        self.cache_enabled = cache_enabled
        self.embedding_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Quantization settings
        self.quantize_embeddings = False
        self.quantization_bits = 8  # int8 quantization

    def _get_domain_instructions(self) -> Dict[str, str]:
        """Get domain-specific instruction prefixes for better embeddings."""
        return {
            "engineering": "This is an engineering and technical document about mechanical design, manufacturing, CAD, specifications, and technical standards. ",
            "medical": "This is a medical document about healthcare, patient care, clinical procedures, and medical research. ",
            "legal": "This is a legal document about laws, regulations, contracts, and legal proceedings. ",
            "scientific": "This is a scientific document about research, experiments, theories, and academic studies. ",
            "general": ""
        }

    def set_domain(self, domain: str):
        """Set the domain for domain-adaptive embeddings."""
        if domain in self.domain_instructions:
            self.domain = domain
            logger.info(f"[BGE] Domain set to: {domain}")
        else:
            logger.warning(f"[BGE] Unknown domain '{domain}', using 'general'")
            self.domain = "general"

    def _apply_domain_adaptation(self, texts: List[str]) -> List[str]:
        """Apply domain-specific prefixes to improve embedding quality."""
        if self.domain == "general" or not self.domain_instructions.get(self.domain):
            return texts
        
        instruction = self.domain_instructions[self.domain]
        
        # Apply instruction prefix to each text
        adapted_texts = []
        for text in texts:
            # For short texts, prepend instruction
            if len(text.split()) < 50:
                adapted_text = instruction + text
            else:
                # For longer texts, insert instruction at the beginning
                adapted_text = instruction + text
            
            # Truncate if too long (BGE has 512 token limit)
            tokens = self.tokenizer.encode(adapted_text, add_special_tokens=True)
            if len(tokens) > 510:  # Leave room for special tokens
                adapted_text = self.tokenizer.decode(tokens[:510], skip_special_tokens=True)
            
            adapted_texts.append(adapted_text)
        
        return adapted_texts

    def enable_quantization(self, bits: int = 8):
        """Enable embedding quantization to reduce storage."""
        self.quantize_embeddings = True
        self.quantization_bits = bits
        logger.info(f"[BGE] Enabled {bits}-bit quantization for embeddings")

    def disable_quantization(self):
        """Disable embedding quantization."""
        self.quantize_embeddings = False
        logger.info("[BGE] Disabled embedding quantization")

    def _quantize_embedding(self, embedding: List[float]) -> List[float]:
        """Quantize embedding to reduce storage requirements."""
        if not self.quantize_embeddings:
            return embedding
        
        import numpy as np
        
        # Convert to numpy for efficient operations
        emb_array = np.array(embedding, dtype=np.float32)
        
        if self.quantization_bits == 8:
            # Scale to int8 range (-128 to 127)
            # Since embeddings are L2 normalized, they range from -1 to 1
            # We'll scale to use the full int8 range
            scale_factor = 127.0
            quantized = np.round(emb_array * scale_factor).astype(np.int8)
            # Store as float for compatibility, but compressed
            return quantized.astype(np.float32) / scale_factor
        else:
            # For other bit depths, implement as needed
            return embedding

    def _dequantize_embedding(self, quantized_embedding: List[float]) -> List[float]:
        """Dequantize embedding back to full precision."""
        if not self.quantize_embeddings:
            return quantized_embedding
        # For int8 quantization, embeddings are already in float32 but scaled
        # No dequantization needed for retrieval since we work in normalized space
        return quantized_embedding

    def _get_text_hash(self, text: str) -> str:
        """Generate hash for text caching."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()

    @torch.inference_mode()
    def embed(self, texts: Union[str, List[str]], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings for text(s) with optimized batch processing, caching, and domain adaptation."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return []
        
        # Apply domain adaptation
        adapted_texts = self._apply_domain_adaptation(texts)
        
        # Check cache for single text
        if len(adapted_texts) == 1 and self.cache_enabled:
            text_hash = self._get_text_hash(adapted_texts[0])
            if text_hash in self.embedding_cache:
                self.cache_hits += 1
                return [self.embedding_cache[text_hash]]
        
        all_embeddings = []
        texts_to_process = []
        cache_indices = []  # Track which positions in result come from cache
        
        # Check cache for each adapted text
        if self.cache_enabled:
            for i, adapted_text in enumerate(adapted_texts):
                text_hash = self._get_text_hash(adapted_text)
                if text_hash in self.embedding_cache:
                    all_embeddings.append(self.embedding_cache[text_hash])
                    self.cache_hits += 1
                else:
                    texts_to_process.append(adapted_text)
                    cache_indices.append(i)
                    self.cache_misses += 1
        else:
            texts_to_process = adapted_texts
            cache_indices = list(range(len(adapted_texts)))
        
        # Process uncached texts in batches
        if texts_to_process:
            for i in range(0, len(texts_to_process), batch_size):
                batch_texts = texts_to_process[i:i + batch_size]
                
                # Tokenize batch with thread safety
                with self.tokenizer_lock:
                    encoded = self.tokenizer(
                        batch_texts, 
                        padding=True, 
                        truncation=True, 
                        max_length=512,  # BGE optimal length
                        return_tensors="pt"
                    ).to(self.device)
                
                # Generate embeddings
                model_output = self.model(**encoded)
                embeddings = model_output.last_hidden_state.mean(dim=1)
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to CPU and collect
                batch_embeddings = embeddings.cpu().tolist()
                
                # Apply quantization if enabled
                if self.quantize_embeddings:
                    batch_embeddings = [self._quantize_embedding(emb) for emb in batch_embeddings]
                
                # Cache embeddings
                if self.cache_enabled:
                    for text, embedding in zip(batch_texts, batch_embeddings):
                        text_hash = self._get_text_hash(text)
                        self.embedding_cache[text_hash] = embedding
                
                all_embeddings.extend(batch_embeddings)
                
                # Memory cleanup
                del encoded, model_output, embeddings
                if torch.cuda.is_available() or torch.backends.mps.is_available():
                    torch.cuda.empty_cache() if torch.cuda.is_available() else torch.mps.empty_cache()
        
        # Reorder results to match input order
        if self.cache_enabled and len(adapted_texts) > 1:
            ordered_embeddings = [None] * len(adapted_texts)
            cache_idx = 0
            process_idx = 0
            
            for i, adapted_text in enumerate(adapted_texts):
                text_hash = self._get_text_hash(adapted_text)
                if text_hash in self.embedding_cache:
                    ordered_embeddings[i] = self.embedding_cache[text_hash]
                else:
                    ordered_embeddings[i] = all_embeddings[len(adapted_texts) - len(texts_to_process) + process_idx]
                    process_idx += 1
            
            all_embeddings = ordered_embeddings
        
        return all_embeddings

    def get_cache_stats(self) -> Dict[str, int]:
        """Get embedding cache statistics."""
        total_requests = self.cache_hits + self.cache_misses
        hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        return {
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate_percent": round(hit_rate, 2),
            "cached_embeddings": len(self.embedding_cache)
        }


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

        # Initialize embedder with engineering domain for CAD/engineering focus
        self.embedder = BGEEmbedder(domain="engineering")

        # Use direct chromadb client
        if chromadb:
            self.client = chromadb.PersistentClient(path=str(self.persist_dir))
            # Create or get collection (without embedding function - we'll handle embeddings manually)
            try:
                self.collection = self.client.get_or_create_collection(
                    name="kalki_documents"
                )
                self.db = self.collection  # For backward compatibility
            except Exception as e:
                logger.error(f"Failed to create Chroma collection: {e}")
                self.collection = None
                self.db = None
        else:
            self.client = None
            self.collection = None
            self.db = None
            logger.warning("ChromaDB is not installed or import failed.")

        self.known_hashes: Set[str] = self._load_known_hashes()
        logger.info("VectorDBManager initialized at %s", self.persist_dir)

    def _load_known_hashes(self) -> Set[str]:
        known = set()
        if not self.collection:
            return known
        try:
            # Get all documents and their metadata
            results = self.collection.get()
            metadatas = results.get("metadatas", [])
            for meta in metadatas:
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
        if not self.collection:
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

        filtered_texts, filtered_metas, ids = [], [], []
        for i, (text, meta) in enumerate(zip(texts, metadatas)):
            if not isinstance(meta, dict):
                meta = {}
            chunk_id = meta.get("chunk_id") or meta.get("hash") or f"{compute_sha256(file_path)}_{i}"
            meta["hash"] = chunk_id
            meta["chunk_id"] = chunk_id
            if chunk_id in self.known_hashes:
                logger.info("Duplicate chunk detected, skipping: %s", chunk_id)
                continue
            meta_clean = self._validate_metadata(meta)
            filtered_texts.append(text)
            filtered_metas.append(meta_clean)
            ids.append(chunk_id)

        if not filtered_texts:
            logger.info("No new chunks to add for %s", file_path)
            return False

        for attempt in range(1, retry + 1):
            try:
                # Generate embeddings for the texts
                embeddings = self.embedder.embed(filtered_texts)

                # Use chromadb collection API with manual embeddings
                self.collection.add(
                    documents=filtered_texts,
                    embeddings=embeddings,
                    metadatas=filtered_metas,
                    ids=ids
                )
                # Update known hashes
                for chunk_id in ids:
                    self.known_hashes.add(chunk_id)
                logger.info("Added %d chunks from %s", len(filtered_texts), file_path)
                return True
            except Exception as e:
                logger.exception("Attempt %d failed to add document: %s", attempt, e)
                if attempt < retry:
                    import time
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

    def rebuild_index(self) -> None:
        """Rebuild the vector database index from scratch."""
        if not self.client:
            return
        try:
            # Delete and recreate collection
            try:
                self.client.delete_collection("kalki_documents")
            except:
                pass  # Collection might not exist

            self.collection = self.client.create_collection(
                name="kalki_documents",
                embedding_function=self.embedder.embed
            )
            self.db = self.collection  # For backward compatibility
            self.known_hashes.clear()
            logger.info("Vector DB index rebuilt.")
        except Exception as e:
            logger.error("Failed to rebuild DB: %s", e)

    def get_stats(self) -> Dict[str, Any]:
        try:
            count = self.collection.count() if self.collection else 0
            return {"collection": str(self.persist_dir), "count": count}
        except Exception as e:
            logger.error(f"[VectorDB] Stats failed: {e}")
            return {"collection": str(self.persist_dir), "count": 0}

    def query(self, query_text: str, k: int = 5, use_reranking: bool = False) -> List[Dict[str, Any]]:
        """Query the vector database for similar documents with optional LLM re-ranking"""
        if not self.collection:
            logger.warning("VectorDB not available for query")
            return []

        try:
            # Generate embedding for the query
            query_embedding = self.embedder.embed(query_text)[0]

            # Query the collection with manual embedding (get more results for re-ranking)
            n_results = k * 2 if use_reranking else k
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Format results
            formatted_results = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for i, (doc, meta, distance) in enumerate(zip(documents, metadatas, distances)):
                formatted_results.append({
                    "text": doc,
                    "metadata": meta or {},
                    "score": 1.0 - distance,  # Convert distance to similarity score
                    "doc_id": meta.get("chunk_id", f"doc_{i}") if meta else f"doc_{i}"
                })

            # Apply LLM re-ranking if requested
            if use_reranking and formatted_results:
                formatted_results = self._rerank_with_llm(query_text, formatted_results, k)

            return formatted_results[:k]  # Return top k results

        except Exception as e:
            logger.error(f"VectorDB query failed: {e}")
            return []

    def _rerank_with_llm(self, query: str, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Re-rank search results using LLM for better relevance scoring."""
        try:
            # Import LLM engine
            from modules.llm import LLMEngine
            import asyncio
            
            if not hasattr(self, '_llm_engine') or self._llm_engine is None:
                try:
                    self._llm_engine = LLMEngine()
                except Exception as e:
                    logger.warning(f"LLM engine not available for re-ranking: {e}")
                    return results
            
            reranked_results = []
            
            # Prepare re-ranking prompt
            rerank_prompt = f"""Given the query: "{query}"

Please score the following document excerpts for relevance on a scale of 0-10 (where 10 is perfectly relevant and 0 is completely irrelevant). Consider:
- Direct relevance to the query
- Technical accuracy and depth
- Completeness of information
- Recency and applicability

Return only a JSON array of scores in the same order as the excerpts.

Excerpts:
"""
            
            # Add excerpts to prompt
            for i, result in enumerate(results):
                excerpt = result['text'][:500] + "..." if len(result['text']) > 500 else result['text']
                rerank_prompt += f"\n{i+1}. {excerpt}\n"
            
            rerank_prompt += "\nScores (JSON array):"
            
            # Get LLM response (handle async properly)
            try:
                # Create event loop if needed
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                if loop.is_running():
                    # If loop is already running, we need to handle this differently
                    # For now, skip re-ranking in this case
                    logger.warning("Event loop already running, skipping LLM re-ranking")
                    return results
                
                response = loop.run_until_complete(self._llm_engine.generate(rerank_prompt, max_tokens=200, temperature=0.1))
                
            except Exception as e:
                logger.warning(f"LLM generation failed: {e}")
                return results
            
            # Parse scores from response
            try:
                # Extract JSON array from response
                import re
                json_match = re.search(r'\[.*\]', response)
                if json_match:
                    scores = json.loads(json_match.group())
                    
                    # Apply scores to results
                    for i, result in enumerate(results):
                        if i < len(scores):
                            llm_score = float(scores[i]) / 10.0  # Normalize to 0-1
                            # Combine vector similarity with LLM score
                            combined_score = 0.7 * result['score'] + 0.3 * llm_score
                            result['llm_score'] = llm_score
                            result['combined_score'] = combined_score
                        else:
                            result['llm_score'] = 0.0
                            result['combined_score'] = result['score']
                else:
                    logger.warning("Could not parse LLM scores, using original ranking")
                    return results
                    
            except Exception as e:
                logger.warning(f"Failed to parse LLM re-ranking scores: {e}")
                return results
            
            # Sort by combined score
            reranked_results = sorted(results, key=lambda x: x.get('combined_score', x['score']), reverse=True)
            
            logger.info(f"LLM re-ranking completed for {len(reranked_results)} results")
            return reranked_results
            
        except ImportError:
            logger.warning("LLM module not available for re-ranking")
            return results
        except Exception as e:
            logger.error(f"LLM re-ranking failed: {e}")
            return results

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