# ============================================================
# Kalki v2.4 — pipeline.py
# ------------------------------------------------------------
# - Concurrent async ingestion (asyncio.gather) for multiple files
# - Per-file and per-batch chunk parallel embedding (asyncio.gather)
# - Exponential backoff for embedding retries
# - Progress bars for both files and chunks (tqdm.asyncio)
# - Partial vector DB writes: writes each successful embedding batch immediately
# - Metrics tracking: per-doc chunk count, embedding and DB times, retries
# - Exception isolation: gathers exceptions, logs and continues
# - CLI: --backend, --collection, --profile, --chunk-size, --batch-size, --concurrency, progress bar, summary
# ============================================================

import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional, Callable
from modules.logging_config import get_logger
from modules.config import get_config
from modules.doc_parser import parse_document_async
from modules.preprocessor import preprocess_text_async
from modules.llm import llm_embed
from modules.vectordb import get_vector_db_adapter

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    tqdm_asyncio = None
    from tqdm import tqdm

logger = get_logger("Kalki.Pipeline")


def _exponential_backoff(attempt: int, base: float = 1.0, factor: float = 2.0, max_backoff: float = 16.0) -> float:
    return min(base * (factor ** attempt), max_backoff)


class Pipeline:
    def __init__(
        self,
        vector_db_backend: str = "chroma",
        collection_name: str = "default",
        llm_backend: str = "openai",
        profile: Optional[str] = None,
        vector_db_config: Optional[dict] = None,
        llm_embed_kwargs: Optional[dict] = None,
        embedding_batch_size: int = 96,
        chunk_size: int = 2000,
        max_retries: int = 2,
        chunk_concurrency: int = 4,
    ):
        self.vector_db = get_vector_db_adapter(vector_db_backend, collection_name, config=vector_db_config)
        self.llm_backend = llm_backend
        self.profile = profile
        self.llm_embed_kwargs = llm_embed_kwargs or {}
        self.embedding_batch_size = embedding_batch_size
        self.chunk_size = chunk_size
        self.max_retries = max_retries
        self.chunk_concurrency = chunk_concurrency

    async def embed_chunks_with_retries(
        self,
        chunk_batch: List[str],
        batch_idx: int,
        total_batches: int,
        progress_bar=None
    ) -> (List[List[float]], List[int], List[str]):
        """
        Embeds a batch of chunks with retries and exponential backoff.
        Returns: (successful_embeddings, failed_indices, failed_errors)
        """
        for retry in range(self.max_retries + 1):
            try:
                start_time = time.time()
                embeddings = await llm_embed(
                    chunk_batch,
                    backend=self.llm_backend,
                    profile=self.profile,
                    batch_size=self.embedding_batch_size,
                    **self.llm_embed_kwargs
                )
                if progress_bar:
                    progress_bar.update(len(chunk_batch))
                elapsed = time.time() - start_time
                logger.info(f"Batch {batch_idx+1}/{total_batches}: Embedded {len(chunk_batch)} chunks in {elapsed:.2f}s")
                return embeddings, [], []
            except Exception as e:
                logger.error(f"Embedding batch {batch_idx+1}/{total_batches} failed on attempt {retry+1}: {e}")
                if retry < self.max_retries:
                    backoff = _exponential_backoff(retry)
                    logger.warning(f"Retrying in {backoff:.1f}s ...")
                    await asyncio.sleep(backoff)
                else:
                    logger.error(f"Batch {batch_idx+1}/{total_batches} failed after {self.max_retries+1} attempts.")
                    if progress_bar:
                        progress_bar.update(len(chunk_batch))
                    return [], list(range(len(chunk_batch))), [str(e)] * len(chunk_batch)

    async def ingest_single_document(
        self,
        path: str,
        preprocess_fn: Optional[Callable] = None,
        parser_fn: Optional[Callable] = None,
        progress_bar=None
    ) -> Dict[str, Any]:
        """
        Async ingestion for a single document, with retries, error capture, and per-batch vector DB writes.
        """
        preprocess_fn = preprocess_fn or preprocess_text_async
        parser_fn = parser_fn or parse_document_async
        result = {
            "path": path, "success": False, "error": None, "chunks": [], "failed_chunks": [],
            "embedding_times": [], "db_times": [], "doc_id": None, "file_hash": None, "num_chunks": 0
        }
        try:
            # 1. Preprocess
            logger.info(f"Preprocessing {path}")
            text = await preprocess_fn(path)
            # 2. Parse
            logger.info(f"Parsing {path}")
            parsed = await parser_fn(path, text)
            # 3. Chunk
            chunks = [text[i:i+self.chunk_size] for i in range(0, len(text), self.chunk_size)]
            doc_uuid = str(uuid.uuid4())
            doc_id = f"{parsed.get('file_hash', path)}-{doc_uuid}"
            result["doc_id"] = doc_id
            result["file_hash"] = parsed.get("file_hash")
            result["num_chunks"] = len(chunks)
            # 4. Embedding in concurrent batches
            batches = [
                chunks[i:i+self.embedding_batch_size]
                for i in range(0, len(chunks), self.embedding_batch_size)
            ]
            all_embeddings = []
            all_failed_indices = []
            all_failed_errors = []
            all_batch_docs = []
            total_batches = len(batches)

            async def embed_and_write(batch_idx, chunk_batch):
                emb, failed_idx, failed_err = await self.embed_chunks_with_retries(
                    chunk_batch, batch_idx, total_batches, progress_bar
                )
                # Write immediately to vector DB
                docs = []
                for idx, chunk in enumerate(chunk_batch):
                    doc = {
                        "doc_id": doc_id,
                        "text": chunk,
                        "metadata": {
                            **parsed,
                            "file_chunk": batch_idx * self.embedding_batch_size + idx,
                            "chunk_start": (batch_idx * self.embedding_batch_size + idx) * self.chunk_size,
                            "chunk_end": min((batch_idx * self.embedding_batch_size + idx + 1) * self.chunk_size, len(text)),
                        }
                    }
                    docs.append(doc)
                # Only add docs for which embeddings exist
                if emb and len(emb) == len(docs):
                    db_start = time.time()
                    await self.vector_db.add_documents(docs, emb)
                    db_elapsed = time.time() - db_start
                    if progress_bar:
                        progress_bar.set_postfix_str(f"DB {db_elapsed:.2f}s")
                    logger.info(f"Batch {batch_idx+1}/{total_batches}: Added {len(emb)} docs to vector DB in {db_elapsed:.2f}s")
                else:
                    logger.warning(f"Batch {batch_idx+1}/{total_batches}: Skipped vector DB write due to embedding failure.")
                return emb, failed_idx, failed_err

            sem = asyncio.Semaphore(self.chunk_concurrency)
            async def bounded_embed_and_write(idx, chunk_batch):
                async with sem:
                    return await embed_and_write(idx, chunk_batch)

            batch_tasks = [
                bounded_embed_and_write(batch_idx, chunk_batch)
                for batch_idx, chunk_batch in enumerate(batches)
            ]
            # Progress bar for chunks
            if tqdm_asyncio and progress_bar is None:
                progress_bar = tqdm_asyncio(total=len(chunks), desc=path.split("/")[-1][:32])
            elif tqdm_asyncio is None and progress_bar is None:
                progress_bar = tqdm(total=len(chunks), desc=path.split("/")[-1][:32])
            # Gather results, keep going on exceptions
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            if progress_bar:
                progress_bar.close()
            # Accumulate embeddings and failures
            for emb, failed_idx, failed_err in batch_results:
                if emb:
                    all_embeddings.extend(emb)
                if failed_idx:
                    all_failed_indices.extend(failed_idx)
                    all_failed_errors.extend(failed_err)
            result.update({
                "success": (not all_failed_indices) and (len(all_embeddings) == len(chunks)),
                "chunks": list(range(len(all_embeddings))),
                "failed_chunks": list(zip(all_failed_indices, all_failed_errors)),
            })
        except Exception as e:
            logger.error(f"Failed to ingest {path}: {e}")
            result["error"] = str(e)
        return result

    async def ingest_documents(
        self,
        filepaths: List[str],
        preprocess_fn: Optional[Callable] = None,
        parser_fn: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        """
        Concurrent async ingestion pipeline for multiple files.
        Returns list of processed document results.
        """
        logger.info(f"Starting concurrent ingestion for {len(filepaths)} files")
        if tqdm_asyncio:
            outer_bar = tqdm_asyncio(total=len(filepaths), desc="Files")
        else:
            outer_bar = tqdm(total=len(filepaths), desc="Files")
        sem = asyncio.Semaphore(self.chunk_concurrency)
        async def bounded_ingest(path):
            async with sem:
                res = await self.ingest_single_document(path)
                if outer_bar:
                    outer_bar.update(1)
                return res

        tasks = [bounded_ingest(path) for path in filepaths]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        if outer_bar:
            outer_bar.close()
        # Exception isolation: filter and log errors
        out_results = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"File ingestion failed: {res}")
                out_results.append({"success": False, "error": str(res)})
            else:
                out_results.append(res)
        logger.info("Ingestion pipeline complete")
        return out_results

    def ingest_documents_sync(
        self,
        filepaths: List[str],
        preprocess_fn: Optional[Callable] = None,
        parser_fn: Optional[Callable] = None,
    ) -> List[Dict[str, Any]]:
        return asyncio.run(self.ingest_documents(filepaths, preprocess_fn, parser_fn))

# CLI with full concurrency/progress options
if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser(description="Kalki Pipeline CLI")
    parser.add_argument("--backend", type=str, default="chroma", help="Vector DB backend")
    parser.add_argument("--collection", type=str, default="default", help="Collection name")
    parser.add_argument("--llm-backend", type=str, default="openai", help="LLM/embedding backend")
    parser.add_argument("--profile", type=str, help="API key profile")
    parser.add_argument("--chunk-size", type=int, default=2000, help="Chunk size")
    parser.add_argument("--batch-size", type=int, default=96, help="Embedding batch size")
    parser.add_argument("--max-retries", type=int, default=2, help="Max embedding retries")
    parser.add_argument("--concurrency", type=int, default=4, help="Max concurrent files/chunk batches")
    parser.add_argument("files", nargs="+", help="File(s) to ingest")
    args = parser.parse_args()

    pipeline = Pipeline(
        vector_db_backend=args.backend,
        collection_name=args.collection,
        llm_backend=args.llm_backend,
        profile=args.profile,
        chunk_size=args.chunk_size,
        embedding_batch_size=args.batch_size,
        max_retries=args.max_retries,
        chunk_concurrency=args.concurrency
    )
    results = pipeline.ingest_documents_sync(args.files)
    print("\nIngestion Summary:")
    for res in results:
        status = "✅" if res.get("success") else "❌"
        print(f"{status} {res.get('path', '?')}")
        if res.get("failed_chunks"):
            print(f"    Failed chunks: {len(res['failed_chunks'])}")
        if res.get("error"):
            print(f"    Error: {res['error']}")
        if res.get("num_chunks") is not None:
            print(f"    Total chunks: {res['num_chunks']}")
    print(f"\n{sum(1 for r in results if r.get('success'))} / {len(results)} succeeded.")

# Kalki v2.4 — pipeline.py