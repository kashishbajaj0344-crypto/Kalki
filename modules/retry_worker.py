# ============================================================
# Kalki v2.0 — retry_worker.py v2.2
# ------------------------------------------------------------
# - Async retry with backoff, event/callback support, dashboard-ready
# - Batch embedding, error/alert streaming
# - Emits events for API/GUI (subscribe_retry_events)
# ============================================================

import os
import json
import asyncio
import time
from datetime import datetime, timedelta

from modules.vector_db import (
    get_embedding, add_documents, load_retry_queue, persist_retry_queue,
    EMBEDDING_RETRY_PATH, SCHEMA_VERSION, EMBEDDER
)
from modules.logging_config import get_logger

logger = get_logger("Kalki.RetryWorker")

# Backoff parameters
MIN_RETRY_DELAY = 30      # seconds
MAX_RETRY_DELAY = 3600    # 1 hour
BACKOFF_MULT = 2
ALERT_THRESHOLD = 5       # Number of failures after which to alert

# --- Event system for dashboard integration ---
retry_event_callbacks = []

def subscribe_retry_events(cb):
    """Dashboard/WS can register callback to receive retry worker events."""
    retry_event_callbacks.append(cb)

def emit_retry_event(event):
    for cb in retry_event_callbacks:
        try:
            cb(event)
        except Exception:
            pass

def now_utc():
    return datetime.utcnow().isoformat() + "Z"

def load_retry_queue_full():
    """Load queue with full metadata for each failed text."""
    if not os.path.exists(EMBEDDING_RETRY_PATH):
        return []
    queue = []
    with open(EMBEDDING_RETRY_PATH, "r") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                # Each obj: {"text": ..., "doc_id": ..., "metadata": ..., "failures": int, "next_try": ISO8601 str}
                # Backwards compatibility
                if "failures" not in obj:
                    obj["failures"] = 0
                if "next_try" not in obj:
                    obj["next_try"] = now_utc()
                queue.append(obj)
            except Exception:
                continue
    return queue

def save_retry_queue_full(queue):
    with open(EMBEDDING_RETRY_PATH, "w") as f:
        for obj in queue:
            f.write(json.dumps(obj) + "\n")

async def process_retry_queue_async(batch_size=8):
    queue = load_retry_queue_full()
    if not queue:
        logger.info("Retry queue empty.")
        emit_retry_event({"event": "retry_queue_empty"})
        return None  # for dynamic sleep logic

    now = datetime.utcnow()
    to_embed, to_embed_objs = [], []

    still_failed = []
    succeeded = 0

    soonest_next_try = None

    # 1. Collect eligible for retry and track next soonest
    for obj in queue:
        next_try = datetime.fromisoformat(obj.get("next_try", now_utc().replace("Z", "")))
        if next_try > now:
            if soonest_next_try is None or next_try < soonest_next_try:
                soonest_next_try = next_try
            still_failed.append(obj)
            continue
        to_embed.append(obj["text"])
        to_embed_objs.append(obj)

    emit_retry_event({
        "event": "retry_started",
        "count": len(to_embed),
        "timestamp": now.isoformat()
    })

    # 2. Batch embedding (if supported)
    batch_embeds = []
    if hasattr(EMBEDDER, "__call__"):  # Chroma embedder supports batch
        logger.info(f"Attempting batch embedding for {len(to_embed)} items...")
        try:
            for i in range(0, len(to_embed), batch_size):
                batch = to_embed[i:i+batch_size]
                embeds = EMBEDDER(batch)
                batch_embeds.extend(embeds)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            for text in to_embed:
                try:
                    emb = EMBEDDER([text])[0]
                except Exception:
                    emb = None
                batch_embeds.append(emb)
    else:
        for text in to_embed:
            emb = await asyncio.get_event_loop().run_in_executor(None, get_embedding, text)
            batch_embeds.append(emb)

    # 3. Handle results, update queue, emit events
    for idx, obj in enumerate(to_embed_objs):
        text = obj["text"]
        doc_id = obj.get("doc_id")
        metadata = obj.get("metadata")
        failures = obj.get("failures", 0)
        emb = batch_embeds[idx] if idx < len(batch_embeds) else None

        if emb is not None and doc_id and metadata:
            logger.info(f"Embedding success for doc_id={doc_id}")
            emit_retry_event({
                "event": "retry_success",
                "doc_id": doc_id,
                "timestamp": now_utc()
            })
            add_documents([doc_id], [emb], [metadata])
            succeeded += 1
        else:
            failures += 1
            delay = min(MIN_RETRY_DELAY * (BACKOFF_MULT ** (failures - 1)), MAX_RETRY_DELAY)
            next_try_time = (now + timedelta(seconds=delay)).isoformat() + "Z"
            obj.update({"failures": failures, "next_try": next_try_time})
            still_failed.append(obj)
            if failures >= ALERT_THRESHOLD:
                logger.error(f"ALERT: Persistent embedding failure for doc_id={doc_id}, failures={failures}")
                emit_retry_event({
                    "event": "retry_alert",
                    "doc_id": doc_id,
                    "failures": failures,
                    "timestamp": now_utc()
                })
            else:
                emit_retry_event({
                    "event": "retry_fail",
                    "doc_id": doc_id,
                    "failures": failures,
                    "timestamp": now_utc()
                })
    save_retry_queue_full(still_failed)
    logger.info(f"Retry cycle: {succeeded} succeeded, {len(still_failed)} remaining.")
    if soonest_next_try:
        return soonest_next_try
    else:
        return datetime.utcnow() + timedelta(seconds=MIN_RETRY_DELAY)

async def run_forever_async():
    logger.info("Starting async embedding retry worker (dynamic sleep). Ctrl+C to stop.")
    while True:
        next_time = await process_retry_queue_async()
        # Dynamic sleep logic
        if next_time:
            now = datetime.utcnow()
            sleep_sec = (next_time - now).total_seconds()
            if sleep_sec < MIN_RETRY_DELAY:
                sleep_sec = MIN_RETRY_DELAY
            logger.info(f"Sleeping {sleep_sec:.1f}s until next eligible retry...")
            await asyncio.sleep(sleep_sec)
        else:
            logger.info(f"No eligible items, sleeping {MIN_RETRY_DELAY}s.")
            await asyncio.sleep(MIN_RETRY_DELAY)

def add_to_retry_queue(text, doc_id=None, metadata=None):
    obj = {
        "text": text,
        "doc_id": doc_id,
        "metadata": metadata,
        "failures": 0,
        "next_try": now_utc()
    }
    q = load_retry_queue_full()
    q.append(obj)
    save_retry_queue_full(q)

if __name__ == "__main__":
    import sys
    if "--once" in sys.argv:
        asyncio.run(process_retry_queue_async())
    else:
        asyncio.run(run_forever_async())

# Kalki v2.0 — retry_worker.py v2.2