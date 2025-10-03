#!/usr/bin/env python3
"""
LLM helpers updated to use vectordb.get_top_k_chunks and robust logging.
"""
import logging
from typing import Tuple
from .config import DEFAULT_CHAT_MODEL, RETRY_ATTEMPTS, QUERY_LOG
from .vectordb import get_top_k_chunks
from .utils import now_ts, load_json, save_json

logger = logging.getLogger("kalki.llm")

try:
    from langchain_community.chat_models import ChatOpenAI
except Exception:
    try:
        from langchain.chat_models import ChatOpenAI
    except Exception:
        logger.exception("ChatOpenAI import failed")
        raise

def retrieve_context_for_query(query: str, top_k: int = 5):
    chunks = get_top_k_chunks(query, top_k=top_k)
    context_pieces = []
    used_ids = []
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        citation = " | ".join(filter(None, [meta.get("title"), meta.get("author"), meta.get("source")])) or meta.get("chunk_id","")
        if c.get("chunk"):
            context_pieces.append(f"{citation}\n{c.get('chunk')}")
            used_ids.append(meta.get("chunk_id") or citation)
    context = "\n\n---\n\n".join(context_pieces) if context_pieces else ""
    return context, used_ids

def call_chat_model(query: str, context: str, model: str = None) -> str:
    chosen = model or DEFAULT_CHAT_MODEL
    last_err = None
    try:
        client = ChatOpenAI(model_name=chosen, temperature=0.2)
    except Exception as e:
        logger.exception("Failed to init ChatOpenAI")
        return f"Error initializing chat client: {e}"
    for attempt in range(RETRY_ATTEMPTS + 1):
        try:
            system = {"role":"system","content":"You are Kalki, a helpful assistant. Use the context to answer and cite sources."}
            user = {"role":"user","content":f"Context:\n{context}\n\nQuery:\n{query}"}
            resp = client.chat([system, user])
            if hasattr(resp, "content"):
                return resp.content
            if isinstance(resp, dict):
                return resp.get("content") or str(resp)
            return str(resp)
        except Exception as e:
            last_err = e
            logger.warning(f"call_chat_model attempt {attempt} failed: {e}")
    logger.exception("LLM failed after retries")
    return f"Error: {last_err}"

def ask_kalki(query: str, model: str = None, top_k: int = 5):
    context, used_ids = retrieve_context_for_query(query, top_k=top_k)
    answer = call_chat_model(query, context, model=model)
    try:
        logs = load_json(QUERY_LOG, [])
        if logs is None:
            logs = []
        logs.append({"timestamp": now_ts(), "query": query, "model": model or DEFAULT_CHAT_MODEL, "chunks_used": used_ids, "answer": answer[:1000]})
        save_json(QUERY_LOG, logs)
    except Exception:
        logger.warning("Failed to write query log")
    return answer
