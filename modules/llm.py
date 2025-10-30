# ============================================================
# Kalki v2.3 — llm.py
# ------------------------------------------------------------
# - Unified async/sync LLM/embedding interface with batching, error handling
# - OpenAI: handles API exceptions, uses unmasked keys, supports profile
# - HuggingFace: supports transformers pipeline (local) and async Inference API
# - Batch embedding for large input (OpenAI, HF)
# - Key management: async-safe, profile-aware, explicit mask=False
# - Service fallback: fallback to other LLMs if primary fails
# - Detailed logging on all errors/requests
# - CLI: --backend, --embed, --profile, --batch, error output
# ============================================================

import os
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
from modules.auth import get_api_key, get_api_key_async
from modules.config import get_config
from modules.logging_config import get_logger

logger = get_logger("Kalki.LLM")

llm_registry = {}

def register_llm(name: str, generate_func: Callable, embed_func: Callable):
    llm_registry[name] = {"generate": generate_func, "embed": embed_func}

def get_llm_backend(name: str):
    return llm_registry.get(name)

# --- OpenAI backend with error handling, batching, profile support ---
async def openai_generate(prompt: str, model: str = None, profile: str = None, **kwargs) -> str:
    import openai
    key = await get_api_key_async("openai", mask=False, profile=profile) if asyncio.iscoroutinefunction(get_api_key_async) else get_api_key("openai", mask=False, profile=profile)
    openai.api_key = key
    model = model or get_config("llm", "openai_model", "gpt-3.5-turbo")
    try:
        resp = await openai.ChatCompletion.acreate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return resp.choices[0].message.content
    except openai.error.OpenAIError as e:
        logger.error(f"OpenAI error (generate): {e}")
        raise
    except Exception as e:
        logger.error(f"OpenAI unknown error (generate): {e}")
        raise

async def openai_embed(texts: Union[str, List[str]], model: str = None, profile: str = None, batch_size: int = 96) -> List[List[float]]:
    import openai
    key = await get_api_key_async("openai", mask=False, profile=profile) if asyncio.iscoroutinefunction(get_api_key_async) else get_api_key("openai", mask=False, profile=profile)
    openai.api_key = key
    model = model or get_config("llm", "openai_embed_model", "text-embedding-ada-002")
    if isinstance(texts, str):
        texts = [texts]
    results = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i:i+batch_size]
        try:
            resp = await openai.Embedding.acreate(input=chunk, model=model)
            results.extend([d["embedding"] for d in resp["data"]])
        except openai.error.OpenAIError as e:
            logger.error(f"OpenAI error (embedding batch {i}-{i+len(chunk)}): {e}")
            raise
        except Exception as e:
            logger.error(f"OpenAI unknown error (embedding batch {i}-{i+len(chunk)}): {e}")
            raise
    return results

register_llm("openai", openai_generate, openai_embed)

# --- HuggingFace backend: transformers pipeline (local) and async Inference API ---
async def huggingface_generate(prompt: str, model: str = None, profile: str = None, **kwargs) -> str:
    try:
        from transformers import pipeline
        model_name = model or get_config("llm", "hf_model", "gpt2")
        pipe = pipeline("text-generation", model=model_name)
        result = pipe(prompt, **kwargs)
        return result[0]["generated_text"]
    except ImportError:
        logger.error("transformers not installed for HuggingFace local generation.")
        raise
    except Exception as e:
        logger.error(f"HuggingFace error (generate): {e}")
        raise

async def huggingface_embed(texts: Union[str, List[str]], model: str = None, profile: str = None, batch_size: int = 32) -> List[List[float]]:
    try:
        from sentence_transformers import SentenceTransformer
        model_name = model or get_config("llm", "hf_embed_model", "all-MiniLM-L6-v2")
        embedder = SentenceTransformer(model_name)
        if isinstance(texts, str):
            texts = [texts]
        results = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i+batch_size]
            emb = embedder.encode(chunk, convert_to_numpy=True).tolist()
            results.extend(emb)
        return results
    except ImportError:
        # Try HF Inference API as fallback
        import httpx
        key = await get_api_key_async("huggingface", mask=False, profile=profile) if asyncio.iscoroutinefunction(get_api_key_async) else get_api_key("huggingface", mask=False, profile=profile)
        endpoint = model or get_config("llm", "hf_embed_endpoint", None)
        if not endpoint:
            logger.error("No HuggingFace Inference API endpoint set for embedding.")
            raise RuntimeError("No HF endpoint")
        if isinstance(texts, str):
            texts = [texts]
        headers = {"Authorization": f"Bearer {key}"}
        results = []
        async with httpx.AsyncClient() as client:
            for t in texts:
                try:
                    resp = await client.post(endpoint, json={"inputs": t}, headers=headers, timeout=30)
                    resp.raise_for_status()
                    results.append(resp.json()["embedding"])
                except Exception as e:
                    logger.error(f"HuggingFace Inference API error (embed): {e}")
                    raise
        return results
    except Exception as e:
        logger.error(f"HuggingFace error (embedding): {e}")
        raise

register_llm("huggingface", huggingface_generate, huggingface_embed)

# --- Main unified async interface with fallback support ---
async def llm_generate(
    prompt: str,
    backend: str = "openai",
    profile: Optional[str] = None,
    fallbacks: Optional[List[str]] = None,
    **kwargs
) -> str:
    backends = [backend] + (fallbacks or [])
    for be in backends:
        backend_entry = get_llm_backend(be)
        if backend_entry and backend_entry["generate"]:
            try:
                return await backend_entry["generate"](prompt, profile=profile, **kwargs)
            except Exception as e:
                logger.error(f"LLM backend {be} failed: {e}")
    raise RuntimeError(f"All LLM backends failed: {backends}")

async def llm_embed(
    texts: Union[str, List[str]],
    backend: str = "openai",
    profile: Optional[str] = None,
    fallbacks: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    **kwargs
) -> List[List[float]]:
    backends = [backend] + (fallbacks or [])
    for be in backends:
        backend_entry = get_llm_backend(be)
        if backend_entry and backend_entry["embed"]:
            try:
                return await backend_entry["embed"](texts, profile=profile, batch_size=(batch_size or 96), **kwargs)
            except Exception as e:
                logger.error(f"LLM backend {be} failed (embed): {e}")
    raise RuntimeError(f"All embedding backends failed: {backends}")

# --- Sync wrappers ---
def llm_generate_sync(prompt: str, backend: str = "openai", profile: Optional[str] = None, **kwargs) -> str:
    return asyncio.run(llm_generate(prompt, backend, profile, **kwargs))

def llm_embed_sync(texts: Union[str, List[str]], backend: str = "openai", profile: Optional[str] = None, **kwargs) -> List[List[float]]:
    return asyncio.run(llm_embed(texts, backend, profile, **kwargs))

# CLI demo with backend, embed, profile, batch
if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser(description="Kalki LLM CLI")
    parser.add_argument("--backend", type=str, default="openai", help="LLM backend (openai, huggingface)")
    parser.add_argument("--embed", action="store_true", help="Call embedding instead of generate")
    parser.add_argument("--profile", type=str, help="Profile to use for API key")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for embedding")
    parser.add_argument("prompt", nargs="+", help="Prompt or text(s)")
    args = parser.parse_args()

    if args.embed:
        res = llm_embed_sync(args.prompt if args.batch > 1 else args.prompt[0], backend=args.backend, profile=args.profile, batch_size=args.batch)
        print("Embeddings:")
        print(res)
    else:
        prompt = " ".join(args.prompt)
        res = llm_generate_sync(prompt, backend=args.backend, profile=args.profile)
        print("Generated:")
        print(res)

# Kalki v2.3 — llm.py