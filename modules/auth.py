"""Temporary auth/credentials shim for local testing.

This module provides a very small-compatible surface so agents that import
`modules.auth` can run locally during development. It should be replaced by
your production credential manager or a proper secrets store.

DO NOT commit real secrets into this file.
"""
from typing import Optional
import asyncio


def get_api_key(provider: str = "openai") -> Optional[str]:
    """Return an API key for the named provider from environment or config.

    This shim returns None by default. Set the environment variables or
    implement a secure retrieval here for real runs.
    """
    # Prefer environment-based secrets if available
    try:
        import os
        key = os.environ.get("KALKI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        return key
    except Exception:
        return None


def get_auth_headers(provider: str = "openai") -> dict:
    """Return simple authorization headers if an API key is present."""
    key = get_api_key(provider=provider)
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


async def get_api_key_async(provider: str = "openai") -> Optional[str]:
    """Async wrapper around get_api_key for compatibility with async callers.

    This implementation is intentionally simple: it runs the sync lookup in
    a thread and returns the result. Replace with an async secrets lookup if
    you integrate a real secret manager later.
    """
    return await asyncio.to_thread(get_api_key, provider)


async def get_auth_headers_async(provider: str = "openai") -> dict:
    """Async wrapper returning auth headers when an API key is available."""
    key = await get_api_key_async(provider)
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


__all__ = ["get_api_key", "get_auth_headers", "get_api_key_async", "get_auth_headers_async"]
