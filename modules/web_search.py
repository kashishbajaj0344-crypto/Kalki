#!/usr/bin/env python3
"""Google Custom Search integration for Kalki desktop."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

import requests


class WebSearchAPI:
    """Thin wrapper around Google Custom Search with safe fallbacks."""

    def __init__(self, session: Optional[requests.Session] = None) -> None:
        self.logger = logging.getLogger(__name__)
        self.providers = ["google_custom_search"]
        self._api_key = os.getenv("GOOGLE_CSE_API_KEY")
        self._cx_id = os.getenv("GOOGLE_CSE_CX")
        self._endpoint = "https://www.googleapis.com/customsearch/v1"
        self._session = session or requests.Session()
        self._last_request = 0.0
        self._min_interval = 1.0  # Cushion to respect Google rate guidance

    def search(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        if not query:
            return []

        if not self._api_key or not self._cx_id:
            self.logger.warning("Google Custom Search credentials unavailable; providing fallback results.")
            return self._mock_results(query, num_results)

        try:
            self._respect_rate_limit()
            params = {
                "key": self._api_key,
                "cx": self._cx_id,
                "q": query,
                "num": max(1, min(num_results, 10)),
                "safe": "active"
            }

            response = self._session.get(self._endpoint, params=params, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except requests.exceptions.RequestException as exc:
            self.logger.error("Google Custom Search request failed: %s", exc)
            return self._mock_results(query, num_results)

        items = payload.get("items", [])
        results: List[Dict[str, Any]] = []

        for item in items[:num_results]:
            results.append(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "source": "Google Custom Search"
                }
            )

        if results:
            return results

        # Empty responses happen on obscure queries; keep UX consistent.
        self.logger.info("Google Custom Search returned no results; providing fallback results.")
        return self._mock_results(query, num_results)

    def _respect_rate_limit(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request = time.time()

    def _mock_results(self, query: str, num_results: int) -> List[Dict[str, Any]]:
        safe_query = query.replace(" ", "+")
        mock_items = [
            {
                "title": f"Result 1 for: {query}",
                "url": f"https://example.com/result1?q={safe_query}",
                "snippet": f"Mock search result for {query}. Replace once live data is available.",
                "source": "Mock Search"
            },
            {
                "title": f"Result 2 for: {query}",
                "url": f"https://example.com/result2?q={safe_query}",
                "snippet": f"Fallback content because live search was unavailable for {query}.",
                "source": "Mock Search"
            },
            {
                "title": f"Result 3 for: {query}",
                "url": f"https://example.com/result3?q={safe_query}",
                "snippet": f"Additional mock resources related to {query}.",
                "source": "Mock Search"
            }
        ]
        return mock_items[:num_results]

    def __str__(self) -> str:
        return f"WebSearchAPI(providers={self.providers})"