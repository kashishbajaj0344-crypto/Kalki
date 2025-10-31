#!/usr/bin/env python3
"""
WebSearchAgent - Internet connectivity and real-time data retrieval
Extends Kalki with web search, API integration, and external data sources.
"""

import asyncio
import logging
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import time

import requests
from bs4 import BeautifulSoup
import aiohttp

from ..base_agent import BaseAgent, AgentCapability
from ..safety.guard import SafetyGuard


class WebSearchAgent(BaseAgent):
    """
    WebSearchAgent provides internet connectivity for Kalki.
    Supports multiple search providers with safety controls and caching.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize capabilities first
        capabilities = [
            AgentCapability.WEB_SEARCH,
            AgentCapability.DATA_RETRIEVAL,
            AgentCapability.API_INTEGRATION
        ]

        # Call parent constructor
        super().__init__(
            name="WebSearchAgent",
            capabilities=capabilities,
            description="Internet connectivity and web search capabilities",
            config=config
        )

        # Agent metadata
        self.agent_name = "WebSearchAgent"
        self.agent_version = "1.0"

        # Search providers configuration
        self.providers = {
            'google': {
                'enabled': bool(os.getenv('GOOGLE_SEARCH_API_KEY') and os.getenv('GOOGLE_CSE_ID')),
                'api_key': os.getenv('GOOGLE_SEARCH_API_KEY'),
                'cse_id': os.getenv('GOOGLE_CSE_ID'),
                'endpoint': 'https://www.googleapis.com/customsearch/v1'
            },
            'bing': {
                'enabled': bool(os.getenv('BING_SEARCH_API_KEY')),
                'api_key': os.getenv('BING_SEARCH_API_KEY'),
                'endpoint': 'https://api.bing.microsoft.com/v7.0/search'
            },
            'serpapi': {
                'enabled': bool(os.getenv('SERPAPI_KEY')),
                'api_key': os.getenv('SERPAPI_KEY'),
                'endpoint': 'https://serpapi.com/search'
            },
            'duckduckgo': {
                'enabled': True,  # Always available as fallback
                'endpoint': 'https://duckduckgo.com/html/'
            }
        }

        # Rate limiting (requests per minute)
        self.rate_limits = {
            'google': {'requests': 100, 'window': 60},  # 100 per minute
            'bing': {'requests': 1000, 'window': 60},   # 1000 per minute
            'serpapi': {'requests': 100, 'window': 60},  # 100 per minute
            'duckduckgo': {'requests': 30, 'window': 60} # 30 per minute (conservative)
        }

        # Request history for rate limiting
        self.request_history: Dict[str, List[datetime]] = {
            provider: [] for provider in self.providers.keys()
        }

        # Caching
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes

        # Safety controls
        self.safety_guard = SafetyGuard()
        self.blocked_domains = {
            'malicious.com', 'phishing.net', 'spam.org'  # Add more as needed
        }

        # HTTP session
        self.session: Optional[aiohttp.ClientSession] = None

        self.logger.info("WebSearchAgent initialized")

    async def initialize(self) -> bool:
        """Initialize the agent"""
        try:
            # Create HTTP session with SSL workaround for macOS
            import ssl
            ssl_context = ssl.create_default_context()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            
            connector = aiohttp.TCPConnector(ssl=ssl_context)
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(connector=connector, timeout=timeout)

            self.logger.info("WebSearchAgent initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Failed to initialize WebSearchAgent: {e}")
            return False

    async def search(self, query: str, max_results: int = 5,
                    provider: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Perform web search with specified query.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            provider: Specific provider to use (auto-select if None)

        Returns:
            List of search results with title, url, snippet
        """
        try:
            # Safety check
            if not await self._is_safe_query(query):
                self.logger.warning(f"Blocked unsafe query: {query}")
                return []

            # Check cache first
            cache_key = f"{query}_{max_results}"
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                self.logger.info(f"Returning cached result for: {query}")
                return cached_result

            # Select provider
            if not provider:
                provider = self._select_provider(query)

            if not provider or not self.providers[provider]['enabled']:
                self.logger.warning(f"No suitable provider available for query: {query}")
                return []

            # Check rate limit
            if not self._check_rate_limit(provider):
                self.logger.warning(f"Rate limit exceeded for {provider}")
                # Try fallback provider
                fallback_provider = self._get_fallback_provider(provider)
                if fallback_provider:
                    provider = fallback_provider
                else:
                    return []

            # Perform search with fallback
            try:
                results = await self._perform_search(provider, query, max_results)
            except Exception as e:
                self.logger.warning(f"Primary provider {provider} failed: {e}")
                results = []
            
            # If no results and we have a fallback, try it
            if not results:
                fallback_provider = self._get_fallback_provider(provider)
                if fallback_provider:
                    self.logger.info(f"Trying fallback provider {fallback_provider} for query: {query}")
                    try:
                        results = await self._perform_search(fallback_provider, query, max_results)
                    except Exception as e:
                        self.logger.warning(f"Fallback provider {fallback_provider} also failed: {e}")
                        results = []

            # Cache results
            self._cache_result(cache_key, results)

            return results

        except Exception as e:
            self.logger.exception(f"Search failed for query '{query}': {e}")
            return []

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a web search task.

        Args:
            task: Task dictionary with action and parameters

        Returns:
            Result dictionary
        """
        try:
            action = task.get('action', '')
            params = task.get('params', {})

            if action == 'search':
                query = params.get('query', '')
                max_results = params.get('max_results', 5)
                provider = params.get('provider')

                results = await self.search(query, max_results, provider)
                return {
                    'status': 'success',
                    'results': results,
                    'query': query,
                    'total_results': len(results),
                    'provider': provider or 'auto'
                }

            elif action == 'fetch_content':
                url = params.get('url', '')
                content = await self.fetch_content(url)
                return {
                    'status': 'success',
                    'result': content,
                    'url': url,
                    'content_length': len(content) if content else 0
                }

            else:
                return {
                    'status': 'error',
                    'error': f"Unknown action: {action}",
                    'supported_actions': ['search', 'fetch_content']
                }

        except Exception as e:
            self.logger.exception(f"Task execution failed: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }

    async def _perform_search(self, provider: str, query: str,
                            max_results: int) -> List[Dict[str, Any]]:
        """Perform search using specified provider"""
        if provider == 'google':
            return await self._google_custom_search(query, max_results)
        elif provider == 'bing':
            return await self._bing_search(query, max_results)
        elif provider == 'serpapi':
            return await self._serpapi_search(query, max_results)
        elif provider == 'duckduckgo':
            return await self._duckduckgo_search(query, max_results)
        else:
            self.logger.error(f"Unknown provider: {provider}")
            return []

    async def _google_custom_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Google Custom Search API"""
        try:
            config = self.providers['google']
            params = {
                'key': config['api_key'],
                'cx': config['cse_id'],
                'q': query,
                'num': min(max_results, 10)  # Google max is 10
            }

            async with self.session.get(config['endpoint'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get('items', [])[:max_results]:
                        result = {
                            'title': item.get('title', ''),
                            'url': item.get('link', ''),
                            'snippet': item.get('snippet', ''),
                            'provider': 'google'
                        }
                        results.append(result)

                    return results
                else:
                    self.logger.error(f"Google search failed: {response.status}")
                    return []

        except Exception as e:
            self.logger.exception(f"Google search error: {e}")
            return []

    async def _bing_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using Bing Search API"""
        try:
            config = self.providers['bing']
            headers = {'Ocp-Apim-Subscription-Key': config['api_key']}
            params = {
                'q': query,
                'count': min(max_results, 50),  # Bing max is 50
                'responseFilter': 'Webpages'
            }

            async with self.session.get(config['endpoint'], headers=headers, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get('webPages', {}).get('value', [])[:max_results]:
                        result = {
                            'title': item.get('name', ''),
                            'url': item.get('url', ''),
                            'snippet': item.get('snippet', ''),
                            'provider': 'bing'
                        }
                        results.append(result)

                    return results
                else:
                    self.logger.error(f"Bing search failed: {response.status}")
                    return []

        except Exception as e:
            self.logger.exception(f"Bing search error: {e}")
            return []

    async def _serpapi_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using SerpApi"""
        try:
            config = self.providers['serpapi']
            params = {
                'api_key': config['api_key'],
                'q': query,
                'num': min(max_results, 10),
                'engine': 'google'
            }

            async with self.session.get(config['endpoint'], params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []

                    for item in data.get('organic_results', [])[:max_results]:
                        result = {
                            'title': item.get('title', ''),
                            'url': item.get('link', ''),
                            'snippet': item.get('snippet', ''),
                            'provider': 'serpapi'
                        }
                        results.append(result)

                    return results
                else:
                    self.logger.error(f"SerpApi search failed: {response.status}")
                    return []

        except Exception as e:
            self.logger.exception(f"SerpApi search error: {e}")
            return []

    async def _duckduckgo_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo (fallback, no API key required)"""
        try:
            # DuckDuckGo doesn't have an official API, so we scrape the HTML
            params = {'q': query}
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            async with self.session.get(self.providers['duckduckgo']['endpoint'],
                                      params=params, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    results = []
                    # Parse DuckDuckGo results (this is a simplified parser)
                    result_divs = soup.find_all('div', class_='result')

                    for div in result_divs[:max_results]:
                        title_elem = div.find('a', class_='result__a')
                        snippet_elem = div.find('a', class_='result__snippet')

                        if title_elem:
                            title = title_elem.get_text().strip()
                            url = title_elem.get('href', '')

                            # DuckDuckGo uses redirect URLs, extract real URL
                            if url.startswith('/l/?uddg='):
                                url = url.split('uddg=')[1].split('&')[0]

                            snippet = snippet_elem.get_text().strip() if snippet_elem else ""

                            result = {
                                'title': title,
                                'url': url,
                                'snippet': snippet,
                                'provider': 'duckduckgo'
                            }
                            results.append(result)

                    return results
                else:
                    self.logger.error(f"DuckDuckGo search failed: {response.status}")
                    return []

        except Exception as e:
            self.logger.exception(f"DuckDuckGo search error: {e}")
            return []

    async def fetch_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract content from a URL.

        Args:
            url: URL to fetch content from

        Returns:
            Extracted text content or None if failed
        """
        try:
            # Safety check
            if not await self._is_safe_url(url):
                self.logger.warning(f"Blocked unsafe URL: {url}")
                return None

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }

            async with self.session.get(url, headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')

                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()

                    # Extract text
                    text = soup.get_text()

                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)

                    return text
                else:
                    self.logger.error(f"Failed to fetch {url}: {response.status}")
                    return None

        except Exception as e:
            self.logger.exception(f"Error fetching content from {url}: {e}")
            return None

    async def _is_safe_query(self, query: str) -> bool:
        """Check if search query is safe"""
        # Use safety guard for content filtering
        is_safe, violations = self.safety_guard.content_filter.check(query)
        if not is_safe:
            self.logger.warning(f"Query blocked due to violations: {violations}")
        return is_safe

    async def _is_safe_url(self, url: str) -> bool:
        """Check if URL is safe to access"""
        # Check blocked domains
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.lower()

        if any(blocked in domain for blocked in self.blocked_domains):
            return False

        return True

    def _select_provider(self, query: str) -> Optional[str]:
        """Select best available provider for query"""
        # Priority: Google > Bing > SerpApi > DuckDuckGo
        priority = ['google', 'bing', 'serpapi', 'duckduckgo']

        for provider in priority:
            if self.providers[provider]['enabled']:
                return provider

        return None

    def _get_fallback_provider(self, current_provider: str) -> Optional[str]:
        """Get fallback provider when rate limited"""
        fallbacks = {
            'google': 'bing',
            'bing': 'serpapi',
            'serpapi': 'duckduckgo',
            'duckduckgo': None
        }

        fallback = fallbacks.get(current_provider)
        if fallback and self.providers[fallback]['enabled']:
            return fallback

        return None

    def _check_rate_limit(self, provider: str) -> bool:
        """Check if request is within rate limits"""
        now = datetime.now()
        limits = self.rate_limits[provider]

        # Clean old requests
        cutoff = now - timedelta(seconds=limits['window'])
        self.request_history[provider] = [
            req_time for req_time in self.request_history[provider]
            if req_time > cutoff
        ]

        # Check if under limit
        if len(self.request_history[provider]) < limits['requests']:
            self.request_history[provider].append(now)
            return True

        return False

    def _get_cached_result(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached result if available and not expired"""
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['results']
            else:
                # Remove expired cache
                del self.cache[cache_key]

        return None

    def _cache_result(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Cache search results"""
        self.cache[cache_key] = {
            'results': results,
            'timestamp': time.time()
        }

    async def shutdown(self) -> bool:
        """Clean shutdown"""
        try:
            if hasattr(self, 'session'):
                await self.session.close()
            self.logger.info("WebSearchAgent shut down successfully")
            return True
        except Exception as e:
            self.logger.exception(f"Shutdown error: {e}")
            return False