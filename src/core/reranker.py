"""Reranking module for Oh My Repos.

This module provides functionality to rerank search results using Jina AI reranker model.

Note:
    Avoid using unescaped backslashes in docstrings (e.g., use `\\(` instead of `\(`).
"""

import logging
import aiohttp
import time
import asyncio
from typing import Dict, List, Optional, Any
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)

# Fix imports for compatibility
try:
    from src.config import settings
except ImportError:
    from config import settings

logger = logging.getLogger(__name__)


class RerankerError(Exception):
    """Base exception for reranking operations."""

    pass


class RerankerServiceUnavailable(RerankerError):
    """Reranker service is temporarily unavailable."""

    pass


class JinaReranker:
    """Jina AI based reranker for repository search results.

    This class provides methods to rerank search results using Jina's reranker model.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        url: Optional[str] = None,
        timeout: Optional[int] = None,
    ) -> None:
        """Initialize the Jina reranker.

        Args:
            api_key: Jina AI API key
            model: Reranker model name
            url: Reranker API URL
            timeout: API request timeout in seconds
        """
        self.api_key = api_key or settings.EMBEDDING_MODEL_API_KEY
        self.model = model or settings.RERANKER_MODEL
        self.url = url or settings.RERANKER_URL
        self.timeout = timeout or settings.RERANKER_TIMEOUT

        if not self.api_key:
            logger.warning("Jina AI API key not provided, reranker will not work")

        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # Create persistent session with connection pooling
        connector = aiohttp.TCPConnector(
            limit=20,  # Maximum number of connections
            limit_per_host=10,  # Maximum connections per host
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=30,
            enable_cleanup_closed=True,
        )

        timeout_config = aiohttp.ClientTimeout(
            total=self.timeout, connect=5, sock_read=self.timeout - 5
        )

        self.session = aiohttp.ClientSession(
            connector=connector, timeout=timeout_config, headers=self.headers
        )

        logger.debug(f"Initialized JinaReranker with model: {self.model}")

    async def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 25,
    ) -> List[Dict[str, Any]]:
        """Rerank search results using Jina AI reranker.

        Args:
            query (str): The original search query.
            results (List[Dict[str, Any]]): List of search results to rerank.
            top_k (int): Number of top results to return after reranking.

        Returns:
            List[Dict[str, Any]]: Reranked list of repository search results.
        """
        if not self.api_key:
            logger.warning("Reranker API key not provided, returning original results")
            return results[:top_k]

        if not results:
            logger.info("No results to rerank")
            return []

        try:
            start_time = time.time()
            logger.info(f"Reranking {len(results)} results with Jina AI")

            # Prepare documents for reranking
            docs = []
            for result in results:
                # Combine relevant fields for better reranking context
                text = f"{result.get('repo_name', '')} - "

                # Add summary/description if available
                if "summary" in result and result["summary"]:
                    text += result["summary"]
                else:
                    text += result.get("description", "")

                # Add language if available
                if "language" in result and result["language"]:
                    text += f" (Language: {result['language']})"

                # Add tags if available
                if "tags" in result and result["tags"]:
                    if isinstance(result["tags"], list):
                        text += f" Tags: {', '.join(result['tags'])}"

                docs.append(text)

            # Prepare payload for API request
            payload = {
                "model": self.model,
                "query": query,
                "documents": docs,
                "top_n": min(len(docs), top_k * 2),  # Request more for robustness
            }

            logger.debug(
                f"JinaReranker: POST {self.url} | model={self.model} | top_n={payload['top_n']}"
            )

            # Make API request with retry logic
            return await self._rerank_with_retry(payload, results, top_k, start_time)

        except RerankerServiceUnavailable as e:
            logger.error(f"Reranker service unavailable: {e}")
            return self._default_ranking(results, top_k)
        except RerankerError as e:
            logger.error(f"Reranker error: {e}")
            return self._default_ranking(results, top_k)
        except Exception as e:
            logger.error(f"Unexpected reranker error: {e}")
            return self._default_ranking(results, top_k)

    def _default_ranking(
        self, results: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Provide a fallback ranking when the API call fails.

        Args:
            results (List[Dict[str, Any]]): Original search results.
            top_k (int): Number of top results to return.

        Returns:
            List[Dict[str, Any]]: List of results with default scoring.
        """
        logger.info("Using default ranking for results")

        # Limit to top_k results
        result_docs = results[:top_k]

        # Ensure all results have a rerank_score field
        for idx, result in enumerate(result_docs):
            result["rerank_score"] = result.get("score", 1.0 - (idx * 0.1))

        return result_docs

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1, max=30, jitter=0.1),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        reraise=True,
    )
    async def _rerank_with_retry(
        self,
        payload: Dict[str, Any],
        results: List[Dict[str, Any]],
        top_k: int,
        start_time: float,
    ) -> List[Dict[str, Any]]:
        """Make reranking API request with retry logic."""
        try:
            async with self.session.post(self.url, json=payload) as response:
                if response.status == 429:
                    # Rate limit - should retry
                    raise RerankerServiceUnavailable("Rate limit exceeded")
                elif response.status >= 500:
                    # Server error - should retry
                    error_text = await response.text()
                    raise RerankerServiceUnavailable(
                        f"Server error {response.status}: {error_text}"
                    )
                elif response.status != 200:
                    # Client error - should not retry
                    error_text = await response.text()
                    raise RerankerError(f"Client error {response.status}: {error_text}")

                data = await response.json()

                logger.debug(
                    f"Jina API response received in {time.time() - start_time:.2f}s"
                )

                # Validate response structure
                if "results" not in data:
                    raise RerankerError(
                        "Invalid response format: missing 'results' field"
                    )

                if not isinstance(data["results"], list):
                    raise RerankerError(
                        "Invalid response format: 'results' is not a list"
                    )

                # Process reranked results
                reranked_results = []
                for item in data["results"]:
                    if "index" not in item or "relevance_score" not in item:
                        logger.warning(f"Skipping invalid result item: {item}")
                        continue

                    idx = item["index"]
                    if idx < 0 or idx >= len(results):
                        logger.warning(f"Invalid result index: {idx}")
                        continue

                    result = results[idx].copy()
                    result["original_score"] = result["score"]
                    result["score"] = float(item["relevance_score"])
                    result["rerank_score"] = float(item["relevance_score"])
                    reranked_results.append(result)

                logger.info(f"Reranking completed in {time.time() - start_time:.2f}s")
                return reranked_results[:top_k]

        except aiohttp.ClientError as e:
            logger.error(f"Reranker API request error: {e}")
            raise RerankerServiceUnavailable(f"Request failed: {e}") from e
        except asyncio.TimeoutError as e:
            logger.error(f"Reranker API timeout after {time.time() - start_time:.2f}s")
            raise RerankerServiceUnavailable("Request timeout") from e

    async def close(self) -> None:
        """Clean up resources."""
        if self.session and not self.session.closed:
            await self.session.close()
