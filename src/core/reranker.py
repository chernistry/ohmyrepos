"""Reranking module for Oh My Repos.

This module provides functionality to rerank search results using Jina AI reranker model.

Note:
    Avoid using unescaped backslashes in docstrings (e.g., use `\\(` instead of `\(`).
"""

import logging
import aiohttp
import time
from typing import Dict, List, Optional, Any, Union

# Исправляем импорты для совместимости
try:
    from src.config import settings
except ImportError:
    from config import settings

logger = logging.getLogger(__name__)


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
            
            # Make API request
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    headers=self.headers,
                    json=payload,
                    timeout=self.timeout,
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Reranker API error: {response.status} - {error_text}")
                        return self._default_ranking(results, top_k)
                    
                    data = await response.json()
                    
                    logger.debug(
                        f"Jina API response received in {time.time() - start_time:.2f}s"
                    )
                    
                    # Process reranked results
                    if "results" in data and isinstance(data["results"], list):
                        reranked_results = []
                        for item in data["results"]:
                            idx = item["index"]
                            if idx < len(results):
                                result = results[idx].copy()
                                result["original_score"] = result["score"]
                                result["score"] = float(item["relevance_score"])
                                result["rerank_score"] = float(item["relevance_score"])
                                reranked_results.append(result)
                        
                        logger.info(f"Reranking completed in {time.time() - start_time:.2f}s")
                        return reranked_results[:top_k]
                    else:
                        logger.warning("Invalid response format from reranker API")
                        return self._default_ranking(results, top_k)
        
        except aiohttp.ClientError as e:
            logger.error(f"Reranker API request error: {e}")
            return self._default_ranking(results, top_k)
        except Exception as e:
            logger.error(f"Reranker error: {e}")
            return self._default_ranking(results, top_k)
    
    def _default_ranking(self, results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
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
    
    async def close(self) -> None:
        """Clean up resources."""
        # Nothing to clean up for the HTTP client
        pass
