"""Hybrid retriever for Oh My Repos.

This module provides hybrid search functionality combining dense (vector) and sparse (BM25) retrieval.
"""

import logging
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

# BM25 implementation
from rank_bm25 import BM25Okapi
from rank_bm25 import BM25Plus
import numpy as np

from src.config import settings
from src.core.storage import QdrantStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid retriever combining vector search and BM25 for repository search.

    This class provides methods to search repositories using both dense and sparse retrieval
    methods, combining the results for better recall and precision.
    """

    def __init__(
        self,
        qdrant_store: Optional[QdrantStore] = None,
        repos_json_path: str = "repos.json",
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
        bm25_variant: str = "plus",
        merge_strategy: str = "rrf",  # 'linear' or 'rrf'
        rrf_k: int = 60,
    ) -> None:
        """Initialize the hybrid retriever.

        Args:
            qdrant_store: Qdrant store for vector search
            repos_json_path: Path to the repos.json file for BM25 indexing
            bm25_weight: Weight for BM25 scores in the final ranking (0.0 to 1.0)
            vector_weight: Weight for vector scores in the final ranking (0.0 to 1.0)
        """
        self.qdrant_store = qdrant_store
        self.repos_json_path = Path(repos_json_path)
        self.bm25_weight = bm25_weight
        self.vector_weight = vector_weight

        # Validate weights
        if not 0.0 <= bm25_weight <= 1.0 or not 0.0 <= vector_weight <= 1.0:
            raise ValueError("Weights must be between 0.0 and 1.0")
        if abs(bm25_weight + vector_weight - 1.0) > 1e-6:
            logger.warning("Weights don't sum to 1.0, normalizing...")
            total = bm25_weight + vector_weight
            self.bm25_weight /= total
            self.vector_weight /= total

        # BM25 index
        self.bm25_index = None
        self.repo_data = []
        self.repo_corpus = []

        self._bm25_variant = bm25_variant.lower() if isinstance(bm25_variant, str) else "plus"
        self.merge_strategy = merge_strategy.lower() if isinstance(merge_strategy, str) else "rrf"
        self.rrf_k = rrf_k

        logger.debug("Initializing HybridRetriever")

    async def initialize(self) -> None:
        """Initialize the retriever by loading repository data and creating BM25 index."""
        # Initialize Qdrant store only when configured
        if self.qdrant_store is None:
            try:
                if settings.qdrant:
                    self.qdrant_store = QdrantStore()
                    await self.qdrant_store.initialize()
                else:
                    logger.warning(
                        "Qdrant not configured; vector search will be disabled"
                    )
            except Exception as e:
                logger.warning(
                    "Qdrant initialization failed (%s); falling back to BM25-only",
                    e,
                )
                self.qdrant_store = None
        else:
            # Ensure provided store is initialized
            try:
                await self.qdrant_store.initialize()
            except Exception as e:
                logger.warning(
                    "Provided Qdrant store failed to initialize (%s); disabling vector search",
                    e,
                )
                self.qdrant_store = None

        # Load repository data for BM25 indexing
        await self._load_repo_data()

        # Create BM25 index
        self._create_bm25_index()

        logger.info("HybridRetriever initialized")

    async def _load_repo_data(self) -> None:
        """Load repository data from repos.json."""
        if not self.repos_json_path.exists():
            # Fallback to project root `repos.json`
            fallback = Path(__file__).resolve().parents[1] / "repos.json"
            if fallback.exists():
                self.repos_json_path = fallback
            else:
                logger.error(
                    f"Repository data file not found: {self.repos_json_path}"
                )
                return

        try:
            with open(self.repos_json_path, "r", encoding="utf-8") as f:
                self.repo_data = json.load(f)

            logger.info(
                f"Loaded {len(self.repo_data)} repositories from {self.repos_json_path}"
            )

            # Prepare corpus for BM25 indexing
            self.repo_corpus = []
            for repo in self.repo_data:
                # Combine relevant fields for indexing (robust to schema variants)
                name = (
                    repo.get("repo_name")
                    or repo.get("full_name")
                    or repo.get("name", "")
                )
                description = repo.get("summary") or repo.get("description", "")
                language = repo.get("language", "")
                text = f"{name} {description} {language}"

                # Add tags if available
                tags = repo.get("tags") or repo.get("topics", [])
                if isinstance(tags, list) and tags:
                    text += " " + " ".join(tags)

                # Add summary if available
                if repo.get("summary"):
                    text += " " + str(repo.get("summary"))

                # Tokenize text with regex for better normalization
                import re
                tokens = [t for t in re.findall(r"\w+", text.lower()) if t]
                self.repo_corpus.append(tokens)

        except Exception as e:
            logger.error(f"Error loading repository data: {e}")
            self.repo_data = []
            self.repo_corpus = []

    def _create_bm25_index(self) -> None:
        """Create BM25 index from repository corpus."""
        if not self.repo_corpus:
            logger.warning("Empty repository corpus, BM25 index not created")
            return

        try:
            if self._bm25_variant == "plus":
                self.bm25_index = BM25Plus(self.repo_corpus)
            else:
                self.bm25_index = BM25Okapi(self.repo_corpus)
            logger.info("BM25 index created successfully (%s)", self._bm25_variant)
        except Exception as e:
            logger.error(f"Error creating BM25 index: {e}")
            self.bm25_index = None

    async def search(
        self,
        query: str,
        limit: int = 25,
        filter_tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for repositories using hybrid retrieval.

        Args:
            query: Search query
            limit: Maximum number of results to return
            filter_tags: Optional list of tags to filter by

        Returns:
            List of repository results sorted by combined score
        """
        logger.info(f"Searching for: '{query}' with limit={limit}")
        
        vector_results = await self._vector_search(
            query, limit=limit * 2, filter_tags=filter_tags
        )
        logger.info(f"Vector search returned {len(vector_results)} results")
        
        bm25_results = await self._bm25_search(query, limit=limit * 2)
        logger.info(f"BM25 search returned {len(bm25_results)} results")
        
        # Combine results
        combined_results = self._combine_results(vector_results, bm25_results, limit)
        logger.info(f"Combined results: {len(combined_results)} items")
        
        return combined_results

    async def _vector_search(
        self,
        query: str,
        limit: int = 25,
        filter_tags: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform vector search using Qdrant.

        Args:
            query: Search query
            limit: Maximum number of results
            filter_tags: Optional list of tags to filter by

        Returns:
            List of search results with scores
        """
        if not self.qdrant_store:
            return []

        try:
            # Здесь запрос пользователя целиком векторизуется через Jina API
            # и затем используется для поиска похожих репозиториев в Qdrant
            results = await self.qdrant_store.search(
                query=query,  # Текстовый запрос передается в search, где он будет векторизован
                limit=limit,
                filter_tags=filter_tags,
            )
            
            # Логирование для диагностики
            if not results:
                logger.warning(f"Vector search returned no results for query: '{query}'")
            else:
                logger.debug(f"Vector search scores: {[r.get('score', 0) for r in results[:3]]}")
                
            return results
        except Exception as e:
            logger.error(f"Error in vector search: {e}")
            return []

    async def _bm25_search(self, query: str, limit: int = 25) -> List[Dict[str, Any]]:
        """Perform BM25 search on repository corpus.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of search results with scores
        """
        if not self.bm25_index or not self.repo_data:
            logger.warning("BM25 index not available")
            return []

        try:
            # Улучшенная токенизация запроса
            import re
            query_tokens = [t for t in re.findall(r"\w+", query.lower()) if t]

            if not query_tokens:
                return []

            # Получить BM25-оценки для всего запроса
            all_scores = self.bm25_index.get_scores(query_tokens)

            # Получить топ индексы
            top_indices = np.argsort(all_scores)[::-1][:limit]

            # Подготовить результаты (нормализуем схему полей)
            results: List[Dict[str, Any]] = []
            for idx in top_indices:
                if idx < len(self.repo_data) and float(all_scores[idx]) > 0.0:
                    repo = self.repo_data[idx]
                    repo_name = (
                        repo.get("repo_name")
                        or repo.get("full_name")
                        or repo.get("name", "")
                    )
                    repo_url = (
                        repo.get("repo_url")
                        or repo.get("html_url")
                        or repo.get("url", "")
                    )
                    summary = repo.get("summary") or repo.get("description", "")
                    tags = repo.get("tags") or repo.get("topics", [])
                    stars = repo.get("stars", repo.get("stargazers_count", 0))

                    results.append(
                        {
                            "repo_name": repo_name,
                            "repo_url": repo_url,
                            "summary": summary,
                            "tags": tags if isinstance(tags, list) else [],
                            "language": repo.get("language"),
                            "stars": stars,
                            "score": float(all_scores[idx]),
                            "bm25_score": float(all_scores[idx]),
                            "vector_score": 0.0,
                        }
                    )

            return results

        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []

    def _combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        """Combine vector and BM25 search results.

        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            limit: Maximum number of results to return

        Returns:
            Combined and ranked results
        """
        # Если один из методов не дал результатов, используем результаты другого
        if not vector_results and bm25_results:
            return bm25_results[:limit]
        if not bm25_results and vector_results:
            return vector_results[:limit]
        if not vector_results and not bm25_results:
            return []
            
        if self.merge_strategy == "rrf":
            # Apply Reciprocal Rank Fusion over individual ranked lists
            ranked_lists = [
                sorted(vector_results, key=lambda x: x["score"], reverse=True),
                sorted(bm25_results, key=lambda x: x["score"], reverse=True),
            ]

            scores: Dict[str, Dict[str, Any]] = {}
            vector_names = {r.get("repo_name") for r in vector_results}
            for lst in ranked_lists:
                for rank, res in enumerate(lst):
                    rr = 1.0 / (self.rrf_k + rank + 1)
                    repo_name = res["repo_name"]
                    if repo_name not in scores:
                        # Start aggregation record
                        scores[repo_name] = {
                            **res,
                            "score": 0.0,
                            "vector_score": 0.0,
                            "bm25_score": 0.0,
                        }
                    scores[repo_name]["score"] += rr
                    if repo_name in vector_names:
                        scores[repo_name]["vector_score"] = res["score"]
                    else:
                        scores[repo_name]["bm25_score"] = res["score"]

            fused = list(scores.values())
            fused.sort(key=lambda x: x["score"], reverse=True)
            return fused[:limit]

        # Fallback to linear blend (existing behaviour)
        combined_map = {}

        # Process vector results
        for result in vector_results:
            repo_name = result["repo_name"]
            score = result["score"] * self.vector_weight
            combined_map[repo_name] = {
                **result,
                "score": score,
                "vector_score": result["score"],
                "bm25_score": 0.0,
            }

        max_bm25_score = max((r["score"] for r in bm25_results), default=1.0)

        for result in bm25_results:
            repo_name = result["repo_name"]
            normalized_bm25_score = result["score"] / max_bm25_score
            weighted_bm25_score = normalized_bm25_score * self.bm25_weight
            if repo_name in combined_map:
                combined_map[repo_name]["score"] += weighted_bm25_score
                combined_map[repo_name]["bm25_score"] = normalized_bm25_score
            else:
                combined_map[repo_name] = {
                    **result,
                    "score": weighted_bm25_score,
                    "vector_score": 0.0,
                    "bm25_score": normalized_bm25_score,
                }

        combined_results = list(combined_map.values())
        combined_results.sort(key=lambda x: x["score"], reverse=True)
        return combined_results[:limit]

    async def close(self) -> None:
        """Close connections."""
        if self.qdrant_store:
            await self.qdrant_store.close()
