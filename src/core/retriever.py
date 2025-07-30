"""Hybrid retriever for Oh My Repos.

This module provides hybrid search functionality combining dense (vector) and sparse (BM25) retrieval.
"""

import logging
from typing import Dict, List, Optional, Any, Union
import json
import os
from pathlib import Path
import asyncio

# BM25 implementation
from rank_bm25 import BM25Okapi
from rank_bm25 import BM25Plus
import numpy as np

# Fix imports for compatibility
try:
    from src.config import settings
    from src.core.storage import QdrantStore
except ImportError:
    from config import settings
    from core.storage import QdrantStore

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
        bm25_weight: float = 0.3,
        vector_weight: float = 0.6,
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
        
        self._bm25_variant = bm25_variant.lower()
        self.merge_strategy = merge_strategy.lower()
        self.rrf_k = rrf_k
        
        logger.debug("Initializing HybridRetriever")
    
    async def initialize(self) -> None:
        """Initialize the retriever by loading repository data and creating BM25 index."""
        # Initialize Qdrant store if not provided
        if self.qdrant_store is None:
            self.qdrant_store = QdrantStore()
            await self.qdrant_store.setup_collection()
        
        # Load repository data for BM25 indexing
        await self._load_repo_data()
        
        # Create BM25 index
        self._create_bm25_index()
        
        logger.info("HybridRetriever initialized")
    
    async def _load_repo_data(self) -> None:
        """Load repository data from repos.json."""
        if not self.repos_json_path.exists():
            logger.error(f"Repository data file not found: {self.repos_json_path}")
            return
        
        try:
            with open(self.repos_json_path, "r", encoding="utf-8") as f:
                self.repo_data = json.load(f)
            
            logger.info(f"Loaded {len(self.repo_data)} repositories from {self.repos_json_path}")
            
            # Prepare corpus for BM25 indexing
            self.repo_corpus = []
            for repo in self.repo_data:
                # Combine name, description and other relevant fields for indexing
                text = f"{repo.get('full_name', '')} {repo.get('description', '')} {repo.get('language', '')}"
                
                # Add tags if available
                if "tags" in repo and isinstance(repo["tags"], list):
                    text += " " + " ".join(repo["tags"])
                    
                # Add summary if available
                if "summary" in repo and repo["summary"]:
                    text += " " + repo["summary"]
                
                # Tokenize text
                tokens = text.lower().split()
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
        vector_results = await self._vector_search(query, limit=limit*2, filter_tags=filter_tags)
        bm25_results = await self._bm25_search(query, limit=limit*2)
        
        # Combine results
        combined_results = self._combine_results(vector_results, bm25_results, limit)
        
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
        try:
            results = await self.qdrant_store.search(
                query=query,
                limit=limit,
                filter_tags=filter_tags,
            )
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
            # Tokenize query
            query_tokens = query.lower().split()
            
            # Get BM25 scores
            bm25_scores = self.bm25_index.get_scores(query_tokens)
            
            # Get top indices
            top_indices = np.argsort(bm25_scores)[::-1][:limit]
            
            # Prepare results
            results = []
            for idx in top_indices:
                if idx < len(self.repo_data):
                    repo = self.repo_data[idx]
                    results.append({
                        "repo_name": repo.get("full_name", ""),
                        "repo_url": repo.get("html_url", ""),
                        "summary": repo.get("description", ""),
                        "tags": repo.get("tags", []),
                        "language": repo.get("language"),
                        "stars": repo.get("stargazers_count", 0),
                        "score": float(bm25_scores[idx]),
                    })
            
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
        if self.merge_strategy == "rrf":
            # Apply Reciprocal Rank Fusion over individual ranked lists
            ranked_lists = [
                sorted(vector_results, key=lambda x: x["score"], reverse=True),
                sorted(bm25_results, key=lambda x: x["score"], reverse=True),
            ]

            scores: Dict[str, Dict[str, Any]] = {}
            for lst in ranked_lists:
                for rank, res in enumerate(lst):
                    rr = 1.0 / (self.rrf_k + rank + 1)
                    repo_name = res["repo_name"]
                    if repo_name not in scores:
                        # Start aggregation record
                        scores[repo_name] = {**res, "score": 0.0, "vector_score": 0.0, "bm25_score": 0.0}
                    scores[repo_name]["score"] += rr
                    if res in vector_results:
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
