"""Search endpoints."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.config import settings
from src.core.retriever import HybridRetriever
from src.core.storage import QdrantStore

router = APIRouter()


class SearchRequest(BaseModel):
    """Search request model."""

    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=25, ge=1, le=100)
    offset: int = Field(default=0, ge=0)
    filter_tags: Optional[List[str]] = Field(default=None)


class RepoResult(BaseModel):
    """Repository search result."""

    repo_name: str
    full_name: str
    description: Optional[str]
    summary: Optional[str]
    tags: List[str]
    language: Optional[str]
    stars: int
    url: str
    score: float


class SearchResponse(BaseModel):
    """Search response model."""

    query: str
    results: List[RepoResult]
    total: int


@router.post("/search", response_model=SearchResponse, status_code=status.HTTP_200_OK)
async def search(request: SearchRequest):
    """Search repositories using hybrid retrieval."""
    if not settings.qdrant:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search service not configured",
        )

    try:
        # Initialize retriever
        qdrant_store = QdrantStore()
        retriever = HybridRetriever(
            qdrant_store=qdrant_store,
            bm25_weight=settings.search.bm25_weight,
            vector_weight=settings.search.vector_weight,
            bm25_variant=settings.search.bm25_variant,
        )
        await retriever.initialize()

        # Perform search
        results = await retriever.search(
            query=request.query,
            limit=request.limit,
            filter_tags=request.filter_tags,
        )

        # Convert to response model
        repo_results = [
            RepoResult(
                repo_name=r.get("repo_name", ""),
                full_name=r.get("full_name", r.get("repo_name", "")),
                description=r.get("description"),
                summary=r.get("summary"),
                tags=r.get("tags", []),
                language=r.get("language"),
                stars=r.get("stars", 0),
                url=r.get("url", ""),
                score=r.get("score", 0.0),
            )
            for r in results
        ]

        return SearchResponse(
            query=request.query,
            results=repo_results,
            total=len(repo_results),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )
