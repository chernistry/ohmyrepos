"""Pydantic models for validation and type safety.

This module defines data models used throughout the Oh My Repos application
with comprehensive validation and type safety.
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class EmbeddingModel(str, Enum):
    """Supported embedding models."""

    JINA_V3 = "jina-embeddings-v3"
    JINA_V2 = "jina-embeddings-v2"


class RerankerModel(str, Enum):
    """Supported reranker models."""

    JINA_M0 = "jina-reranker-m0"


class BM25Variant(str, Enum):
    """Supported BM25 variants."""

    OKAPI = "okapi"
    PLUS = "plus"


class MergeStrategy(str, Enum):
    """Supported result merging strategies."""

    LINEAR = "linear"
    RRF = "rrf"


class EmbeddingRequest(BaseModel):
    """Request model for embedding operations."""

    texts: List[str] = Field(..., min_items=1, max_items=100)
    model: EmbeddingModel = Field(default=EmbeddingModel.JINA_V3)

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, v):
        """Validate text inputs."""
        for i, text in enumerate(v):
            if not isinstance(text, str):
                raise ValueError(f"Text at index {i} must be a string")

            text_stripped = text.strip()
            if len(text_stripped) == 0:
                raise ValueError(f"Text at index {i} cannot be empty")

            if len(text) > 8192:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length of 8192 characters"
                )

        return v


class EmbeddingResponse(BaseModel):
    """Response model for embedding operations."""

    embeddings: List[List[float]] = Field(..., min_items=1)
    model: str = Field(...)
    dimension: int = Field(..., gt=0)

    @field_validator("embeddings")
    @classmethod
    def validate_embeddings(cls, v, info):
        """Validate embedding vectors."""
        if not v:
            raise ValueError("At least one embedding is required")

        # Check dimension consistency
        expected_dim = info.data.get("dimension") if info.data else None
        if expected_dim:
            for i, embedding in enumerate(v):
                if len(embedding) != expected_dim:
                    raise ValueError(
                        f"Embedding {i} has dimension {len(embedding)}, "
                        f"expected {expected_dim}"
                    )

        return v


class RerankerRequest(BaseModel):
    """Request model for reranking operations."""

    query: str = Field(..., min_length=1, max_length=1000)
    documents: List[str] = Field(..., min_items=1, max_items=1000)
    model: RerankerModel = Field(default=RerankerModel.JINA_M0)
    top_n: int = Field(default=25, ge=1, le=1000)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate query string."""
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @field_validator("documents")
    @classmethod
    def validate_documents(cls, v):
        """Validate document list."""
        for i, doc in enumerate(v):
            if not isinstance(doc, str):
                raise ValueError(f"Document at index {i} must be a string")
            if not doc.strip():
                raise ValueError(f"Document at index {i} cannot be empty")
        return v


class RerankerResponse(BaseModel):
    """Response model for reranking operations."""

    results: List[Dict[str, Union[int, float]]] = Field(...)
    model: str = Field(...)

    @field_validator("results")
    @classmethod
    def validate_results(cls, v):
        """Validate reranking results."""
        for i, result in enumerate(v):
            if "index" not in result:
                raise ValueError(f"Result {i} missing 'index' field")
            if "relevance_score" not in result:
                raise ValueError(f"Result {i} missing 'relevance_score' field")

            if not isinstance(result["index"], int) or result["index"] < 0:
                raise ValueError(f"Result {i} has invalid index")

            if not isinstance(result["relevance_score"], (int, float)):
                raise ValueError(f"Result {i} has invalid relevance_score")

        return v


class SearchRequest(BaseModel):
    """Request model for search operations."""

    query: str = Field(..., min_length=1, max_length=500)
    limit: int = Field(default=10, ge=1, le=100)
    filter_tags: Optional[List[str]] = Field(None, max_items=20)
    bm25_weight: Optional[float] = Field(default=0.4, ge=0.0, le=1.0)
    vector_weight: Optional[float] = Field(default=0.6, ge=0.0, le=1.0)
    bm25_variant: BM25Variant = Field(default=BM25Variant.PLUS)
    merge_strategy: MergeStrategy = Field(default=MergeStrategy.RRF)

    @field_validator("query")
    @classmethod
    def validate_query(cls, v):
        """Validate search query."""
        # Remove extra whitespace
        query = re.sub(r"\s+", " ", v.strip())

        if not query:
            raise ValueError("Query cannot be empty")

        # Basic injection prevention
        dangerous_patterns = [
            "<script",
            "javascript:",
            "vbscript:",
            "onload=",
            "onerror=",
        ]
        query_lower = query.lower()
        for pattern in dangerous_patterns:
            if pattern in query_lower:
                raise ValueError("Query contains potentially dangerous content")

        return query

    @field_validator("filter_tags")
    @classmethod
    def validate_filter_tags(cls, v):
        """Validate filter tags."""
        if v is None:
            return v

        for i, tag in enumerate(v):
            if not isinstance(tag, str):
                raise ValueError(f"Tag at index {i} must be a string")
            if not tag.strip():
                raise ValueError(f"Tag at index {i} cannot be empty")
            if len(tag) > 50:
                raise ValueError(
                    f"Tag at index {i} exceeds maximum length of 50 characters"
                )

        # Remove duplicates while preserving order
        return list(dict.fromkeys(v))

    @model_validator(mode="after")
    def validate_weights(self) -> "SearchRequest":
        """Validate that weights sum to approximately 1.0."""
        if abs(self.bm25_weight + self.vector_weight - 1.0) > 0.01:
            raise ValueError("BM25 and vector weights must sum to 1.0")

        return self


class RepositoryData(BaseModel):
    """Model for repository data."""

    repo_name: str = Field(..., min_length=1, max_length=200)
    repo_url: str = Field(...)
    summary: Optional[str] = Field(None, max_length=2000)
    description: Optional[str] = Field(None, max_length=1000)
    tags: List[str] = Field(default_factory=list, max_items=20)
    language: Optional[str] = Field(None, max_length=50)
    stars: int = Field(default=0, ge=0)
    forks: int = Field(default=0, ge=0)
    created_at: Optional[str] = Field(None)
    updated_at: Optional[str] = Field(None)

    @field_validator("repo_url")
    @classmethod
    def validate_repo_url(cls, v):
        """Validate repository URL."""
        url_pattern = re.compile(
            r"^https?://"  # http:// or https://
            r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|"  # domain...
            r"localhost|"  # localhost...
            r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
            r"(?::\d+)?"  # optional port
            r"(?:/?|[/?]\S+)$",
            re.IGNORECASE,
        )

        if not url_pattern.match(v):
            raise ValueError("Invalid repository URL format")

        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        """Validate repository tags."""
        if not v:
            return v

        for i, tag in enumerate(v):
            if not isinstance(tag, str):
                raise ValueError(f"Tag at index {i} must be a string")
            if not tag.strip():
                raise ValueError(f"Tag at index {i} cannot be empty")
            if len(tag) > 50:
                raise ValueError(f"Tag at index {i} exceeds maximum length")

        # Remove duplicates and empty tags
        return list(dict.fromkeys([tag.strip() for tag in v if tag.strip()]))


class SearchResult(BaseModel):
    """Model for search results."""

    repo_name: str = Field(...)
    repo_url: str = Field(...)
    summary: Optional[str] = Field(None)
    tags: List[str] = Field(default_factory=list)
    language: Optional[str] = Field(None)
    stars: int = Field(default=0, ge=0)
    score: float = Field(..., ge=0.0)
    vector_score: Optional[float] = Field(None, ge=0.0)
    bm25_score: Optional[float] = Field(None, ge=0.0)
    rerank_score: Optional[float] = Field(None, ge=0.0)
    original_score: Optional[float] = Field(None, ge=0.0)


class HealthCheckResult(BaseModel):
    """Model for health check results."""

    status: str = Field(..., pattern=r"^(healthy|unhealthy)$")
    checks: Dict[str, Dict[str, Any]] = Field(...)
    timestamp: float = Field(..., gt=0)


class BatchProcessingStatus(BaseModel):
    """Model for batch processing status."""

    total_items: int = Field(..., ge=0)
    processed_items: int = Field(..., ge=0)
    failed_items: int = Field(default=0, ge=0)
    success_rate: float = Field(..., ge=0.0, le=1.0)
    start_time: float = Field(..., gt=0)
    end_time: Optional[float] = Field(None)
    errors: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_processing_status(self) -> "BatchProcessingStatus":
        """Validate processing status consistency."""
        if self.processed_items > self.total_items:
            raise ValueError("Processed items cannot exceed total items")

        if self.failed_items > self.processed_items:
            raise ValueError("Failed items cannot exceed processed items")

        # Calculate success rate
        if self.processed_items > 0:
            self.success_rate = (self.processed_items - self.failed_items) / self.processed_items
        else:
            self.success_rate = 0.0

        return self


class ConfigurationError(Exception):
    """Exception raised for configuration errors."""

    pass


class ValidationError(Exception):
    """Exception raised for validation errors."""

    pass
