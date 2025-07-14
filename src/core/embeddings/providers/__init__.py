"""Embedding providers for Oh My Repos.

This module provides embedding providers for different models.
"""

from src.core.embeddings.providers.jina import JinaEmbeddingProvider

__all__ = ["JinaEmbeddingProvider"]
