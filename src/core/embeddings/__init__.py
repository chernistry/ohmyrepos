"""Embeddings module for Oh My Repos.

This module provides functionality for generating embeddings.
"""

from src.core.embeddings.base import EmbeddingProvider, EmbeddingConfig
from src.core.embeddings.factory import EmbeddingFactory

__all__ = ["EmbeddingProvider", "EmbeddingConfig", "EmbeddingFactory"]
