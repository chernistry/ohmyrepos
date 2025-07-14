"""Embeddings module for Oh My Repos.

This module provides functionality for generating embeddings.
"""

# Исправляем импорты для совместимости
try:
    from src.core.embeddings.base import EmbeddingProvider, EmbeddingConfig
    from src.core.embeddings.factory import EmbeddingFactory
except ImportError:
    from core.embeddings.base import EmbeddingProvider, EmbeddingConfig
    from core.embeddings.factory import EmbeddingFactory

__all__ = ["EmbeddingProvider", "EmbeddingConfig", "EmbeddingFactory"]
