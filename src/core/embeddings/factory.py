"""Embedding provider factory.

This module provides a factory for creating embedding providers.
"""

import logging
from typing import Dict, Type, Optional

from src.config import settings
from src.core.embeddings.base import EmbeddingProvider, EmbeddingConfig
from src.core.embeddings.providers.jina import JinaEmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Factory for creating embedding providers.

    This class provides a factory for creating embedding providers based on configuration.
    """

    # Registry of available embedding providers
    _providers: Dict[str, Type[EmbeddingProvider]] = {
        "jina-embeddings-v3": JinaEmbeddingProvider,
        "jina-embeddings-v2": JinaEmbeddingProvider,
    }

    @classmethod
    def get_provider(
        cls,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        **kwargs,
    ) -> EmbeddingProvider:
        """Get an embedding provider instance.

        Args:
            model_name: Name of the embedding model
            api_key: API key for the provider
            api_url: API URL for the provider
            **kwargs: Additional configuration parameters

        Returns:
            An embedding provider instance

        Raises:
            ValueError: If the provider is not supported
        """
        # Use settings as defaults if not provided
        model_name = model_name or settings.EMBEDDING_MODEL
        api_key = api_key or settings.EMBEDDING_MODEL_API_KEY
        api_url = api_url or settings.EMBEDDING_MODEL_URL

        # Find provider class based on model name
        provider_class = None
        for prefix, cls in cls._providers.items():
            if model_name.startswith(prefix):
                provider_class = cls
                break

        if not provider_class:
            raise ValueError(f"Unsupported embedding model: {model_name}")

        # Create configuration
        config = EmbeddingConfig(
            model_name=model_name,
            api_key=api_key,
            api_url=api_url,
            **kwargs,
        )

        # Create provider instance
        provider = provider_class(config=config)
        logger.debug(f"Created embedding provider for model: {model_name}")

        return provider

    @classmethod
    def register_provider(
        cls, model_prefix: str, provider_class: Type[EmbeddingProvider]
    ) -> None:
        """Register a new embedding provider.

        Args:
            model_prefix: Prefix for model names that this provider supports
            provider_class: Provider class to register
        """
        cls._providers[model_prefix] = provider_class
        logger.debug(f"Registered embedding provider for prefix: {model_prefix}")
