"""Embedding provider factory.

This module provides a factory for creating embedding providers.
"""

import logging
import os
from typing import Dict, Type, Optional

from src.config import settings, EmbeddingProviderType
from src.core.embeddings.base import EmbeddingProvider, EmbeddingConfig
from src.core.embeddings.providers.jina import JinaEmbeddingProvider
from src.core.embeddings.providers.ollama import OllamaEmbeddingProvider

logger = logging.getLogger(__name__)


class EmbeddingFactory:
    """Factory for creating embedding providers.

    This class provides a factory for creating embedding providers based on configuration.
    """

    # Registry of available embedding providers
    _providers: Dict[EmbeddingProviderType, Type[EmbeddingProvider]] = {
        EmbeddingProviderType.JINA: JinaEmbeddingProvider,
        EmbeddingProviderType.OLLAMA: OllamaEmbeddingProvider,
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
        # Get configuration from settings if not provided
        config = settings.embedding
        
        if not config:
             raise ValueError("Embedding configuration is missing")

        provider_type = config.provider
        
        # Override with explicit arguments if provided
        if model_name:
            config.model = model_name
        if api_key:
            config.api_key = api_key
        if api_url:
            config.base_url = api_url

        provider_class = cls._providers.get(provider_type)

        if not provider_class:
            raise ValueError(f"Unsupported embedding provider: {provider_type}")

        # Create provider instance
        provider = provider_class(config=config)
        logger.debug(f"Created embedding provider: {provider_type} with model: {config.model}")

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
