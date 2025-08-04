"""Base classes for embeddings providers.

This module defines the base classes for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional


class EmbeddingProvider(ABC):
    """Base class for embedding providers.

    All embedding providers should inherit from this class.
    """

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """Embed a query text.

        Args:
            text: Text to embed

        Returns:
            List of embedding values
        """
        pass

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources used by the provider."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        pass


class EmbeddingConfig:
    """Configuration for embedding providers."""

    def __init__(
        self,
        model_name: str,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        dimensions: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the embedding configuration.

        Args:
            model_name: Name of the embedding model
            api_key: API key for the provider
            api_url: API URL for the provider
            dimensions: Dimensions of the embeddings
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.dimensions = dimensions
        self.additional_params = kwargs
