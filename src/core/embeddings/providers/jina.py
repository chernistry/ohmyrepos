"""Jina AI embedding provider.

This module provides an embedding provider using Jina AI's embedding models.
"""

import logging
from typing import List, Dict, Any, Optional

import httpx

# Исправляем импорты для совместимости
try:
    from src.config import settings
    from src.core.embeddings.base import EmbeddingProvider, EmbeddingConfig
except ImportError:
    from config import settings
    from core.embeddings.base import EmbeddingProvider, EmbeddingConfig

logger = logging.getLogger(__name__)


class JinaEmbeddingProvider(EmbeddingProvider):
    """Jina AI embedding provider.
    
    This class provides embeddings using Jina AI's embedding models.
    """
    
    def __init__(
        self,
        config: Optional[EmbeddingConfig] = None,
    ) -> None:
        """Initialize the Jina AI embedding provider.
        
        Args:
            config: Configuration for the provider
        """
        self.config = config or EmbeddingConfig(
            model_name=settings.EMBEDDING_MODEL,
            api_key=settings.EMBEDDING_MODEL_API_KEY,
            api_url=settings.EMBEDDING_MODEL_URL,
        )
        
        self.api_url = self.config.api_url or "https://api.jina.ai/v1/embeddings"
        self.api_key = self.config.api_key
        self.model_name = self.config.model_name
        
        if not self.api_key:
            raise ValueError("Jina AI API key is required")
        
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Set default dimension based on model
        if self.config.dimensions is not None:
            self._dimension = self.config.dimensions
        elif "v3" in self.model_name:
            self._dimension = 1024  # jina-embeddings-v3 uses 1024 dimensions
        else:
            self._dimension = 768  # jina-embeddings-v2 uses 768 dimensions
            
        logger.debug(f"Initialized JinaEmbeddingProvider with model: {self.model_name} (dimension: {self._dimension})")
    
    async def embed_query(self, text: str) -> List[float]:
        """Embed a query text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
        """
        embeddings = await self.embed_documents([text])
        return embeddings[0]
    
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Prepare payload
        payload = {
            "model": self.model_name,
            "input": texts,
        }
        
        try:
            response = await self.client.post(
                self.api_url,
                headers=self.headers,
                json=payload,
            )
            response.raise_for_status()
            
            result = response.json()
            embeddings = [item["embedding"] for item in result["data"]]
            
            # Update dimension if not set
            if self.config.dimensions is None and embeddings:
                self._dimension = len(embeddings[0])
                logger.info(f"Detected embedding dimension: {self._dimension}")
            
            # Проверка размерности всех векторов
            if embeddings:
                for i, emb in enumerate(embeddings):
                    if len(emb) != self._dimension:
                        logger.error(f"Embedding dimension mismatch: expected {self._dimension}, got {len(emb)} for text {i}")
            
            return embeddings
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self.client.aclose()
    
    @property
    def dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self._dimension
