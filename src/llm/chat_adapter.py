"""Generic OpenAI-compatible chat adapter.

This module provides a thin wrapper around any LLM endpoint that implements
(OpenAI) `/chat/completions` semantics (OpenRouter, Groq, Together, etc.).
It supports both standard JSON responses and streaming event-source format
(`data: ...\n\n`).

Usage
-----
>>> from src.llm.chat_adapter import ChatAdapter
>>> adapter = ChatAdapter()
>>> resp = await adapter.chat_completion({"messages": [...], "stream": False})
"""

import logging
from typing import Any, AsyncGenerator, Dict, Union, Optional

# Исправляем импорты для совместимости
try:
    from src.config import settings
    from src.llm.providers import get_provider, BaseLLMProvider
except ImportError:
    from config import settings
    from llm.providers import get_provider, BaseLLMProvider

logger = logging.getLogger(__name__)

__all__ = ["ChatAdapter", "chat_completion"]


class ChatAdapter:
    """OpenAI-compatible chat client with optional streaming."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        """Initialize the chat adapter.
        
        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
            provider: Provider name (default: from settings)
        """
        self.provider_name = provider or settings.CHAT_LLM_PROVIDER
        
        # Configure provider-specific parameters
        provider_params = {}
        
        if self.provider_name == "openai":
            provider_params = {
                "base_url": base_url or settings.CHAT_LLM_BASE_URL,
                "api_key": api_key or settings.CHAT_LLM_API_KEY,
                "model": model or settings.CHAT_LLM_MODEL,
            }
        elif self.provider_name == "ollama":
            provider_params = {
                "base_url": base_url or settings.OLLAMA_BASE_URL,
                "timeout": settings.OLLAMA_TIMEOUT,
            }
            # For Ollama, we'll set the model in the payload if not specified
        
        # Create provider instance
        self._provider = get_provider(
            self.provider_name,
            **provider_params
        )
        
        # Store default model for later use
        self.default_model = model or (
            settings.OLLAMA_MODEL if self.provider_name == "ollama" 
            else settings.CHAT_LLM_MODEL
        )
        
        logger.debug("Initialized ChatAdapter with provider=%s", self.provider_name)

    async def chat_completion(
        self,
        payload: Dict[str, Any],
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Generate a chat completion.
        
        Args:
            payload: Chat completion payload
            
        Returns:
            Chat completion response or streaming generator
        """
        # Set default model if not provided in payload
        if "model" not in payload:
            payload["model"] = self.default_model
            
        return await self._provider.chat_completion(payload)

    async def close(self) -> None:
        """Close the provider."""
        await self._provider.close()


# Convenience module-level facade -------------------------------------------

_adapter: Optional[ChatAdapter] = None


async def chat_completion(
    payload: Dict[str, Any],
) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
    """Convenience facade that re-uses a singleton ``ChatAdapter``."""
    global _adapter
    if _adapter is None:
        _adapter = ChatAdapter()
    return await _adapter.chat_completion(payload)
