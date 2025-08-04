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

# Fix imports for compatibility
try:
    from src.config import settings
    from src.llm.providers import get_provider, BaseLLMProvider
    from src.llm.providers.base import ChatCompletionRequest, ChatMessage, ChatCompletionResponse, StreamingChunk
except ImportError:
    from config import settings
    from llm.providers import get_provider
    from llm.providers.base import ChatCompletionRequest, ChatMessage, ChatCompletionResponse, StreamingChunk

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
        self._provider = get_provider(self.provider_name, **provider_params)

        # Store default model for later use
        self.default_model = model or (
            settings.OLLAMA_MODEL
            if self.provider_name == "ollama"
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

        # Convert dict payload to ChatCompletionRequest
        messages = []
        for msg in payload.get("messages", []):
            messages.append(ChatMessage(
                role=msg["role"],
                content=msg["content"]
            ))
        
        request = ChatCompletionRequest(
            messages=messages,
            model=payload["model"],
            temperature=payload.get("temperature", 0.1),
            max_tokens=payload.get("max_tokens", 1000),
            top_p=payload.get("top_p", 1.0),
            frequency_penalty=payload.get("frequency_penalty", 0.0),
            presence_penalty=payload.get("presence_penalty", 0.0),
            stop=payload.get("stop"),
            stream=payload.get("stream", False),
            n=payload.get("n", 1)
        )

        result = await self._provider.chat_completion(request)
        
        # Handle streaming vs non-streaming responses
        if isinstance(result, ChatCompletionResponse):
            # Convert to dict format for backward compatibility
            return {
                "id": result.id,
                "object": "chat.completion",
                "created": result.created,
                "model": result.model,
                "choices": [
                    {
                        "index": choice.index,
                        "message": {
                            "role": choice.message.role,
                            "content": choice.message.content
                        },
                        "finish_reason": choice.finish_reason
                    }
                    for choice in result.choices
                ],
                "usage": {
                    "prompt_tokens": result.usage.prompt_tokens if result.usage else 0,
                    "completion_tokens": result.usage.completion_tokens if result.usage else 0,
                    "total_tokens": result.usage.total_tokens if result.usage else 0
                } if result.usage else None
            }
        else:
            # Return async generator as-is for streaming
            return result

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
