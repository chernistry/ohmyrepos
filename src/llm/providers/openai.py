"""OpenAI-compatible chat provider."""

import logging
import random
import string
from functools import lru_cache
from typing import Dict, Any, Union, AsyncGenerator, Optional

import httpx
from tenacity import AsyncRetrying, stop_after_attempt, wait_exponential

from src.config import settings
from src.llm.providers import register_provider
from src.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)


def _random_suffix(length: int = 4) -> str:
    """Generate a random alphanumeric suffix."""
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """OpenAI-compatible chat provider.
    
    This provider works with any API that implements the OpenAI chat completion
    interface, including OpenAI, Azure OpenAI, and many other providers.
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Initialize the OpenAI provider.
        
        Args:
            base_url: API base URL
            api_key: API key
            model: Model name
        """
        self.base_url = (base_url or settings.CHAT_LLM_BASE_URL).rstrip("/")
        self.api_key = api_key or settings.CHAT_LLM_API_KEY
        self.model = model or settings.CHAT_LLM_MODEL

        if not self.api_key:
            raise ValueError("OpenAI API key is required")

        # Single client instance per provider
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        
        logger.debug(
            "Initialized OpenAI provider with base_url=%s, model=%s", 
            self.base_url, 
            self.model
        )
    
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
        stream = bool(payload.get("stream", False))
        
        # Set defaults
        payload.setdefault("model", self.model)
        payload.setdefault("max_tokens", 1024)
        
        headers = self._build_headers()
        
        # Use tenacity for retries
        retryer = AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=8.0),
            reraise=True,
        )
        
        async for attempt in retryer:
            with attempt:
                resp = await self._client.post(
                    "/chat/completions", 
                    json=payload, 
                    headers=headers
                )
                resp.raise_for_status()
        
        if not stream:
            return resp.json()
        
        async def _stream_generator() -> AsyncGenerator[str, None]:
            try:
                async for line in resp.aiter_lines():
                    if line.strip():
                        yield line
            finally:
                await resp.aclose()
        
        return _stream_generator()
    
    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
    
    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for the API request."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        } 