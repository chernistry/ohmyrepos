"""Enterprise-grade OpenAI provider with comprehensive error handling and monitoring."""

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from .base import (
    BaseLLMProvider,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionUsage,
    ChatMessage,
    StreamingChunk,
    StreamingChoice,
    HealthCheck,
    LLMError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError,
    TokenLimitError,
    register_provider,
)


@register_provider("openai")
class OpenAIProvider(BaseLLMProvider):
    """Enterprise-grade OpenAI API provider with comprehensive features.

    Features:
    - Automatic retry with exponential backoff
    - Rate limiting and concurrency control
    - Comprehensive error handling and classification
    - Health monitoring and metrics
    - Resource management and cleanup
    - Both streaming and non-streaming support
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout: int = 60,
        max_retries: int = 3,
        max_concurrent: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key
            base_url: API base URL (supports OpenRouter, etc.)
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            max_concurrent: Maximum concurrent requests
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.api_key = api_key
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_concurrent = max_concurrent
        self._client: Optional[httpx.AsyncClient] = None
        self._models_cache: Optional[List[str]] = None
        self._models_cache_time: float = 0
        self._models_cache_ttl: int = 3600  # 1 hour

    async def initialize(self) -> None:
        """Initialize the OpenAI provider and validate configuration."""
        if self._initialized:
            return

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "ohmyrepos/1.0.0",
        }

        # Handle different provider-specific headers
        if "openrouter" in self.base_url.lower():
            headers["HTTP-Referer"] = "https://github.com/chernistry/ohmyrepos"
            headers["X-Title"] = "Oh My Repos"

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_connections=self.max_concurrent),
        )

        # Validate API key by attempting a simple request
        try:
            await self.health_check()
            self._initialized = True
        except Exception as e:
            await self.close()
            raise LLMError(f"Failed to initialize OpenAI provider: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[StreamingChunk, None]]:
        """Generate a chat completion with comprehensive error handling.

        Args:
            request: Strongly typed chat completion request

        Returns:
            Chat completion response or streaming generator

        Raises:
            Various LLMError subclasses based on error type
        """
        if not self._initialized:
            await self.initialize()

        # Convert to OpenAI API format
        payload = {
            "model": request.model,
            "messages": [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "frequency_penalty": request.frequency_penalty,
            "presence_penalty": request.presence_penalty,
            "stream": request.stream,
        }

        if request.stop:
            payload["stop"] = request.stop

        async with self.rate_limited(self.max_concurrent):
            if request.stream:
                return self._stream_completion(payload)
            else:
                return await self._complete_chat(payload)

    async def _complete_chat(self, payload: Dict[str, Any]) -> ChatCompletionResponse:
        """Handle non-streaming chat completion."""
        try:
            response = await self._client.post("/chat/completions", json=payload)
            await self._handle_response_errors(response)

            data = response.json()
            
            # Convert to strongly typed response
            choices = [
                ChatCompletionChoice(
                    index=choice["index"],
                    message=ChatMessage(
                        role=choice["message"]["role"],
                        content=choice["message"]["content"],
                    ),
                    finish_reason=choice.get("finish_reason"),
                )
                for choice in data["choices"]
            ]

            usage = None
            if "usage" in data:
                usage = ChatCompletionUsage(
                    prompt_tokens=data["usage"]["prompt_tokens"],
                    completion_tokens=data["usage"]["completion_tokens"],
                    total_tokens=data["usage"]["total_tokens"],
                )

            return ChatCompletionResponse(
                id=data["id"],
                created=data["created"],
                model=data["model"],
                choices=choices,
                usage=usage,
            )

        except httpx.HTTPStatusError as e:
            await self._handle_response_errors(e.response)
            raise  # This should not be reached due to error handling above

    async def _stream_completion(
        self, payload: Dict[str, Any]
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Handle streaming chat completion."""
        try:
            async with self._client.stream(
                "POST", "/chat/completions", json=payload
            ) as response:
                await self._handle_response_errors(response)

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    chunk_data = line[6:].strip()
                    if chunk_data == "[DONE]":
                        break

                    try:
                        data = json.loads(chunk_data)
                        
                        choices = [
                            StreamingChoice(
                                index=choice["index"],
                                delta=choice["delta"],
                                finish_reason=choice.get("finish_reason"),
                            )
                            for choice in data["choices"]
                        ]

                        yield StreamingChunk(
                            id=data["id"],
                            created=data["created"],
                            model=data["model"],
                            choices=choices,
                        )

                    except json.JSONDecodeError:
                        # Skip malformed chunks
                        continue

        except httpx.HTTPStatusError as e:
            await self._handle_response_errors(e.response)

    async def _handle_response_errors(self, response: httpx.Response) -> None:
        """Handle HTTP response errors with proper classification."""
        if response.is_success:
            return

        status_code = response.status_code
        
        try:
            error_data = response.json()
            error_message = error_data.get("error", {}).get("message", "Unknown error")
            error_type = error_data.get("error", {}).get("type", "unknown")
        except (json.JSONDecodeError, KeyError):
            error_message = f"HTTP {status_code}: {response.text}"
            error_type = "unknown"

        # Classify errors based on status code and type
        if status_code == 401:
            raise AuthenticationError(error_message, status_code)
        elif status_code == 404:
            raise ModelNotFoundError(error_message, status_code)
        elif status_code == 429:
            raise RateLimitError(error_message, status_code)
        elif status_code == 400 and "token" in error_message.lower():
            raise TokenLimitError(error_message, status_code)
        else:
            raise LLMError(error_message, status_code)

    async def health_check(self) -> HealthCheck:
        """Perform comprehensive health check."""
        start_time = time.time()
        
        try:
            if not self._client:
                await self.initialize()

            # Try to list models as a health check
            response = await self._client.get("/models", timeout=10.0)
            response.raise_for_status()

            response_time = (time.time() - start_time) * 1000

            return HealthCheck(
                status="healthy",
                response_time_ms=response_time,
                timestamp=time.time(),
            )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            
            return HealthCheck(
                status="unhealthy",
                response_time_ms=response_time,
                error=str(e),
                timestamp=time.time(),
            )

    async def list_models(self) -> List[str]:
        """List available models with caching."""
        current_time = time.time()
        
        # Check cache
        if (
            self._models_cache 
            and current_time - self._models_cache_time < self._models_cache_ttl
        ):
            return self._models_cache

        try:
            if not self._client:
                await self.initialize()

            response = await self._client.get("/models")
            await self._handle_response_errors(response)

            data = response.json()
            models = [model["id"] for model in data.get("data", [])]
            
            # Cache results
            self._models_cache = models
            self._models_cache_time = current_time
            
            return models

        except Exception as e:
            # Return cached results if available, otherwise empty list
            if self._models_cache:
                return self._models_cache
            raise LLMError(f"Failed to list models: {e}")

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        self._models_cache = None