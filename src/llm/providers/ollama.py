"""Enterprise-grade Ollama provider with comprehensive error handling and monitoring."""

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


@register_provider("ollama")
class OllamaProvider(BaseLLMProvider):
    """Enterprise-grade Ollama provider for local LLM deployment.

    Features:
    - Automatic retry with exponential backoff
    - Proper chat message formatting for local models
    - Health monitoring and model management
    - Resource management and cleanup
    - Both streaming and non-streaming support
    - Context window management
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout: int = 60,
        keep_alive: str = "5m",
        num_ctx: int = 4096,
        **kwargs: Any,
    ) -> None:
        """Initialize Ollama provider.

        Args:
            base_url: Ollama server URL
            timeout: Request timeout in seconds
            keep_alive: Model keep-alive duration
            num_ctx: Context window size
            **kwargs: Additional configuration
        """
        super().__init__(**kwargs)
        self.base_url = str(base_url).rstrip("/")
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.num_ctx = num_ctx
        self._client: Optional[httpx.AsyncClient] = None
        self._models_cache: Optional[List[str]] = None
        self._models_cache_time: float = 0
        self._models_cache_ttl: int = 300  # 5 minutes (faster refresh for local)

    async def initialize(self) -> None:
        """Initialize the Ollama provider and validate server connection."""
        if self._initialized:
            return

        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
            headers={"Content-Type": "application/json"},
        )

        # Validate server connection
        try:
            await self.health_check()
            self._initialized = True
        except Exception as e:
            await self.close()
            raise LLMError(f"Failed to initialize Ollama provider: {e}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
    )
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[StreamingChunk, None]]:
        """Generate a chat completion with Ollama.

        Args:
            request: Strongly typed chat completion request

        Returns:
            Chat completion response or streaming generator

        Raises:
            Various LLMError subclasses based on error type
        """
        if not self._initialized:
            await self.initialize()

        # Convert messages to Ollama format
        prompt = self._format_messages(request.messages)

        # Prepare Ollama payload
        payload = {
            "model": request.model,
            "prompt": prompt,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens,
                "top_p": request.top_p,
                "num_ctx": self.num_ctx,
                "stop": request.stop if request.stop else [],
            },
            "keep_alive": self.keep_alive,
        }

        if request.stream:
            return self._stream_completion(payload)
        else:
            return await self._complete_chat(payload, request.model)

    async def _complete_chat(
        self, payload: Dict[str, Any], model: str
    ) -> ChatCompletionResponse:
        """Handle non-streaming chat completion."""
        try:
            response = await self._client.post("/api/generate", json=payload)
            await self._handle_response_errors(response)

            data = response.json()

            # Convert Ollama response to OpenAI-compatible format
            choices = [
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(
                        role="assistant",
                        content=data.get("response", ""),
                    ),
                    finish_reason="stop" if data.get("done", False) else None,
                )
            ]

            usage = None
            if "prompt_eval_count" in data or "eval_count" in data:
                prompt_tokens = data.get("prompt_eval_count", 0)
                completion_tokens = data.get("eval_count", 0)
                usage = ChatCompletionUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                )

            return ChatCompletionResponse(
                id=f"ollama-{model}-{int(time.time())}",
                created=int(time.time()),
                model=model,
                choices=choices,
                usage=usage,
            )

        except httpx.HTTPStatusError as e:
            await self._handle_response_errors(e.response)
            raise

    async def _stream_completion(
        self, payload: Dict[str, Any]
    ) -> AsyncGenerator[StreamingChunk, None]:
        """Handle streaming chat completion."""
        try:
            async with self._client.stream(
                "POST", "/api/generate", json=payload
            ) as response:
                await self._handle_response_errors(response)

                async for line in response.aiter_lines():
                    if not line.strip():
                        continue

                    try:
                        data = json.loads(line)
                        
                        # Convert to OpenAI-compatible streaming format
                        choices = [
                            StreamingChoice(
                                index=0,
                                delta={"content": data.get("response", "")},
                                finish_reason="stop" if data.get("done", False) else None,
                            )
                        ]

                        yield StreamingChunk(
                            id=f"ollama-{payload['model']}-{int(time.time())}",
                            created=int(time.time()),
                            model=payload["model"],
                            choices=choices,
                        )

                        if data.get("done", False):
                            break

                    except json.JSONDecodeError:
                        # Skip malformed chunks
                        continue

        except httpx.HTTPStatusError as e:
            await self._handle_response_errors(e.response)

    def _format_messages(self, messages: List[ChatMessage]) -> str:
        """Format messages for Ollama.

        Ollama typically expects a single prompt string rather than
        a messages array, so we format the conversation appropriately.
        """
        formatted_parts = []

        for message in messages:
            role = message.role
            content = message.content

            if role == "system":
                formatted_parts.append(f"System: {content}")
            elif role == "user":
                formatted_parts.append(f"Human: {content}")
            elif role == "assistant":
                formatted_parts.append(f"Assistant: {content}")

        # Add final prompt for assistant response
        formatted_parts.append("Assistant:")
        
        return "\n\n".join(formatted_parts)

    async def _handle_response_errors(self, response: httpx.Response) -> None:
        """Handle HTTP response errors with proper classification."""
        if response.is_success:
            return

        status_code = response.status_code
        
        try:
            error_data = response.json()
            error_message = error_data.get("error", "Unknown error")
        except (json.JSONDecodeError, KeyError):
            error_message = f"HTTP {status_code}: {response.text}"

        # Classify errors for Ollama
        if status_code == 404:
            if "model" in error_message.lower():
                raise ModelNotFoundError(error_message, status_code)
            else:
                raise LLMError(error_message, status_code)
        elif status_code == 400:
            raise LLMError(error_message, status_code)
        elif status_code >= 500:
            raise LLMError(f"Ollama server error: {error_message}", status_code)
        else:
            raise LLMError(error_message, status_code)

    async def health_check(self) -> HealthCheck:
        """Perform comprehensive health check."""
        start_time = time.time()
        
        try:
            if not self._client:
                await self.initialize()

            # Check server status
            response = await self._client.get("/api/tags", timeout=10.0)
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

            response = await self._client.get("/api/tags")
            await self._handle_response_errors(response)

            data = response.json()
            models = [model["name"] for model in data.get("models", [])]
            
            # Cache results
            self._models_cache = models
            self._models_cache_time = current_time
            
            return models

        except Exception as e:
            # Return cached results if available
            if self._models_cache:
                return self._models_cache
            raise LLMError(f"Failed to list models: {e}")

    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry.

        Args:
            model_name: Name of the model to pull

        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._client:
                await self.initialize()

            payload = {"name": model_name}
            response = await self._client.post("/api/pull", json=payload)
            await self._handle_response_errors(response)
            
            # Clear models cache to force refresh
            self._models_cache = None
            
            return True

        except Exception:
            return False

    async def close(self) -> None:
        """Clean up resources."""
        if self._client:
            await self._client.aclose()
            self._client = None
        self._initialized = False
        self._models_cache = None