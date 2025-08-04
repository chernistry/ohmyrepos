"""Enterprise-grade base class for LLM providers with comprehensive type safety."""

import asyncio
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Protocol, Union

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Strongly typed chat message."""

    role: str = Field(..., pattern=r"^(system|user|assistant|function)$")
    content: str = Field(..., min_length=1)
    name: Optional[str] = Field(default=None)
    function_call: Optional[Dict[str, Any]] = Field(default=None)


class ChatCompletionRequest(BaseModel):
    """Strongly typed chat completion request."""

    messages: List[ChatMessage] = Field(..., min_items=1)
    model: str = Field(..., min_length=1)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=32000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    stream: bool = Field(default=False)
    stop: Optional[Union[str, List[str]]] = Field(default=None)


class ChatCompletionChoice(BaseModel):
    """Chat completion choice with type safety."""

    index: int = Field(..., ge=0)
    message: ChatMessage
    finish_reason: Optional[str] = Field(default=None)


class ChatCompletionUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class ChatCompletionResponse(BaseModel):
    """Strongly typed chat completion response."""

    id: str = Field(...)
    object: str = Field(default="chat.completion")
    created: int = Field(...)
    model: str = Field(...)
    choices: List[ChatCompletionChoice] = Field(..., min_items=1)
    usage: Optional[ChatCompletionUsage] = Field(default=None)


class StreamingChoice(BaseModel):
    """Streaming response choice."""

    index: int = Field(..., ge=0)
    delta: Dict[str, Any] = Field(default_factory=dict)
    finish_reason: Optional[str] = Field(default=None)


class StreamingChunk(BaseModel):
    """Streaming response chunk."""

    id: str = Field(...)
    object: str = Field(default="chat.completion.chunk")
    created: int = Field(...)
    model: str = Field(...)
    choices: List[StreamingChoice] = Field(..., min_items=1)


class LLMError(Exception):
    """Base exception for LLM provider errors."""

    def __init__(self, message: str, status_code: Optional[int] = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class RateLimitError(LLMError):
    """Exception for rate limit errors."""

    pass


class AuthenticationError(LLMError):
    """Exception for authentication errors."""

    pass


class ModelNotFoundError(LLMError):
    """Exception for model not found errors."""

    pass


class TokenLimitError(LLMError):
    """Exception for token limit exceeded errors."""

    pass


class HealthCheck(BaseModel):
    """Health check result."""

    status: str = Field(..., pattern=r"^(healthy|unhealthy|degraded)$")
    response_time_ms: float = Field(..., ge=0)
    error: Optional[str] = Field(default=None)
    timestamp: float = Field(...)


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers with enterprise features.

    This class defines the interface that all LLM providers must implement,
    including health checks, metrics, and proper resource management.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the provider with configuration."""
        self._initialized = False
        self._client = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider and its resources."""
        pass

    @abstractmethod
    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[StreamingChunk, None]]:
        """Generate a chat completion.

        Args:
            request: Strongly typed chat completion request

        Returns:
            Chat completion response or streaming generator

        Raises:
            LLMError: For any provider-specific errors
            RateLimitError: When rate limits are exceeded
            AuthenticationError: For authentication failures
            ModelNotFoundError: When the specified model is not found
            TokenLimitError: When token limits are exceeded
        """
        pass

    @abstractmethod
    async def health_check(self) -> HealthCheck:
        """Perform a health check on the provider.

        Returns:
            Health check result with status and metrics
        """
        pass

    @abstractmethod
    async def list_models(self) -> List[str]:
        """List available models.

        Returns:
            List of available model names
        """
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close any resources used by the provider."""
        pass

    async def __aenter__(self) -> "BaseLLMProvider":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()

    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._initialized

    @asynccontextmanager
    async def rate_limited(self, max_concurrent: int = 10):
        """Rate limiting context manager."""
        if not self._semaphore:
            self._semaphore = asyncio.Semaphore(max_concurrent)
        
        async with self._semaphore:
            yield


class LLMProviderRegistry:
    """Registry for LLM providers with type safety."""

    def __init__(self) -> None:
        self._providers: Dict[str, type[BaseLLMProvider]] = {}

    def register(self, name: str, provider_class: type[BaseLLMProvider]) -> None:
        """Register a provider class.

        Args:
            name: Provider name
            provider_class: Provider class implementing BaseLLMProvider
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(f"Provider {name} must inherit from BaseLLMProvider")
        
        self._providers[name] = provider_class

    def get_provider(self, name: str) -> type[BaseLLMProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            KeyError: If provider is not registered
        """
        if name not in self._providers:
            raise KeyError(f"Provider '{name}' not registered")
        return self._providers[name]

    def list_providers(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def create_provider(self, name: str, **kwargs: Any) -> BaseLLMProvider:
        """Create a provider instance.

        Args:
            name: Provider name
            **kwargs: Provider-specific configuration

        Returns:
            Provider instance
        """
        provider_class = self.get_provider(name)
        return provider_class(**kwargs)


# Global registry instance
_registry = LLMProviderRegistry()


def register_provider(name: str):
    """Decorator to register a provider class.

    Args:
        name: Provider name
    """
    def decorator(cls: type[BaseLLMProvider]) -> type[BaseLLMProvider]:
        _registry.register(name, cls)
        return cls
    return decorator


def get_provider_registry() -> LLMProviderRegistry:
    """Get the global provider registry."""
    return _registry