"""LLM provider adapters.

This package contains adapters for different LLM providers.
"""

from typing import Any

from src.llm.providers.base import BaseLLMProvider, get_provider_registry

# Import built-in providers to trigger registration
from src.llm.providers import openai  # noqa: F401
from src.llm.providers import ollama  # noqa: F401

__all__ = ["get_provider", "BaseLLMProvider"]


def get_provider(name: str, **kwargs: Any) -> BaseLLMProvider:
    """Get a provider instance by name.

    Args:
        name: Provider name
        **kwargs: Additional provider-specific arguments

    Returns:
        A provider instance
    """
    registry = get_provider_registry()
    
    # Get provider class from registry and instantiate
    try:
        provider_class = registry.get_provider(name)
        return provider_class(**kwargs)
    except KeyError:
        raise ValueError(f"Unknown provider: {name}")
