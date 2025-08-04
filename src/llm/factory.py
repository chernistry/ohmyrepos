"""Enterprise-grade LLM provider factory with comprehensive configuration management."""

from typing import Any, Dict, Optional

from ..config import LLMConfig, OllamaConfig, settings
from .providers.base import BaseLLMProvider, get_provider_registry
from .providers.openai import OpenAIProvider
from .providers.ollama import OllamaProvider


class LLMProviderFactory:
    """Factory for creating LLM providers with enterprise configuration.
    
    This factory handles provider instantiation with proper configuration
    management, validation, and error handling.
    """

    @staticmethod
    def create_provider(
        provider_config: Optional[LLMConfig] = None,
        ollama_config: Optional[OllamaConfig] = None,
        **kwargs: Any,
    ) -> BaseLLMProvider:
        """Create an LLM provider instance with configuration.

        Args:
            provider_config: LLM provider configuration
            ollama_config: Ollama-specific configuration
            **kwargs: Additional provider-specific arguments

        Returns:
            Configured LLM provider instance

        Raises:
            ValueError: If provider configuration is invalid
            KeyError: If provider type is not supported
        """
        # Use global settings if no config provided
        if not provider_config:
            provider_config = settings.llm
            if not provider_config:
                raise ValueError("LLM configuration is required")

        if not ollama_config:
            ollama_config = settings.ollama

        provider_type = provider_config.provider.value
        registry = get_provider_registry()

        # Provider-specific configuration
        provider_kwargs = kwargs.copy()

        if provider_type == "openai":
            if not provider_config.api_key:
                raise ValueError("OpenAI API key is required")
            
            provider_kwargs.update({
                "api_key": provider_config.api_key.get_secret_value(),
                "base_url": str(provider_config.base_url),
                "timeout": provider_config.timeout,
                "max_retries": provider_config.max_retries,
                "max_concurrent": provider_config.max_concurrent,
            })

        elif provider_type == "ollama":
            provider_kwargs.update({
                "base_url": str(ollama_config.base_url),
                "timeout": ollama_config.timeout,
                "keep_alive": ollama_config.keep_alive,
                "num_ctx": ollama_config.num_ctx,
            })

        # Create provider instance
        try:
            return registry.create_provider(provider_type, **provider_kwargs)
        except KeyError:
            available_providers = registry.list_providers()
            raise KeyError(
                f"Provider '{provider_type}' not supported. "
                f"Available providers: {available_providers}"
            )

    @staticmethod
    def create_openai_provider(
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "deepseek/deepseek-r1-0528:free",
        **kwargs: Any,
    ) -> OpenAIProvider:
        """Create an OpenAI provider with direct configuration.

        Args:
            api_key: OpenAI API key
            base_url: API base URL
            model: Model identifier
            **kwargs: Additional configuration

        Returns:
            Configured OpenAI provider
        """
        return OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )

    @staticmethod
    def create_ollama_provider(
        base_url: str = "http://127.0.0.1:11434",
        model: str = "phi3.5:3.8b",
        **kwargs: Any,
    ) -> OllamaProvider:
        """Create an Ollama provider with direct configuration.

        Args:
            base_url: Ollama server URL
            model: Model identifier
            **kwargs: Additional configuration

        Returns:
            Configured Ollama provider
        """
        return OllamaProvider(
            base_url=base_url,
            **kwargs
        )

    @staticmethod
    def get_default_provider() -> BaseLLMProvider:
        """Get the default provider based on application settings.

        Returns:
            Default configured LLM provider

        Raises:
            ValueError: If no valid configuration is found
        """
        return LLMProviderFactory.create_provider()

    @staticmethod
    def list_available_providers() -> Dict[str, str]:
        """List all available providers with descriptions.

        Returns:
            Dictionary mapping provider names to descriptions
        """
        return {
            "openai": "OpenAI API (GPT models) and compatible services",
            "ollama": "Local Ollama deployment for private AI",
        }

    @staticmethod
    def validate_provider_config(
        provider_type: str,
        config: Dict[str, Any]
    ) -> bool:
        """Validate provider configuration.

        Args:
            provider_type: Type of provider
            config: Configuration dictionary

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        if provider_type == "openai":
            if "api_key" not in config or not config["api_key"]:
                raise ValueError("OpenAI provider requires 'api_key'")
            if "base_url" not in config:
                raise ValueError("OpenAI provider requires 'base_url'")

        elif provider_type == "ollama":
            if "base_url" not in config:
                raise ValueError("Ollama provider requires 'base_url'")

        else:
            registry = get_provider_registry()
            available = registry.list_providers()
            raise ValueError(
                f"Unknown provider type '{provider_type}'. "
                f"Available: {available}"
            )

        return True


# Convenience functions for direct usage
def create_llm_provider(**kwargs: Any) -> BaseLLMProvider:
    """Create an LLM provider using the factory.
    
    Args:
        **kwargs: Provider configuration arguments
        
    Returns:
        Configured LLM provider instance
    """
    return LLMProviderFactory.create_provider(**kwargs)


def get_default_llm() -> BaseLLMProvider:
    """Get the default LLM provider.
    
    Returns:
        Default configured LLM provider
    """
    return LLMProviderFactory.get_default_provider()