"""Configuration for Oh My Repos.

This module provides configuration settings for the Oh My Repos application.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings.
    
    This class defines all the configuration settings for the application.
    Values are loaded from environment variables or .env file.
    """
    
    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parents[1]
    
    # GitHub API
    GITHUB_USERNAME: str = ""
    GITHUB_TOKEN: str = ""
    
    # LLM settings
    CHAT_LLM_PROVIDER: str = "openai"  # Options: "openai", "ollama"
    CHAT_LLM_BASE_URL: str = "https://openrouter.ai/api/v1"
    CHAT_LLM_MODEL: str = "deepseek/deepseek-r1-0528:free"
    CHAT_LLM_API_KEY: str = ""
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_MODEL: str = "phi3.5:3.8b"
    OLLAMA_TIMEOUT: int = 60
    
    # Vector DB settings
    QDRANT_URL: str = ""
    QDRANT_API_KEY: str = ""
    QDRANT_API_KEY_HEADER: str = "api-key"
    
    # Embedding settings
    EMBEDDING_MODEL: str = "jina-embeddings-v3"
    EMBEDDING_MODEL_API_KEY: str = ""
    EMBEDDING_MODEL_URL: str = "https://api.jina.ai/v1/embeddings"
    
    # Reranker settings
    RERANKER_MODEL: str = "jina-reranker-m0"
    RERANKER_URL: str = "https://api.jina.ai/v1/rerank"
    RERANKER_TIMEOUT: int = 30
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


# Create global settings instance
settings = Settings()
