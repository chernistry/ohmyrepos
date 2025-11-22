"""Enterprise-grade configuration system with secure secret management.

This module provides a comprehensive configuration system with validation,
secure secret handling, and environment-specific settings.
"""

import os
import secrets
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    Field,
    HttpUrl,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env file explicitly before Settings initialization
load_dotenv(Path(__file__).parent.parent / ".env")


class Environment(str, Enum):
    """Application environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LLMProvider(str, Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OLLAMA = "ollama"


class GitHubConfig(BaseModel):
    """GitHub API configuration."""

    username: str = Field(..., min_length=1, description="GitHub username")
    token: SecretStr = Field(..., description="GitHub personal access token")
    api_url: HttpUrl = Field(
        default="https://api.github.com", description="GitHub API base URL"
    )
    max_concurrent_requests: int = Field(
        default=10, ge=1, le=50, description="Maximum concurrent API requests"
    )
    rate_limit_wait_time: int = Field(
        default=60, ge=1, description="Rate limit wait time in seconds"
    )
    request_timeout: int = Field(
        default=30, ge=5, le=300, description="Request timeout in seconds"
    )

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate GitHub username format."""
        if not v.replace("-", "").replace("_", "").isalnum():
            raise ValueError("Invalid GitHub username format")
        return v.lower()

    @field_validator("token")
    @classmethod
    def validate_token(cls, v: SecretStr) -> SecretStr:
        """Validate GitHub token is not empty."""
        if not v.get_secret_value().strip():
            raise ValueError("GitHub token cannot be empty")
        return v


class LLMConfig(BaseModel):
    """LLM provider configuration."""

    provider: LLMProvider = Field(default=LLMProvider.OPENAI)
    base_url: HttpUrl = Field(default="https://api.openai.com/v1")
    model: str = Field(default="deepseek/deepseek-r1-0528:free")
    api_key: Optional[SecretStr] = Field(default=None)
    timeout: int = Field(default=60, ge=5, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    max_concurrent: int = Field(default=4, ge=1, le=20)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1000, ge=1, le=4000)

    @model_validator(mode="after")
    def validate_provider_config(self) -> "LLMConfig":
        """Validate provider-specific configuration."""
        if self.provider == LLMProvider.OPENAI and not self.api_key:
            raise ValueError("OpenAI provider requires API key")
        return self


class OllamaConfig(BaseModel):
    """Ollama-specific configuration."""

    base_url: HttpUrl = Field(default="http://127.0.0.1:11434")
    model: str = Field(default="phi3.5:3.8b")
    timeout: int = Field(default=60, ge=5, le=300)
    keep_alive: str = Field(default="5m")
    num_ctx: int = Field(default=4096, ge=512, le=32768)
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)


class QdrantConfig(BaseModel):
    """Qdrant vector database configuration."""

    url: HttpUrl = Field(..., description="Qdrant server URL")
    api_key: Optional[SecretStr] = Field(default=None)
    collection_name: str = Field(default="repositories")
    vector_size: int = Field(default=1024, ge=128, le=4096)
    distance_metric: Literal["cosine", "dot", "euclidean"] = Field(default="cosine")
    timeout: int = Field(default=30, ge=5, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    batch_size: int = Field(default=100, ge=1, le=1000)

    @field_validator("url")
    @classmethod
    def validate_qdrant_url(cls, v: HttpUrl) -> HttpUrl:
        """Validate Qdrant URL format."""
        parsed = urlparse(str(v))
        if not parsed.netloc:
            raise ValueError("Invalid Qdrant URL")
        return v


class EmbeddingProviderType(str, Enum):
    """Supported embedding providers."""

    JINA = "jina"
    OLLAMA = "ollama"


class EmbeddingConfig(BaseModel):
    """Embedding provider configuration."""

    provider: EmbeddingProviderType = Field(default=EmbeddingProviderType.JINA)
    model: str = Field(default="jina-embeddings-v3")
    api_key: Optional[SecretStr] = Field(default=None, description="Embedding API key")
    base_url: HttpUrl = Field(default="https://api.jina.ai/v1/embeddings")
    batch_size: int = Field(default=32, ge=1, le=100)
    timeout: int = Field(default=30, ge=5, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    dimension: int = Field(default=1024, ge=128, le=4096)


class RerankerConfig(BaseModel):
    """Reranker configuration."""

    model: str = Field(default="jina-reranker-v2-base-multilingual")
    api_key: SecretStr = Field(..., description="Reranker API key")
    base_url: HttpUrl = Field(default="https://api.jina.ai/v1/rerank")
    timeout: int = Field(default=30, ge=5, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    top_n: int = Field(default=25, ge=1, le=100)


class SearchConfig(BaseModel):
    """Search configuration."""

    bm25_variant: Literal["okapi", "plus"] = Field(default="plus")
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0)  # Увеличен вес BM25 для лучших результатов по ключевым словам
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    max_results: int = Field(default=100, ge=1, le=1000)
    enable_reranking: bool = Field(default=True)

    @model_validator(mode="after")
    def validate_weights(self) -> "SearchConfig":
        """Validate that weights sum to 1.0."""
        if abs(self.bm25_weight + self.vector_weight - 1.0) > 0.01:
            raise ValueError("BM25 and vector weights must sum to 1.0")
        return self


class MonitoringConfig(BaseModel):
    """Monitoring and observability configuration."""

    enable_metrics: bool = Field(default=True)
    metrics_port: int = Field(default=8000, ge=1024, le=65535)
    enable_tracing: bool = Field(default=False)
    log_level: LogLevel = Field(default=LogLevel.INFO)
    structured_logging: bool = Field(default=True)
    health_check_interval: int = Field(default=30, ge=1, le=300)


class SecurityConfig(BaseModel):
    """Security configuration."""

    secret_key: SecretStr = Field(default_factory=lambda: SecretStr(secrets.token_urlsafe(32)))
    allowed_hosts: List[str] = Field(default_factory=lambda: ["localhost", "127.0.0.1"])
    enable_rate_limiting: bool = Field(default=True)
    rate_limit_requests: int = Field(default=100, ge=1)
    rate_limit_window: int = Field(default=60, ge=1)
    enable_cors: bool = Field(default=True)
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])


class Settings(BaseSettings):
    """Application settings with comprehensive validation and security.

    This class provides enterprise-grade configuration management with:
    - Environment-specific settings
    - Secure secret handling
    - Comprehensive validation
    - Type safety
    """

    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[1])

    # Component configurations
    github: Optional[GitHubConfig] = Field(default=None)
    llm: Optional[LLMConfig] = Field(default=None)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    qdrant: Optional[QdrantConfig] = Field(default=None)
    embedding: Optional[EmbeddingConfig] = Field(default=None)
    reranker: Optional[RerankerConfig] = Field(default=None)
    search: SearchConfig = Field(default_factory=SearchConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)

    # Legacy flat settings for backward compatibility
    GITHUB_USERNAME: Optional[str] = Field(default=None, exclude=True)
    GITHUB_TOKEN: Optional[str] = Field(default=None, exclude=True)
    CHAT_LLM_PROVIDER: Optional[str] = Field(default=None, exclude=True)
    CHAT_LLM_API_KEY: Optional[str] = Field(default=None, exclude=True)
    QDRANT_URL: Optional[str] = Field(default=None, exclude=True)
    QDRANT_API_KEY: Optional[str] = Field(default=None, exclude=True)
    EMBEDDING_MODEL_API_KEY: Optional[str] = Field(default=None, exclude=True)

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).parent.parent / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra="ignore",
        secrets_dir=os.getenv("SECRETS_DIR"),
    )

    @model_validator(mode="after")
    def build_nested_config(self) -> "Settings":
        """Build nested configuration from flat environment variables."""
        # Handle GitHub config
        if not self.github:
            github_username = self.GITHUB_USERNAME or os.getenv("GITHUB_USERNAME", "")
            github_token = self.GITHUB_TOKEN or os.getenv("GITHUB_TOKEN", "")
            if github_username and github_token:
                self.github = GitHubConfig(
                    username=github_username, token=SecretStr(github_token)
                )

        # Handle LLM config
        if not self.llm:
            provider = self.CHAT_LLM_PROVIDER or os.getenv("CHAT_LLM_PROVIDER", "openai")
            api_key = self.CHAT_LLM_API_KEY or os.getenv("CHAT_LLM_API_KEY", "")
            base_url = os.getenv("CHAT_LLM_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("CHAT_LLM_MODEL", "deepseek/deepseek-r1-0528:free")
            self.llm = LLMConfig(
                provider=LLMProvider(provider),
                base_url=base_url,
                model=model,
                api_key=SecretStr(api_key) if api_key else None,
            )

        # Handle Qdrant config
        if not self.qdrant:
            qdrant_url = self.QDRANT_URL or os.getenv("QDRANT_URL", "")
            qdrant_key = self.QDRANT_API_KEY or os.getenv("QDRANT_API_KEY", "")
            if qdrant_url:
                self.qdrant = QdrantConfig(
                    url=qdrant_url,
                    api_key=SecretStr(qdrant_key) if qdrant_key else None,
                )

        # Handle Embedding config
        if not self.embedding:
            provider_str = os.getenv("EMBEDDINGS_SERVICE", "jina").lower()
            try:
                provider = EmbeddingProviderType(provider_str)
            except ValueError:
                provider = EmbeddingProviderType.JINA

            # Default settings for each provider
            if provider == EmbeddingProviderType.OLLAMA:
                model = os.getenv("EMBEDDING_MODEL", "embeddinggemma:latest")
                base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/api/embeddings")
                api_key = None
            else:
                model = os.getenv("EMBEDDING_MODEL", "jina-embeddings-v3")
                base_url = os.getenv("EMBEDDING_MODEL_URL", "https://api.jina.ai/v1/embeddings")
                api_key_val = self.EMBEDDING_MODEL_API_KEY or os.getenv("EMBEDDING_MODEL_API_KEY", "")
                api_key = SecretStr(api_key_val) if api_key_val else None

            self.embedding = EmbeddingConfig(
                provider=provider,
                model=model,
                base_url=base_url,
                api_key=api_key
            )

        # Handle Reranker config (use same key as embedding for simplicity)
        if not self.reranker:
            reranker_key = (
                self.EMBEDDING_MODEL_API_KEY
                or os.getenv("RERANKER_API_KEY", "")
                or os.getenv("EMBEDDING_MODEL_API_KEY", "")
            )
            if reranker_key:
                self.reranker = RerankerConfig(api_key=SecretStr(reranker_key))

        return self

    @field_validator("environment", mode="before")
    @classmethod
    def validate_environment(cls, v: Any) -> Environment:
        """Validate and convert environment setting."""
        if isinstance(v, str):
            return Environment(v.lower())
        return v

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    def get_secret(self, key: str) -> Optional[str]:
        """Safely retrieve secret values."""
        # Check for secrets in various locations
        # 1. Environment variables
        value = os.getenv(key)
        if value:
            return value

        # 2. Secrets directory (for container deployments)
        secrets_dir = os.getenv("SECRETS_DIR", "/run/secrets")
        if secrets_dir and os.path.exists(secrets_dir):
            secret_file = Path(secrets_dir) / key.lower()
            if secret_file.exists():
                return secret_file.read_text().strip()

        # 3. AWS Secrets Manager (if available)
        try:
            import boto3
            from botocore.exceptions import ClientError

            session = boto3.Session()
            client = session.client("secretsmanager")
            response = client.get_secret_value(SecretId=key)
            return response["SecretString"]
        except (ImportError, ClientError):
            pass

        return None

    def validate_required_settings(self) -> None:
        """Validate that all required settings are configured."""
        errors = []

        if not self.github:
            errors.append("GitHub configuration is required")
        elif not self.github.token.get_secret_value():
            errors.append("GitHub token is required")

        if not self.qdrant:
            errors.append("Qdrant configuration is required")

        if not self.embedding:
            errors.append("Embedding configuration is required")
        elif self.embedding.provider == EmbeddingProviderType.JINA and (
            not self.embedding.api_key or not self.embedding.api_key.get_secret_value()
        ):
            errors.append("Embedding API key is required for Jina provider")

        if self.llm and self.llm.provider == LLMProvider.OPENAI and (
            not self.llm.api_key or not self.llm.api_key.get_secret_value()
        ):
            errors.append("OpenAI API key is required for OpenAI provider")

        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")

    def model_dump_safe(self) -> Dict[str, Any]:
        """Safely dump model excluding secrets."""
        data = self.model_dump(exclude={"github", "llm", "embedding", "reranker"})
        # Add non-secret parts of configs
        if self.github:
            data["github"] = {
                "username": self.github.username,
                "api_url": str(self.github.api_url),
                "max_concurrent_requests": self.github.max_concurrent_requests,
            }
        if self.llm:
            data["llm"] = {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "base_url": str(self.llm.base_url),
            }
        return data


def get_settings() -> Settings:
    """Get application settings with validation."""
    settings = Settings()
    if settings.is_production():
        settings.validate_required_settings()
    return settings


# Global settings instance
settings = get_settings()