"""Production configuration for Oh My Repos.

This module provides production-specific settings and configuration management
with enhanced security, performance tuning, and operational features.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict

try:
    import keyring

    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError

    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False


class ProductionSettings(BaseSettings):
    """Production configuration settings with enhanced security and monitoring."""

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parents[2]

    # Environment
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    DEBUG: bool = Field(default=False, env="DEBUG")

    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json or structured

    # GitHub API (with secure retrieval)
    GITHUB_USERNAME: str = Field(default="", env="GITHUB_USERNAME")
    GITHUB_TOKEN: str = Field(default="", env="GITHUB_TOKEN")

    # LLM settings with production defaults
    CHAT_LLM_PROVIDER: str = Field(default="openai", env="CHAT_LLM_PROVIDER")
    CHAT_LLM_BASE_URL: str = Field(
        default="https://api.openai.com/v1", env="CHAT_LLM_BASE_URL"
    )
    CHAT_LLM_MODEL: str = Field(default="gpt-4", env="CHAT_LLM_MODEL")
    CHAT_LLM_API_KEY: str = Field(default="", env="CHAT_LLM_API_KEY")

    # Ollama settings (for hybrid deployments)
    OLLAMA_BASE_URL: str = Field(
        default="http://127.0.0.1:11434", env="OLLAMA_BASE_URL"
    )
    OLLAMA_MODEL: str = Field(default="phi3.5:3.8b", env="OLLAMA_MODEL")
    OLLAMA_TIMEOUT: int = Field(
        default=120, env="OLLAMA_TIMEOUT"
    )  # Extended for production

    # Vector DB settings
    QDRANT_URL: str = Field(default="", env="QDRANT_URL")
    QDRANT_API_KEY: str = Field(default="", env="QDRANT_API_KEY")
    QDRANT_API_KEY_HEADER: str = Field(default="api-key", env="QDRANT_API_KEY_HEADER")

    # Embedding settings
    EMBEDDING_MODEL: str = Field(default="jina-embeddings-v3", env="EMBEDDING_MODEL")
    EMBEDDING_MODEL_API_KEY: str = Field(default="", env="EMBEDDING_MODEL_API_KEY")
    EMBEDDING_MODEL_URL: str = Field(
        default="https://api.jina.ai/v1/embeddings", env="EMBEDDING_MODEL_URL"
    )

    # Reranker settings
    RERANKER_MODEL: str = Field(default="jina-reranker-m0", env="RERANKER_MODEL")
    RERANKER_URL: str = Field(
        default="https://api.jina.ai/v1/rerank", env="RERANKER_URL"
    )
    RERANKER_TIMEOUT: int = Field(
        default=60, env="RERANKER_TIMEOUT"
    )  # Extended for production

    # Hybrid retriever
    BM25_VARIANT: str = Field(default="plus", env="BM25_VARIANT")
    BM25_WEIGHT: float = Field(default=0.4, env="BM25_WEIGHT")
    VECTOR_WEIGHT: float = Field(default=0.6, env="VECTOR_WEIGHT")

    # Production-specific settings
    MAX_CONCURRENT_REQUESTS: int = Field(default=50, env="MAX_CONCURRENT_REQUESTS")
    MAX_REQUEST_SIZE: int = Field(default=1024 * 1024, env="MAX_REQUEST_SIZE")  # 1MB
    REQUEST_TIMEOUT: int = Field(default=30, env="REQUEST_TIMEOUT")
    CONNECTION_POOL_SIZE: int = Field(default=100, env="CONNECTION_POOL_SIZE")

    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = Field(
        default=1000, env="RATE_LIMIT_REQUESTS_PER_MINUTE"
    )
    RATE_LIMIT_BURST_SIZE: int = Field(default=100, env="RATE_LIMIT_BURST_SIZE")

    # Retry settings\n    RETRY_MAX_ATTEMPTS: int = Field(default=3, env="RETRY_MAX_ATTEMPTS")
    RETRY_INITIAL_DELAY: float = Field(default=1.0, env="RETRY_INITIAL_DELAY")
    RETRY_MAX_DELAY: float = Field(default=60.0, env="RETRY_MAX_DELAY")
    RETRY_EXPONENTIAL_BASE: float = Field(default=2.0, env="RETRY_EXPONENTIAL_BASE")

    # Circuit breaker settings
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = Field(
        default=5, env="CIRCUIT_BREAKER_FAILURE_THRESHOLD"
    )
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = Field(
        default=60, env="CIRCUIT_BREAKER_RECOVERY_TIMEOUT"
    )
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION_TYPES: List[str] = Field(
        default=["httpx.TimeoutException", "httpx.ConnectError", "aiohttp.ClientError"],
        env="CIRCUIT_BREAKER_EXPECTED_EXCEPTION_TYPES",
    )

    # Monitoring and observability
    METRICS_PORT: int = Field(default=8000, env="METRICS_PORT")
    HEALTH_CHECK_PORT: int = Field(default=8001, env="HEALTH_CHECK_PORT")
    ENABLE_PROMETHEUS_METRICS: bool = Field(
        default=True, env="ENABLE_PROMETHEUS_METRICS"
    )
    ENABLE_STRUCTURED_LOGGING: bool = Field(
        default=True, env="ENABLE_STRUCTURED_LOGGING"
    )

    # Security settings
    SECRETS_BACKEND: str = Field(
        default="env", env="SECRETS_BACKEND"
    )  # env, keyring, aws
    AWS_SECRETS_REGION: str = Field(default="us-east-1", env="AWS_SECRETS_REGION")
    AWS_SECRETS_PREFIX: str = Field(default="ohmyrepos/", env="AWS_SECRETS_PREFIX")
    KEYRING_SERVICE_NAME: str = Field(default="ohmyrepos", env="KEYRING_SERVICE_NAME")

    # Data protection
    ENABLE_DATA_ENCRYPTION: bool = Field(default=True, env="ENABLE_DATA_ENCRYPTION")
    ENCRYPTION_KEY: Optional[str] = Field(default=None, env="ENCRYPTION_KEY")

    # Performance tuning
    UVLOOP_ENABLED: bool = Field(default=True, env="UVLOOP_ENABLED")
    ORJSON_ENABLED: bool = Field(default=True, env="ORJSON_ENABLED")
    HTTP2_ENABLED: bool = Field(default=True, env="HTTP2_ENABLED")

    # Batch processing
    BATCH_SIZE: int = Field(default=100, env="BATCH_SIZE")
    MAX_BATCH_CONCURRENCY: int = Field(default=10, env="MAX_BATCH_CONCURRENCY")

    model_config = SettingsConfigDict(
        env_file=[".env.production", ".env"],
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level. Must be one of: {valid_levels}")
        return v.upper()

    @validator("SECRETS_BACKEND")
    def validate_secrets_backend(cls, v):
        """Validate secrets backend."""
        valid_backends = ["env", "keyring", "aws"]
        if v not in valid_backends:
            raise ValueError(
                f"Invalid secrets backend. Must be one of: {valid_backends}"
            )
        return v

    @validator("BM25_WEIGHT", "VECTOR_WEIGHT")
    def validate_weights(cls, v):
        """Validate weight values."""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Weights must be between 0.0 and 1.0")
        return v

    def _get_secret_from_keyring(self, key: str) -> Optional[str]:
        """Get secret from system keyring."""
        if not KEYRING_AVAILABLE:
            return None

        try:
            return keyring.get_password(self.KEYRING_SERVICE_NAME, key)
        except Exception as e:
            logging.warning(f"Failed to retrieve {key} from keyring: {e}")
            return None

    def _get_secret_from_aws(self, key: str) -> Optional[str]:
        """Get secret from AWS Secrets Manager."""
        if not AWS_AVAILABLE:
            return None

        try:
            client = boto3.client("secretsmanager", region_name=self.AWS_SECRETS_REGION)
            secret_name = f"{self.AWS_SECRETS_PREFIX}{key.lower()}"

            response = client.get_secret_value(SecretId=secret_name)
            return response["SecretString"]

        except (ClientError, NoCredentialsError) as e:
            logging.warning(f"Failed to retrieve {key} from AWS Secrets Manager: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error retrieving {key} from AWS: {e}")
            return None

    def get_secret(self, key: str, default: str = "") -> str:
        """Get secret with fallback chain: env -> keyring/aws -> default."""
        # First try environment variable
        env_value = os.getenv(key)
        if env_value:
            return env_value

        # Try configured secrets backend
        if self.SECRETS_BACKEND == "keyring":
            secret = self._get_secret_from_keyring(key)
            if secret:
                return secret
        elif self.SECRETS_BACKEND == "aws":
            secret = self._get_secret_from_aws(key)
            if secret:
                return secret

        return default

    @property
    def github_token(self) -> str:
        """Get GitHub token from secure storage."""
        return self.get_secret("GITHUB_TOKEN", self.GITHUB_TOKEN)

    @property
    def chat_llm_api_key(self) -> str:
        """Get Chat LLM API key from secure storage."""
        return self.get_secret("CHAT_LLM_API_KEY", self.CHAT_LLM_API_KEY)

    @property
    def embedding_model_api_key(self) -> str:
        """Get embedding model API key from secure storage."""
        return self.get_secret("EMBEDDING_MODEL_API_KEY", self.EMBEDDING_MODEL_API_KEY)

    @property
    def qdrant_api_key(self) -> str:
        """Get Qdrant API key from secure storage."""
        return self.get_secret("QDRANT_API_KEY", self.QDRANT_API_KEY)

    def get_database_url(self) -> str:
        """Get database URL for Qdrant."""
        return self.QDRANT_URL

    def get_connection_config(self) -> Dict[str, Any]:
        """Get connection configuration for HTTP clients."""
        return {
            "timeout": self.REQUEST_TIMEOUT,
            "connection_pool_size": self.CONNECTION_POOL_SIZE,
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "http2_enabled": self.HTTP2_ENABLED,
        }

    def get_retry_config(self) -> Dict[str, Any]:
        """Get retry configuration."""
        return {
            "max_attempts": self.RETRY_MAX_ATTEMPTS,
            "initial_delay": self.RETRY_INITIAL_DELAY,
            "max_delay": self.RETRY_MAX_DELAY,
            "exponential_base": self.RETRY_EXPONENTIAL_BASE,
        }

    def get_circuit_breaker_config(self) -> Dict[str, Any]:
        """Get circuit breaker configuration."""
        return {
            "failure_threshold": self.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            "recovery_timeout": self.CIRCUIT_BREAKER_RECOVERY_TIMEOUT,
            "expected_exception_types": self.CIRCUIT_BREAKER_EXPECTED_EXCEPTION_TYPES,
        }

    def validate_configuration(self) -> List[str]:
        """Validate production configuration and return any issues."""
        issues = []

        # Check required secrets
        required_secrets = [
            ("GITHUB_TOKEN", self.github_token),
            ("EMBEDDING_MODEL_API_KEY", self.embedding_model_api_key),
            ("QDRANT_URL", self.QDRANT_URL),
        ]

        for secret_name, secret_value in required_secrets:
            if not secret_value:
                issues.append(f"Missing required secret: {secret_name}")

        # Check weights sum to 1.0
        if abs(self.BM25_WEIGHT + self.VECTOR_WEIGHT - 1.0) > 0.01:
            issues.append("BM25_WEIGHT and VECTOR_WEIGHT must sum to 1.0")

        # Check ports are available
        if self.METRICS_PORT == self.HEALTH_CHECK_PORT:
            issues.append("METRICS_PORT and HEALTH_CHECK_PORT cannot be the same")

        # Check batch settings
        if self.BATCH_SIZE <= 0:
            issues.append("BATCH_SIZE must be greater than 0")

        if self.MAX_BATCH_CONCURRENCY <= 0:
            issues.append("MAX_BATCH_CONCURRENCY must be greater than 0")

        return issues


# Create production settings instance
def get_production_settings() -> ProductionSettings:
    """Get production settings with validation."""
    settings = ProductionSettings()

    # Validate configuration
    issues = settings.validate_configuration()
    if issues:
        error_msg = "Production configuration issues found:\n" + "\n".join(
            f"- {issue}" for issue in issues
        )
        raise ValueError(error_msg)

    return settings
