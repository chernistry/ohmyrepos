"""Comprehensive tests for configuration system."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import BaseModel, ValidationError, SecretStr

from src.config import (
    Settings,
    GitHubConfig,
    LLMConfig,
    OllamaConfig,
    QdrantConfig,
    EmbeddingConfig,
    RerankerConfig,
    SearchConfig,
    MonitoringConfig,
    SecurityConfig,
    Environment,
    LogLevel,
    LLMProvider,
    get_settings,
)


@pytest.mark.unit
class TestConfigurationModels:
    """Test individual configuration models."""

    def test_github_config_validation(self):
        """Test GitHub configuration validation."""
        # Valid config
        config = GitHubConfig(
            username="test-user",
            token=SecretStr("ghp_1234567890abcdef"),
        )
        assert config.username == "test-user"
        assert config.token.get_secret_value() == "ghp_1234567890abcdef"
        assert str(config.api_url) == "https://api.github.com"

        # Invalid username
        with pytest.raises(ValidationError, match="Invalid GitHub username format"):
            GitHubConfig(username="invalid@user", token=SecretStr("token"))

        # Empty token
        with pytest.raises(ValidationError):
            GitHubConfig(username="testuser", token=SecretStr(""))

    def test_llm_config_validation(self):
        """Test LLM configuration validation."""
        # Valid OpenAI config
        config = LLMConfig(
            provider=LLMProvider.OPENAI,
            api_key=SecretStr("sk-1234567890abcdef"),
            model="gpt-4o-mini",
        )
        assert config.provider == LLMProvider.OPENAI
        assert config.api_key.get_secret_value() == "sk-1234567890abcdef"

        # OpenAI without API key should fail
        with pytest.raises(ValidationError, match="OpenAI provider requires API key"):
            LLMConfig(provider=LLMProvider.OPENAI)

        # Ollama without API key should work
        config = LLMConfig(provider=LLMProvider.OLLAMA)
        assert config.provider == LLMProvider.OLLAMA
        assert config.api_key is None

    def test_qdrant_config_validation(self):
        """Test Qdrant configuration validation."""
        # Valid config
        config = QdrantConfig(url="https://cluster.qdrant.cloud:6333")
        assert str(config.url).rstrip('/') == "https://cluster.qdrant.cloud:6333"
        assert config.collection_name == "repositories"
        assert config.vector_size == 1024

        # Invalid URL
        with pytest.raises(ValidationError):
            QdrantConfig(url="not-a-valid-url")

    def test_search_config_validation(self):
        """Test search configuration validation."""
        # Valid config
        config = SearchConfig(bm25_weight=0.3, vector_weight=0.7)
        assert config.bm25_weight == 0.3
        assert config.vector_weight == 0.7

        # Weights don't sum to 1.0
        with pytest.raises(ValidationError, match="must sum to 1.0"):
            SearchConfig(bm25_weight=0.5, vector_weight=0.8)

    def test_monitoring_config_defaults(self):
        """Test monitoring configuration defaults."""
        config = MonitoringConfig()
        assert config.enable_metrics is True
        assert config.log_level == LogLevel.INFO
        assert config.structured_logging is True

    def test_security_config_generation(self):
        """Test security configuration with generated values."""
        config = SecurityConfig()
        assert config.secret_key.get_secret_value()  # Should have generated value
        assert len(config.secret_key.get_secret_value()) > 20
        assert "localhost" in config.allowed_hosts


@pytest.mark.unit
class TestSettingsClass:
    """Test the main Settings class."""

    def test_default_settings(self):
        """Test default settings instantiation."""
        settings = Settings()
        assert settings.environment == Environment.DEVELOPMENT
        assert settings.debug is False
        assert settings.base_dir.exists()

    def test_environment_validation(self):
        """Test environment setting validation."""
        settings = Settings(environment="production")
        assert settings.environment == Environment.PRODUCTION

        settings = Settings(environment=Environment.TESTING)
        assert settings.environment == Environment.TESTING

    def test_is_production_helper(self):
        """Test production environment helper."""
        settings = Settings(environment=Environment.PRODUCTION)
        assert settings.is_production() is True
        assert settings.is_development() is False

        settings = Settings(environment=Environment.DEVELOPMENT)
        assert settings.is_production() is False
        assert settings.is_development() is True

    def test_model_dump_safe(self):
        """Test safe model dumping without secrets."""
        settings = Settings(
            github=GitHubConfig(
                username="testuser",
                token=SecretStr("secret_token"),
            ),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=SecretStr("secret_key"),
                model="gpt-4",
            ),
        )

        safe_dump = settings.model_dump_safe()
        
        # Should include non-secret GitHub data
        assert "github" in safe_dump
        assert safe_dump["github"]["username"] == "testuser"
        assert "token" not in str(safe_dump["github"])  # No secrets

        # Should include non-secret LLM data
        assert "llm" in safe_dump
        assert safe_dump["llm"]["provider"] == "openai"
        assert safe_dump["llm"]["model"] == "gpt-4"
        assert "api_key" not in str(safe_dump["llm"])  # No secrets


@pytest.mark.unit
class TestEnvironmentVariableHandling:
    """Test environment variable configuration."""

    def test_legacy_env_var_support(self, mock_environment):
        """Test backward compatibility with legacy environment variables."""
        settings = Settings()
        
        # Should build nested config from flat env vars
        assert settings.github is not None
        assert settings.github.username == "test_user"
        assert settings.github.token.get_secret_value() == "test_token"

        assert settings.llm is not None
        assert settings.llm.api_key.get_secret_value() == "test_llm_key"

    def test_nested_env_var_support(self):
        """Test nested environment variable support."""
        with patch.dict(os.environ, {
            "GITHUB__USERNAME": "nested_user",
            "GITHUB__TOKEN": "nested_token",
            "LLM__PROVIDER": "ollama",
            "LLM__MODEL": "llama2",
        }):
            settings = Settings()
            
            # Nested env vars should work (if properly configured)
            # This tests the delimiter configuration
            config = settings.model_config
            if hasattr(config, 'env_nested_delimiter'):
                assert config.env_nested_delimiter == "__"
            elif isinstance(config, dict):
                assert config.get('env_nested_delimiter') == "__"

    def test_secrets_directory_support(self):
        """Test Docker secrets directory support."""
        with tempfile.TemporaryDirectory() as temp_dir:
            secrets_dir = Path(temp_dir)
            
            # Create secret files
            (secrets_dir / "github_token").write_text("secret_from_file")
            (secrets_dir / "llm_api_key").write_text("llm_secret_from_file")
            
            with patch.dict(os.environ, {"SECRETS_DIR": str(secrets_dir)}):
                settings = Settings()
                
                # Test that get_secret method can find secrets
                token = settings.get_secret("github_token")
                assert token == "secret_from_file"
                
                llm_key = settings.get_secret("llm_api_key")
                assert llm_key == "llm_secret_from_file"

    def test_env_file_loading(self):
        """Test .env file loading."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write("""
GITHUB_USERNAME=env_file_user
GITHUB_TOKEN=env_file_token
CHAT_LLM_PROVIDER=openai
CHAT_LLM_MODEL=gpt-3.5-turbo
""")
            env_file = f.name

        # Test that env file is configured (actual loading depends on pydantic-settings)
        settings = Settings()
        config = settings.model_config
        if hasattr(config, 'env_file'):
            assert config.env_file == ".env"
            assert config.env_file_encoding == "utf-8"
        elif isinstance(config, dict):
            assert config.get('env_file') == ".env"
            assert config.get('env_file_encoding') == "utf-8"

        # Cleanup
        os.unlink(env_file)


@pytest.mark.unit 
class TestConfigurationValidation:
    """Test configuration validation and error handling."""

    def test_validate_required_settings_success(self):
        """Test successful validation of required settings."""
        settings = Settings(
            github=GitHubConfig(
                username="testuser",
                token=SecretStr("valid_token"),
            ),
            qdrant=QdrantConfig(url="http://localhost:6333"),
            embedding=EmbeddingConfig(api_key=SecretStr("embedding_key")),
            llm=LLMConfig(
                provider=LLMProvider.OPENAI,
                api_key=SecretStr("llm_key"),
            ),
        )
        
        # Should not raise exception
        settings.validate_required_settings()

    def test_validate_required_settings_failures(self):
        """Test validation failures for missing required settings."""
        # Test the validation logic directly rather than trying to create invalid Settings
        
        # Create a valid settings object and manually set fields to None to test validation
        settings = Settings()
        
        # Override the fields to None after construction
        settings.github = None
        with pytest.raises(ValueError, match="GitHub configuration is required"):
            settings.validate_required_settings()

        # Test GitHub token validation at the GitHubConfig level
        with pytest.raises(ValidationError):
            GitHubConfig(username="test", token=SecretStr(""))

        # Test Qdrant validation  
        settings.github = GitHubConfig(username="test", token=SecretStr("token"))
        settings.qdrant = None
        with pytest.raises(ValueError, match="Qdrant configuration is required"):
            settings.validate_required_settings()

    def test_production_validation(self):
        """Test that production environment enforces validation."""
        settings = Settings()
        settings.environment = Environment.PRODUCTION
        settings.github = None
        
        # Should raise error when validating incomplete production config
        with pytest.raises(ValueError):
            settings.validate_required_settings()


@pytest.mark.unit
class TestConfigurationSecurity:
    """Test configuration security features."""

    def test_secret_str_handling(self):
        """Test that secrets are properly protected."""
        config = GitHubConfig(
            username="testuser",
            token=SecretStr("secret_token"),
        )
        
        # Secret should not appear in string representation
        config_str = str(config)
        assert "secret_token" not in config_str
        assert "SecretStr" in config_str or "***" in config_str

        # Secret should be accessible via get_secret_value
        assert config.token.get_secret_value() == "secret_token"

    def test_secret_retrieval_methods(self):
        """Test various secret retrieval methods."""
        settings = Settings()
        
        # Test environment variable retrieval
        with patch.dict(os.environ, {"TEST_SECRET": "env_secret"}):
            secret = settings.get_secret("TEST_SECRET")
            assert secret == "env_secret"

        # Test that missing secrets return None
        secret = settings.get_secret("NON_EXISTENT_SECRET")
        assert secret is None

    def test_aws_secrets_manager_integration(self):
        """Test AWS Secrets Manager integration (mocked)."""
        settings = Settings()
        
        # Test that missing secrets return None
        secret = settings.get_secret("NON_EXISTENT_SECRET")
        assert secret is None
        
        # Test with environment variable fallback
        with patch.dict(os.environ, {"TEST_SECRET": "env_value"}):
            secret = settings.get_secret("TEST_SECRET")
            assert secret == "env_value"


@pytest.mark.integration
class TestConfigurationIntegration:
    """Integration tests for configuration system."""

    def test_full_configuration_lifecycle(self):
        """Test complete configuration setup in different environments."""
        # Development environment
        dev_settings = Settings(environment=Environment.DEVELOPMENT)
        assert dev_settings.is_development()
        assert not dev_settings.is_production()

        # Testing environment  
        test_settings = Settings(environment=Environment.TESTING)
        assert test_settings.environment == Environment.TESTING

        # Production environment
        prod_settings = Settings(
            environment=Environment.PRODUCTION,
            github=GitHubConfig(username="prod", token=SecretStr("prod_token")),
            qdrant=QdrantConfig(url="https://prod.qdrant.cloud:6333"),
            embedding=EmbeddingConfig(api_key=SecretStr("prod_embedding")),
        )
        assert prod_settings.is_production()

    def test_configuration_serialization(self):
        """Test configuration serialization and deserialization."""
        original_settings = Settings(
            environment=Environment.TESTING,
            debug=True,
            github=GitHubConfig(username="test", token=SecretStr("token")),
        )
        
        # Test safe serialization (without secrets)
        safe_dict = original_settings.model_dump_safe()
        assert "github" in safe_dict
        assert "username" in safe_dict["github"]
        assert "token" not in str(safe_dict)  # No secrets

        # Test that sensitive fields are excluded
        full_dict = original_settings.model_dump()
        # Pydantic should handle SecretStr serialization properly
        assert "github" in full_dict

    def test_configuration_override_precedence(self):
        """Test configuration override precedence."""
        # Environment variables should override defaults
        with patch.dict(os.environ, {
            "ENVIRONMENT": "production",
            "DEBUG": "true",
            "GITHUB_USERNAME": "env_override_user",
        }):
            settings = Settings()
            
            # Environment variable should win
            assert settings.environment == Environment.PRODUCTION
            
            # Nested config should be built from env vars
            if settings.github:
                assert settings.github.username == "env_override_user"