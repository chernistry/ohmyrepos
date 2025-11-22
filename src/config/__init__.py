"""Configuration package for Oh My Repos."""

from .production import ProductionSettings, get_production_settings

# Import the main config module
import sys
from pathlib import Path

# Add the src directory to Python path to access the main config
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    # Import from the parent src directory  
    parent_dir = Path(__file__).parent.parent
    config_file = parent_dir / "config.py"
    
    if config_file.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("main_config", config_file)
        main_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_config)
        
        # Extract classes and functions
        Settings = main_config.Settings
        GitHubConfig = main_config.GitHubConfig
        LLMConfig = main_config.LLMConfig
        OllamaConfig = main_config.OllamaConfig
        QdrantConfig = main_config.QdrantConfig
        EmbeddingConfig = main_config.EmbeddingConfig
        RerankerConfig = main_config.RerankerConfig
        SearchConfig = main_config.SearchConfig
        MonitoringConfig = main_config.MonitoringConfig
        SecurityConfig = main_config.SecurityConfig
        Environment = main_config.Environment
        LogLevel = main_config.LogLevel
        LLMProvider = main_config.LLMProvider
        EmbeddingProviderType = main_config.EmbeddingProviderType
        get_settings = main_config.get_settings
        
        # Use the main config system
        settings = get_settings()
    else:
        raise ImportError("Main config.py not found")
    
except (ImportError, AttributeError) as e:
    # Fallback to production settings
    settings = ProductionSettings()
    
    # Create compatibility aliases
    Settings = ProductionSettings
    GitHubConfig = dict
    LLMConfig = dict
    OllamaConfig = dict
    QdrantConfig = dict
    EmbeddingConfig = dict
    RerankerConfig = dict
    SearchConfig = dict
    MonitoringConfig = dict
    SecurityConfig = dict
    Environment = str
    LogLevel = str
    LLMProvider = str
    EmbeddingProviderType = str
    get_settings = get_production_settings

__all__ = [
    "ProductionSettings", 
    "get_production_settings", 
    "settings",
    "Settings",
    "GitHubConfig", 
    "LLMConfig",
    "OllamaConfig",
    "QdrantConfig",
    "EmbeddingConfig",
    "RerankerConfig",
    "SearchConfig",
    "MonitoringConfig",
    "SecurityConfig",
    "Environment",
    "LogLevel",
    "LLMProvider",
    "EmbeddingProviderType",
    "get_settings",
]
