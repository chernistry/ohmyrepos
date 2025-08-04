"""Configuration package for Oh My Repos."""

from .production import ProductionSettings, get_production_settings

# Create a settings instance for compatibility
settings = ProductionSettings()

__all__ = ["ProductionSettings", "get_production_settings", "settings"]
