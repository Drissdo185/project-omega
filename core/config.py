"""
Configuration management for Project Omega using Pydantic Settings.
"""

from pathlib import Path
from typing import Literal, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    All settings can be configured via environment variables or .env file.
    """

    # Storage settings
    STORAGE_PATH: Path = Field(
        default=Path("./docpixie_data"),
        description="Base path for document storage"
    )

    # Vision provider settings
    VISION_PROVIDER: Literal["openai", "anthropic"] = Field(
        default="openai",
        description="Vision model provider (openai or anthropic)"
    )

    VISION_MODEL: str = Field(
        default="gpt-4o",
        description="Vision model to use"
    )

    # API keys
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )

    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )

    # Image processing settings
    IMAGE_DPI: int = Field(
        default=150,
        ge=72,
        le=600,
        description="DPI for PDF to image conversion"
    )

    IMAGE_QUALITY: int = Field(
        default=90,
        ge=1,
        le=100,
        description="JPEG quality (1-100)"
    )

    # Query settings
    MAX_PAGES_PER_SELECTION: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of pages to select per query"
    )

    MAX_TOKENS: int = Field(
        default=4096,
        ge=256,
        le=128000,
        description="Maximum tokens for model responses"
    )

    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    @field_validator("STORAGE_PATH")
    @classmethod
    def validate_storage_path(cls, v: Path) -> Path:
        """Ensure storage path is absolute."""
        if not v.is_absolute():
            v = Path.cwd() / v
        return v

    @field_validator("VISION_PROVIDER")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate vision provider."""
        v = v.lower()
        if v not in ["openai", "anthropic"]:
            raise ValueError(f"Invalid VISION_PROVIDER: {v}. Must be 'openai' or 'anthropic'")
        return v

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        v = v.upper()
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v not in valid_levels:
            raise ValueError(f"Invalid LOG_LEVEL: {v}. Must be one of {valid_levels}")
        return v

    def validate_api_key(self) -> None:
        """
        Validate that the appropriate API key is configured for the selected provider.

        Raises:
            ValueError: If the required API key is not set.
        """
        if self.VISION_PROVIDER == "openai" and not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required when VISION_PROVIDER is 'openai'")
        if self.VISION_PROVIDER == "anthropic" and not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY is required when VISION_PROVIDER is 'anthropic'")

    def get_documents_path(self) -> Path:
        """Get the path to the documents directory."""
        return self.STORAGE_PATH / "documents"

    def __repr__(self) -> str:
        """String representation (hiding API keys)."""
        return (f"Settings(provider={self.VISION_PROVIDER}, model={self.VISION_MODEL}, "
                f"storage={self.STORAGE_PATH}, dpi={self.IMAGE_DPI})")


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get the global settings instance (singleton pattern).

    Returns:
        Settings: The application settings.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """
    Reload settings from environment (useful for testing).

    Returns:
        Settings: The reloaded settings instance.
    """
    global _settings
    _settings = Settings()
    return _settings
