"""
Factory for creating vision AI providers.
"""

import logging
from typing import Dict, Type

from providers.base import BaseProvider
from providers.openai import OpenAIProvider
from providers.anthropic import AnthropicProvider
from exceptions import VisionModelError, ConfigurationError
from core.config import get_settings

logger = logging.getLogger(__name__)


class ProviderFactory:
    """
    Factory for creating vision AI provider instances.
    """

    # Registry of providers
    _providers: Dict[str, Type[BaseProvider]] = {
        'openai': OpenAIProvider,
        'anthropic': AnthropicProvider,
    }

    @classmethod
    def get_provider(cls, provider_name: str = None) -> BaseProvider:
        """
        Get a provider instance.

        Args:
            provider_name: Provider name ('openai' or 'anthropic').
                          If None, uses the configured provider.

        Returns:
            BaseProvider: Provider instance.

        Raises:
            VisionModelError: If provider is not available.
            ConfigurationError: If provider configuration is invalid.
        """
        settings = get_settings()

        # Use configured provider if none specified
        if provider_name is None:
            provider_name = settings.VISION_PROVIDER

        provider_name = provider_name.lower()

        # Get provider class
        provider_class = cls._providers.get(provider_name)

        if provider_class is None:
            raise VisionModelError(
                f"Unknown provider: {provider_name}. "
                f"Available providers: {', '.join(cls._providers.keys())}",
                provider=provider_name
            )

        # Validate API key is configured
        try:
            settings.validate_api_key()
        except ValueError as e:
            raise ConfigurationError(str(e))

        # Create and return provider instance
        try:
            provider = provider_class()
            logger.debug(f"Created provider: {provider}")
            return provider
        except Exception as e:
            raise VisionModelError(
                f"Failed to create provider: {e}",
                provider=provider_name
            )

    @classmethod
    def register_provider(cls, name: str, provider_class: Type[BaseProvider]) -> None:
        """
        Register a new provider.

        Args:
            name: Provider name.
            provider_class: Provider class.
        """
        name = name.lower()
        cls._providers[name] = provider_class
        logger.info(f"Registered provider '{name}': {provider_class.__name__}")

    @classmethod
    def available_providers(cls) -> list:
        """
        Get list of available providers.

        Returns:
            list: List of provider names.
        """
        return list(cls._providers.keys())

    @classmethod
    def __repr__(cls) -> str:
        return f"ProviderFactory(available={cls.available_providers()})"
