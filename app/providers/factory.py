"""Provider factory for creating LLM provider instances"""

import os
from typing import Optional
from loguru import logger

from .base import BaseProvider
from .openai import OpenAIProvider


def create_provider(**kwargs) -> BaseProvider:
    """
    Create an OpenAI provider instance

    Args:
        **kwargs: Additional provider-specific arguments
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL (defaults to OPENAI_BASE_URL env var)
            model: Model name (defaults to OPENAI_MODEL env var or 'gpt-5')

    Returns:
        Initialized OpenAI provider instance

    Example:
        # Use environment variables
        provider = create_provider()

        # Custom configuration
        provider = create_provider(
            api_key='sk-...',
            base_url='https://custom-endpoint.com',
            model='gpt-5'
        )
    """
    logger.info("Creating OpenAI provider")
    return OpenAIProvider(**kwargs)


def create_provider_from_env() -> BaseProvider:
    """
    Create a provider instance from environment variables only

    Environment variables:
        OPENAI_API_KEY: API key (required)
        OPENAI_BASE_URL: Base URL (default: https://aiportalapi.stu-platform.live/use)
        OPENAI_MODEL: Model name (default: gpt-5)

    Returns:
        Initialized OpenAI provider instance
    """
    return OpenAIProvider(
        api_key=os.environ.get("OPENAI_API_KEY"),
        # base_url=os.environ.get("OPENAI_BASE_URL"),
        model_2stage=os.environ.get("OPENAI_MODEL_2STAGE","gpt-4o-mini-2024-07-18"),
        model_3stage=os.environ.get("OPENAI_MODEL_3STAGE","gpt-5-mini-2025-08-07")
    )
