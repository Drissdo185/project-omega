"""Base provider interface for LLM interactions"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers
    Supports both text-only and multimodal (vision) requests
    """

    @abstractmethod
    async def process_text_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        
        **kwargs
    ) -> str:
        """
        Process text-only messages and return response

        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Response text from the model
        """
        pass

    @abstractmethod
    async def process_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """
        Process multimodal messages (text + images) and return response

        Args:
            messages: List of message dicts with multimodal content
                     Content can include 'type': 'text' or 'image_path'
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Response text from the model
        """
        pass

    def get_last_cost(self) -> Optional[float]:
        """
        Get the cost of the last API call (if tracking is enabled)

        Returns:
            Cost in USD or None if not tracked
        """
        return None

    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name of the model being used

        Returns:
            Model name string
        """
        pass
