"""OpenAI provider implementation for OpenAI-compatible endpoints"""

import os
import base64
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import OpenAI
import openai

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider for LLM interactions
    Works with OpenAI-compatible API endpoints
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "gpt-5"
    ):
        """
        Initialize OpenAI provider

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API endpoint (defaults to OPENAI_BASE_URL env var)
            model: Model name to use
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://aiportalapi.stu-platform.live/use")
        self.model_name = model

        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable.")

        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.last_cost = None
        logger.info(f"Initialized OpenAI provider with base_url: {self.base_url}, model: {self.model_name}")

    async def process_text_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Process text-only messages"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            # Track cost if usage is available
            if hasattr(response, 'usage'):
                self._calculate_cost(response.usage)

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI text processing error: {e}")
            raise

    async def process_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 1000,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Process multimodal messages (text + images)"""
        try:
            # Convert messages to OpenAI format
            formatted_messages = self._format_multimodal_messages(messages)

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )

            # Track cost if usage is available
            if hasattr(response, 'usage'):
                self._calculate_cost(response.usage)

            return response.choices[0].message.content.strip()

        except Exception as e:
            logger.error(f"OpenAI multimodal processing error: {e}")
            raise

    def _format_multimodal_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert our internal message format to OpenAI format"""
        formatted = []

        for msg in messages:
            if msg["role"] == "system":
                formatted.append({
                    "role": "system",
                    "content": msg["content"]
                })
            elif msg["role"] == "user":
                # Handle multimodal content
                content = msg.get("content")

                if isinstance(content, str):
                    # Simple text message
                    formatted.append({
                        "role": "user",
                        "content": content
                    })
                elif isinstance(content, list):
                    # Multimodal content (text + images)
                    formatted_content = []

                    for item in content:
                        if item["type"] == "text":
                            formatted_content.append({
                                "type": "text",
                                "text": item["text"]
                            })
                        elif item["type"] == "image_path":
                            # Read image and encode to base64
                            image_data = self._encode_image(item["image_path"])
                            formatted_content.append({
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}",
                                    "detail": item.get("detail", "auto")
                                }
                            })

                    formatted.append({
                        "role": "user",
                        "content": formatted_content
                    })
            else:
                # Assistant or other roles
                formatted.append(msg)

        return formatted

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def _calculate_cost(self, usage) -> None:
        """
        Calculate cost based on token usage
        gpt-5 pricing (approximate):
        - Input: $2.50 per 1M tokens
        - Output: $10.00 per 1M tokens
        """
        try:
            input_cost = (usage.prompt_tokens / 1_000_000) * 2.50
            output_cost = (usage.completion_tokens / 1_000_000) * 10.00
            self.last_cost = input_cost + output_cost
        except Exception as e:
            logger.debug(f"Failed to calculate cost: {e}")
            self.last_cost = None

    def get_last_cost(self) -> Optional[float]:
        """Get the cost of the last API call"""
        return self.last_cost

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name
