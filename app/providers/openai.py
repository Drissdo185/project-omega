"""OpenAI provider implementation for OpenAI-compatible endpoints (Responses API)"""

import os
import base64
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import OpenAI
from dotenv import load_dotenv

from .base import BaseProvider

load_dotenv()

class OpenAIProvider(BaseProvider):
    """
    OpenAI provider for LLM interactions
    Works with OpenAI-compatible API endpoints (Responses API)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini-2024-07-18"
    ):
        """
        Initialize OpenAI provider

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            model: Model name to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_name = model

        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.last_cost = None

        logger.info(f"Initialized OpenAI provider with model: {self.model_name}")

    async def process_text_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 3000,
        **kwargs
    ) -> str:
        """Process text-only messages using the Responses API"""
        try:
            # Flatten messages to text input for Responses API
            # You can concatenate messages or map them depending on your format
            input_data = messages

            response = self.client.responses.create(
                model=self.model_name,
                input=input_data,
                max_output_tokens=max_tokens,
                **kwargs
            )

            if hasattr(response, "usage"):
                self._calculate_cost(response.usage)

            return response.output_text.strip() if hasattr(response, "output_text") else ""

        except Exception as e:
            logger.error(f"OpenAI text processing error: {e}")
            raise

    async def process_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 3000,
        **kwargs
    ) -> str:
        """Process multimodal messages (text + images) using the Responses API"""
        try:
            formatted_messages = self._format_multimodal_messages(messages)

            response = self.client.responses.create(
                model=self.model_name,
                input=formatted_messages,
                max_output_tokens=max_tokens,
                **kwargs
            )

            if hasattr(response, "usage"):
                self._calculate_cost(response.usage)

            return response.output_text.strip() if hasattr(response, "output_text") else ""

        except Exception as e:
            logger.error(f"OpenAI multimodal processing error: {e}")
            raise

    def _format_multimodal_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert internal message format to Responses API input format"""
        formatted = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if isinstance(content, str):
                formatted.append({
                    "role": role,
                    "content": content
                })
            elif isinstance(content, list):
                formatted_content = []
                for item in content:
                    if item["type"] == "text":
                        formatted_content.append({
                            "type": "input_text",
                            "text": item["text"]
                        })
                    elif item["type"] == "image_path":
                        image_data = self._encode_image(item["image_path"])
                        formatted_content.append({
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_data}"
                        })

                formatted.append({
                    "role": role,
                    "content": formatted_content
                })
            else:
                formatted.append(msg)

        return formatted

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            logger.error(f"Failed to encode image {image_path}: {e}")
            raise

    def _calculate_cost(self, usage) -> None:
        """
        Calculate cost based on token usage (approximate)
        gpt-5 pricing (example):
        - Input: $2.50 per 1M tokens
        - Output: $10.00 per 1M tokens
        """
        try:
            input_cost = (usage.input_tokens / 1_000_000) * 2.50
            output_cost = (usage.output_tokens / 1_000_000) * 10.00
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
