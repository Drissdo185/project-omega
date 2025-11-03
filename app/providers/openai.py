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
        model_2stage: Optional[str] = None,
        model_3stage: Optional[str] = None
    ):
        """
        Initialize OpenAI provider with separate models for 2-stage and 3-stage flows

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            model_2stage: Model for 2-stage flow (small docs ≤20 pages) - defaults to gpt-4o-mini
            model_3stage: Model for 3-stage flow (large docs >20 pages) - defaults to gpt-5
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model_2stage = model_2stage or os.getenv("OPENAI_MODEL_2STAGE", "gpt-4o-mini")
        self.model_3stage = model_3stage or os.getenv("OPENAI_MODEL_3STAGE", "gpt-5-mini-2025-08-07")

        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=self.api_key)
        self.last_cost = None

        logger.info(f"Initialized OpenAI provider - 2-stage model: {self.model_2stage}, 3-stage model: {self.model_3stage}")

    async def process_text_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 3000,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Process text-only messages using the Responses API

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens to generate
            model: Specific model to use (if None, uses model_3stage)
            **kwargs: Additional arguments

        Returns:
            Response text from the model
        """
        try:
            # Use specified model or default to 3-stage model
            selected_model = model or self.model_3stage

            # Flatten messages to text input for Responses API
            input_data = messages

            response = self.client.responses.create(
                model=selected_model,
                input=input_data,
                max_output_tokens=max_tokens,
                **kwargs
            )

            self.last_cost = 0.0  # Cost calculation removed

            logger.debug(f"Used model: {selected_model}")
            return response.output_text.strip() if hasattr(response, "output_text") else ""

        except Exception as e:
            logger.error(f"OpenAI text processing error: {e}")
            raise

    async def process_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 3000,
        model: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Process multimodal messages (text + images) using the Responses API

        Args:
            messages: List of message dictionaries with multimodal content
            max_tokens: Maximum tokens to generate
            model: Specific model to use (if None, uses model_2stage)
            **kwargs: Additional arguments

        Returns:
            Response text from the model
        """
        try:
            formatted_messages = self._format_multimodal_messages(messages)

            # Use specified model or default to 2-stage model for vision
            selected_model = model or self.model_2stage

            response = self.client.responses.create(
                model=selected_model,
                input=formatted_messages,
                max_output_tokens=max_tokens,
                **kwargs
            )

            self.last_cost = 0.0  # Cost calculation removed

            logger.debug(f"Used model: {selected_model}")
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


    def get_last_cost(self) -> Optional[float]:
        """Get the cost of the last API call"""
        return self.last_cost

    def get_model_name(self) -> str:
        """Get the 2-stage model name"""
        return self.model_2stage

    def get_model_2stage(self) -> str:
        """Get the 2-stage model name"""
        return self.model_2stage

    def get_model_3stage(self) -> str:
        """Get the 3-stage model name"""
        return self.model_3stage
