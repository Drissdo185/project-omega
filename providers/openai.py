"""
OpenAI vision provider using GPT-4o.
"""

import logging
import base64
from typing import List
from pathlib import Path
import time

try:
    from openai import OpenAI
    from openai import OpenAIError
except ImportError:
    raise ImportError("OpenAI library is required. Install with: pip install openai")

from providers.base import BaseProvider
from exceptions import VisionModelError
from core.config import get_settings

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseProvider):
    """
    OpenAI vision provider using GPT-4o.
    """

    def __init__(self, api_key: str = None, model: str = None, max_retries: int = 3):
        """
        Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (defaults to config).
            model: Model to use (defaults to config).
            max_retries: Maximum number of retries on failure.
        """
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.OPENAI_API_KEY
        self.model = model if model is not None else settings.VISION_MODEL
        self.max_retries = max_retries
        self.max_tokens = settings.MAX_TOKENS

        if not self.api_key:
            raise VisionModelError(
                "OpenAI API key is required",
                provider="openai"
            )

        self.client = OpenAI(api_key=self.api_key)
        logger.info(f"Initialized OpenAIProvider (model={self.model})")

    def _encode_image(self, image_path: str) -> str:
        """
        Encode image to base64.

        Args:
            image_path: Path to image file.

        Returns:
            str: Base64 encoded image.
        """
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def _create_image_content(self, image_path: str) -> dict:
        """
        Create image content for API request.

        Args:
            image_path: Path to image file.

        Returns:
            dict: Image content dictionary.
        """
        base64_image = self._encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }

    def analyze_image(self, image_path: str, prompt: str) -> str:
        """
        Analyze a single image with a prompt.

        Args:
            image_path: Path to the image file.
            prompt: Text prompt for the vision model.

        Returns:
            str: Model's response.

        Raises:
            VisionModelError: If the API call fails.
        """
        return self.analyze_images([image_path], prompt)

    def analyze_images(self, image_paths: List[str], prompt: str) -> str:
        """
        Analyze multiple images with a prompt.

        Args:
            image_paths: List of paths to image files.
            prompt: Text prompt for the vision model.

        Returns:
            str: Model's response.

        Raises:
            VisionModelError: If the API call fails.
        """
        if not image_paths:
            raise VisionModelError(
                "No images provided",
                provider="openai",
                model=self.model
            )

        # Verify all images exist
        for img_path in image_paths:
            if not Path(img_path).exists():
                raise VisionModelError(
                    f"Image not found: {img_path}",
                    provider="openai",
                    model=self.model
                )

        # Build message content
        content = [{"type": "text", "text": prompt}]

        # Add images
        for img_path in image_paths:
            content.append(self._create_image_content(img_path))

        # Make API call with retries
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(f"Calling OpenAI API (attempt {attempt}/{self.max_retries}, images={len(image_paths)})")

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "user",
                            "content": content
                        }
                    ],
                    max_tokens=self.max_tokens
                )

                result = response.choices[0].message.content
                logger.debug(f"OpenAI API call successful (response length: {len(result)})")
                return result

            except OpenAIError as e:
                last_error = e
                logger.warning(f"OpenAI API call failed (attempt {attempt}/{self.max_retries}): {e}")

                if attempt < self.max_retries:
                    # Exponential backoff
                    wait_time = 2 ** attempt
                    logger.debug(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

            except Exception as e:
                raise VisionModelError(
                    f"Unexpected error: {e}",
                    provider="openai",
                    model=self.model
                )

        # All retries failed
        raise VisionModelError(
            f"Failed after {self.max_retries} retries: {last_error}",
            provider="openai",
            model=self.model
        )

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self.model})"
