# """
# Anthropic vision provider using Claude.
# """

# import logging
# import base64
# from typing import List
# from pathlib import Path
# import time

# try:
#     from anthropic import Anthropic
#     from anthropic import AnthropicError
# except ImportError:
#     raise ImportError("Anthropic library is required. Install with: pip install anthropic")

# from providers.base import BaseProvider
# from exceptions import VisionModelError
# from core.config import get_settings

# logger = logging.getLogger(__name__)


# class AnthropicProvider(BaseProvider):
#     """
#     Anthropic vision provider using Claude.
#     """

#     def __init__(self, api_key: str = None, model: str = None, max_retries: int = 3):
#         """
#         Initialize Anthropic provider.

#         Args:
#             api_key: Anthropic API key (defaults to config).
#             model: Model to use (defaults to config).
#             max_retries: Maximum number of retries on failure.
#         """
#         settings = get_settings()
#         self.api_key = api_key if api_key is not None else settings.ANTHROPIC_API_KEY
#         self.model = model if model is not None else settings.VISION_MODEL
#         self.max_retries = max_retries
#         self.max_tokens = settings.MAX_TOKENS

#         if not self.api_key:
#             raise VisionModelError(
#                 "Anthropic API key is required",
#                 provider="anthropic"
#             )

#         self.client = Anthropic(api_key=self.api_key)
#         logger.info(f"Initialized AnthropicProvider (model={self.model})")

#     def _encode_image(self, image_path: str) -> str:
#         """
#         Encode image to base64.

#         Args:
#             image_path: Path to image file.

#         Returns:
#             str: Base64 encoded image.
#         """
#         with open(image_path, 'rb') as f:
#             return base64.standard_b64encode(f.read()).decode('utf-8')

#     def _get_media_type(self, image_path: str) -> str:
#         """
#         Get media type for image.

#         Args:
#             image_path: Path to image file.

#         Returns:
#             str: Media type (e.g., 'image/jpeg').
#         """
#         suffix = Path(image_path).suffix.lower()
#         media_types = {
#             '.jpg': 'image/jpeg',
#             '.jpeg': 'image/jpeg',
#             '.png': 'image/png',
#             '.gif': 'image/gif',
#             '.webp': 'image/webp'
#         }
#         return media_types.get(suffix, 'image/jpeg')

#     def _create_image_content(self, image_path: str) -> dict:
#         """
#         Create image content for API request.

#         Args:
#             image_path: Path to image file.

#         Returns:
#             dict: Image content dictionary.
#         """
#         base64_image = self._encode_image(image_path)
#         media_type = self._get_media_type(image_path)

#         return {
#             "type": "image",
#             "source": {
#                 "type": "base64",
#                 "media_type": media_type,
#                 "data": base64_image
#             }
#         }

#     def analyze_image(self, image_path: str, prompt: str) -> str:
#         """
#         Analyze a single image with a prompt.

#         Args:
#             image_path: Path to the image file.
#             prompt: Text prompt for the vision model.

#         Returns:
#             str: Model's response.

#         Raises:
#             VisionModelError: If the API call fails.
#         """
#         return self.analyze_images([image_path], prompt)

#     def analyze_images(self, image_paths: List[str], prompt: str) -> str:
#         """
#         Analyze multiple images with a prompt.

#         Args:
#             image_paths: List of paths to image files.
#             prompt: Text prompt for the vision model.

#         Returns:
#             str: Model's response.

#         Raises:
#             VisionModelError: If the API call fails.
#         """
#         if not image_paths:
#             raise VisionModelError(
#                 "No images provided",
#                 provider="anthropic",
#                 model=self.model
#             )

#         # Verify all images exist
#         for img_path in image_paths:
#             if not Path(img_path).exists():
#                 raise VisionModelError(
#                     f"Image not found: {img_path}",
#                     provider="anthropic",
#                     model=self.model
#                 )

#         # Build message content
#         content = []

#         # Add images first
#         for img_path in image_paths:
#             content.append(self._create_image_content(img_path))

#         # Add text prompt
#         content.append({
#             "type": "text",
#             "text": prompt
#         })

#         # Make API call with retries
#         last_error = None
#         for attempt in range(1, self.max_retries + 1):
#             try:
#                 logger.debug(f"Calling Anthropic API (attempt {attempt}/{self.max_retries}, images={len(image_paths)})")

#                 response = self.client.messages.create(
#                     model=self.model,
#                     max_tokens=self.max_tokens,
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": content
#                         }
#                     ]
#                 )

#                 # Extract text from response
#                 result = response.content[0].text
#                 logger.debug(f"Anthropic API call successful (response length: {len(result)})")
#                 return result

#             except AnthropicError as e:
#                 last_error = e
#                 logger.warning(f"Anthropic API call failed (attempt {attempt}/{self.max_retries}): {e}")

#                 if attempt < self.max_retries:
#                     # Exponential backoff
#                     wait_time = 2 ** attempt
#                     logger.debug(f"Retrying in {wait_time} seconds...")
#                     time.sleep(wait_time)

#             except Exception as e:
#                 raise VisionModelError(
#                     f"Unexpected error: {e}",
#                     provider="anthropic",
#                     model=self.model
#                 )

#         # All retries failed
#         raise VisionModelError(
#             f"Failed after {self.max_retries} retries: {last_error}",
#             provider="anthropic",
#             model=self.model
#         )

#     def __repr__(self) -> str:
#         return f"AnthropicProvider(model={self.model})"
