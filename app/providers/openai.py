"""OpenAI provider implementation for OpenAI-compatible endpoints"""

import os
import base64
from typing import List, Dict, Any, Optional
from loguru import logger
from openai import OpenAI
import openai
from dotenv import load_dotenv

from .base import BaseProvider

load_dotenv()
class OpenAIProvider(BaseProvider):
    """
    OpenAI provider for LLM interactions
    Works with OpenAI-compatible API endpoints
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize OpenAI provider

        Args:
            api_key: API key (defaults to OPENAI_API_KEY env var)
            base_url: Base URL for API endpoint (defaults to OPENAI_BASE_URL env var)
            model: Model name to use (defaults to OPENAI_MODEL env var)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://aiportalapi.stu-platform.live/use")
        self.model_name = model or os.getenv("OPENAI_MODEL", "GPT-5")

        if not self.api_key:
            raise ValueError("API key is required. Set OPENAI_API_KEY environment variable.")
        
        if not self.api_key.strip():
            raise ValueError("API key cannot be empty or whitespace.")

        # Initialize OpenAI client
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key.strip(),
                base_url=self.base_url.strip() if self.base_url else None,
                timeout=60.0,  # 60 second timeout
                max_retries=2   # Retry failed requests twice
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"Could not create OpenAI client: {e}")

        self.last_cost = None
        logger.info(f"✅ OpenAI provider initialized successfully")
        logger.info(f"   Base URL: {self.base_url}")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   API Key: {self.api_key[:10]}...{self.api_key[-4:]}" if len(self.api_key) > 14 else "   API Key: [hidden]")

    async def process_text_messages(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 2000,
        **kwargs
    ) -> str:
        """Process text-only messages"""
        try:
            logger.debug(f"Processing text messages with model: {self.model_name}")
            logger.debug(f"Messages count: {len(messages)}, Max tokens: {max_tokens}")
            
            # GPT-5 only supports temperature=1, use it as default
            temperature = kwargs.get('temperature', 1.0)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k != 'temperature'}
            )

            # Validate response
            if not response or not response.choices:
                logger.error("API returned empty response or no choices")
                raise ValueError("Empty response from API")
            
            # Track cost if usage is available
            if hasattr(response, 'usage') and response.usage:
                self._calculate_cost(response.usage)
                logger.debug(f"Tokens used - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}")
            
            # Get content
            content = response.choices[0].message.content
            
            if content is None:
                logger.error("API returned None content in text processing")
                raise ValueError("API returned None content")
            
            result = content.strip()
            logger.debug(f"✅ Text processing successful, response length: {len(result)} chars")
            
            return result

        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ValueError(f"API Error: {e}")
        except openai.APIConnectionError as e:
            logger.error(f"OpenAI connection error: {e}")
            raise ConnectionError(f"Connection Error: {e}")
        except openai.RateLimitError as e:
            logger.error(f"OpenAI rate limit error: {e}")
            raise ValueError(f"Rate Limit Error: {e}")
        except Exception as e:
            logger.error(f"Unexpected error in text processing: {type(e).__name__}: {e}")
            raise

    async def process_multimodal_messages(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int = 3000,  # Increased default for detailed defect analysis
        **kwargs
    ) -> str:
        """
        Process multimodal messages (text + images)
        
        Default max_tokens increased to 3000 for defect detection use cases
        which require detailed descriptions of visual anomalies.
        """
        try:
            # Convert messages to OpenAI format
            logger.debug(f"Formatting multimodal messages for {self.model_name}")
            formatted_messages = self._format_multimodal_messages(messages)
            
            logger.info(f"📤 Sending multimodal request:")
            logger.info(f"   Model: {self.model_name}")
            logger.info(f"   Messages: {len(formatted_messages)}")
            logger.info(f"   Max tokens: {max_tokens}")
            
            # Count images in messages
            image_count = 0
            for msg in formatted_messages:
                if isinstance(msg.get('content'), list):
                    image_count += sum(1 for item in msg['content'] if item.get('type') == 'image_url')
            logger.info(f"   Images: {image_count}")

            # GPT-5 only supports temperature=1, use it as default
            temperature = kwargs.get('temperature', 1.0)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=formatted_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                **{k: v for k, v in kwargs.items() if k != 'temperature'}
            )

            # Validate response structure
            if not response:
                logger.error("❌ API returned None response")
                raise ValueError("API returned None response")
            
            if not response.choices:
                logger.error("❌ API response has no choices")
                raise ValueError("API response has no choices")
            
            if not response.choices[0].message:
                logger.error("❌ API response choice has no message")
                raise ValueError("API response has no message")

            # Track cost if usage is available
            if hasattr(response, 'usage') and response.usage:
                self._calculate_cost(response.usage)
                logger.info(f"💰 Token usage - Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}, Cost: ${self.last_cost:.6f}")

            # Validate response content
            content = response.choices[0].message.content
            
            if content is None:
                logger.error("❌ API returned None content - Model may not support vision/multimodal")
                logger.error(f"   Response object: {response}")
                logger.error(f"   Model used: {self.model_name}")
                logger.error(f"   Base URL: {self.base_url}")
                raise ValueError("API returned None content - check if model supports vision/multimodal requests")
            
            if not content.strip():
                finish_reason = response.choices[0].finish_reason if hasattr(response.choices[0], 'finish_reason') else 'unknown'
                logger.error("❌ API returned empty content after strip")
                logger.error(f"   Raw content: '{content}'")
                logger.error(f"   Response finish reason: {finish_reason}")
                logger.error(f"   Model: {self.model_name}")
                
                # Check finish reason
                if finish_reason == "length":
                    raise ValueError("Response was cut off due to max_tokens limit. Please increase max_tokens.")
                elif finish_reason == "content_filter":
                    raise ValueError("Response was filtered by content policy.")
                else:
                    raise ValueError(f"API returned empty response (finish_reason: {finish_reason})")
            
            logger.info(f"✅ Multimodal processing successful: {len(content)} characters")
            return content.strip()

        except openai.APIError as e:
            logger.error(f"❌ OpenAI API error: {e}")
            logger.error(f"   Status code: {getattr(e, 'status_code', 'unknown')}")
            logger.error(f"   Error type: {getattr(e, 'type', 'unknown')}")
            raise ValueError(f"API Error: {e}")
        except openai.APIConnectionError as e:
            logger.error(f"❌ OpenAI connection error: {e}")
            raise ConnectionError(f"Connection Error: Could not reach API at {self.base_url}")
        except openai.RateLimitError as e:
            logger.error(f"❌ OpenAI rate limit error: {e}")
            raise ValueError(f"Rate Limit Error: {e}")
        except Exception as e:
            logger.error(f"❌ Unexpected error in multimodal processing: {type(e).__name__}: {e}")
            logger.error(f"   Model: {self.model_name}")
            logger.error(f"   Base URL: {self.base_url}")
            raise

    def _format_multimodal_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert our internal message format to OpenAI format"""
        formatted = []

        for idx, msg in enumerate(messages):
            try:
                if msg["role"] == "system":
                    formatted.append({
                        "role": "system",
                        "content": msg["content"]
                    })
                    logger.debug(f"Formatted system message {idx+1}")
                    
                elif msg["role"] == "user":
                    # Handle multimodal content
                    content = msg.get("content")

                    if isinstance(content, str):
                        # Simple text message
                        formatted.append({
                            "role": "user",
                            "content": content
                        })
                        logger.debug(f"Formatted text user message {idx+1}: {len(content)} chars")
                        
                    elif isinstance(content, list):
                        # Multimodal content (text + images)
                        formatted_content = []
                        text_count = 0
                        image_count = 0

                        for item_idx, item in enumerate(content):
                            if item["type"] == "text":
                                formatted_content.append({
                                    "type": "text",
                                    "text": item["text"]
                                })
                                text_count += 1
                                
                            elif item["type"] == "image_path":
                                # Read image and encode to base64
                                try:
                                    image_path = item["image_path"]
                                    logger.debug(f"Encoding image: {image_path}")
                                    image_data = self._encode_image(image_path)
                                    
                                    formatted_content.append({
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/jpeg;base64,{image_data}",
                                            "detail": item.get("detail", "high")  # Use "high" for better OCR
                                        }
                                    })
                                    image_count += 1
                                    logger.debug(f"✅ Successfully encoded image {item_idx+1}")
                                    
                                except Exception as img_error:
                                    logger.error(f"❌ Failed to encode image {item_idx+1}: {img_error}")
                                    raise

                        formatted.append({
                            "role": "user",
                            "content": formatted_content
                        })
                        logger.debug(f"Formatted multimodal user message {idx+1}: {text_count} text parts, {image_count} images")
                        
                else:
                    # Assistant or other roles
                    formatted.append(msg)
                    logger.debug(f"Formatted {msg['role']} message {idx+1}")
                    
            except Exception as e:
                logger.error(f"❌ Error formatting message {idx+1}: {e}")
                raise ValueError(f"Failed to format message {idx+1}: {e}")

        logger.debug(f"✅ Formatted {len(formatted)} messages total")
        return formatted

    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 with size validation"""
        try:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                image_size_mb = len(image_bytes) / (1024 * 1024)
                
                logger.debug(f"Image size: {image_size_mb:.2f} MB")
                
                # Warn if image is very large (over 10MB)
                if image_size_mb > 10:
                    logger.warning(f"⚠️ Large image detected ({image_size_mb:.2f} MB) - may cause API issues")
                    
                encoded = base64.b64encode(image_bytes).decode('utf-8')
                encoded_size_mb = len(encoded) / (1024 * 1024)
                
                logger.debug(f"Base64 encoded size: {encoded_size_mb:.2f} MB")
                
                return encoded
                
        except FileNotFoundError:
            logger.error(f"❌ Image file not found: {image_path}")
            raise ValueError(f"Image file not found: {image_path}")
        except Exception as e:
            logger.error(f"❌ Failed to encode image {image_path}: {e}")
            raise ValueError(f"Failed to encode image: {e}")

    def _calculate_cost(self, usage) -> None:
        """
        Calculate cost based on token usage
        
        GPT-5 pricing (approximate):
        - Input: $2.50 per 1M tokens
        - Output: $10.00 per 1M tokens
        
        Note: Vision API calls (with images) consume more tokens due to image processing.
        High-detail images use more tokens for better quality analysis.
        For defect detection, high-detail mode is recommended despite higher cost.
        """
        try:
            input_cost = (usage.prompt_tokens / 1_000_000) * 2.50
            output_cost = (usage.completion_tokens / 1_000_000) * 10.00
            self.last_cost = input_cost + output_cost
            
            # Log detailed token breakdown for defect detection optimization
            logger.debug(f"Token usage breakdown:")
            logger.debug(f"  - Input tokens: {usage.prompt_tokens}")
            logger.debug(f"  - Output tokens: {usage.completion_tokens}")
            logger.debug(f"  - Input cost: ${input_cost:.6f}")
            logger.debug(f"  - Output cost: ${output_cost:.6f}")
            logger.debug(f"  - Total cost: ${self.last_cost:.6f}")
        except Exception as e:
            logger.debug(f"Failed to calculate cost: {e}")
            self.last_cost = None

    def get_last_cost(self) -> Optional[float]:
        """Get the cost of the last API call"""
        return self.last_cost

    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name
