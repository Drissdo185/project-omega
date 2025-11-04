# ai/openai.py
import os
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from loguru import logger


class OpenAIClient:
    """Centralized OpenAI client for all AI operations"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = AsyncOpenAI(api_key=self.api_key)
        
        # Model selection based on document size
        self.model_small = "gpt-4o-mini-2024-07-18"  # For ≤20 pages
        self.model_large = "gpt-5-mini-2025-08-07"  # For >20 pages
        self.model_qa = "gpt-4o-mini-2024-07-18"      # For Q&A
    
    def get_model_for_document(self, page_count: int) -> str:
        """
        Select appropriate model based on document size
        
        Args:
            page_count: Number of pages in document
            
        Returns:
            Model name to use
        """
        if page_count <= 20:
            return self.model_small
        else:
            return self.model_large
    
    async def chat_completion(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.3
    ) -> str:
        """
        Generic chat completion method
        
        Args:
            messages: Chat messages in OpenAI format
            model: Model to use (defaults to model_qa)
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Returns:
            Response content as string
        """
        if model is None:
            model = self.model_qa
        
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise
    
    async def vision_completion(
        self,
        text_prompt: str,
        images: List[str],  # List of base64 encoded images
        model: Optional[str] = None,
        max_tokens: int = 1500,
        temperature: float = 0.3,
        detail: str = "high"
    ) -> str:
        """
        Vision completion with images
        
        Args:
            text_prompt: Text prompt
            images: List of base64 encoded images
            model: Model to use
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            detail: Image detail level ("low" or "high")
            
        Returns:
            Response content as string
        """
        if model is None:
            model = self.model_qa
        
        try:
            # Build content with text and images
            content = [{"type": "text", "text": text_prompt}]
            
            for img_base64 in images:
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img_base64}",
                        "detail": detail
                    }
                })
            
            response = await self.client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": content
                    }
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            logger.error(f"OpenAI Vision API error: {e}")
            raise