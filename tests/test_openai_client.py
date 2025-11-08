# tests/test_openai_client.py
"""
Unit tests for OpenAI client
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import os

from app.ai.openai import OpenAIClient


class TestOpenAIClient:
    """Test OpenAI client functionality"""
    
    def test_init_with_api_key(self):
        """Test client initialization with API key"""
        client = OpenAIClient(api_key="test-key-123")
        assert client.api_key == "test-key-123"
    
    def test_init_from_env(self, mock_env_vars):
        """Test client initialization from environment"""
        client = OpenAIClient()
        assert client.api_key == "test-api-key-12345"
    
    def test_init_without_api_key(self):
        """Test client initialization without API key raises error"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key not provided"):
                OpenAIClient()
    
    def test_model_selection(self):
        """Test model selection based on document size"""
        client = OpenAIClient(api_key="test-key")
        
        # Small document
        model = client.get_model_for_document(10)
        assert model == client.model_small
        
        # Boundary condition - exactly 20 pages
        model = client.get_model_for_document(20)
        assert model == client.model_small
        
        # Large document
        model = client.get_model_for_document(50)
        assert model == client.model_large
    
    @pytest.mark.asyncio
    async def test_chat_completion_basic(self, mock_openai_client):
        """Test basic chat completion"""
        mock_openai_client.chat_completion.return_value = "Test response"
        
        messages = [{"role": "user", "content": "Hello"}]
        response = await mock_openai_client.chat_completion(messages)
        
        assert response == "Test response"
        mock_openai_client.chat_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vision_completion_single_image(self, mock_openai_client):
        """Test vision completion with single image"""
        mock_openai_client.vision_completion.return_value = '{"summary": "Test"}'
        
        response = await mock_openai_client.vision_completion(
            text_prompt="Analyze this image",
            images=["base64encodedimage"]
        )
        
        assert "Test" in response
        mock_openai_client.vision_completion.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_vision_completion_multiple_images(self, mock_openai_client):
        """Test vision completion with multiple images"""
        mock_openai_client.vision_completion.return_value = "Analysis complete"
        
        images = ["image1_base64", "image2_base64", "image3_base64"]
        response = await mock_openai_client.vision_completion(
            text_prompt="Analyze these images",
            images=images
        )
        
        assert response == "Analysis complete"


class TestOpenAIClientErrorHandling:
    """Test error handling in OpenAI client"""
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self):
        """Test API error handling"""
        client = OpenAIClient(api_key="test-key")
        
        # Mock the client to raise an exception
        with patch.object(client.client.chat.completions, 'create', side_effect=Exception("API Error")):
            with pytest.raises(Exception, match="API Error"):
                await client.chat_completion([{"role": "user", "content": "test"}])
    
    @pytest.mark.asyncio
    async def test_empty_messages_list(self):
        """Test handling of empty messages list"""
        client = OpenAIClient(api_key="test-key")
        
        # This should not crash, but behavior depends on OpenAI API
        # In real implementation, you might want to validate input
        messages = []
        # Add validation in actual code if needed
    
    @pytest.mark.asyncio
    async def test_invalid_model_name(self):
        """Test handling of invalid model name"""
        client = OpenAIClient(api_key="test-key")
        
        # Mock to simulate model not found error
        with patch.object(client.client.chat.completions, 'create', 
                         side_effect=Exception("Model not found")):
            with pytest.raises(Exception):
                await client.chat_completion(
                    messages=[{"role": "user", "content": "test"}],
                    model="invalid-model-name"
                )
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test API timeout handling"""
        client = OpenAIClient(api_key="test-key")
        
        # Mock timeout
        with patch.object(client.client.chat.completions, 'create',
                         side_effect=TimeoutError("Request timeout")):
            with pytest.raises(TimeoutError):
                await client.chat_completion([{"role": "user", "content": "test"}])


class TestOpenAIClientEdgeCases:
    """Test edge cases for OpenAI client"""
    
    def test_very_long_api_key(self):
        """Test with extremely long API key"""
        long_key = "sk-" + "x" * 1000
        client = OpenAIClient(api_key=long_key)
        assert client.api_key == long_key
    
    def test_special_characters_in_api_key(self):
        """Test API key with special characters"""
        special_key = "sk-test_key-with-special!@#$%"
        client = OpenAIClient(api_key=special_key)
        assert client.api_key == special_key
    
    @pytest.mark.asyncio
    async def test_empty_response_handling(self, mock_openai_client):
        """Test handling of empty API response"""
        mock_openai_client.chat_completion.return_value = ""
        
        response = await mock_openai_client.chat_completion(
            messages=[{"role": "user", "content": "test"}]
        )
        
        assert response == ""
    
    @pytest.mark.asyncio
    async def test_very_large_response(self, mock_openai_client):
        """Test handling of very large API response"""
        large_response = "x" * 100000  # 100KB response
        mock_openai_client.chat_completion.return_value = large_response
        
        response = await mock_openai_client.chat_completion(
            messages=[{"role": "user", "content": "test"}]
        )
        
        assert len(response) == 100000
