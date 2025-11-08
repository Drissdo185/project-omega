# tests/test_vision_analyzer.py
"""
Unit tests for Vision Analyzer
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, mock_open
import json
from pathlib import Path

from app.ai.vision_analyzer import VisionAnalyzer
from app.processors.document import Document, Page, Partition, DocumentStatus


class TestVisionAnalyzer:
    """Test VisionAnalyzer functionality"""
    
    def test_init_with_storage_root(self, mock_openai_client):
        """Test analyzer initialization with custom storage root"""
        analyzer = VisionAnalyzer(mock_openai_client, storage_root="/custom/path")
        # Use Path comparison to handle Windows/Unix path differences
        assert analyzer.storage_root == Path("/custom/path")
    
    def test_init_from_env(self, mock_openai_client, mock_env_vars):
        """Test analyzer initialization from environment"""
        analyzer = VisionAnalyzer(mock_openai_client)
        # Check path contains expected directory (handles Windows/Unix differences)
        assert "test_flex_rag" in str(analyzer.storage_root)
    
    def test_encode_image(self, mock_openai_client, tmp_path):
        """Test image encoding to base64"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        # Create a test image file
        image_path = tmp_path / "test_image.jpg"
        image_path.write_bytes(b"fake image content")
        
        # Encode
        encoded = analyzer._encode_image(str(image_path))
        
        assert isinstance(encoded, str)
        assert len(encoded) > 0
    
    @pytest.mark.asyncio
    async def test_analyze_single_page_success(self, mock_openai_client, tmp_path):
        """Test successful single page analysis"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        # Mock image file
        image_path = tmp_path / "page_1.jpg"
        image_path.write_bytes(b"fake image")
        
        # Mock API response
        mock_response = json.dumps({
            "summary": "This is page 1 summary",
            "tables": [
                {
                    "table_id": "table_1_1",
                    "title": "Test Table",
                    "summary": "Table summary"
                }
            ],
            "charts": [
                {
                    "chart_id": "chart_1_1",
                    "title": "Test Chart",
                    "chart_type": "line",
                    "summary": "Chart summary"
                }
            ]
        })
        
        mock_openai_client.vision_completion.return_value = mock_response
        
        # Analyze
        result = await analyzer._analyze_single_page(str(image_path), 1)
        
        assert result["summary"] == "This is page 1 summary"
        assert len(result["tables"]) == 1
        assert len(result["charts"]) == 1
        assert result["tables"][0]["table_id"] == "table_1_1"
    
    @pytest.mark.asyncio
    async def test_analyze_single_page_no_tables_charts(self, mock_openai_client, tmp_path):
        """Test page analysis with no tables or charts"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        image_path = tmp_path / "page_1.jpg"
        image_path.write_bytes(b"fake image")
        
        mock_response = json.dumps({
            "summary": "Page with only text",
            "tables": [],
            "charts": []
        })
        
        mock_openai_client.vision_completion.return_value = mock_response
        
        result = await analyzer._analyze_single_page(str(image_path), 1)
        
        assert result["summary"] == "Page with only text"
        assert len(result["tables"]) == 0
        assert len(result["charts"]) == 0
    
    @pytest.mark.asyncio
    async def test_analyze_single_page_json_with_code_fence(self, mock_openai_client, tmp_path):
        """Test handling of JSON response with code fence markers"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        image_path = tmp_path / "page_1.jpg"
        image_path.write_bytes(b"fake image")
        
        # Response with ```json markers
        mock_response = '''```json
        {
            "summary": "Test summary",
            "tables": [],
            "charts": []
        }
        ```'''
        
        mock_openai_client.vision_completion.return_value = mock_response
        
        result = await analyzer._analyze_single_page(str(image_path), 1)
        
        assert result["summary"] == "Test summary"
    
    @pytest.mark.asyncio
    async def test_analyze_single_page_error_handling(self, mock_openai_client, tmp_path):
        """Test error handling in single page analysis"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        image_path = tmp_path / "page_1.jpg"
        image_path.write_bytes(b"fake image")
        
        # Mock API error
        mock_openai_client.vision_completion.side_effect = Exception("API Error")
        
        # Should return empty result, not crash
        result = await analyzer._analyze_single_page(str(image_path), 1)
        
        assert result["summary"] == ""
        assert result["tables"] == []
        assert result["charts"] == []
    
    @pytest.mark.asyncio
    async def test_analyze_single_page_malformed_json(self, mock_openai_client, tmp_path):
        """Test handling of malformed JSON response"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        image_path = tmp_path / "page_1.jpg"
        image_path.write_bytes(b"fake image")
        
        # Malformed JSON
        mock_openai_client.vision_completion.return_value = '{"summary": "test", "tables":'
        
        # Should handle gracefully
        result = await analyzer._analyze_single_page(str(image_path), 1)
        
        assert result["summary"] == ""


class TestVisionAnalyzerPartitions:
    """Test partition-related functionality"""
    
    @pytest.mark.asyncio
    async def test_analyze_partition_batch_basic(self, mock_openai_client, tmp_path):
        """Test basic partition batch analysis"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        # Create sample pages
        pages = []
        for i in range(1, 6):
            image_path = tmp_path / f"page_{i}.jpg"
            image_path.write_bytes(b"fake image")
            
            page = Page(
                page_number=i,
                image_path=str(image_path),
                summary=f"Page {i} summary"
            )
            pages.append(page)
        
        # Mock API response
        mock_response = json.dumps({
            "summary": "Partition summary covering pages 1-5"
        })
        
        mock_openai_client.vision_completion.return_value = mock_response
        
        # Analyze
        result = await analyzer._analyze_partition_batch(pages, 1, (1, 5))
        
        assert result["summary"] == "Partition summary covering pages 1-5"
    
    @pytest.mark.asyncio
    async def test_analyze_partition_batch_large_partition(self, mock_openai_client, tmp_path):
        """Test partition analysis with >10 pages (sampling)"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        # Create 15 pages
        pages = []
        for i in range(1, 16):
            image_path = tmp_path / f"page_{i}.jpg"
            image_path.write_bytes(b"fake image")
            
            page = Page(
                page_number=i,
                image_path=str(image_path),
                summary=f"Page {i} summary"
            )
            pages.append(page)
        
        mock_response = json.dumps({
            "summary": "Large partition summary"
        })
        
        mock_openai_client.vision_completion.return_value = mock_response
        
        result = await analyzer._analyze_partition_batch(pages, 1, (1, 15))
        
        # Should still work with sampling
        assert "summary" in result
    
    @pytest.mark.asyncio
    async def test_analyze_partition_batch_error_handling(self, mock_openai_client, tmp_path):
        """Test error handling in partition analysis"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        # Create pages
        pages = []
        for i in range(1, 4):
            image_path = tmp_path / f"page_{i}.jpg"
            image_path.write_bytes(b"fake image")
            
            page = Page(
                page_number=i,
                image_path=str(image_path),
                summary=f"Page {i} summary"
            )
            pages.append(page)
        
        # Mock API error
        mock_openai_client.vision_completion.side_effect = Exception("API Error")
        
        # Should return empty summary
        result = await analyzer._analyze_partition_batch(pages, 1, (1, 3))
        
        assert result["summary"] == ""


class TestVisionAnalyzerDocumentAnalysis:
    """Test full document analysis"""
    
    @pytest.mark.asyncio
    async def test_analyze_small_document(self, mock_openai_client, sample_small_document, tmp_path):
        """Test analyzing a small document (≤20 pages)"""
        analyzer = VisionAnalyzer(mock_openai_client, storage_root=str(tmp_path))
        
        # Create document directory
        doc_dir = tmp_path / "documents" / sample_small_document.id
        doc_dir.mkdir(parents=True)
        
        # Create fake image files
        for page in sample_small_document.pages:
            image_path = Path(page.image_path)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"fake image")
        
        # Mock API responses
        page_response = json.dumps({
            "summary": "Page analysis result",
            "tables": [],
            "charts": []
        })
        
        mock_openai_client.vision_completion.return_value = page_response
        
        # Analyze
        result_doc = await analyzer.analyze_document(sample_small_document)
        
        # Verify
        assert result_doc.page_count == 10
        assert all(page.summary != "" for page in result_doc.pages)
    
    @pytest.mark.asyncio
    async def test_analyze_large_document(self, mock_openai_client, sample_large_document, tmp_path):
        """Test analyzing a large document (>20 pages)"""
        analyzer = VisionAnalyzer(mock_openai_client, storage_root=str(tmp_path))
        
        # Create document directory
        doc_dir = tmp_path / "documents" / sample_large_document.id
        doc_dir.mkdir(parents=True)
        
        # Create fake image files
        for page in sample_large_document.pages:
            image_path = Path(page.image_path)
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"fake image")
        
        # Mock API responses
        page_response = json.dumps({
            "summary": "Page analysis result",
            "tables": [],
            "charts": []
        })
        
        partition_response = json.dumps({
            "summary": "Partition analysis result"
        })
        
        async def mock_vision_completion(*args, **kwargs):
            # Return different responses based on context
            if "partition" in kwargs.get("text_prompt", "").lower():
                return partition_response
            return page_response
        
        mock_openai_client.vision_completion.side_effect = mock_vision_completion
        
        # Analyze
        result_doc = await analyzer.analyze_document(sample_large_document)
        
        # Verify
        assert result_doc.page_count == 50
        assert result_doc.has_partitions() is True
        assert all(partition.summary != "" for partition in result_doc.partitions)


class TestVisionAnalyzerEdgeCases:
    """Test edge cases in vision analyzer"""
    
    @pytest.mark.asyncio
    async def test_analyze_page_image_not_found(self, mock_openai_client):
        """Test handling of missing image file"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        # Non-existent image path - should return empty result, not raise error
        result = await analyzer._analyze_single_page("/nonexistent/page.jpg", 1)
        
        # The code catches FileNotFoundError and returns default empty result
        assert result["summary"] == ""
        assert result["tables"] == []
        assert result["charts"] == []
    
    @pytest.mark.asyncio
    async def test_analyze_partition_no_pages(self, mock_openai_client):
        """Test partition analysis with empty page list"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        result = await analyzer._analyze_partition_batch([], 1, (1, 1))
        
        # Should handle gracefully
        assert result["summary"] == ""
    
    @pytest.mark.asyncio
    async def test_save_metadata_io_error(self, mock_openai_client, sample_small_document, tmp_path):
        """Test metadata save with IO error"""
        analyzer = VisionAnalyzer(mock_openai_client)
        
        # Try to save to read-only location (simulate IO error)
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(IOError):
                analyzer._save_metadata(sample_small_document, tmp_path)
