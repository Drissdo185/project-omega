# tests/conftest.py
"""
Pytest configuration and shared fixtures
"""
import pytest
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock

# Add app directory to path
app_dir = Path(__file__).parent.parent / "app"
sys.path.insert(0, str(app_dir))

from app.processors.document import (
    Document,
    Page,
    DocumentStatus,
    Partition,
    TableInfo,
    ChartInfo
)


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables"""
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key-12345")
    monkeypatch.setenv("FLEX_RAG_DATA_LOCATION", "/tmp/test_flex_rag")


@pytest.fixture
def sample_small_document():
    """Create a sample small document (≤20 pages)"""
    pages = []
    for i in range(1, 11):  # 10 pages
        pages.append(Page(
            page_number=i,
            image_path=f"/tmp/test_doc/pages/page_{i}.jpg",
            width=800,
            height=1000,
            summary=f"This is page {i} content summary.",
            tables=[],
            charts=[]
        ))
    
    return Document(
        id="doc_test_small",
        name="test_document_small.pdf",
        page_count=10,
        pages=pages,
        partitions=[],
        status=DocumentStatus.READY,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_large_document():
    """Create a sample large document (>20 pages) with partitions"""
    pages = []
    for i in range(1, 51):  # 50 pages
        partition_id = ((i - 1) // 20) + 1  # 20 pages per partition
        pages.append(Page(
            page_number=i,
            image_path=f"/tmp/test_doc/pages/page_{i}.jpg",
            width=800,
            height=1000,
            summary=f"This is page {i} content summary.",
            partition_id=partition_id,
            tables=[],
            charts=[]
        ))
    
    partitions = [
        Partition(partition_id=1, page_range=(1, 20), summary="Partition 1 summary"),
        Partition(partition_id=2, page_range=(21, 40), summary="Partition 2 summary"),
        Partition(partition_id=3, page_range=(41, 50), summary="Partition 3 summary")
    ]
    
    return Document(
        id="doc_test_large",
        name="test_document_large.pdf",
        page_count=50,
        pages=pages,
        partitions=partitions,
        status=DocumentStatus.READY,
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )


@pytest.fixture
def sample_page_with_tables():
    """Create a sample page with tables"""
    tables = [
        TableInfo(
            table_id="table_5_1",
            title="Revenue by Quarter",
            summary="Shows quarterly revenue for 2023"
        ),
        TableInfo(
            table_id="table_5_2",
            title="Employee Count",
            summary="Department-wise employee distribution"
        )
    ]
    
    return Page(
        page_number=5,
        image_path="/tmp/test_doc/pages/page_5.jpg",
        width=800,
        height=1000,
        summary="This page contains financial tables and employee data.",
        tables=tables,
        charts=[]
    )


@pytest.fixture
def sample_page_with_charts():
    """Create a sample page with charts"""
    charts = [
        ChartInfo(
            chart_id="chart_7_1",
            title="Sales Trend",
            chart_type="line",
            summary="Monthly sales trend over 12 months"
        ),
        ChartInfo(
            chart_id="chart_7_2",
            title="Market Share",
            chart_type="pie",
            summary="Market share distribution by product"
        )
    ]
    
    return Page(
        page_number=7,
        image_path="/tmp/test_doc/pages/page_7.jpg",
        width=800,
        height=1000,
        summary="This page shows sales analytics and market data.",
        tables=[],
        charts=charts
    )


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    mock_client.model_small = "gpt-4-vision-preview"
    mock_client.model_large = "gpt-4-vision-preview"
    mock_client.model_qa = "gpt-4-vision-preview"
    
    # Mock async methods
    mock_client.chat_completion = AsyncMock()
    mock_client.vision_completion = AsyncMock()
    mock_client.get_model_for_document = Mock(return_value="gpt-4-vision-preview")
    
    return mock_client


@pytest.fixture
def temp_test_dir(tmp_path):
    """Create a temporary test directory"""
    test_dir = tmp_path / "test_documents"
    test_dir.mkdir()
    
    # Create subdirectories
    (test_dir / "documents").mkdir()
    (test_dir / "cache").mkdir()
    (test_dir / "uploads").mkdir()
    
    return test_dir


@pytest.fixture
def sample_pdf_path():
    """Path to a sample PDF file for testing"""
    # This would be a real PDF file in a test fixtures directory
    return Path(__file__).parent / "fixtures" / "sample.pdf"
