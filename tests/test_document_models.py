# tests/test_document_models.py
"""
Unit tests for document data models
"""
import pytest
from datetime import datetime, timezone

from app.processors.document import (
    Document,
    Page,
    DocumentStatus,
    Partition,
    TableInfo,
    ChartInfo,
    TableInfoWithPage,
    ChartInfoWithPage,
    PartitionDetail,
    PartitionDetails
)


class TestTableInfo:
    """Test TableInfo model"""
    
    def test_create_table_info(self):
        """Test creating a TableInfo instance"""
        table = TableInfo(
            table_id="table_1_1",
            title="Test Table",
            summary="A test table summary"
        )
        
        assert table.table_id == "table_1_1"
        assert table.title == "Test Table"
        assert table.summary == "A test table summary"
    
    def test_table_info_to_dict(self):
        """Test TableInfo serialization"""
        table = TableInfo(
            table_id="table_1_1",
            title="Test Table",
            summary="A test table summary"
        )
        
        data = table.to_dict()
        assert data["table_id"] == "table_1_1"
        assert data["title"] == "Test Table"
        assert data["summary"] == "A test table summary"
    
    def test_table_info_from_dict(self):
        """Test TableInfo deserialization"""
        data = {
            "table_id": "table_1_1",
            "title": "Test Table",
            "summary": "A test table summary"
        }
        
        table = TableInfo.from_dict(data)
        assert table.table_id == "table_1_1"
        assert table.title == "Test Table"


class TestChartInfo:
    """Test ChartInfo model"""
    
    def test_create_chart_info(self):
        """Test creating a ChartInfo instance"""
        chart = ChartInfo(
            chart_id="chart_2_1",
            title="Sales Chart",
            chart_type="line",
            summary="Monthly sales data"
        )
        
        assert chart.chart_id == "chart_2_1"
        assert chart.chart_type == "line"
    
    def test_chart_info_to_dict(self):
        """Test ChartInfo serialization"""
        chart = ChartInfo(
            chart_id="chart_2_1",
            title="Sales Chart",
            chart_type="bar",
            summary="Quarterly revenue"
        )
        
        data = chart.to_dict()
        assert data["chart_type"] == "bar"
    
    def test_chart_info_from_dict(self):
        """Test ChartInfo deserialization"""
        data = {
            "chart_id": "chart_2_1",
            "title": "Sales Chart",
            "chart_type": "pie",
            "summary": "Market share"
        }
        
        chart = ChartInfo.from_dict(data)
        assert chart.chart_type == "pie"


class TestPage:
    """Test Page model"""
    
    def test_create_page(self):
        """Test creating a Page instance"""
        page = Page(
            page_number=1,
            image_path="/path/to/page_1.jpg",
            width=800,
            height=1000
        )
        
        assert page.page_number == 1
        assert page.image_path == "/path/to/page_1.jpg"
        assert page.width == 800
        assert page.height == 1000
    
    def test_page_with_tables(self, sample_page_with_tables):
        """Test page with tables"""
        page = sample_page_with_tables
        
        assert page.has_tables() is True
        assert page.get_table_count() == 2
        assert page.tables[0].table_id == "table_5_1"
    
    def test_page_with_charts(self, sample_page_with_charts):
        """Test page with charts"""
        page = sample_page_with_charts
        
        assert page.has_charts() is True
        assert page.get_chart_count() == 2
        assert page.charts[0].chart_type == "line"
    
    def test_page_without_tables_or_charts(self):
        """Test page without tables or charts"""
        page = Page(
            page_number=1,
            image_path="/path/to/page_1.jpg"
        )
        
        assert page.has_tables() is False
        assert page.has_charts() is False
        assert page.get_table_count() == 0
        assert page.get_chart_count() == 0
    
    def test_page_serialization(self, sample_page_with_tables):
        """Test Page serialization and deserialization"""
        page = sample_page_with_tables
        
        # Serialize
        data = page.to_dict()
        assert data["page_number"] == 5
        assert len(data["tables"]) == 2
        
        # Deserialize
        restored_page = Page.from_dict(data)
        assert restored_page.page_number == 5
        assert restored_page.has_tables() is True


class TestPartition:
    """Test Partition model"""
    
    def test_create_partition(self):
        """Test creating a Partition instance"""
        partition = Partition(
            partition_id=1,
            page_range=(1, 20),
            summary="First partition summary"
        )
        
        assert partition.partition_id == 1
        assert partition.page_range == (1, 20)
        assert partition.get_page_count() == 20
    
    def test_partition_page_count(self):
        """Test partition page count calculation"""
        partition = Partition(
            partition_id=2,
            page_range=(21, 35),
            summary="Second partition"
        )
        
        assert partition.get_page_count() == 15
    
    def test_partition_serialization(self):
        """Test Partition serialization"""
        partition = Partition(
            partition_id=1,
            page_range=(1, 20),
            summary="Test summary"
        )
        
        data = partition.to_dict()
        assert data["partition_id"] == 1
        assert data["page_range"] == [1, 20]
        
        restored = Partition.from_dict(data)
        assert restored.page_range == (1, 20)


class TestDocument:
    """Test Document model"""
    
    def test_create_small_document(self, sample_small_document):
        """Test creating a small document"""
        doc = sample_small_document
        
        assert doc.page_count == 10
        assert doc.is_large_document() is False
        assert doc.has_partitions() is False
        assert doc.status == DocumentStatus.READY
    
    def test_create_large_document(self, sample_large_document):
        """Test creating a large document"""
        doc = sample_large_document
        
        assert doc.page_count == 50
        assert doc.is_large_document() is True
        assert doc.has_partitions() is True
        assert len(doc.partitions) == 3
    
    def test_document_serialization(self, sample_small_document):
        """Test Document serialization and deserialization"""
        doc = sample_small_document
        
        # Serialize
        data = doc.to_dict()
        assert data["id"] == "doc_test_small"
        assert data["page_count"] == 10
        assert data["status"] == "ready"
        
        # Deserialize
        restored_doc = Document.from_dict(data)
        assert restored_doc.page_count == 10
        assert restored_doc.status == DocumentStatus.READY
    
    def test_document_with_partitions(self, sample_large_document):
        """Test document with partitions"""
        doc = sample_large_document
        
        # Check first partition
        assert doc.partitions[0].partition_id == 1
        assert doc.partitions[0].page_range == (1, 20)
        
        # Check pages are assigned to partitions
        page_1 = doc.pages[0]
        assert page_1.partition_id == 1
        
        page_25 = doc.pages[24]
        assert page_25.partition_id == 2


class TestPartitionDetails:
    """Test PartitionDetails models"""
    
    def test_partition_detail_with_tables_and_charts(self):
        """Test PartitionDetail with tables and charts"""
        tables = [
            TableInfoWithPage(
                table_id="table_5_1",
                page_number=5,
                title="Revenue Table",
                summary="Q1-Q4 revenue"
            )
        ]
        
        charts = [
            ChartInfoWithPage(
                chart_id="chart_7_1",
                page_number=7,
                title="Sales Chart",
                chart_type="line",
                summary="Monthly sales trend"
            )
        ]
        
        detail = PartitionDetail(
            partition_id=1,
            page_range=(1, 20),
            page_count=20,
            summary="First partition summary",
            tables=tables,
            charts=charts
        )
        
        assert detail.partition_id == 1
        assert len(detail.tables) == 1
        assert len(detail.charts) == 1
        assert detail.tables[0].page_number == 5
    
    def test_partition_details_serialization(self):
        """Test PartitionDetails serialization"""
        detail1 = PartitionDetail(
            partition_id=1,
            page_range=(1, 20),
            page_count=20,
            summary="Partition 1"
        )
        
        detail2 = PartitionDetail(
            partition_id=2,
            page_range=(21, 40),
            page_count=20,
            summary="Partition 2"
        )
        
        details = PartitionDetails(
            document_id="doc_test",
            document_name="test.pdf",
            total_partitions=2,
            partitions=[detail1, detail2]
        )
        
        data = details.to_dict()
        assert data["total_partitions"] == 2
        assert len(data["partitions"]) == 2
        
        restored = PartitionDetails.from_dict(data)
        assert restored.total_partitions == 2
        assert restored.partitions[0].partition_id == 1


class TestEdgeCases:
    """Test edge cases and error conditions"""
    
    def test_empty_page_list(self):
        """Test document with no pages"""
        doc = Document(
            id="doc_empty",
            name="empty.pdf",
            page_count=0,
            pages=[],
            status=DocumentStatus.ERROR
        )
        
        assert doc.page_count == 0
        assert len(doc.pages) == 0
        assert doc.is_large_document() is False
    
    def test_page_with_no_dimensions(self):
        """Test page without width/height"""
        page = Page(
            page_number=1,
            image_path="/path/to/page.jpg"
        )
        
        assert page.width is None
        assert page.height is None
    
    def test_partition_single_page(self):
        """Test partition with single page"""
        partition = Partition(
            partition_id=1,
            page_range=(1, 1),
            summary="Single page partition"
        )
        
        assert partition.get_page_count() == 1
    
    def test_document_boundary_20_pages(self):
        """Test document with exactly 20 pages (boundary condition)"""
        pages = [
            Page(page_number=i, image_path=f"/page_{i}.jpg")
            for i in range(1, 21)
        ]
        
        doc = Document(
            id="doc_20",
            name="twenty_pages.pdf",
            page_count=20,
            pages=pages,
            status=DocumentStatus.READY
        )
        
        # Exactly 20 pages should NOT be considered large
        assert doc.is_large_document() is False
    
    def test_document_boundary_21_pages(self):
        """Test document with 21 pages (just over boundary)"""
        pages = [
            Page(page_number=i, image_path=f"/page_{i}.jpg")
            for i in range(1, 22)
        ]
        
        doc = Document(
            id="doc_21",
            name="twentyone_pages.pdf",
            page_count=21,
            pages=pages,
            status=DocumentStatus.READY
        )
        
        # 21 pages should be considered large
        assert doc.is_large_document() is True
