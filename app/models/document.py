# app/models/document.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional
from enum import Enum

class DocumentStatus(Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"


@dataclass
class TableInfo:
    """Minimal table information"""
    table_id: str
    title: str
    summary: str
    
    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "title": self.title,
            "summary": self.summary
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TableInfo':
        return cls(
            table_id=data["table_id"],
            title=data["title"],
            summary=data["summary"]
        )


@dataclass
class ChartInfo:
    """Minimal chart information"""
    chart_id: str
    title: str
    chart_type: str  # line, bar, pie, scatter, area
    summary: str
    
    def to_dict(self) -> dict:
        return {
            "chart_id": self.chart_id,
            "title": self.title,
            "chart_type": self.chart_type,
            "summary": self.summary
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'ChartInfo':
        return cls(
            chart_id=data["chart_id"],
            title=data["title"],
            chart_type=data["chart_type"],
            summary=data["summary"]
        )


@dataclass
class Page:
    """Page with tables and charts detection"""
    page_number: int
    image_path: str
    width: Optional[int] = None
    height: Optional[int] = None
    summary: str = ""
    partition_id: Optional[int] = None  # Link to partition (for large documents >20 pages)
    tables: List[TableInfo] = field(default_factory=list)
    charts: List[ChartInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "summary": self.summary,
            "width": self.width,
            "height": self.height,
            "partition_id": self.partition_id,
            "tables": [t.to_dict() for t in self.tables],
            "charts": [c.to_dict() for c in self.charts]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Page':
        return cls(
            page_number=data["page_number"],
            image_path=data["image_path"],
            width=data.get("width"),
            height=data.get("height"),
            summary=data.get("summary", ""),
            partition_id=data.get("partition_id"),
            tables=[TableInfo.from_dict(t) for t in data.get("tables", [])],
            charts=[ChartInfo.from_dict(c) for c in data.get("charts", [])]
        )
    
    def has_tables(self) -> bool:
        """Check if page has any tables"""
        return len(self.tables) > 0
    
    def has_charts(self) -> bool:
        """Check if page has any charts"""
        return len(self.charts) > 0
    
    def get_table_count(self) -> int:
        """Get number of tables on page"""
        return len(self.tables)
    
    def get_chart_count(self) -> int:
        """Get number of charts on page"""
        return len(self.charts)


@dataclass
class Partition:
    """Logical grouping of pages for large documents (>20 pages)"""
    partition_id: int  # 1-based partition number
    page_range: tuple  # (start_page, end_page) inclusive
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "partition_id": self.partition_id,
            "page_range": list(self.page_range),
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Partition':
        return cls(
            partition_id=data["partition_id"],
            page_range=tuple(data["page_range"]),
            summary=data.get("summary", "")
        )

    def get_page_count(self) -> int:
        """Get number of pages in this partition"""
        return self.page_range[1] - self.page_range[0] + 1


@dataclass
class TableInfoWithPage:
    """Table information with page number for partition_details.json"""
    table_id: str
    page_number: int
    title: str
    summary: str

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "page_number": self.page_number,
            "title": self.title,
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TableInfoWithPage':
        return cls(
            table_id=data["table_id"],
            page_number=data["page_number"],
            title=data["title"],
            summary=data["summary"]
        )


@dataclass
class ChartInfoWithPage:
    """Chart information with page number for partition_details.json"""
    chart_id: str
    page_number: int
    title: str
    chart_type: str
    summary: str

    def to_dict(self) -> dict:
        return {
            "chart_id": self.chart_id,
            "page_number": self.page_number,
            "title": self.title,
            "chart_type": self.chart_type,
            "summary": self.summary
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChartInfoWithPage':
        return cls(
            chart_id=data["chart_id"],
            page_number=data["page_number"],
            title=data["title"],
            chart_type=data["chart_type"],
            summary=data["summary"]
        )


@dataclass
class PartitionDetail:
    """Detailed partition information with aggregated tables and charts"""
    partition_id: int
    page_range: tuple  # (start_page, end_page) inclusive
    page_count: int
    summary: str
    tables: List[TableInfoWithPage] = field(default_factory=list)
    charts: List[ChartInfoWithPage] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "partition_id": self.partition_id,
            "page_range": list(self.page_range),
            "page_count": self.page_count,
            "summary": self.summary,
            "tables": [t.to_dict() for t in self.tables],
            "charts": [c.to_dict() for c in self.charts]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PartitionDetail':
        return cls(
            partition_id=data["partition_id"],
            page_range=tuple(data["page_range"]),
            page_count=data["page_count"],
            summary=data["summary"],
            tables=[TableInfoWithPage.from_dict(t) for t in data.get("tables", [])],
            charts=[ChartInfoWithPage.from_dict(c) for c in data.get("charts", [])]
        )


@dataclass
class PartitionDetails:
    """Container for partition_details.json file structure"""
    document_id: str
    document_name: str
    total_partitions: int
    partitions: List[PartitionDetail] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "total_partitions": self.total_partitions,
            "partitions": [p.to_dict() for p in self.partitions]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PartitionDetails':
        return cls(
            document_id=data["document_id"],
            document_name=data["document_name"],
            total_partitions=data["total_partitions"],
            partitions=[PartitionDetail.from_dict(p) for p in data.get("partitions", [])]
        )


@dataclass
class Document:
    """Document with vision-based pages"""
    id: str
    name: str
    page_count: int
    pages: List[Page] = field(default_factory=list)
    partitions: List[Partition] = field(default_factory=list)  # For large documents (>20 pages)
    status: DocumentStatus = DocumentStatus.PROCESSING
    summary: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "page_count": self.page_count,
            "status": self.status.value,
            "summary": self.summary,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "pages": [p.to_dict() for p in self.pages],
            "partitions": [part.to_dict() for part in self.partitions]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Document':
        return cls(
            id=data["id"],
            name=data["name"],
            page_count=data["page_count"],
            status=DocumentStatus(data["status"]),
            summary=data.get("summary"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            pages=[Page.from_dict(p) for p in data.get("pages", [])],
            partitions=[Partition.from_dict(p) for p in data.get("partitions", [])]
        )

    def is_large_document(self) -> bool:
        """Check if document should use partition-based approach (>20 pages)"""
        return self.page_count > 20

    def has_partitions(self) -> bool:
        """Check if document has partition summaries"""
        return len(self.partitions) > 0