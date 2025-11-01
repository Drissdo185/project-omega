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
    tables: List[TableInfo] = field(default_factory=list)
    charts: List[ChartInfo] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "summary": self.summary,
            "width": self.width,
            "height": self.height,
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
class Document:
    """Document with vision-based pages"""
    id: str
    name: str
    page_count: int
    pages: List[Page] = field(default_factory=list)
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
            "pages": [p.to_dict() for p in self.pages]
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
            pages=[Page.from_dict(p) for p in data.get("pages", [])]
        )