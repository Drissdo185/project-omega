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
class Page:
    """Represents a single page as image"""
    page_number: int
    image_path: str  # Path to JPEG file
    width: Optional[int] = None
    height: Optional[int] = None
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "summary": self.summary,
            "width": self.width,
            "height": self.height
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Page':
        return cls(
            page_number=data["page_number"],
            image_path=data["image_path"],
            width=data.get("width"),
            height=data.get("height"),
            summary=data.get("summary", "")
        )

@dataclass
class Document:
    """Document with vision-based pages"""
    id: str
    name: str
    page_count: int
    pages: List[Page] = field(default_factory=list)
    folder: str = "Other"
    status: DocumentStatus = DocumentStatus.PROCESSING
    summary: Optional[str] = None  # Vision-generated summary
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
            folder=data["folder"],
            status=DocumentStatus(data["status"]),
            summary=data.get("summary"),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            pages=[Page.from_dict(p) for p in data.get("pages", [])]
        )