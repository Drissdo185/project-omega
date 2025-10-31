# app/models/document.py
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from enum import Enum

class DocumentStatus(Enum):
    PROCESSING = "processing"
    READY = "ready"
    ERROR = "error"

@dataclass
class Page:
    """Represents a single page within a combined image"""
    page_number: int
    image_path: str  # Path to combined JPEG file containing this page
    width: Optional[int] = None  # Original page width
    height: Optional[int] = None  # Original page height
    summary: str = ""
    isImage: bool = False  # Whether this page contains images/graphics
    combined_image_number: Optional[int] = None  # Which combined image this page is in
    grid_position: Optional[Dict[str, int]] = None  # {"row": 0, "col": 0} position in grid

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "summary": self.summary,
            "width": self.width,
            "height": self.height,
            "isImage": self.isImage,
            "combined_image_number": self.combined_image_number,
            "grid_position": self.grid_position
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Page':
        return cls(
            page_number=data["page_number"],
            image_path=data["image_path"],
            width=data.get("width"),
            height=data.get("height"),
            summary=data.get("summary", ""),
            isImage=data.get("isImage", False),
            combined_image_number=data.get("combined_image_number"),
            grid_position=data.get("grid_position")
        )

@dataclass
class Document:
    """Document with vision-based pages in combined images"""
    id: str
    name: str
    page_count: int
    pages: List[Page] = field(default_factory=list)
    combined_images: List[Dict[str, Any]] = field(default_factory=list)  # Info about combined image files
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
            "pages": [p.to_dict() for p in self.pages],
            "combined_images": self.combined_images
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
            combined_images=data.get("combined_images", [])
        )