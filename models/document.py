"""
Document and Page data models for Vision-Based RAG system.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid


@dataclass
class Page:
    """
    Represents a single page within a document.
    """
    page_number: int
    image_path: str
    width: int
    height: int
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Page instance to dictionary.
        """
        return {
            "page_number": self.page_number,
            "image_path": self.image_path,
            "width": self.width,
            "height": self.height,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Page':
        """
        Create Page instance from dictionary.
        """
        return cls(
            page_number=data["page_number"],
            image_path=data["image_path"],
            width=data["width"],
            height=data["height"],
        )
    
    def __repr__(self) -> str:
        return f"Page(number={self.page_number}, size={self.width}x{self.height})"

@dataclass
class Document:
    """Represents a document with its pages and metadata."""
    
    id: str
    name: str
    page_count: int
    status: str  # "processing", "ready", "failed"
    pages: List[Page] = field(default_factory=list)
    summary: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    
    def __post_init__(self):
        """Set timestamps if not provided."""
        if self.created_at is None:
            self.created_at = datetime.now(datetime.timezone.utc)
        if self.updated_at is None:
            self.updated_at = datetime.now(datetime.timezone.utc)
            
    @staticmethod
    def generate_id() -> str:
        """Generate a unique document ID."""
        return f"doc_{uuid.uuid4().hex[:12]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Document to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'name': self.name,
            'page_count': self.page_count,
            'status': self.status,
            'summary': self.summary,
            'pages': [page.to_dict() for page in self.pages],
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        """Create Document from dictionary."""
        # Parse pages
        pages = [Page.from_dict(p) for p in data.get('pages', [])]
        
        # Parse timestamps
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        updated_at = None
        if data.get('updated_at'):
            updated_at = datetime.fromisoformat(data['updated_at'])
        
        return cls(
            id=data['id'],
            name=data['name'],
            page_count=data['page_count'],
            status=data['status'],
            pages=pages,
            summary=data.get('summary'),
            created_at=created_at,
            updated_at=updated_at,
            metadata=data.get('metadata', {})
        )
    
    def add_page(self, page: Page) -> None:
        """Add a page to the document."""
        self.pages.append(page)
        self.page_count = len(self.pages)
        self.updated_at = datetime.now(datetime.timezone.utc)
        
    
    def get_page(self, page_number: int) -> Optional[Page]:
        """Get a specific page by number."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None
    
    def update_status(self, status: str) -> None:
        """Update document status."""
        self.status = status
        self.updated_at = datetime.now(datetime.timezone.utc)
    
    def set_summary(self, summary: str) -> None:
        """Set document summary."""
        self.summary = summary
        self.updated_at = datetime.now(datetime.timezone.utc)
    
    def is_ready(self) -> bool:
        """Check if document is ready for querying."""
        return self.status == "ready" and self.summary is not None
    
    def __repr__(self) -> str:
        return (f"Document(id={self.id}, name={self.name}, "
                f"pages={self.page_count}, status={self.status})")
            
    
        
    
    
    


