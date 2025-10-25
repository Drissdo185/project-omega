""" AI analysis documents analysis, labeling, and and metadata extraction """

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class DocumentCategory(str, Enum):
    """Document categories for classification"""
    HR_POLICY = "hr_policy"
    FINANCIAL_REPORT = "financial_report"
    GENERAL = "general"
    UNKNOWN = "unknown"
    

class ContentType(str, Enum):
    """Types of content found in documents"""
    TEXT = "text"
    TABLE = "table"
    CHART = "chart"
    FORM = "form"
    IMAGE = "image"
    MIXED = "mixed"
    UNKNOWN = "unknown"
    

@dataclass
class PageLabel:
    """Labels and metadata for a single page in a document"""
    page_number: int
    content_type: ContentType
    topics: List[str] = field(default_factory=list)
    language: str = "en"
    confidence_score: float = 0.0
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert PageLabel to dictionary"""
        return {
            "page_number": self.page_number,
            "content_type": self.content_type.value,
            "topics": self.topics,
            "language": self.language,
            "confidence_score": self.confidence_score
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PageLabel':
        """Create PageLabel from dictionary"""
        return cls(
            page_number=data.get("page_number", 0),
            content_type=ContentType(data.get("content_type", "text")),
            topics=data.get("topics", []),
            language=data.get("language", "en"),
            confidence_score=data.get("confidence_score", 0.0)
        )
        
@dataclass
class PageAnalysis:
    """Detailed analysis of a single page"""

    page_number: int
    labels: PageLabel
    extracted_data: Dict[str, Any] = field(default_factory=dict)  # Structured data
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "page_number": self.page_number,
            "labels": self.labels.to_dict(),
            "extracted_data": self.extracted_data,
            "timestamp": self.timestamp.isoformat()
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PageAnalysis':
        return cls(
            page_number=data["page_number"],
            labels=PageLabel.from_dict(data["labels"]),
            extracted_data=data.get("extracted_data", {}),
            timestamp=datetime.fromisoformat(data["timestamp"])
        )

@dataclass
class DocumentAnalysis:
    """Complete analysis of a document"""
    document_id: str
    document_name: str
    overall_summary: str
    total_cost: float = 0.0

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "overall_summary": self.overall_summary,

            "total_cost": self.total_cost
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DocumentAnalysis':
        return cls(
            document_id=data["document_id"],
            document_name=data["document_name"],
            overall_summary=data["overall_summary"],
            total_cost=data.get("total_cost", 0.0)
        )
    


    
    