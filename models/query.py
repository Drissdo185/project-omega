"""
Query result models for Vision-Based RAG system.
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class QueryResult:
    """Represents the result of a query against a document."""
    
    answer: str
    selected_pages: List[int]
    document_id: str
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert QueryResult to dictionary."""
        return {
            'answer': self.answer,
            'selected_pages': self.selected_pages,
            'document_id': self.document_id,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryResult':
        """Create QueryResult from dictionary."""
        timestamp = None
        if data.get('timestamp'):
            timestamp = datetime.fromisoformat(data['timestamp'])
        
        return cls(
            answer=data['answer'],
            selected_pages=data['selected_pages'],
            document_id=data['document_id'],
            confidence=data.get('confidence'),
            reasoning=data.get('reasoning'),
            metadata=data.get('metadata', {}),
            timestamp=timestamp
        )
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the result."""
        self.metadata[key] = value
    
    def get_page_count(self) -> int:
        """Get number of pages selected."""
        return len(self.selected_pages)
    
    def __repr__(self) -> str:
        return (f"QueryResult(doc={self.document_id}, "
                f"pages={self.selected_pages}, "
                f"confidence={self.confidence})")