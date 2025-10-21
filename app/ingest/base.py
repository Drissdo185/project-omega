# app/processors/base.py
from abc import ABC, abstractmethod
from typing import List
from app.models.document import Page

class BaseProcessor(ABC):
    """Base processor for documents"""
    
    @abstractmethod
    async def process(self, file_path: str) -> List[Page]:
        """Process document and return pages as images"""
        pass
    
    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """Check if processor supports file type"""
        pass