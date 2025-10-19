"""
Abstract base class for document processors.
"""

from abc import ABC, abstractmethod
from models.document import Document
from storage.base import BaseStorage


class BaseProcessor(ABC):
    """
    Abstract base class for document processors.

    Defines the interface for processing different document types.
    """

    @abstractmethod
    def process(self, file_path: str, doc_id: str, storage: BaseStorage) -> Document:
        """
        Process a document file.

        Args:
            file_path: Path to the document file.
            doc_id: Unique identifier for the document.
            storage: Storage instance to save processed data.

        Returns:
            Document: Processed document object with status="processing".

        Raises:
            ProcessingError: If processing fails.
        """
        pass

    @abstractmethod
    def supports(self, file_path: str) -> bool:
        """
        Check if this processor supports the given file type.

        Args:
            file_path: Path to the document file.

        Returns:
            bool: True if this processor can handle the file.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
