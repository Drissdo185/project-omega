"""Base storage interface for document management"""

from abc import ABC, abstractmethod
from typing import List, Optional
from app.models.document import Document


class BaseStorage(ABC):
    """
    Abstract base class for document storage
    Provides interface for loading and managing documents
    """

    @abstractmethod
    async def get_all_documents(self) -> List[Document]:
        """
        Get all documents from storage

        Returns:
            List of Document objects
        """
        pass

    @abstractmethod
    async def get_document(self, doc_id: str) -> Optional[Document]:
        """
        Get a specific document by ID

        Args:
            doc_id: Document identifier

        Returns:
            Document object or None if not found
        """
        pass

    @abstractmethod
    async def save_document(self, document: Document) -> bool:
        """
        Save a document to storage

        Args:
            document: Document to save

        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from storage

        Args:
            doc_id: Document identifier

        Returns:
            True if successful, False otherwise
        """
        pass
