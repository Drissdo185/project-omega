"""
Abstract base class for storage implementations.
"""

from abc import ABC, abstractmethod
from typing import List
from models.document import Document


class BaseStorage(ABC):
    """
    Abstract base class for document storage.

    Defines the interface that all storage implementations must follow.
    """

    @abstractmethod
    def save_document(self, document: Document) -> bool:
        """
        Save document metadata to storage.

        Args:
            document: Document object to save.

        Returns:
            bool: True if successful.

        Raises:
            StorageError: If save operation fails.
        """
        pass

    @abstractmethod
    def load_document(self, doc_id: str) -> Document:
        """
        Load document metadata from storage.

        Args:
            doc_id: Unique document identifier.

        Returns:
            Document: The loaded document object.

        Raises:
            DocumentNotFoundError: If document does not exist.
            StorageError: If load operation fails.
        """
        pass

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document and all associated data from storage.

        Args:
            doc_id: Unique document identifier.

        Returns:
            bool: True if successful.

        Raises:
            DocumentNotFoundError: If document does not exist.
            StorageError: If delete operation fails.
        """
        pass

    @abstractmethod
    def list_documents(self) -> List[Document]:
        """
        List all documents in storage.

        Returns:
            List[Document]: List of all document objects.

        Raises:
            StorageError: If list operation fails.
        """
        pass

    @abstractmethod
    def save_page_image(self, doc_id: str, page_num: int, image_bytes: bytes) -> str:
        """
        Save a page image to storage.

        Args:
            doc_id: Document identifier.
            page_num: Page number (1-indexed).
            image_bytes: Image data as bytes (JPEG format).

        Returns:
            str: Path or identifier where the image was saved.

        Raises:
            StorageError: If save operation fails.
        """
        pass

    @abstractmethod
    def load_page_image(self, doc_id: str, page_num: int) -> bytes:
        """
        Load a page image from storage.

        Args:
            doc_id: Document identifier.
            page_num: Page number (1-indexed).

        Returns:
            bytes: Image data as bytes.

        Raises:
            DocumentNotFoundError: If document or page does not exist.
            StorageError: If load operation fails.
        """
        pass

    @abstractmethod
    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists in storage.

        Args:
            doc_id: Document identifier.

        Returns:
            bool: True if document exists, False otherwise.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
