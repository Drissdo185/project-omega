"""
In-memory storage implementation for testing.
"""

import logging
from typing import Dict, List
from copy import deepcopy

from storage.base import BaseStorage
from models.document import Document
from exceptions import DocumentNotFoundError, StorageError

logger = logging.getLogger(__name__)


class MemoryStorage(BaseStorage):
    """
    In-memory storage implementation for testing.

    All data is stored in RAM using dictionaries. Data is lost when the process ends.
    """

    def __init__(self):
        """Initialize in-memory storage."""
        self._documents: Dict[str, Document] = {}
        self._images: Dict[str, Dict[int, bytes]] = {}  # {doc_id: {page_num: image_bytes}}
        logger.info("Initialized MemoryStorage")

    def save_document(self, document: Document) -> bool:
        """
        Save document to memory.

        Args:
            document: Document object to save.

        Returns:
            bool: True if successful.

        Raises:
            StorageError: If save operation fails.
        """
        try:
            # Deep copy to avoid external mutations
            self._documents[document.id] = deepcopy(document)
            logger.debug(f"Saved document to memory: {document.id}")
            return True
        except Exception as e:
            raise StorageError(
                f"Failed to save document to memory: {e}",
                operation="save_document"
            )

    def load_document(self, doc_id: str) -> Document:
        """
        Load document from memory.

        Args:
            doc_id: Unique document identifier.

        Returns:
            Document: The loaded document object.

        Raises:
            DocumentNotFoundError: If document does not exist.
            StorageError: If load operation fails.
        """
        if doc_id not in self._documents:
            raise DocumentNotFoundError(doc_id)

        try:
            # Return a deep copy to prevent external mutations
            document = deepcopy(self._documents[doc_id])
            logger.debug(f"Loaded document from memory: {doc_id}")
            return document
        except Exception as e:
            raise StorageError(
                f"Failed to load document from memory: {e}",
                operation="load_document"
            )

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document from memory.

        Args:
            doc_id: Unique document identifier.

        Returns:
            bool: True if successful.

        Raises:
            DocumentNotFoundError: If document does not exist.
            StorageError: If delete operation fails.
        """
        if doc_id not in self._documents:
            raise DocumentNotFoundError(doc_id)

        try:
            del self._documents[doc_id]
            if doc_id in self._images:
                del self._images[doc_id]
            logger.debug(f"Deleted document from memory: {doc_id}")
            return True
        except Exception as e:
            raise StorageError(
                f"Failed to delete document from memory: {e}",
                operation="delete_document"
            )

    def list_documents(self) -> List[Document]:
        """
        List all documents in memory.

        Returns:
            List[Document]: List of all document objects.

        Raises:
            StorageError: If list operation fails.
        """
        try:
            # Return deep copies to prevent external mutations
            documents = [deepcopy(doc) for doc in self._documents.values()]
            logger.debug(f"Listed {len(documents)} documents from memory")
            return documents
        except Exception as e:
            raise StorageError(
                f"Failed to list documents from memory: {e}",
                operation="list_documents"
            )

    def save_page_image(self, doc_id: str, page_num: int, image_bytes: bytes) -> str:
        """
        Save a page image to memory.

        Args:
            doc_id: Document identifier.
            page_num: Page number (1-indexed).
            image_bytes: Image data as bytes.

        Returns:
            str: Identifier where the image was saved.

        Raises:
            StorageError: If save operation fails.
        """
        try:
            if doc_id not in self._images:
                self._images[doc_id] = {}

            # Store a copy of the bytes
            self._images[doc_id][page_num] = bytes(image_bytes)
            image_id = f"memory://{doc_id}/page_{page_num}.jpg"
            logger.debug(f"Saved page image to memory: {image_id}")
            return image_id
        except Exception as e:
            raise StorageError(
                f"Failed to save page image to memory: {e}",
                operation="save_page_image"
            )

    def load_page_image(self, doc_id: str, page_num: int) -> bytes:
        """
        Load a page image from memory.

        Args:
            doc_id: Document identifier.
            page_num: Page number (1-indexed).

        Returns:
            bytes: Image data as bytes.

        Raises:
            DocumentNotFoundError: If document or page does not exist.
            StorageError: If load operation fails.
        """
        if doc_id not in self._images or page_num not in self._images[doc_id]:
            raise DocumentNotFoundError(f"{doc_id}/page_{page_num}")

        try:
            # Return a copy of the bytes
            image_bytes = bytes(self._images[doc_id][page_num])
            logger.debug(f"Loaded page image from memory: {doc_id}/page_{page_num}")
            return image_bytes
        except Exception as e:
            raise StorageError(
                f"Failed to load page image from memory: {e}",
                operation="load_page_image"
            )

    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists in memory.

        Args:
            doc_id: Document identifier.

        Returns:
            bool: True if document exists, False otherwise.
        """
        return doc_id in self._documents

    def clear(self) -> None:
        """
        Clear all data from memory (useful for testing).
        """
        self._documents.clear()
        self._images.clear()
        logger.debug("Cleared all data from memory")

    def __repr__(self) -> str:
        return f"MemoryStorage(documents={len(self._documents)}, images={sum(len(pages) for pages in self._images.values())})"
