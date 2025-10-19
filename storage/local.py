"""
Local filesystem storage implementation.
"""

import json
import logging
from pathlib import Path
from typing import List
from PIL import Image
import io

from storage.base import BaseStorage
from models.document import Document
from exceptions import DocumentNotFoundError, StorageError

logger = logging.getLogger(__name__)


class LocalStorage(BaseStorage):
    """
    Local filesystem-based storage implementation.

    Stores documents in a directory structure:
    {base_path}/documents/{doc_id}/
        - metadata.json (document metadata)
        - pages/
            - page_1.jpg
            - page_2.jpg
            - ...
    """

    def __init__(self, base_path: Path):
        """
        Initialize local storage.

        Args:
            base_path: Base directory for storage.
        """
        self.base_path = Path(base_path)
        self.documents_path = self.base_path / "documents"
        self._ensure_base_directories()
        logger.info(f"Initialized LocalStorage at {self.base_path}")

    def _ensure_base_directories(self) -> None:
        """Create base directories if they don't exist."""
        try:
            self.documents_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise StorageError(
                f"Failed to create base directories: {e}",
                operation="init",
                path=str(self.documents_path)
            )

    def _get_document_path(self, doc_id: str) -> Path:
        """Get the directory path for a document."""
        return self.documents_path / doc_id

    def _get_metadata_path(self, doc_id: str) -> Path:
        """Get the path to document metadata file."""
        return self._get_document_path(doc_id) / "metadata.json"

    def _get_pages_path(self, doc_id: str) -> Path:
        """Get the directory path for document pages."""
        return self._get_document_path(doc_id) / "pages"

    def _get_page_image_path(self, doc_id: str, page_num: int) -> Path:
        """Get the path to a specific page image."""
        return self._get_pages_path(doc_id) / f"page_{page_num}.jpg"

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
        try:
            doc_path = self._get_document_path(document.id)
            doc_path.mkdir(parents=True, exist_ok=True)

            metadata_path = self._get_metadata_path(document.id)
            metadata = document.to_dict()

            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved document metadata: {document.id}")
            return True

        except Exception as e:
            raise StorageError(
                f"Failed to save document: {e}",
                operation="save_document",
                path=str(metadata_path)
            )

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
        metadata_path = self._get_metadata_path(doc_id)

        if not metadata_path.exists():
            raise DocumentNotFoundError(doc_id)

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            document = Document.from_dict(metadata)
            logger.debug(f"Loaded document: {doc_id}")
            return document

        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to load document: {e}",
                operation="load_document",
                path=str(metadata_path)
            )

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
        doc_path = self._get_document_path(doc_id)

        if not doc_path.exists():
            raise DocumentNotFoundError(doc_id)

        try:
            # Delete all files in the document directory
            import shutil
            shutil.rmtree(doc_path)
            logger.info(f"Deleted document: {doc_id}")
            return True

        except Exception as e:
            raise StorageError(
                f"Failed to delete document: {e}",
                operation="delete_document",
                path=str(doc_path)
            )

    def list_documents(self) -> List[Document]:
        """
        List all documents in storage.

        Returns:
            List[Document]: List of all document objects.

        Raises:
            StorageError: If list operation fails.
        """
        try:
            documents = []

            if not self.documents_path.exists():
                return documents

            for doc_dir in self.documents_path.iterdir():
                if doc_dir.is_dir():
                    metadata_path = doc_dir / "metadata.json"
                    if metadata_path.exists():
                        try:
                            with open(metadata_path, 'r', encoding='utf-8') as f:
                                metadata = json.load(f)
                            documents.append(Document.from_dict(metadata))
                        except Exception as e:
                            logger.warning(f"Failed to load document {doc_dir.name}: {e}")
                            continue

            logger.debug(f"Listed {len(documents)} documents")
            return documents

        except Exception as e:
            raise StorageError(
                f"Failed to list documents: {e}",
                operation="list_documents",
                path=str(self.documents_path)
            )

    def save_page_image(self, doc_id: str, page_num: int, image_bytes: bytes) -> str:
        """
        Save a page image to storage.

        Args:
            doc_id: Document identifier.
            page_num: Page number (1-indexed).
            image_bytes: Image data as bytes (JPEG format).

        Returns:
            str: Path where the image was saved.

        Raises:
            StorageError: If save operation fails.
        """
        try:
            pages_path = self._get_pages_path(doc_id)
            pages_path.mkdir(parents=True, exist_ok=True)

            image_path = self._get_page_image_path(doc_id, page_num)

            # Save image bytes directly
            with open(image_path, 'wb') as f:
                f.write(image_bytes)

            logger.debug(f"Saved page image: {doc_id}/page_{page_num}.jpg")
            return str(image_path)

        except Exception as e:
            raise StorageError(
                f"Failed to save page image: {e}",
                operation="save_page_image",
                path=str(image_path)
            )

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
        image_path = self._get_page_image_path(doc_id, page_num)

        if not image_path.exists():
            raise DocumentNotFoundError(f"{doc_id}/page_{page_num}")

        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            logger.debug(f"Loaded page image: {doc_id}/page_{page_num}.jpg")
            return image_bytes

        except DocumentNotFoundError:
            raise
        except Exception as e:
            raise StorageError(
                f"Failed to load page image: {e}",
                operation="load_page_image",
                path=str(image_path)
            )

    def document_exists(self, doc_id: str) -> bool:
        """
        Check if a document exists in storage.

        Args:
            doc_id: Document identifier.

        Returns:
            bool: True if document exists, False otherwise.
        """
        metadata_path = self._get_metadata_path(doc_id)
        return metadata_path.exists()

    def __repr__(self) -> str:
        return f"LocalStorage(base_path={self.base_path})"
