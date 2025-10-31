# app/storage/document_store.py
import json
import os
from pathlib import Path
from typing import List, Optional, Dict
from loguru import logger

from app.models.document import Document, DocumentStatus
from .base import BaseStorage


class DocumentStore(BaseStorage):
    """Storage manager for vision-based document structure"""

    def __init__(self, storage_root: str = None):
        if storage_root is None:
            storage_root = os.environ.get("FLEX_RAG_DATA_LOCATION", "./app/flex_rag_data_location")
    
        self.storage_root = Path(storage_root)
        self.documents_dir = self.storage_root / "documents"
        self.cache_dir = self.storage_root / "cache" / "summaries"

        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_document(self, doc_id: str) -> Optional[Document]:
        """Load a document by its ID with caching"""
        # Simple in-memory cache (10 seconds)
        cache_key = f'_doc_cache_{doc_id}'
        cache_time_key = f'_doc_time_{doc_id}'
        
        import time
        current_time = time.time()
        
        if hasattr(self, cache_time_key):
            cache_age = current_time - getattr(self, cache_time_key)
            if cache_age < 10 and hasattr(self, cache_key):
                logger.debug(f"Using cached document {doc_id}")
                return getattr(self, cache_key)
        
        metadata_path = self.documents_dir / doc_id / "metadata.json"

        if not metadata_path.exists():
            logger.warning(f"Document metadata not found: {doc_id}")
            return None

        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            document = Document.from_dict(data)
            
            # Cache the result
            setattr(self, cache_key, document)
            setattr(self, cache_time_key, current_time)
            
            return document
        except Exception as e:
            logger.error(f"Failed to load document {doc_id}: {e}")
            return None

    def load_index(self) -> List[Dict]:
        """Load the global document index with caching"""
        # Cache index for 5 seconds to reduce I/O
        cache_key = '_index_cache'
        cache_time_key = '_index_cache_time'
        
        import time
        current_time = time.time()
        
        if hasattr(self, cache_time_key):
            cache_age = current_time - getattr(self, cache_time_key)
            if cache_age < 5 and hasattr(self, cache_key):
                logger.debug("Using cached index")
                return getattr(self, cache_key)
        
        index_path = self.documents_dir / "index.json"

        if not index_path.exists():
            logger.debug("Index not found, returning empty list")
            return []

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            
            # Cache the result
            setattr(self, cache_key, index)
            setattr(self, cache_time_key, current_time)
            
            return index
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            return []

    def list_all_documents(self) -> List[Dict]:
        """Get all documents from index"""
        return self.load_index()

    async def get_all_documents(self) -> List[Document]:
        """Load all documents from storage"""
        index = self.load_index()
        documents = []

        for entry in index:
            doc_id = entry.get("id")
            if doc_id:
                doc = self.load_document(doc_id)
                if doc:
                    documents.append(doc)

        return documents

    async def get_document(self, doc_id: str) -> Optional[Document]:
        """Get a specific document by ID (async version of load_document)"""
        return self.load_document(doc_id)

    async def save_document(self, document: Document) -> bool:
        """Save a document to storage and invalidate cache"""
        try:
            doc_dir = self.documents_dir / document.id
            doc_dir.mkdir(parents=True, exist_ok=True)

            metadata_path = doc_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)

            self._update_index_entry(document)
            
            # Invalidate caches
            self._invalidate_cache(document.id)
            
            return True
        except Exception as e:
            logger.error(f"Failed to save document {document.id}: {e}")
            return False

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and update the index"""
        doc_dir = self.documents_dir / doc_id

        if not doc_dir.exists():
            logger.warning(f"Document directory not found: {doc_id}")
            return False

        try:
            # Remove document directory
            import shutil
            shutil.rmtree(doc_dir)

            # Update index
            index = self.load_index()
            index = [e for e in index if e.get("id") != doc_id]

            index_path = self.documents_dir / "index.json"
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

            logger.info(f"Deleted document: {doc_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {doc_id}: {e}")
            return False

    def update_document_status(self, doc_id: str, status: DocumentStatus) -> bool:
        """Update the status of a document"""
        doc = self.load_document(doc_id)
        if not doc:
            return False

        doc.status = status

        # Save updated metadata
        metadata_path = self.documents_dir / doc_id / "metadata.json"
        try:
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)

            # Update index
            self._update_index_entry(doc)
            return True

        except Exception as e:
            logger.error(f"Failed to update document status: {e}")
            return False

    def _update_index_entry(self, document: Document):
        """Update a single document entry in the index"""
        # Clear index cache before loading to get fresh data
        if hasattr(self, '_index_cache'):
            delattr(self, '_index_cache')
        if hasattr(self, '_index_cache_time'):
            delattr(self, '_index_cache_time')
        
        index = self.load_index()

        index_entry = {
            "id": document.id,
            "name": document.name,
            "page_count": document.page_count,
            "status": document.status.value,
            "created_at": document.created_at.isoformat(),
            "updated_at": document.updated_at.isoformat()
        }

        # Remove existing entry
        index = [e for e in index if e.get("id") != document.id]
        index.append(index_entry)

        # Save updated index
        index_path = self.documents_dir / "index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2, ensure_ascii=False)
        
        # Invalidate cache after update
        if hasattr(self, '_index_cache'):
            delattr(self, '_index_cache')
        if hasattr(self, '_index_cache_time'):
            delattr(self, '_index_cache_time')
    
    def _invalidate_cache(self, doc_id: str):
        """Invalidate cache for a specific document"""
        cache_key = f'_doc_cache_{doc_id}'
        cache_time_key = f'_doc_time_{doc_id}'
        
        if hasattr(self, cache_key):
            delattr(self, cache_key)
        if hasattr(self, cache_time_key):
            delattr(self, cache_time_key)

    def get_document_pages_dir(self, doc_id: str) -> Optional[Path]:
        """Get the pages directory path for a document"""
        pages_dir = self.documents_dir / doc_id / "pages"
        if pages_dir.exists():
            return pages_dir
        return None

    def save_summary(self, doc_id: str, summary: str):
        """Save a summary for a document to the cache"""
        summary_path = self.cache_dir / f"{doc_id}_summary.txt"
        try:
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            logger.debug(f"Saved summary for {doc_id}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")

    def load_summary(self, doc_id: str) -> Optional[str]:
        """Load a cached summary for a document"""
        summary_path = self.cache_dir / f"{doc_id}_summary.txt"
        if not summary_path.exists():
            return None

        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load summary: {e}")
            return None
