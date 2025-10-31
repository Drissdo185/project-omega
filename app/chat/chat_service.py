"""
High-level chat service that combines document loading and chat agent.
Provides a simple interface for asking questions about documents.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger

from app.providers.base import BaseProvider
from app.providers.factory import create_provider_from_env
from app.ai.vision_analysis import VisionAnalysisService
from app.models.document import Document
from .chat_agent import ChatAgent


class ChatService:
    """
    High-level service for chatting with documents.
    Handles document loading, caching, and question answering.
    """

    def __init__(
        self,
        provider: Optional[BaseProvider] = None,
        storage_root: Optional[str] = None
    ):
        """
        Initialize chat service
        
        Args:
            provider: LLM provider (creates from env if not provided)
            storage_root: Root directory for document storage
        """
        # Initialize provider
        self.provider = provider or create_provider_from_env()
        
        # Initialize storage root
        if storage_root is None:
            storage_root = os.environ.get("FLEX_RAG_DATA_LOCATION", "./app/flex_rag_data_location")
        self.storage_root = Path(storage_root)
        
        # Initialize services
        self.vision_service = VisionAnalysisService(self.provider, str(self.storage_root))
        self.chat_agent = ChatAgent(self.provider)
        
        logger.info(f"ChatService initialized with storage: {self.storage_root}")

    async def ask(
        self,
        document_id: str,
        question: str,
        max_pages: int = None,
        use_all_pages: bool = False
    ) -> Dict:
        """
        Ask a question about a document
        
        Args:
            document_id: ID of the document to query
            question: User's question
            max_pages: Optional hard limit on pages (None = agent decides dynamically)
            use_all_pages: If True, analyze all pages
            
        Returns:
            Dict with answer, page_numbers, and cost information
            
        Example:
            result = await service.ask(
                document_id="doc_abc123",
                question="What is the candidate's education background?"
            )
            print(result["answer"])
            print(f"Used pages: {result['page_numbers']}")
        """
        # Load document
        document = self.vision_service.load_document(document_id)
        
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        # Check if document has summaries
        if not any(page.summary for page in document.pages):
            logger.warning(f"Document {document_id} has no page summaries. Consider analyzing it first.")
        
        # Ask question
        result = await self.chat_agent.answer_question(
            document=document,
            question=question,
            max_pages=max_pages,
            use_all_pages=use_all_pages
        )
        
        return result

    async def ask_by_name(
        self,
        document_name: str,
        question: str,
        max_pages: int = None
    ) -> Dict:
        """
        Ask a question about a document by its name
        
        Args:
            document_name: Name of the document (filename)
            question: User's question
            max_pages: Optional hard limit on pages (None = agent decides)
            
        Returns:
            Dict with answer and metadata
        """
        # Find document by name
        documents = self.vision_service.get_all_documents()
        
        document = None
        for doc in documents:
            if doc.name == document_name or doc.name.startswith(document_name):
                document = doc
                break
        
        if not document:
            available = [doc.name for doc in documents]
            raise ValueError(f"Document '{document_name}' not found. Available: {available}")
        
        logger.info(f"Found document: {document.name} (ID: {document.id})")
        
        # Ask question
        return await self.chat_agent.answer_question(
            document=document,
            question=question,
            max_pages=max_pages
        )

    async def conversation(
        self,
        document_id: str,
        questions: List[str],
        max_pages_per_question: int = None
    ) -> List[Dict]:
        """
        Have a multi-turn conversation about a document
        
        Args:
            document_id: ID of the document
            questions: List of questions
            max_pages_per_question: Max pages per question (None = agent decides)
            
        Returns:
            List of result dicts
            
        Example:
            results = await service.conversation(
                document_id="doc_abc123",
                questions=[
                    "What is this document about?",
                    "Who is the author?",
                    "What are the main findings?"
                ]
            )
        """
        # Load document
        document = self.vision_service.load_document(document_id)
        
        if not document:
            raise ValueError(f"Document not found: {document_id}")
        
        # Run conversation
        results = await self.chat_agent.multi_turn_conversation(
            document=document,
            questions=questions,
            max_pages_per_question=max_pages_per_question
        )
        
        return results

    def list_documents(self) -> List[Dict]:
        """
        List all available documents (optimized with caching)
        
        Returns:
            List of document info dicts with id, name, page_count, and image info
        """
        # Cache document list for 5 seconds to avoid repeated disk I/O
        cache_key = '_doc_list_cache'
        cache_time_key = '_doc_list_cache_time'
        
        import time
        current_time = time.time()
        
        if hasattr(self, cache_time_key):
            cache_age = current_time - getattr(self, cache_time_key)
            if cache_age < 5 and hasattr(self, cache_key):
                return getattr(self, cache_key)
        
        documents = self.vision_service.get_all_documents()
        
        result = [
            {
                "id": doc.id,
                "name": doc.name,
                "page_count": doc.page_count,
                "status": doc.status.value,
                "has_summaries": any(page.summary for page in doc.pages),
                "pages_with_images": sum(1 for p in doc.pages if p.isImage),  # Faster than len([])
                "combined_images_count": len(getattr(doc, 'combined_images', []))
            }
            for doc in documents
        ]
        
        # Cache the result
        setattr(self, cache_key, result)
        setattr(self, cache_time_key, current_time)
        
        return result

    def get_document_info(self, document_id: str) -> Optional[Dict]:
        """
        Get detailed information about a document (optimized with caching)
        
        Args:
            document_id: Document ID
            
        Returns:
            Dict with document details including page summaries and image info
        """
        # Cache individual document info for 10 seconds
        cache_key = f'_doc_info_cache_{document_id}'
        cache_time_key = f'_doc_info_time_{document_id}'
        
        import time
        current_time = time.time()
        
        if hasattr(self, cache_time_key):
            cache_age = current_time - getattr(self, cache_time_key)
            if cache_age < 10 and hasattr(self, cache_key):
                return getattr(self, cache_key)
        
        document = self.vision_service.load_document(document_id)
        
        if not document:
            return None
        
        # Optimized counting
        pages_with_images = sum(1 for p in document.pages if p.isImage)
        
        result = {
            "id": document.id,
            "name": document.name,
            "page_count": document.page_count,
            "status": document.status.value,
            "summary": document.summary,
            "pages_with_images": pages_with_images,
            "combined_images_count": len(document.combined_images),
            "pages": [
                {
                    "page_number": page.page_number,
                    "summary": page.summary,
                    "width": page.width,
                    "height": page.height,
                    "isImage": page.isImage,
                    "combined_image_number": page.combined_image_number
                }
                for page in document.pages
            ]
        }
        
        # Cache the result
        setattr(self, cache_key, result)
        setattr(self, cache_time_key, current_time)
        
        return result

    def get_total_cost(self) -> float:
        """Get total cost of all queries in this session"""
        return self.chat_agent.get_total_cost()

    def reset_cost(self):
        """Reset cost tracking"""
        self.chat_agent.reset_cost()

