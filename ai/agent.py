"""
RAG Agent - Main orchestrator for document processing and querying.
"""

import logging
from pathlib import Path
from typing import List

from models.document import Document
from models.query import QueryResult
from storage.base import BaseStorage
from storage.local import LocalStorage
from processors.factory import ProcessorFactory
from providers.factory import ProviderFactory
from providers.base import BaseProvider
from ai.summarizer import DocumentSummarizer
from ai.page_selector import PageSelector
from ai.synthesizer import ResponseSynthesizer
from ai.prompts import ANALYSIS_PROMPT
from exceptions import DocumentNotFoundError, ProcessingError
from core.config import get_settings

logger = logging.getLogger(__name__)


class RAGAgent:
    """
    Main RAG agent that orchestrates document processing and querying.

    Workflow:
    1. add_document: PDF → Images → Summarize → Ready
    2. query: Select pages → Analyze pages → Synthesize answer
    """

    def __init__(
        self,
        storage: BaseStorage = None,
        provider: BaseProvider = None
    ):
        """
        Initialize RAG agent.

        Args:
            storage: Storage implementation (defaults to LocalStorage).
            provider: Vision provider (defaults to configured provider).
        """
        settings = get_settings()

        # Initialize storage
        if storage is None:
            storage_path = settings.get_documents_path()
            self.storage = LocalStorage(storage_path)
        else:
            self.storage = storage

        # Initialize provider
        if provider is None:
            self.provider = ProviderFactory.get_provider()
        else:
            self.provider = provider

        # Initialize AI components
        self.summarizer = DocumentSummarizer(self.provider)
        self.page_selector = PageSelector(self.provider)
        self.synthesizer = ResponseSynthesizer(self.provider)

        logger.info(f"Initialized RAGAgent (storage={self.storage}, provider={self.provider})")

    def add_document(self, pdf_path: str) -> Document:
        """
        Add a document to the system.

        Workflow:
        1. Process PDF → convert to images
        2. Summarize document using vision model
        3. Update status to "ready"
        4. Save to storage

        Args:
            pdf_path: Path to PDF file.

        Returns:
            Document: Processed document with status="ready".

        Raises:
            ProcessingError: If processing fails.
        """
        pdf_path_obj = Path(pdf_path)

        if not pdf_path_obj.exists():
            raise ProcessingError(f"PDF file not found: {pdf_path}")

        logger.info(f"Adding document: {pdf_path}")

        try:
            # Step 1: Process PDF
            logger.info("Step 1: Processing PDF to images...")
            processor = ProcessorFactory.get_processor_for_file(pdf_path)
            doc_id = Document.generate_id()

            document = processor.process(
                file_path=pdf_path,
                doc_id=doc_id,
                storage=self.storage
            )

            logger.info(f"PDF processed: {document.page_count} pages")

            # Save initial document state
            self.storage.save_document(document)

            # Step 2: Summarize document
            logger.info("Step 2: Generating document summary...")
            summary = self.summarizer.summarize(document, self.storage)
            document.set_summary(summary)

            logger.info(f"Summary generated (length: {len(summary)})")

            # Step 3: Update status to ready
            document.update_status("ready")

            # Save final document
            self.storage.save_document(document)

            logger.info(f"Document added successfully: {document.id}")
            return document

        except Exception as e:
            logger.error(f"Failed to add document: {e}")

            # Update status to failed if document was created
            try:
                if 'document' in locals():
                    document.update_status("failed")
                    self.storage.save_document(document)
            except Exception:
                pass

            raise ProcessingError(f"Failed to add document: {e}")

    def query(self, doc_id: str, question: str) -> QueryResult:
        """
        Query a document with a question.

        Workflow:
        1. Load document
        2. Select relevant pages using vision model
        3. Analyze selected pages
        4. Synthesize final answer

        Args:
            doc_id: Document identifier.
            question: User's question.

        Returns:
            QueryResult: Query result with answer and metadata.

        Raises:
            DocumentNotFoundError: If document doesn't exist.
            VisionModelError: If querying fails.
        """
        logger.info(f"Querying document {doc_id}: '{question}'")

        # Load document
        document = self.storage.load_document(doc_id)

        if not document.is_ready():
            raise ProcessingError(
                f"Document {doc_id} is not ready (status: {document.status})",
                doc_id=doc_id
            )

        try:
            # Step 1: Select relevant pages
            logger.info("Step 1: Selecting relevant pages...")
            selected_pages = self.page_selector.select_pages(
                query=question,
                document=document,
                storage=self.storage
            )

            if not selected_pages:
                logger.warning("No relevant pages found")
                return QueryResult(
                    answer="I couldn't find any relevant information in the document to answer your question.",
                    selected_pages=[],
                    document_id=doc_id,
                    confidence=0.0,
                    reasoning="No relevant pages were identified."
                )

            logger.info(f"Selected {len(selected_pages)} pages: {selected_pages}")

            # Step 2: Analyze selected pages
            logger.info("Step 2: Analyzing selected pages...")
            analyses = self._analyze_pages(
                question=question,
                page_numbers=selected_pages,
                document=document
            )

            # Step 3: Synthesize answer
            logger.info("Step 3: Synthesizing final answer...")
            final_answer = self.synthesizer.synthesize(
                query=question,
                analyses=analyses,
                page_numbers=selected_pages
            )

            # Create result
            result = QueryResult(
                answer=final_answer,
                selected_pages=selected_pages,
                document_id=doc_id,
                confidence=None,  # Could add confidence scoring
                reasoning=f"Analyzed {len(selected_pages)} pages"
            )

            logger.info(f"Query completed successfully (answer length: {len(final_answer)})")
            return result

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise

    def _analyze_pages(
        self,
        question: str,
        page_numbers: List[int],
        document: Document
    ) -> List[str]:
        """
        Analyze specific pages to answer a question.

        Args:
            question: User's question.
            page_numbers: List of page numbers to analyze.
            document: Document object.

        Returns:
            List[str]: List of analysis results (one per page).
        """
        analyses = []

        # Get image paths for selected pages
        for page_num in page_numbers:
            page = document.get_page(page_num)
            if page is None:
                logger.warning(f"Page {page_num} not found in document")
                continue

            # Analyze this page
            prompt = ANALYSIS_PROMPT.format(query=question)

            try:
                analysis = self.provider.analyze_image(
                    image_path=page.image_path,
                    prompt=prompt
                )
                analyses.append(analysis)
                logger.debug(f"Analyzed page {page_num} (result length: {len(analysis)})")

            except Exception as e:
                logger.warning(f"Failed to analyze page {page_num}: {e}")
                analyses.append(f"[Error analyzing page {page_num}]")

        return analyses

    def list_documents(self) -> List[Document]:
        """
        List all documents in the system.

        Returns:
            List[Document]: List of all documents.
        """
        return self.storage.list_documents()

    def get_document(self, doc_id: str) -> Document:
        """
        Get a specific document.

        Args:
            doc_id: Document identifier.

        Returns:
            Document: The document.

        Raises:
            DocumentNotFoundError: If document doesn't exist.
        """
        return self.storage.load_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document.

        Args:
            doc_id: Document identifier.

        Returns:
            bool: True if successful.

        Raises:
            DocumentNotFoundError: If document doesn't exist.
        """
        logger.info(f"Deleting document: {doc_id}")
        result = self.storage.delete_document(doc_id)
        logger.info(f"Document deleted: {doc_id}")
        return result

    def __repr__(self) -> str:
        return f"RAGAgent(storage={self.storage}, provider={self.provider})"
