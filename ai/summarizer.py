"""
Document summarizer using vision models.
"""

import logging
from typing import List

from models.document import Document
from storage.base import BaseStorage
from providers.base import BaseProvider
from ai.prompts import SUMMARIZATION_PROMPT
from exceptions import VisionModelError

logger = logging.getLogger(__name__)


class DocumentSummarizer:
    """
    Generates comprehensive summaries of documents by analyzing all pages.
    """

    def __init__(self, provider: BaseProvider):
        """
        Initialize document summarizer.

        Args:
            provider: Vision model provider to use.
        """
        self.provider = provider
        logger.info(f"Initialized DocumentSummarizer with {provider}")

    def summarize(self, document: Document, storage: BaseStorage) -> str:
        """
        Generate a comprehensive summary of the document.

        Args:
            document: Document to summarize.
            storage: Storage to load page images from.

        Returns:
            str: Document summary.

        Raises:
            VisionModelError: If summarization fails.
        """
        logger.info(f"Summarizing document: {document.id} ({document.page_count} pages)")

        try:
            # Get all page image paths
            image_paths = self._get_page_image_paths(document, storage)

            if not image_paths:
                raise VisionModelError(
                    f"No page images found for document {document.id}"
                )

            logger.debug(f"Analyzing {len(image_paths)} pages for summarization")

            # Send all pages to vision model for analysis
            summary = self.provider.analyze_images(
                image_paths=image_paths,
                prompt=SUMMARIZATION_PROMPT
            )

            logger.info(f"Generated summary for {document.id} (length: {len(summary)})")
            return summary

        except VisionModelError:
            raise
        except Exception as e:
            raise VisionModelError(f"Failed to summarize document: {e}")

    def _get_page_image_paths(self, document: Document, storage: BaseStorage) -> List[str]:
        """
        Get paths to all page images for a document.

        Args:
            document: Document object.
            storage: Storage instance.

        Returns:
            List[str]: List of image paths ordered by page number.
        """
        # Sort pages by page number
        sorted_pages = sorted(document.pages, key=lambda p: p.page_number)

        # Get image paths
        image_paths = [page.image_path for page in sorted_pages]

        return image_paths

    def __repr__(self) -> str:
        return f"DocumentSummarizer(provider={self.provider})"
