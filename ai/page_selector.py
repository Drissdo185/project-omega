"""
Page selector using vision models to identify relevant pages.
"""

import logging
import json
from typing import List

from models.document import Document
from storage.base import BaseStorage
from providers.base import BaseProvider
from ai.prompts import PAGE_SELECTION_PROMPT
from exceptions import VisionModelError
from core.config import get_settings

logger = logging.getLogger(__name__)


class PageSelector:
    """
    Selects relevant pages from a document based on a query using vision models.
    """

    def __init__(self, provider: BaseProvider, max_pages: int = None):
        """
        Initialize page selector.

        Args:
            provider: Vision model provider to use.
            max_pages: Maximum number of pages to select (defaults to config).
        """
        self.provider = provider
        settings = get_settings()
        self.max_pages = max_pages if max_pages is not None else settings.MAX_PAGES_PER_SELECTION
        logger.info(f"Initialized PageSelector with {provider} (max_pages={self.max_pages})")

    def select_pages(
        self,
        query: str,
        document: Document,
        storage: BaseStorage
    ) -> List[int]:
        """
        Select relevant pages for answering a query.

        Args:
            query: User's question or query.
            document: Document to search.
            storage: Storage to load page images from.

        Returns:
            List[int]: List of selected page numbers (1-indexed), ordered by relevance.

        Raises:
            VisionModelError: If page selection fails.
        """
        logger.info(f"Selecting pages for query: '{query}' (doc={document.id})")

        if not document.summary:
            raise VisionModelError(
                f"Document {document.id} has no summary. Cannot select pages."
            )

        try:
            # Get all page image paths
            image_paths = self._get_page_image_paths(document, storage)

            if not image_paths:
                raise VisionModelError(
                    f"No page images found for document {document.id}"
                )

            # Build prompt with query and summary
            prompt = PAGE_SELECTION_PROMPT.format(
                summary=document.summary,
                query=query,
                max_pages=self.max_pages
            )

            logger.debug(f"Analyzing {len(image_paths)} pages for relevance")

            # Send all pages to vision model
            response = self.provider.analyze_images(
                image_paths=image_paths,
                prompt=prompt
            )

            # Parse JSON response
            selected_pages = self._parse_page_numbers(response, document.page_count)

            # Limit to max_pages
            if len(selected_pages) > self.max_pages:
                logger.warning(f"Model returned {len(selected_pages)} pages, limiting to {self.max_pages}")
                selected_pages = selected_pages[:self.max_pages]

            logger.info(f"Selected {len(selected_pages)} pages: {selected_pages}")
            return selected_pages

        except VisionModelError:
            raise
        except Exception as e:
            raise VisionModelError(f"Failed to select pages: {e}")

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

    def _parse_page_numbers(self, response: str, max_page: int) -> List[int]:
        """
        Parse page numbers from model response.

        Args:
            response: Model's response (should be JSON array).
            max_page: Maximum valid page number.

        Returns:
            List[int]: List of valid page numbers.
        """
        try:
            # Try to find JSON array in response
            response = response.strip()

            # Find array boundaries
            start_idx = response.find('[')
            end_idx = response.rfind(']')

            if start_idx == -1 or end_idx == -1:
                logger.warning(f"No JSON array found in response: {response[:100]}")
                return []

            json_str = response[start_idx:end_idx + 1]
            page_numbers = json.loads(json_str)

            if not isinstance(page_numbers, list):
                logger.warning(f"Response is not a list: {type(page_numbers)}")
                return []

            # Validate and filter page numbers
            valid_pages = []
            for page_num in page_numbers:
                if isinstance(page_num, int) and 1 <= page_num <= max_page:
                    valid_pages.append(page_num)
                else:
                    logger.warning(f"Invalid page number: {page_num}")

            return valid_pages

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response[:200]}")
            return []
        except Exception as e:
            logger.warning(f"Error parsing page numbers: {e}")
            return []

    def __repr__(self) -> str:
        return f"PageSelector(provider={self.provider}, max_pages={self.max_pages})"
