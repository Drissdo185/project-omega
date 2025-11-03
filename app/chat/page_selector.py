"""
Page selector that uses page summaries to identify relevant pages for a query.
This reduces costs by only sending relevant images to the LLM.
"""

import json
from typing import List, Dict, Tuple, Optional
from loguru import logger

from app.models.document import Document, Page, Partition
from app.providers.base import BaseProvider


class PageSelector:
    """Selects relevant pages based on summaries and user query"""

    def __init__(self, provider: BaseProvider):
        """
        Initialize page selector
        
        Args:
            provider: LLM provider for analyzing relevance
        """
        self.provider = provider

    async def select_relevant_pages(
        self,
        document: Document,
        user_question: str,
        max_pages: int = None,
        partitions: Optional[List[Partition]] = None,
        model: Optional[str] = None
    ) -> List[Page]:
        """
        Select most relevant pages for a user question based on page summaries.
        The agent decides how many pages are needed based on question complexity.

        Args:
            document: Document with page summaries
            user_question: User's question
            max_pages: Optional hard limit on pages (None = let agent decide)
            partitions: Optional list of partitions to filter pages by (for large documents)

        Returns:
            List of selected Page objects, ordered by relevance
        """
        try:
            logger.info(f"Selecting relevant pages for question: {user_question}")

            # Filter pages by partitions if provided (for large documents)
            if partitions:
                candidate_pages = self._filter_pages_by_partitions(document, partitions)
                logger.info(f"Filtering to {len(candidate_pages)} pages from {len(partitions)} selected partitions")
            else:
                candidate_pages = document.pages

            # Build context with page summaries (only from candidate pages)
            pages_context = self._build_pages_context_from_pages(document, candidate_pages)

            # Build prompt for page selection (agent decides count)
            prompt = self._build_selection_prompt(user_question, pages_context, max_pages)

            # Ask LLM to select relevant pages
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert document analyst. Your task is to identify which pages of a document are most relevant to answer a user's question. Select as many or as few pages as needed - quality over quantity."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]

            response = await self.provider.process_text_messages(
                messages=messages,
                max_tokens=500,
                model=model
            )

            # Parse selected page numbers
            selected_page_numbers = self._parse_page_numbers(response)

            # Apply max_pages limit only if specified
            if max_pages and len(selected_page_numbers) > max_pages:
                logger.info(f"Agent selected {len(selected_page_numbers)} pages, limiting to {max_pages}")
                selected_page_numbers = selected_page_numbers[:max_pages]

            # Get actual Page objects from candidate pages
            selected_pages = self._get_pages_by_numbers_from_list(candidate_pages, selected_page_numbers)

            # Track cost
            cost = self.provider.get_last_cost() or 0.0
            logger.info(f"Agent selected {len(selected_pages)} pages (cost: ${cost:.4f})")
            logger.debug(f"Selected page numbers: {selected_page_numbers}")

            return selected_pages

        except Exception as e:
            logger.error(f"Failed to select relevant pages: {e}")
            # Fallback: return first few pages from candidates
            fallback_count = max_pages if max_pages else 3
            candidate_pages = self._filter_pages_by_partitions(document, partitions) if partitions else document.pages
            return candidate_pages[:fallback_count]

    def _filter_pages_by_partitions(self, document: Document, partitions: List[Partition]) -> List[Page]:
        """
        Filter pages to only include those within the given partitions.
        Uses partition_id field on pages if available, otherwise falls back to page_range.

        Args:
            document: Document with all pages
            partitions: List of partitions to filter by

        Returns:
            List of pages within the partition ranges
        """
        if not partitions:
            return document.pages

        # Get partition IDs
        partition_ids = [p.partition_id for p in partitions]

        filtered_pages = []
        for page in document.pages:
            # Method 1: Direct partition_id match (preferred for large documents)
            if page.partition_id is not None and page.partition_id in partition_ids:
                filtered_pages.append(page)
            # Method 2: Fallback to page_range matching (for backward compatibility)
            elif page.partition_id is None:
                for partition in partitions:
                    start_page, end_page = partition.page_range
                    if start_page <= page.page_number <= end_page:
                        filtered_pages.append(page)
                        break

        return filtered_pages

    def _build_pages_context(self, document: Document) -> str:
        """Build formatted context of all page summaries"""
        return self._build_pages_context_from_pages(document, document.pages)

    def _build_pages_context_from_pages(self, document: Document, pages: List[Page]) -> str:
        """Build formatted context from a filtered list of pages"""
        lines = [f"Document: {document.name}"]
        lines.append(f"Total pages in document: {document.page_count}")
        lines.append(f"Candidate pages to select from: {len(pages)}")
        lines.append("\nPage summaries:")

        for page in pages:
            summary = page.summary or "No summary available"
            lines.append(f"\nPage {page.page_number}: {summary}")

        return "\n".join(lines)

    def _build_selection_prompt(
        self,
        user_question: str,
        pages_context: str,
        max_pages: int = None
    ) -> str:
        """Build prompt for LLM to select relevant pages"""
        
        if max_pages:
            # Hard limit specified
            constraint = f"2. Select up to {max_pages} most relevant pages (hard limit)"
        else:
            # Let agent decide
            constraint = """2. Decide how many pages are needed based on the question:
   - Simple questions (e.g., "What is X?") may need 1-2 pages
   - Complex questions (e.g., "Compare X and Y") may need more pages
   - Choose quality over quantity - only select pages that will help"""
        
        return f"""Given a user's question and summaries of all pages in a document, identify which pages are most relevant to answer the question.

User Question:
{user_question}

{pages_context}

Task:
1. Analyze which pages contain information relevant to the user's question
{constraint}
3. Return ONLY a JSON array of page numbers in order of relevance (most relevant first)

Example response format:
[1, 3, 7]

Your response (JSON array only):"""

    def _parse_page_numbers(self, response: str) -> List[int]:
        """Parse page numbers from LLM response"""
        try:
            # Clean response
            response = response.strip()
            
            # Try to find JSON array in response
            start = response.find('[')
            end = response.rfind(']') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                page_numbers = json.loads(json_str)
                
                # Validate it's a list of integers
                if isinstance(page_numbers, list):
                    return [int(num) for num in page_numbers if isinstance(num, (int, str)) and str(num).isdigit()]
            
            logger.warning(f"Could not parse page numbers from: {response}")
            return []

        except Exception as e:
            logger.error(f"Failed to parse page numbers: {e}")
            return []

    def _get_pages_by_numbers(
        self,
        document: Document,
        page_numbers: List[int]
    ) -> List[Page]:
        """Get Page objects by their page numbers from entire document"""
        return self._get_pages_by_numbers_from_list(document.pages, page_numbers)

    def _get_pages_by_numbers_from_list(
        self,
        pages: List[Page],
        page_numbers: List[int]
    ) -> List[Page]:
        """Get Page objects by their page numbers from a given list of pages"""
        selected_pages = []

        for page_num in page_numbers:
            # Find page with matching page_number
            for page in pages:
                if page.page_number == page_num:
                    selected_pages.append(page)
                    break

        return selected_pages

    async def select_all_pages(self, document: Document) -> List[Page]:
        """
        Return all pages (no filtering)
        Useful when user wants comprehensive analysis
        """
        return document.pages

