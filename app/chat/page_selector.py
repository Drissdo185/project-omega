"""
Page selector that uses page summaries to identify relevant pages for a query.
This reduces costs by only sending relevant images to the LLM.
"""

import json
from typing import List, Dict, Tuple
from loguru import logger

from app.models.document import Document, Page
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
        max_pages: int = None
    ) -> List[Page]:
        """
        Select most relevant pages for a user question based on page summaries.
        The agent decides how many pages are needed based on question complexity.
        
        Args:
            document: Document with page summaries
            user_question: User's question
            max_pages: Optional hard limit on pages (None = let agent decide)
            
        Returns:
            List of selected Page objects, ordered by relevance
        """
        try:
            logger.info(f"Selecting relevant pages for question: {user_question}")

            # Build context with all page summaries
            pages_context = self._build_pages_context(document)

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
                max_tokens=500
            )

            # Parse selected page numbers
            selected_page_numbers = self._parse_page_numbers(response)

            # Apply max_pages limit only if specified
            if max_pages and len(selected_page_numbers) > max_pages:
                logger.info(f"Agent selected {len(selected_page_numbers)} pages, limiting to {max_pages}")
                selected_page_numbers = selected_page_numbers[:max_pages]

            # Get actual Page objects
            selected_pages = self._get_pages_by_numbers(document, selected_page_numbers)

            # Track cost
            cost = self.provider.get_last_cost() or 0.0
            logger.info(f"Agent selected {len(selected_pages)} pages (cost: ${cost:.4f})")
            logger.debug(f"Selected page numbers: {selected_page_numbers}")

            return selected_pages

        except Exception as e:
            logger.error(f"Failed to select relevant pages: {e}")
            # Fallback: return first few pages
            fallback_count = max_pages if max_pages else 3
            return document.pages[:fallback_count]

    def _build_pages_context(self, document: Document) -> str:
        """Build formatted context of all page summaries"""
        lines = [f"Document: {document.name}"]
        lines.append(f"Total pages: {document.page_count}")
        lines.append("\nPage summaries:")
        
        for page in document.pages:
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
        """Get Page objects by their page numbers"""
        selected_pages = []
        
        for page_num in page_numbers:
            # Find page with matching page_number
            for page in document.pages:
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

