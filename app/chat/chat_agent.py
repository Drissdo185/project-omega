"""
Chat agent that answers questions using vision-based document analysis.
Uses page summaries to select relevant pages, then analyzes images to answer.
"""

from typing import List, Optional, Dict
from loguru import logger

from app.models.document import Document, Page
from app.providers.base import BaseProvider
from .page_selector import PageSelector


class ChatAgent:
    """
    Intelligent agent that answers questions about documents using vision analysis.
    
    Workflow:
    1. User asks a question
    2. Agent selects relevant pages based on summaries (cost-effective filtering)
    3. Agent sends selected page images to LLM for detailed analysis
    4. Returns answer with source pages
    """

    def __init__(self, provider: BaseProvider):
        """
        Initialize chat agent
        
        Args:
            provider: LLM provider with vision capabilities
        """
        self.provider = provider
        self.page_selector = PageSelector(provider)
        self.total_cost = 0.0

    async def answer_question(
        self,
        document: Document,
        question: str,
        max_pages: int = None,
        use_all_pages: bool = False
    ) -> Dict:
        """
        Answer a question about a document
        
        Args:
            document: Document with page summaries
            question: User's question
            max_pages: Optional hard limit on pages (None = agent decides dynamically)
            use_all_pages: If True, use all pages (no selection)
            
        Returns:
            Dict with answer, selected_pages, and cost information
        """
        try:
            logger.info(f"Answering question: {question}")
            logger.info(f"Document: {document.name} ({document.page_count} pages)")

            # Step 1: Select relevant pages
            if use_all_pages:
                logger.info("Using all pages (no filtering)")
                selected_pages = await self.page_selector.select_all_pages(document)
            else:
                if max_pages:
                    logger.info(f"Agent selecting relevant pages (hard limit: {max_pages})...")
                else:
                    logger.info("Agent selecting relevant pages (deciding count dynamically)...")
                selected_pages = await self.page_selector.select_relevant_pages(
                    document=document,
                    user_question=question,
                    max_pages=max_pages
                )

            if not selected_pages:
                logger.warning("No pages selected, using first page as fallback")
                selected_pages = [document.pages[0]] if document.pages else []

            selection_cost = self.provider.get_last_cost() or 0.0

            if not selected_pages:
                return {
                    "answer": "No pages available to analyze.",
                    "selected_pages": [],
                    "page_numbers": [],
                    "total_cost": selection_cost,
                    "selection_cost": selection_cost,
                    "analysis_cost": 0.0
                }

            # Step 2: Analyze selected pages with vision to answer question
            logger.info(f"Analyzing {len(selected_pages)} selected pages...")
            answer = await self._analyze_pages_for_answer(
                pages=selected_pages,
                question=question,
                document_name=document.name
            )

            analysis_cost = self.provider.get_last_cost() or 0.0
            total_cost = selection_cost + analysis_cost

            # Track cumulative cost
            self.total_cost += total_cost

            result = {
                "answer": answer,
                "selected_pages": selected_pages,
                "page_numbers": [p.page_number for p in selected_pages],
                "total_cost": total_cost,
                "selection_cost": selection_cost,
                "analysis_cost": analysis_cost
            }

            logger.info(f"Question answered successfully!")
            logger.info(f"Cost breakdown - Selection: ${selection_cost:.4f}, Analysis: ${analysis_cost:.4f}, Total: ${total_cost:.4f}")

            return result

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise

    async def _analyze_pages_for_answer(
        self,
        pages: List[Page],
        question: str,
        document_name: str
    ) -> str:
        """
        Analyze selected page images to answer the question
        
        Args:
            pages: Selected pages to analyze
            question: User's question
            document_name: Name of the document
            
        Returns:
            Answer text from LLM
        """
        try:
            # Build multimodal prompt
            prompt = self._build_answer_prompt(question, pages, document_name)

            # Build multimodal message with images
            content = [{"type": "text", "text": prompt}]

            # Add all selected page images
            for page in pages:
                content.append({
                    "type": "image_path",
                    "image_path": page.image_path,
                    "detail": "high"
                })

            messages = [
                {
                    "role": "system",
                    "content": "You are an expert document analyst with vision capabilities. Analyze the provided document pages and answer the user's question accurately and concisely."
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            # Get answer from vision model
            response = await self.provider.process_multimodal_messages(
                messages=messages,
                max_tokens=2000
            )

            # Validate response
            if not response or not response.strip():
                logger.warning("LLM returned empty response")
                return "I apologize, but I couldn't generate an answer. The AI model returned an empty response. Please try rephrasing your question or check if the document pages contain relevant information."
            
            return response.strip()

        except Exception as e:
            logger.error(f"Failed to analyze pages: {e}")
            return f"Error analyzing pages: {str(e)}"

    def _build_answer_prompt(
        self,
        question: str,
        pages: List[Page],
        document_name: str
    ) -> str:
        """Build prompt for answering question with page images"""
        page_nums = [str(p.page_number) for p in pages]
        pages_str = ", ".join(page_nums)

        return f"""Document: {document_name}
Analyzing pages: {pages_str}

User Question:
{question}

Instructions:
1. Carefully examine the provided page images
2. Extract relevant information to answer the question
3. Provide a clear, accurate answer based on what you see
4. If the answer isn't in the images, say so clearly
5. Reference specific page numbers when citing information

Your answer:"""

    def get_total_cost(self) -> float:
        """Get total cost of all queries in this session"""
        return self.total_cost

    def reset_cost(self):
        """Reset cost counter"""
        self.total_cost = 0.0

    async def multi_turn_conversation(
        self,
        document: Document,
        questions: List[str],
        max_pages_per_question: int = None
    ) -> List[Dict]:
        """
        Handle multiple questions in sequence (conversation mode)
        
        Args:
            document: Document to query
            questions: List of questions
            max_pages_per_question: Max pages per question (None = agent decides)
            
        Returns:
            List of result dictionaries, one per question
        """
        results = []
        
        for i, question in enumerate(questions, 1):
            logger.info(f"\n--- Question {i}/{len(questions)} ---")
            result = await self.answer_question(
                document=document,
                question=question,
                max_pages=max_pages_per_question
            )
            results.append(result)
        
        logger.info(f"\n=== Conversation Complete ===")
        logger.info(f"Total questions: {len(questions)}")
        logger.info(f"Total cost: ${self.total_cost:.4f}")
        
        return results

