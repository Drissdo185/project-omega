"""
Chat agent that answers questions using vision-based document analysis.
Uses page summaries to select relevant pages, then analyzes images to answer.
"""

from typing import List, Optional, Dict
from loguru import logger

from app.models.document import Document, Page, Partition
from app.providers.base import BaseProvider
from .page_selector import PageSelector
from .partition_selector import PartitionSelector


class ChatAgent:
    """
    Intelligent agent that answers questions about documents using vision analysis.

    Workflow for small documents (≤20 pages) - 2-stage:
    1. User asks a question
    2. Agent selects relevant pages based on summaries (cost-effective filtering)
    3. Agent sends selected page images to LLM for detailed analysis
    4. Returns answer with source pages

    Workflow for large documents (>20 pages) - 3-stage hierarchical:
    1. User asks a question
    2. Agent selects 1-2 relevant partitions based on partition summaries
    3. Agent selects 2-5 relevant pages from those partitions only
    4. Agent sends selected page images to LLM for detailed analysis
    5. Returns answer with source pages
    """

    def __init__(self, provider: BaseProvider):
        """
        Initialize chat agent

        Args:
            provider: LLM provider with vision capabilities
        """
        self.provider = provider
        self.page_selector = PageSelector(provider)
        self.partition_selector = PartitionSelector(provider)
        self.total_cost = 0.0

    async def answer_question(
        self,
        document: Document,
        question: str,
        max_pages: int = None,
        use_all_pages: bool = False
    ) -> Dict:
        """
        Answer a question about a document using conditional 2-stage or 3-stage flow

        Args:
            document: Document with page summaries (and partitions if >20 pages)
            question: User's question
            max_pages: Optional hard limit on pages (None = agent decides dynamically)
            use_all_pages: If True, use all pages (no selection)

        Returns:
            Dict with answer, selected_pages, partition info, and cost information
        """
        try:
            logger.info(f"Answering question: {question}")
            logger.info(f"Document: {document.name} ({document.page_count} pages)")

            
            if document.is_large_document():

                logger.info("Using 3-stage hierarchical flow (large document)")
                selected_pages, partition_cost, selection_cost, selected_partitions = await self._three_stage_selection(
                    document=document,
                    question=question,
                    max_pages=max_pages
                )
            else:
                # 2-STAGE FLOW for small documents (≤20 pages)
                logger.info("Using 2-stage flow (small document)")
                selected_pages, selection_cost = await self._two_stage_selection(
                    document=document,
                    question=question,
                    max_pages=max_pages,
                    use_all_pages=use_all_pages
                )
                partition_cost = 0.0
                selected_partitions = []

            if not selected_pages:
                logger.warning("No pages selected, using first page as fallback")
                selected_pages = [document.pages[0]] if document.pages else []

            if not selected_pages:
                return {
                    "answer": "No pages available to analyze.",
                    "selected_pages": [],
                    "page_numbers": [],
                    "total_cost": selection_cost + partition_cost,
                    "partition_cost": partition_cost,
                    "selection_cost": selection_cost,
                    "analysis_cost": 0.0,
                    "selected_partitions": []
                }

            # Final stage: Analyze selected pages with vision to answer question
            # Use model based on flow: 2-stage uses gpt-4o-mini, 3-stage uses gpt-5
            model_for_vision = self.provider.get_model_3stage() if document.is_large_document() else self.provider.get_model_2stage()            
            logger.info(f"Analyzing {len(selected_pages)} selected pages with vision model {model_for_vision}...")
            answer = await self._analyze_pages_for_answer(
                pages=selected_pages,
                question=question,
                document_name=document.name,
                model=model_for_vision
            )

            analysis_cost = self.provider.get_last_cost() or 0.0
            total_cost = partition_cost + selection_cost + analysis_cost

            # Track cumulative cost
            self.total_cost += total_cost

            result = {
                "answer": answer,
                "selected_pages": selected_pages,
                "page_numbers": [p.page_number for p in selected_pages],
                "total_cost": total_cost,
                "partition_cost": partition_cost,
                "selection_cost": selection_cost,
                "analysis_cost": analysis_cost,
                "selected_partitions": [p.partition_id for p in selected_partitions] if selected_partitions else []
            }

            logger.info(f"Question answered successfully!")
            logger.info(f"Cost breakdown - Partition: ${partition_cost:.4f}, Selection: ${selection_cost:.4f}, Analysis: ${analysis_cost:.4f}, Total: ${total_cost:.4f}")

            return result

        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            raise

    async def _two_stage_selection(
        self,
        document: Document,
        question: str,
        max_pages: int = None,
        use_all_pages: bool = False
    ) -> tuple:
        """
        2-stage selection for small documents (≤20 pages)
        Uses gpt-4o-mini model

        Returns:
            tuple: (selected_pages, selection_cost)
        """
        if use_all_pages:
            logger.info("Stage 1/2: Using all pages (no filtering)")
            selected_pages = await self.page_selector.select_all_pages(document)
        else:
            logger.info(f"Stage 1/2: Selecting relevant pages using {self.provider.get_model_2stage()}...")
            selected_pages = await self.page_selector.select_relevant_pages(
                document=document,
                user_question=question,
                max_pages=max_pages,
                model=self.provider.get_model_2stage()  # Use gpt-4o-mini for 2-stage
            )

        selection_cost = self.provider.get_last_cost() or 0.0
        return selected_pages, selection_cost

    async def _three_stage_selection(
        self,
        document: Document,
        question: str,
        max_pages: int = None
    ) -> tuple:
        """
        3-stage hierarchical selection for large documents (>20 pages)
        Uses gpt-5 model for all stages

        Returns:
            tuple: (selected_pages, partition_cost, selection_cost, selected_partitions)
        """
        # Stage 1: Select 1-2 relevant partitions using gpt-5 (if partitions exist)
        if document.has_partitions():
            logger.info(f"Stage 1/3: Selecting relevant partitions using {self.provider.get_model_3stage()}...")
            selected_partitions = await self.partition_selector.select_relevant_partitions(
                document=document,
                user_question=question,
                max_partitions=2,
                model=self.provider.get_model_3stage()
            )
            partition_cost = self.provider.get_last_cost() or 0.0

            if not selected_partitions:
                logger.warning("No partitions selected, falling back to first partition")
                selected_partitions = document.partitions[:1] if document.partitions else []

            # Log selected partitions
            for partition in selected_partitions:
                logger.info(f"  Selected Partition {partition.partition_id}: Pages {partition.page_range[0]}-{partition.page_range[1]}")
        else:
            logger.warning("Document has no partitions, skipping partition selection stage")
            selected_partitions = []
            partition_cost = 0.0

        # Stage 2: Select 2-5 relevant pages from selected partitions using gpt-5
        logger.info(f"Stage 2/3: Selecting relevant pages using {self.provider.get_model_3stage()}...")
        selected_pages = await self.page_selector.select_relevant_pages(
            document=document,
            user_question=question,
            max_pages=max_pages,
            partitions=selected_partitions if selected_partitions else None,
            model=self.provider.get_model_3stage()
        )
        selection_cost = self.provider.get_last_cost() or 0.0

        # Log selected pages
        logger.info(f"  Selected {len(selected_pages)} pages: {[p.page_number for p in selected_pages]}")

        return selected_pages, partition_cost, selection_cost, selected_partitions

    async def _analyze_pages_for_answer(
        self,
        pages: List[Page],
        question: str,
        document_name: str,
        model: Optional[str] = None
    ) -> str:
        """
        Analyze selected page images to answer the question

        Args:
            pages: Selected pages to analyze
            question: User's question
            document_name: Name of the document
            model: Specific model to use for vision analysis

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
                max_tokens=3000,
                model=model
            )

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

