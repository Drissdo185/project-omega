"""
Chat agent that answers questions using vision-based document analysis.
Uses page summaries to select relevant pages, then analyzes images to answer.
OPTIMIZED: Phase 1 - Vision-Specific Prompting, Dynamic Token Budgeting
"""

from typing import List, Optional, Dict
from loguru import logger

from app.models.document import Document, Page
from app.providers.base import BaseProvider
from .page_selector import PageSelector
from app.ai.vision_prompts import VisionPromptLibrary, AnalysisTask
from app.ai.token_optimizer import TokenBudgetOptimizer, ContentComplexity, TaskPriority


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
        OPTIMIZED: Uses vision-specific Q&A prompts and dynamic token budgeting
        
        Args:
            pages: Selected pages to analyze
            question: User's question
            document_name: Name of the document
            
        Returns:
            Answer text from LLM
        """
        try:
            # Detect content complexity from page summaries
            all_summaries = " ".join([p.summary for p in pages if p.summary])
            complexity = TokenBudgetOptimizer.detect_complexity_from_text(all_summaries)
            
            # Calculate DYNAMIC token budget
            has_tables = any("table" in (p.summary or "").lower() for p in pages)
            has_diagrams = any(word in (p.summary or "").lower() 
                             for p in pages 
                             for word in ["diagram", "chart", "figure", "graph"])
            
            token_budget = TokenBudgetOptimizer.calculate_vision_budget(
                num_pages=len(pages),
                complexity=complexity,
                priority=TaskPriority.HIGH,  # Q&A is high priority
                has_tables=has_tables,
                has_diagrams=has_diagrams
            )
            
            logger.info(f"Dynamic token budget for Q&A: {token_budget} (complexity: {complexity.value})")

            # Build OPTIMIZED multimodal prompt using vision-specific library
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
                    "content": """You are a helpful and professional document analyst with vision capabilities. Your goal is to provide clear, accurate, and user-friendly answers.

RESPONSE STYLE:
- Use natural, conversational language (avoid overly technical jargon unless necessary)
- Structure your answer with clear sections if covering multiple points
- Use bullet points or numbered lists for better readability when appropriate
- Bold important terms or key findings using **markdown**
- If citing specific information, mention the page number naturally (e.g., "According to page 3...")

ACCURACY REQUIREMENTS:
- Answer ONLY based on what you can see in the provided document pages
- If information is not in the pages, clearly state "I don't see this information in the provided pages"
- Be specific and precise - quote or paraphrase relevant text when appropriate
- If something is unclear in the document, acknowledge the ambiguity

FORMATTING:
- Start with a direct answer to the question
- Then provide supporting details or context
- End with additional relevant information if helpful
- Keep paragraphs short (2-4 sentences max) for easy reading"""
                },
                {
                    "role": "user",
                    "content": content
                }
            ]

            # Get answer from vision model with OPTIMIZED dynamic token budget
            response = await self.provider.process_multimodal_messages(
                messages=messages,
                max_tokens=token_budget,  # Dynamic budget optimization
                temperature=1.0  # GPT-5 only supports temperature=1
            )

            # Validate response
            if not response or not response.strip():
                logger.warning("LLM returned empty response, trying text-based fallback")
                return await self._fallback_text_based_answer(pages, question)
            
            return response.strip()

        except Exception as e:
            logger.error(f"Failed to analyze pages with vision: {e}")
            logger.info("Attempting text-based fallback answer")
            try:
                return await self._fallback_text_based_answer(pages, question)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return f"I apologize, but I encountered an error while analyzing the document: {str(e)}"
    
    async def _fallback_text_based_answer(
        self,
        pages: List[Page],
        question: str
    ) -> str:
        """
        Fallback method to answer questions using page summaries instead of images
        
        Args:
            pages: Selected pages
            question: User's question
            
        Returns:
            Answer based on page summaries
        """
        logger.info("Using text-based fallback with page summaries")
        
        # Build context from page summaries
        context_parts = []
        for page in pages:
            if page.summary and page.summary.strip():
                context_parts.append(f"Page {page.page_number}:\n{page.summary}\n")
        
        if not context_parts:
            return "I couldn't find relevant information to answer your question. The document pages may not contain text or the analysis failed."
        
        context = "\n".join(context_parts)
        
        # Ask LLM to answer based on text summaries
        prompt = f"""📚 **Document Content (Summaries):**

{context}

❓ **User Question:** {question}

**INSTRUCTIONS:**
- Answer the question based on the content above
- Format your answer for easy reading (use **bold**, bullet points, etc.)
- Cite page numbers when referencing specific information
- If the answer isn't in the content, clearly state: "I don't see this information in the provided pages."
- Keep your answer concise but complete

**Your Answer:**"""

        messages = [
            {
                "role": "system",
                "content": """You are a helpful and professional document analyst. Provide clear, well-formatted answers that are easy to read and understand.

RESPONSE GUIDELINES:
- Write in a natural, conversational tone
- Use **bold** for important terms or key findings
- Use bullet points or numbered lists when presenting multiple items
- Keep paragraphs short (2-4 sentences) for easy reading
- Start with the direct answer, then provide supporting details
- Mention page numbers when citing information (e.g., "According to page 5...")
- If the answer is not in the content, clearly state that

Remember: You are answering based on document summaries (not full images), so be clear and accurate with the information available."""
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        # Use text-only processing
        response = await self.provider.process_text_messages(
            messages=messages,
            max_tokens=1500,  # Increased for better formatted fallback answers
            temperature=1.0
        )
        
        return response.strip() if response else "Unable to generate an answer."

    def _build_answer_prompt(
        self,
        question: str,
        pages: List[Page],
        document_name: str
    ) -> str:
        """Build prompt for answering question with page images"""
        page_nums = [str(p.page_number) for p in pages]
        pages_str = ", ".join(page_nums)

        return f"""📄 **Document:** {document_name}
📖 **Analyzing Pages:** {pages_str}

❓ **Question:** {question}

**INSTRUCTIONS:**
1. **Examine the page images carefully** - Look at all text, tables, diagrams, and visual elements
2. **Answer the question directly** - Start with the main answer, then provide details
3. **Use only what you see** - Don't use external knowledge or make assumptions
4. **Be specific** - Quote relevant text and mention page numbers (e.g., "On page 3, the document states...")
5. **Format for readability:**
   - Use **bold** for key terms or findings
   - Use bullet points (•) or numbered lists when listing multiple items
   - Keep paragraphs short and scannable
6. **If not found** - If the answer isn't in these pages, clearly say: "I don't see this information in the provided pages."

**Your Answer:**"""

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

