# ai/page_selection_agent.py
import base64
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger

from app.processors.document import Document, Page, Partition
from app.ai.openai import OpenAIClient


class PageSelectionAgent:
    """
    Intelligent agent to select relevant pages for answering user questions
    Uses different strategies based on document size
    """
    
    def __init__(self, openai_client: OpenAIClient, storage_root: str = None):
        self.client = openai_client
        self.max_pages_to_analyze = 5  # Maximum pages to send to final Q&A
        self.max_partitions = 2  # Maximum partitions to select for large docs
        
        if storage_root is None:
            import os
            storage_root = os.environ.get("FLEX_RAG_DATA_LOCATION", "/flex_rag_data_location")
        
        self.storage_root = Path(storage_root)
        self.documents_dir = self.storage_root / "documents"
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def _load_partition_summary(self, document: Document) -> Optional[Dict]:
        """
        Load partition_summary.json for large documents
        
        Returns:
            Parsed partition summary dict or None
        """
        try:
            doc_dir = self.documents_dir / document.id
            summary_path = doc_dir / "partition_summary.json"
            
            if not summary_path.exists():
                logger.warning(f"partition_summary.json not found for document {document.id}")
                return None
            
            with open(summary_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        except Exception as e:
            logger.error(f"Failed to load partition_summary.json: {e}")
            return None
    
    async def _select_partitions_from_summary(
        self,
        question: str,
        partition_summary: Dict
    ) -> List[int]:
        """
        Select relevant partitions using partition_summary.json
        
        Args:
            question: User question
            partition_summary: Loaded partition_summary.json data
            
        Returns:
            List of selected partition IDs (max 2)
        """
        partitions_info = []
        
        for partition in partition_summary.get("partitions", []):
            partition_id = partition.get("partition_id")
            page_range = partition.get("page_range")
            summary = partition.get("summary", "")
            tables = partition.get("tables", [])
            charts = partition.get("charts", [])
            
            # Build rich context for each partition
            info_text = f"Partition {partition_id} (Pages {page_range[0]}-{page_range[1]}):\n"
            info_text += f"Summary: {summary}\n"
            
            if tables:
                info_text += f"Tables ({len(tables)}): "
                table_titles = [f"{t.get('title', 'Untitled')} (page {t.get('page_number')})" 
                               for t in tables[:3]]  # Show first 3
                info_text += ", ".join(table_titles)
                if len(tables) > 3:
                    info_text += f", and {len(tables) - 3} more..."
                info_text += "\n"
            
            if charts:
                info_text += f"Charts ({len(charts)}): "
                chart_titles = [f"{c.get('title', 'Untitled')} ({c.get('chart_type', 'unknown')}, page {c.get('page_number')})" 
                               for c in charts[:3]]  # Show first 3
                info_text += ", ".join(chart_titles)
                if len(charts) > 3:
                    info_text += f", and {len(charts) - 3} more..."
                info_text += "\n"
            
            partitions_info.append({
                "partition_id": partition_id,
                "page_range": page_range,
                "info": info_text
            })
        
        # Build prompt for partition selection
        partitions_text = "\n\n".join([p["info"] for p in partitions_info])
        
        prompt = f"""Given this user question: "{question}"

Here are detailed summaries of all document partitions, including their tables and charts:

{partitions_text}

Analyze which partitions are most likely to contain the answer to the user's question.
Consider:
1. Content relevance to the question
2. Presence of relevant tables or charts
3. Context needed to understand the answer

Select the top {self.max_partitions} most relevant partitions.

Return ONLY valid JSON:
{{
  "selected_partitions": [partition_id1, partition_id2],
  "reasoning": "Brief explanation of why these partitions were selected and what they contain that's relevant"
}}"""

        try:
            response = await self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.client.model_large,
                max_completion_tokens=500,
                temperature=0.2
            )
            
            # Parse response
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            selected_partition_ids = result.get("selected_partitions", [])
            reasoning = result.get("reasoning", "")
            
            logger.info(f"🎯 Selected partitions: {selected_partition_ids}")
            logger.info(f"📝 Reasoning: {reasoning}")
            
            # Ensure we don't select more than max_partitions
            return selected_partition_ids[:self.max_partitions]
        
        except Exception as e:
            logger.error(f"Partition selection failed: {e}")
            # Fallback: return first partition(s)
            fallback = [1, 2] if len(partitions_info) >= 2 else [1]
            logger.warning(f"Using fallback partitions: {fallback}")
            return fallback
    
    async def _select_pages_within_partitions(
        self,
        question: str,
        document: Document,
        selected_partition_ids: List[int],
        partition_summary: Dict
    ) -> List[Page]:
        """
        Select specific pages within the selected partitions
        
        Args:
            question: User question
            document: Document object
            selected_partition_ids: IDs of selected partitions
            partition_summary: Loaded partition_summary.json data
            
        Returns:
            List of selected pages (max max_pages_to_analyze)
        """
        # Get candidate pages from selected partitions
        candidate_pages = [
            p for p in document.pages
            if p.partition_id in selected_partition_ids
        ]
        
        logger.info(f"📄 Found {len(candidate_pages)} candidate pages in partitions {selected_partition_ids}")
        
        # Build detailed page information including metadata
        page_infos = []
        
        for page in candidate_pages:
            info_text = f"Page {page.page_number}:\n"
            info_text += f"Summary: {page.summary}\n"
            
            # Add table information
            if page.has_tables():
                info_text += f"Tables ({len(page.tables)}): "
                table_info = ", ".join([f"{t.title}" for t in page.tables])
                info_text += table_info + "\n"
            
            # Add chart information
            if page.has_charts():
                info_text += f"Charts ({len(page.charts)}): "
                chart_info = ", ".join([f"{c.title} ({c.chart_type})" for c in page.charts])
                info_text += chart_info + "\n"
            
            page_infos.append({
                "page_number": page.page_number,
                "info": info_text,
                "has_tables": page.has_tables(),
                "has_charts": page.has_charts()
            })
        
        # Build prompt for page selection
        pages_text = "\n".join([p["info"] for p in page_infos])
        
        # Get partition context for additional information
        partition_contexts = []
        for partition_id in selected_partition_ids:
            for p in partition_summary.get("partitions", []):
                if p.get("partition_id") == partition_id:
                    partition_contexts.append(
                        f"Partition {partition_id} context: {p.get('summary', '')}"
                    )
        
        context_text = "\n".join(partition_contexts)
        
        prompt = f"""Given this user question: "{question}"

Partition context:
{context_text}

Here are all pages from the selected partitions with their summaries, tables, and charts:

{pages_text}

Select the top {self.max_pages_to_analyze} most relevant pages to answer this question.
Consider:
1. Direct content relevance to the question
2. Presence of specific tables or charts that contain relevant data
3. Contextual pages that help understand the answer
4. Pages that together provide a complete answer

Return ONLY valid JSON:
{{
  "selected_pages": [page_number1, page_number2, page_number3, ...],
  "reasoning": "Brief explanation of why these specific pages were selected"
}}"""

        try:
            response = await self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.client.model_large,
                max_completion_tokens=600,
                temperature=0.2
            )
            
            # Parse response
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            selected_page_numbers = result.get("selected_pages", [])
            reasoning = result.get("reasoning", "")
            
            logger.info(f"📑 Selected pages: {selected_page_numbers}")
            logger.info(f"📝 Reasoning: {reasoning}")
            
            # Get Page objects
            selected_pages = [
                p for p in candidate_pages
                if p.page_number in selected_page_numbers
            ]
            
            # Sort by page number
            selected_pages.sort(key=lambda p: p.page_number)
            
            # Limit to max_pages_to_analyze
            return selected_pages[:self.max_pages_to_analyze]
        
        except Exception as e:
            logger.error(f"Page selection within partitions failed: {e}")
            # Fallback: return first N candidate pages
            logger.warning(f"Using fallback: first {self.max_pages_to_analyze} candidate pages")
            return candidate_pages[:self.max_pages_to_analyze]
    
    async def _select_pages_small_doc(
        self,
        document: Document,
        question: str
    ) -> List[Page]:
        """
        Strategy for small documents (≤20 pages)
        Use metadata to find relevant pages
        
        Args:
            document: Document object
            question: User question
            
        Returns:
            List of relevant pages (up to max_pages_to_analyze)
        """
        logger.info(f"📘 Small doc strategy: Analyzing {document.page_count} pages")
        
        # Build context from all page summaries
        page_infos = []
        for page in document.pages:
            info_text = f"Page {page.page_number}:\n"
            info_text += f"Summary: {page.summary}\n"
            
            # Add table info
            if page.has_tables():
                table_info = ", ".join([f"{t.title}" for t in page.tables])
                info_text += f"Tables: {table_info}\n"
            
            # Add chart info
            if page.has_charts():
                chart_info = ", ".join([f"{c.title} ({c.chart_type})" for c in page.charts])
                info_text += f"Charts: {chart_info}\n"
            
            page_infos.append({
                "page_number": page.page_number,
                "info": info_text
            })
        
        # Build prompt
        pages_text = "\n".join([p["info"] for p in page_infos])
        
        prompt = f"""Given this user question: "{question}"

Here are summaries of all pages in the document:

{pages_text}

Select the top {self.max_pages_to_analyze} most relevant pages to answer this question.
Consider:
- Direct content relevance to the question
- Tables or charts that might contain relevant data
- Context needed to understand the answer

Return ONLY valid JSON:
{{
  "selected_pages": [page_number1, page_number2, ...],
  "reasoning": "Brief explanation of why these pages were selected"
}}"""

        try:
            response = await self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.client.model_small,
                max_completion_tokens=500,
                temperature=0.2
            )
            
            # Parse response
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            selected_page_numbers = result.get("selected_pages", [])
            reasoning = result.get("reasoning", "")
            
            logger.info(f"📑 Selected pages: {selected_page_numbers}")
            logger.info(f"📝 Reasoning: {reasoning}")
            
            # Get Page objects
            selected_pages = [
                p for p in document.pages
                if p.page_number in selected_page_numbers
            ]
            
            # Sort by page number
            selected_pages.sort(key=lambda p: p.page_number)
            
            return selected_pages[:self.max_pages_to_analyze]
        
        except Exception as e:
            logger.error(f"Page selection failed: {e}")
            # Fallback: return first N pages
            logger.warning(f"Falling back to first {self.max_pages_to_analyze} pages")
            return document.pages[:self.max_pages_to_analyze]
    
    async def _select_pages_large_doc(
        self,
        document: Document,
        question: str
    ) -> List[Page]:
        """
        Strategy for large documents (>20 pages)
        Step 1: Select relevant partitions (max 2) using partition_summary.json
        Step 2: Select specific pages within those partitions
        
        Args:
            document: Document object
            question: User question
            
        Returns:
            List of relevant pages (up to max_pages_to_analyze)
        """
        logger.info(f"📚 Large doc strategy: {document.page_count} pages, {len(document.partitions)} partitions")
        
        # Load partition_summary.json
        partition_summary = self._load_partition_summary(document)
        
        if not partition_summary:
            logger.error("❌ partition_summary.json not available, falling back to partition objects")
            # Fallback to using partition objects from metadata.json
            return await self._select_pages_large_doc_fallback(document, question)
        
        # Step 1: Select top 2 partitions
        logger.info("🎯 Step 1: Selecting relevant partitions...")
        selected_partition_ids = await self._select_partitions_from_summary(
            question,
            partition_summary
        )
        
        if not selected_partition_ids:
            logger.error("No partitions selected, using first partition")
            selected_partition_ids = [1]
        
        # Step 2: Select specific pages within partitions
        logger.info(f"📄 Step 2: Selecting pages within partitions {selected_partition_ids}...")
        selected_pages = await self._select_pages_within_partitions(
            question,
            document,
            selected_partition_ids,
            partition_summary
        )
        
        return selected_pages
    
    async def _select_pages_large_doc_fallback(
        self,
        document: Document,
        question: str
    ) -> List[Page]:
        """
        Fallback strategy for large documents when partition_summary.json is not available
        Uses partition objects from metadata.json
        """
        logger.info("Using fallback partition selection strategy")
        
        # Build partition info from document.partitions
        partition_infos = []
        for partition in document.partitions:
            info_text = f"Partition {partition.partition_id} (Pages {partition.page_range[0]}-{partition.page_range[1]}): {partition.summary}"
            partition_infos.append({
                "partition_id": partition.partition_id,
                "page_range": partition.page_range,
                "summary": info_text
            })
        
        partitions_text = "\n".join([p["summary"] for p in partition_infos])
        
        prompt = f"""Given this user question: "{question}"

Here are document partition summaries:

{partitions_text}

Select the top {self.max_partitions} most relevant partitions to answer this question.

Return ONLY valid JSON:
{{
  "selected_partitions": [partition_id1, partition_id2],
  "reasoning": "Brief explanation"
}}"""

        try:
            response = await self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.client.model_large,
                max_completion_tokens=300
            )
            
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            selected_partition_ids = result.get("selected_partitions", [1])
            
            logger.info(f"Selected partitions (fallback): {selected_partition_ids}")
            
        except Exception as e:
            logger.error(f"Fallback partition selection failed: {e}")
            selected_partition_ids = [1, 2] if len(document.partitions) >= 2 else [1]
        
        # Get pages from selected partitions
        candidate_pages = [
            p for p in document.pages
            if p.partition_id in selected_partition_ids
        ]
        
        # Select pages within partitions (simplified)
        page_summaries = []
        for page in candidate_pages:
            summary = f"Page {page.page_number}: {page.summary}"
            if page.has_tables():
                summary += f" [Tables: {len(page.tables)}]"
            if page.has_charts():
                summary += f" [Charts: {len(page.charts)}]"
            page_summaries.append({
                "page_number": page.page_number,
                "summary": summary
            })
        
        pages_text = "\n".join([p["summary"] for p in page_summaries])
        
        prompt = f"""Given this user question: "{question}"

Here are pages from relevant partitions:

{pages_text}

Select the top {self.max_pages_to_analyze} most relevant pages.

Return ONLY valid JSON:
{{
  "selected_pages": [page_number1, page_number2, ...],
  "reasoning": "Brief explanation"
}}"""

        try:
            response = await self.client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                model=self.client.model_large,
                max_completion_tokens=400
            )
            
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            result = json.loads(content)
            selected_page_numbers = result.get("selected_pages", [])
            
            selected_pages = [
                p for p in candidate_pages
                if p.page_number in selected_page_numbers
            ]
            
            selected_pages.sort(key=lambda p: p.page_number)
            
            return selected_pages[:self.max_pages_to_analyze]
        
        except Exception as e:
            logger.error(f"Page selection failed: {e}")
            return candidate_pages[:self.max_pages_to_analyze]
    
    async def select_relevant_pages(
        self,
        document: Document,
        question: str
    ) -> List[Page]:
        """
        Main method to select relevant pages based on document size
        
        For ≤20 pages: Direct page selection
        For >20 pages: Two-step process
            1. Select top 2 partitions using partition_summary.json
            2. Select specific pages within those partitions
        
        Args:
            document: Document object
            question: User question
            
        Returns:
            List of relevant pages (up to 5 pages)
        """
        logger.info(f"🔍 Selecting relevant pages for question: {question[:100]}...")
        logger.info(f"📊 Document: {document.name} ({document.page_count} pages)")
        
        if document.page_count <= 20:
            # Small document: direct page selection
            pages = await self._select_pages_small_doc(document, question)
        else:
            # Large document: partition → pages selection
            pages = await self._select_pages_large_doc(document, question)
        
        logger.info(f"✅ Selected {len(pages)} pages: {[p.page_number for p in pages]}")
        
        return pages
    
    async def answer_question(
        self,
        document: Document,
        question: str,
        selected_pages: List[Page]
    ) -> Dict[str, any]:
        """
        Answer question using selected pages
        
        Args:
            document: Document object
            question: User question
            selected_pages: Pre-selected relevant pages
            
        Returns:
            {
                "answer": "Answer text",
                "pages_used": [page_numbers],
                "confidence": "high/medium/low",
                "selected_pages": [page_numbers shown]
            }
        """
        logger.info(f"💭 Answering question with {len(selected_pages)} pages")
        
        # Encode images
        images = []
        page_numbers = []
        for page in selected_pages:
            try:
                img_base64 = self._encode_image(page.image_path)
                images.append(img_base64)
                page_numbers.append(page.page_number)
            except Exception as e:
                logger.error(f"Failed to encode page {page.page_number}: {e}")
        
        if not images:
            return {
                "answer": "❌ Error: Could not load page images",
                "pages_used": [],
                "confidence": "low",
                "selected_pages": page_numbers
            }
        
        # Build context from page summaries
        context = []
        for page in selected_pages:
            context_text = f"Page {page.page_number}: {page.summary}"
            
            if page.has_tables():
                table_info = "; ".join([f"{t.title}" for t in page.tables])
                context_text += f" | Tables: {table_info}"
            
            if page.has_charts():
                chart_info = "; ".join([f"{c.title} ({c.chart_type})" for c in page.charts])
                context_text += f" | Charts: {chart_info}"
            
            context.append(context_text)
        
        context_text = "\n".join(context)
        
        # Build prompt
        prompt = f"""You are analyzing a document to answer a user's question.

**Document:** {document.name}
**Total Pages:** {document.page_count}
**Pages Being Analyzed:** {page_numbers}

**Page Context:**
{context_text}

**User Question:** {question}

Please analyze the provided page images carefully and answer the question.

**Instructions:**
- Provide a clear, comprehensive, and detailed answer
- Reference specific page numbers when citing information (e.g., "On page 5...")
- If referring to tables or charts, mention them specifically
- If the answer is not found in these pages, state that clearly
- Assess your confidence level based on how well these pages answer the question

**Format your response as JSON:**
{{
  "answer": "Your detailed answer here, referencing specific pages...",
  "confidence": "high/medium/low",
  "pages_referenced": [list of page numbers you actually used in your answer]
}}"""

        try:
            response = await self.client.vision_completion(
                text_prompt=prompt,
                images=images,
                model=self.client.get_model_for_document(document.page_count),
                max_completion_tokens=2000,
                detail="high",
                temperature=0.3
            )
            
            # Parse response
            content = response.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                result = json.loads(content)
                
                return {
                    "answer": result.get("answer", response),
                    "pages_used": result.get("pages_referenced", page_numbers),
                    "confidence": result.get("confidence", "medium"),
                    "selected_pages": page_numbers
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw response
                logger.warning("Could not parse JSON response, using raw answer")
                return {
                    "answer": response,
                    "pages_used": page_numbers,
                    "confidence": "medium",
                    "selected_pages": page_numbers
                }
        
        except Exception as e:
            logger.error(f"Failed to answer question: {e}")
            return {
                "answer": f"❌ Error processing question: {str(e)}",
                "pages_used": page_numbers,
                "confidence": "low",
                "selected_pages": page_numbers
            }