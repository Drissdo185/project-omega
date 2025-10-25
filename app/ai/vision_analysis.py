"""
Vision-based document analysis service
Analyzes document images and generates summaries with labels
"""

import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from loguru import logger

from app.models.document import Document, Page
from app.ai.analysis import (
    DocumentAnalysis, PageAnalysis, PageLabel,
    DocumentCategory, ContentType
)
from app.providers.base import BaseProvider


class VisionAnalysisService:
    """Service for analyzing documents using vision models"""
    
    def __init__(self, provider: BaseProvider, storage_root: str = None):
        """
        Initialize the vision analysis service
        
        Args:
            provider: LLM provider with vision capabilities
            storage_root: Root directory for storing analysis results
        """
        self.provider = provider
        
        if storage_root is None:
            import os
            storage_root = os.environ.get("FLEX_RAG_DATA_LOCATION", "./app/flex_rag_data_location")
        
        self.storage_root = Path(storage_root)
        self.documents_dir = self.storage_root / "documents"
        self.documents_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"VisionAnalysisService initialized with storage: {self.documents_dir}")

    async def analyze_page(
        self,
        page: Page,
        context: str = "",
        detailed: bool = True
    ) -> tuple[PageAnalysis, str]:
        """
        Analyze a single page using vision model

        Args:
            page: Page object with image path
            context: Optional context about the document
            detailed: Whether to do detailed analysis or quick summary

        Returns:
            tuple: (PageAnalysis object, page_summary string)
        """
        try:
            logger.info(f"Analyzing page {page.page_number}: {page.image_path}")

            # Build the prompt based on detail level
            if detailed:
                prompt = self._build_detailed_analysis_prompt(context)
            else:
                prompt = self._build_quick_analysis_prompt(context)

            # Prepare multimodal message
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert document analyst. Analyze images and provide structured information."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_path",
                            "image_path": page.image_path,
                            "detail": "high"
                        }
                    ]
                }
            ]

            # Get analysis from LLM
            response = await self.provider.process_multimodal_messages(
                messages=messages,
                max_tokens=2000 if detailed else 1000,
            )

            # Log the raw response for debugging
            logger.debug(f"LLM response for page {page.page_number}: {response[:200]}...")

            # Parse the structured response
            page_analysis, page_summary = self._parse_analysis_response(response, page.page_number)

            # Get cost
            cost = self.provider.get_last_cost() or 0.0

            logger.info(f"Page {page.page_number} analyzed (cost: ${cost:.4f})")

            return page_analysis, page_summary

        except Exception as e:
            logger.error(f"Failed to analyze page {page.page_number}: {e}")
            # Return minimal analysis on error
            return PageAnalysis(
                page_number=page.page_number,
                labels=PageLabel(
                    page_number=page.page_number,
                    content_type=ContentType.UNKNOWN
                )
            ), "Analysis failed"
    
    async def analyze_document(
        self,
        document: Document,
        detailed: bool = True,
        max_pages: Optional[int] = None
    ) -> DocumentAnalysis:
        """
        Analyze entire document with all pages

        Args:
            document: Document object with pages
            detailed: Whether to do detailed analysis
            max_pages: Optional limit on number of pages to analyze

        Returns:
            DocumentAnalysis with all page analyses
        """
        try:
            logger.info(f"Starting analysis of document: {document.name} ({document.page_count} pages)")

            total_cost = 0.0
            page_summaries = {}  # Store summaries for each page

            # Analyze each page
            pages_to_analyze = document.pages[:max_pages] if max_pages else document.pages

            for page in pages_to_analyze:
                # Provide context about the document
                context = f"This is page {page.page_number} of {document.page_count} from '{document.name}'"

                page_analysis, page_summary = await self.analyze_page(page, context, detailed)
                page_summaries[page.page_number] = page_summary

                # Update the page object's summary field
                page.summary = page_summary

                # Track cost
                cost = self.provider.get_last_cost() or 0.0
                total_cost += cost

            # Convert page_analyses list to dictionary format: {"page_1": {...}, "page_2": {...}}
            page_analyses_dict = {}
            for pa in page_analyses:
                page_key = f"page_{pa.page_number}"
                page_analyses_dict[page_key] = pa.to_dict()

            # Create document-level summary
            overall_summary = f"Document with {len(page_analyses)} pages analyzed"

            # Create document analysis
            doc_analysis = DocumentAnalysis(
                document_id=document.id,
                document_name=document.name,
                overall_summary=overall_summary,
                page_analyses=page_analyses_dict,
                total_cost=total_cost
            )

            # Update document's summary field
            document.summary = overall_summary

            # Save complete metadata to JSON
            self._save_analysis(doc_analysis, document)

            logger.info(f"Document analysis complete. Total cost: ${total_cost:.4f}")

            return doc_analysis

        except Exception as e:
            logger.error(f"Failed to analyze document {document.id}: {e}")
            raise
    
    def _build_detailed_analysis_prompt(self, context: str = "") -> str:
        """Build prompt for detailed page analysis"""

        return f"""Analyze this document page in detail. {context}

IMPORTANT: Return ONLY valid JSON with no additional text or markdown formatting.

Required JSON structure:
{{
    "page_summary": "Brief 1-2 sentence summary of the page content",
    "content_type": "text",
    "topics": ["topic1", "topic2"],
    "language": "en",
    "confidence_score": 0.95,
    "extracted_data": {{}}
}}

Valid content_type values: text, table, chart, form, image, mixed
Language should be ISO code (en, vi, etc.)
Confidence score should be between 0.0 and 1.0

Return ONLY the JSON object, no explanations."""
    
    def _build_quick_analysis_prompt(self, context: str = "") -> str:
        """Build prompt for quick page analysis"""
        return f"""Quickly analyze this document page. {context}

IMPORTANT: Return ONLY valid JSON with no additional text.

Required JSON structure:
{{
    "page_summary": "Brief summary (1 sentence)",
    "content_type": "text",
    "topics": ["topic1", "topic2"]
}}

Valid content_type: text, table, chart, form, image, mixed
Return ONLY the JSON object."""
    
    def _parse_analysis_response(self, response: str, page_number: int) -> tuple[PageAnalysis, str]:
        """Parse LLM response into PageAnalysis object and page summary

        Returns:
            tuple: (PageAnalysis, page_summary)
        """
        try:
            # Try to extract JSON from response
            data = self._parse_json_response(response)

            # Build PageLabel
            label = PageLabel(
                page_number=page_number,
                content_type=ContentType(data.get("content_type", "text")),
                topics=data.get("topics", []),
                language=data.get("language", "en"),
                confidence_score=data.get("confidence_score", 0.8)
            )

            page_analysis = PageAnalysis(
                page_number=page_number,
                labels=label,
                extracted_data=data.get("extracted_data", {})
            )

            # Extract page summary for the pages array
            page_summary = data.get("page_summary", "")

            return page_analysis, page_summary

        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            logger.error(f"Full response was: {response[:500]}...")

            # Return minimal analysis with extracted summary if possible
            page_summary = self._extract_summary_fallback(response)
            
            return PageAnalysis(
                page_number=page_number,
                labels=PageLabel(
                    page_number=page_number,
                    content_type=ContentType.TEXT
                )
            ), page_summary
    
    def _parse_json_response(self, response: str) -> dict:
        """Extract and parse JSON from LLM response with robust error handling"""
        import re

        try:
            # Try direct parse first
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        code_blocks = re.findall(code_block_pattern, response, re.DOTALL)

        for block in code_blocks:
            try:
                # Clean the block first
                cleaned_block = self._clean_json_string(block.strip())
                return json.loads(cleaned_block)
            except json.JSONDecodeError:
                continue

        # Try to find JSON object in response
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = response[start:end]
            
            # Clean and fix the JSON string
            json_str = self._clean_json_string(json_str)

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                # If still failing, try to extract just the essential fields
                logger.warning(f"JSON parse error, attempting fallback: {e}")
                return self._extract_essential_fields(response)

        # Final fallback - return minimal structure
        logger.warning("No valid JSON found, using minimal fallback")
        return {
            "page_summary": "Analysis incomplete",
            "content_type": "text",
            "topics": [],
            "language": "en",
            "confidence_score": 0.0,
            "extracted_data": {}
        }
    
    def _clean_json_string(self, json_str: str) -> str:
        """Clean and fix common JSON issues"""
        import re
        
        # Remove trailing commas before closing braces/brackets
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        
        # Fix incomplete strings at the end
        # If the JSON ends with an incomplete string, close it
        if json_str.count('"') % 2 != 0:
            # Find the last quote and check if it's properly closed
            last_quote_pos = json_str.rfind('"')
            # Check if there's content after the last quote that looks incomplete
            after_quote = json_str[last_quote_pos+1:].strip()
            if after_quote and not after_quote.startswith(',') and not after_quote.startswith('}') and not after_quote.startswith(']'):
                # Likely an incomplete string, close it
                json_str = json_str[:last_quote_pos+1] + '"'
                
                # Now we need to close any open brackets/braces
                # Count open brackets and braces
                open_braces = json_str.count('{') - json_str.count('}')
                open_brackets = json_str.count('[') - json_str.count(']')
                
                # Close any incomplete arrays first
                if open_brackets > 0:
                    json_str += ']' * open_brackets
                    
                # Close any incomplete objects
                if open_braces > 0:
                    json_str += '}' * open_braces
        
        # Handle truncated arrays/objects
        open_braces = json_str.count('{') - json_str.count('}')
        open_brackets = json_str.count('[') - json_str.count(']')
        
        if open_brackets > 0:
            json_str += ']' * open_brackets
        if open_braces > 0:
            json_str += '}' * open_braces
            
        return json_str
    
    def _extract_essential_fields(self, response: str) -> dict:
        """Extract essential fields using regex when JSON parsing fails"""
        import re
        
        result = {
            "page_summary": "Analysis incomplete",
            "content_type": "text",
            "topics": [],
            "language": "en",
            "confidence_score": 0.0,
            "extracted_data": {}
        }
        
        # Try to extract page_summary
        summary_match = re.search(r'"page_summary"\s*:\s*"([^"]*)"', response)
        if summary_match:
            result["page_summary"] = summary_match.group(1)
        
        # Try to extract content_type
        content_match = re.search(r'"content_type"\s*:\s*"([^"]*)"', response)
        if content_match:
            result["content_type"] = content_match.group(1)
        
        # Try to extract topics array
        topics_match = re.search(r'"topics"\s*:\s*\[([^\]]*)\]', response)
        if topics_match:
            topics_str = topics_match.group(1)
            # Extract individual topics
            topics = re.findall(r'"([^"]*)"', topics_str)
            result["topics"] = topics
        
        # Try to extract language
        lang_match = re.search(r'"language"\s*:\s*"([^"]*)"', response)
        if lang_match:
            result["language"] = lang_match.group(1)
        
        # Try to extract confidence score
        conf_match = re.search(r'"confidence_score"\s*:\s*([0-9.]+)', response)
        if conf_match:
            try:
                result["confidence_score"] = float(conf_match.group(1))
            except:
                pass
        
        return result
    
    def _extract_summary_fallback(self, response: str) -> str:
        """Extract page summary from response when JSON parsing fails"""
        import re
        
        # Try to extract page_summary field
        summary_match = re.search(r'"page_summary"\s*:\s*"([^"]*)"', response)
        if summary_match:
            return summary_match.group(1)
        
        # Fallback to generic message
        return "Page analysis incomplete - parsing error"
    
    def _save_analysis(self, analysis: DocumentAnalysis, document: Document):
        """Save complete metadata.json file with document metadata and page summaries

        Note: Detailed page analyses are NOT saved to reduce file size and complexity.
        Only the document metadata and page summaries are persisted.

        Args:
            analysis: DocumentAnalysis object (used for overall_summary only)
            document: Document object with pages (includes LLM-generated summaries)
        """
        try:
            # Save to documents directory
            doc_dir = self.documents_dir / analysis.document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            output_path = doc_dir / "metadata.json"

            # Create complete metadata structure (only document info and pages with summaries)
            metadata = {
                "id": document.id,
                "name": document.name,
                "page_count": document.page_count,
                "status": document.status.value,
                "summary": analysis.overall_summary,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "pages": [p.to_dict() for p in document.pages]
            }

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved complete metadata to: {output_path}")

            # Update global index
            self._update_index(document)

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _update_index(self, document: Document):
        """Update or create the global document index"""
        try:
            index_path = self.documents_dir / "index.json"

            # Load existing index
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            else:
                index = []

            # Add or update document entry
            index_entry = {
                "id": document.id,
                "name": document.name,
                "page_count": document.page_count,
                "status": document.status.value,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat()
            }

            # Remove existing entry if present
            index = [e for e in index if e.get("id") != document.id]
            index.append(index_entry)

            # Save updated index
            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

            logger.debug(f"Updated index at {index_path}")

        except Exception as e:
            logger.error(f"Failed to update index: {e}")
    
    def load_document(self, document_id: str) -> Optional[Document]:
        """Load previously saved document from metadata.json

        Note: This loads the document with page summaries, not the detailed analysis.
        Detailed analysis (page_analyses) is no longer saved to reduce file size.
        """
        try:
            metadata_path = self.documents_dir / document_id / "metadata.json"

            if not metadata_path.exists():
                return None

            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Reconstruct Document from metadata
            return Document.from_dict(data)

        except Exception as e:
            logger.error(f"Failed to load document for {document_id}: {e}")
            return None

    def get_all_documents(self) -> List[Document]:
        """Load all saved documents from metadata.json files

        Note: Returns documents with page summaries, not detailed analysis.
        Detailed analysis (page_analyses) is no longer saved to reduce file size.
        """
        documents = []

        # Iterate through all document directories
        for doc_dir in self.documents_dir.iterdir():
            if not doc_dir.is_dir() or doc_dir.name == "index.json":
                continue

            metadata_file = doc_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Load document
                document = Document.from_dict(data)
                documents.append(document)
            except Exception as e:
                logger.error(f"Failed to load {metadata_file}: {e}")

        return documents