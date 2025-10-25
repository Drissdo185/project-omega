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
            page_analyses = []
            page_summaries = {}  # Store summaries for each page

            # Analyze each page
            pages_to_analyze = document.pages[:max_pages] if max_pages else document.pages

            for page in pages_to_analyze:
                # Provide context about the document
                context = f"This is page {page.page_number} of {document.page_count} from '{document.name}'"

                page_analysis, page_summary = await self.analyze_page(page, context, detailed)
                page_analyses.append(page_analysis)
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
            logger.error(f"Full response was: {response}")

            # Return minimal analysis
            return PageAnalysis(
                page_number=page_number,
                labels=PageLabel(
                    page_number=page_number,
                    content_type=ContentType.TEXT
                )
            ), "Could not parse analysis"
    
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
                return json.loads(block.strip())
            except json.JSONDecodeError:
                continue

        # Try to find JSON object in response
        start = response.find('{')
        end = response.rfind('}') + 1

        if start >= 0 and end > start:
            json_str = response[start:end]

            # Try to fix common JSON issues
            # Remove trailing commas before closing braces/brackets
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)

            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.debug(f"Attempted to parse: {json_str[:500]}...")
                raise ValueError(f"Invalid JSON in response: {e}")

        raise ValueError("No valid JSON found in response")
    
    def _save_analysis(self, analysis: DocumentAnalysis, document: Document):
        """Save complete metadata.json file with document and analysis data

        Args:
            analysis: DocumentAnalysis object with page analyses
            document: Document object with pages (includes updated summaries)
        """
        try:
            # Save to documents directory
            doc_dir = self.documents_dir / analysis.document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            output_path = doc_dir / "metadata.json"

            # Create complete metadata structure
            metadata = {
                "id": document.id,
                "name": document.name,
                "page_count": document.page_count,
                "status": document.status.value,
                "summary": analysis.overall_summary,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "pages": [p.to_dict() for p in document.pages],
                "page_analyses": analysis.page_analyses,
                "total_cost": analysis.total_cost
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
    
    def load_analysis(self, document_id: str) -> Optional[DocumentAnalysis]:
        """Load previously saved document analysis from metadata.json"""
        try:
            metadata_path = self.documents_dir / document_id / "metadata.json"

            if not metadata_path.exists():
                return None

            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Check if analysis data exists in metadata
            if 'page_analyses' not in data:
                return None

            # Reconstruct DocumentAnalysis from metadata
            analysis_data = {
                'document_id': data.get('id', document_id),
                'document_name': data.get('name', ''),
                'overall_summary': data.get('summary', ''),
                'page_analyses': data.get('page_analyses', {}),
                'total_cost': data.get('total_cost', 0.0)
            }

            return DocumentAnalysis.from_dict(analysis_data)

        except Exception as e:
            logger.error(f"Failed to load analysis for {document_id}: {e}")
            return None

    def get_all_analyses(self) -> List[DocumentAnalysis]:
        """Load all saved document analyses from metadata.json files"""
        analyses = []

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

                # Only include if analysis data exists
                if 'page_analyses' in data:
                    analysis_data = {
                        'document_id': data.get('id', doc_dir.name),
                        'document_name': data.get('name', ''),
                        'overall_summary': data.get('summary', ''),
                        'page_analyses': data.get('page_analyses', {}),
                        'total_cost': data.get('total_cost', 0.0)
                    }
                    analyses.append(DocumentAnalysis.from_dict(analysis_data))
            except Exception as e:
                logger.error(f"Failed to load {metadata_file}: {e}")

        return analyses