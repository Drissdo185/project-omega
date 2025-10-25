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
    ) -> PageAnalysis:
        """
        Analyze a single page using vision model
        
        Args:
            page: Page object with image path
            context: Optional context about the document
            detailed: Whether to do detailed analysis or quick summary
        
        Returns:
            PageAnalysis object with summary and labels
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
            
            # Parse the structured response
            analysis_data = self._parse_analysis_response(response, page.page_number)
            
            # Get cost
            cost = self.provider.get_last_cost() or 0.0
            
            logger.info(f"Page {page.page_number} analyzed (cost: ${cost:.4f})")
            
            return analysis_data
            
        except Exception as e:
            logger.error(f"Failed to analyze page {page.page_number}: {e}")
            # Return minimal analysis on error
            return PageAnalysis(
                page_number=page.page_number,
                summary="Analysis failed",
                detailed_content="",
                labels=PageLabel(
                    page_number=page.page_number,
                    content_type=ContentType.UNKNOWN
                )
            )
    
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
            
            # Analyze each page
            pages_to_analyze = document.pages[:max_pages] if max_pages else document.pages

            for page in pages_to_analyze:
                # Provide context about the document
                context = f"This is page {page.page_number} of {document.page_count} from '{document.name}' (Category: {document.folder})"

                page_analysis = await self.analyze_page(page, context, detailed)
                page_analyses.append(page_analysis)

                # Track cost
                cost = self.provider.get_last_cost() or 0.0
                total_cost += cost

            # Collect all topics from page analyses
            all_topics = []
            for pa in page_analyses:
                all_topics.extend(pa.labels.topics)
            unique_topics = list(set(all_topics))

            # Convert page_analyses list to dictionary format: {"page_1": [...], "page_2": [...]}
            page_analyses_dict = {}
            for pa in page_analyses:
                page_key = f"page_{pa.page_number}"
                page_analyses_dict[page_key] = pa.to_dict()

            # Create document analysis directly without synthesis
            doc_analysis = DocumentAnalysis(
                document_id=document.id,
                document_name=document.name,
                category=DocumentCategory.GENERAL,  # Default category
                overall_summary=f"Document with {len(page_analyses)} pages analyzed",
                page_analyses=page_analyses_dict,
                document_topics=unique_topics,
                metadata={
                    "page_count": document.page_count,
                    "folder": document.folder
                },
                total_cost=total_cost
            )
            
            # Save analysis to JSON
            self._save_analysis(doc_analysis)
            
            logger.info(f"Document analysis complete. Total cost: ${total_cost:.4f}")
            
            return doc_analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze document {document.id}: {e}")
            raise
    
    def _build_detailed_analysis_prompt(self, context: str = "") -> str:
        """Build prompt for detailed page analysis"""
        
        return f"""Analyze this document page in detail. {context}

        Provide the following information in JSON format:

        {{
            "summary": "Brief 1-2 sentence summary of the page content",
            "detailed_content": "Detailed description of what's on the page (3-5 sentences)",
            "content_type": "text|table|chart|form|image|mixed",
            "topics": ["list", "of", "main", "topics"],
            "language": "en",
            "confidence_score": 0.0-1.0,
            "extracted_data": {{
                "any_structured_data": "like dates, numbers, etc."
            }}
        }}

        Be thorough and extract all relevant information."""
    
    def _build_quick_analysis_prompt(self, context: str = "") -> str:
        """Build prompt for quick page analysis"""
        return f"""Quickly analyze this document page. {context}

        Provide in JSON format:

        {{
            "summary": "Brief summary (1 sentence)",
            "content_type": "text|table|chart|form|image|mixed",
            "topics": ["main", "topics"],
            "keywords": ["key", "words"]
        }}"""
    
    def _parse_analysis_response(self, response: str, page_number: int) -> PageAnalysis:
        """Parse LLM response into PageAnalysis object"""
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
            
            # Build PageAnalysis
            return PageAnalysis(
                page_number=page_number,
                summary=data.get("summary", ""),
                detailed_content=data.get("detailed_content", ""),
                labels=label,
                extracted_data=data.get("extracted_data", {})
            )
            
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            logger.debug(f"Response was: {response}")
            
            # Return minimal analysis
            return PageAnalysis(
                page_number=page_number,
                summary="Could not parse analysis",
                detailed_content=response[:500],
                labels=PageLabel(
                    page_number=page_number,
                    content_type=ContentType.TEXT
                )
            )
    
    def _parse_json_response(self, response: str) -> dict:
        """Extract and parse JSON from LLM response"""
        try:
            # Try direct parse
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to find JSON in response
            start = response.find('{')
            end = response.rfind('}') + 1
            
            if start >= 0 and end > start:
                json_str = response[start:end]
                return json.loads(json_str)
            
            raise ValueError("No valid JSON found in response")
    
    def _save_analysis(self, analysis: DocumentAnalysis):
        """Save document analysis to metadata.json file"""
        try:
            # Save to documents directory
            doc_dir = self.documents_dir / analysis.document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            output_path = doc_dir / "metadata.json"

            # Load existing metadata if it exists
            existing_data = {}
            if output_path.exists():
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)

            # Extract only analysis-specific fields (avoid duplicating document fields)
            analysis_dict = analysis.to_dict()
            analysis_fields = {
                'summary': analysis_dict.get('overall_summary'),  # Map overall_summary to summary field
                'page_analyses': analysis_dict.get('page_analyses'),
                'total_cost': analysis_dict.get('total_cost')
            }

            # Merge only analysis fields with existing metadata
            merged_data = {**existing_data, **analysis_fields}

            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved analysis to: {output_path}")

        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
    
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

            # Reconstruct DocumentAnalysis from merged metadata
            # DocumentAnalysis.from_dict expects document_id and document_name
            # which are stored as 'id' and 'name' in the metadata
            analysis_data = {
                'document_id': data.get('id', document_id),
                'document_name': data.get('name', ''),
                'category': data.get('category', 'general'),
                'overall_summary': data.get('overall_summary', ''),
                'page_analyses': data.get('page_analyses', []),
                'document_topics': data.get('document_topics', []),
                'metadata': {
                    'page_count': data.get('page_count'),
                    'folder': data.get('folder')
                },
                'analysis_timestamp': data.get('analysis_timestamp', datetime.now().isoformat()),
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
                        'category': data.get('category', 'general'),
                        'overall_summary': data.get('overall_summary', ''),
                        'page_analyses': data.get('page_analyses', {}),
                        'document_topics': data.get('document_topics', []),
                        'metadata': {
                            'page_count': data.get('page_count'),
                            'folder': data.get('folder')
                        },
                        'analysis_timestamp': data.get('analysis_timestamp', datetime.now().isoformat()),
                        'total_cost': data.get('total_cost', 0.0)
                    }
                    analyses.append(DocumentAnalysis.from_dict(analysis_data))
            except Exception as e:
                logger.error(f"Failed to load {metadata_file}: {e}")

        return analyses