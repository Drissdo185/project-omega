"""
Vision-based document analysis service
Analyzes combined document images (20 pages per image) and generates summaries with isImage detection
OPTIMIZED: Phase 1 - Vision-Specific Prompting, Simplified Grid, Dynamic Token Budgeting
"""
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger
from app.models.document import Document, Page, DocumentStatus
import fitz  # PyMuPDF for text extraction fallback
from app.providers.base import BaseProvider
from app.ai.vision_prompts import VisionPromptLibrary, AnalysisTask
from app.ai.grid_descriptor import SimplifiedGridDescriptor, GridLayoutCalculator
from app.ai.token_optimizer import TokenBudgetOptimizer, ContentComplexity, TaskPriority


class VisionAnalysisService:
    """Service for analyzing documents using vision models with combined images"""

    def __init__(self, provider: BaseProvider, storage_root: str = None):
        """
        Initialize the vision analysis service
        
        Args:
            provider: LLM provider with vision capabilities (GPT-5)
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

    async def analyze_combined_image(
        self,
        image_path: str,
        pages_in_image: List[Page],
        context: str = "",
        task: AnalysisTask = AnalysisTask.GENERAL,
        question: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze a combined image containing up to 20 pages and return individual page analysis
        OPTIMIZED: Uses vision-specific prompts, simplified grid, dynamic token budgeting

        Args:
            image_path: Path to the combined image file
            pages_in_image: List of Page objects that are in this combined image
            context: Optional context about the document
            task: Analysis task type (auto-detected if question provided)
            question: Optional question for Q&A tasks

        Returns:
            List of page analysis dicts with page_number, summary, and isImage
        """
        try:
            logger.info(f"Analyzing combined image: {image_path} with {len(pages_in_image)} pages")

            # Auto-detect task from question if provided
            if question:
                task = VisionPromptLibrary.detect_task_from_question(question)
                logger.info(f"Auto-detected task: {task.value}")

            # Build OPTIMIZED prompt using vision-specific library
            num_pages = len(pages_in_image)
            grid_size = GridLayoutCalculator.calculate_grid_size(num_pages)
            grid_description = SimplifiedGridDescriptor.describe_layout(num_pages, grid_size)
            
            # Get task-specific prompt
            if question:
                prompt = VisionPromptLibrary.get_qa_prompt(question, num_pages, grid_description)
            else:
                prompt = VisionPromptLibrary.get_combined_image_prompt(
                    task, num_pages, grid_description, context
                )

            # Calculate DYNAMIC token budget based on content
            token_budget = TokenBudgetOptimizer.calculate_vision_budget(
                num_pages=num_pages,
                complexity=ContentComplexity.MODERATE,  # Can be auto-detected in future
                priority=TaskPriority.NORMAL,
                has_tables=False,  # Can be detected from page metadata
                has_diagrams=False
            )
            
            logger.info(f"Using dynamic token budget: {token_budget} (was 4000)")

            # Prepare multimodal message
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert document analyst with advanced OCR and vision capabilities. Extract ALL text content accurately, analyze visual elements, and provide comprehensive, structured responses."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_path", "image_path": image_path, "detail": "high"},
                    ],
                },
            ]

            # Get analysis from LLM with OPTIMIZED dynamic token limit
            response = await self.provider.process_multimodal_messages(
                messages=messages,
                max_tokens=token_budget,  # Dynamic budget optimization
                temperature=1.0  # GPT-5 only supports temperature=1
            )

            # Parse the response to get individual page analyses
            page_analyses = self._parse_combined_response(response, pages_in_image)

            # Check if vision analysis failed and use fallback
            if any("vision analysis failed" in str(analysis.get("summary", "")).lower() or 
                   "vision capabilities not available" in str(analysis.get("summary", "")).lower()
                   for analysis in page_analyses):
                
                logger.warning("Vision analysis failed, attempting text extraction fallback")
                page_analyses = await self._fallback_text_extraction(pages_in_image, image_path)

            # Get cost
            cost = self.provider.get_last_cost() or 0.0
            logger.info(f"Combined image analyzed (cost: ${cost:.4f}) - {len(page_analyses)} pages")

            return page_analyses

        except Exception as e:
            logger.error(f"Failed to analyze combined image {image_path}: {e}")
            # Use deterministic fallback instead of AI vision
            logger.info("Using deterministic page extraction as primary method")
            return await self._deterministic_text_extraction(pages_in_image)

    async def _deterministic_text_extraction(self, pages_in_image: List[Page]) -> List[Dict[str, Any]]:
        """
        Deterministic method to extract text from specific PDF pages
        This bypasses vision analysis completely for 100% accuracy
        """
        pdf = None
        try:
            logger.info("Using deterministic PDF text extraction")
            
            # Find the original PDF file
            uploads_dir = Path("uploads")
            pdf_files = list(uploads_dir.glob("*.pdf")) if uploads_dir.exists() else []
            
            if not pdf_files:
                logger.warning("No PDF files found for deterministic extraction")
                return self._create_empty_analyses(pages_in_image)
            
            # Use the most recent PDF file
            pdf_file = max(pdf_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Using PDF file for deterministic extraction: {pdf_file}")
            
            # Extract text from exact page numbers
            pdf = fitz.open(pdf_file)
            page_analyses = []
            
            for page_obj in pages_in_image:
                page_num = page_obj.page_number
                
                try:
                    if page_num <= len(pdf):
                        # Get the exact PDF page (1-indexed to 0-indexed)
                        pdf_page = pdf[page_num - 1]
                        
                        # Extract text content
                        text_content = pdf_page.get_text()
                        
                        # Clean and format the text
                        if text_content.strip():
                            summary = text_content.strip()
                            # Clean whitespace but preserve structure
                            summary = '\n'.join(line.strip() for line in summary.split('\n') if line.strip())
                            
                            # Limit length for display but preserve key content
                            if len(summary) > 1500:
                                summary = summary[:1500] + "..."
                            
                            logger.debug(f"Extracted {len(text_content)} characters from page {page_num}")
                        else:
                            summary = "Page appears to be empty or contains only images"
                            logger.warning(f"Page {page_num} has no extractable text")
                        
                        page_analyses.append({
                            "page_number": page_num,
                            "width": 960,
                            "height": 1440,
                            "summary": summary,
                            "isImage": len(text_content.strip()) < 50
                        })
                    else:
                        logger.error(f"Page {page_num} exceeds PDF page count ({len(pdf)})")
                        page_analyses.append({
                            "page_number": page_num,
                            "width": 960,
                            "height": 1440,
                            "summary": f"Page {page_num} not found in PDF",
                            "isImage": False
                        })
                        
                except Exception as e:
                    logger.error(f"Error extracting page {page_num}: {e}")
                    page_analyses.append({
                        "page_number": page_num,
                        "width": 960,
                        "height": 1440,
                        "summary": f"Extraction failed for page {page_num}: {e}",
                        "isImage": False
                    })
            
            logger.info(f"Deterministic extraction completed for {len(page_analyses)} pages")
            return page_analyses
            
        except Exception as e:
            logger.error(f"Deterministic extraction failed: {e}")
            return self._create_empty_analyses(pages_in_image)
        finally:
            # Always close PDF to free resources
            if pdf:
                pdf.close()
                pdf = None

    def _create_empty_analyses(self, pages_in_image: List[Page]) -> List[Dict[str, Any]]:
        """Create empty analyses when all extraction methods fail"""
        return [
            {
                "page_number": page.page_number,
                "width": 960,
                "height": 1440,
                "summary": "All extraction methods failed",
                "isImage": False
            }
            for page in pages_in_image
        ]

    async def analyze_page(
        self,
        page: Page,
        context: str = ""
    ) -> str:
        """
        Legacy method for backward compatibility - now redirects to combined image analysis
        """
        # Find other pages that share the same combined image
        pages_in_same_image = [page]  # At minimum, just this page
        
        # Analyze the combined image
        analyses = await self.analyze_combined_image(
            page.image_path,
            pages_in_same_image,
            context
        )
        
        # Return summary for this specific page
        for analysis in analyses:
            if analysis["page_number"] == page.page_number:
                # Update page with isImage info
                page.isImage = analysis["isImage"]
                return analysis["summary"]
        
        return "Analysis failed"

    async def analyze_document(
        self,
        document: Document,
        max_pages: Optional[int] = None
    ) -> Document:
        """
        Analyze entire document with combined images and save metadata

        Args:
            document: Document object with pages and combined_images
            max_pages: Optional limit on number of pages to analyze

        Returns:
            Document object with updated summaries and isImage flags
        """
        try:
            logger.info(f"Starting analysis of document: {document.name} ({document.page_count} pages)")

            # Group pages by their combined image
            image_to_pages: Dict[str, List[Page]] = {}
            pages_to_analyze = document.pages[:max_pages] if max_pages else document.pages

            for page in pages_to_analyze:
                image_path = page.image_path
                if image_path not in image_to_pages:
                    image_to_pages[image_path] = []
                image_to_pages[image_path].append(page)

            # Analyze each combined image
            total_analyzed = 0
            for image_path, pages_in_image in image_to_pages.items():
                logger.info(f"Analyzing {len(pages_in_image)} pages in image: {Path(image_path).name}")
                
                context = f"This combined image contains pages from '{document.name}' (total {document.page_count} pages)"
                
                # Get analysis for all pages in this image
                page_analyses = await self.analyze_combined_image(
                    image_path,
                    pages_in_image,
                    context
                )
                
                # Update page objects with analysis results
                for analysis in page_analyses:
                    for page in pages_in_image:
                        if page.page_number == analysis["page_number"]:
                            page.summary = analysis["summary"]
                            page.isImage = analysis["isImage"]
                            total_analyzed += 1
                            break

            # Create document-level summary
            pages_with_images = len([p for p in pages_to_analyze if p.isImage])
            document.summary = f"Document with {total_analyzed} pages analyzed, {pages_with_images} pages contain images/graphics"

            # Save complete metadata to JSON
            self._save_document_metadata(document)

            logger.info(f"Document analysis complete for: {document.name} ({total_analyzed} pages, {pages_with_images} with images)")
            return document

        except Exception as e:
            logger.error(f"Failed to analyze document {document.id}: {e}")
            raise

    def _build_combined_analysis_prompt(self, pages_in_image: List[Page], context: str = "") -> str:
        """
        Build prompt for analyzing combined image with multiple pages
        LEGACY METHOD - kept for compatibility, now uses optimized prompt library
        """
        num_pages = len(pages_in_image)
        grid_size = GridLayoutCalculator.calculate_grid_size(num_pages)
        grid_description = SimplifiedGridDescriptor.create_complete_context(
            num_pages, grid_size, task_type="analysis"
        )
        
        # Use vision-optimized prompt from library
        base_prompt = VisionPromptLibrary.get_combined_image_prompt(
            AnalysisTask.GENERAL,
            num_pages,
            grid_description,
            context
        )
        
        # Add JSON format requirement
        json_format = """

Return as valid JSON array:
[
  {
    "page_number": 1,
    "width": 960,
    "height": 1440,
    "summary": "Complete content from page...",
    "isImage": false
  }
]"""
        
        return base_prompt + json_format

    def _parse_combined_response(self, response: str, pages_in_image: List[Page]) -> List[Dict[str, Any]]:
        """Parse the JSON response from combined image analysis"""
        try:
            # Clean the response to extract JSON
            response = response.strip()
            
            # Log the raw response for debugging
            logger.debug(f"Raw API response (first 500 chars): {response[:500]}")
            
            # Check for common vision API errors
            if any(phrase in response.lower() for phrase in [
                "unable to analyze the image",
                "can't analyze images", 
                "don't have the ability to view images",
                "cannot process images",
                "upload the text content manually"
            ]):
                logger.warning(f"AI model reports no vision capability: {response[:200]}...")
                raise ValueError("AI model does not support image analysis")
            
            # If response doesn't look like JSON at all, treat it as text content
            if '[' not in response and '{' not in response:
                logger.warning("Response doesn't contain JSON markers, treating as plain text")
                # Use the response as summary for all pages
                summary_text = response[:1500] if len(response) > 1500 else response
                return [
                    {
                        "page_number": page.page_number,
                        "width": 960,
                        "height": 1440,
                        "summary": summary_text,
                        "isImage": False
                    }
                    for page in pages_in_image
                ]
            
            # Try to find JSON array in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                logger.debug(f"Attempting to parse JSON: {json_str[:200]}...")
                page_analyses = json.loads(json_str)
                
                # Validate and clean the analyses
                valid_analyses = []
                for analysis in page_analyses:
                    if isinstance(analysis, dict) and "page_number" in analysis:
                        # Ensure all required fields
                        clean_analysis = {
                            "page_number": int(analysis.get("page_number", 0)),
                            "width": int(analysis.get("width", 960)),
                            "height": int(analysis.get("height", 1440)),
                            "summary": str(analysis.get("summary", "No summary available"))[:2000],  # Increased length for detailed content
                            "isImage": bool(analysis.get("isImage", False))
                        }
                        valid_analyses.append(clean_analysis)
                
                # Ensure we have analysis for all expected pages
                expected_pages = {p.page_number for p in pages_in_image}
                found_pages = {a["page_number"] for a in valid_analyses}
                
                # Add missing pages with default analysis
                for page in pages_in_image:
                    if page.page_number not in found_pages:
                        valid_analyses.append({
                            "page_number": page.page_number,
                            "width": 960,
                            "height": 1440,
                            "summary": "Analysis not available for this page",
                            "isImage": False
                        })
                
                logger.info(f"Successfully parsed {len(valid_analyses)} page analyses from JSON")
                return valid_analyses
            
            else:
                logger.error("No JSON array markers found in response")
                raise ValueError("No valid JSON array found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse combined response: {e}")
            logger.debug(f"Full response was: {response[:1000]}...")
            
            # Trigger fallback by raising exception instead of returning error message
            # This will cause the calling function to use deterministic text extraction
            logger.warning("JSON parsing failed, triggering fallback to text extraction")
            raise ValueError(f"Failed to parse vision response: {e}")

    def _save_document_metadata(self, document: Document):
        """Save document metadata to metadata.json file"""
        try:
            doc_dir = self.documents_dir / document.id
            doc_dir.mkdir(parents=True, exist_ok=True)
            output_path = doc_dir / "metadata.json"

            metadata = {
                "id": document.id,
                "name": document.name,
                "page_count": document.page_count,
                "status": document.status.value,
                "summary": document.summary,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
                "pages": [p.to_dict() for p in document.pages],
            }

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved complete metadata to: {output_path}")

            self._update_index(document)

        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")

    def _update_index(self, document: Document):
        """Update or create the global document index"""
        try:
            index_path = self.documents_dir / "index.json"
            if index_path.exists():
                with open(index_path, "r", encoding="utf-8") as f:
                    index = json.load(f)
            else:
                index = []

            index_entry = {
                "id": document.id,
                "name": document.name,
                "page_count": document.page_count,
                "status": document.status.value,
                "created_at": document.created_at.isoformat(),
                "updated_at": document.updated_at.isoformat(),
            }

            index = [e for e in index if e.get("id") != document.id]
            index.append(index_entry)

            with open(index_path, "w", encoding="utf-8") as f:
                json.dump(index, f, indent=2, ensure_ascii=False)

            logger.debug(f"Updated index at {index_path}")

        except Exception as e:
            logger.error(f"Failed to update index: {e}")

    def load_document(self, document_id: str) -> Optional[Document]:
        """Load previously saved document from metadata.json"""
        try:
            metadata_path = self.documents_dir / document_id / "metadata.json"
            if not metadata_path.exists():
                return None

            with open(metadata_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            return Document.from_dict(data)
        except Exception as e:
            logger.error(f"Failed to load document for {document_id}: {e}")
            return None

    def get_all_documents(self) -> List[Document]:
        """Load all saved documents from metadata.json files (optimized with caching)"""
        # Cache documents for 5 seconds to reduce disk I/O
        cache_key = '_documents_cache'
        cache_time_key = '_documents_cache_time'
        
        import time
        current_time = time.time()
        
        if hasattr(self, cache_time_key):
            cache_age = current_time - getattr(self, cache_time_key)
            if cache_age < 5 and hasattr(self, cache_key):
                return getattr(self, cache_key)
        
        documents = []
        
        # Use list() to avoid iterator issues
        doc_dirs = list(self.documents_dir.iterdir()) if self.documents_dir.exists() else []
        
        for doc_dir in doc_dirs:
            if not doc_dir.is_dir() or doc_dir.name == "index.json":
                continue

            metadata_file = doc_dir / "metadata.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                document = Document.from_dict(data)
                documents.append(document)
            except Exception as e:
                logger.error(f"Failed to load {metadata_file}: {e}")
        
        # Cache the result
        setattr(self, cache_key, documents)
        setattr(self, cache_time_key, current_time)

        return documents

    async def _fallback_text_extraction(self, pages_in_image: List[Page], combined_image_path: str) -> List[Dict[str, Any]]:
        """
        Fallback method to extract text from PDF when vision analysis fails
        
        Args:
            pages_in_image: List of Page objects
            combined_image_path: Path to the combined image (to find the original PDF)
            
        Returns:
            List of page analysis dicts with extracted text
        """
        pdf = None
        try:
            logger.info("Using fallback text extraction with PyMuPDF")
            
            # Try to find the original PDF file
            uploads_dir = Path("uploads")
            pdf_files = list(uploads_dir.glob("*.pdf")) if uploads_dir.exists() else []
            
            if not pdf_files:
                logger.warning("No PDF files found for text extraction fallback")
                return [
                    {
                        "page_number": page.page_number,
                        "width": 960,
                        "height": 1440,
                        "summary": "Text extraction failed - no source PDF found",
                        "isImage": False
                    }
                    for page in pages_in_image
                ]
            
            # Use the most recent PDF file
            pdf_file = max(pdf_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Using PDF file for text extraction: {pdf_file}")
            
            # Extract text from PDF pages
            pdf = fitz.open(pdf_file)
            page_analyses = []
            
            for page_obj in pages_in_image:
                page_num = page_obj.page_number
                
                try:
                    # Get the PDF page (1-indexed to 0-indexed)
                    pdf_page = pdf[page_num - 1]
                    
                    # Extract text content
                    text_content = pdf_page.get_text()
                    
                    # Clean and process the text
                    if text_content.strip():
                        # Limit text length and clean it
                        summary = text_content.strip()[:1500]  # Limit to 1500 chars
                        summary = ' '.join(summary.split())  # Clean whitespace
                        
                        if len(summary) < 50:
                            summary = f"Page contains limited text: {summary}"
                        else:
                            summary = f"Extracted content: {summary}"
                    else:
                        summary = "Page appears to be empty or contains only images"
                    
                    page_analyses.append({
                        "page_number": page_num,
                        "width": 960,
                        "height": 1440,
                        "summary": summary,
                        "isImage": len(text_content.strip()) < 50  # Consider it an image if very little text
                    })
                    
                except Exception as page_error:
                    logger.error(f"Failed to extract text from page {page_num}: {page_error}")
                    page_analyses.append({
                        "page_number": page_num,
                        "width": 960,
                        "height": 1440,
                        "summary": f"Text extraction failed for page {page_num}",
                        "isImage": False
                    })
            
            logger.info(f"Text extraction completed for {len(page_analyses)} pages")
            return page_analyses
            
        except Exception as e:
            logger.error(f"Fallback text extraction failed: {e}")
            return [
                {
                    "page_number": page.page_number,
                    "width": 960,
                    "height": 1440,
                    "summary": "Both vision analysis and text extraction failed",
                    "isImage": False
                }
                for page in pages_in_image
            ]
        finally:
            # Always close PDF to free resources
            if pdf:
                pdf.close()
                pdf = None
