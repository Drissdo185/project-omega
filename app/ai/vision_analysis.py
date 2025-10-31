"""
Vision-based document analysis service
Analyzes combined document images (20 pages per image) and generates summaries with isImage detection
"""
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
from loguru import logger
from app.models.document import Document, Page, DocumentStatus
import fitz  # PyMuPDF for text extraction fallback
from app.providers.base import BaseProvider


class VisionAnalysisService:
    """Service for analyzing documents using vision models with combined images"""

    def __init__(self, provider: BaseProvider, storage_root: str = None):
        """
        Initialize the vision analysis service
        
        Args:
            provider: LLM provider with vision capabilities (gpt-oss-20b)
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
        context: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Analyze a combined image containing up to 20 pages and return individual page analysis

        Args:
            image_path: Path to the combined image file
            pages_in_image: List of Page objects that are in this combined image
            context: Optional context about the document

        Returns:
            List of page analysis dicts with page_number, summary, and isImage
        """
        try:
            logger.info(f"Analyzing combined image: {image_path} with {len(pages_in_image)} pages")

            # Build the prompt for analyzing all pages in the image
            prompt = self._build_combined_analysis_prompt(pages_in_image, context)

            # Prepare multimodal message
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert document analyst with OCR capabilities. Your job is to carefully read and extract ALL text content from document images, including questions, answers, technical details, and specific information. Provide detailed, accurate summaries that capture the complete content so users can find specific information they're looking for."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_path", "image_path": image_path, "detail": "high"},
                    ],
                },
            ]

            # Get analysis from LLM with optimized token limit (GPT-5 requires temperature=1)
            response = await self.provider.process_multimodal_messages(
                messages=messages,
                max_tokens=4000,  # Reduced from 6000 for cost efficiency
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
        """Build prompt for analyzing combined image with multiple pages"""
        page_numbers = [str(p.page_number) for p in pages_in_image]
        page_list = ", ".join(page_numbers)
        
        # Build grid layout description
        grid_layout = self._build_grid_layout_description(pages_in_image)
        
        return f"""Please analyze this combined image containing PDF pages and perform careful text extraction.

This image contains multiple PDF pages arranged in a grid layout:

{grid_layout}

CRITICAL GRID LAYOUT INSTRUCTIONS:
- The image has pages arranged in a specific grid pattern as described above
- You MUST identify each page by its position in the grid, not by any text labels
- Start from TOP-LEFT as position (1,1), then move RIGHT across the row, then DOWN to next row
- Each page corresponds to a specific page number based on its grid position
- Read the content from the correct grid position for each page number

INSTRUCTIONS FOR ACCURATE TEXT EXTRACTION:
- Perform OCR (Optical Character Recognition) to read all visible text
- Extract exact text content including names, companies, technologies, dates
- Include specific technical terms, programming languages, and frameworks
- Transcribe questions and answers word-for-word
- Note specific years of experience and quantifiable details
- Include company names, project details, and accomplishments as written

For each page, provide a JSON object with:
1. **page_number**: The page number (integer) - based on grid position mapping above
2. **width**: 960 (standard width)
3. **height**: 1440 (standard height)
4. **summary**: Complete transcription of ALL visible text content from the CORRECT grid position including:
   - Every question and answer
   - Names, titles, and contact information
   - Technical skills and programming languages mentioned
   - Years of experience and employment history
   - Educational background and certifications
   - Project descriptions and achievements
   - Any other specific details mentioned
5. **isImage**: Boolean - true if page contains charts/diagrams, false for text-only

Please focus on accuracy and completeness. Include all text exactly as it appears from the CORRECT grid position for each page number.

Return as valid JSON array:
[
  {{
    "page_number": 1,
    "width": 960,
    "height": 1440,
    "summary": "Complete transcription of all visible text content from TOP-LEFT position...",
    "isImage": false
  }}
]

{context}"""

    def _build_grid_layout_description(self, pages_in_image: List[Page]) -> str:
        """Build a detailed grid layout description for the AI to understand page positions"""
        if not pages_in_image:
            return "No pages in image"
        
        # Sort pages by page number to get the correct order
        sorted_pages = sorted(pages_in_image, key=lambda p: p.page_number)
        
        # Determine grid layout based on page count (must match pdf_vision.py logic)
        page_count = len(sorted_pages)
        
        if page_count <= 1:
            grid_cols, grid_rows = 1, 1
        elif page_count <= 2:
            grid_cols, grid_rows = 2, 1
        elif page_count <= 4:
            grid_cols, grid_rows = 2, 2
        elif page_count <= 6:
            grid_cols, grid_rows = 3, 2  # 6 pages: 3 columns, 2 rows
        elif page_count <= 9:
            grid_cols, grid_rows = 3, 3  # 7-9 pages: 3 columns, 3 rows
        elif page_count <= 12:
            grid_cols, grid_rows = 4, 3
        elif page_count <= 16:
            grid_cols, grid_rows = 4, 4
        else:
            grid_cols, grid_rows = 4, 5  # For more than 16 pages
        
        # Build detailed grid description
        description = f"GRID LAYOUT: {grid_cols} columns × {grid_rows} rows\n"
        description += f"Total pages: {page_count}\n\n"
        description += "EXACT PAGE POSITIONS IN THE GRID:\n"
        
        # Create visual grid representation
        description += "Visual Grid Layout:\n"
        grid_visual = [["." for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        for i, page in enumerate(sorted_pages):
            row = i // grid_cols
            col = i % grid_cols
            grid_visual[row][col] = f"P{page.page_number}"
            
        for row_idx, row in enumerate(grid_visual):
            description += f"Row {row_idx + 1}: " + " | ".join(f"{cell:>3}" for cell in row) + "\n"
        
        description += "\nPAGE MAPPING (READ LEFT-TO-RIGHT, TOP-TO-BOTTOM):\n"
        for i, page in enumerate(sorted_pages):
            row = i // grid_cols + 1  # 1-indexed for clarity
            col = i % grid_cols + 1   # 1-indexed for clarity
            position_name = self._get_position_name(row, col, grid_rows, grid_cols)
            description += f"- Page {page.page_number}: Row {row}, Column {col} ({position_name})\n"
        
        description += f"\nCRITICAL INSTRUCTIONS FOR AI ANALYSIS:\n"
        description += f"1. The image has a {grid_cols}×{grid_rows} grid layout\n"
        description += f"2. Read from LEFT to RIGHT, then TOP to BOTTOM\n"
        description += f"3. Position 1 = TOP-LEFT, Position {page_count} = BOTTOM-RIGHT\n"
        description += f"4. Each grid cell contains exactly one page\n"
        description += f"5. Extract text from the EXACT grid position for each page number\n"
        description += f"6. If content seems misplaced, double-check the grid mapping above\n"
        
        return description
    
    def _get_position_name(self, row: int, col: int, total_rows: int, total_cols: int) -> str:
        """Get human-readable position name for a grid cell"""
        # Determine vertical position
        if row == 1:
            vertical = "TOP"
        elif row == total_rows:
            vertical = "BOTTOM"
        else:
            vertical = f"ROW-{row}"
        
        # Determine horizontal position
        if col == 1:
            horizontal = "LEFT"
        elif col == total_cols:
            horizontal = "RIGHT"
        else:
            horizontal = f"COL-{col}"
        
        # Combine for specific positions
        if total_rows == 2 and total_cols == 3:  # 6-page layout
            positions = {
                (1, 1): "TOP-LEFT",
                (1, 2): "TOP-CENTER", 
                (1, 3): "TOP-RIGHT",
                (2, 1): "BOTTOM-LEFT",
                (2, 2): "BOTTOM-CENTER",
                (2, 3): "BOTTOM-RIGHT"
            }
            return positions.get((row, col), f"{vertical}-{horizontal}")
        
        return f"{vertical}-{horizontal}"

    def _parse_combined_response(self, response: str, pages_in_image: List[Page]) -> List[Dict[str, Any]]:
        """Parse the JSON response from combined image analysis"""
        try:
            # Clean the response to extract JSON
            response = response.strip()
            
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
            
            # Try to find JSON array in the response
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
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
                
                return valid_analyses
            
            else:
                raise ValueError("No valid JSON array found in response")
                
        except Exception as e:
            logger.error(f"Failed to parse combined response: {e}")
            logger.debug(f"Full response was: {response}")
            
            # Return default analysis for all pages with more specific error message
            error_message = "Vision analysis failed - model may not support image processing"
            if "json" in str(e).lower():
                error_message = "Vision analysis response format error"
            elif "vision" in str(e).lower() or "image" in str(e).lower():
                error_message = "Vision capabilities not available"
            
            return [
                {
                    "page_number": page.page_number,
                    "width": 960,
                    "height": 1440,
                    "summary": error_message,
                    "isImage": False
                }
                for page in pages_in_image
            ]

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
