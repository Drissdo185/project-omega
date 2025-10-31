# app/processors/pdf_vision.py
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import os
import json
import hashlib
import math
from typing import Optional, List, Tuple, Any
from datetime import datetime, timezone
from loguru import logger

from .base import BaseProcessor
from app.models.document import Page, Document, DocumentStatus

class VisionPDFProcessor(BaseProcessor):
    """
    PDF processor with intelligent page combining for optimal quality:
    - If PDF has < 20 pages: combines all pages into 1 high-quality image
    - If PDF has ≥ 20 pages: uses maximum 20 pages per combined image
    - Adaptive grid layouts based on page count
    - Enhanced image quality with higher resolution and JPEG quality
    """

    def __init__(
        self,
        render_scale: float = 2.0,  # Increased for better quality
        jpeg_quality: int = 95,     # Increased for better quality
        max_image_size: tuple = (4000, 4000),  # Increased max size for better quality
        max_pages_per_image: int = 20,
        storage_root: str = None
    ):
        if storage_root is None:
            storage_root = os.environ.get("FLEX_RAG_DATA_LOCATION", "./app/flex_rag_data_location")
    
        self.storage_root = Path(storage_root)
        self.render_scale = render_scale
        self.jpeg_quality = jpeg_quality
        self.max_image_size = max_image_size
        self.max_pages_per_image = max_pages_per_image
        self.documents_dir = self.storage_root / "documents"
        self.cache_dir = self.storage_root / "cache" / "summaries"

        # Ensure directories exist
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_doc_id(self, file_path: str) -> str:
        """Generate unique document ID from file path and timestamp"""
        content = f"{file_path}_{datetime.now(timezone.utc).isoformat()}"
        return f"doc_{hashlib.md5(content.encode()).hexdigest()[:12]}"

    def _create_document_directory(self, doc_id: str) -> Path:
        """Create directory structure for a document"""
        doc_dir = self.documents_dir / doc_id
        pages_dir = doc_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        return doc_dir


    def _create_combined_image(self, pdf_pages: List[Any], start_page: int) -> Tuple[Image.Image, List[dict]]:
        """
        Create a combined image from PDF pages arranged in an optimal grid
        
        Args:
            pdf_pages: List of PyMuPDF page objects
            start_page: Starting page number (1-indexed)
            
        Returns:
            Tuple of (combined_image, page_info_list)
        """
        actual_pages = len(pdf_pages)
        
        # Calculate optimal grid layout based on number of pages
        if actual_pages <= 1:
            cols, rows = 1, 1
        elif actual_pages <= 2:
            cols, rows = 2, 1
        elif actual_pages <= 4:
            cols, rows = 2, 2
        elif actual_pages <= 6:
            cols, rows = 3, 2  # 6 pages: 3 columns, 2 rows
        elif actual_pages <= 9:
            cols, rows = 3, 3  # 7-9 pages: 3 columns, 3 rows
        elif actual_pages <= 12:
            cols, rows = 4, 3
        elif actual_pages <= 16:
            cols, rows = 4, 4
        else:
            # For more than 16 pages, use 4x5 grid (max 20 pages)
            cols, rows = 4, 5
        
        logger.info(f"Creating combined image with {actual_pages} pages in {cols}x{rows} grid")
        
        # Render individual pages with higher quality
        page_images = []
        page_info = []
        
        for i, page in enumerate(pdf_pages):
            page_num = start_page + i
            
            # Render page to pixmap with higher quality
            mat = fitz.Matrix(self.render_scale, self.render_scale)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            # Store original dimensions before resizing
            original_width, original_height = img.size
            
            page_images.append(img)
            page_info.append({
                "page_number": page_num,
                "width": original_width,
                "height": original_height,
                "grid_position": {"row": i // cols, "col": i % cols}
            })
        
        # Calculate cell size for grid layout with better quality
        if page_images:
            # Find max dimensions
            max_width = max(img.width for img in page_images)
            max_height = max(img.height for img in page_images)
            
            # Calculate cell size to fit within max_image_size while maintaining quality
            cell_width = min(max_width, self.max_image_size[0] // cols)
            cell_height = min(max_height, self.max_image_size[1] // rows)
            
            # Ensure minimum cell size for quality
            min_cell_size = 300  # Minimum pixels per cell
            cell_width = max(cell_width, min_cell_size)
            cell_height = max(cell_height, min_cell_size)
            
            # Create combined image
            combined_width = cell_width * cols
            combined_height = cell_height * rows
            combined_img = Image.new("RGB", (combined_width, combined_height), "white")
            
            # Place each page in the grid with high quality resampling
            for i, (img, info) in enumerate(zip(page_images, page_info)):
                row = i // cols
                col = i % cols
                
                # Resize page to fit cell while maintaining aspect ratio and quality
                img_resized = img.copy()
                img_resized.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
                
                # Calculate position to center the image in the cell
                x_offset = col * cell_width + (cell_width - img_resized.width) // 2
                y_offset = row * cell_height + (cell_height - img_resized.height) // 2
                
                # Paste the resized image
                combined_img.paste(img_resized, (x_offset, y_offset))
                
                # Add page number annotation with better visibility
                try:
                    draw = ImageDraw.Draw(combined_img)
                    # Try to load a better font
                    try:
                        font = ImageFont.truetype("arial.ttf", 20)  # Larger font
                    except:
                        try:
                            font = ImageFont.truetype("calibri.ttf", 20)
                        except:
                            font = ImageFont.load_default()
                    
                    # Draw page number in the top-left corner with better styling
                    text = f"P{info['page_number']}"
                    text_x = col * cell_width + 8
                    text_y = row * cell_height + 8
                    
                    # Draw background rectangle for text with padding
                    bbox = draw.textbbox((text_x, text_y), text, font=font)
                    padding = 4
                    bg_rect = [bbox[0] - padding, bbox[1] - padding, bbox[2] + padding, bbox[3] + padding]
                    draw.rectangle(bg_rect, fill="white", outline="black", width=2)
                    draw.text((text_x, text_y), text, fill="black", font=font)
                    
                except Exception as e:
                    logger.warning(f"Failed to add page number annotation: {e}")
            
            return combined_img, page_info
        else:
            # Return empty image if no pages
            return Image.new("RGB", (100, 100), "white"), []

    async def process(self, file_path: str, doc_id: Optional[str] = None) -> Document:
        """
        Convert PDF to combined images with intelligent page grouping:
        - If PDF has < 20 pages: combine all pages into 1 image
        - If PDF has ≥ 20 pages: maximum 20 pages per combined image

        Args:
            file_path: Path to the PDF file
            doc_id: Optional document ID (generated if not provided)

        Returns:
            Document object with combined image metadata
        """
        try:
            logger.info(f"Processing PDF with intelligent page combining: {file_path}")

            # Generate document ID if not provided
            if not doc_id:
                doc_id = self._generate_doc_id(file_path)

            # Create document directory structure
            doc_dir = self._create_document_directory(doc_id)
            pages_dir = doc_dir / "pages"

            # Open PDF with PyMuPDF
            pdf = fitz.open(file_path)
            total_pages = len(pdf)
            
            logger.info(f"PDF has {total_pages} pages")
            
            # Determine page grouping strategy
            if total_pages < 20:
                # Combine all pages into 1 image
                pages_per_group = total_pages
                logger.info(f"PDF has < 20 pages → combining all {total_pages} pages into 1 image")
            else:
                # Use maximum 20 pages per image
                pages_per_group = self.max_pages_per_image
                logger.info(f"PDF has ≥ 20 pages → using maximum {pages_per_group} pages per image")
            
            # Process pages in groups
            combined_images = []
            all_page_info = []
            
            for start_idx in range(0, total_pages, pages_per_group):
                end_idx = min(start_idx + pages_per_group, total_pages)
                page_group = [pdf[i] for i in range(start_idx, end_idx)]
                start_page_num = start_idx + 1
                
                logger.info(f"Processing pages {start_page_num} to {start_idx + len(page_group)} ({len(page_group)} pages in this group)")
                
                # Create combined image for this group
                combined_img, page_info = self._create_combined_image(page_group, start_page_num)
                
                # Save combined image with high quality
                image_num = (start_idx // pages_per_group) + 1
                output_path = pages_dir / f"combined_{image_num}.jpg"
                
                # Apply intelligent resizing to maintain quality
                original_size = combined_img.size
                if (combined_img.width > self.max_image_size[0] or 
                    combined_img.height > self.max_image_size[1]):
                    # Calculate resize ratio to maintain quality
                    ratio = min(
                        self.max_image_size[0] / combined_img.width,
                        self.max_image_size[1] / combined_img.height
                    )
                    new_width = int(combined_img.width * ratio)
                    new_height = int(combined_img.height * ratio)
                    combined_img = combined_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    logger.debug(f"Resized combined image {image_num} from {original_size} to {combined_img.size}")
                
                # Save with high quality settings
                combined_img.save(
                    output_path,
                    "JPEG",
                    quality=self.jpeg_quality,
                    optimize=True,
                    progressive=True  # Progressive JPEG for better quality
                )
                
                # Store image info
                combined_images.append({
                    "image_path": str(output_path),
                    "image_number": image_num,
                    "width": combined_img.width,
                    "height": combined_img.height,
                    "page_range": (start_page_num, start_idx + len(page_group)),
                    "pages_in_image": len(page_group)
                })
                
                # Add page info with image reference
                for page in page_info:
                    page["image_path"] = str(output_path)
                    page["combined_image_number"] = image_num
                    all_page_info.append(page)
                
                logger.debug(f"Created high-quality combined image {image_num} with {len(page_group)} pages → {output_path}")

            pdf.close()

            # Create Page objects for each individual page (for compatibility)
            pages = []
            for page_info in all_page_info:
                pages.append(Page(
                    page_number=page_info["page_number"],
                    image_path=page_info["image_path"],
                    width=page_info["width"],
                    height=page_info["height"],
                    combined_image_number=page_info["combined_image_number"],
                    grid_position=page_info["grid_position"]
                ))

            # Create Document object
            document = Document(
                id=doc_id,
                name=Path(file_path).name,
                page_count=total_pages,
                pages=pages,
                combined_images=combined_images,
                status=DocumentStatus.READY,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            logger.info(f"Successfully processed {total_pages} pages into {len(combined_images)} high-quality combined image(s)")
            logger.info(f"Document directory: {doc_dir}")
            logger.info(f"Strategy: {'All pages in 1 image' if total_pages < 20 else f'Max {self.max_pages_per_image} pages per image'}")

            return document

        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            raise
    
    def supports(self, file_path: str) -> bool:
        """Check if file is PDF"""
        return file_path.lower().endswith('.pdf')