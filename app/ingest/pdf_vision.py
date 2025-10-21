# app/processors/pdf_vision.py
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import tempfile
import os
from typing import List
from loguru import logger

from .base import BaseProcessor
from app.models.document import Page

class VisionPDFProcessor(BaseProcessor):
    """PDF processor that converts to images (no text extraction)"""
    
    def __init__(
        self,
        render_scale: float = 2.0,
        jpeg_quality: int = 90,
        max_image_size: tuple = (1400, 1400)
    ):
        self.render_scale = render_scale
        self.jpeg_quality = jpeg_quality
        self.max_image_size = max_image_size
    
    async def process(self, file_path: str) -> List[Page]:
        """
        Convert PDF pages to JPEG images.
        NO text extraction - pure vision approach.
        """
        try:
            logger.info(f"Processing PDF with vision: {file_path}")
            
            # Create temp directory for images
            temp_dir = Path(tempfile.mkdtemp(prefix="vision_pdf_"))
            
            # Open PDF with PyMuPDF
            pdf = fitz.open(file_path)
            pages = []
            
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                
                # Render page to high-quality pixmap
                mat = fitz.Matrix(self.render_scale, self.render_scale)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                # Convert to PIL Image
                img = Image.frombytes(
                    "RGB",
                    [pix.width, pix.height],
                    pix.samples
                )
                
                # Resize if exceeds max size
                if (img.width > self.max_image_size[0] or 
                    img.height > self.max_image_size[1]):
                    img.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
                    logger.debug(f"Resized page {page_num+1} to {img.size}")
                
                # Save as optimized JPEG
                output_path = temp_dir / f"page_{page_num + 1}.jpg"
                img.save(
                    output_path,
                    "JPEG",
                    quality=self.jpeg_quality,
                    optimize=True
                )
                
                # Create Page object
                pages.append(Page(
                    page_number=page_num + 1,
                    image_path=str(output_path),
                    width=img.width,
                    height=img.height
                ))
                
                logger.debug(f"Converted page {page_num+1} → {output_path}")
            
            pdf.close()
            logger.info(f"Processed {len(pages)} pages from {file_path}")
            return pages
            
        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            raise
    
    def supports(self, file_path: str) -> bool:
        """Check if file is PDF"""
        return file_path.lower().endswith('.pdf')