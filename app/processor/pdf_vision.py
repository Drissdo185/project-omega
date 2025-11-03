# app/processors/pdf_vision.py
import fitz  # PyMuPDF
from PIL import Image
from pathlib import Path
import os
import json
import hashlib
from typing import Optional
from datetime import datetime, timezone
from loguru import logger

from .base import BaseProcessor
from app.models.document import Page, Document, DocumentStatus

class VisionPDFProcessor(BaseProcessor):
    """PDF processor that converts to images (no text extraction)"""

    def __init__(
        self,
        render_scale: float = 1.5,
        jpeg_quality: int = 75,
        max_image_size: tuple = (1024, 1024),
        storage_root: str = None
    ):
        if storage_root is None:
            storage_root = os.environ.get("FLEX_RAG_DATA_LOCATION", "../../flex_rag_data_location")
    
        self.storage_root = Path(storage_root)
        self.render_scale = render_scale
        self.jpeg_quality = jpeg_quality
        self.max_image_size = max_image_size
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


    async def process(self, file_path: str, doc_id: Optional[str] = None) -> Document:
        """
        Convert PDF pages to JPEG images and save in structured directory.
        Note: Does NOT save metadata.json - that happens after analysis.

        Args:
            file_path: Path to the PDF file
            doc_id: Optional document ID (generated if not provided)

        Returns:
            Document object with all metadata
        """
        try:
            logger.info(f"Processing PDF with vision: {file_path}")

            # Generate document ID if not provided
            if not doc_id:
                doc_id = self._generate_doc_id(file_path)

            # Create document directory structure
            doc_dir = self._create_document_directory(doc_id)
            pages_dir = doc_dir / "pages"

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

                # Save as optimized JPEG in pages directory
                output_path = pages_dir / f"page_{page_num + 1}.jpg"
                img.save(
                    output_path,
                    "JPEG",
                    quality=self.jpeg_quality,
                    optimize=True
                )

                # Create Page object with relative path
                pages.append(Page(
                    page_number=page_num + 1,
                    image_path=str(output_path),
                    width=img.width,
                    height=img.height
                ))

                logger.debug(f"Converted page {page_num+1} → {output_path}")

            pdf.close()

            # Create Document object
            document = Document(
                id=doc_id,
                name=Path(file_path).name,
                page_count=len(pages),
                pages=pages,
                status=DocumentStatus.READY,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc)
            )

            logger.info(f"Successfully processed {len(pages)} pages from {file_path}")
            logger.info(f"Document directory: {doc_dir}")
            logger.info(f"Metadata will be saved after analysis is complete")

            return document

        except Exception as e:
            logger.error(f"Failed to process PDF: {e}")
            raise
    
    def supports(self, file_path: str) -> bool:
        """Check if file is PDF"""
        return file_path.lower().endswith('.pdf')